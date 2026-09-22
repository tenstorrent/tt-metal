# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Nightly: every 16-bit value, every ULP-gateable unary SFPU op.

The functional drivers in test_eltwise_unary_sfpu.py sample a few thousand points from
an op's safe domain, and the budgets in helpers/sfpu_accuracy_budget.yaml used to be
measured the same way -- so the gate re-sampled the domain its own number came from and
could not fail. Approximate ``Reciprocal`` reads 1 ULP that way and 128 ULP over every
bf16 value there is.

This harness measures the second number. One device run per variant covers the whole
format: 65,279 finite bfloat16 values or 63,487 float16 ones, auto-sized to 64 tiles.
``Bfp8_b`` is swept in bfloat16 and packed on the way in.

Run it as a gate (the nightly default)::

    pytest test_unary_sfpu_ulp.py

Or re-measure and fold the results back into the table::

    pytest test_unary_sfpu_ulp.py --ulp-emit
"""

import pytest
import torch
from helpers import ulp_sweep
from helpers.chip_architecture import get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.sfpu_accuracy_budget import Metric, accuracy_contract
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.ulp import ulp_distance, ulp_stats
from helpers.ulp_sweep import (
    SWEEP_FORMATS,
    measurable_mask,
    stimuli_format_for,
    sweep_spec,
)

#: 64 tiles: the whole bf16/fp16 value set in one run, and the generator's own ceiling.
SWEEP_DIMENSIONS = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * 64]


def run_sweep(
    mathop: MathOperation,
    formats: InputOutputFormat,
    approx_mode: ApproximationMode,
    dest_acc: DestAccumulation,
):
    """One exhaustive variant on hardware. Returns (src, golden, result)."""
    torch.manual_seed(0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=stimuli_format_for(formats.input_format),
        input_dimensions_A=SWEEP_DIMENSIONS,
        stimuli_format_B=stimuli_format_for(formats.input_format),
        input_dimensions_B=SWEEP_DIMENSIONS,
        spec_A=sweep_spec(),
    )

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        SWEEP_DIMENSIONS,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        SWEEP_DIMENSIONS,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(SWEEP_DIMENSIONS, SWEEP_DIMENSIONS),
            APPROX_MODE(approx_mode),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=mathop),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
    )
    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    return src_A, golden_tensor, res_tensor


def measure(mathop, formats, approx_mode, dest_acc):
    """The variant's ULP distribution over the lanes a step count can describe."""
    src, golden, result = run_sweep(mathop, formats, approx_mode, dest_acc)
    mask = measurable_mask(mathop, src, golden, result, formats.input_format)
    distance = ulp_distance(golden, result)
    return ulp_stats(distance, mask), int(mask.sum())


# ─────────────────────────────────────────────────────────────────────────────
# The nightly gate
# ─────────────────────────────────────────────────────────────────────────────


def _ulp_gateable_unary_ops():
    """Every unary op the table gives a step budget to on some variant.

    Read from the registry rather than listed here: an op enrolled tomorrow is swept
    tomorrow, and one demoted to the tolerance metric stops being swept, with no edit
    to this file.
    """
    from helpers.sfpu_accuracy_budget import _SFPU_ACCURACY_BUDGET
    from helpers.sfpu_domains import sfpu_unary_ops

    unary = set(sfpu_unary_ops())
    return sorted(
        (
            op
            for op, table in _SFPU_ACCURACY_BUDGET.items()
            if op in unary and any(c.metric == Metric.ULP for c in table.values())
        ),
        key=lambda op: op.name,
    )


SWEEP_OPS = _ulp_gateable_unary_ops()


@pytest.mark.parametrize(
    "dest_acc", list(DestAccumulation), ids=lambda d: f"dest_acc:{d.name}"
)
@pytest.mark.parametrize(
    "approx_mode", list(ApproximationMode), ids=lambda a: f"approx:{a.name}"
)
@pytest.mark.parametrize("out_fmt", SWEEP_FORMATS, ids=lambda f: f"out:{f.name}")
@pytest.mark.parametrize("in_fmt", SWEEP_FORMATS, ids=lambda f: f"in:{f.name}")
@pytest.mark.parametrize("mathop", SWEEP_OPS, ids=lambda op: op.name)
def test_unary_sfpu_ulp_sweep(mathop, in_fmt, out_fmt, approx_mode, dest_acc):
    """Every non-special value of the input format, against the op's declared budget.

    The functional driver samples this op's safe domain; this one leaves nothing out, so
    a budget that only held on the sample fails here.
    """
    formats = InputOutputFormat(in_fmt, out_fmt)
    contract = accuracy_contract(
        mathop,
        output_format=out_fmt,
        input_format=in_fmt,
        approx_mode=approx_mode,
        dest_acc=dest_acc,
        arch=get_chip_architecture(),
    )
    if contract.metric != Metric.ULP and not ulp_sweep.EMIT:
        # Gating: an op on the tolerance metric has no step budget to check. Emitting is
        # the opposite case -- skipping on the table's current verdict would only ever
        # re-measure what is already enrolled, and the whole point of the sweep is to
        # decide enrolment from the measurement rather than the other way round.
        pytest.skip(f"{mathop.name} is on the tolerance metric for this variant")

    try:
        stats, lanes = measure(mathop, formats, approx_mode, dest_acc)
    except (OverflowError, ValueError) as exc:
        # The golden is computed on the host in float64, and a full-range sweep feeds it
        # inputs no sampled domain reaches -- cosh(3.4e38) overflows Python's float
        # before the kernel is ever consulted. That is a limit of the reference, not a
        # measurement, so it must not silently become a budget.
        pytest.skip(f"golden cannot be computed over the full range: {exc}")

    if ulp_sweep.EMIT:
        # --ulp-emit: this run *is* the measurement, so there is nothing to gate against.
        ulp_sweep.record(
            mathop.name,
            (in_fmt.name, out_fmt.name, approx_mode.name, dest_acc.name),
            int(stats["max"]),
        )
        return

    assert stats["max"] <= contract.max_ulp, (
        f"{mathop.name} {in_fmt.name}->{out_fmt.name} approx={approx_mode.name} "
        f"dest_acc={dest_acc.name}: {stats['max']} ULP over {lanes} swept lanes, "
        f"budget {contract.max_ulp}. Worst lane at flat index {stats['worst_index']}. "
        "The budget was measured on a sample; this sweep leaves nothing out."
    )


@pytest.fixture(scope="session", autouse=True)
def _emit_measured_table():
    """Under ``--ulp-emit``, fold the session's measurements into the table on the way out.

    Written once at the end rather than per test: an op's rows collapse across the whole
    variant space, so a row cannot be decided until every cell for that op has been seen.
    """
    yield
    if not ulp_sweep.EMIT or not ulp_sweep.MEASURED:
        return
    from datetime import date

    from helpers.sfpu_accuracy_budget import _TABLE_PATH

    suffix = (
        f"exhaustive {'/'.join(f.name for f in SWEEP_FORMATS)} sweep, "
        f"{get_chip_architecture().value}, {date.today().isoformat()}"
    )
    n = ulp_sweep.write_table(_TABLE_PATH, suffix)
    print(f"\n--ulp-emit: rewrote {n} op block(s) in {_TABLE_PATH.name}")
