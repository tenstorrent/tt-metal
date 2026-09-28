# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Nightly: every 16-bit value, every ULP-gateable unary SFPU op.

The functional drivers in test_eltwise_unary_sfpu.py sample a few thousand points from
an op's safe domain, so a budget measured that way can only ever be re-confirmed by
them. Approximate ``Reciprocal`` reads 1 ULP there and 128 ULP over every bf16 value.

One device run per variant covers the whole format: 65,279 finite bfloat16 values or
63,487 float16 ones, in 64 tiles. ``Bfp8_b`` is swept in bfloat16 and packed on the way
in. Run it as a gate (the nightly default)::

    pytest test_unary_sfpu_ulp.py

or re-measure and fold the results back into helpers/sfpu_accuracy_budget.yaml::

    pytest test_unary_sfpu_ulp.py --ulp-emit
"""

import sys

import pytest
import torch
from helpers import ulp_sweep
from helpers.chip_architecture import get_chip_architecture
from helpers.format_config import InputOutputFormat
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
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.sfpu_accuracy_budget import (
    _SFPU_ACCURACY_BUDGET,
    Metric,
    accuracy_contract,
)
from helpers.sfpu_domains import _UNARY_OPS_NOT_SWEPT, sfpu_unary_ops
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
    nonfinite_failures,
    stimuli_format_for,
    sweep_spec,
)
from helpers.utils import passed_test

#: ~7 minutes of 64-tile device runs. `accuracy` is the marker every LLK workflow
#: deselects; `nightly` is deselected only by the PR gate, so llk-e2e would still run it.
pytestmark = pytest.mark.accuracy

#: 64 tiles: the whole bf16/fp16 value set in one run, and the generator's own ceiling.
SWEEP_DIMENSIONS = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * 64]


def run_sweep(mathop, formats, approx_mode, dest_acc):
    """One exhaustive variant on hardware. Returns ``(src, golden, result)``."""
    torch.manual_seed(0)
    stimuli_format = stimuli_format_for(formats.input_format)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=stimuli_format,
        input_dimensions_A=SWEEP_DIMENSIONS,
        stimuli_format_B=stimuli_format,
        input_dimensions_B=SWEEP_DIMENSIONS,
        spec_A=sweep_spec(),
    )
    golden = get_golden_generator(UnarySFPUGolden)(
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
    result = configuration.run().result
    assert len(result) == len(golden), (
        f"{mathop.name}: hardware returned {len(result)} values against "
        f"{len(golden)} in the golden"
    )
    return src_A, golden, torch.tensor(result, dtype=format_dict[formats.output_format])


def _emitting():
    """Whether this session measures rather than gates. ``ulp_sweep.EMIT`` is set in
    ``pytest_configure``, too late for collection, and an xdist worker's argv lacks the
    flag -- so both are read, or ``--compile-producer -n N`` builds the wrong set."""
    return ulp_sweep.EMIT or "--ulp-emit" in sys.argv


def _sweep_ops():
    """Gating sweeps every unary op with a step budget on some variant. Emitting sweeps
    every sweepable unary op, enrolled or not: a measurement decides enrolment, so it
    cannot be limited to what is already enrolled."""
    unary = sfpu_unary_ops()
    if _emitting():
        ops = set(unary) - set(_UNARY_OPS_NOT_SWEPT)
    else:
        ops = {
            op
            for op, table in _SFPU_ACCURACY_BUDGET.items()
            if op in unary and any(c.metric == Metric.ULP for c in table.values())
        }
    return sorted(ops, key=lambda op: op.name)


@pytest.mark.parametrize(
    "dest_acc", list(DestAccumulation), ids=lambda d: f"dest_acc:{d.name}"
)
@pytest.mark.parametrize(
    "approx_mode", list(ApproximationMode), ids=lambda a: f"approx:{a.name}"
)
@pytest.mark.parametrize("out_fmt", SWEEP_FORMATS, ids=lambda f: f"out:{f.name}")
@pytest.mark.parametrize("in_fmt", SWEEP_FORMATS, ids=lambda f: f"in:{f.name}")
@pytest.mark.parametrize("mathop", _sweep_ops(), ids=lambda op: op.name)
def test_unary_sfpu_ulp_sweep(mathop, in_fmt, out_fmt, approx_mode, dest_acc):
    """Every non-special value of the input format, against the op's declared budget."""
    formats = InputOutputFormat(in_fmt, out_fmt)
    cell = (
        f"{mathop.name} {in_fmt.name}->{out_fmt.name} approx={approx_mode.name} "
        f"dest_acc={dest_acc.name}"
    )
    contract = accuracy_contract(
        mathop,
        output_format=out_fmt,
        input_format=in_fmt,
        approx_mode=approx_mode,
        dest_acc=dest_acc,
        arch=get_chip_architecture(),
    )
    if contract.metric != Metric.ULP and not ulp_sweep.EMIT:
        # Nothing to gate. An emit run measures it anyway: that is how it gets enrolled.
        pytest.skip(f"{mathop.name} is on the tolerance metric for this variant")

    try:
        src, golden, result = run_sweep(mathop, formats, approx_mode, dest_acc)
    except OverflowError as exc:
        # The float64 host golden overflows on inputs no sampled domain reaches
        # (cosh(3.4e38)): a limit of the reference, not a measurement. Only
        # OverflowError -- a ValueError here is a real "Unsupported operation".
        pytest.skip(f"golden cannot be computed over the full range: {exc}")

    mask = measurable_mask(src, golden, result, in_fmt)
    overflowed = nonfinite_failures(mathop, src, golden, result, in_fmt, out_fmt)
    stats = ulp_stats(ulp_distance(golden, result), mask)
    lanes = int(mask.sum())

    # Before the emit return: an empty mask reports `max: 0` and would be recorded as
    # bit-exact. A gate fails, since an unmeasurable budget is a gate not running; an
    # emit run skips, since "not measurable" is an answer and a failure blocks the write.
    if lanes == 0:
        unmeasurable = "no lane a step count can describe"
    elif overflowed.any():
        unmeasurable = (
            f"{int(overflowed.sum())} lane(s) disagree about being non-finite. No "
            "budget buys an overflow, and a step count cannot describe one."
        )
    else:
        unmeasurable = None
    if unmeasurable:
        if ulp_sweep.EMIT:
            pytest.skip(f"{cell}: not measurable -- {unmeasurable}")
        raise AssertionError(f"{cell}: {unmeasurable}")

    if ulp_sweep.EMIT:
        ulp_sweep.record(
            mathop.name,
            (in_fmt.name, out_fmt.name, approx_mode.name, dest_acc.name),
            int(stats["max"]),
        )
        return

    # The contract's own verdict rather than `stats["max"]`, so a `near_zero_atol` floor
    # (Erfinv bf16->bf16 holds 84 lanes at 14690 steps with it) is honoured.
    assert passed_test(
        golden, result, out_fmt, mask=mask, **contract.passed_test_kwargs()
    ), (
        f"{cell}: {stats['max']} ULP over {lanes} swept lanes, budget "
        f"{contract.max_ulp}. Worst lane at flat index {stats['worst_index']}."
    )
