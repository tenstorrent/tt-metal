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
    MathOperation,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.sfpu_accuracy_budget import MEASURED_ARCH, Metric, accuracy_contract
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

#: Every variant here is a real 64-tile device run, and the whole sweep is ~7 minutes on
#: hardware. Unmarked, the PR gate collects it and runs it with `--timeout=60 -x` under a
#: 15-minute budget, where the first slow variant reds the gate.
#:
#: `accuracy` rather than `nightly`, which pytest.ini defines as exactly this -- "marks
#: SFPU accuracy-sweep tests". It is the marker every LLK workflow deselects
#: (`not perf and not quasar and not accuracy`); `nightly` is deselected only by
#: pr-gate.yaml, so marking it that way would still have left llk-e2e running the sweep
#: on every push. Run it deliberately, by name.
pytestmark = pytest.mark.accuracy

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
    assert len(res_from_L1) == len(golden_tensor), (
        f"{mathop.name}: hardware returned {len(res_from_L1)} values against "
        f"{len(golden_tensor)} in the golden"
    )
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    return src_A, golden_tensor, res_tensor


def measure(mathop, formats, approx_mode, dest_acc):
    """One variant, as the tensors a verdict needs plus the distribution.

    Returns ``(golden, result, mask, overflowed, stats)``. *mask* is the lanes a step
    count can describe; *overflowed* is the lanes it drops that are a *failure* rather
    than a non-question, which a caller ranking only the mask would never see.
    """
    src, golden, result = run_sweep(mathop, formats, approx_mode, dest_acc)
    mask = measurable_mask(src, golden, result, formats.input_format)
    overflowed = nonfinite_failures(
        mathop, src, golden, result, formats.input_format, formats.output_format
    )
    stats = ulp_stats(ulp_distance(golden, result), mask)
    return golden, result, mask, overflowed, stats


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


def _measurable_unary_ops():
    """Every unary op a measurement pass may reach, enrolled or not.

    Wider than the gate's set on purpose. Restricting emit to ops that already carry a
    ULP row is the loop the sweep exists to break: the ops a measurement would decide
    enrolment for -- ``SigmoidAppx`` and ``GeluAppx``, each on a single tolerance row,
    and every registered unary op with no entry at all -- are exactly the unreachable
    ones, and hand-seeding a row to get in is what the table's header forbids.
    """
    from helpers.sfpu_domains import _UNARY_OPS_NOT_SWEPT, sfpu_unary_ops

    return sorted(
        set(sfpu_unary_ops()) - set(_UNARY_OPS_NOT_SWEPT), key=lambda o: o.name
    )


#: Resolved at collection, so it has to read `EMIT` from the command line rather than
#: from `ulp_sweep`, which `pytest_configure` sets at the same point.
SWEEP_OPS = (
    _measurable_unary_ops()
    if any(arg == "--ulp-emit" for arg in sys.argv)
    else _ulp_gateable_unary_ops()
)


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
        golden, result, mask, overflowed, stats = measure(
            mathop, formats, approx_mode, dest_acc
        )
    except OverflowError as exc:
        # The golden is computed on the host in float64, and a full-range sweep feeds it
        # inputs no sampled domain reaches -- cosh(3.4e38) overflows Python's float
        # before the kernel is ever consulted. That is a limit of the reference, not a
        # measurement, so it must not silently become a budget.
        #
        # `OverflowError` only: every unary golden that could raise a domain `ValueError`
        # is routed through torch instead, so catching it here would swallow
        # `Unsupported operation` out of `generate_golden` -- reachable by a YAML edit
        # alone, and silently skipping all 36 variants of the op it enrolled.
        pytest.skip(f"golden cannot be computed over the full range: {exc}")

    lanes = int(mask.sum())
    # Before the emit return, not after: an empty mask makes `ulp_stats` report `max: 0`,
    # so a variant that compared nothing would pass the gate and record a bit-exact
    # `max_ulp: 0`.
    assert lanes > 0, (
        f"{mathop.name} {in_fmt.name}->{out_fmt.name} approx={approx_mode.name} "
        f"dest_acc={dest_acc.name}: no lane a step count can describe"
    )
    assert not bool(overflowed.any()), (
        f"{mathop.name} {in_fmt.name}->{out_fmt.name} approx={approx_mode.name} "
        f"dest_acc={dest_acc.name}: {int(overflowed.sum())} lane(s) disagree about "
        "being non-finite. No budget buys an overflow, and a step count cannot "
        "describe one -- so it is neither gated nor measured below."
    )

    if ulp_sweep.EMIT:
        # --ulp-emit: this run *is* the measurement, so there is nothing to gate against.
        ulp_sweep.record(
            mathop.name,
            (in_fmt.name, out_fmt.name, approx_mode.name, dest_acc.name),
            int(stats["max"]),
        )
        return

    # The verdict the contract declares, not a bare `stats["max"]`: five rows carry a
    # `near_zero_atol` floor, and asserting the maximum alone drops it -- making the gate
    # *stricter* than the registry says. Erfinv bf16->bf16 notes 84 near-zero points at
    # 14690 steps held by that floor, against a `max_ulp: 2`.
    assert passed_test(
        golden,
        result,
        out_fmt,
        mask=mask,
        **contract.passed_test_kwargs(),
    ), (
        f"{mathop.name} {in_fmt.name}->{out_fmt.name} approx={approx_mode.name} "
        f"dest_acc={dest_acc.name}: {stats['max']} ULP over {lanes} swept lanes, "
        f"budget {contract.max_ulp}. Worst lane at flat index {stats['worst_index']}. "
        "The budget was measured on a sample; this sweep leaves nothing out."
    )


@pytest.fixture(scope="session", autouse=True)
def _emit_measured_table(request):
    """Under ``--ulp-emit``, fold the session's measurements into the table on the way out.

    Written once at the end rather than per test: an op's rows collapse across the whole
    variant space, so a row cannot be decided until every cell for that op has been seen.

    Three preconditions, because this rewrites a checked-in file:

    * **the measured architecture.** ``_render`` never emits ``arch``, and
      ``accuracy_contract`` reads an unkeyed ULP row as a Wormhole measurement, so a
      Blackhole emit run would replace the Wormhole budgets with Blackhole numbers
      wearing a Wormhole badge. The arch reaches ``suffix``, but only after the ``#``.
    * **the controller.** ``pytest_configure`` sets ``EMIT`` in xdist workers too, and
      each would write the file from its own shard.
    * **a clean session.** A run that failed or was interrupted has measured a subset,
      and ``write_table`` would replace the cells it holds while `_collapse` widens a
      row over cells it never touched.
    """
    yield
    if not ulp_sweep.EMIT or not ulp_sweep.MEASURED:
        return
    from datetime import date

    from helpers.sfpu_accuracy_budget import _TABLE_PATH

    arch = get_chip_architecture()
    if arch != MEASURED_ARCH:
        pytest.fail(
            f"--ulp-emit ran on {arch.value}, but the table's unkeyed rows are read as "
            f"{MEASURED_ARCH.value} measurements and `_render` does not emit `arch`. "
            "Nothing written."
        )
    if hasattr(request.config, "workerinput"):
        return  # an xdist worker; the controller owns the file
    if request.session.testsfailed:
        pytest.fail(
            f"--ulp-emit saw {request.session.testsfailed} failure(s), so the session "
            "measured a subset. Nothing written -- emit from a clean run."
        )

    suffix = (
        f"exhaustive {'/'.join(f.name for f in SWEEP_FORMATS)} sweep, "
        f"{arch.value}, {date.today().isoformat()}"
    )
    n = ulp_sweep.write_table(_TABLE_PATH, suffix)
    print(f"\n--ulp-emit: rewrote {n} op block(s) in {_TABLE_PATH.name}")
