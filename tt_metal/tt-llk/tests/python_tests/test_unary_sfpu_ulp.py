# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every distinct finite 16-bit value, every ULP-gateable unary SFPU op.

The functional drivers in test_eltwise_unary_sfpu.py sample a few thousand points from
an op's safe domain, so a budget measured that way can only ever be re-confirmed by
them: it cannot see a tail the sample never reaches.

One device run per variant covers the whole format: 65,279 finite bfloat16 values or
63,487 float16 ones, in 64 tiles. ``Bfp8_b`` is swept in bfloat16 and packed on the way
in. Marked ``accuracy``, which every LLK workflow deselects, so it runs only by name
or with ``-m accuracy``. Run it as a gate::

    pytest test_unary_sfpu_ulp.py

or re-measure and fold the results back into helpers/sfpu_accuracy_budget.yaml::

    pytest test_unary_sfpu_ulp.py --ulp-emit            # every op with a key line
    pytest test_unary_sfpu_ulp.py --ulp-emit --op MyOp  # one op, matched exactly
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
    BlocksCalculationAlgorithm,
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
    measurable_mask,
    nonfinite_failures,
    nonfinite_reason,
    stimuli_format_for,
    sweep_cells,
    sweep_spec,
)
from helpers.utils import passed_test

#: ~7 minutes of 64-tile device runs. `accuracy` is the marker every LLK workflow
#: deselects; `nightly` would not do, since llk-e2e runs it.
pytestmark = pytest.mark.accuracy

#: 64 tiles x 1024 lanes = 65,536: every finite bf16 and fp16 value in one run, and the
#: generator's own ceiling. A smaller count would silently keep only the lowest-sorted
#: values, so test_ulp_sweep.py pins it against `swept_value_count`.
SWEEP_TILE_COUNT = 64
SWEEP_DIMENSIONS = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * SWEEP_TILE_COUNT]

#: Subnormal outputs flushed on every format, fp16 included, by the emit's ranking and
#: the gate's verdict alike. The metric keeps fp16's subnormal band by default, but the
#: golden keeps IEEE subnormals the pack path does not reproduce: an exact op read 512
#: steps on Float16_b->Float16 from that band alone. A difference below 6.1e-05 is the
#: store's, not the op's. One constant, so emit and gate cannot rank differently.
_FLUSH_SUBNORMALS = True


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
            # The table has no axis for either: every budget is measured with these two
            # compiled in and holds for a kernel built that way (see the YAML header).
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
        # Every swept input is 16-bit or a block float, so nothing unpacks to Dest.
        unpack_to_dest=False,
    )
    # `sweep_cells` leaves out the cells TestConfig would promote to another Dest, so
    # every cell swept here must be built with the dest_acc it asks for; a cell built
    # as another would key its measurement on a kernel that never ran. Fail here rather
    # than record it.
    assert configuration.dest_acc == dest_acc, (
        f"{mathop.name}: asked for dest_acc={dest_acc.name}, TestConfig built "
        f"{configuration.dest_acc.name} -- sweep_cells() is out of step with it"
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
    every unary op with a key line in the table, whatever its rows say: the key line is
    the enrolment (SFPU_ULP.md, step 2), the measurement fills it in. An op without one
    has nowhere to be written, so sweeping it would only turn the emit red after the
    rest was written -- and `sfpu_unary_ops()` admits every newly registered op, so a
    whole-table emit would go red the day anyone adds an op."""
    unary = sfpu_unary_ops()
    if _emitting():
        ops = {op for op in _SFPU_ACCURACY_BUDGET if op in unary} - set(
            _UNARY_OPS_NOT_SWEPT
        )
    else:
        ops = {
            op
            for op, table in _SFPU_ACCURACY_BUDGET.items()
            if op in unary and any(c.metric == Metric.ULP for c in table.values())
        }
    return sorted(ops, key=lambda op: op.name)


@pytest.mark.parametrize(
    "in_fmt, out_fmt, approx_mode, dest_acc",
    [
        pytest.param(
            *cell,
            id=f"in:{cell[0].name}-out:{cell[1].name}-approx:{cell[2].name}"
            f"-dest_acc:{cell[3].name}",
        )
        for cell in sweep_cells()
    ],
)
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
        # Nothing to gate, so the cell is not swept here at all: neither the step count
        # nor the non-finite check runs on it. Its safe domain is the functional driver's
        # (test_eltwise_unary_sfpu.py, tolerance arm); the whole-format tail is measured
        # only by an emit run, which is what wrote the row's comment. The nightly sweep
        # that measures every cell and holds a tolerance row to its recorded maximum
        # arrives with the headroom report (#57527).
        pytest.skip(
            f"{cell}: on the tolerance metric, so unswept at gate time; its safe "
            "domain is the functional driver's, the full-format tail only an emit's"
        )

    try:
        src, golden, result = run_sweep(mathop, formats, approx_mode, dest_acc)
    except OverflowError as exc:
        if not ulp_sweep.EMIT:
            # A gated cell's golden already ran over the full range when its budget was
            # measured, so an overflow now is a regression, not a limit of the reference.
            raise
        # The float64 host golden overflows on inputs no sampled domain reaches
        # (cosh(3.4e38)): a limit of the reference, not a measurement, for an op emit
        # is measuring for the first time. Only OverflowError -- a ValueError here is a
        # real "Unsupported operation".
        pytest.skip(f"golden cannot be computed over the full range: {exc}")

    mask = measurable_mask(src, golden, result, in_fmt)
    overflowed = nonfinite_failures(
        mathop,
        src,
        golden,
        result,
        in_fmt,
        out_fmt,
        approx_mode=approx_mode,
        dest_acc=dest_acc,
    )
    if not ulp_sweep.EMIT:
        # An excused lane that agrees again means the defect its issue tracks is gone
        # from this cell; the entry has to go with it, or its lanes stay ungated.
        stale = ulp_sweep.stale_excuses(
            mathop, src, golden, result, in_fmt, out_fmt, approx_mode, dest_acc
        )
        assert not stale, (
            f"{cell}: no lane the _KNOWN_NONFINITE_LANES entry for "
            f"{', '.join(entry.issue for entry in stale)} names disagrees with the "
            "golden any more; drop the entry so those lanes are gated again"
        )
    stats = ulp_stats(
        ulp_distance(golden, result, flush_subnormals=_FLUSH_SUBNORMALS), mask
    )
    lanes = int(mask.sum())
    key = (in_fmt.name, out_fmt.name, approx_mode.name, dest_acc.name)

    # Before the emit return: an empty mask reports `max: 0` and would be recorded as
    # bit-exact. A gate fails, since an unmeasurable budget is a gate not running; an
    # emit run records the reason as the cell's verdict, so the op's grid stays whole.
    if lanes == 0:
        reason = "no lane a step count can describe"
        unmeasurable = reason
    elif overflowed.any():
        reason = nonfinite_reason(overflowed, src, golden, result, stats, lanes)
        unmeasurable = (
            f"{reason}. No budget buys an overflow, and a step count cannot describe "
            "one."
        )
    else:
        unmeasurable = None
    if unmeasurable:
        if not ulp_sweep.EMIT:
            raise AssertionError(f"{cell}: {unmeasurable}")
        ulp_sweep.record_unmeasurable(mathop.name, key, reason)
        return

    if ulp_sweep.EMIT:
        ulp_sweep.record(mathop.name, key, int(stats["max"]))
        return

    # The contract's own verdict rather than `stats["max"]`, so a `near_zero_atol` floor
    # (Erfinv Float16_b->Float16_b reads 14704 steps raw, and 2 with it) is honoured.
    # That makes `stats` the raw maximum, which can be a lane the floor rescued: the
    # failing lanes are the ones passed_test logs.
    assert passed_test(
        golden,
        result,
        out_fmt,
        mask=mask,
        flush_subnormals=_FLUSH_SUBNORMALS,
        **contract.passed_test_kwargs(),
    ), (
        f"{cell}: failed a {contract.max_ulp}-step budget over {lanes} swept lanes; "
        "the failing lanes are in the ULP-budget log above. Raw maximum before any "
        f"near_zero_atol floor: {stats['max']} ULP at flat index "
        f"{stats['worst_index']}."
    )
