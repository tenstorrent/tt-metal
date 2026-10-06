# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every distinct finite 16-bit value -- a stride of Float32 -- every enrolled unary SFPU
op.

The functional drivers in test_eltwise_unary_sfpu.py sample a few thousand points from
an op's safe domain, so a budget measured that way can only ever be re-confirmed by
them: it cannot see a tail the sample never reaches.

One device run per variant covers the whole format: 65,279 finite bfloat16 values or
63,487 float16 ones, in 64 tiles. ``Bfp8_b`` is swept in bfloat16 and packed on the way
in; a ``Float32`` input has 2**32 values, so it is walked with a stride instead.
Marked ``accuracy``, which every LLK workflow deselects, so it runs only by name or with
``-m accuracy``. Run it as a gate::

    pytest test_unary_sfpu_ulp.py

or re-measure and fold the results back into helpers/sfpu_accuracy_budget.yaml::

    pytest test_unary_sfpu_ulp.py --ulp-emit            # every op with a key line
    pytest test_unary_sfpu_ulp.py --ulp-emit --op MyOp  # one op, matched exactly
"""

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
    FLUSH_SUBNORMAL_OUTPUTS,
    MEASURED_ARCH,
    Metric,
    accuracy_contract,
)
from helpers.sfpu_domains import (
    _UNARY_OPS_NOT_SWEPT,
    sfpu_unary_ops,
    unpacks_to_dest,
)
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
    golden_input,
    measurable_mask,
    nonfinite_failures,
    nonfinite_reason,
    stimuli_format_for,
    sweep_cells,
    sweep_spec,
)
from helpers.utils import _record_ulp_measurement, passed_test

#: `accuracy` is the marker every LLK workflow deselects; `nightly` would not do, since
#: llk-e2e runs it.
pytestmark = [
    pytest.mark.accuracy,
    # Every unkeyed budget is a Wormhole measurement and binds nowhere else
    # (sfpu_accuracy_budget.MEASURED_ARCH), so off it every cell resolves to tolerance,
    # nothing can fail, and the headroom report would hold another chip's figures
    # against this table's. --ulp-emit refuses off Wormhole for the same reason.
    pytest.mark.skipif(
        get_chip_architecture() != MEASURED_ARCH,
        reason=f"the table's budgets are {MEASURED_ARCH.value} measurements; this sweep "
        "gates nothing on another architecture",
    ),
]

#: 64 tiles x 1024 lanes = 65,536: every finite bf16 and fp16 value in one run, the
#: Float32 stride's sample count, and the generator's own ceiling. A smaller count would
#: silently keep only the lowest-sorted values, so test_ulp_sweep.py pins it against
#: `swept_value_count`.
SWEEP_TILE_COUNT = 64
SWEEP_DIMENSIONS = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * SWEEP_TILE_COUNT]

#: The emit's ranking and the gate's verdict alike, so they cannot rank differently; the
#: policy and its reason live with the binary/ternary gate's, which ranks the same way.
_FLUSH_SUBNORMALS = FLUSH_SUBNORMAL_OUTPUTS


def run_sweep(mathop, formats, approx_mode, dest_acc):
    """One variant on hardware, over every value of a 16-bit input or a stride of a
    Float32 one (``ulp_sweep.is_exhaustive``). Returns ``(src, golden, result)``."""
    torch.manual_seed(0)
    stimuli_format = stimuli_format_for(formats.input_format)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=stimuli_format,
        input_dimensions_A=SWEEP_DIMENSIONS,
        stimuli_format_B=stimuli_format,
        input_dimensions_B=SWEEP_DIMENSIONS,
        spec_A=sweep_spec(formats.input_format),
    )
    golden = get_golden_generator(UnarySFPUGolden)(
        mathop,
        golden_input(src_A, formats.input_format, dest_acc),
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
        unpack_to_dest=unpacks_to_dest(formats.input_format, dest_acc),
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


def _sweep_ops():
    """Every unary op with a key line in the table, whatever its rows say, gating or
    emitting. The key line is the enrolment (SFPU_ULP.md, step 2) and the measurement
    fills it in; an op without one has nowhere to be written, and `sfpu_unary_ops()`
    admits every newly registered op, so a wider set would turn a whole-table emit red
    the day anyone adds an op.

    The gate run used to take only the ops with a step budget somewhere. Since every
    cell is measured, gated or not, and the headroom report holds a tolerance cell to
    the figure its row records, that left the ops on tolerance everywhere -- Digamma,
    Erfc, Expm1Cw, GeluAppx, Lgamma, Polygamma, SigmoidAppx, Softplus, 333 rows -- with
    recorded figures and no run behind them."""
    unary = sfpu_unary_ops()
    ops = {op for op in _SFPU_ACCURACY_BUDGET if op in unary} - set(
        _UNARY_OPS_NOT_SWEPT
    )
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
    """Every non-special value of a 16-bit input format, or a stride of Float32, against
    the op's declared budget."""
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
    # A tolerance cell has no budget to gate, but is measured anyway: its row records the
    # last sweep's worst lane, and the nightly's headroom report fails a run that exceeds
    # it. Skipping it made a demoted cell's regressions and recoveries invisible.
    gated = contract.metric == Metric.ULP

    # No OverflowError skip. cosh/sinh were the last goldens that could raise on a finite
    # input; they go through torch now. Some still call `math.*` (atan, asinh, the tanh
    # family, gelu_derivative, xielu, the rounding ops), but none can raise on a finite
    # argument: each is bounded, takes a non-positive argument, or guards with isfinite.
    # A new `math.exp`-style golden would, and then fails the cell loudly rather than
    # turning it into a skip nobody reads.
    src, golden, result = run_sweep(mathop, formats, approx_mode, dest_acc)

    mask = measurable_mask(src, golden, result, in_fmt, out_fmt, dest_acc)
    overflowed = nonfinite_failures(
        mathop,
        src,
        golden,
        result,
        in_fmt,
        out_fmt,
        dest_acc=dest_acc,
        approx_mode=approx_mode,
    )
    if not ulp_sweep.EMIT:
        # An excused lane that agrees again means the defect its issue tracks is gone
        # from this cell; the entry has to go with it, or its lanes stay ungated.
        stale = ulp_sweep.stale_excuses(
            mathop,
            src,
            golden,
            result,
            in_fmt,
            out_fmt,
            approx_mode=approx_mode,
            dest_acc=dest_acc,
        )
        assert not stale, (
            f"{cell}: no lane the _KNOWN_NONFINITE_LANES entry for "
            f"{', '.join(entry.issue for entry in stale)} names disagrees with the "
            "golden any more; drop the entry so those lanes are gated again"
        )
    distance = ulp_distance(golden, result, flush_subnormals=_FLUSH_SUBNORMALS)
    stats = ulp_stats(distance, mask)
    lanes = int(mask.sum())
    key = (in_fmt.name, out_fmt.name, approx_mode.name, dest_acc.name)

    # Before the emit return: an empty mask reports `max: 0` and would be recorded as
    # bit-exact. A gated cell fails, since an unmeasurable budget is a gate not running.
    # An emit run records the reason as the cell's verdict, so the op's grid stays
    # whole; a tolerance cell skips, having no budget to hold.
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
        if ulp_sweep.EMIT:
            ulp_sweep.record_unmeasurable(mathop.name, key, reason)
            return
        if not gated:
            # Skipped, but on the record: the row names how many lanes went non-finite,
            # and the headroom report fails a run where that count appears or grows.
            # Without this a re-emitted overflow was a permanent, silent skip.
            _record_ulp_measurement(
                distance,
                mask=mask,
                nonfinite=int(overflowed.sum()),
            )
            pytest.skip(f"{cell}: not measurable -- {unmeasurable}")
        raise AssertionError(f"{cell}: {unmeasurable}")

    if ulp_sweep.EMIT or not gated:
        # Measured, not gated. The JSONL row is what `--ulp-measure` would have written
        # from inside passed_test, which this branch returns before; a tolerance cell is
        # held against its row's "max N ULP" by the headroom report.
        _record_ulp_measurement(distance, mask=mask)
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
        **contract.passed_test_kwargs(flush_subnormals=_FLUSH_SUBNORMALS),
    ), (
        f"{cell}: failed a {contract.max_ulp}-step budget over {lanes} swept lanes; "
        "the failing lanes are in the ULP-budget log above. Raw maximum before any "
        f"near_zero_atol floor: {stats['max']} ULP at flat index "
        f"{stats['worst_index']}."
    )
