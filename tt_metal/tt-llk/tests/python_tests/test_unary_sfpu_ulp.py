# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every normal 16-bit value and zero -- a stride of Float32 -- every enrolled unary
SFPU op.

The functional drivers in test_eltwise_unary_sfpu.py sample a few thousand points from
an op's safe domain, so a budget measured that way can only ever be re-confirmed by
them: it cannot see a tail the sample never reaches.

One device run per variant covers the whole format: 65,025 bfloat16 values or 61,441
float16 ones -- every normal value and zero -- in 64 tiles; subnormals are not fed,
since the unpack flushes them. A ``Float32`` input has 2**32 values, so it is walked
with a stride instead. Only the formats the SFPU computes in are swept: a block float's
error is the unpacker's and packer's, not the op's.
Marked ``accuracy``, which every LLK workflow deselects, so it runs only by name or with
``-m accuracy``. Run it as a gate::

    pytest test_unary_sfpu_ulp.py

which fails a gated cell over its budget and skips a tolerance cell with its measured
maximum in the reason. To hold the tolerance cells to the figures their rows record,
measure and compare::

    pytest test_unary_sfpu_ulp.py --ulp-measure=ulp.jsonl
    python -m helpers.ulp_budget_diff headroom \\
        --table helpers/sfpu_accuracy_budget.yaml --measured ulp.jsonl

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
    Metric,
    accuracy_contract,
    domains_of,
    usable_budget_ceiling,
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
from helpers.ulp_domains import largest_figure, partition
from helpers.ulp_sweep import (
    band_mask,
    domain_figures,
    entry_readings,
    floored_profile,
    golden_input,
    near_zero_floor,
    nonfinite_failures,
    nonfinite_reason,
    padding_lanes,
    ranked_mask,
    received_inputs,
    sweep_cells,
    sweep_spec,
)
from helpers.ulp_table import MEASURED_ARCH
from helpers.utils import _record_ulp_measurement, passed_test

#: `accuracy` is the marker every LLK workflow deselects; `nightly` would not do, since
#: llk-e2e runs it.
pytestmark = [
    pytest.mark.accuracy,
    # Every unkeyed budget is a Wormhole measurement and binds nowhere else
    # (ulp_table.MEASURED_ARCH), so off it every cell resolves to tolerance,
    # nothing can fail, and the headroom report would hold another chip's figures
    # against this table's. --ulp-emit refuses off Wormhole for the same reason.
    pytest.mark.skipif(
        get_chip_architecture() != MEASURED_ARCH,
        reason=f"the table's budgets are {MEASURED_ARCH.value} measurements; this sweep "
        "gates nothing on another architecture",
    ),
]

#: 64 tiles x 1024 lanes = 65,536: every normal bf16 and fp16 value and zero in one run,
#: the Float32 stride's 65,024 samples, and the generator's own ceiling. A smaller count would
#: silently keep only the lowest-sorted values, so test_ulp_sweep.py pins it against
#: `swept_value_count`.
SWEEP_TILE_COUNT = 64
SWEEP_DIMENSIONS = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * SWEEP_TILE_COUNT]

#: The emit's ranking and the gate's verdict alike, so they cannot rank differently; the
#: policy and its reason live with the binary/ternary gate's, which ranks the same way.
_FLUSH_SUBNORMALS = FLUSH_SUBNORMAL_OUTPUTS


def run_sweep(mathop, formats, approx_mode, dest_acc):
    """One variant on hardware, over every normal value (and zero) of a 16-bit input or
    a stride of a Float32 one (``ulp_sweep.is_exhaustive``). Returns ``(src, golden, result)``.
    """
    torch.manual_seed(0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=SWEEP_DIMENSIONS,
        stimuli_format_B=formats.input_format,
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
            # compiled in and holds for a kernel built that way (SFPU_ULP.md, "The rules
            # the table enforces").
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
    fills it in; the sweep collects only the ops the table names, and `sfpu_unary_ops()`
    admits every newly registered op, so a wider set would sweep each new op the day
    anyone adds it, whether or not it was meant to be swept."""
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
    """Every normal value of a 16-bit input format, or a stride of Float32. A row with a
    ``domains`` list is held entry by entry: each entry within the output's ceiling to
    its figure (``ulp_domains``). A plain gated cell is held to its step budget; a
    tolerance cell or entry is measured and recorded, and the headroom report holds it
    to the figure its row records."""
    formats = InputOutputFormat(in_fmt, out_fmt)
    cell = (
        f"{mathop.name} {in_fmt.name}->{out_fmt.name} approx={approx_mode.name} "
        f"dest_acc={dest_acc.name}"
    )
    arch = get_chip_architecture()
    contract = accuracy_contract(
        mathop,
        output_format=out_fmt,
        input_format=in_fmt,
        approx_mode=approx_mode,
        dest_acc=dest_acc,
        arch=arch,
    )
    # The row's list, if it has one: then the entries are the verdict, and the cell is
    # gated wherever an entry is. A tolerance cell is measured too: the headroom report
    # holds it, entry by entry where it has them, to its row's figures.
    domains, floor_atol = domains_of(
        mathop,
        output_format=out_fmt,
        input_format=in_fmt,
        approx_mode=approx_mode,
        dest_acc=dest_acc,
        arch=arch,
    )
    ceiling = usable_budget_ceiling(out_fmt)
    gated = contract.metric == Metric.ULP or any(d.gated(ceiling) for d in domains)

    # No OverflowError skip. cosh/sinh were the last goldens that could raise on a finite
    # input; they go through torch now. Some still call `math.*` (atan, asinh, the tanh
    # family, gelu_derivative, xielu, the rounding ops), but none can raise on a finite
    # argument: each is bounded, takes a non-positive argument, or guards with isfinite.
    # A new `math.exp`-style golden would, and then fails the cell loudly rather than
    # turning it into a skip nobody reads.
    src, golden, result = run_sweep(mathop, formats, approx_mode, dest_acc)

    # Ranked over the op's claim as well as the measurable lanes: a lane past an
    # argument-reduction limit or on a pole is one the op answers nothing for, so it
    # can no more set a budget than be a non-finite failure (which already read the
    # claim). Sin's and Cos's budgets were set by |x| ~ 1e13 lanes answering 2e38.
    mask = ranked_mask(mathop, src, golden, result, in_fmt, out_fmt, dest_acc)
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
    key = ulp_sweep.cell_key(in_fmt, out_fmt, approx_mode, dest_acc)

    # Before the emit return, or an empty mask is recorded as bit-exact `max: 0`. The
    # overflow first: an all-non-finite cell also has `lanes == 0`, and only this
    # verdict carries the lane count the headroom report reads back (`nonfinite`).
    if overflowed.any():
        verdict = ulp_sweep.nonfinite_verdict(overflowed, src, golden, result, stats)
        reason = nonfinite_reason(verdict, lanes)
        unmeasurable = (
            f"{reason}. No budget buys an overflow, and a step count cannot describe "
            "one."
        )
    elif lanes == 0:
        verdict = ulp_sweep.Unmeasurable(why="no lane a step count can describe")
        unmeasurable = verdict.why
    else:
        unmeasurable = None
    if unmeasurable:
        if ulp_sweep.EMIT or not gated:
            # On the record either way: the row names how many lanes went non-finite,
            # and the headroom report fails a run where that count appears or grows.
            _record_ulp_measurement(
                distance,
                mask=mask,
                nonfinite=verdict.nonfinite,
                required=True,
            )
        if ulp_sweep.EMIT:
            ulp_sweep.record_unmeasurable(mathop, key, verdict)
            return
        if not gated:
            pytest.skip(
                f"{cell}: measured, not gated -- not measurable: {unmeasurable}"
            )
        raise AssertionError(f"{cell}: {unmeasurable}")

    # Where each lane's input sits, as the kernel received it, for the domains; and the
    # lanes the near-zero floor judges on absolute error, which the entries leave out
    # as the cell's residual does.
    x = received_inputs(src, in_fmt, out_fmt, dest_acc).to(torch.float64)
    present = ~padding_lanes(src, in_fmt)

    if ulp_sweep.EMIT:
        # A cell that only a near-zero lane demotes is recorded with the floor that
        # covers it, and the emitter enrols it on the lanes outside that floor; the list
        # is drawn on those same lanes, so an entry's figure is the cell's residual over
        # fewer lanes.
        floor = near_zero_floor(golden, result, distance, mask, out_fmt)
        # The profile counts a lane the floor covers as ranked at 0 steps: it is held,
        # by the floor, so it draws no edge of its own, and the entry it falls in is
        # figured over the lanes outside the floor, as the cell is.
        judged, bins = floored_profile(
            golden, result, x, distance, mask, present, floor[1] if floor else None
        )
        drawn = domain_figures(partition(bins, ceiling), x, distance, mask, judged)
        _record_ulp_measurement(
            distance,
            mask=mask,
            bins=bins,
            readings=entry_readings(drawn, x, distance, mask, judged),
            required=True,
        )
        ulp_sweep.record(mathop, key, int(stats["max"]), floor=floor, domains=drawn)
        return

    if not domains:
        if not gated:
            # Measured, not gated. The JSONL row is what `--ulp-measure` would have
            # written from inside passed_test, which this branch returns before; a
            # tolerance cell is held against its row's `measured` figure by the headroom
            # report, which judges only the cells it reads: the record is required.
            _record_ulp_measurement(distance, mask=mask, required=True)
            # A skip, not a pass: nothing here judged the cell. Without --ulp-measure
            # there is no record for the headroom report either, and a regression would
            # show green.
            pytest.skip(
                f"{cell}: measured, not gated -- on the tolerance metric; measured max "
                f"{int(stats['max'])} ULP over {lanes} lanes. The headroom report "
                "(ulp_budget_diff.py headroom over --ulp-measure) holds it to its row's "
                "recorded figure."
            )
        # The contract's own verdict rather than `stats["max"]`, so a `near_zero_atol`
        # floor is honoured. That makes `stats` the raw maximum, which can be a lane the floor
        # rescued: the failing lanes are the ones passed_test logs.
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
        return

    # A row with a list: the entries are the verdict (ulp_domains). Record the cell
    # once, with its reading over every entry -- which consumes the one lookup above, so
    # the comparisons below record nothing of their own -- then hold each gateable entry
    # to its figure over the lanes outside the floor, and the ranked lanes a null entry
    # leaves unjudged to the row's largest.
    judged, bins = floored_profile(
        golden, result, x, distance, mask, present, floor_atol
    )
    readings = entry_readings(domains, x, distance, mask, judged)
    _record_ulp_measurement(
        distance, mask=mask, bins=bins, readings=readings, required=True
    )
    if not gated:
        pytest.skip(
            f"{cell}: measured, not gated -- every entry is past the ceiling; measured max "
            f"{int(stats['max'])} ULP over {lanes} lanes. The headroom report holds each "
            "entry to its figure."
        )
    largest = largest_figure(domains)
    failed = []
    inside = torch.zeros_like(mask)
    for entry, reading in zip(domains, readings):
        if entry.ulp is None:
            continue
        # Exactly the lanes the emitter figured the entry on: ranked, outside the
        # cell-wide floor. The floor is not applied again here -- scoped to one entry's
        # lanes its band would differ from the emitter's. A covered lane is one whose
        # absolute error is within the floor today; one that drifted out of it is in
        # `judged`, and held to the figure by steps like any other.
        band = band_mask(entry, x, judged)
        inside |= band
        if entry.ulp > ceiling:
            continue  # measured, not gated: the headroom report holds it
        if not bool(band.any()):
            # Every ranked lane held by the floor passes; an entry that ranks no lane at
            # all any more -- its lanes left the mask -- judged nothing, so it fails.
            if not bool(band_mask(entry, x, mask).any()):
                failed.append(
                    f"x in [{entry.low:g}, {entry.high:g}): no lane ranked against "
                    f"its figure {entry.ulp}"
                )
            continue
        if not passed_test(
            golden,
            result,
            out_fmt,
            mask=band,
            max_ulp=entry.ulp,
            flush_subnormals=_FLUSH_SUBNORMALS,
        ):
            failed.append(
                f"x in [{entry.low:g}, {entry.high:g}): worst "
                f"lane {reading.worst} over its figure {entry.ulp}"
            )
    leftover = judged & ~inside
    if (
        bool(leftover.any())
        and largest <= ceiling
        and not passed_test(
            golden,
            result,
            out_fmt,
            mask=leftover,
            max_ulp=largest,
            flush_subnormals=_FLUSH_SUBNORMALS,
        )
    ):
        failed.append(
            f"{int(leftover.sum())} ranked lane(s) in a null entry, over the row's "
            f"largest figure {largest}"
        )
    assert not failed, (
        f"{cell}: " + "; ".join(failed) + ". The failing lanes are in the ULP-budget "
        "log above."
    )
