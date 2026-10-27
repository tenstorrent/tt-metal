# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every normal 16-bit value, or a stride of Float32, through every enrolled unary SFPU
op: one device run per variant. A gated cell is held to its row's figure (entry by
entry for a ``domains`` list); a tolerance cell is measured and skipped, and the
headroom report (``python -m helpers.ulp_budget_diff headroom`` over ``--ulp-measure``)
holds it to its row's recorded figure. ``--ulp-emit`` writes the table instead of gating
against it. Marked ``accuracy``; see SFPU_ULP.md for the commands."""

import pytest
import torch
from helpers import ulp_sweep
from helpers.chip_architecture import get_chip_architecture
from helpers.data_format_inference import unpacks_to_dest
from helpers.format_config import InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import FastMode, format_dict
from helpers.sfpu_accuracy_budget import accuracy_contract, domains_of, enrolled_ops
from helpers.sfpu_domains import swept_unary_ops
from helpers.stimuli_generator import generate_stimuli
from helpers.ulp import FLUSH_SUBNORMAL_OUTPUTS, ulp_distance, ulp_stats
from helpers.ulp_domains import largest_figure, partition, usable_budget_ceiling
from helpers.ulp_table import MEASURED_ARCH, Metric
from helpers.utils import _record_ulp_measurement, passed_test
from test_eltwise_unary_sfpu import build_eltwise_unary_sfpu_config

# `accuracy` is the marker every LLK workflow deselects. Every unkeyed budget is a
# Wormhole measurement and binds nowhere else, so off Wormhole nothing here can fail.
pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.skipif(
        get_chip_architecture() != MEASURED_ARCH,
        reason=f"the table's budgets are {MEASURED_ARCH.value} measurements; this sweep "
        "gates nothing on another architecture",
    ),
]

# 64 tiles x 1024 lanes = 65,536: every normal bf16 and fp16 value and zero, and the
# Float32 stride's 65,024 samples; test_ulp_sweep.py pins it against swept_value_count.
SWEEP_TILE_COUNT = 64
SWEEP_DIMENSIONS = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * SWEEP_TILE_COUNT]

_LOG_TAIL = ". The failing lanes are in the ULP-budget log above."


def run_sweep(mathop, formats, approx_mode, dest_acc):
    """One variant on hardware over the whole swept input. Returns ``(src, golden,
    result)``."""
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=SWEEP_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=SWEEP_DIMENSIONS,
        spec_A=ulp_sweep.sweep_spec(formats.input_format),
    )
    golden = get_golden_generator(UnarySFPUGolden)(
        mathop,
        ulp_sweep.golden_input(src_A, formats.input_format, dest_acc),
        formats.output_format,
        dest_acc,
        formats.input_format,
        SWEEP_DIMENSIONS,
    )
    configuration = build_eltwise_unary_sfpu_config(
        formats,
        dest_acc,
        approx_mode,
        mathop,
        SWEEP_DIMENSIONS,
        src_A,
        src_B,
        tile_cnt_A,
        tile_cnt_B,
        # The table has no axis for either: every budget holds for a kernel built
        # with these two compiled in.
        fast_mode=FastMode.No,
        clamp_negative=True,
        unpack_to_dest=unpacks_to_dest(formats.input_format, dest_acc),
    )
    # A cell built as another Dest would key its measurement on a kernel that never
    # ran: sweep_cells leaves the promoted cells out, and this holds it to that.
    assert configuration.dest_acc == dest_acc, (
        f"{mathop.name}: asked for dest_acc={dest_acc.name}, TestConfig built "
        f"{configuration.dest_acc.name}; sweep_cells() is out of step with it"
    )
    result = configuration.run().result
    assert len(result) == len(
        golden
    ), f"{mathop.name}: {len(result)} values back against {len(golden)} in the golden"
    return src_A, golden, torch.tensor(result, dtype=format_dict[formats.output_format])


def _sweep_ops():
    """Every unary op with a row in the table, gating or emitting: the row is the
    enrolment, and the sweep collects only the ops the table names."""
    return sorted(set(enrolled_ops()) & swept_unary_ops(), key=lambda op: op.name)


def _skip_measured(cell, why):
    pytest.skip(f"{cell}: measured, not gated -- {why}")


@pytest.mark.parametrize(
    "in_fmt, out_fmt, approx_mode, dest_acc",
    [
        pytest.param(
            *cell,
            id=f"in:{cell[0].name}-out:{cell[1].name}-approx:{cell[2].name}"
            f"-dest_acc:{cell[3].name}",
        )
        for cell in ulp_sweep.sweep_cells()
    ],
)
@pytest.mark.parametrize("mathop", _sweep_ops(), ids=lambda op: op.name)
def test_unary_sfpu_ulp_sweep(mathop, in_fmt, out_fmt, approx_mode, dest_acc):
    formats = InputOutputFormat(in_fmt, out_fmt)
    cell = (
        f"{mathop.name} {in_fmt.name}->{out_fmt.name} approx={approx_mode.name} "
        f"dest_acc={dest_acc.name}"
    )
    arch = get_chip_architecture()
    variant = dict(
        output_format=out_fmt,
        input_format=in_fmt,
        approx_mode=approx_mode,
        dest_acc=dest_acc,
        arch=arch,
    )
    contract = accuracy_contract(mathop, **variant)
    domains, floor_atol = domains_of(mathop, **variant)
    ceiling = usable_budget_ceiling(out_fmt)
    gated = contract.metric == Metric.ULP or any(
        d.ulp is not None and d.ulp <= ceiling for d in domains
    )

    src, golden, result = run_sweep(mathop, formats, approx_mode, dest_acc)

    mask = ulp_sweep.ranked_mask(mathop, src, golden, result, in_fmt, out_fmt, dest_acc)
    lanes_kw = dict(approx_mode=approx_mode, dest_acc=dest_acc)
    overflowed = ulp_sweep.nonfinite_failures(
        mathop, src, golden, result, in_fmt, out_fmt, **lanes_kw
    )
    if not ulp_sweep.EMIT:
        # An excused lane that agrees again means the tracked defect is gone from this
        # cell; the entry has to go with it, or its lanes stay ungated.
        stale = ulp_sweep.stale_excuses(
            mathop, src, golden, result, in_fmt, out_fmt, **lanes_kw
        )
        assert not stale, (
            f"{cell}: no lane the _KNOWN_NONFINITE_LANES entry for "
            f"{', '.join(f'#{entry.issue}' for entry in stale)} names disagrees with "
            "the golden any more; drop the entry so those lanes are gated again"
        )
    distance = ulp_distance(golden, result, flush_subnormals=FLUSH_SUBNORMAL_OUTPUTS)
    stats = ulp_stats(distance, mask)
    lanes = int(mask.sum())
    key = ulp_sweep.cell_key(in_fmt, out_fmt, approx_mode, dest_acc)

    # The overflow verdict first: an all-non-finite cell has `lanes == 0` too, and only
    # this verdict carries the lane count the headroom report reads back.
    if overflowed.any():
        verdict = ulp_sweep.nonfinite_verdict(overflowed, src, golden, result, stats)
        why = f"{verdict.why}: {verdict.nonfinite} such lane(s)"
        if verdict.measured is not None:
            why += f"; max {verdict.measured} ULP over the {lanes} measurable lane(s)"
        why += ". No budget buys an overflow, and a step count cannot describe one."
    elif lanes == 0:
        verdict = ulp_sweep.Unmeasurable(why="no lane a step count can describe")
        why = verdict.why
    else:
        verdict = None
    if verdict is not None:
        if ulp_sweep.EMIT or not gated:
            _record_ulp_measurement(
                distance, mask=mask, nonfinite=verdict.nonfinite, required=True
            )
        if ulp_sweep.EMIT:
            ulp_sweep.record_unmeasurable(mathop, key, verdict)
            return
        if not gated:
            _skip_measured(cell, f"not measurable: {why}")
        raise AssertionError(f"{cell}: {why}")

    # Each lane's input as the kernel received it, for the domains.
    x = ulp_sweep.received_inputs(src, in_fmt, out_fmt, dest_acc).to(torch.float64)
    present = ~ulp_sweep.padding_lanes(src, in_fmt)

    if ulp_sweep.EMIT:
        floor = ulp_sweep.near_zero_floor(golden, result, distance, mask, out_fmt)
        judged, bins = ulp_sweep.floored_profile(
            golden, result, x, distance, mask, present, floor[1] if floor else None
        )
        drawn = ulp_sweep.domain_figures(
            partition(bins, ceiling), x, distance, mask, judged
        )
        _record_ulp_measurement(
            distance,
            mask=mask,
            bins=bins,
            readings=ulp_sweep.entry_readings(drawn, x, distance, mask, judged),
            required=True,
        )
        ulp_sweep.record(mathop, key, int(stats["max"]), floor=floor, domains=drawn)
        return

    if not domains:
        if not gated:
            # The headroom report judges only the cells it reads: required.
            _record_ulp_measurement(distance, mask=mask, required=True)
            _skip_measured(
                cell,
                f"on the tolerance metric; measured max {int(stats['max'])} ULP over "
                f"{lanes} lanes. The headroom report holds it to its row's figure.",
            )
        # The contract's own verdict, so a near_zero_atol floor is honoured; `stats` is
        # the raw maximum, which can be a lane the floor rescued.
        assert passed_test(
            golden,
            result,
            out_fmt,
            mask=mask,
            **contract.passed_test_kwargs(flush_subnormals=FLUSH_SUBNORMAL_OUTPUTS),
        ), (
            f"{cell}: failed a {contract.max_ulp}-step budget over {lanes} swept lanes "
            f"(raw maximum before any floor: {stats['max']} ULP at flat index "
            f"{stats['worst_index']}){_LOG_TAIL}"
        )
        return

    # A list row: record the cell once with its reading over every entry (consuming the
    # one lookup above), then hold each gateable entry to its figure over the lanes
    # outside the floor, and a null entry's ranked strays to the row's largest.
    judged, bins = ulp_sweep.floored_profile(
        golden, result, x, distance, mask, present, floor_atol
    )
    readings = ulp_sweep.entry_readings(domains, x, distance, mask, judged)
    _record_ulp_measurement(
        distance, mask=mask, bins=bins, readings=readings, required=True
    )
    if not gated:
        _skip_measured(
            cell,
            f"every entry is past the ceiling; measured max {int(stats['max'])} ULP "
            f"over {lanes} lanes. The headroom report holds each entry to its figure.",
        )
    largest = largest_figure(domains)
    gate = dict(mask=None, max_ulp=None, flush_subnormals=FLUSH_SUBNORMAL_OUTPUTS)
    failed = []
    inside = torch.zeros_like(mask)
    for entry, reading in zip(domains, readings):
        if entry.ulp is None:
            continue
        band = ulp_sweep.band_mask(entry.low, entry.high, x, judged)
        inside |= band
        if entry.ulp > ceiling:
            continue  # measured, not gated: the headroom report holds it
        if not bool(band.any()):
            # Every ranked lane held by the floor passes; an entry that ranks no lane
            # at all any more judged nothing, so it fails.
            if not bool(ulp_sweep.band_mask(entry.low, entry.high, x, mask).any()):
                failed.append(
                    f"x in [{entry.low:g}, {entry.high:g}): no lane ranked against "
                    f"its figure {entry.ulp}"
                )
            continue
        gate.update(mask=band, max_ulp=entry.ulp)
        if not passed_test(golden, result, out_fmt, **gate):
            failed.append(
                f"x in [{entry.low:g}, {entry.high:g}): worst lane {reading.worst} "
                f"over its figure {entry.ulp}"
            )
    leftover = judged & ~inside
    gate.update(mask=leftover, max_ulp=largest)
    if (
        bool(leftover.any())
        and largest <= ceiling
        and not passed_test(golden, result, out_fmt, **gate)
    ):
        failed.append(
            f"{int(leftover.sum())} ranked lane(s) in a null entry, over the row's "
            f"largest figure {largest}"
        )
    assert not failed, f"{cell}: " + "; ".join(failed) + _LOG_TAIL
