# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the ULP gate in ``passed_test(max_ulp=...)``: every other call
site relies on ``max_ulp=None`` changing nothing, and the rest is what the gate is for.
"""

from contextlib import contextmanager

import pytest
import torch
from conftest import refuses
from helpers.format_config import DataFormat
from helpers.llk_params import format_dict
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.ulp import mantissa_bits, ulp_distance, ulp_dtype
from helpers.utils import PCC_SIGNAL_FLOOR, calculate_pcc, passed_test, tolerances

TILE_SIZE = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM

FLOAT_FORMATS = [DataFormat.Float32, DataFormat.Float16, DataFormat.Float16_b]
TORCH_DTYPE = {
    DataFormat.Float32: torch.float32,
    DataFormat.Float16: torch.float16,
    DataFormat.Float16_b: torch.bfloat16,
    DataFormat.Bfp8_b: torch.bfloat16,
}


@contextmanager
def _loguru_sink(level="TRACE"):
    """Every loguru record at or above *level*; ``caplog`` sits behind a level filter."""
    from loguru import logger as loguru_logger

    records = []
    sink_id = loguru_logger.add(records.append, level=level, format="{message}")
    try:
        yield records
    finally:
        loguru_logger.remove(sink_id)


@pytest.fixture
def captured_logs():
    with _loguru_sink() as records:
        yield records


def _logs_for(call, level="TRACE"):
    with _loguru_sink(level) as records:
        call()
    return records


def _tile(value, fmt):
    return torch.full((TILE_SIZE,), value, dtype=TORCH_DTYPE[fmt])


def _ramp(fmt, low=1.0, high=512.0):
    """A tile spanning decades, so PCC is dominated by its large-magnitude lanes."""
    return torch.linspace(low, high, TILE_SIZE, dtype=torch.float32).to(
        TORCH_DTYPE[fmt]
    )


def _step(tensor, steps):
    """*tensor* moved *steps* representable values toward +inf (or -inf if negative)."""
    target = float("inf") if steps > 0 else float("-inf")
    out = tensor.clone()
    for _ in range(abs(steps)):
        out = torch.nextafter(out, torch.full_like(out, target))
    return out


# ── The default must not move ──────────────────────────────────────────────────


@pytest.mark.parametrize("fmt", FLOAT_FORMATS, ids=lambda f: f.name)
def test_without_max_ulp_the_verdict_is_unchanged(fmt):
    golden = _tile(1.0, fmt)
    assert passed_test(golden, golden.clone(), fmt)
    assert passed_test(golden, golden + 0.01, fmt)  # inside the per-format atol of 0.05
    assert not passed_test(golden, golden + 10.0, fmt, print_errors=False)


# ── Bounds on the arguments ────────────────────────────────────────────────────


def test_an_argument_only_the_ulp_arm_reads_is_refused_without_a_budget():
    """The tolerance arm is ``torch.isclose``: a silently ignored ``mask`` would judge
    every lane."""
    fmt = DataFormat.Float16
    golden = _tile(1.0, fmt)
    for name, value in (
        ("near_zero_atol", 1e-6),
        ("flush_subnormals", True),
        ("mask", torch.ones_like(golden, dtype=torch.bool)),
    ):
        with refuses(name):
            passed_test(golden, golden.clone(), fmt, **{name: value})
    with refuses("flush_subnormals, near_zero_atol"):
        passed_test(
            golden, golden.clone(), fmt, near_zero_atol=1e-6, flush_subnormals=True
        )
    assert passed_test(golden, golden.clone(), fmt, max_ulp=0, flush_subnormals=True)
    assert passed_test(
        golden,
        golden.clone(),
        fmt,
        max_ulp=0,
        mask=torch.ones_like(golden, dtype=torch.bool),
    )


def test_a_negative_budget_or_floor_is_refused_rather_than_made_inert():
    """A negative budget fails every lane (-1 is UNMEASURABLE); a negative floor is
    inert. 0 and 0.0 stay legal: the bit-exact gate and "no floor"."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    with refuses("max_ulp must not be negative"):
        passed_test(golden, golden.clone(), fmt, max_ulp=-1)
    with refuses("near_zero_atol must not be negative"):
        passed_test(golden, golden.clone(), fmt, max_ulp=1, near_zero_atol=-1e-6)
    assert passed_test(golden, golden.clone(), fmt, max_ulp=0, near_zero_atol=0.0)


@pytest.mark.parametrize(
    "kwarg, value",
    [
        ("custom_atol", 0.13),
        ("custom_rtol", 0.05),
        ("custom_pcc_threshold", 0.97),
        ("custom_bfp4_max_ulp_diff", 2),
    ],
)
def test_a_budget_alongside_a_tolerance_override_raises(kwarg, value):
    golden = _tile(1.0, DataFormat.Float16_b)
    with refuses(kwarg):
        passed_test(
            golden, golden.clone(), DataFormat.Float16_b, max_ulp=1, **{kwarg: value}
        )


def test_a_budget_alongside_multiple_l1_passes_raises():
    """``L1_to_L1_iterations`` defaults to 1, not None, so the sibling check cannot see
    it."""
    golden = _tile(1.0, DataFormat.Float16_b)
    with refuses("L1_to_L1_iterations"):
        passed_test(
            golden,
            golden.clone(),
            DataFormat.Float16_b,
            L1_to_L1_iterations=2,
            max_ulp=1,
        )


def test_an_empty_tensor_is_refused_before_any_verdict_is_logged(captured_logs):
    """``torch.all`` on an empty mask is True, so an empty read-back would pass having
    compared nothing, and must not record a passing datapoint first."""
    empty = torch.zeros(0, dtype=torch.bfloat16)
    with refuses("nothing to compare"):
        passed_test(empty, empty.clone(), DataFormat.Float16_b, max_ulp=0)
    logged = "\n".join(captured_logs)
    assert "ULP within budget" not in logged, logged[:300]
    assert (
        "no measurable lane" not in logged and "no lane under judgement" not in logged
    )


# ── The budget itself ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("fmt", FLOAT_FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("direction", [1, -1], ids=["up", "down"])
def test_the_budget_is_an_inclusive_bound_in_both_directions(fmt, direction):
    golden = _tile(1.0, fmt)
    assert passed_test(golden, golden.clone(), fmt, max_ulp=0)
    assert not passed_test(
        golden, _step(golden, direction), fmt, max_ulp=0, print_errors=False
    )
    assert passed_test(golden, _step(golden, direction), fmt, max_ulp=1)
    assert passed_test(golden, _step(golden, 3 * direction), fmt, max_ulp=3)
    assert not passed_test(
        golden, _step(golden, 4 * direction), fmt, max_ulp=3, print_errors=False
    )


def test_one_bad_lane_fails_the_tile():
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = golden.clone()
    result[517] = _step(golden, 2)[517]
    assert not passed_test(golden, result, fmt, max_ulp=1, print_errors=False)
    assert passed_test(golden, result, fmt, max_ulp=2)


# ── Why it replaces the tolerance and PCC checks ───────────────────────────────


def test_the_budget_catches_what_the_flat_tolerance_waves_through():
    """``atol=0.05`` is ~6 bf16 steps at 1.0. The first assertion keeps the test from
    demonstrating nothing."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = _step(golden, 6)
    assert passed_test(golden, result, fmt)
    assert not passed_test(golden, result, fmt, max_ulp=1, print_errors=False)


def test_the_budget_catches_what_pcc_waves_through():
    """PCC is dominated by the large-magnitude lanes: one small element off by a whole
    binade leaves it at ~1.0."""
    fmt = DataFormat.Float16_b
    golden = _ramp(fmt)
    golden[-1] = 0.001
    result = golden.clone()
    result[-1] = 0.002
    assert calculate_pcc(result, golden) > 0.99
    assert passed_test(golden, result, fmt)
    assert not passed_test(golden, result, fmt, max_ulp=8, print_errors=False)


def _move_magnitude(tensor, steps):
    """Every magnitude moved *steps* representable values away from zero, sign kept;
    bit arithmetic, so an fp32 budget of 419,430 steps is one operation."""
    width = tensor.element_size() * 8
    bits = tensor.view(torch.int16 if width == 16 else torch.int32).to(torch.int64)
    bits &= (1 << width) - 1
    sign_bit = 1 << (width - 1)
    moved = ((bits & (sign_bit - 1)) + steps) | (bits & sign_bit)
    signed = torch.where(moved >= sign_bit, moved - (1 << width), moved)
    return signed.to(torch.int16 if width == 16 else torch.int32).view(tensor.dtype)


@pytest.mark.parametrize("fmt", FLOAT_FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("direction", [1, -1], ids=["away_from_zero", "toward_zero"])
def test_a_budget_within_the_ceiling_passes_nothing_the_tolerance_check_rejects(
    fmt, direction
):
    """N steps from a golden in binade 2**e span at most ``N * 2**(e - mantissa_bits)``,
    which is within ``rtol * |golden|`` for N up to ``rtol * 2**mantissa_bits``."""
    ceiling = tolerances[fmt].rtol * (1 << mantissa_bits(ulp_dtype(fmt)))
    budget = int(ceiling)
    low, high = (-6, 6) if fmt is DataFormat.Float16 else (-20, 20)
    magnitudes = torch.logspace(low, high, TILE_SIZE // 2, base=2.0)
    golden = torch.cat([magnitudes, -magnitudes]).to(TORCH_DTYPE[fmt])
    result = _move_magnitude(golden, direction * budget)
    assert passed_test(golden, result, fmt, max_ulp=budget)
    assert passed_test(golden, result, fmt)


def test_pcc_is_not_consulted_under_a_budget():
    """The budget is the whole verdict: a ramp well above PCC_SIGNAL_FLOOR with half of
    it shifted a whole binade is inside ``max_ulp=128`` and nowhere near 0.99."""
    fmt = DataFormat.Float16_b
    golden = _ramp(fmt)
    result = golden.clone()
    result[: TILE_SIZE // 2] = golden[: TILE_SIZE // 2] * 2.0
    assert float(golden.abs().max()) > PCC_SIGNAL_FLOOR
    assert calculate_pcc(result, golden) < 0.99
    assert not passed_test(golden, result, fmt)
    assert passed_test(golden, result, fmt, max_ulp=128)
    assert not passed_test(golden, result, fmt, max_ulp=127, print_errors=False)


# ── The near-zero floor ────────────────────────────────────────────────────────


def test_the_floor_rescues_a_cancellation_lane_without_raising_the_budget():
    fmt = DataFormat.Float32
    golden = torch.linspace(1.0, 100.0, TILE_SIZE, dtype=torch.float32)
    golden[-1] = 1e-8
    result = golden.clone()
    result[-1] = 2e-8
    assert not passed_test(golden, result, fmt, max_ulp=2, print_errors=False)
    assert passed_test(golden, result, fmt, max_ulp=2, near_zero_atol=1e-7)
    result[0] = 1.5  # not near zero: the floor buys nothing
    assert not passed_test(
        golden, result, fmt, max_ulp=2, near_zero_atol=1e-7, print_errors=False
    )


def test_the_headline_names_the_lane_that_failed_not_one_the_floor_rescued(
    captured_logs,
):
    """A rescued lane holds the largest step count by construction."""
    fmt = DataFormat.Float32
    golden = torch.linspace(1.0, 100.0, TILE_SIZE, dtype=torch.float32)
    golden[-1] = 1e-8
    result = golden.clone()
    result[-1] = 2e-8  # rescued, enormous step count
    result[5] = golden[5] + 1.0  # the real failure, far fewer steps
    assert not passed_test(
        golden, result, fmt, max_ulp=2, near_zero_atol=1e-7, print_errors=False
    )
    logged = "\n".join(captured_logs)
    assert "@ [5]" in logged, logged[:400]
    assert f"@ [{TILE_SIZE - 1}]" not in logged


def test_a_floor_looser_than_the_atol_it_replaces_warns(captured_logs):
    """``near_zero_atol`` is the atol half of the ``isclose`` this arm replaces, so a
    floor above it is strictly looser than what it displaced."""
    fmt = DataFormat.Float16_b
    golden = _tile(100.0, fmt)
    result = golden.clone()
    golden[-1], result[-1] = 0.5, 0.7
    assert int(ulp_distance(golden, result).max()) > 1
    assert passed_test(golden, result, fmt, max_ulp=1, near_zero_atol=0.2)
    assert "near_zero_atol=0.2 is looser than the atol=0.05" in "\n".join(captured_logs)
    quiet = _logs_for(
        lambda: passed_test(golden, result, fmt, max_ulp=1, near_zero_atol=0.05)
    )
    assert "is looser than the atol" not in "\n".join(quiet)


def test_the_near_zero_band_follows_the_mask_through_passed_test():
    """A large masked-out golden must not widen the floor over the lanes masked in."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = golden + 0.02
    golden.reshape(-1)[0] = 1e6
    result.reshape(-1)[0] = 1e6
    keep = torch.ones_like(golden, dtype=torch.bool)
    keep.reshape(-1)[0] = False
    kwargs = dict(max_ulp=0, near_zero_atol=0.05, print_errors=False)
    assert passed_test(
        golden, result, fmt, **kwargs
    )  # unmasked, 1e6 stretches the band
    assert not passed_test(golden, result, fmt, mask=keep, **kwargs)


def test_the_log_counts_only_the_floor_held_lanes_past_the_budget(captured_logs):
    """A covered lane inside the step budget needed no rescue."""
    fmt = DataFormat.Float32
    golden = _tile(100.0, fmt)
    golden[1], golden[2] = 1e-4, 2e-4
    result = golden.clone()
    result[1] = 1.1e-4  # far past the budget, inside the floor
    result[2] = _step(golden[2:3], 1)[0]  # covered by the floor, and within budget
    with _ulp_report_enabled():
        assert passed_test(golden, result, fmt, max_ulp=1, near_zero_atol=1e-4)
    assert "1 held by the floor" in "\n".join(captured_logs)


# ── Non-finites ────────────────────────────────────────────────────────────────


def test_matching_nans_pass_and_a_missing_nan_fails():
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    golden[3] = float("nan")
    assert passed_test(golden, golden.clone(), fmt, max_ulp=0)
    missing = golden.clone()
    missing[3] = 1.0
    assert not passed_test(golden, missing, fmt, max_ulp=0, print_errors=False)


def test_an_overflow_to_inf_fails_and_reports_a_finite_step(captured_logs):
    fmt = DataFormat.Float16_b
    golden = _tile(float(torch.finfo(torch.bfloat16).max), fmt)
    result = _tile(float("inf"), fmt)
    assert not passed_test(golden, result, fmt, max_ulp=4, print_errors=False)
    assert "1 ULP = inf" not in "\n".join(captured_logs)


# ── Format gating ──────────────────────────────────────────────────────────────


def test_a_fp16_budget_sees_the_subnormal_band_it_used_to_flush():
    """fp16 keeps its subnormal band here; collapsing it let a 0-step budget accept up
    to 1023 steps of error near zero."""
    fmt = DataFormat.Float16
    smallest = 2.0**-24
    golden, result = _tile(smallest, fmt), _tile(1023 * smallest, fmt)
    assert int(ulp_distance(golden, result).max()) == 1022
    assert not passed_test(golden, result, fmt, max_ulp=1021, print_errors=False)
    assert passed_test(golden, result, fmt, max_ulp=1022)


def test_one_bfp8_b_step_is_two_bf16_steps():
    """Bfp8_b's 7 magnitude bits include an explicit leading 1, so a bf16 budget buys
    half as many format steps."""
    fmt = DataFormat.Bfp8_b
    golden = _tile(1.0, fmt)  # a constant tile: every lane is its own block maximum
    one_bfp8_step = _step(golden, 2)
    assert int(ulp_distance(golden, one_bfp8_step).max()) == 2
    assert passed_test(golden, one_bfp8_step, fmt, max_ulp=2)
    assert not passed_test(golden, one_bfp8_step, fmt, max_ulp=1, print_errors=False)


def test_a_bfp8_b_budget_charges_for_legal_block_quantization():
    """A lane small relative to its block is many bf16 steps out while perfectly legal
    (69, measured, for 0.06 in a block with amax 2.77), so a Bfp8_b budget is usable
    only where the block quantization is exact."""
    fmt = DataFormat.Bfp8_b
    golden = torch.zeros(TILE_SIZE, dtype=torch.bfloat16)
    golden[0::16] = 2.77
    for offset in range(1, 16):
        golden[offset::16] = 0.06
    quantized = golden.clone()
    quantized[1::16] = float(torch.tensor(0.06 + 2.0 ** (1 - 6), dtype=torch.bfloat16))
    assert int(ulp_distance(golden, quantized).max()) == 69
    assert not passed_test(golden, quantized, fmt, max_ulp=68, print_errors=False)
    assert passed_test(golden, quantized, fmt, max_ulp=69)
    assert passed_test(golden, quantized, fmt)  # the tolerance arm is block-aware


def test_a_bfp8_b_budget_is_exact_where_the_block_quantization_is(captured_logs):
    fmt = DataFormat.Bfp8_b
    golden = torch.zeros(TILE_SIZE, dtype=torch.bfloat16)
    for offset in range(16):
        golden[offset::16] = float(64 + offset)  # one binade, integer-valued
    assert passed_test(golden, golden.clone(), fmt, max_ulp=0)
    assert not passed_test(golden, _step(golden, 2), fmt, max_ulp=1, print_errors=False)
    assert "max 2 ULP @ [0] (budget 1)" in "\n".join(captured_logs)


@pytest.mark.parametrize(
    "fmt", [DataFormat.Bfp4_b, DataFormat.MxFp4], ids=lambda f: f.name
)
def test_a_budget_on_a_block_format_without_a_per_element_ulp_raises(fmt):
    golden = torch.ones(TILE_SIZE, dtype=torch.bfloat16)
    with refuses("no per-element ULP"):
        passed_test(golden, golden.clone(), fmt, max_ulp=1)


def test_a_budget_on_an_integer_format_raises():
    fmt = DataFormat.Int32
    golden = torch.ones(TILE_SIZE, dtype=format_dict[fmt])
    with refuses("no per-element ULP"):
        passed_test(golden, golden.clone(), fmt, max_ulp=0)
    assert passed_test(golden, golden.clone(), fmt)


# ── Reporting ──────────────────────────────────────────────────────────────────


def test_a_pass_reports_the_distribution_too(captured_logs):
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    assert passed_test(golden, _step(golden, 1), fmt, max_ulp=2)
    logged = "\n".join(captured_logs)
    assert "ULP within budget" in logged and "max 1 ULP" in logged
    assert "100.0% exact" not in logged  # it really measured the perturbed lanes


@pytest.mark.parametrize(
    "budget, warns", [(200, True), (128, False)], ids=["past", "inside"]
)
def test_a_budget_past_the_meaningful_ceiling_warns_but_still_gates(budget, warns):
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    verdict = []
    logged = "\n".join(
        _logs_for(
            lambda: verdict.append(
                passed_test(golden, _step(golden, 3), fmt, max_ulp=budget)
            )
        )
    )
    assert ("exceeds the largest meaningful ULP budget" in logged) is warns
    assert verdict == [True]


@pytest.mark.parametrize(
    "budget, warns", [(64, True), (4, False)], ids=["loose", "tight"]
)
def test_a_budget_looser_than_the_tolerance_it_replaces_warns(budget, warns):
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    logged = "\n".join(
        _logs_for(lambda: passed_test(golden, _step(golden, 3), fmt, max_ulp=budget))
    )
    assert ("looser than the rtol" in logged) is warns


def test_the_displaced_figure_is_not_rounded_to_read_as_equal(captured_logs):
    """Bfp8_b's displaced figure is 25.6; a budget of 26 must not read as looser than
    "~26 steps"."""
    fmt = DataFormat.Bfp8_b
    golden = _tile(1.0, fmt)
    passed_test(golden, golden.clone(), fmt, max_ulp=26)
    logged = "\n".join(captured_logs)
    assert "max_ulp=26 is looser" in logged and "~25.6 steps" in logged
    assert "~26 steps" not in logged


def test_a_mask_narrows_the_gate_to_the_lanes_it_selects():
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = golden.clone()
    result.reshape(-1)[0] = _step(golden, 5).reshape(-1)[0]
    assert not passed_test(golden, result, fmt, max_ulp=0, print_errors=False)
    keep = torch.ones_like(golden, dtype=torch.bool)
    keep.reshape(-1)[0] = False
    assert passed_test(golden, result, fmt, max_ulp=0, mask=keep, print_errors=False)
    with refuses("mask must be bool"):
        passed_test(golden, result, fmt, max_ulp=0, mask=torch.ones_like(golden))


def test_a_masked_out_lane_stays_out_of_the_summary(captured_logs):
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = _step(golden, 3)
    result.reshape(-1)[0] = _step(golden, 40).reshape(-1)[0]
    keep = torch.ones_like(golden, dtype=torch.bool)
    keep.reshape(-1)[0] = False
    assert not passed_test(golden, result, fmt, max_ulp=1, mask=keep, print_errors=True)
    logged = "\n".join(captured_logs)
    assert "max 3 ULP" in logged and "max 40 ULP" not in logged


# ── --ulp-report ────────────────────────────────────────────────────────────


@contextmanager
def _ulp_report_enabled(value=True):
    """``--ulp-report`` as the plugin sets it; takes a value so a test can pin it off."""
    from helpers import utils

    previous = utils._ULP_REPORT
    utils._ULP_REPORT = value
    try:
        yield
    finally:
        utils._ULP_REPORT = previous


def test_the_report_measures_an_op_that_carries_no_budget():
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = _step(golden, 6)  # inside atol=0.05, invisible to the default gate
    with _ulp_report_enabled(False):
        quiet = _logs_for(lambda: passed_test(golden, result, fmt))
    assert not any("ULP" in record for record in quiet)
    with _ulp_report_enabled():
        logged = "\n".join(_logs_for(lambda: passed_test(golden, result, fmt)))
    assert "ULP report" in logged and "max 6 ULP" in logged and "p99" in logged


def test_the_report_raises_an_enrolled_pass_out_of_debug():
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    with _ulp_report_enabled():
        records = _logs_for(
            lambda: passed_test(golden, _step(golden, 1), fmt, max_ulp=2), "INFO"
        )
    assert any(
        "ULP within budget" in record and "max 1 ULP" in record for record in records
    )


def test_a_non_finite_failure_names_itself_in_the_log(captured_logs):
    """A NaN lane drops out of the statistics, so the disagreement must lead the line."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    golden[7] = float("nan")
    result = golden.clone()
    result[7] = 1.0
    assert not passed_test(golden, result, fmt, max_ulp=0, print_errors=False)
    logged = "\n".join(captured_logs)
    assert "non-finite disagreement @ [7]" in logged and "1 such lane(s)" in logged


def test_a_failure_logs_the_worst_lane_ahead_of_the_tile_dump(captured_logs):
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = golden.clone()
    result[42] = _step(golden, 9)[42]
    assert not passed_test(golden, result, fmt, max_ulp=1, print_errors=True)
    logged = "\n".join(captured_logs)
    assert "max 9 ULP @ [42]" in logged
    assert logged.index("ULP budget exceeded") < logged.index("Result tile")


@pytest.mark.parametrize("print_errors", [True, False], ids=["reported", "silenced"])
def test_print_errors_controls_the_level_not_the_message(print_errors):
    """The headline's ERROR sink appends to the test_errors.log CI uploads; silence
    drops the level, not the line."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = _step(golden, 9)

    def verdict():
        passed_test(golden, result, fmt, max_ulp=1, print_errors=print_errors)

    assert bool(_logs_for(verdict, "ERROR")) is print_errors
    assert "ULP budget exceeded" in "\n".join(_logs_for(verdict))


def test_a_silenced_failure_is_reported_at_info_under_the_report_flag():
    """A reporting sweep that dropped the failures would derive its budget from the
    lanes that passed: INFO under the flag, still not ERROR."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = _step(golden, 9)

    def verdict():
        passed_test(golden, result, fmt, max_ulp=1, print_errors=False)

    with _ulp_report_enabled():
        assert not _logs_for(verdict, "ERROR")
        assert "ULP budget exceeded" in "\n".join(_logs_for(verdict, "INFO"))
    with _ulp_report_enabled(False):
        assert not _logs_for(verdict, "INFO")


def test_the_report_cannot_change_a_verdict():
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    cases = [
        (golden.clone(), {}, True),
        (_step(golden, 6), {}, True),  # inside the tolerance, outside a 1-step budget
        (golden + 10.0, {"print_errors": False}, False),
        (_step(golden, 1), {"max_ulp": 0, "print_errors": False}, False),
        (_step(golden, 1), {"max_ulp": 1}, True),
    ]
    for (
        result,
        kwargs,
        expected,
    ) in cases:  # bool(): the tolerance arm returns a 0-d tensor
        assert bool(passed_test(golden, result, fmt, **kwargs)) is expected
        with _ulp_report_enabled():
            assert bool(passed_test(golden, result, fmt, **kwargs)) is expected


def test_the_report_stays_off_a_format_with_no_per_element_ulp():
    golden = torch.ones(TILE_SIZE, dtype=torch.bfloat16)
    with _ulp_report_enabled():
        logged = "\n".join(
            _logs_for(lambda: passed_test(golden, golden.clone(), DataFormat.Bfp4_b))
        )
    assert "ULP report" not in logged


def test_the_report_skips_a_shape_the_verdict_broadcast_rather_than_raising():
    """The tolerance arm broadcasts; a report that raised here would change a verdict."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = golden.clone().unsqueeze(0)  # (1, N) against (N,)
    assert bool(passed_test(golden, result, fmt))
    with _ulp_report_enabled():
        logged = "\n".join(_logs_for(lambda: passed_test(golden, result, fmt), "INFO"))
    assert "ULP report skipped" in logged and "ULP report -" not in logged
