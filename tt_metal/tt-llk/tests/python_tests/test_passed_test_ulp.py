# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the ULP gate in ``passed_test(max_ulp=...)``.

No kernel, no device: ``passed_test`` takes two torch tensors and returns a bool, so the
whole verdict is testable on the host. That matters more here than for the metric itself —
``passed_test`` has ~295 call sites, and the one property every one of them depends on is
that ``max_ulp=None`` changes nothing.

The rest of the file is about what the gate is *for*. Two tests construct results that the
tolerance check and the PCC check respectively wave through, and show the step budget
catching them; those are the reason this exists, and if either ever starts passing under a
budget the gate has stopped being stronger than what it replaced.
"""

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.utils import calculate_pcc, passed_test

TILE_SIZE = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM

FLOAT_FORMATS = [DataFormat.Float32, DataFormat.Float16, DataFormat.Float16_b]
TORCH_DTYPE = {
    DataFormat.Float32: torch.float32,
    DataFormat.Float16: torch.float16,
    DataFormat.Float16_b: torch.bfloat16,
    DataFormat.Bfp8_b: torch.bfloat16,
}


@pytest.fixture
def captured_logs():
    """Every loguru record emitted during the test, whatever level the session is at.

    ``caplog`` is not enough here. ``helpers.logger`` bridges loguru into stdlib logging
    through a sink that carries its own level filter — INFO by default — so a debug
    message never reaches the stdlib logger ``caplog`` watches. A sink added here sees
    everything and is removed again before the next test.
    """
    from loguru import logger as loguru_logger

    records = []
    sink_id = loguru_logger.add(records.append, level="TRACE", format="{message}")
    try:
        yield records
    finally:
        loguru_logger.remove(sink_id)


def _tile(value, fmt):
    return torch.full((TILE_SIZE,), value, dtype=TORCH_DTYPE[fmt])


def _step(tensor, steps):
    """*tensor* moved *steps* representable values toward +inf (or -inf if negative)."""
    target = float("inf") if steps > 0 else float("-inf")
    out = tensor.clone()
    for _ in range(abs(steps)):
        out = torch.nextafter(out, torch.full_like(out, target))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The default must not move
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fmt", FLOAT_FORMATS, ids=lambda f: f.name)
def test_without_max_ulp_the_verdict_is_unchanged(fmt):
    """The property all ~295 call sites rely on."""
    golden = _tile(1.0, fmt)
    assert passed_test(golden, golden.clone(), fmt)
    # Inside the per-format atol of 0.05, which is what those call sites are gated on.
    assert passed_test(golden, golden + 0.01, fmt)
    assert not passed_test(golden, golden + 10.0, fmt, print_errors=False)


def test_near_zero_atol_alone_is_rejected():
    golden = _tile(1.0, DataFormat.Float16_b)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="does nothing on its own"
    ):
        passed_test(golden, golden.clone(), DataFormat.Float16_b, near_zero_atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# The budget itself
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fmt", FLOAT_FORMATS, ids=lambda f: f.name)
@pytest.mark.parametrize("direction", [1, -1], ids=["up", "down"])
def test_a_one_step_perturbation_fails_a_zero_step_budget(fmt, direction):
    """The proposal's P1 verification, both directions."""
    golden = _tile(1.0, fmt)
    perturbed = _step(golden, direction)
    assert not passed_test(golden, perturbed, fmt, max_ulp=0, print_errors=False)
    assert passed_test(golden, perturbed, fmt, max_ulp=1)


@pytest.mark.parametrize("fmt", FLOAT_FORMATS, ids=lambda f: f.name)
def test_the_budget_boundary_is_inclusive(fmt):
    golden = _tile(1.0, fmt)
    assert passed_test(golden, _step(golden, 3), fmt, max_ulp=3)
    assert not passed_test(golden, _step(golden, 4), fmt, max_ulp=3, print_errors=False)


@pytest.mark.parametrize("fmt", FLOAT_FORMATS, ids=lambda f: f.name)
def test_an_exact_result_passes_a_zero_step_budget(fmt):
    golden = _tile(1.0, fmt)
    assert passed_test(golden, golden.clone(), fmt, max_ulp=0)


def test_a_one_step_perturbation_in_a_single_lane_is_enough_to_fail():
    """The gate is a max over lanes, not a mean: one bad element fails the tile."""
    golden = _tile(1.0, DataFormat.Float16_b)
    result = golden.clone()
    result[517] = _step(golden, 2)[517]
    assert not passed_test(
        golden, result, DataFormat.Float16_b, max_ulp=1, print_errors=False
    )
    assert passed_test(golden, result, DataFormat.Float16_b, max_ulp=2)


# ─────────────────────────────────────────────────────────────────────────────
# Why it replaces the tolerance and PCC checks
# ─────────────────────────────────────────────────────────────────────────────


def test_the_budget_catches_what_the_flat_tolerance_waves_through():
    """``atol=0.05`` is ~6 bf16 steps at 1.0, so today's gate cannot see a 6-step error.

    Both assertions matter: the first shows the tolerance check passing this result, the
    second shows the budget rejecting it. If the first ever starts failing, this test has
    stopped demonstrating anything.
    """
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = _step(golden, 6)

    assert passed_test(golden, result, fmt)
    assert not passed_test(golden, result, fmt, max_ulp=1, print_errors=False)


def test_the_budget_catches_what_pcc_waves_through():
    """PCC is a shape metric dominated by the large-magnitude lanes.

    One small-magnitude element wrong by a factor of two — a full binade, 128 bf16 steps —
    leaves PCC indistinguishable from 1.0.
    """
    fmt = DataFormat.Float16_b
    golden = torch.linspace(1.0, 512.0, TILE_SIZE, dtype=torch.float32).to(
        TORCH_DTYPE[fmt]
    )
    golden[-1] = 0.001
    result = golden.clone()
    result[-1] = 0.002

    assert calculate_pcc(result, golden) > 0.99
    assert passed_test(golden, result, fmt)
    assert not passed_test(golden, result, fmt, max_ulp=8, print_errors=False)


def test_pcc_is_not_consulted_under_a_budget():
    """A result inside the budget passes even where PCC would not be computed at all —
    the all-but-zero golden that trips ``PCC_SIGNAL_FLOOR`` — so the budget is the whole
    verdict rather than one of three."""
    fmt = DataFormat.Float16_b
    golden = _tile(1e-8, fmt)
    assert passed_test(golden, golden.clone(), fmt, max_ulp=0)
    assert not passed_test(golden, _step(golden, 2), fmt, max_ulp=1, print_errors=False)


# ─────────────────────────────────────────────────────────────────────────────
# The near-zero floor
# ─────────────────────────────────────────────────────────────────────────────


def test_the_floor_rescues_a_cancellation_lane_without_raising_the_budget():
    fmt = DataFormat.Float32
    golden = torch.linspace(1.0, 100.0, TILE_SIZE, dtype=torch.float32)
    golden[-1] = 1e-8
    result = golden.clone()
    result[-1] = 2e-8

    assert not passed_test(golden, result, fmt, max_ulp=2, print_errors=False)
    assert passed_test(golden, result, fmt, max_ulp=2, near_zero_atol=1e-7)
    # And the floor does not buy anything for a lane that is not near zero.
    result[0] = 1.5
    assert not passed_test(
        golden, result, fmt, max_ulp=2, near_zero_atol=1e-7, print_errors=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Non-finites
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fmt", FLOAT_FORMATS, ids=lambda f: f.name)
def test_matching_nans_pass_and_a_missing_nan_fails(fmt):
    golden = _tile(1.0, fmt)
    golden[3] = float("nan")

    matching = golden.clone()
    assert passed_test(golden, matching, fmt, max_ulp=0)

    missing = golden.clone()
    missing[3] = 1.0
    assert not passed_test(golden, missing, fmt, max_ulp=0, print_errors=False)


def test_an_overflow_to_inf_fails_even_at_the_top_of_the_range():
    fmt = DataFormat.Float16_b
    golden = _tile(float(torch.finfo(torch.bfloat16).max), fmt)
    result = _tile(float("inf"), fmt)
    assert not passed_test(golden, result, fmt, max_ulp=4, print_errors=False)


def test_a_non_finite_failure_names_itself_in_the_log(captured_logs):
    """A NaN lane is unmeasurable and drops out of the statistics, so this verdict used to
    log "max 0 ULP (budget 0)" — accurate and useless. The disagreement now leads."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    golden[7] = float("nan")
    result = golden.clone()
    result[7] = 1.0

    assert not passed_test(golden, result, fmt, max_ulp=0, print_errors=False)
    logged = "\n".join(captured_logs)
    assert "non-finite disagreement @ [7]" in logged
    assert "1 such lane(s)" in logged


def test_a_failure_at_the_top_of_the_range_does_not_log_an_infinite_step(captured_logs):
    fmt = DataFormat.Float16_b
    golden = _tile(float(torch.finfo(torch.bfloat16).max), fmt)
    result = _tile(float("inf"), fmt)
    assert not passed_test(golden, result, fmt, max_ulp=4, print_errors=False)
    assert "1 ULP = inf" not in "\n".join(captured_logs)


# ─────────────────────────────────────────────────────────────────────────────
# Format gating
# ─────────────────────────────────────────────────────────────────────────────


def test_bfp8_b_is_gated_in_bf16_step_space():
    """``passed_test`` has already cast Bfp8_b to bfloat16, and Bfp8_b's 7 magnitude bits
    are bf16's mantissa width, so the budget is a real per-element criterion here."""
    fmt = DataFormat.Bfp8_b
    golden = _tile(1.0, fmt)
    assert passed_test(golden, golden.clone(), fmt, max_ulp=0)
    assert not passed_test(golden, _step(golden, 2), fmt, max_ulp=1, print_errors=False)


@pytest.mark.parametrize(
    "fmt",
    [DataFormat.Bfp4_b, DataFormat.Bfp2_b, DataFormat.MxFp8P, DataFormat.MxFp4],
    ids=lambda f: f.name,
)
def test_a_budget_on_a_block_format_without_a_per_element_ulp_raises(fmt):
    """Raised, not silently applied: these keep their block-aware lattice compares, and
    nobody should be able to think they have a step budget they do not have."""
    golden = torch.ones(TILE_SIZE, dtype=torch.bfloat16)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="no per-element ULP"
    ):
        passed_test(golden, golden.clone(), fmt, max_ulp=1)


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
    """Both cannot apply, so the conflicting argument must not be silently dropped. This
    is also the shape the budget registry needs: an op declares a step budget or a
    tolerance, never both."""
    golden = _tile(1.0, DataFormat.Float16_b)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match=f"{kwarg}"
    ):
        passed_test(
            golden,
            golden.clone(),
            DataFormat.Float16_b,
            max_ulp=1,
            **{kwarg: value},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def test_a_failure_logs_the_worst_lane_and_still_prints_the_tile(captured_logs):
    """Both, in that order: the ULP headline names the point, then the existing coloured
    tile dump shows it in context."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = golden.clone()
    result[42] = _step(golden, 9)[42]

    assert not passed_test(golden, result, fmt, max_ulp=1, print_errors=True)
    logged = "\n".join(captured_logs)
    assert "ULP budget exceeded" in logged
    assert "max 9 ULP @ [42]" in logged
    assert "budget 1" in logged
    assert "Float16_b" in logged
    # The tile printer's own output, which must still run.
    assert "Result tile" in logged
    assert "Golden tile" in logged
    assert logged.index("ULP budget exceeded") < logged.index("Result tile")


def test_a_pass_reports_the_distribution_too(captured_logs):
    """Logged on a pass as well, which is what makes a per-test accuracy report cheap."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    assert passed_test(golden, _step(golden, 1), fmt, max_ulp=2)
    logged = "\n".join(captured_logs)
    assert "ULP within budget" in logged
    assert "max 1 ULP" in logged
    assert "100.0% exact" not in logged  # it really measured the perturbed lanes


def test_a_budget_past_the_meaningful_ceiling_warns_but_still_gates(captured_logs):
    """ttnn's sanity ceiling, reached through the gate: 200 steps is past bf16's 128, so
    the two values differ by more than an order of magnitude and the op belongs on the
    tolerance metric. It is a warning, not an error — the verdict still stands."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    assert passed_test(golden, _step(golden, 3), fmt, max_ulp=200)
    logged = "\n".join(captured_logs)
    assert "exceeds the largest meaningful ULP budget" in logged
    assert "2**mantissa_bits" in logged


def test_a_budget_inside_the_ceiling_does_not_warn(captured_logs):
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    assert passed_test(golden, _step(golden, 3), fmt, max_ulp=128)
    assert "exceeds the largest meaningful" not in "\n".join(captured_logs)
