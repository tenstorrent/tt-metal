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
from helpers.ulp import ulp_distance
from helpers.utils import PCC_SIGNAL_FLOOR, calculate_pcc, passed_test

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
    """The budget is the whole verdict, not one of three: a result every lane of which is
    inside a loose budget passes even though PCC is far under ``target_pcc``.

    An all-but-zero golden cannot pin this. ``1e-8`` is below ``PCC_SIGNAL_FLOOR``, so the
    pre-existing floor guard further down ``passed_test`` returns the same verdict on the
    same value and the new early return is never what made the test pass. This golden is
    well above the floor, and half the tile is shifted a whole binade -- 128 bf16 steps,
    inside ``max_ulp=128`` and nowhere near a 0.99 correlation."""
    fmt = DataFormat.Float16_b
    golden = torch.linspace(1.0, 512.0, TILE_SIZE, dtype=torch.float32).to(
        TORCH_DTYPE[fmt]
    )
    result = golden.clone()
    result[: TILE_SIZE // 2] = golden[: TILE_SIZE // 2] * 2.0

    assert float(golden.abs().max()) > PCC_SIGNAL_FLOOR
    assert calculate_pcc(result, golden) < 0.99, "PCC must be the thing being skipped"
    assert not passed_test(golden, result, fmt)  # the tolerance+PCC gate rejects it
    assert passed_test(golden, result, fmt, max_ulp=128)
    # And the budget is still a real bound one step below what the shift costs.
    assert not passed_test(golden, result, fmt, max_ulp=127, print_errors=False)


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


def test_a_fp16_budget_sees_the_subnormal_band_it_used_to_flush():
    """``ulp_elementwise_valid`` hardcoded ``flush_subnormals=True``, so the gate overrode
    the per-dtype default the metric deliberately sets to ``False`` for fp16.

    fp16 keeps its subnormals in this harness -- ``golden_generators._FTZ_THRESHOLD`` is
    ``2**-24`` there, the smallest fp16 *subnormal* -- so an fp16 golden legitimately
    carries the whole band. Collapsing it let the gate accept up to 1023 representable
    steps of error near zero under a 0-step budget.
    """
    fmt = DataFormat.Float16
    smallest = 2.0**-24
    golden = _tile(smallest, fmt)
    result = _tile(1023 * smallest, fmt)
    assert float(golden[0]) != 0.0 and float(result[0]) != 0.0

    assert int(ulp_distance(golden, result).max()) == 1022
    assert not passed_test(golden, result, fmt, max_ulp=0, print_errors=False)
    assert not passed_test(golden, result, fmt, max_ulp=1021, print_errors=False)
    assert passed_test(golden, result, fmt, max_ulp=1022)


def test_bfp8_b_is_gated_in_bf16_step_space():
    """Bfp8_b has no float dtype of its own, so the budget is counted in bfloat16 steps
    against the tensor ``passed_test`` has already cast. Nothing is ORed in beside it.
    """
    fmt = DataFormat.Bfp8_b
    golden = torch.zeros(TILE_SIZE, dtype=torch.bfloat16)
    # One large element per block, so every other lane is small relative to its own block.
    golden[0::16] = 64.0
    golden[1::16] = 0.5
    for offset in range(2, 16):
        golden[offset::16] = 0.25

    assert passed_test(golden, golden.clone(), fmt, max_ulp=0)

    # A lane wrong by far more than the budget allows must still fail.
    broken = golden.clone()
    broken[1] = 8.0
    assert not passed_test(golden, broken, fmt, max_ulp=1, print_errors=False)


def test_one_bfp8_b_step_is_two_bf16_steps():
    """Bfp8_b's 7 magnitude bits *include* an explicit leading 1, so it has 6 fractional
    bits against bfloat16's 7. A budget denominated in bf16 steps therefore buys half as
    many format steps, and an odd budget buys the same as the even one below it."""
    fmt = DataFormat.Bfp8_b
    golden = _tile(1.0, fmt)  # a constant tile: every lane is its own block maximum
    one_bfp8_step = _step(golden, 2)

    assert int(ulp_distance(golden, one_bfp8_step).max()) == 2
    assert passed_test(golden, one_bfp8_step, fmt, max_ulp=2)
    assert not passed_test(golden, one_bfp8_step, fmt, max_ulp=1, print_errors=False)


def test_a_bfp8_b_budget_does_charge_for_legal_block_quantization():
    """The price of the proxy gate, and the reason a Bfp8_b budget is only usable where
    the block quantization is exact.

    One step of the Bfp8_b lattice is ``2**(floor(log2 amax) - floor(log2 x) + 1)`` bf16
    steps, so a lane small relative to its block is many bf16 steps from the golden while
    being perfectly legal -- 69 steps for a lane at 0.06 inside a block with amax 2.77.
    The budget charges for all of it. ORing the block-aware lattice compare in would have
    hidden that, at the cost of ``max_ulp`` no longer being the enforced maximum for this
    format: a lane 69 steps out would pass a ``max_ulp=0`` budget on the lattice's say-so,
    and the reported worst lane would be one the verdict had already accepted.
    ``near_zero_atol`` cannot absorb it either -- its band is relative to the *tensor*
    maximum, and such a lane is only small relative to its own block.
    """
    fmt = DataFormat.Bfp8_b
    golden = torch.zeros(TILE_SIZE, dtype=torch.bfloat16)
    golden[0::16] = 2.77
    for offset in range(1, 16):
        golden[offset::16] = 0.06

    quantized = golden.clone()
    quantized[1::16] = float(torch.tensor(0.06 + 2.0 ** (1 - 6), dtype=torch.bfloat16))
    assert int(ulp_distance(golden, quantized).max()) == 69

    assert not passed_test(golden, quantized, fmt, max_ulp=1, print_errors=False)
    assert not passed_test(golden, quantized, fmt, max_ulp=68, print_errors=False)
    assert passed_test(golden, quantized, fmt, max_ulp=69)
    # The tolerance arm is where such an op belongs: its lattice compare is block-aware.
    assert passed_test(golden, quantized, fmt)


def test_a_bfp8_b_budget_is_exact_where_the_block_quantization_is(captured_logs):
    """The enrollable case, and the one the per-op budget registry uses: integer-valued
    results inside a single binade quantize exactly, so a 0-step budget is legitimate.

    Also the case the removed lattice OR used to blur -- with nothing ORed in, the
    reported worst lane is a lane the verdict actually judged.
    """
    fmt = DataFormat.Bfp8_b
    golden = torch.zeros(TILE_SIZE, dtype=torch.bfloat16)
    for offset in range(16):
        golden[offset::16] = float(64 + offset)  # 64..79, one binade, integer-valued

    assert passed_test(golden, golden.clone(), fmt, max_ulp=0)
    assert not passed_test(golden, _step(golden, 2), fmt, max_ulp=1, print_errors=False)
    assert "max 2 ULP @ [0] (budget 1)" in "\n".join(captured_logs)


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


def test_a_silenced_failure_does_not_append_to_the_persistent_error_log(captured_logs):
    """``print_errors=False`` asks for silence, but the ULP headline sat outside that
    guard — and its sink is ``mode="a"`` on ``test_errors.log``, which CI uploads. The
    line is still emitted so the message stays assertable; only its level drops."""
    from loguru import logger as loguru_logger

    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    errors = []
    sink = loguru_logger.add(errors.append, level="ERROR", format="{message}")
    try:
        assert not passed_test(
            golden, _step(golden, 9), fmt, max_ulp=1, print_errors=False
        )
    finally:
        loguru_logger.remove(sink)

    assert errors == [], "print_errors=False must not emit an ERROR record"
    assert "ULP budget exceeded" in "\n".join(captured_logs)


def test_a_reported_failure_still_logs_at_error_level(captured_logs):
    """``captured_logs`` is a TRACE sink, so a substring check against it cannot tell
    ``logger.error`` from the ``logger.debug`` fallback -- swap the call and it still
    passes. An ERROR-level sink is what pins the positive direction of the
    ``print_errors and not _RECORD_TEST_ORDER`` guard, mirroring
    ``test_a_silenced_failure_does_not_append_to_the_persistent_error_log``."""
    from loguru import logger as loguru_logger

    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    errors = []
    sink = loguru_logger.add(errors.append, level="ERROR", format="{message}")
    try:
        assert not passed_test(
            golden, _step(golden, 9), fmt, max_ulp=1, print_errors=True
        )
    finally:
        loguru_logger.remove(sink)

    assert any("ULP budget exceeded" in record for record in errors), errors
    assert "ULP budget exceeded" in "\n".join(captured_logs)


def test_the_headline_names_the_lane_that_failed_not_one_the_floor_rescued(
    captured_logs,
):
    """Ranking every lane let a rescued lane -- which holds the largest step count in the
    tensor by construction -- take the headline while the actual failure went unmentioned.
    With ``print_errors=False`` that line is the only output there is."""
    fmt = DataFormat.Float32
    golden = torch.linspace(1.0, 100.0, TILE_SIZE, dtype=torch.float32)
    golden[-1] = 1e-8  # rescued by the floor, enormous step count
    result = golden.clone()
    result[-1] = 2e-8
    result[5] = golden[5] + 1.0  # the real failure, far fewer steps

    assert not passed_test(
        golden, result, fmt, max_ulp=2, near_zero_atol=1e-7, print_errors=False
    )
    logged = "\n".join(captured_logs)
    assert "@ [5]" in logged, logged[:400]
    assert f"@ [{TILE_SIZE - 1}]" not in logged


def test_a_budget_alongside_multiple_l1_passes_raises():
    """``L1_to_L1_iterations`` only ever fed ``target_pcc = pow(0.99, n)``, which this arm
    returns before reaching, so a caller passing it with a budget silently lost the
    multi-pass allowance while its four siblings raised. Its default is ``1``, not
    ``None``, so the `is not None` check could not see it."""
    golden = _tile(1.0, DataFormat.Float16_b)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="L1_to_L1_iterations"
    ):
        passed_test(
            golden,
            golden.clone(),
            DataFormat.Float16_b,
            2,  # L1_to_L1_iterations, positional on purpose -- see the docstring
            max_ulp=1,
        )


def test_an_empty_tensor_is_not_reported_as_a_pass():
    """``torch.all`` on an empty mask is ``True``, and the budget arm returns before the
    ``abs().max()`` that used to raise, so an empty read-back would pass having compared
    nothing."""
    empty = torch.zeros(0, dtype=torch.bfloat16)
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="nothing to compare"
    ):
        passed_test(empty, empty.clone(), DataFormat.Float16_b, max_ulp=0)


def test_a_budget_looser_than_the_tolerance_it_replaces_warns(captured_logs):
    """The bound is the rtol half, not the atol one: ``rtol * |v|`` is about
    ``rtol * 2**mantissa_bits`` steps at large magnitude, ~6 for bf16. Past that the
    budget is looser than what it displaced and PCC is no longer behind it."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    assert passed_test(golden, _step(golden, 3), fmt, max_ulp=64)
    assert "looser than the rtol" in "\n".join(captured_logs)


def test_a_tight_budget_does_not_warn_about_the_tolerance_it_replaces(captured_logs):
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    assert passed_test(golden, _step(golden, 3), fmt, max_ulp=4)
    assert "looser than the rtol" not in "\n".join(captured_logs)
