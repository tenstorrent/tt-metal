# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the ULP gate in ``passed_test(max_ulp=...)``.

No kernel, no device: ``passed_test`` takes two tensors and returns a bool. It has ~295
call sites, and the property every one of them relies on is that ``max_ulp=None`` changes
nothing. The rest is what the gate is *for*: two tests build results the tolerance check
and PCC respectively wave through, and show the step budget catching them.
"""

from contextlib import contextmanager

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import format_dict
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.ulp import INTEGER_FORMATS, ulp_distance
from helpers.utils import PCC_SIGNAL_FLOOR, calculate_pcc, passed_test

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
    """Every loguru record at or above *level*, whatever level the session is at.

    ``caplog`` is not enough: ``helpers.logger`` bridges loguru into stdlib logging
    through a sink with its own level filter, so a debug record never reaches it."""
    from loguru import logger as loguru_logger

    records = []
    sink_id = loguru_logger.add(records.append, level=level, format="{message}")
    try:
        yield records
    finally:
        loguru_logger.remove(sink_id)


@pytest.fixture
def captured_logs():
    """Every loguru record emitted during the test. See :func:`_loguru_sink`."""
    with _loguru_sink() as records:
        yield records


def _logs_for(call, level="TRACE"):
    """The records one call emits, for a test that needs two verdicts."""
    with _loguru_sink(level) as records:
        call()
    return records


def _refuses(match):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(ValueError, match=match)  # allow-pytest.raises: host-only test


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
    """The property all ~295 call sites rely on."""
    golden = _tile(1.0, fmt)
    assert passed_test(golden, golden.clone(), fmt)
    assert passed_test(golden, golden + 0.01, fmt)  # inside the per-format atol of 0.05
    assert not passed_test(golden, golden + 10.0, fmt, print_errors=False)


# ── Bounds on the arguments, not just on the result ────────────────────────────


def test_an_argument_only_the_ulp_arm_reads_is_refused_without_a_budget():
    """The tolerance arm is ``torch.isclose``, which has no flush concept and no floor, so
    either of these without a budget was accepted and did nothing at all."""
    fmt = DataFormat.Float16
    golden = _tile(1.0, fmt)
    for kwargs in ({"near_zero_atol": 1e-6}, {"flush_subnormals": True}):
        with _refuses("does nothing on its own"):
            passed_test(golden, golden.clone(), fmt, **kwargs)
    with _refuses("flush_subnormals and near_zero_atol are read only"):
        passed_test(
            golden, golden.clone(), fmt, near_zero_atol=1e-6, flush_subnormals=True
        )
    # With a budget both are accepted, so the guard has not swallowed the feature.
    assert passed_test(golden, golden.clone(), fmt, max_ulp=0, flush_subnormals=True)


@pytest.mark.parametrize("budget", [-1, -128], ids=str)
def test_a_negative_budget_is_refused_rather_than_failing_every_lane(budget):
    """``distance <= max_ulp`` is false on every lane for a negative budget, so a
    bit-identical pair failed reading like a harness bug -- and ``-1`` is exactly
    ``UNMEASURABLE``. ``0`` stays legal: that is the bit-exact gate."""
    golden = _tile(1.0, DataFormat.Float16_b)
    with _refuses("must not be negative"):
        passed_test(golden, golden.clone(), DataFormat.Float16_b, max_ulp=budget)
    assert passed_test(golden, golden.clone(), DataFormat.Float16_b, max_ulp=0)


def test_a_negative_floor_is_refused_rather_than_made_inert():
    """A negative floor makes ``magnitude <= absolute_cut`` false on every lane, so it
    fails closed -- but then nothing tells an inert floor from a real regression. ``0.0``
    stays legal as a deliberate "no floor"."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    with _refuses("near_zero_atol must not be negative"):
        passed_test(golden, golden.clone(), fmt, max_ulp=1, near_zero_atol=-1e-6)
    assert passed_test(golden, golden.clone(), fmt, max_ulp=1, near_zero_atol=0.0)


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
    """Both cannot apply, so the conflicting argument must not be silently dropped."""
    golden = _tile(1.0, DataFormat.Float16_b)
    with _refuses(kwarg):
        passed_test(
            golden, golden.clone(), DataFormat.Float16_b, max_ulp=1, **{kwarg: value}
        )


def test_a_budget_alongside_multiple_l1_passes_raises():
    """``L1_to_L1_iterations`` only ever fed ``target_pcc = pow(0.99, n)``, which this arm
    returns before reaching. Its default is ``1``, not ``None``, so the ``is not None``
    check that catches its four siblings could not see it."""
    golden = _tile(1.0, DataFormat.Float16_b)
    with _refuses("L1_to_L1_iterations"):
        passed_test(
            golden,
            golden.clone(),
            DataFormat.Float16_b,
            2,  # L1_to_L1_iterations, positional on purpose -- see the docstring
            max_ulp=1,
        )


def test_an_empty_tensor_is_refused_before_any_verdict_is_logged(captured_logs):
    """``torch.all`` on an empty mask is ``True``, so an empty read-back would pass
    having compared nothing -- and the refusal has to come *before* the reporting block,
    which otherwise recorded a passing datapoint and only then raised."""
    empty = torch.zeros(0, dtype=torch.bfloat16)
    with _refuses("nothing to compare"):
        passed_test(empty, empty.clone(), DataFormat.Float16_b, max_ulp=0)
    logged = "\n".join(captured_logs)
    assert "ULP within budget" not in logged, logged[:300]
    # Both spellings of the empty verdict: an empty tile has no unmeasurable lane either.
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
    """The gate is a max over lanes, not a mean."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = golden.clone()
    result[517] = _step(golden, 2)[517]
    assert not passed_test(golden, result, fmt, max_ulp=1, print_errors=False)
    assert passed_test(golden, result, fmt, max_ulp=2)


# ── Why it replaces the tolerance and PCC checks ───────────────────────────────


def test_the_budget_catches_what_the_flat_tolerance_waves_through():
    """``atol=0.05`` is ~6 bf16 steps at 1.0, so today's gate cannot see a 6-step error.
    Both assertions matter: if the first stops passing, this has stopped demonstrating
    anything."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = _step(golden, 6)
    assert passed_test(golden, result, fmt)
    assert not passed_test(golden, result, fmt, max_ulp=1, print_errors=False)


def test_the_budget_catches_what_pcc_waves_through():
    """PCC is a shape metric dominated by the large-magnitude lanes: one small element
    wrong by a factor of two -- a whole binade, 128 bf16 steps -- leaves it at ~1.0."""
    fmt = DataFormat.Float16_b
    golden = _ramp(fmt)
    golden[-1] = 0.001
    result = golden.clone()
    result[-1] = 0.002

    assert calculate_pcc(result, golden) > 0.99
    assert passed_test(golden, result, fmt)
    assert not passed_test(golden, result, fmt, max_ulp=8, print_errors=False)


def test_pcc_is_not_consulted_under_a_budget():
    """The budget is the whole verdict, not one of three.

    An all-but-zero golden cannot pin this: below ``PCC_SIGNAL_FLOOR`` the pre-existing
    floor guard returns the same verdict, so the early return would never be what passed
    the test. This ramp is well above the floor, and half of it is shifted a whole binade
    -- inside ``max_ulp=128`` and nowhere near a 0.99 correlation."""
    fmt = DataFormat.Float16_b
    golden = _ramp(fmt)
    result = golden.clone()
    result[: TILE_SIZE // 2] = golden[: TILE_SIZE // 2] * 2.0

    assert float(golden.abs().max()) > PCC_SIGNAL_FLOOR
    assert calculate_pcc(result, golden) < 0.99, "PCC must be the thing being skipped"
    assert not passed_test(golden, result, fmt)  # the tolerance+PCC gate rejects it
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
    # ...and buys nothing for a lane that is not near zero.
    result[0] = 1.5
    assert not passed_test(
        golden, result, fmt, max_ulp=2, near_zero_atol=1e-7, print_errors=False
    )


def test_the_headline_names_the_lane_that_failed_not_one_the_floor_rescued(
    captured_logs,
):
    """Ranking every lane let a rescued lane -- which holds the largest step count by
    construction -- take the headline while the real failure went unmentioned."""
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


def test_a_floor_looser_than_the_atol_it_replaces_warns(captured_logs):
    """``near_zero_atol`` *is* the atol half of the ``isclose`` this arm replaces,
    reintroduced inside the band, so a floor above it is strictly looser than what it
    displaced -- with PCC gone and the lane kept out of the log by the rescued mask.

    Measured: one bf16 lane at ``golden=0.5`` against ``result=0.7`` is 51 steps, rescued
    under a 1-step budget, where the tolerance would have rejected it at 0.085 -- a silent
    40% error. The absolute cut cannot close it: both intervals scale with the atol and
    are anchored at zero, so they always overlap."""
    fmt = DataFormat.Float16_b
    golden = _tile(100.0, fmt)
    result = golden.clone()
    golden[-1], result[-1] = 0.5, 0.7

    assert int(ulp_distance(golden, result).max()) > 1
    assert passed_test(golden, result, fmt, max_ulp=1, near_zero_atol=0.2)
    assert "near_zero_atol=0.2 is looser than the atol=0.05" in "\n".join(captured_logs)

    # A floor at or under the atol it replaces is the intended use, and stays quiet.
    quiet = _logs_for(
        lambda: passed_test(golden, result, fmt, max_ulp=1, near_zero_atol=0.05)
    )
    assert "is looser than the atol" not in "\n".join(quiet)


# ── Non-finites ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fmt", FLOAT_FORMATS, ids=lambda f: f.name)
def test_matching_nans_pass_and_a_missing_nan_fails(fmt):
    golden = _tile(1.0, fmt)
    golden[3] = float("nan")
    assert passed_test(golden, golden.clone(), fmt, max_ulp=0)

    missing = golden.clone()
    missing[3] = 1.0
    assert not passed_test(golden, missing, fmt, max_ulp=0, print_errors=False)


def test_a_non_finite_failure_names_itself_in_the_log(captured_logs):
    """A NaN lane is unmeasurable and drops out of the statistics, so this verdict used to
    log "max 0 ULP (budget 0)" -- accurate and useless. The disagreement now leads."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    golden[7] = float("nan")
    result = golden.clone()
    result[7] = 1.0

    assert not passed_test(golden, result, fmt, max_ulp=0, print_errors=False)
    logged = "\n".join(captured_logs)
    assert "non-finite disagreement @ [7]" in logged and "1 such lane(s)" in logged


def test_an_overflow_to_inf_fails_and_reports_a_finite_step(captured_logs):
    """No budget buys an overflow -- and the step it is measured against has to be a
    number, since ``nextafter`` from the largest finite goes to ``Inf``."""
    fmt = DataFormat.Float16_b
    golden = _tile(float(torch.finfo(torch.bfloat16).max), fmt)
    result = _tile(float("inf"), fmt)
    assert not passed_test(golden, result, fmt, max_ulp=4, print_errors=False)
    assert "1 ULP = inf" not in "\n".join(captured_logs)


# ── Format gating ──────────────────────────────────────────────────────────────


def test_a_fp16_budget_sees_the_subnormal_band_it_used_to_flush():
    """``ulp_elementwise_valid`` hardcoded ``flush_subnormals=True``, overriding the
    per-dtype default of ``False`` for fp16 -- which keeps its whole subnormal band here.
    Collapsing it let a 0-step budget accept up to 1023 steps of error near zero."""
    fmt = DataFormat.Float16
    smallest = 2.0**-24
    golden, result = _tile(smallest, fmt), _tile(1023 * smallest, fmt)

    assert int(ulp_distance(golden, result).max()) == 1022
    assert not passed_test(golden, result, fmt, max_ulp=1021, print_errors=False)
    assert passed_test(golden, result, fmt, max_ulp=1022)


def test_one_bfp8_b_step_is_two_bf16_steps():
    """Bfp8_b has no float dtype of its own, so the budget is counted in bfloat16 steps
    against the tensor ``passed_test`` already cast -- and its 7 magnitude bits *include*
    an explicit leading 1, leaving 6 fractional against bfloat16's 7. So a bf16 budget
    buys half as many format steps, and an odd one buys the same as the even below it.
    """
    fmt = DataFormat.Bfp8_b
    golden = _tile(1.0, fmt)  # a constant tile: every lane is its own block maximum
    one_bfp8_step = _step(golden, 2)

    assert int(ulp_distance(golden, one_bfp8_step).max()) == 2
    assert passed_test(golden, one_bfp8_step, fmt, max_ulp=2)
    assert not passed_test(golden, one_bfp8_step, fmt, max_ulp=1, print_errors=False)


def test_a_bfp8_b_budget_charges_for_legal_block_quantization():
    """Why a Bfp8_b budget is only usable where the block quantization is exact: a lane
    small relative to its block is many bf16 steps out while being perfectly legal -- 69,
    measured, for a lane at 0.06 in a block with amax 2.77. (Measured, not a closed form:
    ``2**(e_amax - e_x + 1)`` is a ratio of step *sizes*, a step count only inside one
    binade, and says 128 here.)

    ORing the lattice compare in would hide it, at the cost of ``max_ulp`` no longer
    being the enforced maximum. ``near_zero_atol`` cannot absorb it either: its band is
    relative to the *tensor* maximum, and such a lane is only small within its block.
    """
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
    # The tolerance arm is where such an op belongs: its lattice compare is block-aware.
    assert passed_test(golden, quantized, fmt)


def test_a_bfp8_b_budget_is_exact_where_the_block_quantization_is(captured_logs):
    """The enrollable case the budget registry uses: integer-valued results inside one
    binade quantize exactly, so a 0-step budget is legitimate."""
    fmt = DataFormat.Bfp8_b
    golden = torch.zeros(TILE_SIZE, dtype=torch.bfloat16)
    for offset in range(16):
        golden[offset::16] = float(64 + offset)  # 64..79: one binade, integer-valued

    assert passed_test(golden, golden.clone(), fmt, max_ulp=0)
    assert not passed_test(golden, _step(golden, 2), fmt, max_ulp=1, print_errors=False)
    assert "max 2 ULP @ [0] (budget 1)" in "\n".join(captured_logs)


@pytest.mark.parametrize(
    "fmt",
    [DataFormat.Bfp4_b, DataFormat.Bfp2_b, DataFormat.MxFp8P, DataFormat.MxFp4],
    ids=lambda f: f.name,
)
def test_a_budget_on_a_block_format_without_a_per_element_ulp_raises(fmt):
    """Raised, not silently applied: these keep their block-aware lattice compares."""
    golden = torch.ones(TILE_SIZE, dtype=torch.bfloat16)
    with _refuses("no per-element ULP"):
        passed_test(golden, golden.clone(), fmt, max_ulp=1)


# ── Reporting ──────────────────────────────────────────────────────────────────


def test_a_failure_logs_the_worst_lane_ahead_of_the_tile_dump(captured_logs):
    """In that order: the headline names the point, then the tile dump shows context."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = golden.clone()
    result[42] = _step(golden, 9)[42]

    assert not passed_test(golden, result, fmt, max_ulp=1, print_errors=True)
    logged = "\n".join(captured_logs)
    assert "max 9 ULP @ [42]" in logged
    assert logged.index("ULP budget exceeded") < logged.index("Result tile")


def test_a_pass_reports_the_distribution_too(captured_logs):
    """Logged on a pass as well, which is what makes a per-test accuracy report cheap."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    assert passed_test(golden, _step(golden, 1), fmt, max_ulp=2)
    logged = "\n".join(captured_logs)
    assert "ULP within budget" in logged and "max 1 ULP" in logged
    assert "100.0% exact" not in logged  # it really measured the perturbed lanes


@pytest.mark.parametrize("print_errors", [True, False], ids=["reported", "silenced"])
def test_print_errors_controls_the_level_not_the_message(print_errors):
    """``print_errors=False`` asks for silence, but the ULP headline sat outside that
    guard -- and its sink appends to the ``test_errors.log`` CI uploads. The line is
    still emitted either way so it stays assertable; only its level drops. An ERROR sink
    is what pins that: a TRACE sink cannot tell ``logger.error`` from ``logger.debug``.
    """
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = _step(golden, 9)

    def verdict():
        passed_test(golden, result, fmt, max_ulp=1, print_errors=print_errors)

    assert bool(_logs_for(verdict, "ERROR")) is print_errors
    assert "ULP budget exceeded" in "\n".join(_logs_for(verdict))


@pytest.mark.parametrize(
    "budget, warns", [(200, True), (128, False)], ids=["past", "inside"]
)
def test_a_budget_past_the_meaningful_ceiling_warns_but_still_gates(budget, warns):
    """ttnn's sanity ceiling through the gate: 200 steps is past bf16's 128, so the two
    values differ by more than a binade. A warning, not an error: the verdict stands."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    logged = "\n".join(
        _logs_for(lambda: passed_test(golden, _step(golden, 3), fmt, max_ulp=budget))
    )
    assert ("exceeds the largest meaningful ULP budget" in logged) is warns


@pytest.mark.parametrize(
    "budget, warns", [(64, True), (4, False)], ids=["loose", "tight"]
)
def test_a_budget_looser_than_the_tolerance_it_replaces_warns(budget, warns):
    """The bound is the rtol half -- ``rtol * 2**mantissa_bits`` steps at large
    magnitude, ~6 for bf16 -- past which the budget is looser than what it displaced."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    logged = "\n".join(
        _logs_for(lambda: passed_test(golden, _step(golden, 3), fmt, max_ulp=budget))
    )
    assert ("looser than the rtol" in logged) is warns


def test_the_displaced_figure_is_not_rounded_to_read_as_equal(captured_logs):
    """Bfp8_b's ``displaced`` is 25.6, and ``{:.0f}`` printed the minimal triggering
    budget of 26 as looser than "~26 steps" -- looser than a figure shown equal to it.
    """
    fmt = DataFormat.Bfp8_b
    golden = _tile(1.0, fmt)
    passed_test(golden, golden.clone(), fmt, max_ulp=26)
    logged = "\n".join(captured_logs)
    assert "max_ulp=26 is looser" in logged and "~25.6 steps" in logged
    assert "~26 steps" not in logged


# ── Integers are not ULP territory ──────────────────────────────────────────


@pytest.mark.parametrize("fmt", INTEGER_FORMATS, ids=lambda f: f.name)
def test_a_budget_on_an_integer_format_raises(fmt):
    """ULP is not a weaker gate for an integer format, it is a meaningless one: the
    values are exact and the only sensible verdict is bit equality."""
    golden = torch.ones(TILE_SIZE, dtype=format_dict[fmt])
    with _refuses("no per-element ULP"):
        passed_test(golden, golden.clone(), fmt, max_ulp=0)
    # ...and refusing the budget must not have broken the ordinary path.
    assert passed_test(golden, golden.clone(), fmt)


# ── --ulp-report ────────────────────────────────────────────────────────────


@contextmanager
def _ulp_report_enabled():
    """``--ulp-report`` as the plugin sets it, restored afterwards."""
    from helpers import utils

    previous = utils._ULP_REPORT
    utils._ULP_REPORT = True
    try:
        yield
    finally:
        utils._ULP_REPORT = previous


def test_the_report_measures_an_op_that_carries_no_budget():
    """Where the flag earns its keep: an op still on the tolerance metric has no number
    watching its drift, so the report is the only signal before someone picks one."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    result = _step(golden, 6)  # inside atol=0.05, invisible to the default gate

    quiet = _logs_for(lambda: passed_test(golden, result, fmt))
    assert not any("ULP" in record for record in quiet)

    with _ulp_report_enabled():
        logged = "\n".join(_logs_for(lambda: passed_test(golden, result, fmt)))
    assert "ULP report" in logged
    # Max *and* the distribution behind it: a p99 is what separates one unlucky lane
    # from a drift across the tile, and it is the number the report exists to trend.
    assert "max 6 ULP" in logged
    assert "p99" in logged


def test_the_report_raises_an_enrolled_pass_out_of_debug():
    """With a budget the summary already existed, at a level CI drops. The flag is the
    ask for it, so it comes back at INFO -- still on the pass path."""
    fmt = DataFormat.Float16_b
    golden = _tile(1.0, fmt)
    with _ulp_report_enabled():
        records = _logs_for(
            lambda: passed_test(golden, _step(golden, 1), fmt, max_ulp=2), "INFO"
        )
    assert any(
        "ULP within budget" in record and "max 1 ULP" in record for record in records
    )


@pytest.mark.parametrize("fmt", FLOAT_FORMATS, ids=lambda f: f.name)
def test_the_report_cannot_change_a_verdict(fmt):
    """Reporting only. It runs after the verdict and is never read back into one, so
    every pass and every failure has to land the same way with the flag on."""
    golden = _tile(1.0, fmt)
    cases = [
        (golden.clone(), {}, True),
        (_step(golden, 6), {}, True),  # inside the tolerance, outside a 1-step budget
        (golden + 10.0, {"print_errors": False}, False),
        (_step(golden, 1), {"max_ulp": 0, "print_errors": False}, False),
        (_step(golden, 1), {"max_ulp": 1}, True),
    ]
    # `bool(...)`: the tolerance arm returns a 0-d tensor rather than a Python bool.
    for result, kwargs, expected in cases:
        assert bool(passed_test(golden, result, fmt, **kwargs)) is expected
        with _ulp_report_enabled():
            assert bool(passed_test(golden, result, fmt, **kwargs)) is expected


def test_the_report_stays_off_a_format_with_no_per_element_ulp():
    """Nothing to measure: a Bfp4_b verdict has no per-element step count, and asking
    for one would raise inside a path that must not be able to fail a test."""
    golden = torch.ones(TILE_SIZE, dtype=torch.bfloat16)
    with _ulp_report_enabled():
        logged = "\n".join(
            _logs_for(lambda: passed_test(golden, golden.clone(), DataFormat.Bfp4_b))
        )
    assert "ULP report" not in logged
