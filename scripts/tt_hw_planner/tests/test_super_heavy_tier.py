"""Unit tests for the 3-tier model escalation (haiku → sonnet → opus).

Previously the tiered switching was 2-tier only (haiku → sonnet for
claude). Opus was opt-in via explicit ``--auto-model-heavy=opus``.

Phi-3.5 attention stuck on the same TypeError across 8 sonnet iters
showed that sonnet alone wasn't always enough — the LLM needs deeper
reasoning to break out of "patches that look right but produce the
same error" cycles.

This module adds a 3rd tier (super_heavy = opus by default) with
auto-escalation triggers:
  * attempts_so_far ≥ 5: heavy has had 3+ shots without convergence
  * consecutive_same_class ≥ 3: heavy is producing patches that all
    hit the same error class (stuck pattern)
"""

from __future__ import annotations

import pytest


# ─── _pick_agent_model_for_iter — super_heavy escalation ────────────


def test_super_heavy_fires_on_attempts_5():
    from scripts.tt_hw_planner._cli_helpers.agent import _pick_agent_model_for_iter

    model, reason = _pick_agent_model_for_iter(
        model_default="haiku",
        model_light="haiku",
        model_heavy="sonnet",
        model_super_heavy="opus",
        complexity_bonus=0,
        failure_class="API_SIGNATURE",
        attempts_so_far=5,
    )
    assert model == "opus"
    assert "super_heavy" in reason
    assert "attempts=5" in reason


def test_super_heavy_fires_on_consecutive_same_class_threshold():
    """consec_same_class >= threshold triggers opus even when attempts
    are still below the attempts threshold. Tests with attempts=2
    (below 3) and consec=2 (at threshold) so the trigger is unambiguous.
    """
    from scripts.tt_hw_planner._cli_helpers.agent import (
        _SUPER_HEAVY_CONSEC_SAME_CLASS_THRESHOLD,
        _pick_agent_model_for_iter,
    )

    model, reason = _pick_agent_model_for_iter(
        model_default="haiku",
        model_light="haiku",
        model_heavy="sonnet",
        model_super_heavy="opus",
        complexity_bonus=0,
        failure_class="API_SIGNATURE",
        attempts_so_far=2,
        consecutive_same_class=_SUPER_HEAVY_CONSEC_SAME_CLASS_THRESHOLD,
    )
    assert model == "opus"
    assert "super_heavy" in reason
    assert "consec_same" in reason


def test_super_heavy_not_fires_when_thresholds_not_met():
    from scripts.tt_hw_planner._cli_helpers.agent import (
        _SUPER_HEAVY_ATTEMPTS_THRESHOLD,
        _SUPER_HEAVY_CONSEC_SAME_CLASS_THRESHOLD,
        _pick_agent_model_for_iter,
    )

    model, reason = _pick_agent_model_for_iter(
        model_default="haiku",
        model_light="haiku",
        model_heavy="sonnet",
        model_super_heavy="opus",
        complexity_bonus=0,
        failure_class="API_SIGNATURE",
        attempts_so_far=_SUPER_HEAVY_ATTEMPTS_THRESHOLD - 1,
        consecutive_same_class=_SUPER_HEAVY_CONSEC_SAME_CLASS_THRESHOLD - 1,
    )
    # Both below threshold → heavy escalation only, not super_heavy
    assert model == "sonnet"


def test_super_heavy_only_fires_when_explicitly_enabled():
    """Without model_super_heavy passed, system stays 2-tier (haiku → sonnet)
    even at high attempt counts."""
    from scripts.tt_hw_planner._cli_helpers.agent import _pick_agent_model_for_iter

    model, reason = _pick_agent_model_for_iter(
        model_default="haiku",
        model_light="haiku",
        model_heavy="sonnet",
        # NO model_super_heavy
        complexity_bonus=0,
        failure_class="API_SIGNATURE",
        attempts_so_far=10,
        consecutive_same_class=5,
    )
    assert model == "sonnet"  # caps at heavy tier
    assert "heavy" in reason
    assert "opus" not in reason


def test_light_still_fires_on_iter_1_with_super_heavy_enabled():
    """Adding super_heavy must not break the light tier — iter 1 still
    starts with haiku."""
    from scripts.tt_hw_planner._cli_helpers.agent import _pick_agent_model_for_iter

    model, reason = _pick_agent_model_for_iter(
        model_default="haiku",
        model_light="haiku",
        model_heavy="sonnet",
        model_super_heavy="opus",
        complexity_bonus=0,
        failure_class="OTHER",
        attempts_so_far=0,
        consecutive_same_class=0,
    )
    assert model == "haiku"
    assert reason == "light"


def test_tier_ladder_end_to_end():
    """End-to-end tier ladder smoke test.

    With the 2026-06-03 thresholds (attempts=3, consec=2), the ladder
    is:
      attempts=0 → haiku   (light, iter 1)
      attempts=1 → haiku   (light, no heavy trigger yet)
      attempts=2 → sonnet  (heavy:attempts=2)
      attempts=3 → opus    (super_heavy:attempts=3)
      attempts≥3 → opus    (continues)

    Opus engages a full iter earlier than the old (5, 3) defaults,
    matching the Phi-3.5 attention observation that two sonnet iters
    on the same STATE_DICT_KEY didn't break through."""
    from scripts.tt_hw_planner._cli_helpers.agent import (
        _SUPER_HEAVY_ATTEMPTS_THRESHOLD,
        _pick_agent_model_for_iter,
    )

    kwargs = dict(
        model_default="haiku",
        model_light="haiku",
        model_heavy="sonnet",
        model_super_heavy="opus",
        complexity_bonus=0,
        failure_class="API_SIGNATURE",
        consecutive_same_class=0,
    )
    # iter 1: light
    assert _pick_agent_model_for_iter(attempts_so_far=0, **kwargs)[0] == "haiku"
    assert _pick_agent_model_for_iter(attempts_so_far=1, **kwargs)[0] == "haiku"
    # iter 2: heavy
    assert _pick_agent_model_for_iter(attempts_so_far=2, **kwargs)[0] == "sonnet"
    # iter 3+ (at attempts threshold): super_heavy
    assert _pick_agent_model_for_iter(attempts_so_far=_SUPER_HEAVY_ATTEMPTS_THRESHOLD, **kwargs)[0] == "opus"
    assert _pick_agent_model_for_iter(attempts_so_far=10, **kwargs)[0] == "opus"


# ─── _resolve_tiered_model_aliases — provider defaults ──────────────


def test_claude_tiered_defaults_are_sonnet_opus():
    """2026-06-03: haiku was dropped from the iter-loop reasoning
    ladder (Phi-3.5 attention case showed iter-1 haiku output was
    systematically thrown away by later sonnet iters). The claude
    tiered ladder is now sonnet → opus; both light and heavy resolve
    to sonnet so the picker's existing tier-discrimination code paths
    still work."""
    from scripts.tt_hw_planner._cli_helpers.agent import _resolve_tiered_model_aliases

    light, heavy, super_heavy = _resolve_tiered_model_aliases(
        provider="claude",
        auto_model="sonnet",
        auto_model_light=None,
        auto_model_heavy=None,
        auto_model_super_heavy=None,
        auto_model_tiered=True,
    )
    assert light == "sonnet", "iter-1 starts at sonnet (haiku no longer in reasoning ladder)"
    assert heavy == "sonnet", "heavy tier also resolves to sonnet — opus is via super_heavy"
    assert super_heavy == "opus", "claude tiered mode must default super_heavy=opus"


def test_explicit_super_heavy_overrides_default():
    from scripts.tt_hw_planner._cli_helpers.agent import _resolve_tiered_model_aliases

    _, _, super_heavy = _resolve_tiered_model_aliases(
        provider="claude",
        auto_model="haiku",
        auto_model_light=None,
        auto_model_heavy=None,
        auto_model_super_heavy="opus-experimental",
        auto_model_tiered=True,
    )
    assert super_heavy == "opus-experimental"


def test_disabled_tiered_returns_none_triple():
    from scripts.tt_hw_planner._cli_helpers.agent import _resolve_tiered_model_aliases

    light, heavy, super_heavy = _resolve_tiered_model_aliases(
        provider="claude",
        auto_model="haiku",
        auto_model_light=None,
        auto_model_heavy=None,
        auto_model_super_heavy=None,
        auto_model_tiered=False,
    )
    assert (light, heavy, super_heavy) == (None, None, None)


def test_cursor_provider_does_not_default_super_heavy():
    """Cursor's tier ladder is different and we don't have an opus-like
    super tier mapped for it yet."""
    from scripts.tt_hw_planner._cli_helpers.agent import _resolve_tiered_model_aliases

    _, _, super_heavy = _resolve_tiered_model_aliases(
        provider="cursor",
        auto_model="sonnet-4",
        auto_model_light=None,
        auto_model_heavy=None,
        auto_model_super_heavy=None,
        auto_model_tiered=True,
    )
    assert super_heavy is None


# ─── Phi-3.5 attention scenario ────────────────────────────────────


def test_phi35_attention_scenario_escalates_to_opus():
    """End-to-end check: Phi-3.5 attention's stuck pattern (attempts ≥ 8,
    consec_same ≥ 5) should fire super_heavy=opus."""
    from scripts.tt_hw_planner._cli_helpers.agent import _pick_agent_model_for_iter

    model, reason = _pick_agent_model_for_iter(
        model_default="haiku",
        model_light="haiku",
        model_heavy="sonnet",
        model_super_heavy="opus",
        complexity_bonus=0,
        failure_class="API_SIGNATURE",
        attempts_so_far=8,
        consecutive_same_class=5,
    )
    assert model == "opus", (
        "Phi-3.5 attention stuck at attempts=8 + consec=5 must escalate to opus; "
        "without this, sonnet keeps reproducing the same TypeError"
    )
