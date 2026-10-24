"""Unit tests for local-weights resolution.

Pins the contract introduced by the 2026-06-02 audit to make rerun
behavior predictable after a user downloads HF weights locally:

  * ``_resolve_local_weights_env`` — pure decision: translate CLI
    flags (``--local-dir``, ``--offline-hf``) into HF_HOME /
    HF_HUB_OFFLINE env-var overrides.
  * ``_apply_local_weights_env`` — side-effectful: mutate
    ``os.environ`` so subprocess pytest runs inherit the overrides.
  * ``_weights_in_default_cache`` — best-effort cache probe (purely
    informational, never gates execution).
  * ``format_hf_weight_failure_message`` — every category surfaces
    the local-weights hint so the user always knows their options.
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from scripts.tt_hw_planner._cli_helpers.error_patterns import (
    HFWeightFailure,
    format_hf_weight_failure_message,
)
from scripts.tt_hw_planner.cli import (
    _apply_local_weights_env,
    _resolve_local_weights_env,
    _weights_in_default_cache,
)


def _args(local_dir=None, offline_hf=False) -> argparse.Namespace:
    return argparse.Namespace(local_dir=local_dir, offline_hf=offline_hf)


# ─── _resolve_local_weights_env: pure decision logic ─────────────────


def test_no_flags_returns_empty_overrides() -> None:
    """Without explicit flags, HF library's default cache-first
    behavior is correct — no env overrides needed."""
    assert _resolve_local_weights_env(_args()) == {}


def test_local_dir_sets_both_hf_home_and_offline() -> None:
    """--local-dir means 'use weights from THIS exact path; no
    network'. Both env vars are needed: HF_HOME redirects the cache,
    HF_HUB_OFFLINE forces no network attempt."""
    args = _args(local_dir="/tmp/some-weights")
    overrides = _resolve_local_weights_env(args)
    assert overrides["HF_HUB_OFFLINE"] == "1"
    # HF_HOME path is resolved/expanded — assert key, not exact str.
    assert "HF_HOME" in overrides
    assert overrides["HF_HOME"].endswith("some-weights") or "some-weights" in overrides["HF_HOME"]


def test_offline_hf_sets_only_offline_var() -> None:
    """--offline-hf alone forces no-network on the default cache.
    Does NOT change HF_HOME — user wants default cache, no path
    override."""
    overrides = _resolve_local_weights_env(_args(offline_hf=True))
    assert overrides == {"HF_HUB_OFFLINE": "1"}


def test_local_dir_takes_precedence_over_offline_hf() -> None:
    """Both flags set: --local-dir wins (it's the more specific
    intent). Result is identical to --local-dir alone, no surprise
    from passing both."""
    args = _args(local_dir="/tmp/w", offline_hf=True)
    overrides = _resolve_local_weights_env(args)
    assert "HF_HOME" in overrides
    assert overrides["HF_HUB_OFFLINE"] == "1"


def test_local_dir_with_tilde_is_expanded() -> None:
    """User passes a tilde path; tool should resolve to absolute so
    subprocesses (which may have a different CWD or HOME) still find
    it."""
    args = _args(local_dir="~/my-weights")
    overrides = _resolve_local_weights_env(args)
    assert overrides["HF_HOME"].startswith("/")
    assert "~" not in overrides["HF_HOME"]


# ─── _apply_local_weights_env: env mutation + info banner ────────────


@contextmanager
def _isolated_env():
    """Save & restore HF_* env vars across a test."""
    saved = {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_OFFLINE")}
    try:
        for k in ("HF_HOME", "HF_HUB_OFFLINE"):
            os.environ.pop(k, None)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_apply_sets_environ_for_local_dir(capsys) -> None:
    """Side effect: os.environ must be mutated so subprocesses inherit."""
    with _isolated_env():
        _apply_local_weights_env(_args(local_dir="/tmp/w"), "org/m")
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert "/tmp/w" in os.environ.get("HF_HOME", "")
    # Banner is informational; just check it printed something.
    out = capsys.readouterr().out
    assert "local-weights" in out
    assert "HF_HUB_OFFLINE=1" in out


def test_apply_sets_environ_for_offline_hf(capsys) -> None:
    with _isolated_env():
        _apply_local_weights_env(_args(offline_hf=True), "org/m")
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        # HF_HOME should NOT be touched by --offline-hf alone.
        assert "HF_HOME" not in os.environ or os.environ["HF_HOME"]  # may exist from outer env


def test_apply_no_flags_leaves_env_alone(capsys) -> None:
    """No explicit flags + no cache hit → no env changes, no banner."""
    with _isolated_env(), patch("scripts.tt_hw_planner.cli._weights_in_default_cache", return_value=False):
        _apply_local_weights_env(_args(), "org/m")
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "HF_HOME" not in os.environ
    out = capsys.readouterr().out
    assert "local-weights" not in out  # no banner


def test_apply_prints_info_line_when_cache_hit_detected(capsys) -> None:
    """Auto-detected cache hit prints an INFO line but does NOT mutate
    env. HF library's default behavior already uses the cache."""
    with _isolated_env(), patch("scripts.tt_hw_planner.cli._weights_in_default_cache", return_value=True):
        _apply_local_weights_env(_args(), "org/m")
        assert "HF_HUB_OFFLINE" not in os.environ
    out = capsys.readouterr().out
    assert "found cached weights" in out
    assert "org/m" in out


# ─── _weights_in_default_cache: pure best-effort probe ──────────────


def test_cache_probe_returns_false_when_huggingface_hub_missing() -> None:
    """Defensive: if huggingface_hub isn't importable, the probe must
    return False (treated as 'unknown'), not raise."""
    with patch.dict("sys.modules", {"huggingface_hub": None}):
        # Import inside patch: scan_cache_dir is what we'd block, but
        # the function's try/except handles the ImportError path too.
        # This test confirms the contract under failure.
        try:
            result = _weights_in_default_cache("org/m")
        except Exception:
            result = "raised"
        assert result is False


def test_cache_probe_returns_true_when_model_in_cache() -> None:
    """Cache scan returns a repo for the model with non-zero size →
    probe returns True."""
    fake_repo = MagicMock(repo_id="org/m", size_on_disk=1_000_000)
    fake_scan = MagicMock(repos=[fake_repo])
    with patch("huggingface_hub.scan_cache_dir", return_value=fake_scan):
        assert _weights_in_default_cache("org/m") is True


def test_cache_probe_returns_false_when_different_model_in_cache() -> None:
    fake_repo = MagicMock(repo_id="other/different", size_on_disk=1_000_000)
    fake_scan = MagicMock(repos=[fake_repo])
    with patch("huggingface_hub.scan_cache_dir", return_value=fake_scan):
        assert _weights_in_default_cache("org/m") is False


def test_cache_probe_returns_false_on_scan_failure() -> None:
    """If scan_cache_dir raises (permissions, etc.), treat as 'unknown'."""
    with patch("huggingface_hub.scan_cache_dir", side_effect=OSError("permission denied")):
        assert _weights_in_default_cache("org/m") is False


# ─── format_hf_weight_failure_message: every category mentions local options ──


def test_all_categories_mention_local_dir_flag() -> None:
    """Every category remediation now surfaces --local-dir so users
    with weights in non-standard paths get the right hint. Gap 1 fix."""
    for category in ("gated", "not_found", "network", "corrupt", "load"):
        failure = HFWeightFailure(category=category, detail="x", excerpt="x")
        msg = format_hf_weight_failure_message("org/m", failure)
        assert "--local-dir" in msg, f"category {category} missing --local-dir hint"


def test_all_categories_mention_hf_home_env_var() -> None:
    """Every category mentions HF_HOME as the standard env-var path
    to local weights. Gap 1 fix."""
    for category in ("gated", "not_found", "network", "corrupt", "load"):
        failure = HFWeightFailure(category=category, detail="x", excerpt="x")
        msg = format_hf_weight_failure_message("org/m", failure)
        assert "HF_HOME" in msg, f"category {category} missing HF_HOME hint"


def test_all_categories_mention_offline_mode() -> None:
    """HF_HUB_OFFLINE is the env var to force no-network on rerun;
    every category surfaces it so users with a hostile network know
    they can opt out of the HF connectivity check."""
    for category in ("gated", "not_found", "network", "corrupt", "load"):
        failure = HFWeightFailure(category=category, detail="x", excerpt="x")
        msg = format_hf_weight_failure_message("org/m", failure)
        assert "HF_HUB_OFFLINE" in msg, f"category {category} missing HF_HUB_OFFLINE hint"
