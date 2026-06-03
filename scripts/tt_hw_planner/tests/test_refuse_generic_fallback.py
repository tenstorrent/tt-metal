"""Tests for the refuse-generic-catch-all-fallback gate (2026-06-03).

Background: the silent path that bit seamless-m4t was
``_enforce_backend_match_quality_or_abort`` letting hf_eager (a generic
catch-all backend) through with quality="category-default" because
``routing_mode=generic`` was treated as "intentionally catch-all, not
wrong-template." That's defensible for non-iterating callers but bites
LLM-driven ``up`` runs: hf_eager scaffolds torch-wrapper stubs, the
auto-PCC tests SKIP on novel architectures, and the OUTCOME banner
declares "all graduated rc=0" with no real TT-native code.

This module pins the gate's behavior:
  * Refuse the generic-catch-all path by default — return rc=2.
  * Caller can opt in via --accept-closest-backend.
  * Auto-onboard gets a chance first; if it produces a real backend,
    the gate accepts that.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _import_cli():
    return importlib.import_module("scripts.tt_hw_planner.cli")


def _build_generic_backend():
    from scripts.tt_hw_planner.family_backends import FamilyBackend

    return FamilyBackend(
        category="TTS",
        name="hf_eager universal (TTS)",
        demo_path="models/demos/hf_eager/demo.py",
        routing_mode="generic",
        canonical_hf_id=None,
    )


class _FakeProbe:
    def __init__(self, category="TTS", pipeline_tag=None):
        self.category = category
        self.pipeline_tag = pipeline_tag
        self.raw_config = {"model_type": "novel_arch_xyz"}


def test_refuse_generic_catchall_in_auto_default(capsys):
    """Default behavior: refuse when generic catch-all would fire AND
    auto-onboard can't produce a real backend AND user didn't opt-in."""
    cli = _import_cli()

    fake_backend = _build_generic_backend()
    fake_probe = _FakeProbe()

    with patch.object(cli, "probe_model", return_value=fake_probe):
        with patch(
            "scripts.tt_hw_planner.family_backends.pick_backend_with_quality",
            return_value=(fake_backend, "category-default"),
        ):
            with patch.object(cli, "_try_auto_onboard_inline", return_value=None):
                rc = cli._enforce_backend_match_quality_or_abort(
                    "test/novel-model",
                    accept_closest=False,
                )

    out = capsys.readouterr().out
    assert rc == 2, f"expected refuse with rc=2; got {rc}"
    assert "GENERIC-CATCH-ALL FALLBACK refused" in out
    assert "--accept-closest-backend" in out, "error must show the opt-in flag"


def test_accept_closest_overrides_generic_refusal(capsys):
    """When the user explicitly passes --accept-closest-backend, the
    gate proceeds against the catch-all (CPU-only mode)."""
    cli = _import_cli()

    fake_backend = _build_generic_backend()
    fake_probe = _FakeProbe()

    with patch.object(cli, "probe_model", return_value=fake_probe):
        with patch(
            "scripts.tt_hw_planner.family_backends.pick_backend_with_quality",
            return_value=(fake_backend, "category-default"),
        ):
            rc = cli._enforce_backend_match_quality_or_abort(
                "test/novel-model",
                accept_closest=True,
            )

    out = capsys.readouterr().out
    assert rc is None, "with --accept-closest-backend, gate should proceed"
    assert "--accept-closest-backend set" in out


def test_auto_onboard_real_backend_overrides_refusal(capsys):
    """When auto-onboard drafts a real backend, that's accepted in place
    of refusal."""
    cli = _import_cli()

    fake_backend = _build_generic_backend()
    fake_probe = _FakeProbe()
    onboarded_backend = _build_generic_backend()  # any backend stand-in

    with patch.object(cli, "probe_model", return_value=fake_probe):
        with patch(
            "scripts.tt_hw_planner.family_backends.pick_backend_with_quality",
            return_value=(fake_backend, "category-default"),
        ):
            with patch.object(
                cli,
                "_try_auto_onboard_inline",
                return_value=(onboarded_backend, "exact"),
            ):
                rc = cli._enforce_backend_match_quality_or_abort(
                    "test/novel-model",
                    accept_closest=False,
                )

    out = capsys.readouterr().out
    assert rc is None, "auto-onboard should result in 'proceed'"
    assert "auto-onboard" in out.lower(), "should log that auto-onboard drafted the backend"


def test_exact_match_still_proceeds_silently():
    """An exact backend match (known model_type) must still proceed
    without the refuse-generic logic firing."""
    cli = _import_cli()

    from scripts.tt_hw_planner.family_backends import FamilyBackend

    # An exact (non-generic) backend.
    real_backend = FamilyBackend(
        category="LLM",
        name="tt_transformers",
        demo_path="models/tt_transformers/demo/simple_text_demo.py",
        routing_mode="template",
        canonical_hf_id=None,
        model_type_keys=["llama"],
    )
    fake_probe = _FakeProbe(category="LLM")
    fake_probe.raw_config = {"model_type": "llama"}

    with patch.object(cli, "probe_model", return_value=fake_probe):
        with patch(
            "scripts.tt_hw_planner.family_backends.pick_backend_with_quality",
            return_value=(real_backend, "exact"),
        ):
            rc = cli._enforce_backend_match_quality_or_abort(
                "meta/llama-3",
                accept_closest=False,
            )

    # No refusal for exact match.
    assert rc is None
