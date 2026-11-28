"""An empty raw_config has three causes that need three different answers.

Regression pins: a public LoRA repo was told to set HF_TOKEN (an access fix for a
non-access problem), and a composite was told the same. Metadata is published for
gated repos, so probe fields prove nothing about file readability -- the hub's own
error type must decide.
"""

from __future__ import annotations

from scripts.tt_hw_planner import probe as pr


class _Probe:
    """Minimal stand-in: only the fields missing_config_reason reads."""

    def __init__(self, is_composite=False, submodels=None, pipeline_tag="text-generation"):
        self.is_composite = is_composite
        self.submodels = submodels or []
        self.pipeline_tag = pipeline_tag
        self.weight_bytes_total = 1


def test_composite_is_explained_as_composite(monkeypatch) -> None:
    monkeypatch.setattr(pr, "_repo_access_status", lambda *a, **k: "ok")
    out = pr.missing_config_reason(_Probe(is_composite=True, submodels=["transformer", "vae"]), "org/pipe")
    assert "composite" in out
    assert "transformer" in out and "vae" in out
    assert "HF_TOKEN" not in out


def test_readable_repo_without_identity_is_not_an_access_problem(monkeypatch) -> None:
    """The LoRA case: files readable, no config anywhere."""
    monkeypatch.setattr(pr, "_repo_access_status", lambda *a, **k: "absent")
    out = pr.missing_config_reason(_Probe(), "org/some-lora")
    assert "not a standalone model" in out
    assert "HF_TOKEN" not in out
    assert "huggingface-cli login" not in out


def test_denied_repo_is_an_access_problem(monkeypatch) -> None:
    """Gated repos publish metadata, so only the error type reveals this."""
    monkeypatch.setattr(pr, "_repo_access_status", lambda *a, **k: "denied")
    out = pr.missing_config_reason(_Probe(), "org/gated")
    assert "denied" in out
    assert "HF_TOKEN" in out


def test_unknown_status_does_not_assert_a_cause(monkeypatch) -> None:
    monkeypatch.setattr(pr, "_repo_access_status", lambda *a, **k: "unknown")
    out = pr.missing_config_reason(_Probe(), "org/mystery")
    assert "could not be determined" in out


def test_metadata_presence_never_decides_readability(monkeypatch) -> None:
    """A fully-populated probe must still be called denied when access is denied --
    this is the exact confusion that mislabelled a gated model as 'not a model'."""
    monkeypatch.setattr(pr, "_repo_access_status", lambda *a, **k: "denied")
    rich = _Probe(pipeline_tag="text-generation")
    rich.weight_bytes_total = 10**10
    assert "denied" in pr.missing_config_reason(rich, "org/gated-but-listed")
