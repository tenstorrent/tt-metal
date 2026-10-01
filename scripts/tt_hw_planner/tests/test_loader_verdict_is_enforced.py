"""A verdict nothing acts on is a comment.

The gate could already tell that a generated reference contradicted itself or the constants its own
checkpoint declares. Nothing downstream asked. `_load_reference_module` gated on whether the loader
FILE existed -- which it does, broken or not -- so a condemned reference was loaded and used to
enumerate the module tree anyway, and every PCC measured against it inherited the fault.
"""

from __future__ import annotations

import importlib.util

import pytest

from scripts.tt_hw_planner import module_tree
from scripts.tt_hw_planner import reference_loader_resolver as rlr

requires_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")

_BLOCKER = ".loader_blocker.txt"


def _decoder(*, acausal: bool):
    import torch
    import torch.nn as nn

    class Cfg:
        vocab_size, hidden_size, is_decoder = 32, 8, True

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()
            self.embed = nn.Embedding(Cfg.vocab_size, Cfg.hidden_size)

        def forward(self, input_ids: torch.LongTensor):
            h = self.embed(input_ids)
            # Acausal: every position averaged over the whole sequence, so each one sees the future.
            return h.mean(dim=1, keepdim=True).expand_as(h) if acausal else torch.cumsum(h, dim=1)

    return LM().eval()


@pytest.fixture
def loaded(monkeypatch, tmp_path):
    """Drive `_load_reference_module` with a chosen reference and a loader file that exists."""

    def _run(ref):
        monkeypatch.setattr(rlr, "has_loader", lambda *a, **k: True)
        monkeypatch.setattr(rlr, "load_reference", lambda *a, **k: ref)
        monkeypatch.setattr(rlr, "loader_path", lambda d: tmp_path / "_reference_loader.py")
        return module_tree._load_reference_module("some/model", demo_dir=tmp_path)

    return _run


@requires_torch
def test_a_condemned_reference_is_refused_rather_than_used(loaded, tmp_path) -> None:
    assert loaded(_decoder(acausal=True)) is None
    blocker = tmp_path / _BLOCKER
    assert blocker.exists(), "refused it without leaving any trace of why"
    assert "causality" in blocker.read_text(), blocker.read_text()


@requires_torch
def test_a_sound_reference_still_passes_straight_through(loaded, tmp_path) -> None:
    """The gate must not become a wall: refusing everything would be its own kind of broken."""
    assert loaded(_decoder(acausal=False)) is not None
    assert not (tmp_path / _BLOCKER).exists()


@requires_torch
def test_the_verdict_can_be_asked_of_a_model_already_built() -> None:
    """`assess` exists so consulting the verdict does not mean loading a 4B model twice."""
    assert rlr.assess(_decoder(acausal=False), "some/model")["ok"] is True
    condemned = rlr.assess(_decoder(acausal=True), "some/model")
    assert condemned["ok"] is False and "causality" in condemned["reason"]
