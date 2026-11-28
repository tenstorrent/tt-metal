# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""publish-hf writes the id HF serves the repo under now, not the one it was given.

MiniCPM5-2B, 2026-11-10: the HF account was renamed and the old id only redirected; a republish given the old
id would have written it into the card's pull/serve commands and the manifest.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.commands import publish_hf as ph  # noqa: E402


def _hf(monkeypatch, info):
    hub = pytest.importorskip("huggingface_hub")

    class _Api:
        def __init__(self, token=None):
            pass

        def model_info(self, repo_id):
            return info(repo_id)

    monkeypatch.setattr(hub, "HfApi", _Api)


def test_a_renamed_repo_is_named_by_its_current_id(monkeypatch):
    _hf(monkeypatch, lambda repo_id: SimpleNamespace(id="new-owner/model-tt"))
    assert ph._current_repo_id("old-owner/model-tt", None) == "new-owner/model-tt"


def test_an_id_hf_cannot_resolve_is_kept(monkeypatch):
    def missing(repo_id):
        raise OSError("404")

    _hf(monkeypatch, missing)
    assert ph._current_repo_id("owner/new-model-tt", None) == "owner/new-model-tt"
    _hf(monkeypatch, lambda repo_id: SimpleNamespace(id=None))
    assert ph._current_repo_id("owner/model-tt", None) == "owner/model-tt"
