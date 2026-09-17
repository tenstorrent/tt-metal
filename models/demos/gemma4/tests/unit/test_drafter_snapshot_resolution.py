# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drafter checkpoints resolve from the HF cache the RUNNER configured.

``~/.cache/huggingface/hub`` is only the default. The tt-metal
vllm-model-tests runner sets ``HF_HUB_CACHE=/mnt/MLPerf/huggingface/hub``, and
a resolver that globs the default alone finds nothing there. That is not a soft
failure: a missing drafter is rejected at config time, so the server never
starts.
"""

import os

import pytest

# The gemma4 vLLM generator imports vllm at module scope (through
# tt_transformers.generator_vllm), so COLLECTING this file fails outright on a
# runner without vLLM installed -- which is the tt-metal unit-test job. Skip
# before the import rather than inside the tests: the failure is at import.
pytest.importorskip("vllm")
from models.demos.gemma4.tt import generator_vllm as gv


def _mk(root, repo):
    snap = os.path.join(root, repo, "snapshots", "abc123")
    os.makedirs(snap, exist_ok=True)
    return snap


def test_hf_hub_cache_is_searched_before_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "mlperf" / "hub"))
    roots = gv._hf_hub_cache_dirs()
    assert roots[0] == str(tmp_path / "mlperf" / "hub")
    assert roots[-1] == os.path.expanduser("~/.cache/huggingface/hub")


def test_hf_home_contributes_its_hub_subdirectory(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    assert str(tmp_path / "hfhome" / "hub") in gv._hf_hub_cache_dirs()


def test_roots_are_deduplicated(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    roots = gv._hf_hub_cache_dirs()
    assert len(roots) == len(set(roots))


def test_dflash_drafter_is_found_in_the_configured_cache(monkeypatch, tmp_path):
    root = str(tmp_path / "mlperf" / "hub")
    snap = _mk(root, "models--z-lab--gemma-4-31B-it-DFlash")
    monkeypatch.setenv("HF_HUB_CACHE", root)
    got = gv._dflash_default_snapshot()
    assert got is not None and got.rstrip("/") == snap


def test_mtp_assistant_is_found_in_the_configured_cache(monkeypatch, tmp_path):
    root = str(tmp_path / "mlperf" / "hub")
    snap = _mk(root, "models--google--gemma-4-12B-it-assistant")
    monkeypatch.setenv("HF_HUB_CACHE", root)
    got = gv._assistant_default_snapshot("google/gemma-4-12B-it")
    assert got.rstrip("/") == snap


def test_a_missing_drafter_still_reports_absence(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty2"))
    # No assertion on the default root: a developer box may legitimately have
    # the drafter cached there. Absence is only asserted for the configured
    # roots, which is what the resolver adds.
    assert gv._hf_snapshot_glob("models--z-lab--does-not-exist") is None
