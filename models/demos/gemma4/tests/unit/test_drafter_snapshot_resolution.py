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


def test_a_read_only_cache_does_not_attempt_a_fetch(monkeypatch, tmp_path):
    """A :ro weights mount is cache-only.

    HF writes metadata on every HEAD call, so a fetch into a read-only mount
    fails with "Read-only file system" rather than populating anything. The
    caller reports the miss instead.
    """
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(ro)])
        called = []
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            lambda *a, **k: called.append(k) or "/never",
        )
        assert gv._hf_resolve_repo("z-lab/nope", "models--z-lab--nope") is None
        assert called == []
    finally:
        ro.chmod(0o755)


def test_a_writable_cache_fetches_even_with_offline_set(monkeypatch, tmp_path):
    """Write access is the operator asking for the fetch.

    HF_HUB_OFFLINE is set for the read-only case; it must not veto a fetch into
    a cache the operator deliberately made writable, and it is restored after.
    """
    root = tmp_path / "rw"
    root.mkdir()
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(root)])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    seen = {}

    def fake_download(*a, **k):
        seen["offline_during_call"] = os.environ.get("HF_HUB_OFFLINE")
        seen.update(k)
        return str(root / "fetched")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    got = gv._hf_resolve_repo("z-lab/x", "models--z-lab--x")
    assert got == str(root / "fetched")
    assert seen["offline_during_call"] is None, "offline must be cleared for the call"
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "and restored afterwards"


def test_a_cache_hit_never_reaches_the_hub(monkeypatch, tmp_path):
    root = str(tmp_path / "hub")
    snap = _mk(root, "models--z-lab--gemma-4-31B-it-DFlash")
    monkeypatch.setenv("HF_HUB_CACHE", root)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    called = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *a, **k: called.append(k) or "/never",
    )
    assert gv._dflash_default_snapshot().rstrip("/") == snap
    assert called == []


def test_an_online_miss_fetches_the_repo(monkeypatch, tmp_path):
    """Cache-then-hub, the way transformers resolves the target model."""
    # Pin the search roots: ~/.cache is the final fallback and a dev box may
    # genuinely have the drafter cached there, which would mask the fetch.
    (tmp_path / "empty").mkdir(exist_ok=True)
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(tmp_path / "empty")])
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    asked = {}

    def fake_download(*a, **k):
        asked.update(k)
        return str(tmp_path / "fetched")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    got = gv._dflash_default_snapshot()
    assert got == str(tmp_path / "fetched")
    assert asked.get("repo_id") == "z-lab/gemma-4-31B-it-DFlash"


def test_a_failed_fetch_is_reported_as_a_miss_not_an_exception(monkeypatch, tmp_path):
    (tmp_path / "empty").mkdir(exist_ok=True)
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(tmp_path / "empty")])
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def boom(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    assert gv._dflash_default_snapshot() is None
