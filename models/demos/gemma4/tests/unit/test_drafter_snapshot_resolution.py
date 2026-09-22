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

    Asserting the ENV VAR alone is not enough, and that is what let the real bug
    ship: huggingface_hub reads HF_HUB_OFFLINE once at import into
    constants.HF_HUB_OFFLINE, and every offline gate goes through
    constants.is_offline_mode(), which returns that module global. Clearing only
    the env var left the hub offline, so a :rw CI run still raised
    LocalEntryNotFoundError and the drafter never downloaded. Assert the gate the
    hub actually consults.
    """
    from huggingface_hub import constants as hf_constants

    root = tmp_path / "rw"
    root.mkdir()
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(root)])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    seen = {}

    def fake_download(*a, **k):
        seen["env_during_call"] = os.environ.get("HF_HUB_OFFLINE")
        seen["offline_mode_during_call"] = hf_constants.is_offline_mode()
        seen.update(k)
        return str(root / "fetched")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    got = gv._hf_resolve_repo("z-lab/x", "models--z-lab--x")
    assert got == str(root / "fetched")
    assert seen["env_during_call"] is None, "the env var is cleared for the call"
    assert seen["offline_mode_during_call"] is False, (
        "the hub's own offline gate must be off for the call, or the fetch raises "
        "LocalEntryNotFoundError however the env var is set"
    )
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "env restored afterwards"
    assert hf_constants.HF_HUB_OFFLINE is True, "and the hub global restored too"


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


# ── Target weights directory for the drafter's embedding loader ──────────────


def test_model_weights_dir_is_used_when_it_exists(monkeypatch, tmp_path):
    """tt-inference-server exports MODEL_WEIGHTS_DIR; prefer it."""
    d = tmp_path / "weights"
    d.mkdir()
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(d))
    monkeypatch.delenv("HF_MODEL", raising=False)
    assert gv._target_weights_dir() == str(d)


def test_a_repo_id_in_hf_model_resolves_through_the_hub_cache(monkeypatch, tmp_path):
    """tt-metal's vLLM CI sets HF_MODEL to a REPO ID and never sets
    MODEL_WEIGHTS_DIR, so the only way to the checkpoint is the hub cache.
    Relying on the env var alone built 'None/model.safetensors.index.json' and
    killed the engine during warmup after the drafter had downloaded fine."""
    root = tmp_path / "hub"
    snap = root / "models--google--gemma-4-31B-it" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(root)])
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("GEMMA4_MODEL_PATH", raising=False)
    monkeypatch.setenv("HF_MODEL", "google/gemma-4-31B-it")
    got = gv._target_weights_dir()
    assert got is not None, "a repo id must resolve through the hub cache"
    assert got.rstrip("/") == str(snap)


def test_a_local_hf_model_directory_is_used_directly(monkeypatch, tmp_path):
    d = tmp_path / "local-ckpt"
    d.mkdir()
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("HF_MODEL", str(d))
    assert gv._target_weights_dir() == str(d)


def test_nothing_resolvable_reports_absence(monkeypatch, tmp_path):
    """None must be returned so the caller can raise something readable,
    rather than formatting it into a path."""
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(tmp_path / "empty")])
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("GEMMA4_MODEL_PATH", raising=False)
    monkeypatch.setenv("HF_MODEL", "google/gemma-4-31B-it")
    assert gv._target_weights_dir() is None
