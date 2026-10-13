# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit test of the model-path and weight-cache resolution in ``tt/paths.py`` (no device, no snapshot).

    pytest models/demos/pplx_decider_v1_27b/tests/test_paths.py -q
"""

from __future__ import annotations

import os

import pytest

from models.demos.pplx_decider_v1_27b.tt import paths

ENV = ("HF_MODEL", "PPLX_DECIDER_SNAPSHOT", "TT_CACHE_PATH", "PPLX_DECIDER_WEIGHT_CACHE", "HF_HOME")


def make_checkpoint(directory, *, config='{"model_type": "qwen3_5"}', index='{"weight_map": {}}', readout=b"r"):
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(config)
    (directory / "model.safetensors.index.json").write_text(index)
    (directory / "readout.safetensors").write_bytes(readout)
    return directory


@pytest.fixture
def env(tmp_path, monkeypatch):
    """No resolution variable set; the HF cache lookup is replaced by a fake HF-cache snapshot."""
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    hub = tmp_path / "hf" / "hub" / "models--perplexity-ai--pplx-decider-v1-27b" / "snapshots" / paths.REVISION
    hf_snapshot = make_checkpoint(hub)
    calls = []

    def fake_hf_snapshot(repo_id, revision):
        calls.append((repo_id, revision))
        if repo_id != paths.MODEL_ID:
            raise FileNotFoundError(f"hf download {repo_id}")
        return hf_snapshot

    monkeypatch.setattr(paths, "_hf_snapshot", fake_hf_snapshot)
    return dict(tmp=tmp_path, hf_snapshot=hf_snapshot, calls=calls, mp=monkeypatch)


def test_model_path_order(env):
    tmp, mp = env["tmp"], env["mp"]
    arg, hf_model, legacy = (make_checkpoint(tmp / n) for n in ("arg", "hf_model", "legacy"))
    mp.setenv("HF_MODEL", str(hf_model))
    mp.setenv("PPLX_DECIDER_SNAPSHOT", str(legacy))

    assert paths.resolve_model_path(arg) == (arg, "argument")
    assert paths.resolve_model_path() == (hf_model, "$HF_MODEL")
    mp.delenv("HF_MODEL")
    assert paths.resolve_model_path()[0] == legacy
    mp.delenv("PPLX_DECIDER_SNAPSHOT")
    assert paths.resolve_model_path()[0] == env["hf_snapshot"]
    assert env["calls"][-1] == (paths.MODEL_ID, paths.REVISION)  # the pinned revision
    assert paths.snapshot_path() == env["hf_snapshot"]


def test_model_path_repo_id(env, expect_error):
    env["mp"].setenv("HF_MODEL", paths.MODEL_ID)
    assert paths.resolve_model_path() == (env["hf_snapshot"], "$HF_MODEL")
    assert env["calls"][-1] == (paths.MODEL_ID, paths.REVISION)
    with expect_error(FileNotFoundError, "hf download other/model"):
        paths.resolve_model_path("other/model")
    with expect_error(FileNotFoundError, "neither a local directory"):
        paths.resolve_model_path(env["tmp"] / "missing")


def test_cache_dir_order(env):
    tmp, mp = env["tmp"], env["mp"]
    local = make_checkpoint(tmp / "local")
    key = paths.cache_key(local)
    mp.setenv("TT_CACHE_PATH", str(tmp / "tt_cache_path"))
    mp.setenv("PPLX_DECIDER_WEIGHT_CACHE", str(tmp / "legacy_cache"))
    mp.setenv("HF_HOME", str(tmp / "hf"))

    resolve = paths.resolve_model_cache_path
    assert resolve(local) == (tmp / "tt_cache_path" / "tt_weight_cache" / key, "$TT_CACHE_PATH")
    mp.delenv("TT_CACHE_PATH")
    assert resolve(local) == (tmp / "legacy_cache" / "tt_weight_cache" / key, "$PPLX_DECIDER_WEIGHT_CACHE")
    mp.delenv("PPLX_DECIDER_WEIGHT_CACHE")
    assert resolve(local) == (local / "tt_weight_cache" / key, "model directory")
    # An HF repo id is not a local directory: the cache goes under $HF_HOME/tt_cache (gemma4 naming).
    expected = tmp / "hf" / "tt_cache" / "perplexity-ai--pplx-decider-v1-27b" / "tt_weight_cache" / paths.REVISION
    assert resolve(paths.MODEL_ID) == (expected, "$HF_HOME/tt_cache")
    assert expected.is_dir()  # created (create=True is the default)
    assert resolve(local, create=False)[0] == local / "tt_weight_cache" / key


def test_cache_key(env):
    tmp = env["tmp"]
    # An HF-cache snapshot directory is keyed by its revision; the cache sits inside the snapshot dir.
    snap = env["hf_snapshot"]
    assert paths.cache_key(snap) == paths.REVISION
    assert paths.resolve_model_cache_path(snap, create=False)[0] == snap / "tt_weight_cache" / paths.REVISION
    # Two different local checkpoints never share a key.
    a = make_checkpoint(tmp / "a")
    b = make_checkpoint(tmp / "b", config='{"model_type": "qwen3_5", "num_hidden_layers": 2}')
    c = make_checkpoint(tmp / "c", readout=b"a different readout")
    keys = {paths.cache_key(p) for p in (a, b, c)}
    assert len(keys) == 3 and all(k.startswith("local-") for k in keys)
    # Same checkpoint, same key; a rewritten readout (new mtime) is a new key.
    assert paths.cache_key(a) == paths.cache_key(tmp / "a")
    st = (a / "readout.safetensors").stat()
    os.utime(a / "readout.safetensors", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert paths.cache_key(a) not in keys
