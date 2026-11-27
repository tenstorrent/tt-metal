# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the checkpoint directory the adapter reads. No device is opened.

tt-inference-server passes ``--model`` as the repo id, places the pinned checkpoint outside ``$HF_HOME`` and exports
it as ``MODEL_WEIGHTS_DIR`` (and an ``HF_MODEL`` symlink); bare-metal runs keep ``snapshot_download(local_files_only)``.
"""

import json

import huggingface_hub
import pytest

from agentic_models.models.nanbeige_4_3_33b.vllm.tt import generator_vllm as adapter

HUB_DIR = "hub/models--Nanbeige--Nanbeige4.3-33B/snapshots"


def checkpoint(root, safetensors=True):
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    if safetensors:
        (root / "model-00001-of-00014.safetensors").write_bytes(b"")
    return root


@pytest.fixture
def env(monkeypatch, tmp_path):
    for name in ("MODEL_WEIGHTS_DIR", "HF_MODEL"):
        monkeypatch.delenv(name, raising=False)
    hub = checkpoint(tmp_path / "hf_home" / HUB_DIR / adapter.HF_REVISION)
    downloads = []

    def snapshot_download(repo, revision, local_files_only):
        downloads.append((repo, revision, local_files_only))
        return str(hub)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    env = type("Env", (), {})()
    env.set, env.tmp, env.hub, env.downloads = monkeypatch.setenv, tmp_path, hub, downloads
    return env


def test_tti_weights_dir_is_preferred(env):
    weights = checkpoint(env.tmp / "cache_root/weights" / HUB_DIR / adapter.HF_REVISION)
    env.set("MODEL_WEIGHTS_DIR", str(weights))
    # vLLM's hf_config._name_or_path is the repo id here, not a directory.
    assert adapter.checkpoint_path(adapter.HF_MODEL) == weights
    assert env.downloads == []


def test_hf_model_symlink_is_used(env):
    weights = checkpoint(env.tmp / "weights/snapshots" / adapter.HF_REVISION)
    link = env.tmp / "model_symlinks/Nanbeige4.3-33B"
    link.parent.mkdir()
    link.symlink_to(weights)
    env.set("HF_MODEL", str(link))
    assert adapter.checkpoint_path() == link


def test_pinned_hub_snapshot_without_a_mounted_checkpoint(env):
    env.set("MODEL_WEIGHTS_DIR", str(checkpoint(env.tmp / "metadata_only", safetensors=False)))
    env.set("HF_MODEL", adapter.HF_MODEL)
    assert adapter.checkpoint_path(adapter.HF_MODEL) == env.hub
    assert env.downloads == [(adapter.HF_MODEL, adapter.HF_REVISION, True)]


def test_local_model_directory_wins(env):
    # vLLM's --model can be a local directory; hf_config._name_or_path is then that directory.
    env.set("MODEL_WEIGHTS_DIR", str(checkpoint(env.tmp / "mounted")))
    explicit = checkpoint(env.tmp / "explicit")
    assert adapter.checkpoint_path(explicit) == explicit


def test_mounted_snapshot_must_be_the_pinned_revision(env):
    env.set("MODEL_WEIGHTS_DIR", str(checkpoint(env.tmp / "snapshots" / ("0" * 40))))
    with pytest.raises(ValueError, match=adapter.HF_REVISION):
        adapter.checkpoint_path()
