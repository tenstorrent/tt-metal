# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the checkpoint directory the model reads. No device is opened.

tt-inference-server passes ``--model`` as the repo id, places the pinned checkpoint outside ``$HF_HOME`` and exports
it as ``MODEL_WEIGHTS_DIR`` (and an ``HF_MODEL`` symlink); bare-metal runs keep ``snapshot_download(local_files_only)``.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_models.models.qwen3_6_27b.vllm.tt import model

TT = Path(__file__).resolve().parents[2] / "tt"


def checkpoint(root, safetensors=True):
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    if safetensors:
        (root / "model-00001-of-00001.safetensors").write_bytes(b"")
    return root


@pytest.fixture
def env(monkeypatch, tmp_path):
    for name in ("MODEL_WEIGHTS_DIR", "HF_MODEL"):
        monkeypatch.delenv(name, raising=False)
    hub = checkpoint(tmp_path / "hf_home/hub/models--Qwen--Qwen3.6-27B/snapshots" / model.REVISION)
    downloads, loaded = [], []

    def snapshot_download(repo, revision, local_files_only):
        downloads.append((repo, revision, local_files_only))
        return str(hub)

    def from_pretrained(path, **kwargs):
        loaded.append(Path(path))
        return SimpleNamespace(text_config=SimpleNamespace())

    monkeypatch.setattr(model, "snapshot_download", snapshot_download)
    monkeypatch.setattr(model, "AutoConfig", SimpleNamespace(from_pretrained=from_pretrained))
    return SimpleNamespace(set=monkeypatch.setenv, tmp=tmp_path, hub=hub, downloads=downloads, loaded=loaded)


def test_tti_weights_dir_is_preferred(env):
    weights = checkpoint(env.tmp / "cache_root/weights/hub/models--Qwen--Qwen3.6-27B/snapshots" / model.REVISION)
    env.set("MODEL_WEIGHTS_DIR", str(weights))
    assert model.Checkpoint().path == weights
    assert env.loaded == [weights] and env.downloads == []


def test_hf_model_symlink_is_used(env):
    weights = checkpoint(env.tmp / "weights/snapshots" / model.REVISION)
    link = env.tmp / "model_symlinks/Qwen3.6-27B"
    link.parent.mkdir()
    link.symlink_to(weights)
    env.set("HF_MODEL", str(link))
    assert model.Checkpoint().path == link


def test_pinned_hub_snapshot_without_a_mounted_checkpoint(env):
    env.set("MODEL_WEIGHTS_DIR", str(checkpoint(env.tmp / "metadata_only", safetensors=False)))
    env.set("HF_MODEL", model.MODEL_ID)
    assert model.Checkpoint().path == env.hub
    assert env.downloads == [(model.MODEL_ID, model.REVISION, True)]


def test_explicit_path_wins(env):
    env.set("MODEL_WEIGHTS_DIR", str(checkpoint(env.tmp / "mounted")))
    explicit = checkpoint(env.tmp / "explicit")
    assert model.Checkpoint(explicit).path == explicit


def test_mounted_snapshot_must_be_the_pinned_revision(env):
    env.set("MODEL_WEIGHTS_DIR", str(checkpoint(env.tmp / "snapshots" / ("0" * 40))))
    with pytest.raises(ValueError, match=model.REVISION):
        model.Checkpoint()


def test_tokenizer_loads_from_the_checkpoint_directory():
    # model.checkpoint is Checkpoint(...).path (tt/model.py), so tokenizer, config and weights share one directory.
    assert "self.checkpoint = source.path" in (TT / "model.py").read_text()
    assert "AutoTokenizer.from_pretrained(model.checkpoint, local_files_only=True)" in (TT / "generator.py").read_text()
