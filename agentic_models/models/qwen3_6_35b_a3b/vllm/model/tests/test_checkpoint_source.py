# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the checkpoint directory the model reads. No device is opened.

tt-inference-server passes ``--model`` as the repo id, places the pinned checkpoint outside ``$HF_HOME`` and exports
it as ``MODEL_WEIGHTS_DIR`` (and an ``HF_MODEL`` symlink); bare-metal runs keep the ``$HF_HOME`` hub snapshot.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_models.models.qwen3_6_35b_a3b.vllm.tt import model

TT = Path(__file__).resolve().parents[2] / "tt"
HUB_DIR = "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"


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
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    loaded = []

    def from_pretrained(path, **kwargs):
        loaded.append(Path(path))
        return SimpleNamespace(text_config=SimpleNamespace())

    monkeypatch.setattr(model, "AutoConfig", SimpleNamespace(from_pretrained=from_pretrained))
    return SimpleNamespace(set=monkeypatch.setenv, tmp=tmp_path, loaded=loaded)


def test_tti_weights_dir_is_preferred(env):
    weights = checkpoint(env.tmp / "cache_root/weights/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots" / model.HF_REVISION)
    checkpoint(env.tmp / "hf_home" / HUB_DIR / model.HF_REVISION)
    env.set("MODEL_WEIGHTS_DIR", str(weights))
    assert model.Checkpoint().path == weights
    assert env.loaded == [weights]


def test_hf_model_symlink_is_used(env):
    weights = checkpoint(env.tmp / "weights/snapshots" / model.HF_REVISION)
    link = env.tmp / "model_symlinks/Qwen3.6-35B-A3B"
    link.parent.mkdir()
    link.symlink_to(weights)
    env.set("HF_MODEL", str(link))
    assert model.Checkpoint().path == link


def test_hf_home_snapshot_without_a_mounted_checkpoint(env):
    hub = checkpoint(env.tmp / "hf_home" / HUB_DIR / model.HF_REVISION)
    env.set("MODEL_WEIGHTS_DIR", str(checkpoint(env.tmp / "metadata_only", safetensors=False)))
    env.set("HF_MODEL", "Qwen/Qwen3.6-35B-A3B")
    assert model.Checkpoint().path == hub


def test_explicit_path_wins(env):
    env.set("MODEL_WEIGHTS_DIR", str(checkpoint(env.tmp / "mounted")))
    explicit = checkpoint(env.tmp / "explicit")
    assert model.Checkpoint(explicit).path == explicit


def test_mounted_snapshot_must_be_the_pinned_revision(env):
    env.set("MODEL_WEIGHTS_DIR", str(checkpoint(env.tmp / "snapshots" / ("0" * 40))))
    with pytest.raises(ValueError, match=model.HF_REVISION):
        model.Checkpoint()


def test_tokenizer_loads_from_the_checkpoint_directory():
    assert "AutoTokenizer.from_pretrained(model.checkpoint.path, local_files_only=True)" in (
        TT / "generator.py"
    ).read_text()
