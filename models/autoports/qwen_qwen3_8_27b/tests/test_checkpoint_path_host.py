# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.autoports.qwen_qwen3_8_27b.tt import model


def test_checkpoint_path_prefers_inference_server_staged_weights(tmp_path, monkeypatch):
    staged = tmp_path / "staged"
    staged.mkdir()
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(staged))
    monkeypatch.delenv("CACHE_ROOT", raising=False)

    index = staged / "model.safetensors.index.json"
    index.write_text("{}")

    assert model.checkpoint_path() == staged


def test_checkpoint_path_uses_cache_root_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    staged = tmp_path / "weights" / "Qwen3.8-27B"
    staged.mkdir(parents=True)
    (staged / "model.safetensors.index.json").write_text("{}")

    assert model.checkpoint_path() == staged
