# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Qwen3-TTS reference implementation.
Verifies that the HuggingFace model loads and produces valid output.
"""

import os

import pytest
import torch


@pytest.fixture
def model_path():
    path = os.getenv("QWEN3_TTS_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    return path


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class TestReferenceConfig:
    def test_config_from_pretrained(self, model_path):
        from models.demos.qwen3_tts.reference.functional import Qwen3TTSConfig

        config = Qwen3TTSConfig.from_pretrained(model_path)
        assert config.talker_hidden_size == 2048
        assert config.talker_num_layers == 28
        assert config.talker_num_heads == 16
        assert config.talker_num_kv_heads == 8
        assert config.talker_intermediate_size == 6144
        assert config.talker_vocab_size == 3072
        assert config.talker_text_vocab_size == 151936
        assert config.talker_num_code_groups == 16
        assert config.cp_hidden_size == 1024
        assert config.cp_num_layers == 5
        assert config.codec_language_ids.get("japanese") == 2058

    def test_default_config(self):
        from models.demos.qwen3_tts.reference.functional import Qwen3TTSConfig

        config = Qwen3TTSConfig()
        assert config.talker_hidden_size == 2048
        assert config.spk_enc_dim == 2048
        assert config.spk_sample_rate == 24000


class TestActivationCapture:
    def test_capture_and_save(self, tmp_path):
        from models.demos.qwen3_tts.reference.functional import ActivationCapture

        capture = ActivationCapture()
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
        )
        capture.register(model[0], "linear1")
        capture.register(model[2], "linear2")

        x = torch.randn(2, 32)
        _ = model(x)

        assert "linear1" in capture.activations
        assert "linear2" in capture.activations
        assert capture.activations["linear1"].shape == (2, 64)
        assert capture.activations["linear2"].shape == (2, 16)

        capture.save(str(tmp_path / "activations"))
        loaded = ActivationCapture.load(str(tmp_path / "activations"))
        assert "linear1" in loaded
        assert torch.allclose(loaded["linear1"], capture.activations["linear1"])

        capture.remove_hooks()
        capture.clear()
        assert len(capture.activations) == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for reference model inference",
)
class TestReferenceInference:
    def test_generate_basic(self, model_path, device):
        from models.demos.qwen3_tts.reference.functional import (
            generate_reference,
            load_reference_model,
        )

        model = load_reference_model(model_path, device=device)
        wavs, sr = generate_reference(
            model,
            text="こんにちは",
            language="japanese",
            max_new_tokens=256,
        )
        assert sr > 0
        assert len(wavs) > 0
        wav = wavs[0]
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        assert len(wav) > 0, "Generated waveform should not be empty"

    def test_generate_with_capture(self, model_path, device):
        from models.demos.qwen3_tts.reference.functional import (
            ActivationCapture,
            generate_reference,
            load_reference_model,
        )

        model = load_reference_model(model_path, device=device)
        capture = ActivationCapture()
        wavs, sr = generate_reference(
            model,
            text="テスト",
            language="japanese",
            max_new_tokens=128,
            capture=capture,
        )
        assert len(capture.activations) > 0, "Should have captured activations"
        for name, tensor in capture.activations.items():
            assert tensor.ndim >= 2, f"Activation {name} should be at least 2D"
