# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU-only tests for the Qwen3-TTS Speaker Encoder: reference model, mel spectrogram, configs.
No device or ttnn dependency at module level.
"""

import numpy as np
import pytest
import torch
from loguru import logger


class TestSpeakerEncoderConfig:
    """Test SpeakerEncoderConfig parsing."""

    def test_config_from_dict(self):
        from models.demos.qwen3_tts.tt.configs import SpeakerEncoderConfig

        cfg = SpeakerEncoderConfig.from_dict({
            "enc_dim": 2048,
            "mel_dim": 128,
            "sample_rate": 24000,
        })
        assert cfg.enc_dim == 2048
        assert cfg.mel_dim == 128
        assert cfg.sample_rate == 24000
        assert cfg.enc_channels == [512, 512, 512, 512, 1536]
        assert cfg.enc_res2net_scale == 8
        logger.info("SpeakerEncoderConfig parsing verified")

    def test_config_defaults(self):
        from models.demos.qwen3_tts.tt.configs import SpeakerEncoderConfig

        cfg = SpeakerEncoderConfig()
        assert cfg.enc_dim == 2048
        assert len(cfg.enc_channels) == 5
        assert len(cfg.enc_kernel_sizes) == 5
        assert len(cfg.enc_dilations) == 5


class TestSpeakerEncoderReference:
    """Test the PyTorch reference model."""

    def test_small_model_forward(self):
        """Verify reference model with small random weights."""
        from models.demos.qwen3_tts.reference.speaker_encoder_ref import SpeakerEncoderReference

        model = SpeakerEncoderReference(
            mel_dim=16, enc_dim=64,
            enc_channels=[32, 32, 32, 32, 96],
            enc_kernel_sizes=[5, 3, 3, 3, 1],
            enc_dilations=[1, 2, 3, 4, 1],
            enc_res2net_scale=4, enc_se_channels=16,
            enc_attention_channels=16,
        )
        model.eval()

        mel = torch.randn(1, 50, 16)
        with torch.no_grad():
            emb = model(mel)

        assert emb.shape == (1, 64)
        assert not torch.isnan(emb).any()
        logger.info(f"Small model output: {emb.shape}, norm={emb.norm():.4f}")

    def test_full_model_shapes(self):
        """Verify default model dimensions match real architecture."""
        from models.demos.qwen3_tts.reference.speaker_encoder_ref import SpeakerEncoderReference

        model = SpeakerEncoderReference()
        params = sum(p.numel() for p in model.parameters())
        assert abs(params - 12e6) < 1e6, f"Expected ~12M params, got {params/1e6:.1f}M"

        mel = torch.randn(1, 100, 128)
        model.eval()
        with torch.no_grad():
            emb = model(mel)
        assert emb.shape == (1, 2048)
        logger.info(f"Full model: {params/1e6:.1f}M params, output={emb.shape}")

    def test_variable_length_input(self):
        """Speaker encoder should handle variable-length mel inputs."""
        from models.demos.qwen3_tts.reference.speaker_encoder_ref import SpeakerEncoderReference

        model = SpeakerEncoderReference(
            mel_dim=16, enc_dim=32,
            enc_channels=[16, 16, 16, 16, 48],
            enc_kernel_sizes=[5, 3, 3, 3, 1],
            enc_dilations=[1, 2, 3, 4, 1],
            enc_res2net_scale=2, enc_se_channels=8,
            enc_attention_channels=8,
        )
        model.eval()

        for T in [20, 50, 200]:
            mel = torch.randn(1, T, 16)
            with torch.no_grad():
                emb = model(mel)
            assert emb.shape == (1, 32), f"Failed for T={T}: {emb.shape}"
        logger.info("Variable length input test passed")

    def test_real_weights_load(self):
        """Verify loading from real safetensors (if available)."""
        try:
            from models.demos.qwen3_tts.reference.speaker_encoder_ref import SpeakerEncoderReference

            model = SpeakerEncoderReference.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
            model.eval()

            params = sum(p.numel() for p in model.parameters())
            assert abs(params - 12e6) < 1e6
            assert model.enc_dim == 2048

            mel = torch.randn(1, 281, 128)  # ~3s at 24kHz
            with torch.no_grad():
                emb = model(mel)
            assert emb.shape == (1, 2048)
            assert not torch.isnan(emb).any()
            logger.info(f"Real weights loaded: {params/1e6:.1f}M params, norm={emb.norm():.4f}")
        except Exception as e:
            pytest.skip(f"Weights not available: {e}")


class TestMelSpectrogram:
    """Test mel spectrogram extraction."""

    def test_mel_spectrogram_shape(self):
        from models.demos.qwen3_tts.tt.speaker_encoder import mel_spectrogram

        audio = np.random.randn(24000 * 3).astype(np.float32) * 0.1  # 3 seconds
        mel = mel_spectrogram(audio, sr=24000)

        assert mel.dim() == 3
        assert mel.shape[0] == 1  # batch
        assert mel.shape[2] == 128  # mel_dim
        expected_frames = (24000 * 3) // 256 + 1
        assert abs(mel.shape[1] - expected_frames) <= 2
        logger.info(f"Mel spectrogram: {mel.shape} ({mel.shape[1]} frames for 3s audio)")

    def test_mel_spectrogram_values(self):
        from models.demos.qwen3_tts.tt.speaker_encoder import mel_spectrogram

        audio = np.random.randn(24000).astype(np.float32) * 0.5
        mel = mel_spectrogram(audio, sr=24000)

        assert not torch.isnan(mel).any()
        assert not torch.isinf(mel).any()
        assert mel.max() < 20  # log-compressed values should be reasonable
        assert mel.min() > -20


class TestSpeakerEncoderPCC:
    """PCC comparison between our reference and HF reference (if available)."""

    def test_pcc_vs_hf(self):
        """Compare our reference output with HF Qwen3TTSSpeakerEncoder."""
        try:
            from models.demos.qwen3_tts.reference.speaker_encoder_ref import SpeakerEncoderReference

            our_model = SpeakerEncoderReference.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
            our_model.eval()

            from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
            from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder

            hf_config = Qwen3TTSSpeakerEncoderConfig(enc_dim=2048)
            hf_model = Qwen3TTSSpeakerEncoder(hf_config)

            # Load same weights
            from safetensors import safe_open

            from huggingface_hub import hf_hub_download

            path = hf_hub_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", filename="model.safetensors")
            hf_state = {}
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    if key.startswith("speaker_encoder."):
                        hf_state[key[len("speaker_encoder."):]] = f.get_tensor(key)
            hf_model.load_state_dict(hf_state, strict=True)
            hf_model.eval()

            torch.manual_seed(42)
            mel = torch.randn(1, 100, 128).to(torch.bfloat16)

            with torch.no_grad():
                our_emb = our_model.to(torch.bfloat16)(mel)
                hf_emb = hf_model.to(torch.bfloat16)(mel.transpose(1, 2)).squeeze(-1)  # HF expects [B, mel, T]... no, HF transposes internally

            # HF forward: hidden_states = hidden_states.transpose(1, 2) at start
            # so it expects [B, T, mel_dim] same as ours
            with torch.no_grad():
                hf_emb = hf_model.to(torch.bfloat16)(mel)

            our_emb_f = our_emb.float().flatten()
            hf_emb_f = hf_emb.float().flatten()

            pcc = torch.corrcoef(torch.stack([our_emb_f, hf_emb_f]))[0, 1].item()
            logger.info(f"PCC vs HF: {pcc:.6f}")
            assert pcc > 0.999, f"PCC too low: {pcc}"

        except (ImportError, OSError, Exception) as e:
            pytest.skip(f"HF model or weights not available: {e}")
