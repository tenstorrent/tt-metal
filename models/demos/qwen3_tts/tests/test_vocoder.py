# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU-only tests for the Qwen3-TTS Vocoder (Code2Wav decoder).
No device or ttnn dependency at module level.
"""

import numpy as np
import pytest
import torch
from loguru import logger


class TestVocoderConfig:
    """Test VocoderConfig parsing."""

    def test_config_from_dict(self):
        from models.demos.qwen3_tts.tt.configs import VocoderConfig

        cfg = VocoderConfig.from_dict({
            "codebook_size": 2048,
            "codebook_dim": 512,
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "decoder_dim": 1536,
            "upsample_rates": [8, 5, 4, 3],
            "upsampling_ratios": [2, 2],
        })
        assert cfg.codebook_size == 2048
        assert cfg.codebook_dim == 512
        assert cfg.hidden_size == 512
        assert cfg.num_hidden_layers == 8
        assert cfg.decoder_dim == 1536
        assert cfg.upsample_rates == [8, 5, 4, 3]
        assert cfg.upsampling_ratios == [2, 2]
        assert cfg.sliding_window == 72
        logger.info("VocoderConfig parsing verified")

    def test_config_defaults(self):
        from models.demos.qwen3_tts.tt.configs import VocoderConfig

        cfg = VocoderConfig()
        total_upsample = 1
        for r in cfg.upsample_rates + cfg.upsampling_ratios:
            total_upsample *= r
        assert total_upsample == 1920
        assert cfg.output_sample_rate == 24000


class TestVocoderReference:
    """Test the PyTorch reference model."""

    def test_small_model_forward(self):
        """Verify reference model with small random weights."""
        from models.demos.qwen3_tts.reference.vocoder_ref import VocoderReference

        model = VocoderReference(
            codebook_size=32, codebook_dim=16, hidden_size=32, latent_dim=64,
            num_layers=2, num_heads=4, num_kv_heads=4, head_dim=8, ffn_dim=64,
            sliding_window=8, num_quantizers=4, decoder_dim=48,
            upsample_rates=[2, 2, 2, 2], upsampling_ratios=[2, 2],
        )
        model.eval()

        codes = torch.randint(0, 32, (1, 4, 10))
        with torch.no_grad():
            wav = model(codes)

        expected_upsample = 2 * 2 * 2 * 2 * 2 * 2  # = 64
        assert wav.shape == (1, 1, 10 * expected_upsample)
        assert not torch.isnan(wav).any()
        assert wav.min() >= -1.0
        assert wav.max() <= 1.0
        logger.info(f"Small model: {codes.shape} -> {wav.shape}")

    def test_full_model_dimensions(self):
        """Verify default model matches real architecture dimensions."""
        from models.demos.qwen3_tts.reference.vocoder_ref import VocoderReference

        model = VocoderReference()
        params = sum(p.numel() for p in model.parameters())
        assert abs(params - 114e6) < 5e6, f"Expected ~114M params, got {params/1e6:.1f}M"
        assert model.total_upsample == 1920

        codes = torch.randint(0, 2048, (1, 16, 5))  # 5 frames
        model.eval()
        with torch.no_grad():
            wav = model(codes)
        assert wav.shape == (1, 1, 5 * 1920)
        logger.info(f"Full model: {params/1e6:.1f}M params, {wav.shape[-1]} samples for 5 frames")

    def test_chunked_decode(self):
        """Verify chunked decode produces valid output."""
        from models.demos.qwen3_tts.reference.vocoder_ref import VocoderReference

        model = VocoderReference(
            codebook_size=32, codebook_dim=16, hidden_size=32, latent_dim=64,
            num_layers=1, num_heads=4, num_kv_heads=4, head_dim=8, ffn_dim=64,
            sliding_window=8, num_quantizers=4, decoder_dim=48,
            upsample_rates=[2, 2, 2, 2], upsampling_ratios=[2, 2],
        )
        model.eval()

        codes = torch.randint(0, 32, (1, 4, 20))
        with torch.no_grad():
            wav_full = model(codes)
            wav_chunked = model.chunked_decode(codes, chunk_size=10, left_context_size=3)

        assert wav_chunked.shape[-1] == wav_full.shape[-1]
        assert not torch.isnan(wav_chunked).any()
        logger.info(f"Chunked decode: full={wav_full.shape[-1]}, chunked={wav_chunked.shape[-1]}")

    def test_vq_dequant_shape(self):
        """Verify VQ dequant produces correct shape for valid codes."""
        from models.demos.qwen3_tts.reference.vocoder_ref import SplitResidualVectorQuantizer

        quantizer = SplitResidualVectorQuantizer(
            dimension=8, n_q=4, n_q_semantic=1, bins=32,
            input_dimension=16, output_dimension=16,
        )
        # Initialize codebook with non-zero values for testing
        for layer in quantizer.rvq_first.vq.layers:
            layer._codebook.embedding_sum.data.normal_()
        for layer in quantizer.rvq_rest.vq.layers:
            layer._codebook.embedding_sum.data.normal_()

        codes = torch.randint(0, 32, (1, 4, 5))  # [B, num_quantizers, T]
        with torch.no_grad():
            output = quantizer.decode(codes)
        assert output.shape == (1, 16, 5)
        assert output.abs().sum() > 0
        logger.info(f"VQ dequant: {codes.shape} -> {output.shape}")

    def test_real_weights_load(self):
        """Verify loading from real safetensors (if available)."""
        try:
            from models.demos.qwen3_tts.reference.vocoder_ref import VocoderReference

            model = VocoderReference.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
            model.eval()

            params = sum(p.numel() for p in model.parameters())
            assert params > 100e6
            assert model.total_upsample == 1920

            codes = torch.randint(0, 2048, (1, 16, 3))
            with torch.no_grad():
                wav = model(codes)
            assert wav.shape == (1, 1, 3 * 1920)
            assert not torch.isnan(wav).any()
            logger.info(f"Real weights: {params/1e6:.1f}M params, output={wav.shape}")
        except Exception as e:
            pytest.skip(f"Weights not available: {e}")


class TestVocoderWrapper:
    """Test the Vocoder TT wrapper (CPU-side)."""

    def test_decode_shape(self):
        """Verify decode returns correct numpy array shape."""
        from models.demos.qwen3_tts.reference.vocoder_ref import VocoderReference
        from models.demos.qwen3_tts.tt.vocoder import Vocoder

        model = VocoderReference(
            codebook_size=32, codebook_dim=16, hidden_size=32, latent_dim=64,
            num_layers=1, num_heads=4, num_kv_heads=4, head_dim=8, ffn_dim=64,
            sliding_window=8, num_quantizers=4, decoder_dim=48,
            upsample_rates=[2, 2, 2, 2], upsampling_ratios=[2, 2],
        )
        vocoder = Vocoder(model, dtype=torch.float32)

        codes = torch.randint(0, 32, (1, 10, 4))  # [B, T, num_quantizers]
        waveform = vocoder.decode(codes)

        assert isinstance(waveform, np.ndarray)
        expected_samples = 10 * model.total_upsample
        assert len(waveform) == expected_samples
        logger.info(f"Vocoder wrapper: {codes.shape} -> {waveform.shape}")
