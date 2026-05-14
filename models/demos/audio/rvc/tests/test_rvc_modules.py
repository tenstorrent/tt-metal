# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for RVC TTNN implementation components.
"""

import pytest
import torch

import ttnn
from models.demos.audio.rvc.tt.reference_rvc import (
    RVCModel,
    PosteriorEncoder,
    HiFiGANVocoder,
    FlowDecoder,
    RMVPE,
    ResBlock,
    EncoderBlock,
)
from models.demos.audio.rvc.tt.rvc_parameter_preprocessing import preprocess_rvc_model


class TestReferenceModel:
    """Test the PyTorch reference model components."""

    def test_posterior_encoder(self):
        """Test posterior encoder forward pass."""
        encoder = PosteriorEncoder(80, 192, 192, kernel_size=5, n_layers=6)
        x = torch.randn(1, 80, 100)
        with torch.no_grad():
            z, m, logs = encoder(x)
        assert z.shape == (1, 192, 100)
        assert m.shape == (1, 192, 100)
        assert logs.shape == (1, 192, 100)
        assert not torch.isnan(z).any()

    def test_hifigan_vocoder(self):
        """Test HiFi-GAN vocoder forward pass."""
        vocoder = HiFiGANVocoder(in_channels=192)
        mel = torch.randn(1, 192, 50)
        with torch.no_grad():
            audio = vocoder(mel)
        assert audio.shape[0] == 1
        assert audio.shape[1] == 1
        assert audio.shape[2] > mel.shape[2]  # Upsampled
        assert audio.abs().max() <= 1.0  # tanh bounded

    def test_flow_decoder(self):
        """Test flow decoder forward and reverse pass."""
        flow = FlowDecoder(192, 192, n_flows=4)
        z = torch.randn(1, 192, 50)

        # Forward
        with torch.no_grad():
            out_forward = flow(z, reverse=False)
        assert out_forward.shape == z.shape

        # Reverse
        with torch.no_grad():
            out_reverse = flow(z, reverse=True)
        assert out_reverse.shape == z.shape

    def test_rmvpe(self):
        """Test RMVPE pitch extraction."""
        rmvpe = RMVPE(in_channels=128, hidden_channels=256)
        mel = torch.randn(1, 128, 100)
        with torch.no_grad():
            f0 = rmvpe(mel)
        assert f0.shape[0] == 1
        assert f0.shape[1] == 100
        assert (f0 >= 0).all()

    def test_resblock(self):
        """Test residual block."""
        resblock = ResBlock(256, [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        x = torch.randn(1, 256, 100)
        with torch.no_grad():
            out = resblock(x)
        assert out.shape == x.shape

    def test_encoder_block(self):
        """Test single encoder block."""
        block = EncoderBlock(192, 768, 2, 3)
        x = torch.randn(1, 192, 100)
        with torch.no_grad():
            out = block(x)
        assert out.shape == x.shape

    def test_full_model(self):
        """Test full RVC model forward pass."""
        model = RVCModel()
        mel = torch.randn(1, 80, 50)
        features = torch.randn(200, 192)
        features = torch.nn.functional.normalize(features, dim=-1)

        with torch.no_grad():
            output = model(mel, features, index_rate=0.5)

        assert output.shape[0] == 1
        assert output.shape[1] == 1
        assert output.shape[2] > mel.shape[2]
        assert not torch.isnan(output).any()

    def test_batch_processing(self):
        """Test batch inference."""
        model = RVCModel()
        mel = torch.randn(2, 80, 50)

        with torch.no_grad():
            output = model(mel, index_rate=0.0)

        assert output.shape[0] == 2

    def test_different_lengths(self):
        """Test with different sequence lengths."""
        model = RVCModel()
        for length in [32, 64, 100, 128]:
            mel = torch.randn(1, 80, length)
            with torch.no_grad():
                output = model(mel, index_rate=0.0)
            assert output.shape[0] == 1
            assert not torch.isnan(output).any()


class TestParameterPreprocessing:
    """Test parameter preprocessing pipeline."""

    @pytest.mark.parametrize("device_params", [{"l1_small_size": 1600}], indirect=True)
    def test_preprocess(self, device):
        """Test full parameter preprocessing."""
        model = RVCModel()
        params = preprocess_rvc_model(model, device)

        assert params.device == device
        assert params.encoder is not None
        assert params.flow is not None
        assert params.vocoder is not None
        assert params.rmvpe is not None

    @pytest.mark.parametrize("device_params", [{"l1_small_size": 1600}], indirect=True)
    def test_encoder_params(self, device):
        """Test encoder parameter extraction."""
        model = RVCModel()
        params = preprocess_rvc_model(model, device)

        enc = params.encoder
        assert enc.proj_m_weight is not None
        assert enc.proj_logs_weight is not None
        assert len(enc.enc_layers) == 6  # Default n_layers


class TestTTNNComponents:
    """Test TTNN-specific component implementations."""

    @pytest.mark.parametrize("device_params", [{"l1_small_size": 1600}], indirect=True)
    def test_feature_retrieval(self, device):
        """Test index-based feature retrieval."""
        from models.demos.audio.rvc.tt.ttnn_rvc import feature_retrieval

        source = torch.randn(1, 192, 50)
        index = torch.randn(200, 192)
        index = torch.nn.functional.normalize(index, dim=-1)

        # CPU test
        result = feature_retrieval(source, index, index_rate=0.5)
        assert result.shape == source.shape
        assert not torch.isnan(result).any()

        # Zero index_rate should return source unchanged
        result_no_retrieval = feature_retrieval(source, index, index_rate=0.0)
        assert torch.allclose(result_no_retrieval, source, atol=1e-6)

    @pytest.mark.parametrize("device_params", [{"l1_small_size": 1600}], indirect=True)
    def test_mel_conversion(self, device):
        """Test audio to mel conversion."""
        from models.demos.audio.rvc.tt.ttnn_rvc import _audio_to_mel

        audio = torch.randn(1, 16000)  # 1 second
        mel = _audio_to_mel(audio, None)
        assert mel.shape[0] == 1
        assert mel.shape[1] == 80  # n_mels
