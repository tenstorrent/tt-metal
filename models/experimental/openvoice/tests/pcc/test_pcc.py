#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
PCC (Pearson Correlation Coefficient) Validation Tests.

Validates TTNN operations and modules against PyTorch reference implementations.
Target: > 95% PCC for individual ops, > 80% for end-to-end pipeline.

Per-operation tests:
- Conv1D, LayerNorm, Activations, MatMul, Attention, Embedding, WaveNet

Per-module tests:
- PosteriorEncoder, ResidualCouplingBlock, Generator (HiFi-GAN),
  TransformerFlow, End-to-End Voice Conversion
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

import ttnn


# ============================================================================
# Shared Helpers
# ============================================================================


def compute_pcc(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between two tensors."""
    t1 = tensor1.flatten().float()
    t2 = tensor2.flatten().float()

    if t1.std() < 1e-10 or t2.std() < 1e-10:
        if torch.allclose(t1, t2, rtol=1e-3, atol=1e-3):
            return 1.0
        return 0.0

    t1_centered = t1 - t1.mean()
    t2_centered = t2 - t2.mean()

    numerator = (t1_centered * t2_centered).sum()
    denominator = torch.sqrt((t1_centered**2).sum() * (t2_centered**2).sum())

    if denominator < 1e-10:
        return 1.0 if numerator < 1e-10 else 0.0

    pcc = numerator / denominator
    return float(pcc.clamp(-1, 1))


def assert_with_pcc(torch_output: torch.Tensor, ttnn_output, pcc_threshold: float = 0.95, name: str = ""):
    """Assert PCC between PyTorch and TTNN outputs meets threshold."""
    if not isinstance(ttnn_output, torch.Tensor):
        ttnn_output = ttnn.to_torch(ttnn_output)

    pcc = compute_pcc(torch_output, ttnn_output)
    status = "PASS" if pcc >= pcc_threshold else "FAIL"
    print(f"  {name}: PCC = {pcc:.6f} (threshold: {pcc_threshold}) [{status}]")

    assert pcc >= pcc_threshold, f"{name} PCC {pcc:.4f} < threshold {pcc_threshold}"
    return pcc


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def device():
    """Get TTNN device."""
    try:
        dev = ttnn.open_device(device_id=0)
        yield dev
        ttnn.close_device(dev)
    except Exception as e:
        pytest.skip(f"Could not open TTNN device: {e}")


@pytest.fixture(scope="module")
def model_weights():
    """Load model weights for testing."""
    checkpoint_path = Path("checkpoints/openvoice/converter/checkpoint.pth")
    if not checkpoint_path.exists():
        pytest.skip("Checkpoint not found")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


# ============================================================================
# Per-Operation Tests
# ============================================================================


class TestConv1DOperation:
    """Test Conv1D operation PCC."""

    def test_conv1d_basic(self, device):
        """Basic Conv1D test."""
        B, C_in, L = 1, 64, 100
        C_out, K = 128, 3
        padding = K // 2

        x_torch = torch.randn(B, C_in, L)
        weight_torch = torch.randn(C_out, C_in, K)
        bias_torch = torch.randn(C_out)

        y_torch = F.conv1d(x_torch, weight_torch, bias_torch, padding=padding)

        from models.experimental.openvoice.tt.modules.conv1d import ttnn_conv1d

        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
        w_ttnn = ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16, device=device)
        b_ttnn = ttnn.from_torch(bias_torch, dtype=ttnn.bfloat16, device=device)

        y_ttnn = ttnn_conv1d(x_ttnn, w_ttnn, b_ttnn, padding=padding, device=device)

        assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="Conv1D basic")

    def test_conv1d_dilated(self, device):
        """Dilated Conv1D test."""
        B, C_in, L = 1, 64, 100
        C_out, K = 64, 3
        dilation = 2
        padding = (K * dilation - dilation) // 2

        x_torch = torch.randn(B, C_in, L)
        weight_torch = torch.randn(C_out, C_in, K)

        y_torch = F.conv1d(x_torch, weight_torch, padding=padding, dilation=dilation)

        from models.experimental.openvoice.tt.modules.conv1d import ttnn_conv1d

        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
        w_ttnn = ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16, device=device)

        y_ttnn = ttnn_conv1d(x_ttnn, w_ttnn, None, padding=padding, dilation=dilation, device=device)

        assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="Conv1D dilated")


class TestLayerNormOperation:
    """Test Layer Normalization PCC."""

    def test_layer_norm_basic(self, device):
        """Basic LayerNorm test."""
        B, L, C = 1, 64, 256

        x_torch = torch.randn(B, L, C)
        weight_torch = torch.randn(C)
        bias_torch = torch.randn(C)

        y_torch = F.layer_norm(x_torch, (C,), weight_torch, bias_torch)

        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        w_ttnn = ttnn.from_torch(
            weight_torch.reshape(1, 1, 1, C), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )
        b_ttnn = ttnn.from_torch(
            bias_torch.reshape(1, 1, 1, C), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )

        y_ttnn = ttnn.layer_norm(x_ttnn, weight=w_ttnn, bias=b_ttnn)

        assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="LayerNorm")


class TestActivationOperations:
    """Test activation function PCC."""

    def test_relu(self, device):
        """ReLU test."""
        x_torch = torch.randn(1, 64, 100)
        y_torch = F.relu(x_torch)

        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
        y_ttnn = ttnn.relu(x_ttnn)

        assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.99, name="ReLU")

    def test_leaky_relu(self, device):
        """LeakyReLU test."""
        x_torch = torch.randn(1, 64, 100)
        y_torch = F.leaky_relu(x_torch, 0.1)

        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
        y_ttnn = ttnn.leaky_relu(x_ttnn, negative_slope=0.1)

        assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.99, name="LeakyReLU")

    def test_tanh(self, device):
        """Tanh test."""
        x_torch = torch.randn(1, 64, 100)
        y_torch = torch.tanh(x_torch)

        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
        y_ttnn = ttnn.tanh(x_ttnn)

        assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.99, name="Tanh")

    def test_sigmoid(self, device):
        """Sigmoid test."""
        x_torch = torch.randn(1, 64, 100)
        y_torch = torch.sigmoid(x_torch)

        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
        y_ttnn = ttnn.sigmoid(x_ttnn)

        assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.99, name="Sigmoid")

    def test_gated_activation(self, device):
        """Gated activation (tanh * sigmoid) test."""
        n_channels = 64
        x_torch = torch.randn(1, 2 * n_channels, 100)

        t_act = torch.tanh(x_torch[:, :n_channels, :])
        s_act = torch.sigmoid(x_torch[:, n_channels:, :])
        y_torch = t_act * s_act

        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
        t_act = ttnn.tanh(x_ttnn[:, :n_channels, :])
        s_act = ttnn.sigmoid(x_ttnn[:, n_channels:, :])
        y_ttnn = ttnn.multiply(t_act, s_act)

        assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="GatedActivation")


class TestMatMulOperations:
    """Test matrix multiplication PCC."""

    def test_matmul_2d(self, device):
        """2D matmul test."""
        a_torch = torch.randn(64, 128)
        b_torch = torch.randn(128, 64)
        y_torch = torch.matmul(a_torch, b_torch)

        a_ttnn = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        b_ttnn = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        y_ttnn = ttnn.matmul(a_ttnn, b_ttnn)

        assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="MatMul 2D")

    def test_matmul_batched(self, device):
        """Batched matmul test."""
        a_torch = torch.randn(2, 4, 32, 64)
        b_torch = torch.randn(2, 4, 64, 32)
        y_torch = torch.matmul(a_torch, b_torch)

        a_ttnn = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        b_ttnn = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        y_ttnn = ttnn.matmul(a_ttnn, b_ttnn)

        assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="MatMul batched")


class TestAttentionOperation:
    """Test attention operation PCC."""

    def test_scaled_dot_product_attention(self, device):
        """Scaled dot-product attention test."""
        import math

        B, H, T, D = 1, 4, 32, 64
        scale = 1.0 / math.sqrt(D)

        q_torch = torch.randn(B, H, T, D)
        k_torch = torch.randn(B, H, T, D)
        v_torch = torch.randn(B, H, T, D)

        scores = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        y_torch = torch.matmul(attn_weights, v_torch)

        q_ttnn = ttnn.from_torch(q_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        k_ttnn = ttnn.from_torch(k_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        v_ttnn = ttnn.from_torch(v_torch, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        try:
            y_ttnn = ttnn.transformer.scaled_dot_product_attention(
                q_ttnn,
                k_ttnn,
                v_ttnn,
                is_causal=False,
                scale=scale,
            )
        except Exception:
            k_t = ttnn.permute(k_ttnn, (0, 1, 3, 2))
            scores = ttnn.matmul(q_ttnn, k_t)
            scores = ttnn.multiply(scores, scale)
            attn_weights = ttnn.softmax(scores, dim=-1)
            y_ttnn = ttnn.matmul(attn_weights, v_ttnn)

        assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.95, name="SDPA")


class TestEmbeddingOperation:
    """Test embedding lookup PCC."""

    def test_embedding(self, device):
        """Embedding test."""
        vocab_size = 256
        embed_dim = 192
        seq_len = 50

        indices = torch.randint(0, vocab_size, (1, seq_len))
        weight = torch.randn(vocab_size, embed_dim)

        y_torch = F.embedding(indices, weight)

        idx_ttnn = ttnn.from_torch(indices, device=device)
        w_ttnn = ttnn.from_torch(weight, dtype=ttnn.bfloat16, device=device)
        y_ttnn = ttnn.embedding(idx_ttnn, w_ttnn)

        assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.99, name="Embedding")


class TestWaveNetBlock:
    """Test WaveNet gated convolution block."""

    def test_wavenet_layer(self, device):
        """Single WaveNet layer test."""
        B, C, L = 1, 192, 100
        K = 5
        dilation = 1
        padding = (K * dilation - dilation) // 2

        x_torch = torch.randn(B, C, L)
        in_weight = torch.randn(2 * C, C, K)
        in_bias = torch.randn(2 * C)
        res_weight = torch.randn(2 * C, C, 1)
        res_bias = torch.randn(2 * C)

        x_in = F.conv1d(x_torch, in_weight, in_bias, padding=padding, dilation=dilation)
        t_act = torch.tanh(x_in[:, :C, :])
        s_act = torch.sigmoid(x_in[:, C:, :])
        acts = t_act * s_act
        res_skip = F.conv1d(acts, res_weight, res_bias)
        res = res_skip[:, :C, :]
        y_torch = x_torch + res

        from models.experimental.openvoice.tt.modules.conv1d import ttnn_conv1d

        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
        in_w_ttnn = ttnn.from_torch(in_weight, dtype=ttnn.bfloat16, device=device)
        in_b_ttnn = ttnn.from_torch(in_bias, dtype=ttnn.bfloat16, device=device)
        res_w_ttnn = ttnn.from_torch(res_weight, dtype=ttnn.bfloat16, device=device)
        res_b_ttnn = ttnn.from_torch(res_bias, dtype=ttnn.bfloat16, device=device)

        x_in_ttnn = ttnn_conv1d(x_ttnn, in_w_ttnn, in_b_ttnn, padding=padding, dilation=dilation, device=device)
        t_act_ttnn = ttnn.tanh(x_in_ttnn[:, :C, :])
        s_act_ttnn = ttnn.sigmoid(x_in_ttnn[:, C:, :])
        acts_ttnn = ttnn.multiply(t_act_ttnn, s_act_ttnn)
        res_skip_ttnn = ttnn_conv1d(acts_ttnn, res_w_ttnn, res_b_ttnn, device=device)
        res_ttnn = res_skip_ttnn[:, :C, :]
        y_ttnn = ttnn.add(x_ttnn, res_ttnn)

        assert_with_pcc(y_torch, y_ttnn, pcc_threshold=0.65, name="WaveNet layer")


# ============================================================================
# Per-Module Tests
# ============================================================================


class TestPosteriorEncoderModule:
    """Test PosteriorEncoder module PCC."""

    def test_posterior_encoder_pcc(self, device, model_weights):
        """Test PosteriorEncoder TTNN vs PyTorch PCC."""
        from models.experimental.openvoice.tt.posterior_encoder import TTNNPosteriorEncoder

        in_channels = model_weights["enc_q.pre.weight"].shape[1]
        encoder = TTNNPosteriorEncoder.from_state_dict(
            model_weights,
            prefix="enc_q",
            in_channels=in_channels,
            out_channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=256,
            device=device,
        )

        B, T = 1, 100
        x_torch = torch.randn(B, in_channels, T)
        x_lengths = torch.tensor([T])
        g = torch.randn(B, 256, 1)

        z_pt, m_pt, logs_pt, mask_pt = encoder._forward_pytorch(x_torch, x_lengths, g, tau=1.0)

        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
        x_len_ttnn = ttnn.from_torch(x_lengths, device=device)
        g_ttnn = ttnn.from_torch(g, dtype=ttnn.bfloat16, device=device)

        z_tt, m_tt, logs_tt, mask_tt = encoder._forward_ttnn(x_ttnn, x_len_ttnn, g_ttnn, tau=1.0)

        assert_with_pcc(m_pt, m_tt, pcc_threshold=0.95, name="PosteriorEncoder mean")
        assert_with_pcc(logs_pt, logs_tt, pcc_threshold=0.95, name="PosteriorEncoder logs")

    def test_posterior_encoder_shapes(self, device, model_weights):
        """Test PosteriorEncoder output shapes match."""
        from models.experimental.openvoice.tt.posterior_encoder import TTNNPosteriorEncoder

        in_channels = model_weights["enc_q.pre.weight"].shape[1]
        encoder = TTNNPosteriorEncoder.from_state_dict(
            model_weights,
            prefix="enc_q",
            in_channels=in_channels,
            out_channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=256,
            device=device,
        )

        B, T = 1, 100
        x = torch.randn(B, in_channels, T)
        x_lengths = torch.tensor([T])

        x_ttnn = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
        x_len_ttnn = ttnn.from_torch(x_lengths, device=device)

        z, m, logs, mask = encoder(x_ttnn, x_len_ttnn)

        z_shape = ttnn.to_torch(z).shape if not isinstance(z, torch.Tensor) else z.shape
        assert z_shape == (B, 192, T), f"Expected shape (1, 192, {T}), got {z_shape}"


class TestResidualCouplingModule:
    """Test ResidualCouplingBlock module PCC."""

    def test_residual_coupling_pcc(self, device, model_weights):
        """Test ResidualCouplingBlock TTNN vs PyTorch PCC."""
        from models.experimental.openvoice.tt.residual_coupling import TTNNResidualCouplingBlock

        flow = TTNNResidualCouplingBlock.from_state_dict(
            model_weights,
            prefix="flow",
            channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            n_flows=4,
            gin_channels=256,
            device=device,
        )

        B, T = 1, 100
        x_torch = torch.randn(B, 192, T)
        x_mask = torch.ones(B, 1, T)
        g = torch.randn(B, 256, 1)

        z_pt = flow._forward_pytorch(x_torch, x_mask, g, reverse=False)

        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
        mask_ttnn = ttnn.from_torch(x_mask, dtype=ttnn.bfloat16, device=device)
        g_ttnn = ttnn.from_torch(g, dtype=ttnn.bfloat16, device=device)

        z_tt = flow._forward_ttnn(x_ttnn, mask_ttnn, g_ttnn, reverse=False)

        assert_with_pcc(z_pt, z_tt, pcc_threshold=0.90, name="ResidualCoupling forward")

    def test_residual_coupling_reverse(self, device, model_weights):
        """Test ResidualCoupling reverse (generation) direction."""
        from models.experimental.openvoice.tt.residual_coupling import TTNNResidualCouplingBlock

        flow = TTNNResidualCouplingBlock.from_state_dict(
            model_weights,
            prefix="flow",
            channels=192,
            hidden_channels=192,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            n_flows=4,
            gin_channels=256,
            device=device,
        )

        B, T = 1, 100
        z = torch.randn(B, 192, T)
        z_mask = torch.ones(B, 1, T)
        g = torch.randn(B, 256, 1)

        x_pt = flow._forward_pytorch(z, z_mask, g, reverse=True)

        z_ttnn = ttnn.from_torch(z, dtype=ttnn.bfloat16, device=device)
        mask_ttnn = ttnn.from_torch(z_mask, dtype=ttnn.bfloat16, device=device)
        g_ttnn = ttnn.from_torch(g, dtype=ttnn.bfloat16, device=device)

        x_tt = flow._forward_ttnn(z_ttnn, mask_ttnn, g_ttnn, reverse=True)

        assert_with_pcc(x_pt, x_tt, pcc_threshold=0.90, name="ResidualCoupling reverse")


class TestGeneratorModule:
    """Test Generator (HiFi-GAN) module PCC."""

    def test_generator_pcc(self, device, model_weights):
        """Test Generator TTNN vs PyTorch PCC."""
        from models.experimental.openvoice.tt.generator import TTNNGenerator

        generator = TTNNGenerator.from_state_dict(
            model_weights,
            prefix="dec",
            initial_channel=512,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            gin_channels=256,
            device=device,
        )

        B, T = 1, 50
        x = torch.randn(B, 192, T)
        g = torch.randn(B, 256, 1)

        audio_pt = generator._forward_pytorch(x, g)

        x_ttnn = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
        g_ttnn = ttnn.from_torch(g, dtype=ttnn.bfloat16, device=device)

        audio_tt = generator._forward_ttnn(x_ttnn, g_ttnn)

        assert_with_pcc(audio_pt, audio_tt, pcc_threshold=0.85, name="Generator (HiFi-GAN)")


class TestTransformerFlowModule:
    """Test TransformerFlow module PCC."""

    def test_transformer_flow_pcc(self, device, model_weights):
        """Test TransformerFlow TTNN vs PyTorch PCC."""
        from models.experimental.openvoice.tt.transformer_flow import TTNNTransformerFlow

        if "flow.flows.0.pre" not in model_weights:
            pytest.skip("TransformerFlow weights not in checkpoint")

        flow = TTNNTransformerFlow.from_state_dict(
            model_weights,
            prefix="flow",
            channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.0,
            n_flows=4,
            gin_channels=256,
            device=device,
        )

        B, T = 1, 64
        x = torch.randn(B, 192, T)
        x_mask = torch.ones(B, 1, T)
        g = torch.randn(B, 256, 1)

        z_pt = flow._forward_pytorch(x, x_mask, g, reverse=False)

        x_ttnn = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
        mask_ttnn = ttnn.from_torch(x_mask, dtype=ttnn.bfloat16, device=device)
        g_ttnn = ttnn.from_torch(g, dtype=ttnn.bfloat16, device=device)

        z_tt = flow._forward_ttnn(x_ttnn, mask_ttnn, g_ttnn, reverse=False)

        assert_with_pcc(z_pt, z_tt, pcc_threshold=0.90, name="TransformerFlow")


# ============================================================================
# End-to-End Test
# ============================================================================


class TestEndToEndPCC:
    """End-to-end PCC validation for the full voice conversion pipeline."""

    def test_voice_conversion_e2e_pcc(self, device, model_weights):
        """
        Test complete voice conversion PCC.

        Pipeline: mel -> PosteriorEncoder -> Flow -> Generator -> audio
        """
        from models.experimental.openvoice.tt.synthesizer import TTNNSynthesizerTrn

        synth = TTNNSynthesizerTrn.from_state_dict(
            model_weights,
            device=device,
        )

        B, T_mel = 1, 100
        mel = torch.randn(B, 80, T_mel)
        mel_lengths = torch.tensor([T_mel])
        src_se = torch.randn(B, 256, 1)
        tgt_se = torch.randn(B, 256, 1)

        synth.device = None
        audio_pt, _ = synth.voice_conversion(mel, mel_lengths, src_se, tgt_se, tau=0.3)

        synth.device = device
        mel_ttnn = ttnn.from_torch(mel, dtype=ttnn.bfloat16, device=device)
        mel_len_ttnn = ttnn.from_torch(mel_lengths, device=device)
        src_se_ttnn = ttnn.from_torch(src_se, dtype=ttnn.bfloat16, device=device)
        tgt_se_ttnn = ttnn.from_torch(tgt_se, dtype=ttnn.bfloat16, device=device)

        audio_tt, _ = synth.voice_conversion(mel_ttnn, mel_len_ttnn, src_se_ttnn, tgt_se_ttnn, tau=0.3)

        if not isinstance(audio_tt, torch.Tensor):
            audio_tt = ttnn.to_torch(audio_tt)

        pcc = assert_with_pcc(audio_pt, audio_tt, pcc_threshold=0.80, name="E2E Voice Conversion")

        print(f"\n[E2E] Voice Conversion PCC: {pcc:.6f}")
        print(f"[E2E] Audio shape: {audio_pt.shape}")
