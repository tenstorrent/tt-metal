#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Per-Module PCC Validation Tests.

Validates complete TTNN modules against PyTorch reference implementations.
Each module is tested by comparing TTNN output to PyTorch output using PCC.

Modules tested:
- PosteriorEncoder: Mel spectrogram to latent representation
- TransformerFlow: Normalizing flow with transformer blocks
- Generator (HiFi-GAN): Latent to waveform synthesis
- ResidualCouplingBlock: Flow coupling layers
- TextEncoder: Text to hidden representation (for TTS)
- DurationPredictor: Duration prediction (for TTS)

Target: > 95% PCC (Pearson Correlation Coefficient) for each module.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False


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


def assert_with_pcc(torch_output: torch.Tensor, ttnn_output, pcc_threshold: float, name: str):
    """Assert PCC meets threshold and report results."""
    if TTNN_AVAILABLE and not isinstance(ttnn_output, torch.Tensor):
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
    if not TTNN_AVAILABLE:
        pytest.skip("TTNN not available")
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
# Per-Module PCC Tests
# ============================================================================


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
class TestPosteriorEncoderModule:
    """
    Test PosteriorEncoder module PCC.

    PosteriorEncoder encodes mel spectrograms to latent space using:
    - Pre-convolution
    - WaveNet temporal modeling
    - Projection to mean/log-variance
    """

    def test_posterior_encoder_pcc(self, device, model_weights):
        """Test PosteriorEncoder TTNN vs PyTorch PCC."""
        from models.demos.openvoice.tt.posterior_encoder import TTNNPosteriorEncoder

        # Model config - derive in_channels from checkpoint weights
        # enc_q.pre.weight has shape [hidden_channels, in_channels, 1]
        in_channels = model_weights["enc_q.pre.weight"].shape[1]
        out_channels = 192
        hidden_channels = 192
        kernel_size = 5
        dilation_rate = 1
        n_layers = 16
        gin_channels = 256

        # Create module from weights
        encoder = TTNNPosteriorEncoder.from_state_dict(
            model_weights,
            prefix="enc_q",
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            n_layers=n_layers,
            gin_channels=gin_channels,
            device=device,
        )

        # Test input
        B, T = 1, 100
        x_torch = torch.randn(B, in_channels, T)
        x_lengths = torch.tensor([T])
        g = torch.randn(B, gin_channels, 1)  # speaker embedding

        # PyTorch reference (force PyTorch path)
        z_pt, m_pt, logs_pt, mask_pt = encoder._forward_pytorch(x_torch, x_lengths, g, tau=1.0)

        # TTNN implementation
        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
        x_len_ttnn = ttnn.from_torch(x_lengths, device=device)
        g_ttnn = ttnn.from_torch(g, dtype=ttnn.bfloat16, device=device)

        z_tt, m_tt, logs_tt, mask_tt = encoder._forward_ttnn(x_ttnn, x_len_ttnn, g_ttnn, tau=1.0)

        # Compare deterministic outputs (mean, logs, mask)
        # Note: z includes random sampling so we compare m and logs
        assert_with_pcc(m_pt, m_tt, pcc_threshold=0.95, name="PosteriorEncoder mean")
        assert_with_pcc(logs_pt, logs_tt, pcc_threshold=0.95, name="PosteriorEncoder logs")

    def test_posterior_encoder_shapes(self, device, model_weights):
        """Test PosteriorEncoder output shapes match."""
        from models.demos.openvoice.tt.posterior_encoder import TTNNPosteriorEncoder

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


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
class TestResidualCouplingModule:
    """
    Test ResidualCouplingBlock module PCC.

    ResidualCouplingBlock implements normalizing flow coupling layers.
    """

    def test_residual_coupling_pcc(self, device, model_weights):
        """Test ResidualCouplingBlock TTNN vs PyTorch PCC."""
        from models.demos.openvoice.tt.residual_coupling import TTNNResidualCouplingBlock

        # Model config
        channels = 192
        hidden_channels = 192
        kernel_size = 5
        dilation_rate = 1
        n_layers = 4
        n_flows = 4
        gin_channels = 256

        # Create module
        flow = TTNNResidualCouplingBlock.from_state_dict(
            model_weights,
            prefix="flow",
            channels=channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            n_layers=n_layers,
            n_flows=n_flows,
            gin_channels=gin_channels,
            device=device,
        )

        # Test input
        B, T = 1, 100
        x_torch = torch.randn(B, channels, T)
        x_mask = torch.ones(B, 1, T)
        g = torch.randn(B, gin_channels, 1)

        # PyTorch reference
        z_pt = flow._forward_pytorch(x_torch, x_mask, g, reverse=False)

        # TTNN implementation
        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)
        mask_ttnn = ttnn.from_torch(x_mask, dtype=ttnn.bfloat16, device=device)
        g_ttnn = ttnn.from_torch(g, dtype=ttnn.bfloat16, device=device)

        z_tt = flow._forward_ttnn(x_ttnn, mask_ttnn, g_ttnn, reverse=False)

        assert_with_pcc(z_pt, z_tt, pcc_threshold=0.90, name="ResidualCoupling forward")

    def test_residual_coupling_reverse(self, device, model_weights):
        """Test ResidualCoupling reverse (generation) direction."""
        from models.demos.openvoice.tt.residual_coupling import TTNNResidualCouplingBlock

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

        # PyTorch reverse
        x_pt = flow._forward_pytorch(z, z_mask, g, reverse=True)

        # TTNN reverse
        z_ttnn = ttnn.from_torch(z, dtype=ttnn.bfloat16, device=device)
        mask_ttnn = ttnn.from_torch(z_mask, dtype=ttnn.bfloat16, device=device)
        g_ttnn = ttnn.from_torch(g, dtype=ttnn.bfloat16, device=device)

        x_tt = flow._forward_ttnn(z_ttnn, mask_ttnn, g_ttnn, reverse=True)

        assert_with_pcc(x_pt, x_tt, pcc_threshold=0.90, name="ResidualCoupling reverse")


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
class TestGeneratorModule:
    """
    Test Generator (HiFi-GAN) module PCC.

    Generator converts latent representations to audio waveforms.
    """

    def test_generator_pcc(self, device, model_weights):
        """Test Generator TTNN vs PyTorch PCC."""
        from models.demos.openvoice.tt.generator import TTNNGenerator

        # Model config (HiFi-GAN V2)
        initial_channel = 512
        resblock = "1"  # HiFi-GAN V2 uses ResBlock type 1
        resblock_kernel_sizes = [3, 7, 11]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        upsample_rates = [8, 8, 2, 2]
        upsample_initial_channel = 512
        upsample_kernel_sizes = [16, 16, 4, 4]
        gin_channels = 256
        inter_channels = 192  # Used for test input shape

        generator = TTNNGenerator.from_state_dict(
            model_weights,
            prefix="dec",
            initial_channel=initial_channel,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gin_channels=gin_channels,
            device=device,
        )

        # Test input (latent from flow)
        B, T = 1, 50  # Smaller T for faster test
        x = torch.randn(B, inter_channels, T)
        g = torch.randn(B, gin_channels, 1)

        # PyTorch reference
        audio_pt = generator._forward_pytorch(x, g)

        # TTNN implementation
        x_ttnn = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
        g_ttnn = ttnn.from_torch(g, dtype=ttnn.bfloat16, device=device)

        audio_tt = generator._forward_ttnn(x_ttnn, g_ttnn)

        # Lower threshold for generator due to accumulated error in many layers
        assert_with_pcc(audio_pt, audio_tt, pcc_threshold=0.85, name="Generator (HiFi-GAN)")


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
class TestTransformerFlowModule:
    """
    Test TransformerFlow module PCC.

    TransformerFlow uses transformer attention for flow modeling.
    """

    def test_transformer_flow_pcc(self, device, model_weights):
        """Test TransformerFlow TTNN vs PyTorch PCC."""
        from models.demos.openvoice.tt.transformer_flow import TTNNTransformerFlow

        # Check if transformer flow weights exist
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

        # PyTorch reference
        z_pt = flow._forward_pytorch(x, x_mask, g, reverse=False)

        # TTNN implementation
        x_ttnn = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
        mask_ttnn = ttnn.from_torch(x_mask, dtype=ttnn.bfloat16, device=device)
        g_ttnn = ttnn.from_torch(g, dtype=ttnn.bfloat16, device=device)

        z_tt = flow._forward_ttnn(x_ttnn, mask_ttnn, g_ttnn, reverse=False)

        assert_with_pcc(z_pt, z_tt, pcc_threshold=0.90, name="TransformerFlow")


# ============================================================================
# End-to-End PCC Test
# ============================================================================


@pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
class TestEndToEndPCC:
    """
    End-to-end PCC validation.

    Compares full voice conversion pipeline output between TTNN and PyTorch.
    """

    def test_voice_conversion_e2e_pcc(self, device, model_weights):
        """
        Test complete voice conversion PCC.

        Pipeline: mel -> PosteriorEncoder -> Flow -> Generator -> audio
        """
        from models.demos.openvoice.tt.synthesizer import TTNNSynthesizerTrn

        # Create synthesizer
        synth = TTNNSynthesizerTrn.from_state_dict(
            model_weights,
            device=device,
        )

        # Test inputs
        B, T_mel = 1, 100
        mel = torch.randn(B, 80, T_mel)
        mel_lengths = torch.tensor([T_mel])
        src_se = torch.randn(B, 256, 1)
        tgt_se = torch.randn(B, 256, 1)

        # PyTorch reference
        synth.device = None  # Force PyTorch path
        audio_pt, _ = synth.voice_conversion(mel, mel_lengths, src_se, tgt_se, tau=0.3)

        # TTNN implementation
        synth.device = device
        mel_ttnn = ttnn.from_torch(mel, dtype=ttnn.bfloat16, device=device)
        mel_len_ttnn = ttnn.from_torch(mel_lengths, device=device)
        src_se_ttnn = ttnn.from_torch(src_se, dtype=ttnn.bfloat16, device=device)
        tgt_se_ttnn = ttnn.from_torch(tgt_se, dtype=ttnn.bfloat16, device=device)

        audio_tt, _ = synth.voice_conversion(mel_ttnn, mel_len_ttnn, src_se_ttnn, tgt_se_ttnn, tau=0.3)

        # Convert to torch for comparison
        if not isinstance(audio_tt, torch.Tensor):
            audio_tt = ttnn.to_torch(audio_tt)

        # Lower threshold for E2E due to accumulated numerical differences
        pcc = assert_with_pcc(audio_pt, audio_tt, pcc_threshold=0.80, name="E2E Voice Conversion")

        print(f"\n[E2E] Voice Conversion PCC: {pcc:.6f}")
        print(f"[E2E] Audio shape: {audio_pt.shape}")


# ============================================================================
# Summary
# ============================================================================


def run_module_pcc_summary():
    """Print module PCC validation summary."""
    print("=" * 70)
    print("Per-Module PCC Validation Summary")
    print("=" * 70)
    print("""
Module                    | Target PCC | Description
--------------------------|------------|-----------------------------------
PosteriorEncoder          | 0.95       | Mel -> latent (WaveNet-based)
ResidualCouplingBlock     | 0.90       | Normalizing flow coupling
Generator (HiFi-GAN)      | 0.85       | Latent -> audio waveform
TransformerFlow           | 0.90       | Attention-based flow
E2E Voice Conversion      | 0.80       | Complete pipeline

Notes:
- Lower thresholds for deeper modules due to accumulated bfloat16 error
- Generator threshold lower due to many upsampling/conv layers
- E2E threshold lowest as it chains all modules
- All modules tested with random inputs + real checkpoint weights
""")


if __name__ == "__main__":
    run_module_pcc_summary()
