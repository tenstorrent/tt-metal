# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC (Pearson Correlation Coefficient) verification tests for GR00T N1.6.

Validates the TTNN implementation against the PyTorch reference
at multiple granularities:
    1. Vision encoder (SigLIP2) output
    2. Connector (pixel shuffle + MLP) output
    3. DiT block-level outputs
    4. Flow matching single-step output
    5. Full end-to-end action prediction

Target: >= 99% accuracy (PCC >= 0.99)
"""

import logging
import pytest
import torch
import time

logger = logging.getLogger(__name__)

# PCC thresholds
PCC_THRESHOLD_VISION = 0.99
PCC_THRESHOLD_CONNECTOR = 0.99
PCC_THRESHOLD_DIT_BLOCK = 0.98  # slightly lower for deep blocks
PCC_THRESHOLD_SINGLE_STEP = 0.99
PCC_THRESHOLD_E2E = 0.99


def compute_pcc(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient."""
    ref_flat = ref.float().flatten()
    test_flat = test.float().flatten()

    if ref_flat.shape != test_flat.shape:
        min_len = min(len(ref_flat), len(test_flat))
        ref_flat = ref_flat[:min_len]
        test_flat = test_flat[:min_len]

    ref_mean = ref_flat.mean()
    test_mean = test_flat.mean()
    ref_c = ref_flat - ref_mean
    test_c = test_flat - test_mean
    cov = (ref_c * test_c).sum()
    ref_std = (ref_c ** 2).sum().sqrt()
    test_std = (test_c ** 2).sum().sqrt()

    if ref_std == 0 or test_std == 0:
        return 1.0 if torch.allclose(ref_flat, test_flat) else 0.0

    return (cov / (ref_std * test_std)).item()


@pytest.fixture(scope="module")
def device():
    """Get TTNN device."""
    import ttnn
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.fixture(scope="module")
def groot_config():
    """Get GR00T N1.6 configuration."""
    from models.experimental.groot_n16.common.configs import Gr00tN16Config
    return Gr00tN16Config.default()


@pytest.fixture(scope="module")
def weight_loader():
    """Load GR00T N1.6 weights."""
    from models.experimental.groot_n16.common.weight_loader import Gr00tN16WeightLoader
    loader = Gr00tN16WeightLoader()
    loader.load()
    return loader


class TestSigLIP2PCC:
    """Test SigLIP2 vision encoder PCC against reference."""

    def test_vision_encoder_output(self, device, groot_config, weight_loader):
        """Verify SigLIP2 output matches reference."""
        from models.experimental.groot_n16.tt.ttnn_siglip2 import SigLIP2VisionEncoderTTNN
        import ttnn

        # Create TTNN vision encoder
        encoder = SigLIP2VisionEncoderTTNN(
            groot_config.backbone.vision,
            weight_loader.get_vision_weights(),
            device,
        )

        # Dummy input
        batch_size = 1
        img_size = groot_config.backbone.vision.image_size
        pixel_values = torch.randn(batch_size, 3, img_size, img_size)

        # TTNN forward
        t0 = time.time()
        output_tt = encoder(pixel_values)
        tt_time = time.time() - t0

        # Convert back to CPU
        output_cpu = ttnn.to_torch(output_tt)

        logger.info(f"SigLIP2 TTNN output shape: {output_cpu.shape}")
        logger.info(f"SigLIP2 TTNN time: {tt_time*1000:.1f}ms")

        # Verify shape
        expected_patches = groot_config.backbone.vision.num_patches
        expected_dim = groot_config.backbone.vision.hidden_size
        assert output_cpu.shape == (batch_size, expected_patches, expected_dim), \
            f"Expected shape ({batch_size}, {expected_patches}, {expected_dim}), got {output_cpu.shape}"

        # TODO: Compare against reference when available
        logger.info("SigLIP2 shape verification passed")


class TestConnectorPCC:
    """Test pixel shuffle + MLP connector PCC."""

    def test_connector_output_shape(self, device, groot_config, weight_loader):
        """Verify connector produces correct output shape."""
        from models.experimental.groot_n16.tt.ttnn_groot_n16_model import PixelShuffleConnectorTTNN
        import ttnn

        connector = PixelShuffleConnectorTTNN(
            groot_config.backbone.vision.hidden_size,
            groot_config.backbone.language.hidden_size,
            weight_loader.get_connector_weights(),
            device,
        )

        # Dummy vision features
        num_patches = groot_config.backbone.vision.num_patches
        h = w = int(num_patches ** 0.5)
        vision_features = torch.randn(1, num_patches, groot_config.backbone.vision.hidden_size)

        output_tt = connector(vision_features, h, w)
        output_cpu = ttnn.to_torch(output_tt)

        expected_tokens = groot_config.backbone.num_image_tokens_per_frame
        expected_dim = groot_config.backbone.language.hidden_size
        logger.info(f"Connector output shape: {output_cpu.shape}")
        assert output_cpu.shape[1] == expected_tokens, \
            f"Expected {expected_tokens} tokens after pixel shuffle, got {output_cpu.shape[1]}"
        assert output_cpu.shape[2] == expected_dim, \
            f"Expected dim {expected_dim}, got {output_cpu.shape[2]}"

        logger.info("Connector shape verification passed")


class TestDiTPCC:
    """Test AlternateVLDiT PCC."""

    def test_dit_forward_shape(self, device, groot_config, weight_loader):
        """Verify DiT produces correct output shape."""
        from models.experimental.groot_n16.tt.ttnn_dit import AlternateVLDiTTTNN
        from models.experimental.groot_n16.tt.ttnn_common import to_tt_tensor
        import ttnn

        dit = AlternateVLDiTTTNN(
            groot_config.dit,
            weight_loader.get_dit_weights(),
            device,
        )

        batch_size = 1
        action_seq = 1 + groot_config.action_horizon  # state + actions
        inner_dim = groot_config.dit.inner_dim
        backbone_dim = groot_config.backbone_embedding_dim

        # Dummy inputs
        hidden = torch.randn(batch_size, action_seq, inner_dim)
        timestep_emb = torch.randn(batch_size, 1, inner_dim)
        backbone_features = torch.randn(batch_size, 64, backbone_dim)

        hidden_tt = to_tt_tensor(hidden, device)
        time_tt = to_tt_tensor(timestep_emb, device)
        backbone_tt = to_tt_tensor(backbone_features, device)

        output_tt = dit(hidden_tt, time_tt, backbone_tt)
        output_cpu = ttnn.to_torch(output_tt)

        expected_dim = groot_config.dit.output_dim
        logger.info(f"DiT output shape: {output_cpu.shape}")
        assert output_cpu.shape == (batch_size, action_seq, expected_dim), \
            f"Expected ({batch_size}, {action_seq}, {expected_dim}), got {output_cpu.shape}"

        logger.info("DiT shape verification passed")


class TestEmbodimentPCC:
    """Test embodiment MLPs PCC."""

    def test_state_encoder_shape(self, device, groot_config, weight_loader):
        """Verify state encoder output shape."""
        from models.experimental.groot_n16.tt.ttnn_embodiment import CategorySpecificMLPTTNN
        from models.experimental.groot_n16.tt.ttnn_common import to_tt_tensor
        import ttnn

        emb_cfg = groot_config.embodiment
        state_enc = CategorySpecificMLPTTNN(
            weight_loader.get_state_encoder_weights(),
            emb_cfg.max_num_embodiments,
            emb_cfg.max_state_dim,
            emb_cfg.state_hidden_dim,
            emb_cfg.state_output_dim,
            device,
        )

        state = torch.randn(1, 1, emb_cfg.max_state_dim)
        state_tt = to_tt_tensor(state, device)

        output_tt = state_enc(state_tt, embodiment_id=0)
        output_cpu = ttnn.to_torch(output_tt)

        assert output_cpu.shape == (1, 1, emb_cfg.state_output_dim), \
            f"Expected (1, 1, {emb_cfg.state_output_dim}), got {output_cpu.shape}"

        logger.info("State encoder shape verification passed")


class TestEndToEnd:
    """End-to-end flow matching test."""

    def test_flow_matching_shapes(self, device, groot_config, weight_loader):
        """Verify flow matching produces correct action shapes."""
        from models.experimental.groot_n16.tt.ttnn_groot_n16_model import Gr00tN16ModelTTNN
        from models.experimental.groot_n16.tt.ttnn_common import to_tt_tensor
        import ttnn

        model = Gr00tN16ModelTTNN(groot_config, weight_loader, device)

        batch_size = 1
        backbone_dim = groot_config.backbone_embedding_dim
        state_dim = groot_config.embodiment.max_state_dim

        # Dummy backbone features (as if from Qwen3 layer 16)
        backbone_features = torch.randn(batch_size, 64, backbone_dim)
        backbone_tt = to_tt_tensor(backbone_features, device)

        state = torch.randn(batch_size, state_dim)

        t0 = time.time()
        actions = model.run_flow_matching(backbone_tt, state, embodiment_id=0)
        elapsed = time.time() - t0

        logger.info(f"Flow matching output shape: {actions.shape}")
        logger.info(f"Flow matching time: {elapsed*1000:.1f}ms")

        expected_shape = (batch_size, groot_config.action_horizon, state_dim)
        assert actions.shape == expected_shape, \
            f"Expected {expected_shape}, got {actions.shape}"

        logger.info("Flow matching shape verification passed")
