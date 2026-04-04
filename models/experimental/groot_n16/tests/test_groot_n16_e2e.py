# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end inference test for GR00T N1.6 on Blackhole.

Tests the full pipeline: vision encoding + flow matching (4 Euler steps).
Backbone features are provided as dummy inputs since Qwen3 is not yet ported.

Run from the pi0 tt-metal directory:
    cd /home/ttuser/experiments/pi0/tt-metal
    export TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole
    PYTHONPATH=/home/ttuser/experiments/gr00t_n16/tt-metal pytest \\
        /home/ttuser/experiments/gr00t_n16/tt-metal/models/experimental/groot_n16/tests/test_groot_n16_e2e.py -svv
"""

import logging
import sys
import time

import pytest
import torch

sys.path.insert(0, "/home/ttuser/experiments/gr00t_n16/tt-metal")

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def tt_device():
    """Open and close TT device for the test module."""
    import ttnn

    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


@pytest.fixture(scope="module")
def weight_loader():
    """Load GR00T N1.6 weights (cached across tests)."""
    from models.experimental.groot_n16.common.weight_loader import Gr00tN16WeightLoader

    loader = Gr00tN16WeightLoader()
    loader.load()
    return loader


@pytest.fixture(scope="module")
def config():
    """GR00T N1.6 configuration."""
    from models.experimental.groot_n16.common.configs import Gr00tN16Config

    return Gr00tN16Config.default()


@pytest.fixture(scope="module")
def model(config, weight_loader, tt_device):
    """Build the full GR00T N1.6 TTNN model."""
    from models.experimental.groot_n16.tt.ttnn_groot_n16_model import Gr00tN16ModelTTNN

    return Gr00tN16ModelTTNN(config, weight_loader, tt_device)


class TestVisionEncoding:
    """Test vision encoding pipeline: SigLIP2 + pixel shuffle + connector."""

    def test_output_shape(self, model, config):
        """Verify vision encoding produces correct output shape."""
        import ttnn

        pixel_values = torch.randn(1, 3, 224, 224)
        image_tokens = model.encode_vision(pixel_values)
        output = ttnn.to_torch(image_tokens)

        expected_tokens = config.backbone.num_image_tokens_per_frame  # 64
        expected_dim = config.backbone.language.hidden_size  # 2048
        assert output.shape == (1, expected_tokens, expected_dim), \
            f"Expected (1, {expected_tokens}, {expected_dim}), got {output.shape}"

    def test_determinism(self, model):
        """Verify vision encoding is deterministic."""
        import ttnn

        torch.manual_seed(42)
        pv = torch.randn(1, 3, 224, 224)

        out1 = ttnn.to_torch(model.encode_vision(pv))
        out2 = ttnn.to_torch(model.encode_vision(pv))

        assert torch.allclose(out1, out2, atol=1e-3), \
            f"Non-deterministic: max diff = {(out1 - out2).abs().max()}"

    def test_latency(self, model):
        """Measure vision encoding latency."""
        pv = torch.randn(1, 3, 224, 224)
        model.encode_vision(pv)  # warmup

        times = []
        for _ in range(5):
            t0 = time.time()
            model.encode_vision(pv)
            times.append(time.time() - t0)

        avg_ms = sum(times) / len(times) * 1000
        logger.info(f"Vision encoding latency: {avg_ms:.1f}ms")
        assert avg_ms < 100, f"Vision encoding too slow: {avg_ms:.1f}ms (threshold: 100ms)"


class TestFlowMatching:
    """Test flow matching inference with dummy backbone features."""

    def test_output_shape(self, model, config, tt_device):
        """Verify flow matching produces correct action output shape."""
        from models.experimental.groot_n16.tt.ttnn_common import to_tt_tensor

        backbone = to_tt_tensor(torch.randn(1, 64, 2048), tt_device)
        state = torch.randn(1, config.embodiment.max_state_dim)

        actions = model.run_flow_matching(backbone, state, embodiment_id=0)

        assert actions.shape == (1, config.action_horizon, config.embodiment.max_action_dim), \
            f"Expected (1, {config.action_horizon}, {config.embodiment.max_action_dim}), got {actions.shape}"

    def test_finite_output(self, model, config, tt_device):
        """Verify flow matching produces finite (no NaN/Inf) outputs."""
        from models.experimental.groot_n16.tt.ttnn_common import to_tt_tensor

        backbone = to_tt_tensor(torch.randn(1, 64, 2048), tt_device)
        state = torch.randn(1, config.embodiment.max_state_dim)

        actions = model.run_flow_matching(backbone, state, embodiment_id=0)

        assert not actions.isnan().any(), "Actions contain NaN"
        assert not actions.isinf().any(), "Actions contain Inf"

    def test_different_timesteps_produce_different_actions(self, model, config, tt_device):
        """Verify that different noise seeds produce different action predictions."""
        from models.experimental.groot_n16.tt.ttnn_common import to_tt_tensor

        backbone = to_tt_tensor(torch.randn(1, 64, 2048), tt_device)
        state = torch.randn(1, config.embodiment.max_state_dim)

        torch.manual_seed(1)
        actions1 = model.run_flow_matching(backbone, state, embodiment_id=0)
        torch.manual_seed(2)
        actions2 = model.run_flow_matching(backbone, state, embodiment_id=0)

        # Different seeds should produce different actions
        assert not torch.allclose(actions1, actions2, atol=1e-2), \
            "Different seeds produced identical actions"

    def test_latency(self, model, config, tt_device):
        """Measure flow matching latency."""
        from models.experimental.groot_n16.tt.ttnn_common import to_tt_tensor

        backbone = to_tt_tensor(torch.randn(1, 64, 2048), tt_device)
        state = torch.randn(1, config.embodiment.max_state_dim)
        model.run_flow_matching(backbone, state, embodiment_id=0)  # warmup

        times = []
        for _ in range(3):
            b = to_tt_tensor(torch.randn(1, 64, 2048), tt_device)
            t0 = time.time()
            model.run_flow_matching(b, state, embodiment_id=0)
            times.append(time.time() - t0)

        avg_ms = sum(times) / len(times) * 1000
        logger.info(f"Flow matching latency (4 steps): {avg_ms:.1f}ms")
        assert avg_ms < 500, f"Flow matching too slow: {avg_ms:.1f}ms (threshold: 500ms)"


class TestEndToEnd:
    """Full end-to-end inference test."""

    def test_full_pipeline(self, model, config, tt_device):
        """Run vision encoding + flow matching as full pipeline."""
        import ttnn
        from models.experimental.groot_n16.tt.ttnn_common import to_tt_tensor

        # Step 1: Vision encoding
        pixel_values = torch.randn(1, 3, 224, 224)
        image_tokens = model.encode_vision(pixel_values)
        image_tokens_cpu = ttnn.to_torch(image_tokens)

        assert image_tokens_cpu.shape == (1, 64, 2048)

        # Step 2: Use image tokens as backbone features (simulating Qwen3 output)
        backbone_features = to_tt_tensor(image_tokens_cpu, tt_device)

        # Step 3: Flow matching
        state = torch.randn(1, config.embodiment.max_state_dim)
        actions = model.run_flow_matching(backbone_features, state, embodiment_id=0)

        assert actions.shape == (1, config.action_horizon, config.embodiment.max_action_dim)
        assert not actions.isnan().any()
        assert not actions.isinf().any()

        logger.info(f"E2E actions shape: {actions.shape}")
        logger.info(f"E2E actions range: [{actions.min():.4f}, {actions.max():.4f}]")

    def test_full_pipeline_latency(self, model, config, tt_device):
        """Measure full E2E latency."""
        import ttnn
        from models.experimental.groot_n16.tt.ttnn_common import to_tt_tensor

        pixel_values = torch.randn(1, 3, 224, 224)
        state = torch.randn(1, config.embodiment.max_state_dim)

        # Warmup
        img = model.encode_vision(pixel_values)
        backbone = to_tt_tensor(ttnn.to_torch(img), tt_device)
        model.run_flow_matching(backbone, state, embodiment_id=0)

        # Measure
        times = []
        for _ in range(3):
            t0 = time.time()
            img = model.encode_vision(pixel_values)
            backbone = to_tt_tensor(ttnn.to_torch(img), tt_device)
            actions = model.run_flow_matching(backbone, state, embodiment_id=0)
            times.append(time.time() - t0)

        avg_ms = sum(times) / len(times) * 1000
        actions_per_sec = 1000.0 / avg_ms
        logger.info(f"E2E latency: {avg_ms:.1f}ms")
        logger.info(f"Actions/sec: {actions_per_sec:.2f}")

        assert avg_ms < 500, f"E2E too slow: {avg_ms:.1f}ms"


class TestPCCValidation:
    """PCC validation against PyTorch reference for key components."""

    @staticmethod
    def compute_pcc(ref: torch.Tensor, test: torch.Tensor) -> float:
        ref_f = ref.float().flatten()
        test_f = test.float().flatten()
        min_len = min(len(ref_f), len(test_f))
        ref_f, test_f = ref_f[:min_len], test_f[:min_len]
        rc = ref_f - ref_f.mean()
        tc = test_f - test_f.mean()
        cov = (rc * tc).sum()
        return (cov / (rc.pow(2).sum().sqrt() * tc.pow(2).sum().sqrt())).item()

    def test_vision_encoder_pcc(self, model, config, weight_loader, tt_device):
        """Validate SigLIP2 + connector against PyTorch reference."""
        import ttnn
        from models.experimental.groot_n16.tests.test_groot_n16_pcc import _build_siglip2_ref_model

        torch.manual_seed(42)
        pv = torch.randn(1, 3, 224, 224)

        # TTNN
        tt_out = ttnn.to_torch(model.vision_encoder(pv))

        # Reference
        ref_model = _build_siglip2_ref_model(weight_loader.get_vision_weights())
        ref_model.eval()
        with torch.no_grad():
            ref_out = ref_model(pv)

        pcc = self.compute_pcc(ref_out, tt_out)
        logger.info(f"SigLIP2 PCC: {pcc:.6f}")
        assert pcc >= 0.95, f"SigLIP2 PCC {pcc:.4f} below 0.95 threshold"

    def test_dit_pcc(self, config, weight_loader, tt_device):
        """Validate DiT against upstream Isaac-GR00T reference."""
        import ttnn
        import importlib.util
        from models.experimental.groot_n16.tt.ttnn_dit import AlternateVLDiTTTNN
        from models.experimental.groot_n16.tt.ttnn_embodiment import TimestepEncoderTTNN
        from models.experimental.groot_n16.tt.ttnn_common import to_tt_tensor

        torch.manual_seed(42)
        hidden = torch.randn(1, 51, 1536)
        backbone = torch.randn(1, 64, 2048)

        # TTNN DiT
        tt_dit = AlternateVLDiTTTNN(config.dit, weight_loader.get_dit_weights(), tt_device)
        ts_enc = TimestepEncoderTTNN(weight_loader.get_timestep_encoder_weights(), tt_device)
        timestep_emb = ts_enc(torch.tensor([0]))

        tt_out = ttnn.to_torch(
            tt_dit(to_tt_tensor(hidden, tt_device), timestep_emb, to_tt_tensor(backbone, tt_device))
        ).float()

        # Reference DiT
        dit_path = "/home/ttuser/experiments/gr00t_n16/Isaac-GR00T/gr00t/model/modules/dit.py"
        spec = importlib.util.spec_from_file_location("gr00t_dit", dit_path)
        dit_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dit_mod)

        ref_dit = dit_mod.AlternateVLDiT(
            num_layers=32, num_attention_heads=32, attention_head_dim=48,
            norm_type="ada_norm", dropout=0.0, final_dropout=True, output_dim=1024,
            interleave_self_attention=True, cross_attention_dim=2048, attend_text_every_n_blocks=2,
        )
        dit_sd = {
            k[len("action_head.model."):]: v
            for k, v in weight_loader.state_dict.items()
            if k.startswith("action_head.model.")
        }
        ref_dit.load_state_dict(dit_sd, strict=False)
        ref_dit.float().eval()

        with torch.no_grad():
            ref_out = ref_dit(
                hidden_states=hidden, encoder_hidden_states=backbone,
                timestep=torch.tensor([0]),
                image_mask=torch.ones(1, 64, dtype=torch.bool),
                backbone_attention_mask=torch.ones(1, 64, dtype=torch.bool),
            )

        pcc = self.compute_pcc(ref_out, tt_out)
        logger.info(f"DiT PCC: {pcc:.6f}")
        assert pcc >= 0.95, f"DiT PCC {pcc:.4f} below 0.95 threshold"

    def test_embodiment_pcc(self, config, weight_loader, tt_device):
        """Validate embodiment state encoder PCC."""
        import ttnn
        from models.experimental.groot_n16.tt.ttnn_embodiment import CategorySpecificMLPTTNN
        from models.experimental.groot_n16.tt.ttnn_common import to_tt_tensor

        emb_cfg = config.embodiment
        state_weights = weight_loader.get_state_encoder_weights()

        # Reference
        w1 = state_weights["layer1.W"][0].float()
        b1 = state_weights["layer1.b"][0].float()
        w2 = state_weights["layer2.W"][0].float()
        b2 = state_weights["layer2.b"][0].float()

        torch.manual_seed(42)
        state_input = torch.randn(1, 1, 128)

        with torch.no_grad():
            h = torch.nn.functional.silu(torch.matmul(state_input, w1) + b1)
            ref_out = torch.matmul(h, w2) + b2

        # TTNN
        state_enc = CategorySpecificMLPTTNN(
            state_weights, emb_cfg.max_num_embodiments,
            emb_cfg.max_state_dim, emb_cfg.state_hidden_dim,
            emb_cfg.state_output_dim, tt_device,
        )
        tt_out = ttnn.to_torch(state_enc(to_tt_tensor(state_input, tt_device), 0))

        pcc = self.compute_pcc(ref_out, tt_out)
        logger.info(f"Embodiment PCC: {pcc:.6f}")
        assert pcc >= 0.99, f"Embodiment PCC {pcc:.4f} below 0.99 threshold"
