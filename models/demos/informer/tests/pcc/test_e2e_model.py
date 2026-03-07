# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC tests for Informer model."""

import pytest
import torch

from models.demos.informer.reference.torch_reference import (
    TorchInformerModel,
    compute_metrics,
    informer_torch_forward,
    ttnn_state_dict,
)
from models.demos.informer.tt.config import InformerConfig
from models.demos.informer.tt.model import InformerModel
from models.demos.informer.tt.ops import to_torch


class TestInformerE2E:
    """End-to-end Informer model tests."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize(
        "seq_len,label_len,pred_len",
        [
            (96, 48, 24),
            (64, 32, 16),
        ],
    )
    def test_informer_output_shape(self, device, batch_size, seq_len, label_len, pred_len):
        """Test Informer produces correct output shape."""
        config = InformerConfig(
            enc_in=7,
            dec_in=7,
            c_out=7,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=64,
            n_heads=2,
            d_ff=256,
            e_layers=2,
            d_layers=1,
            time_feature_dim=4,
            dtype="bfloat16",
        )

        model = InformerModel(config, device=device, seed=42)

        # Inputs
        past_values = torch.randn(batch_size, seq_len, config.enc_in, dtype=torch.float32)
        past_time = torch.randn(batch_size, seq_len, config.time_feature_dim, dtype=torch.float32)
        future_time = torch.randn(batch_size, pred_len, config.time_feature_dim, dtype=torch.float32)

        # Forward pass
        out = model(past_values, past_time, future_time)
        out_torch = to_torch(out)

        # Check shape
        expected_shape = (batch_size, pred_len, config.c_out)
        assert out_torch.shape == expected_shape, f"Expected {expected_shape}, got {out_torch.shape}"
        assert torch.isfinite(out_torch).all(), "Output contains non-finite values"
        ref_out = informer_torch_forward(model, past_values, past_time, future_time)
        mse, mae, corr = compute_metrics(out_torch.float(), ref_out.float())
        assert mse < 1e-2, f"Output-shape test MSE {mse:.6f} too high"
        assert mae < 1e-1, f"Output-shape test MAE {mae:.6f} too high"
        assert corr > 0.90, f"Output-shape test PCC {corr:.4f} too low"

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_informer_vs_torch_reference(self, device, batch_size):
        """Test Informer output matches PyTorch reference."""
        config = InformerConfig(
            enc_in=7,
            dec_in=7,
            c_out=7,
            seq_len=96,
            label_len=48,
            pred_len=24,
            d_model=64,
            n_heads=2,
            d_ff=256,
            e_layers=2,
            d_layers=1,
            time_feature_dim=4,
            dtype="bfloat16",
            attention_type="prob",
        )

        model = InformerModel(config, device=device, seed=42)

        # Inputs
        past_values = torch.randn(batch_size, config.seq_len, config.enc_in, dtype=torch.float32)
        past_time = torch.randn(batch_size, config.seq_len, config.time_feature_dim, dtype=torch.float32)
        future_time = torch.randn(batch_size, config.pred_len, config.time_feature_dim, dtype=torch.float32)

        # TTNN forward
        ttnn_out = model(past_values, past_time, future_time)
        ttnn_out_torch = to_torch(ttnn_out)

        # PyTorch reference forward
        torch_out = informer_torch_forward(model, past_values, past_time, future_time)

        # Compare
        mse, mae, pcc = compute_metrics(ttnn_out_torch, torch_out)

        print(f"E2E PCC: {pcc:.4f}, MSE: {mse:.6f}, MAE: {mae:.6f}")

        # Targets
        assert pcc > 0.90, f"E2E PCC {pcc:.4f} < 0.90"

    def test_state_dict_roundtrip(self, device, torch_seed):
        """Test loading a torch reference state dict into TTNN model."""
        torch.manual_seed(torch_seed)
        cfg = InformerConfig(
            enc_in=4,
            dec_in=4,
            c_out=4,
            seq_len=32,
            label_len=16,
            pred_len=8,
            d_model=64,
            n_heads=2,
            d_ff=128,
            time_feature_dim=4,
            dtype="bfloat16",
        )
        torch_model = TorchInformerModel(cfg)
        torch_model.eval()
        state = ttnn_state_dict(torch_model)

        ttnn_model = InformerModel(cfg, device=device, seed=0)
        ttnn_model.load_state_dict(state, strict=True)

        batch = 2
        past_values = torch.randn(batch, cfg.seq_len, cfg.enc_in, dtype=torch.float32)
        past_time = torch.randn(batch, cfg.seq_len, cfg.time_feature_dim, dtype=torch.float32)
        future_time = torch.randn(batch, cfg.pred_len, cfg.time_feature_dim, dtype=torch.float32)

        with torch.no_grad():
            torch_out = torch_model(past_values, past_time, future_time)
        ttnn_out = to_torch(ttnn_model(past_values, past_time, future_time)).float()

        assert ttnn_out.shape == torch_out.shape
        assert torch.isfinite(ttnn_out).all()

        mse, mae, corr = compute_metrics(ttnn_out, torch_out)
        assert mse < 1e-2, f"MSE {mse:.6f} too high"
        assert mae < 5e-2, f"MAE {mae:.6f} too high"
        assert corr > 0.95, f"Correlation {corr:.4f} too low"

    @pytest.mark.parametrize("e_layers", [1, 2, 3])
    @pytest.mark.parametrize("d_layers", [1, 2])
    def test_informer_different_depths(self, device, e_layers, d_layers):
        """Test Informer with different encoder/decoder depths."""
        config = InformerConfig(
            enc_in=7,
            dec_in=7,
            c_out=7,
            seq_len=64,
            label_len=32,
            pred_len=16,
            d_model=64,
            n_heads=2,
            d_ff=256,
            e_layers=e_layers,
            d_layers=d_layers,
            time_feature_dim=4,
            dtype="bfloat16",
        )

        model = InformerModel(config, device=device, seed=42)

        # Inputs
        batch_size = 2
        past_values = torch.randn(batch_size, config.seq_len, config.enc_in, dtype=torch.float32)
        past_time = torch.randn(batch_size, config.seq_len, config.time_feature_dim, dtype=torch.float32)
        future_time = torch.randn(batch_size, config.pred_len, config.time_feature_dim, dtype=torch.float32)

        # Forward pass
        out = model(past_values, past_time, future_time)
        out_torch = to_torch(out)

        assert out_torch.shape == (batch_size, config.pred_len, config.c_out)
        assert torch.isfinite(out_torch).all()
        ref_out = informer_torch_forward(model, past_values, past_time, future_time)
        mse, mae, corr = compute_metrics(out_torch.float(), ref_out.float())
        assert mse < 2e-2, f"Depth test MSE {mse:.6f} too high"
        assert mae < 1.5e-1, f"Depth test MAE {mae:.6f} too high"
        assert corr > 0.85, f"Depth test PCC {corr:.4f} too low"


class TestInformerMultivariate:
    """Test multivariate forecasting capabilities."""

    @pytest.mark.parametrize("features", [1, 4, 7, 21])
    def test_different_feature_counts(self, device, features):
        """Test Informer with different numbers of features (univariate to multivariate)."""
        config = InformerConfig(
            enc_in=features,
            dec_in=features,
            c_out=features,
            seq_len=64,
            label_len=32,
            pred_len=24,
            d_model=64,
            n_heads=2,
            d_ff=256,
            e_layers=2,
            d_layers=1,
            time_feature_dim=4,
            dtype="bfloat16",
        )

        model = InformerModel(config, device=device, seed=42)

        # Inputs
        batch_size = 2
        past_values = torch.randn(batch_size, config.seq_len, features, dtype=torch.float32)
        past_time = torch.randn(batch_size, config.seq_len, config.time_feature_dim, dtype=torch.float32)
        future_time = torch.randn(batch_size, config.pred_len, config.time_feature_dim, dtype=torch.float32)

        # Forward pass
        out = model(past_values, past_time, future_time)
        out_torch = to_torch(out)

        assert out_torch.shape == (batch_size, config.pred_len, features)
        assert torch.isfinite(out_torch).all()
        ref_out = informer_torch_forward(model, past_values, past_time, future_time)
        mse, mae, corr = compute_metrics(out_torch.float(), ref_out.float())
        assert mse < 2e-2, f"Feature-count test MSE {mse:.6f} too high"
        assert mae < 1.5e-1, f"Feature-count test MAE {mae:.6f} too high"
        assert corr > 0.85, f"Feature-count test PCC {corr:.4f} too low"


class TestInformerPredictionHorizons:
    """Test different prediction horizons."""

    @pytest.mark.parametrize("pred_len", [24, 48, 96, 168, 336])
    def test_prediction_horizons(self, device, pred_len):
        """Test Informer with different prediction horizons."""
        config = InformerConfig(
            enc_in=7,
            dec_in=7,
            c_out=7,
            seq_len=96,
            label_len=48,
            pred_len=pred_len,
            d_model=64,
            n_heads=2,
            d_ff=256,
            e_layers=2,
            d_layers=1,
            time_feature_dim=4,
            dtype="bfloat16",
        )

        model = InformerModel(config, device=device, seed=42)

        # Inputs
        batch_size = 2
        past_values = torch.randn(batch_size, config.seq_len, config.enc_in, dtype=torch.float32)
        past_time = torch.randn(batch_size, config.seq_len, config.time_feature_dim, dtype=torch.float32)
        future_time = torch.randn(batch_size, pred_len, config.time_feature_dim, dtype=torch.float32)

        # Forward pass
        out = model(past_values, past_time, future_time)
        out_torch = to_torch(out)

        assert out_torch.shape == (batch_size, pred_len, config.c_out)
        assert torch.isfinite(out_torch).all()
        ref_out = informer_torch_forward(model, past_values, past_time, future_time)
        mse, mae, corr = compute_metrics(out_torch.float(), ref_out.float())
        assert mse < 2e-2, f"Horizon test MSE {mse:.6f} too high"
        assert mae < 1.5e-1, f"Horizon test MAE {mae:.6f} too high"
        assert corr > 0.85, f"Horizon test PCC {corr:.4f} too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
