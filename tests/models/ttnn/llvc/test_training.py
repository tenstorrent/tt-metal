# tests/models/ttnn/llvc/test_training.py

import pytest
import torch
import ttnn
import numpy as np
from models.ttnn.llvc import LLVC  # assuming model import path
from ttnn.device import DeviceContext

# Mark as integration test requiring TT device
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not ttnn.has_tt_device(),
        reason="TTNN device not available"
    ),
]


def create_synthetic_data(batch_size: int = 2, seq_len: int = 160, feat_dim: int = 80):
    """
    Generate a synthetic batch of mel-spectrogram data for training.
    Args:
        batch_size: Number of samples in batch.
        seq_len: Number of time frames.
        feat_dim: Number of mel bands.
    Returns:
        input_tensor: (batch_size, feat_dim, seq_len) torch tensor on CPU.
        target_tensor: (batch_size, feat_dim, seq_len) torch tensor on CPU.
    """
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, feat_dim, seq_len)
    # Target: slightly shifted version of input to simulate conversion target
    target_tensor = torch.roll(input_tensor, shifts=1, dims=-1)
    target_tensor[:, :, 0] = 0.0
    return input_tensor, target_tensor


def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple L1 loss for training verification."""
    return torch.nn.functional.l1_loss(pred, target)


class TestLLVCTraining:
    """Integration tests for LLVC training loop."""

    @pytest.fixture(scope="class")
    def device(self):
        """Provide a TTNN device context for the test class."""
        with DeviceContext(device_id=0) as dev:
            yield dev

    @pytest.fixture(scope="class")
    def model(self, device):
        """Instantiate a small LLVC model on device."""
        # Use a minimal configuration for testing
        config = {
            "input_dim": 80,
            "hidden_dim": 64,
            "num_layers": 2,
            "output_dim": 80,
        }
        model = LLVC(config)
        model.to(device)
        model.train()
        return model

    def test_training_loss_decrease(self, device, model):
        """
        Verify that after a few training steps the loss decreases.
        Runs a minimal training loop with synthetic data.
        """
        # Hyperparameters
        batch_size = 2
        seq_len = 160
        feat_dim = 80
        num_steps = 5
        learning_rate = 1e-3

        # Dataloader-like iteration
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Loss history for comparison
        losses = []

        for step in range(num_steps):
            # Generate synthetic data on CPU
            input_cpu, target_cpu = create_synthetic_data(batch_size, seq_len, feat_dim)

            # Move to TTNN device
            input_tt = ttnn.from_torch(input_cpu, device=device, dtype=ttnn.bfloat16)
            target_tt = ttnn.from_torch(target_cpu, device=device, dtype=ttnn.bfloat16)

            # Forward pass
            optimizer.zero_grad()
            output_tt = model(input_tt)

            # Move output back to CPU for loss calculation
            output_cpu = ttnn.to_torch(output_tt)
            loss = compute_loss(output_cpu, target_cpu)

            # Backward pass (if model parameters are differentiable through TTNN ops)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Assert that the final loss is lower than the initial loss
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )

        # Optional: print summary for debugging
        print(f"Training loss progression: {[f'{l:.4f}' for l in losses]}")

    def test_training_runs_without_error(self, device, model):
        """
        Minimal sanity - training loop completes without exceptions
        even if loss does not decrease (e.g., due to optimizer settings).
        """
        batch_size = 1
        seq_len = 80
        feat_dim = 80
        input_cpu, target_cpu = create_synthetic_data(batch_size, seq_len, feat_dim)
        input_tt = ttnn.from_torch(input_cpu, device=device, dtype=ttnn.bfloat16)
        target_tt = ttnn.from_torch(target_cpu, device=device, dtype=ttnn.bfloat16)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        output_tt = model(input_tt)
        output_cpu = ttnn.to_torch(output_tt)
        loss = compute_loss(output_cpu, target_cpu)
        loss.backward()
        optimizer.step()
        # If we get here without error, test passes
        assert True