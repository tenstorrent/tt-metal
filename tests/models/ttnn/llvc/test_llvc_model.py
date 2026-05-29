"""
Unit tests for LLVC model forward and backward passes on TTNN devices.
Tests ensure correct tensor shapes, gradient flow, and device compatibility.
"""

import pytest
import torch
import ttnn
from ttnn import Tensor, Device, operations
from typing import Tuple, Optional


class TtnnLLVCModel(torch.nn.Module):
    """
    Minimal LLVC-like model using TTNN operations for testing forward/backward passes.
    Composed of a linear encoder, ReLU, and linear decoder.
    """

    def __init__(self, input_dim: int = 80, hidden_dim: int = 64, output_dim: int = 80):
        super().__init__()
        self.encoder = ttnn.linear(input_dim, hidden_dim, bias=True)
        self.decoder = ttnn.linear(hidden_dim, output_dim, bias=True)
        self.activation = ttnn.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode, activate, decode."""
        hidden = self.encoder(x)
        hidden = self.activation(hidden)
        out = self.decoder(hidden)
        return out


@pytest.fixture(params=[0], ids=["device0"])
def device(request) -> Device:
    """
    Fixture providing a single TTNN device for testing.
    Can be parametrized with multiple device IDs for multi-device tests.
    """
    device_id = request.param
    try:
        dev = ttnn.open_device(device_id=device_id)
        yield dev
    except RuntimeError as e:
        pytest.fail(f"Failed to open TTNN device {device_id}: {e}")
    finally:
        ttnn.close_device(dev)


@pytest.fixture
def sample_batch() -> torch.Tensor:
    """Fixture for a random batch of mel-spectrogram frames (1 x 80)."""
    return torch.randn(1, 80)


class TestLLVCModelForwardBackward:
    """Suite for forward and backward pass tests on TTNN devices."""

    def test_forward_pass_shape(self, device: Device, sample_batch: torch.Tensor):
        """
        Verify forward pass returns tensor with correct output shape.
        """
        model = TtnnLLVCModel()
        input_tensor = ttnn.from_torch(sample_batch, device=device, dtype=ttnn.bfloat16)
        output_tensor = model.forward(input_tensor)
        output_torch = ttnn.to_torch(output_tensor)
        assert output_torch.shape == sample_batch.shape, (
            f"Expected output shape {sample_batch.shape}, got {output_torch.shape}"
        )

    def test_forward_backward_gradients(self, device: Device, sample_batch: torch.Tensor):
        """
        Validate that gradients flow through the model and all parameters receive gradients.
        """
        model = TtnnLLVCModel()
        # Move model parameters to device (TTNN tensors)
        for param in model.parameters():
            if isinstance(param, ttnn.Tensor):
                param = param.to(device)

        input_tensor = ttnn.from_torch(sample_batch, device=device, dtype=ttnn.bfloat16)
        output = model.forward(input_tensor)

        # Compute a simple loss (MSE) and backprop
        target = ttnn.from_torch(torch.randn_like(sample_batch), device=device, dtype=ttnn.bfloat16)
        loss = ttnn.mse_loss(output, target)
        loss_tensor = ttnn.to_torch(loss).sum()
        loss_tensor.backward()

        # Check that parameters have gradients (non-None and finite)
        for name, param in model.named_parameters():
            if param.grad is None:
                pytest.fail(f"Parameter {name} has no gradient after backward pass.")
            if not torch.isfinite(param.grad).all():
                pytest.fail(f"Parameter {name} has non-finite gradient: {param.grad}")

    def test_device_consistency(self, device: Device, sample_batch: torch.Tensor):
        """
        Ensure all tensors reside on the same device after forward pass.
        """
        model = TtnnLLVCModel()
        input_tensor = ttnn.from_torch(sample_batch, device=device, dtype=ttnn.bfloat16)
        output = model.forward(input_tensor)
        # Check that output device matches input device
        assert output.device() == device, "Output tensor not on the expected device."
        # Check internal parameters are on the same device (if they are TTNN tensors)
        for param in model.parameters():
            if isinstance(param, ttnn.Tensor):
                assert param.device() == device, f"Parameter not on device: {param}"

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_varying_batch_size(self, device: Device, batch_size: int):
        """
        Test forward pass with different batch sizes to ensure dynamic shape handling.
        """
        model = TtnnLLVCModel()
        x = torch.randn(batch_size, 80)
        input_tensor = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16)
        output = model.forward(input_tensor)
        output_torch = ttnn.to_torch(output)
        assert output_torch.size(0) == batch_size, (
            f"Batch dimension mismatch: expected {batch_size}, got {output_torch.size(0)}"
        )

    def test_all_gather_support(self, device: Device, sample_batch: torch.Tensor):
        """
        Verify that the model can be used with ttnn.all_gather (mesh operation).
        Note: This test requires multi-device setup; skipped if only one device available.
        """
        try:
            num_devices = len(ttnn.get_device_ids())
        except AttributeError:
            pytest.skip("ttnn.get_device_ids() not available")
        if num_devices < 2:
            pytest.skip("Need at least 2 devices to test all_gather")

        # Prepare data on first device
        x = sample_batch
        input_tensor = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16)
        model = TtnnLLVCModel()
        # Move model to device
        for param in model.parameters():
            if isinstance(param, ttnn.Tensor):
                param = param.to(device)

        # Run all_gather on model output (simplified: just a single all_gather call)
        output = model.forward(input_tensor)
        gathered = ttnn.all_gather(output, dim=0)  # gather across devices
        assert gathered is not None, "all_gather returned None"
        assert gathered.shape[0] == x.shape[0] * num_devices, (
            f"Expected first dim {x.shape[0]*num_devices}, got {gathered.shape[0]}"
        )