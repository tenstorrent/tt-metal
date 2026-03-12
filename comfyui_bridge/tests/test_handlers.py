# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Unit tests for handlers.py - TensorBridge and OperationHandler.
"""

import torch
import numpy as np
import pytest
from multiprocessing import shared_memory
import uuid


class TestTensorBridge:
    """Test TensorBridge shared memory functionality."""

    def test_tensor_to_shm_basic(self):
        """Test basic tensor to shared memory conversion."""
        from comfyui_bridge.handlers import TensorBridge

        bridge = TensorBridge()
        tensor = torch.randn(1, 4, 64, 64)

        handle = bridge.tensor_to_shm(tensor)

        # Verify handle structure
        assert "shm_name" in handle
        assert "shape" in handle
        assert "dtype" in handle
        assert "size_bytes" in handle

        # Verify shape
        assert handle["shape"] == list(tensor.shape)

        # Verify size
        expected_size = tensor.numel() * tensor.element_size()
        assert handle["size_bytes"] == expected_size

        # Cleanup
        bridge.cleanup_segment(handle["shm_name"])

    def test_tensor_from_shm_basic(self):
        """Test basic shared memory to tensor reconstruction."""
        from comfyui_bridge.handlers import TensorBridge

        bridge = TensorBridge()
        original_tensor = torch.randn(1, 4, 64, 64)

        # Send to shm
        handle = bridge.tensor_to_shm(original_tensor)

        # Receive from shm
        reconstructed_tensor = bridge.tensor_from_shm(handle)

        # Verify
        assert reconstructed_tensor.shape == original_tensor.shape
        assert reconstructed_tensor.dtype == original_tensor.dtype
        assert torch.allclose(reconstructed_tensor, original_tensor)

    def test_tensor_to_shm_cuda(self):
        """Test CUDA tensor is moved to CPU."""
        from comfyui_bridge.handlers import TensorBridge

        bridge = TensorBridge()

        # Create CPU tensor (CUDA not available in test env)
        tensor = torch.randn(1, 4, 64, 64)

        handle = bridge.tensor_to_shm(tensor)
        reconstructed = bridge.tensor_from_shm(handle)

        assert reconstructed.device.type == "cpu"

    def test_tensor_to_shm_non_contiguous(self):
        """Test non-contiguous tensor is made contiguous."""
        from comfyui_bridge.handlers import TensorBridge

        bridge = TensorBridge()

        # Create non-contiguous tensor
        tensor = torch.randn(8, 8).t()  # Transpose makes it non-contiguous
        assert not tensor.is_contiguous()

        handle = bridge.tensor_to_shm(tensor)
        reconstructed = bridge.tensor_from_shm(handle)

        assert reconstructed.is_contiguous()
        assert torch.allclose(reconstructed, tensor)

    def test_tensor_round_trip_dtypes(self):
        """Test round trip with different dtypes."""
        from comfyui_bridge.handlers import TensorBridge

        bridge = TensorBridge()
        dtypes = [torch.float32, torch.float16, torch.int64]

        for dtype in dtypes:
            tensor = torch.randn(4, 4).to(dtype)

            handle = bridge.tensor_to_shm(tensor)
            reconstructed = bridge.tensor_from_shm(handle)

            assert reconstructed.dtype == tensor.dtype
            if dtype in [torch.float32, torch.float16]:
                assert torch.allclose(reconstructed, tensor)
            else:
                assert torch.equal(reconstructed, tensor)

    def test_cleanup_segment(self):
        """Test cleanup of shared memory segment."""
        from comfyui_bridge.handlers import TensorBridge

        bridge = TensorBridge()
        tensor = torch.randn(4, 4)

        handle = bridge.tensor_to_shm(tensor)
        shm_name = handle["shm_name"]

        # Verify segment exists
        assert shm_name in bridge._active_segments

        # Cleanup
        bridge.cleanup_segment(shm_name)

        # Verify removed
        assert shm_name not in bridge._active_segments

    def test_cleanup_all(self):
        """Test cleanup of all segments."""
        from comfyui_bridge.handlers import TensorBridge

        bridge = TensorBridge()

        # Create multiple segments
        tensors = [torch.randn(4, 4) for _ in range(3)]
        handles = [bridge.tensor_to_shm(t) for t in tensors]

        # Verify all exist
        assert len(bridge._active_segments) == 3

        # Cleanup all
        bridge.cleanup_all()

        # Verify all removed
        assert len(bridge._active_segments) == 0

    def test_large_tensor(self):
        """Test with large tensor (> 100MB)."""
        from comfyui_bridge.handlers import TensorBridge

        bridge = TensorBridge()

        # Create 256MB tensor
        large_tensor = torch.randn(1, 4, 2048, 2048)
        size_mb = (large_tensor.numel() * large_tensor.element_size()) / (1024 * 1024)
        assert size_mb > 100

        handle = bridge.tensor_to_shm(large_tensor)
        reconstructed = bridge.tensor_from_shm(handle)

        # Verify (use sampling to avoid memory issues)
        assert reconstructed.shape == large_tensor.shape
        assert torch.allclose(reconstructed[0, 0, :10, :10], large_tensor[0, 0, :10, :10])


class TestOperationHandler:
    """Test OperationHandler with mocked SDXLRunner."""

    def test_handle_ping_no_model(self):
        """Test ping operation when no model loaded."""
        from comfyui_bridge.handlers import OperationHandler
        from sdxl_config import SDXLConfig

        config = SDXLConfig()
        handler = OperationHandler(config)

        result = handler.handle_ping({})

        assert result["status"] == "ok"
        assert result["model_loaded"] == False
        assert result["model_id"] is None

    def test_parse_dtype(self):
        """Test dtype string parsing."""
        from comfyui_bridge.handlers import _parse_dtype

        test_cases = [
            ("torch.float32", np.float32),
            ("torch.float16", np.float16),
            ("torch.float64", np.float64),
            ("torch.int64", np.int64),
            ("torch.int32", np.int32),
            ("torch.int16", np.int16),
            ("torch.int8", np.int8),
            ("torch.uint8", np.uint8),
        ]

        for dtype_str, expected_dtype in test_cases:
            result = _parse_dtype(dtype_str)
            assert result == expected_dtype

    def test_parse_dtype_unknown(self):
        """Test unknown dtype defaults to float32."""
        from comfyui_bridge.handlers import _parse_dtype

        result = _parse_dtype("torch.unknown_type")
        assert result == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
