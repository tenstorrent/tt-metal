# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for trace cache key computation and output cloning fixes."""

import os
import pytest
import torch
import torch.nn as nn
import ttnn

from models.experimental.tt_symbiote.core.run_config import _compute_args_signature, TracedRun
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.linear import TTNNLinear


def _is_traced_mode():
    """Check if traced mode is active."""
    return os.environ.get("TT_SYMBIOTE_RUN_MODE") == "TRACED"


# =============================================================================
# Fix 1: Cache Key Tests (no device required)
# =============================================================================


def test_same_tensor_signature_produces_cache_hit():
    """Tensors with identical shape/dtype but different values should produce same signature."""
    tensor1 = torch.randn(2, 3, 4, dtype=torch.float32)
    tensor2 = torch.randn(2, 3, 4, dtype=torch.float32)  # Different values, same shape/dtype

    sig1 = _compute_args_signature((tensor1,))
    sig2 = _compute_args_signature((tensor2,))

    assert sig1 == sig2, f"Expected same signature for tensors with same shape/dtype, got {sig1} vs {sig2}"


def test_different_tensor_signature_produces_cache_miss():
    """Tensors with different shapes or dtypes should produce different signatures."""
    tensor_a = torch.randn(2, 3, 4, dtype=torch.float32)
    tensor_b = torch.randn(2, 3, 5, dtype=torch.float32)  # Different shape
    tensor_c = torch.randn(2, 3, 4, dtype=torch.bfloat16)  # Different dtype

    sig_a = _compute_args_signature((tensor_a,))
    sig_b = _compute_args_signature((tensor_b,))
    sig_c = _compute_args_signature((tensor_c,))

    assert sig_a != sig_b, f"Expected different signatures for different shapes, got {sig_a} vs {sig_b}"
    assert sig_a != sig_c, f"Expected different signatures for different dtypes, got {sig_a} vs {sig_c}"


def test_non_tensor_args_ignored_in_cache_key():
    """Non-tensor arguments should not affect the signature."""
    tensor = torch.randn(2, 3, dtype=torch.float32)

    # Different non-tensor args should produce same signature
    sig1 = _compute_args_signature((tensor, 42, "hello"))
    sig2 = _compute_args_signature((tensor, 100, "world"))
    sig3 = _compute_args_signature((tensor,))

    assert sig1 == sig2, f"Non-tensor args should not affect signature, got {sig1} vs {sig2}"
    assert sig1 == sig3, f"Additional non-tensor args should not affect signature, got {sig1} vs {sig3}"


def test_tensor_kwargs_included_in_signature():
    """Tensor kwargs should be included in the signature."""
    tensor1 = torch.randn(2, 3, dtype=torch.float32)
    tensor2 = torch.randn(4, 5, dtype=torch.float32)
    tensor3 = torch.randn(6, 7, dtype=torch.float32)

    sig1 = _compute_args_signature((), kwargs={"input": tensor1})
    sig2 = _compute_args_signature((), kwargs={"input": tensor2})  # Different shape in kwarg
    sig3 = _compute_args_signature((), kwargs={"other": tensor1})  # Different key name

    assert sig1 != sig2, f"Different tensor kwargs should produce different signatures, got {sig1} vs {sig2}"
    assert sig1 != sig3, f"Different kwarg names should produce different signatures, got {sig1} vs {sig3}"


# =============================================================================
# Fix 2: Clone Tests (device required)
# =============================================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_clone_trace_output_creates_independent_copy(device):
    """Cloning a TorchTTNNTensor should create an independent copy with different buffer."""
    # Create a torch tensor and convert to TTNN
    torch_tensor = torch.randn(1, 32, 64, dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor, device=device, layout=ttnn.TILE_LAYOUT)

    original = TorchTTNNTensor(ttnn_tensor, dtype=torch.bfloat16)
    cloned = TracedRun._clone_trace_output(original)

    # Verify different tensor objects
    assert cloned is not original, "Clone should be a different object"
    assert cloned.ttnn_tensor is not original.ttnn_tensor, "Clone should have different TTNN tensor"

    # Verify buffer addresses are different
    original_buffer = original.ttnn_tensor.buffer_address()
    cloned_buffer = cloned.ttnn_tensor.buffer_address()
    assert original_buffer != cloned_buffer, f"Buffer addresses should differ: {original_buffer} vs {cloned_buffer}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_clone_modification_does_not_affect_original(device):
    """Modifying a clone should not affect the original tensor."""
    # Create tensor with known values
    torch_tensor = torch.ones(1, 32, 64, dtype=torch.bfloat16) * 5.0
    ttnn_tensor = ttnn.from_torch(torch_tensor, device=device, layout=ttnn.TILE_LAYOUT)

    original = TorchTTNNTensor(ttnn_tensor, dtype=torch.bfloat16)
    cloned = TracedRun._clone_trace_output(original)

    # Modify the clone by adding to it
    modified_ttnn = ttnn.add(cloned.ttnn_tensor, 10.0)
    cloned_modified = TorchTTNNTensor(modified_ttnn, dtype=torch.bfloat16)

    # Convert back to torch to verify values
    original_torch = ttnn.to_torch(original.ttnn_tensor)
    cloned_modified_torch = ttnn.to_torch(cloned_modified.ttnn_tensor)

    # Original should still have value ~5.0, modified clone should have ~15.0
    assert torch.allclose(
        original_torch, torch.ones_like(original_torch) * 5.0, atol=0.1
    ), "Original tensor should be unchanged"
    assert torch.allclose(
        cloned_modified_torch, torch.ones_like(cloned_modified_torch) * 15.0, atol=0.1
    ), "Modified clone should have new values"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_clone_handles_nested_structures(device):
    """Clone should handle nested lists and dicts of TorchTTNNTensors."""
    # Create test tensors
    torch_tensor1 = torch.randn(1, 32, 64, dtype=torch.bfloat16)
    torch_tensor2 = torch.randn(1, 32, 64, dtype=torch.bfloat16)

    ttnn_tensor1 = ttnn.from_torch(torch_tensor1, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_tensor2 = ttnn.from_torch(torch_tensor2, device=device, layout=ttnn.TILE_LAYOUT)

    tensor1 = TorchTTNNTensor(ttnn_tensor1, dtype=torch.bfloat16)
    tensor2 = TorchTTNNTensor(ttnn_tensor2, dtype=torch.bfloat16)

    # Test list cloning
    original_list = [tensor1, tensor2]
    cloned_list = TracedRun._clone_trace_output(original_list)

    assert isinstance(cloned_list, list), "Cloned list should be a list"
    assert len(cloned_list) == 2, "Cloned list should have same length"
    assert cloned_list[0] is not tensor1, "List elements should be cloned"
    assert cloned_list[1] is not tensor2, "List elements should be cloned"
    assert cloned_list[0].ttnn_tensor.buffer_address() != tensor1.ttnn_tensor.buffer_address()
    assert cloned_list[1].ttnn_tensor.buffer_address() != tensor2.ttnn_tensor.buffer_address()

    # Test dict cloning
    original_dict = {"a": tensor1, "b": tensor2}
    cloned_dict = TracedRun._clone_trace_output(original_dict)

    assert isinstance(cloned_dict, dict), "Cloned dict should be a dict"
    assert set(cloned_dict.keys()) == {"a", "b"}, "Cloned dict should have same keys"
    assert cloned_dict["a"] is not tensor1, "Dict values should be cloned"
    assert cloned_dict["b"] is not tensor2, "Dict values should be cloned"
    assert cloned_dict["a"].ttnn_tensor.buffer_address() != tensor1.ttnn_tensor.buffer_address()
    assert cloned_dict["b"].ttnn_tensor.buffer_address() != tensor2.ttnn_tensor.buffer_address()


# =============================================================================
# Integration Tests: Full Trace Flow with TTNN Modules
# Requires: TT_SYMBIOTE_RUN_MODE=TRACED
# =============================================================================


@pytest.mark.skipif(not _is_traced_mode(), reason="Requires TT_SYMBIOTE_RUN_MODE=TRACED")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_traced_linear_captures_and_reuses_trace(device):
    """Verify trace is captured on first run and reused on subsequent runs."""
    TracedRun.release_all()
    assert TracedRun.cache_size() == 0, "Cache should be empty at start"

    # Create TTNNLinear module
    torch_linear = nn.Linear(64, 128, bias=True)
    ttnn_linear = TTNNLinear.from_torch(torch_linear)
    ttnn_linear._unique_name = "test_linear_capture"
    ttnn_linear.to_device(device)
    ttnn_linear.preprocess_weights()
    ttnn_linear.move_weights_to_device()

    # First run - should capture trace
    input1 = torch.randn(1, 32, 64, dtype=torch.bfloat16)
    output1 = ttnn_linear(input1)

    assert TracedRun.cache_size() == 1, "Trace should be captured on first run"

    # Second run with same shape - should reuse trace
    input2 = torch.randn(1, 32, 64, dtype=torch.bfloat16)
    output2 = ttnn_linear(input2)

    assert TracedRun.cache_size() == 1, "Trace should be reused (cache size unchanged)"

    TracedRun.release_all()


@pytest.mark.skipif(not _is_traced_mode(), reason="Requires TT_SYMBIOTE_RUN_MODE=TRACED")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_traced_linear_different_shapes_create_new_trace(device):
    """Verify different tensor shapes create separate cache entries."""
    TracedRun.release_all()

    # Create TTNNLinear with larger input to support multiple shapes
    torch_linear = nn.Linear(64, 128, bias=True)
    ttnn_linear = TTNNLinear.from_torch(torch_linear)
    ttnn_linear._unique_name = "test_linear_shapes"
    ttnn_linear.to_device(device)
    ttnn_linear.preprocess_weights()
    ttnn_linear.move_weights_to_device()

    # Run with shape A
    input_a = torch.randn(1, 32, 64, dtype=torch.bfloat16)
    _ = ttnn_linear(input_a)
    assert TracedRun.cache_size() == 1, "First shape should create one trace"

    # Run with shape B (different batch dimension)
    input_b = torch.randn(2, 32, 64, dtype=torch.bfloat16)
    _ = ttnn_linear(input_b)
    assert TracedRun.cache_size() == 2, "Different shape should create new trace"

    # Run with shape A again - should reuse first trace
    input_a2 = torch.randn(1, 32, 64, dtype=torch.bfloat16)
    _ = ttnn_linear(input_a2)
    assert TracedRun.cache_size() == 2, "Same shape should reuse existing trace"

    TracedRun.release_all()


@pytest.mark.skipif(not _is_traced_mode(), reason="Requires TT_SYMBIOTE_RUN_MODE=TRACED")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_traced_linear_outputs_are_independent(device):
    """Verify multiple trace executions return independent outputs (not aliased)."""
    TracedRun.release_all()

    torch_linear = nn.Linear(64, 128, bias=True)
    ttnn_linear = TTNNLinear.from_torch(torch_linear)
    ttnn_linear._unique_name = "test_linear_independent"
    ttnn_linear.to_device(device)
    ttnn_linear.preprocess_weights()
    ttnn_linear.move_weights_to_device()

    input_tensor = torch.randn(1, 32, 64, dtype=torch.bfloat16)

    # First run captures trace
    _ = ttnn_linear(input_tensor)

    # Second and third runs use cached trace
    output1 = ttnn_linear(input_tensor)
    output2 = ttnn_linear(input_tensor)

    # Verify outputs are different objects
    assert output1 is not output2, "Outputs should be different objects"

    # Verify buffer addresses are different (not aliased)
    if hasattr(output1, "ttnn_tensor") and output1.ttnn_tensor is not None:
        buf1 = output1.ttnn_tensor.buffer_address()
        buf2 = output2.ttnn_tensor.buffer_address()
        assert buf1 != buf2, f"Buffer addresses should differ: {buf1} vs {buf2}"

    TracedRun.release_all()
