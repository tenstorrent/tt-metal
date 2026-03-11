# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Qwen3.5 Linear Attention (GatedDeltaNet) implementation.

Test coverage:
- Forward pass (prefill mode) with various seq_len and batch_size combinations
- Decode mode with cached state
- Edge cases: seq_len < kernel_size, seq_len = CHUNK_SIZE, seq_len > CHUNK_SIZE
"""

import pytest
import torch
from loguru import logger
from transformers import AutoModelForCausalLM

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen35.tt.linear_attention import TtLinearAttention
from models.tt_transformers.tt.load_checkpoints import standardize_hf_keys_multimodal


def get_reference_model(model_name="Qwen/Qwen3.5-9B"):
    """Load HuggingFace reference model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    return model


@pytest.mark.parametrize(
    "layer_idx,seq_len,batch_size",
    [
        (1, 32, 1),  # Standard case: Layer 1, smaller seq for memory
        (1, 2, 1),  # Edge case: seq_len < conv_kernel_size (tests padding logic)
        (1, 64, 1),  # Edge case: seq_len = CHUNK_SIZE (no padding needed)
        (1, 65, 1),  # Edge case: seq_len > CHUNK_SIZE (tests large padding)
        (1, 32, 2),  # Multi-batch: tests batched operations
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_linear_attention_forward(layer_idx, seq_len, batch_size, mesh_device):
    """Test full forward pass against HuggingFace reference."""

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Load reference model
    ref_model = get_reference_model()
    ref_linear_attn = ref_model.model.layers[layer_idx].linear_attn

    # Get model config from reference
    from models.demos.qwen35.tt.model_config import Qwen35ModelArgs

    model_args = Qwen35ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
        dummy_weights=False,
    )

    # Verify layer is linear attention
    assert model_args.is_linear_attention_layer(layer_idx), f"Layer {layer_idx} is not linear attention"

    # Create TT model (state_dict passed to constructor, matching tt_transformers pattern)
    state_dict = standardize_hf_keys_multimodal(ref_model.state_dict())
    tt_model = TtLinearAttention(mesh_device, state_dict, model_args, layer_idx)

    # Create test input
    hidden_size = model_args.dim
    x_pt = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Run reference
    with torch.no_grad():
        ref_output = ref_linear_attn(x_pt)

    # Run TT model
    x_tt = ttnn.from_torch(
        x_pt,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_output = tt_model(x_tt)
    tt_output_torch = ttnn.to_torch(tt_output)

    # Compare outputs
    # Use 0.97 threshold - complex attention with many ops has accumulated bfloat16 precision loss,
    # especially with the decay-weighted state updates for proper decode mode support
    passing, pcc = comp_pcc(ref_output, tt_output_torch, pcc=0.97)
    logger.info(f"Forward pass PCC: {pcc} (seq_len={seq_len}, batch_size={batch_size})")

    assert passing, f"Forward pass PCC {pcc} below threshold"
    logger.info("✓ Forward pass matches reference")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_linear_attention_decode_mode(mesh_device):
    """
    Test decode mode (seq_len=1) with cached state.

    This tests the _recurrent_gated_delta_rule path and cache state management.
    """
    # Set seed for reproducibility
    torch.manual_seed(42)

    layer_idx = 1
    prefill_seq_len = 32
    batch_size = 1

    # Load reference model
    ref_model = get_reference_model()
    ref_linear_attn = ref_model.model.layers[layer_idx].linear_attn

    # Get model config
    from models.demos.qwen35.tt.model_config import Qwen35ModelArgs

    model_args = Qwen35ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=prefill_seq_len + 10,  # Extra for decode tokens
        dummy_weights=False,
    )

    # Create TT model (state_dict passed to constructor, matching tt_transformers pattern)
    state_dict = standardize_hf_keys_multimodal(ref_model.state_dict())
    tt_model = TtLinearAttention(mesh_device, state_dict, model_args, layer_idx)

    hidden_size = model_args.dim

    # === Phase 1: Prefill (build up cache state) ===
    x_prefill_pt = torch.randn(batch_size, prefill_seq_len, hidden_size, dtype=torch.bfloat16)

    # Run reference prefill
    with torch.no_grad():
        ref_prefill_output = ref_linear_attn(x_prefill_pt)

    # Run TT prefill with cache_params to capture state
    cache_params = {
        "conv_state": None,
        "recurrent_state": None,
        "has_previous_state": False,
    }

    x_prefill_tt = ttnn.from_torch(
        x_prefill_pt,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_prefill_output = tt_model(x_prefill_tt, cache_params=cache_params)
    tt_prefill_torch = ttnn.to_torch(tt_prefill_output)

    # Verify prefill matches
    passing, pcc = comp_pcc(ref_prefill_output, tt_prefill_torch, pcc=0.97)
    logger.info(f"Prefill PCC: {pcc}")
    assert passing, f"Prefill PCC {pcc} below threshold"

    # Verify cache was populated
    assert cache_params["has_previous_state"], "Cache should be marked as having previous state"
    assert cache_params["conv_state"] is not None, "Conv state should be populated"
    assert cache_params["recurrent_state"] is not None, "Recurrent state should be populated"

    # === Phase 2: Decode (single token with cache) ===
    x_decode_pt = torch.randn(batch_size, 1, hidden_size, dtype=torch.bfloat16)

    # For reference, we need to run the full sequence (prefill + decode token)
    # since HF reference doesn't have native cache support in the same way
    x_full_pt = torch.cat([x_prefill_pt, x_decode_pt], dim=1)
    with torch.no_grad():
        ref_full_output = ref_linear_attn(x_full_pt)
    # Take only the last token's output
    ref_decode_output = ref_full_output[:, -1:, :]

    # Run TT decode with cached state
    x_decode_tt = ttnn.from_torch(
        x_decode_pt,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_decode_output = tt_model(x_decode_tt, cache_params=cache_params)
    tt_decode_torch = ttnn.to_torch(tt_decode_output)

    # Compare decode outputs
    # Note: Decode mode may have slightly lower PCC due to state approximation
    passing, pcc = comp_pcc(ref_decode_output, tt_decode_torch, pcc=0.95)
    logger.info(f"Decode PCC: {pcc}")

    assert passing, f"Decode PCC {pcc} below threshold"
    logger.info("✓ Decode mode matches reference")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_linear_attention_conv_state_none_decode(mesh_device):
    """
    Test decode mode edge case: conv_state=None but has_previous_state=True.

    This can happen on first decode call if cache_params is provided but states
    aren't initialized. The code should handle this gracefully.
    """
    torch.manual_seed(42)

    layer_idx = 1
    batch_size = 1

    # Load reference model
    ref_model = get_reference_model()

    # Get model config
    from models.demos.qwen35.tt.model_config import Qwen35ModelArgs

    model_args = Qwen35ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=32,
        dummy_weights=False,
    )

    # Create TT model (state_dict passed to constructor, matching tt_transformers pattern)
    state_dict = standardize_hf_keys_multimodal(ref_model.state_dict())
    tt_model = TtLinearAttention(mesh_device, state_dict, model_args, layer_idx)

    hidden_size = model_args.dim

    # Create decode input (seq_len=1) with incomplete cache
    x_decode_pt = torch.randn(batch_size, 1, hidden_size, dtype=torch.bfloat16)

    # Edge case: has_previous_state=True but conv_state=None
    # This tests the fallback padding path in _causal_conv1d
    cache_params = {
        "conv_state": None,  # Intentionally None
        "recurrent_state": None,
        "has_previous_state": True,  # But marked as having state
    }

    x_decode_tt = ttnn.from_torch(
        x_decode_pt,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Should not crash - code should handle None conv_state gracefully
    tt_output = tt_model(x_decode_tt, cache_params=cache_params)
    tt_output_torch = ttnn.to_torch(tt_output)

    # Just verify output shape is correct (no reference comparison for this edge case)
    expected_shape = (batch_size, 1, hidden_size)
    assert tt_output_torch.shape == expected_shape, f"Output shape {tt_output_torch.shape} != expected {expected_shape}"
    logger.info("✓ Gracefully handled conv_state=None in decode mode")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
