# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for Molmo2 Image Pooling (Cross-Attention).

Validates ImagePooling against PyTorch reference implementation.
"""


import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_pcc


def get_pooling_weights(model_id: str = "allenai/Molmo2-8B"):
    """Load pooling weights from HuggingFace."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    prefix = "model.vision_backbone.image_pooling_2d"
    keys = [
        f"{prefix}.wq.weight",
        f"{prefix}.wq.bias",
        f"{prefix}.wk.weight",
        f"{prefix}.wk.bias",
        f"{prefix}.wv.weight",
        f"{prefix}.wv.bias",
        f"{prefix}.wo.weight",
        f"{prefix}.wo.bias",
    ]
    return load_state_dict_from_safetensors(model_id, keys)


class ReferenceCrossAttention(nn.Module):
    """PyTorch reference implementation of cross-attention pooling."""

    def __init__(self, input_dim=2304, hidden_dim=1152, num_heads=16, head_dim=72):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.wq = nn.Linear(input_dim, num_heads * head_dim, bias=True)
        self.wk = nn.Linear(input_dim, num_heads * head_dim, bias=True)
        self.wv = nn.Linear(input_dim, num_heads * head_dim, bias=True)
        self.wo = nn.Linear(num_heads * head_dim, hidden_dim, bias=True)

    def forward(self, query, key_value, attn_mask=None):
        batch_size, num_queries, _ = query.shape
        _, pool_size, _ = key_value.shape

        # Project Q, K, V
        q = self.wq(query)
        k = self.wk(key_value)
        v = self.wv(key_value)

        # Reshape for multi-head attention
        q = q.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, pool_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, pool_size, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, num_queries, -1)
        out = self.wo(out)

        return out


@pytest.mark.parametrize("pool_size", [9, 16, 25])
def test_image_pooling(pool_size, device):
    """
    Test ImagePooling against PyTorch reference.

    Args:
        pool_size: Size of the pooling window (K_pool)
        device: TTNN device fixture
    """
    from models.demos.molmo2.tt.image_pooling import ImagePooling

    model_id = "allenai/Molmo2-8B"
    input_dim = 2304
    hidden_dim = 1152
    num_heads = 16
    head_dim = 72
    num_queries = 64  # Number of output tokens

    # Load weights
    state_dict = get_pooling_weights(model_id)

    # Create reference model
    ref_model = ReferenceCrossAttention(input_dim, hidden_dim, num_heads, head_dim)
    prefix = "model.vision_backbone.image_pooling_2d"
    ref_model.wq.weight.data = state_dict[f"{prefix}.wq.weight"]
    ref_model.wq.bias.data = state_dict[f"{prefix}.wq.bias"]
    ref_model.wk.weight.data = state_dict[f"{prefix}.wk.weight"]
    ref_model.wk.bias.data = state_dict[f"{prefix}.wk.bias"]
    ref_model.wv.weight.data = state_dict[f"{prefix}.wv.weight"]
    ref_model.wv.bias.data = state_dict[f"{prefix}.wv.bias"]
    ref_model.wo.weight.data = state_dict[f"{prefix}.wo.weight"]
    ref_model.wo.bias.data = state_dict[f"{prefix}.wo.bias"]
    ref_model.eval()

    # Create random input
    torch.manual_seed(42)
    # Query: mean of gathered features [batch, num_queries, input_dim]
    query_torch = torch.randn(1, num_queries, input_dim, dtype=torch.float32)
    # Key/Value: gathered features [batch, pool_size, input_dim]
    kv_torch = torch.randn(1, pool_size, input_dim, dtype=torch.float32)

    # Reference forward (no mask for this test)
    with torch.no_grad():
        ref_output = ref_model(query_torch, kv_torch)

    # Create TTNN pooling
    tt_pooling = ImagePooling(
        mesh_device=device,
        state_dict=state_dict,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=ttnn.bfloat8_b,
    )

    # Convert inputs to TTNN
    query_ttnn = ttnn.from_torch(
        query_torch.unsqueeze(0),  # [1, 1, num_queries, input_dim]
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    kv_ttnn = ttnn.from_torch(
        kv_torch.unsqueeze(0),  # [1, 1, pool_size, input_dim]
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # TTNN forward
    tt_output = tt_pooling(query_ttnn, kv_ttnn)

    # Convert back to torch
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0)

    # Compare with PCC
    passing, pcc_msg = comp_pcc(ref_output, tt_output_torch, pcc=0.99)
    print(f"ImagePooling (pool_size={pool_size}) PCC: {pcc_msg}")

    assert passing, f"ImagePooling failed PCC check: {pcc_msg}"


@pytest.mark.parametrize("use_mask", [False, True])
def test_image_pooling_with_mask(use_mask, device):
    """
    Test ImagePooling with attention mask.

    Args:
        use_mask: Whether to use attention mask
        device: TTNN device fixture
    """
    from models.demos.molmo2.tt.image_pooling import ImagePooling

    model_id = "allenai/Molmo2-8B"
    input_dim = 2304
    hidden_dim = 1152
    num_heads = 16
    head_dim = 72
    num_queries = 32
    pool_size = 16

    # Load weights
    state_dict = get_pooling_weights(model_id)

    # Create reference model
    ref_model = ReferenceCrossAttention(input_dim, hidden_dim, num_heads, head_dim)
    prefix = "model.vision_backbone.image_pooling_2d"
    ref_model.wq.weight.data = state_dict[f"{prefix}.wq.weight"]
    ref_model.wq.bias.data = state_dict[f"{prefix}.wq.bias"]
    ref_model.wk.weight.data = state_dict[f"{prefix}.wk.weight"]
    ref_model.wk.bias.data = state_dict[f"{prefix}.wk.bias"]
    ref_model.wv.weight.data = state_dict[f"{prefix}.wv.weight"]
    ref_model.wv.bias.data = state_dict[f"{prefix}.wv.bias"]
    ref_model.wo.weight.data = state_dict[f"{prefix}.wo.weight"]
    ref_model.wo.bias.data = state_dict[f"{prefix}.wo.bias"]
    ref_model.eval()

    # Create random input
    torch.manual_seed(42)
    query_torch = torch.randn(1, num_queries, input_dim, dtype=torch.float32)
    kv_torch = torch.randn(1, pool_size, input_dim, dtype=torch.float32)

    # Create mask if needed
    if use_mask:
        # Mask out some positions (simulate invalid patches)
        valid = torch.ones(1, 1, 1, pool_size)
        valid[0, 0, 0, pool_size // 2 :] = 0  # Mask second half
        attn_mask = torch.where(valid == 0, float("-inf"), 0.0)
    else:
        attn_mask = None

    # Reference forward
    with torch.no_grad():
        ref_output = ref_model(query_torch, kv_torch, attn_mask)

    # Create TTNN pooling
    tt_pooling = ImagePooling(
        mesh_device=device,
        state_dict=state_dict,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=ttnn.bfloat8_b,
    )

    # Convert inputs to TTNN
    query_ttnn = ttnn.from_torch(
        query_torch.unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    kv_ttnn = ttnn.from_torch(
        kv_torch.unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if attn_mask is not None:
        attn_mask_ttnn = ttnn.from_torch(
            attn_mask,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        attn_mask_ttnn = None

    # TTNN forward
    tt_output = tt_pooling(query_ttnn, kv_ttnn, attn_mask_ttnn)

    # Convert back to torch
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0)

    # Compare with PCC
    passing, pcc_msg = comp_pcc(ref_output, tt_output_torch, pcc=0.99)
    print(f"ImagePooling (use_mask={use_mask}) PCC: {pcc_msg}")

    assert passing, f"ImagePooling with mask failed PCC check: {pcc_msg}"


if __name__ == "__main__":
    # Quick standalone test
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        test_image_pooling(16, device)
        print("Test passed!")
    finally:
        ttnn.close_device(device)
