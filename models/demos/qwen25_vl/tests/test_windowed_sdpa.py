#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest test cases for the windowed scaled dot product attention operation.
This demonstrates how cu_window_seqlens can be used to create block-diagonal
attention patterns without explicitly passing an attention mask.
"""

import os
import time

import pytest
import torch

import ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 26015744, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,head_dim,qk_chunk_size,cu_window_seqlens",
    [
        # basic cases
        (1, 16, 32, 96, 32, [0, 4, 20, 32]),
        (1, 16, 32, 96, 32, [0, 4, 9, 24, 32]),
        (1, 16, 32, 96, 32, [0, 16, 20]),
        (1, 16, 32, 96, 32, [0, 16, 20, 24, 28, 32]),
        # medium cases
        (1, 1, 64, 32, 32, [0, 16, 32, 48, 60]),
        (1, 1, 64, 32, 32, [0, 16, 48, 64]),
        (1, 1, 64, 32, 64, [0, 16, 32, 48, 60]),  # todo)) this case is failing
    ],
    ids=["basic-1", "basic-2", "basic-3", "basic-4", "medium-1", "medium-2", "medium-3"],
)
def test_windowed_sdpa_basic(mesh_device, batch_size, num_heads, seq_len, head_dim, qk_chunk_size, cu_window_seqlens):
    """Test windowed scaled dot product attention against standard SDPA with equivalent mask."""
    # Get windows from the provided function
    # cu_window_seqlens = get_cu_window_seqlens()
    if callable(cu_window_seqlens):
        pt_cu_window_seqlens = cu_window_seqlens()
    else:
        assert isinstance(cu_window_seqlens, list), "cu_window_seqlens must be a callable or a torch.Tensor"
        pt_cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.uint8)

    cu_window_seqlens_tt = ttnn.from_torch(
        pt_cu_window_seqlens, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
    )

    # Create input tensors
    torch.manual_seed(42)  # For reproducible results
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # Convert to TTNN tensors
    q_tt = ttnn.from_torch(q, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    k_tt = ttnn.from_torch(k, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    v_tt = ttnn.from_torch(v, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    # Run standard SDPA with explicit mask for comparison
    attention_mask = create_windowed_attention_mask(seq_len, cu_window_seqlens)
    # # print each value in attention_mask using nested for loops with aligned spacing
    # for i in range(attention_mask.shape[0]):
    #     for j in range(attention_mask.shape[1]):
    #         value = attention_mask[i, j].item()
    #         if value == float("-inf"):
    #             print("  -inf", end=" ")
    #         else:
    #             print(f"{value:6.1f}", end=" ")
    #     print()
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
    attention_mask_tt = ttnn.from_torch(
        attention_mask, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b
    )

    print("-------------------- Running standard SDPA --------------------")
    output_standard_tt = ttnn.transformer.scaled_dot_product_attention(
        q_tt,
        k_tt,
        v_tt,
        attn_mask=attention_mask_tt,
        is_causal=False,
        scale=0.1,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=qk_chunk_size,
            k_chunk_size=qk_chunk_size,
        ),
    )
    output_standard = ttnn.to_torch(output_standard_tt)

    # sleep for 1 seconds
    time.sleep(1)

    # Run windowed SDPA
    print("-------------------- Running windowed SDPA --------------------")
    output_tt = ttnn.transformer.windowed_scaled_dot_product_attention(
        q_tt,
        k_tt,
        v_tt,
        cu_window_seqlens_tt,
        scale=0.1,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=qk_chunk_size,
            k_chunk_size=qk_chunk_size,
        ),
    )

    # Convert back to torch for verification
    # NOTE: there is an implicit synchronization during the to_torch call, so the call getting here is not a guarantee that the computation above is complete.
    output = ttnn.to_torch(output_tt)

    # sleep for 1 seconds
    time.sleep(1)

    print("-------------------- Comparing outputs --------------------")

    # Compare outputs
    print(f"Windowed SDPA output shape: {output.shape}")
    print(f"Standard SDPA output shape: {output_standard.shape}")
    print(f"Max difference: {torch.max(torch.abs(output - output_standard)).item()}")

    # Print first few values for comparison (batch=0, head=0)
    compare_outputs(output, output_standard, seq_end=16, head_end=16)
    compare_outputs(output, output_standard, seq_start=0, seq_end=16, head_start=16, head_end=32)
    compare_outputs(output, output_standard, seq_start=16, seq_end=32, head_start=0, head_end=16)
    compare_outputs(output, output_standard, seq_start=16, seq_end=32, head_start=16, head_end=32)

    # Assert that outputs are close
    max_diff = torch.max(torch.abs(output - output_standard)).item()
    assert max_diff < 1e-2, f"Max difference {max_diff} exceeds tolerance"

    # Assert shapes match
    assert output.shape == output_standard.shape, f"Shape mismatch: {output.shape} vs {output_standard.shape}"


def create_windowed_attention_mask(seq_len, cu_window_seqlens):
    """Create the attention mask that would be implicitly generated by windowed SDPA."""
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32)

    # For each window, allow attention only within that window
    for i in range(1, len(cu_window_seqlens)):
        start = cu_window_seqlens[i - 1]
        end = cu_window_seqlens[i]
        mask[start:end, start:end] = 0.0

    return mask


def compare_outputs(output, output_standard, seq_start=0, seq_end=None, head_start=0, head_end=None):
    """Compare output tensors in specified ranges and print windowed vs standard values."""
    seq_end = seq_end or output.shape[2]
    head_end = head_end or output.shape[3]

    seq_range = min(seq_end, output.shape[2]) - seq_start
    head_range = min(head_end, output.shape[3]) - head_start

    print(
        f"Windowed vs Standard ({seq_range}x{head_range} values, seq[{seq_start}:{seq_end}], head[{head_start}:{head_end}]):"
    )
    for i in range(seq_start, min(seq_end, output.shape[2])):
        for j in range(head_start, min(head_end, output.shape[3])):
            value = output[0, 0, i, j].item()
            value_standard = output_standard[0, 0, i, j].item()
            print(f"{value:6.3f}/{value_standard:6.3f}", end=" ")
        print()
