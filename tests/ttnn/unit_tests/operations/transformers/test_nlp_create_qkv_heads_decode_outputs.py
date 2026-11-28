# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""nlp_create_qkv_heads_decode must not ignore caller output tensors.

The public signature takes `output_tensors` (three optional preallocated tensors), but the function
always allocated and returned three new tensors, so a caller that passed outputs got different
tensors back with no error.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 96
BATCH = 32


def _input(device):
    total_heads = NUM_HEADS + 2 * NUM_KV_HEADS
    torch_input = torch.rand(1, 1, BATCH, HEAD_DIM * total_heads)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_input, tt_input


def test_nlp_create_qkv_heads_decode_rejects_preallocated_outputs(device, expect_error):
    _, tt_input = _input(device)
    # Shapes are irrelevant: the call is rejected for passing outputs at all.
    outputs = [
        ttnn.from_torch(
            torch.zeros(1, BATCH, NUM_HEADS, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
        ttnn.from_torch(
            torch.zeros(1, BATCH, NUM_KV_HEADS, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
        ttnn.from_torch(
            torch.zeros(1, BATCH, NUM_KV_HEADS, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
    ]
    with expect_error(RuntimeError, "preallocated output_tensors"):
        ttnn.experimental.nlp_create_qkv_heads_decode(
            tt_input, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, output_tensors=outputs
        )


def test_nlp_create_qkv_heads_decode_default_outputs(device):
    # Control that passes with and without the fix: no outputs requested, three are allocated.
    torch_input, tt_input = _input(device)
    height_sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
        tt_input,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        memory_config=height_sharded,
    )

    proj = torch_input.reshape(1, BATCH, HEAD_DIM * (NUM_HEADS + 2 * NUM_KV_HEADS))
    q_ref = proj[:, :, : HEAD_DIM * NUM_HEADS].reshape(1, BATCH, NUM_HEADS, HEAD_DIM)
    k_ref = proj[:, :, HEAD_DIM * NUM_HEADS : HEAD_DIM * (NUM_HEADS + NUM_KV_HEADS)].reshape(
        1, BATCH, NUM_KV_HEADS, HEAD_DIM
    )
    v_ref = proj[:, :, HEAD_DIM * (NUM_HEADS + NUM_KV_HEADS) :].reshape(1, BATCH, NUM_KV_HEADS, HEAD_DIM)

    assert_with_pcc(q_ref.float(), ttnn.to_torch(q).float(), 0.999)
    assert_with_pcc(k_ref.float(), ttnn.to_torch(k).float(), 0.999)
    assert_with_pcc(v_ref.float(), ttnn.to_torch(v).float(), 0.999)
