# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import ttnn

from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa.op import (
    EncoderSDPAConfig,
    bge_encoder_sdpa_experimental,
)
from models.common.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.skipif(not is_wormhole_b0(), reason="BGE model-local SDPA targets Wormhole")
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("runtime_lengths", [False, True])
def test_bge_encoder_sdpa_reduce_auxiliary(device, streaming, runtime_lengths):
    """Exercise the model-local writer recipe with both compute paths and partial KV masking."""
    grid = device.compute_with_storage_grid_size()
    config = EncoderSDPAConfig(
        batch=1,
        num_q_heads=2,
        num_kv_heads=1,
        q_seq_len=grid.x * grid.y * 16,
        kv_seq_len=128,
        head_dim=64,
        q_chunk_size=32,
        k_chunk_size=64,
        grid_x=grid.x,
        grid_y=grid.y,
        scale=1 / math.sqrt(64),
        use_streaming=streaming,
        use_runtime_lengths=runtime_lengths,
    )
    torch.manual_seed(29)
    q, k, v = [torch.randn(shape, dtype=torch.bfloat16) for shape in (config.q_shape, config.kv_shape, config.kv_shape)]
    inputs = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v)]
    valid_length = 95
    lengths = (
        ttnn.from_torch(torch.tensor([[valid_length]], dtype=torch.int32), dtype=ttnn.uint32, device=device)
        if runtime_lengths
        else None
    )
    result = bge_encoder_sdpa_experimental(*inputs, config=config, valid_lengths=lengths)
    mask = (torch.arange(config.kv_seq_len) < valid_length).view(1, -1) if runtime_lengths else None
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float().repeat_interleave(2, dim=1),
        v.float().repeat_interleave(2, dim=1),
        attn_mask=mask,
        is_causal=False,
        scale=config.scale,
    )
    assert_with_pcc(expected, ttnn.to_torch(result).float(), 0.99)
