# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx):
    cos = cos_cached[:, :, token_idx : token_idx + 1, ...]
    sin = sin_cached[:, :, token_idx : token_idx + 1, ...]
    return (x * cos) + (rotate_half(x) * sin)


@pytest.mark.parametrize("cache_size", [2048])
@pytest.mark.parametrize("head_dim", [64])
def test_rotary_embedding_decode_program_cache_reuse(cache_size, head_dim, device):
    # Regression: token_idx's value is excluded from the hash, so decode positions hit the same cached
    # program and the token_idx-derived args must be re-applied each hit or they freeze. The existing
    # decode test runs each token_idx in isolation (cache miss), so it can't catch this.
    torch.manual_seed(0)
    input_shape = [1, 1, 32, head_dim]
    sin_cos_shape = [1, 1, cache_size, head_dim]
    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    sin_cached = torch.randn(sin_cos_shape).bfloat16().float()
    cost = ttnn.Tensor(cos_cached, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    sint = ttnn.Tensor(sin_cached, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    # Positions that move both within a tile (cos_sin_offset) and across tiles (cos_sin_start_id).
    for token_idx in [0, 1, 31, 32, 33, 128, 1025]:
        x = torch.randn(input_shape).bfloat16().float()
        xt = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
        xtt = ttnn.experimental.rotary_embedding(xt, cost, sint, token_idx)
        tt_out = xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        pt_out = apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx)
        passing, out = comp_pcc(pt_out, tt_out)
        logger.info(f"token_idx={token_idx}: {out}")
        assert passing, f"rotary_embedding wrong at token_idx={token_idx} (program-cache freeze)"

        # Force the next input to a fresh address so the cache-hit path must also re-patch addresses.
        _dummy = ttnn.Tensor(torch.randn(input_shape), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    # token_idx is excluded from the hash, so every position reused ONE program (no per-token recompile).
    assert (
        device.num_program_cache_entries() == 1
    ), f"expected 1 cached program, got {device.num_program_cache_entries()}"
