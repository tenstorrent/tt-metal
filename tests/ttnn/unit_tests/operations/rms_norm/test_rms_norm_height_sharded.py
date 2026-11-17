# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4: HEIGHT_SHARDED TILE input consumed in place; sharded output via memory_config."""

import pytest
import torch
import ttnn

from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm.rms_norm import rms_norm

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _ref(x, g, eps=1e-6):
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return y * g.float() if g is not None else y


@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (2, 1, 96, 160), (1, 1, 333, 1000), (3, 1, 555, 1025), (99991, 64)])
@pytest.mark.parametrize("use_gamma", [False, True])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_height_sharded_in_place(device, shape, use_gamma, out_sharded):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16) if use_gamma else None
    mc = auto_shard_config(list(shape), HS, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device) if use_gamma else None
    out_mc = tx.memory_config() if out_sharded else ttnn.DRAM_MEMORY_CONFIG
    out = rms_norm(tx, gamma=tg, memory_config=out_mc)
    assert (out.memory_config().memory_layout == HS) == out_sharded
    res = ttnn.to_torch(out).float()
    ref = _ref(x, g)
    assert torch.allclose(res, ref, rtol=0.05, atol=0.1)
