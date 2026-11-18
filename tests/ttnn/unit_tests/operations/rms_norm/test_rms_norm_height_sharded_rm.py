# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4a: ROW_MAJOR HEIGHT_SHARDED (flat stick shards, ragged tail, W-padded sticks)."""

import pytest
import torch
import ttnn

from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm.rms_norm import rms_norm

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _ref(x, g, eps=1e-6):
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return y * g.float() if g is not None else y


@pytest.mark.parametrize(
    "shape", [(1, 1, 64, 128), (2, 1, 96, 160), (1, 1, 333, 1000), (3, 1, 555, 1025), (1, 1, 256, 512), (4097, 64)]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("use_gamma", [False, True])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_rm_height_sharded(device, shape, dtype, use_gamma, out_sharded):
    torch.manual_seed(0)
    tdt = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    x = torch.randn(shape, dtype=tdt)
    g = torch.randn(1, 1, 1, shape[-1], dtype=tdt) if use_gamma else None
    mc = auto_shard_config(list(shape), HS, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
    tx = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    tg = ttnn.from_torch(g, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device) if use_gamma else None
    out_mc = tx.memory_config() if out_sharded else ttnn.DRAM_MEMORY_CONFIG
    out = rms_norm(tx, gamma=tg, memory_config=out_mc)
    assert (out.memory_config().memory_layout == HS) == out_sharded
    res = ttnn.to_torch(out).float()
    assert torch.allclose(res, _ref(x, g), rtol=0.05, atol=0.1)
