# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 5: WIDTH_SHARDED / BLOCK_SHARDED input with cross-core sum-of-squares combine."""

import pytest
import torch
import ttnn

from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm.rms_norm import rms_norm

ML = ttnn.TensorMemoryLayout


def _ref(x, g, eps=1e-6):
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return y * g.float() if g is not None else y


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("ml", [ML.WIDTH_SHARDED, ML.BLOCK_SHARDED])
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 2048), (1, 1, 256, 512), (100, 5120), (1, 1, 32, 200), (1, 1, 40, 40), (2, 1, 96, 1000), (333, 1025)],
)
@pytest.mark.parametrize("use_gamma", [False, True])
def test_cross_core_sharded(device, shape, use_gamma, ml, layout):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16) if use_gamma else None
    mc = auto_shard_config(list(shape), ml, layout=layout, dtype=ttnn.bfloat16, device=device)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=layout, device=device) if use_gamma else None
    out = rms_norm(tx, gamma=tg, memory_config=tx.memory_config())
    assert out.memory_config().memory_layout == ml
    res = ttnn.to_torch(out).float()
    assert torch.allclose(res, _ref(x, g), rtol=0.05, atol=0.1)
