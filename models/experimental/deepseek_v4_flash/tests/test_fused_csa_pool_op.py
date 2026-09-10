# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit test for ``ttnn.experimental.deepseek.csa_pool_window``.

The op fuses CSA compressor pooling (attention.py ``_pool_window``): add
``position_bias``, take prev Ca + current Cb, softmax-weight over the 2*cr window,
and emit ``[1, 1, users, Dh]``. Inputs are ROW_MAJOR WIDTH_SHARDED over ``2*Dh``.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc


PCC_THRESHOLD = 0.999
TILE = 32
MASK_NEG = -1.0e9


def _torch_reference(prev_kv, prev_gate, win_kv, win_gate, bias, head_dim):
    """Mirror of CSA ``_pool_window`` lines 827-840 in float32."""
    users, _, cr, feat = prev_kv.shape
    prev_g = prev_gate + bias
    cur_g = win_gate + bias
    new_kv = torch.cat([prev_kv[..., :head_dim], win_kv[..., head_dim:]], dim=2)
    new_gate = torch.cat([prev_g[..., :head_dim], cur_g[..., head_dim:]], dim=2)
    weights = torch.softmax(new_gate, dim=2)
    compressed = (new_kv * weights).sum(dim=2)
    return compressed.reshape(1, 1, users, head_dim)


def _width_sharded_cfg(rows: int, width: int, num_cores: int) -> ttnn.MemoryConfig:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard = ttnn.ShardSpec(grid, [rows, width // num_cores], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard)


def _to_rm_ws(t: torch.Tensor, device, num_cores: int) -> ttnn.Tensor:
    packed = t.reshape(1, 1, t.shape[0] * t.shape[-2], t.shape[-1])
    return ttnn.to_memory_config(
        ttnn.from_torch(packed, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
        _width_sharded_cfg(packed.shape[-2], packed.shape[-1], num_cores),
    )


@pytest.mark.parametrize(
    "users, cr, dh, num_cores, mask_prev",
    (
        (1, 4, 64, 4, False),  # small CSA-like: 2*Dh=128 over 4 cores
        (1, 4, 64, 4, True),  # first window: prev_gate is -inf
        (2, 4, 64, 4, False),
        (1, 4, 128, 8, False),
    ),
)
def test_csa_pool_window_op(device, reset_seeds, users, cr, dh, num_cores, mask_prev):
    feat = 2 * dh
    prev_kv = torch.randn(users, 1, cr, feat, dtype=torch.float32)
    prev_gate = torch.randn(users, 1, cr, feat, dtype=torch.float32)
    win_kv = torch.randn(users, 1, cr, feat, dtype=torch.float32)
    win_gate = torch.randn(users, 1, cr, feat, dtype=torch.float32)
    bias = torch.randn(1, 1, cr, feat, dtype=torch.float32)
    if mask_prev:
        prev_gate = torch.full_like(prev_gate, MASK_NEG)

    ref = _torch_reference(prev_kv, prev_gate, win_kv, win_gate, bias, dh)

    prev_kv_tt = _to_rm_ws(prev_kv, device, num_cores)
    prev_gate_tt = _to_rm_ws(prev_gate, device, num_cores)
    win_kv_tt = _to_rm_ws(win_kv, device, num_cores)
    win_gate_tt = _to_rm_ws(win_gate, device, num_cores)
    bias_tt = _to_rm_ws(bias, device, num_cores)

    out_tt = ttnn.experimental.deepseek.csa_pool_window(prev_kv_tt, prev_gate_tt, win_kv_tt, win_gate_tt, bias_tt)
    assert out_tt.layout == ttnn.ROW_MAJOR_LAYOUT
    got = ttnn.to_torch(out_tt).reshape(ref.shape).float()
    passing, pcc_message = comp_pcc(ref, got, pcc=PCC_THRESHOLD)
    tag = f"users={users} cr={cr} dh={dh} cores={num_cores} mask_prev={mask_prev}"
    logger.info(f"[csa_pool_window {tag}] {comp_allclose(ref, got)}")
    logger.info(f"[csa_pool_window {tag}] PCC: {pcc_message}")
    assert passing, f"csa_pool_window PCC < {PCC_THRESHOLD} ({tag}): {pcc_message}"
