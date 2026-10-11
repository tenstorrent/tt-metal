# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused ``ttnn.experimental.deepseek.engram_gate`` op against a torch reference of
the Engram gate math in ``DeepSeekV41Engram.forward`` (everything after the ``wkv`` matmul).

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_engram_gate_op.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config

HC = 4
DIM = 4096
EPS = 1e-6
GATE_CLAMP = 1e-6


def _reference(x, kv, gate_weight):
    """``x`` ``[B, 1, hc, D]``, ``kv`` ``[1, 1, B, (hc+1)*D]``, ``gate_weight`` ``[hc, D]``; fp32."""
    b, _, hc, d = x.shape
    key = kv[..., : hc * d].reshape(b, 1, hc, d)
    value = kv[..., hc * d :].reshape(b, 1, 1, d)
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * torch.rsqrt(key.pow(2).mean(-1, keepdim=True) + EPS)
    dot = (x * gate_weight * key).sum(-1, keepdim=True) * rstd
    gate = torch.sigmoid(dot.abs().clamp_min(GATE_CLAMP).sqrt() * dot.sign())
    return x + gate * value


@pytest.mark.parametrize("batch", (1, 4, 32))
# The op computes on the gate weight's cores; with fewer streams cores (the model's 8 collapse cores)
# each streams shard is split over several of them.
@pytest.mark.parametrize("streams_cores, gate_cores", ((16, 16), (32, 32), (8, 16), (8, 32)))
# 64 cores -> 320-wide kv shards, which the streams slices straddle.
@pytest.mark.parametrize("kv_cores", (40, 64, 80))
def test_engram_gate_op(device, reset_seeds, batch, streams_cores, gate_cores, kv_cores):
    rows = batch * HC
    x = torch.randn(batch, 1, HC, DIM).bfloat16().float()
    kv = torch.randn(1, 1, batch, (HC + 1) * DIM).bfloat16().float()
    q_weight = 1.0 + 0.1 * torch.randn(HC, DIM)
    k_weight = 1.0 + 0.1 * torch.randn(HC, DIM)
    gate_weight = q_weight * k_weight * DIM**-0.5
    expected = _reference(x, kv, gate_weight)

    streams_mem = width_sharded_l1_config(rows, DIM, device, num_cores=streams_cores, tile_height=1)
    kv_mem = width_sharded_l1_config(batch, (HC + 1) * DIM, device, num_cores=kv_cores, tile_height=1)
    gate_mem = width_sharded_l1_config(ttnn.TILE_SIZE, DIM, device, num_cores=gate_cores)

    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=streams_mem
    )
    tt_kv = ttnn.from_torch(kv, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=kv_mem)
    tt_gate = ttnn.from_torch(
        gate_weight.repeat(ttnn.TILE_SIZE // HC, 1).reshape(1, 1, ttnn.TILE_SIZE, DIM),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_mem,
    )

    # Twice: the second call runs from the program cache, after the exchange semaphore was reset.
    for _ in range(2):
        tt_out = ttnn.experimental.deepseek.engram_gate(tt_x, tt_kv, tt_gate, EPS)
    assert tt_out.memory_config() == streams_mem
    actual = ttnn.to_torch(tt_out).float()

    passing, pcc = comp_pcc(expected, actual, 0.9999)
    logger.info(f"out PCC {pcc}")
    assert passing, f"out PCC {pcc}"
    passing, pcc = comp_pcc(expected - x, actual - x, 0.999)
    logger.info(f"out - x PCC {pcc}")
    assert passing, f"out - x PCC {pcc}"
