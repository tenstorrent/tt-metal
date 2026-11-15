# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``ttnn.experimental.deepseek.engram_gate`` on a DSpark block: streams [1, 1, hc*h, D] in (h x 32) tiles (row
s*h + j = stream s of block row j), kv [1, 1, h, (hc+1)*D] in the same tiles, and a gate weight whose row r
holds copy (r // h) % hc.

    pytest tests/ttnn/unit_tests/operations/experimental/deepseek/dspark_rows/test_engram_gate_rows.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config

HC = 4
D = 5120
EPS = 1e-6
CLAMP = 1e-6
STREAMS_CORES, KV_CORES, GATE_CORES = 8, 80, 16


def _reference(x, kv, gate_weight):
    """x [B, hc, D], kv [B, (hc+1)*D], gate_weight [hc, D] -> [B, hc, D] (fp32)."""
    b, hc, d = x.shape
    key = kv[:, : hc * d].reshape(b, hc, d)
    value = kv[:, hc * d :].reshape(b, 1, d)
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * torch.rsqrt(key.pow(2).mean(-1, keepdim=True) + EPS)
    dot = (x * gate_weight * key).sum(-1, keepdim=True) * rstd
    gate = torch.sigmoid(dot.abs().clamp_min(CLAMP).sqrt() * dot.sign())
    return x + gate * value


def _gate_weight(device, gw, block_h):
    rows = torch.arange(ttnn.TILE_SIZE)
    copy = (rows // block_h) % HC if block_h else rows % HC
    return ttnn.from_torch(
        gw[copy].reshape(1, 1, ttnn.TILE_SIZE, D),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=width_sharded_l1_config(ttnn.TILE_SIZE, D, device, num_cores=GATE_CORES),
    )


def _inputs(device, rows, h, seed=0):
    """A block of `rows` (<= h) rows: torch x [rows, hc, D], kv [rows, (hc+1)*D], gw [hc, D] and the device
    tensors (h == 0: today's ROW_MAJOR users [rows, 1, hc, D])."""
    torch.manual_seed(seed)
    x = torch.randn(rows, HC, D).bfloat16().float()
    kv = torch.randn(rows, (HC + 1) * D).bfloat16().float()
    gw = (1.0 + 0.1 * torch.randn(HC, D)) * (1.0 + 0.1 * torch.randn(HC, D)) * D**-0.5
    if h == 0:
        x_tt = ttnn.from_torch(
            x.reshape(rows, 1, HC, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=width_sharded_l1_config(rows * HC, D, device, num_cores=STREAMS_CORES, tile_height=1),
        )
        kv_tt = ttnn.from_torch(
            kv.reshape(1, 1, rows, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=width_sharded_l1_config(rows, (HC + 1) * D, device, num_cores=KV_CORES, tile_height=1),
        )
    else:
        tile = ttnn.Tile((h, 32))
        # Stream-major: row s*h + j; block rows past `rows` are padding.
        xs = torch.zeros(HC, h, D)
        xs[:, :rows] = x.transpose(0, 1)
        x_tt = ttnn.from_torch(
            xs.reshape(1, 1, HC * h, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=tile,
            device=device,
            memory_config=width_sharded_l1_config(HC * h, D, device, num_cores=STREAMS_CORES, tile_height=h),
        )
        kv_tt = ttnn.from_torch(
            kv.reshape(1, 1, rows, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=tile,
            device=device,
            memory_config=width_sharded_l1_config(h, (HC + 1) * D, device, num_cores=KV_CORES, tile_height=h),
        )
    return x, kv, gw, x_tt, kv_tt, _gate_weight(device, gw, h)


@pytest.mark.parametrize("rows, h", [(4, 4), (1, 1), (2, 2), (5, 8), (8, 8)], ids=str)
def test_engram_gate_rows(device, rows, h):
    x, kv, gw, x_tt, kv_tt, g_tt = _inputs(device, rows, h)
    expected = _reference(x, kv, gw)
    for _ in range(2):  # the second call runs from the program cache
        out_tt = ttnn.experimental.deepseek.engram_gate(x_tt, kv_tt, g_tt, EPS)
    assert out_tt.memory_config() == x_tt.memory_config()
    assert out_tt.layout == ttnn.TILE_LAYOUT and tuple(out_tt.tile.tile_shape) == (h, 32)
    out = ttnn.to_torch(out_tt).float().reshape(HC, h, D)[:, :rows].transpose(0, 1)
    for j in range(rows):
        for s in range(HC):
            passing, pcc = comp_pcc(expected[j, s], out[j, s], 0.9999)
            passing_d, pcc_d = comp_pcc(expected[j, s] - x[j, s], out[j, s] - x[j, s], 0.999)
            logger.info(f"row {j} stream {s}: out PCC {pcc}, out - x PCC {pcc_d}")
            assert passing and passing_d, f"row {j} stream {s}: {pcc} / {pcc_d}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 10240}], indirect=True)
def test_engram_gate_rows_perf(device):
    """Production: streams on the 8 collapse cores, kv on wkv's 80 cores, the gate on 16 cores."""
    from tracy import signpost

    *_, x1, kv1, g1 = _inputs(device, 1, 0)
    *_, x4, kv4, g4 = _inputs(device, 4, 4)
    for name, args in {"s1": (x1, kv1, g1), "s4": (x4, kv4, g4)}.items():
        ttnn.deallocate(ttnn.experimental.deepseek.engram_gate(*args, EPS))
        ttnn.synchronize_device(device)
        signpost(name)
        for _ in range(20):
            ttnn.deallocate(ttnn.experimental.deepseek.engram_gate(*args, EPS))
        ttnn.synchronize_device(device)
