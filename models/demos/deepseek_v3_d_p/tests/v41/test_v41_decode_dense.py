# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The decode tile's dense projections as one DRAM-rate program (``tt/v41/decode_dense.py``,
``V41_DECODE_DENSE=matvec``) against ``ttnn.linear`` on the served compute config and against an fp32
reference of the
dequantized weight, on the attention's five projection shapes (``wq_a``, ``wkv``, ``wq_b``, ``wo_a`` with its
two
groups, ``wo_b``) at TP 4: random bf16 rows (every row of the tile, every chip its own), the distance of both
paths
to the reference, then each under trace (``V41_DD_REPLAYS`` replays of ``V41_DD_CALLS`` calls) for the
per-call
device time. No weights; ~1 minute on the Galaxy. Results: ``V41_DD_RESULT``, ``V41_DD_OUT``."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tt.v41.attention import (
    DECODE_1D,
    DENSE_COMPUTE_CONFIG,
    DENSE_IN0_BLOCK_W,
    dense_program_config,
)
from models.demos.deepseek_v3_d_p.tt.v41.decode_dense import DecodeDense, column_memory_config

TP = 4
# name -> (B, K, N) per chip at TP 4
SHAPES = {
    "wq_a": (1, C.EMB_SIZE // TP, C.Q_LORA_RANK),
    "wkv": (1, C.EMB_SIZE // TP, C.HEAD_DIM + C.QK_ROPE_HEAD_DIM),
    "wq_b": (1, C.Q_LORA_RANK, C.NUM_ATTENTION_HEADS // TP * C.HEAD_DIM),
    "wo_a": (C.O_GROUPS // TP, C.NUM_ATTENTION_HEADS * C.HEAD_DIM // C.O_GROUPS, C.O_LORA_RANK),
    "wo_b": (1, C.O_GROUPS * C.O_LORA_RANK // TP, C.EMB_SIZE),
}


def _trace_us(mesh_device, fn, calls: int, replays: int) -> float:
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outs = [fn() for _ in range(calls)]
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    best = None
    for _ in range(replays):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        dt = (time.perf_counter() - t0) / calls
        best = dt if best is None else min(best, dt)
    ttnn.release_trace(mesh_device, tid)
    for o in outs:
        ttnn.deallocate(o)
    return round(best * 1e6, 1)


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double().flatten(), b.double().flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    rel = ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
    return {"pcc": round(pcc, 7), "rel": round(rel, 6), "max_abs": round((a - b).abs().max().item(), 5)}


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=64 * 1024 * 1024), indirect=True)
def test_v41_decode_dense(mesh_device, device_params):
    replays = int(os.environ.get("V41_DD_REPLAYS", 20))
    calls = int(os.environ.get("V41_DD_CALLS", 32))
    sp, tp = tuple(mesh_device.shape)
    chips = sp * tp
    torch.manual_seed(130)
    result = {"mesh": list(mesh_device.shape), "shapes": {}}
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    shard = ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, 1))
    for name, (B, K, N) in SHAPES.items():
        w_host = torch.randn(1, B, K, N) * 0.02
        # the bfp8 rounding is the host's (``from_torch`` without a device) for both layouts: ``from_torch``
        # with a
        # device and a sharded memory config rounds differently (1 ulp on a tenth of the words)
        w_h = ttnn.from_torch(w_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate)
        w = ttnn.to_device(w_h, mesh_device)
        w_deq = ttnn.to_torch(ttnn.get_device_tensors(w)[0]).float()  # the bfp8 weight as the device holds it
        w_col = ttnn.to_device(w_h, mesh_device, memory_config=column_memory_config(mesh_device, K))
        assert torch.equal(
            ttnn.to_torch(ttnn.get_device_tensors(w_col)[0]), ttnn.to_torch(ttnn.get_device_tensors(w)[0])
        )
        x_host = torch.randn(sp, tp * B, 32, K).to(torch.bfloat16)  # chip (r, c): its B row blocks
        x = ttnn.from_torch(x_host, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard)
        x_host = x_host.view(chips, B, 32, K)
        dense = DecodeDense(w, mesh_device)
        dense_col = DecodeDense(w_col, mesh_device)
        assert dense_col.col_contig and not dense.col_contig

        def linear():
            if name in DENSE_IN0_BLOCK_W:
                cfg = dense_program_config(32, K, N, DENSE_IN0_BLOCK_W[name], one_d=name in DECODE_1D)
                return ttnn.linear(x, w, program_config=cfg, compute_kernel_config=DENSE_COMPUTE_CONFIG)
            return ttnn.linear(x, w, compute_kernel_config=DENSE_COMPUTE_CONFIG)

        ref = torch.matmul(x_host.float(), w_deq)  # [chips, B, 32, N]
        got_lin = torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(linear())])
        got_mv = torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(dense(x))])
        assert got_mv.shape == ref.shape, (got_mv.shape, ref.shape)
        got_col = torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(dense_col(x))])
        assert torch.equal(got_col, got_mv), name  # the same program on the same bytes: bit for bit
        entry = {
            "B": B,
            "K": K,
            "N": N,
            "cols": dense.cols,
            "S": dense.S,
            "L": dense.L,
            "blk": dense.BLK,
            "units_max": dense.units_max,
            "linear_vs_ref": _stats(got_lin, ref),
            "matvec_vs_ref": _stats(got_mv, ref),
            "matvec_vs_linear": _stats(got_mv, got_lin),
            "mb": round(B * K * N * 1088 / 1024 / 1e6, 2),
        }
        # the timing under trace
        entry["linear_us"] = _trace_us(mesh_device, linear, calls, replays)
        entry["matvec_us"] = _trace_us(mesh_device, lambda: dense(x), calls, replays)
        entry["matvec_col_us"] = _trace_us(mesh_device, lambda: dense_col(x), calls, replays)
        for k in ("linear", "matvec", "matvec_col"):
            entry[f"{k}_gbps"] = round(entry["mb"] / entry[f"{k}_us"] * 1e3, 1)
        # the prefill's matmul on the column-sharded weight (the layout can replace the served one only if
        # ttnn.linear reads it on the chunk shapes no slower): 640 rows a chip (5,120 / sp) on the 2D config
        rows = 5120 // sp
        xp_host = torch.randn(sp, tp * B, rows, K).to(torch.bfloat16)
        xp = ttnn.from_torch(
            xp_host, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard
        )
        cfg = dense_program_config(rows, K, N, DENSE_IN0_BLOCK_W.get(name, 8))

        def plinear(wt):
            return ttnn.linear(xp, wt, program_config=cfg, compute_kernel_config=DENSE_COMPUTE_CONFIG)

        try:
            p_int = torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(plinear(w))])
            p_col = torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(plinear(w_col))])
            entry["prefill_col_equal"] = bool(torch.equal(p_int, p_col))
            entry["prefill_linear_us"] = _trace_us(mesh_device, lambda: plinear(w), 4, replays)
            entry["prefill_linear_col_us"] = _trace_us(mesh_device, lambda: plinear(w_col), 4, replays)
        except Exception as e:  # noqa: BLE001
            entry["prefill_col_error"] = str(e).splitlines()[0][:200]
        ttnn.deallocate(xp)
        logger.info(f"decode dense {name}: {entry}")
        result["shapes"][name] = entry
        assert entry["matvec_vs_ref"]["pcc"] > 0.9999, (name, entry)
        assert entry["matvec_vs_ref"]["rel"] <= entry["linear_vs_ref"]["rel"] * 1.5 + 1e-6, (name, entry)
        ttnn.deallocate(w)
        ttnn.deallocate(w_col)
        ttnn.deallocate(x)
    logger.info(f"V41_DD_RESULT {json.dumps(result)}")
    out = os.environ.get("V41_DD_OUT")
    if out:
        json.dump(result, open(out, "w"), indent=1)
