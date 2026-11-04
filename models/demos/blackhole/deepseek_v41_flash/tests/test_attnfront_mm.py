# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention-front POC, step M1: the blaze (deepseek_v3_b1) DRAM-streaming matmul at the front-chain shapes of DSV4.1 decode (M = 8 tokens per mesh row as tiny 8x32 tiles, weights bfp8 / bfp4
replicated on all 32 chips, 8 compute cores = the 8 DRAM-bank-adjacent workers), measured in trace replay under the device profiler (the digest has METAL TRACE ID, per chip kernel durations).
Env: AF_CFGS ';'-separated 'name:K:N:dtype:fidelity:fp32:subblock_k' (default: the QKV and QB shapes, bfp8, LoFi), AF_M (8), AF_REP (ops per trace, 4), AF_TREPS (3), FD_OUT (output dir).
Output: <FD_OUT>/layer_arms.json (arms with trace_id, so dsv4-logs/attnfront/optab.py works on it), PCC of chip 0 against torch."""

import json
import os

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import DRAMStreamingMatmul
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles

OUT = os.environ.get("FD_OUT", "/mnt/tt-data/ssinghal/dsv4-logs/attnfront/run_mm")
M = int(os.environ.get("AF_M", "8"))
REP = int(os.environ.get("AF_REP", "4"))
TREPS = int(os.environ.get("AF_TREPS", "3"))
DEFAULT = "qkv:5120:1792:bfp8:LoFi:0:0;qb:1280:4096:bfp8:LoFi:0:0;qkv_hf2:5120:1792:bfp8:HiFi2:0:0;qb_hf2:1280:4096:bfp8:HiFi2:0:0"
CFGS = os.environ.get("AF_CFGS") or DEFAULT
DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16}


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 100_000_000}], indirect=True)
@pytest.mark.timeout(2400)
@torch.no_grad()
def test_attnfront_mm(mesh_device):
    md = mesh_device
    os.makedirs(OUT, exist_ok=True)
    nb = len(ttnn.get_optimal_dram_bank_to_logical_worker_assignment(md, ttnn.NOC.RISCV_0_default))
    cores = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(md, ttnn.NOC.RISCV_0_default)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
    rep = ttnn.ReplicateTensorToMesh(md)
    arms = {}
    for spec in CFGS.split(";"):
        name, K, N, dt, fid, f32, sbk = spec.split(":")
        K, N, f32, sbk = int(K), int(N), int(f32), int(sbk)
        Kt = K // 32
        sbk = sbk or max(1, Kt // 2)
        while Kt % sbk:
            sbk -= 1
        Np = -(-N // (32 * nb)) * 32 * nb
        torch.manual_seed(1)
        x = torch.randn(1, 1, M, K).bfloat16().float()
        w = (torch.randn(1, 1, K, Np) * 0.05).bfloat16().float()
        tile_in = ttnn.Tile([M, 32])
        in0 = ttnn.from_torch(
            x.repeat(1, 1, nb, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            tile=tile_in,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs, [M, K], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=rep,
        )
        dram_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(md.dram_grid_size().x - 1, md.dram_grid_size().y - 1))}
        )
        in1 = ttnn.from_torch(
            shuffle_tensor_tiles(w, 32, nb),
            dtype=DT[dt],
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_crs, [K, Np // nb], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=rep,
        )
        out = ttnn.from_torch(
            torch.zeros(1, 1, M, Np),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            tile=ttnn.Tile([M, 32]),
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs, (M, Np // nb), ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=rep,
        )

        def run():
            DRAMStreamingMatmul.op(
                in0,
                in1,
                out,
                fp32_dest_acc_en=bool(f32),
                math_fidelity=getattr(ttnn.MathFidelity, fid),
                math_approx_mode=False,
                subblock_k=sbk,
            )

        run()
        ttnn.synchronize_device(md)
        got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
        # reference with the same weight quantisation: torch matmul of the dequantised weights
        wq = ttnn.to_torch(
            ttnn.get_device_tensors(
                ttnn.from_torch(w, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=rep)
            )[0]
        ).float()
        pcc = _pcc(got.reshape(M, Np), x.reshape(M, K) @ wq.reshape(K, Np))
        ttnn.ReadDeviceProfiler(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(REP):
            run()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)
        for _ in range(TREPS):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            ttnn.ReadDeviceProfiler(md)
        arms[name] = dict(trace_id=int(tid), pcc=pcc, K=K, N=N, Np=Np, dt=dt, fid=fid, f32=f32, sbk=sbk)
        print("MMARM " + json.dumps({name: arms[name]}), flush=True)
        json.dump(dict(arms=arms, layer=-1), open(os.path.join(OUT, "layer_arms.json"), "w"))
    print("AF_DONE", flush=True)
