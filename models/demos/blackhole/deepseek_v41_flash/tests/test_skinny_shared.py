# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit test + trace-replay timing of the shared-expert chain (tt/skinny_shared.py) against torch (dequantised bfp8 weights) and the stock column-parallel shared expert (tt/shared_expert_tp.py).
Env: SK_T (8), SK_ARMS (comma list of 'skinny', 'skinny_c' (compact x), 'stock'; default skinny,stock), SK_REP (4), SK_TREPS (3), FD_OUT."""

import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import skinny_shared as SS
from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert_tp import DSV41SharedExpertColTP

OUT = os.environ.get("FD_OUT", "/mnt/tt-data/ssinghal/dsv4-logs/skinnymm/run_shared")
T = int(os.environ.get("SK_T", "8"))
REP = int(os.environ.get("SK_REP", "4"))
TREPS = int(os.environ.get("SK_TREPS", "3"))
ARMS = os.environ.get("SK_ARMS", "skinny,stock").split(",")
L = 10.0


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 100_000_000}], indirect=True)
@pytest.mark.timeout(2400)
@torch.no_grad()
def test_skinny_shared(mesh_device):
    md = mesh_device
    os.makedirs(OUT, exist_ok=True)
    rows, cols = tuple(md.shape)
    torch.manual_seed(5)
    w0 = (torch.randn(1, 1, 5120, 2304) * 0.03).bfloat16().float()
    w1 = (torch.randn(1, 1, 5120, 2304) * 0.03).bfloat16().float()
    w2 = (torch.randn(1, 1, 2304, 5120) * 0.03).bfloat16().float()
    h_host = (torch.randn(1, 1, T, 5120) * 1.5).bfloat16().float()
    h = ttnn.from_torch(
        h_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    # dequantised weights as the device sees them (bfp8), one copy
    deq = lambda w: ttnn.to_torch(
        ttnn.get_device_tensors(
            ttnn.from_torch(
                w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=ttnn.ReplicateTensorToMesh(md)
            )
        )[0]
    ).float()
    d0, d1, d2 = deq(w0).reshape(5120, 2304), deq(w1).reshape(5120, 2304), deq(w2).reshape(2304, 5120)
    x = h_host.reshape(T, 5120)
    pccs, arms = {}, {}

    def check(out, name):
        res = []
        for chip in (0, 5, 13, 31):
            c = chip % cols
            sl = slice(288 * c, 288 * (c + 1))
            g = (x @ d0[:, sl]).bfloat16().float().clamp(max=L)
            u = (x @ d1[:, sl]).bfloat16().float().clamp(-L, L)
            act = (torch.nn.functional.silu(g) * u).bfloat16().float()
            ref = act @ d2[sl]
            got = ttnn.to_torch(ttnn.get_device_tensors(out)[chip]).float().reshape(-1, 5120)[:T]
            res.append(_pcc(got, ref))
        pccs[name] = res
        print("PCC", name, res, flush=True)

    fns = {}
    if "skinny" in ARMS or "skinny_c" in ARMS:
        sk = SS.SkinnyShared(md, w0, w1, w2)
    if "skinny" in ARMS:
        fns["skinny"] = lambda: sk(h, T)
    if "skinny_c" in ARMS:
        mt = 1 if T <= 8 else 2
        buf = torch.zeros(mt, 160, 2, 8, 16)
        hp = torch.zeros(16, 5120)
        hp[:T] = h_host[0, 0]
        for m in range(mt):
            for kt in range(160):
                blk = hp[m * 8 : m * 8 + 8, kt * 32 : kt * 32 + 32]
                buf[m, kt, 0] = blk[:, :16]
                buf[m, kt, 1] = blk[:, 16:]
        hc = ttnn.from_torch(
            buf.reshape(5 * mt, 8192),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        fns["skinny_c"] = lambda: sk(hc, T)
    if "stock" in ARMS:
        st = DSV41SharedExpertColTP(md, w0, w1, w2)
        fns["stock"] = lambda: st.forward(h)
    for name, fn in fns.items():
        o = fn()
        ttnn.synchronize_device(md)
        check(o, name)
        ttnn.ReadDeviceProfiler(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(REP):
            fn()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)
        for _ in range(TREPS):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            ttnn.ReadDeviceProfiler(md)
        arms[name] = dict(trace_id=int(tid), pcc=pccs.get(name))
        json.dump(dict(arms=arms, layer=-1), open(os.path.join(OUT, "layer_arms.json"), "w"))
    print("SK_DONE " + json.dumps(arms), flush=True)
