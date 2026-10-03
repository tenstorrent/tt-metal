# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSV41DeviceEngram.forward_v2 vs forward: PCC on random and realistic inputs, and chain timing (T = 4/8/16/32)."""

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram, _MappedTable
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

D = 5120
PARAMS = [
    pytest.param(
        {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 300_000_000},
        id="ring",
    )
]


def chain_ms(md, fn):
    def run(k):
        def f():
            for _ in range(k):
                fn()

        fn()
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        f()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        t = time.perf_counter()
        for _ in range(20):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        r = (time.perf_counter() - t) / 20 * 1e3
        ttnn.release_trace(md, tid)
        return r

    return (run(3) - run(1)) / 2


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_engram_v2(mesh_device):
    md = mesh_device
    rows_, cols_ = tuple(md.shape)
    sh = _Shards()
    cfg, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    eng = DSV41DeviceEngram(md, 1, sh, mesh_config=cfg, ccl=ccl)
    table = _MappedTable(sh, 1)  # realistic rows: random rows of the real Engram table (24 rows of 256 per token)
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows_, cols_))
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard
    )
    torch.manual_seed(0)
    nrows = table.w.shape[0]
    variants = {
        "default (compact, rm_key, addcmul)": {},
        "mode=rms (rms_norm + sum)": dict(mode="rms"),
        "compact=False": dict(mode="rms", compact=False),
        "rm_key=False": dict(mode="rms", rm_key=False),
        "fuse_out=False": dict(mode="rms", fuse_out=False),
        "compact=False,rm_key=False,fuse_out=False": dict(mode="rms", compact=False, rm_key=False, fuse_out=False),
    }
    for T in (4, 8, 16, 32):
        worst = {n: 1.0 for n in variants}
        maxd = 0.0
        dpcc = 1.0
        for trial in range(6):
            kind = trial % 3
            if kind == 0:  # random streams
                xs = torch.randn(rows_ * T, 1, 4, D)
            elif kind == 1:  # heavy-tailed, per-stream scales like real mHC streams
                xs = torch.randn(rows_ * T, 1, 4, D) * torch.exp(torch.randn(rows_ * T, 1, 4, 1))
            else:
                xs = torch.randn(rows_ * T, 1, 4, D) * 3 + torch.randn(1, 1, 4, D)
            if trial < 3:
                rw = torch.randn(rows_ * T, 1, 1, eng.kin).to(torch.bfloat16)
            else:  # real table rows
                rw = table.rows(torch.randint(0, nrows, (rows_ * T, eng.kin // 256))).reshape(rows_ * T, 1, 1, eng.kin)
            x, r = up(xs), up(rw, ttnn.bfloat16)
            ref = ttnn.to_torch(ttnn.get_device_tensors(eng.forward(x, r))[0]).float()
            for n, kw in variants.items():
                if trial == 0 or n.startswith("default"):
                    try:
                        o = ttnn.to_torch(ttnn.get_device_tensors(eng.forward_v2(x, r, **kw))[0]).float()
                        worst[n] = min(worst[n], R.pcc(ref, o))
                        if n.startswith("default"):
                            xt = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()
                            dpcc = min(dpcc, R.pcc(ref - xt, o - xt))
                        if n.startswith("default"):
                            maxd = max(maxd, (ref - o).abs().max().item())
                    except Exception as e:
                        worst[n] = float("nan")
                        print(f"EV2 T={T} {n} FAILED {str(e)[:200]}", flush=True)
        for n in variants:
            print(f"EV2 T={T:2d} PCC {n:50s} worst {worst[n]:.9f}", flush=True)
        # ref: gate dot is small -> also compare the gate contribution (out - x)
        print(
            f"EV2 T={T:2d} default max|out - ref| {maxd:.3e}  worst PCC of the gate term (out - x) {dpcc:.7f}",
            flush=True,
        )
        x, r = up(torch.randn(rows_ * T, 1, 4, D)), up(torch.randn(rows_ * T, 1, 1, eng.kin), ttnn.bfloat16)
        p = lambda n, fn: print(f"EVT T={T:2d} {n:50s} {chain_ms(md, fn) * 1e3:8.1f} us", flush=True)
        p(
            "kv only (matmul + allgather)",
            lambda: ttnn.typecast(
                cfg.allgather(
                    ttnn.matmul(
                        r,
                        eng.wkv_T,
                        compute_kernel_config=eng.ckc,
                        dtype=ttnn.bfloat16,
                        core_grid=ttnn.CoreGrid(y=8, x=8),
                    ),
                    ccl,
                    axis=1,
                    dim=3,
                ),
                ttnn.float32,
            ),
        )
        p("forward (current)", lambda: eng.forward(x, r))
        for n, kw in variants.items():
            try:
                p("forward_v2 " + n, lambda: eng.forward_v2(x, r, **kw))
            except Exception as e:
                print(f"EVT T={T} {n} FAILED {str(e)[:150]}", flush=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_engram_v2_stages(mesh_device):
    """Per-op cost of the forward_v2 default path (each op alone, repeated in a trace) + alternatives."""
    md = mesh_device
    rows_, cols_ = tuple(md.shape)
    U, UT = ttnn.UnaryWithParam, ttnn.UnaryOpType
    sh = _Shards()
    eng = DSV41DeviceEngram(md, 1, sh)
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows_, cols_))
    up = lambda t, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t, device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard
    )
    hc = 4
    for T in (4, 32):
        R_ = T * hc
        x = up(torch.randn(rows_ * T, 1, 4, D))
        kv = up(torch.randn(rows_ * T, 1, 1, 5 * D), ttnn.bfloat16)
        w = eng._weight_rows(T)
        ck = eng.ckc_norm
        st = {}

        def S(name, fn):
            st[name] = fn
            return fn()

        kv_rm = S("to_layout kv -> RM", lambda: ttnn.to_layout(kv, ttnn.ROW_MAJOR_LAYOUT))
        key_rm = S("slice key + reshape view (RM)", lambda: ttnn.reshape(kv_rm[:, :, :, : hc * D], [1, 1, R_, D]))
        key_t = S("to_layout key -> TILE", lambda: ttnn.to_layout(key_rm, ttnn.TILE_LAYOUT))
        key = S("typecast key fp32", lambda: ttnn.typecast(key_t, ttnn.float32))
        val_rm = S("slice value (RM)", lambda: kv_rm[:, :, :, hc * D :])
        val = S(
            "value to_layout TILE + typecast",
            lambda: ttnn.typecast(ttnn.to_layout(val_rm, ttnn.TILE_LAYOUT), ttnn.float32),
        )
        xr = S("reshape x -> [1,1,R,D]", lambda: ttnn.reshape(x, [1, 1, R_, D]))
        xn = S("rms_norm x", lambda: ttnn.rms_norm(xr, epsilon=1e-6, compute_kernel_config=ck))
        kn = S("rms_norm key", lambda: ttnn.rms_norm(key, epsilon=1e-6, compute_kernel_config=ck))
        xw = S("xn * w'", lambda: ttnn.multiply(xn, w))
        p_ = S("* kn", lambda: ttnn.multiply(xw, kn))
        dot = S("sum(-1)", lambda: ttnn.sum(p_, dim=-1, keepdim=True))
        gate = S(
            "gate fused op",
            lambda: ttnn.multiply(
                dot,
                dot,
                input_tensor_a_activations=[U(UT.ABS), U(UT.MAXIMUM, 1e-6), U(UT.SQRT)],
                input_tensor_b_activations=[U(UT.SIGN)],
                activations=[U(UT.SIGMOID)],
            ),
        )
        g4 = S("reshape gate [T,1,4,1]", lambda: ttnn.reshape(gate, [T, 1, hc, 1]))
        S("addcmul(x, gate, value)", lambda: ttnn.addcmul(x, g4, val))
        tot = 0
        for n, fn in st.items():
            ms = chain_ms(md, fn)
            tot += ms
            print(f"EVS T={T:2d} {n:40s} {ms * 1e3:8.1f} us", flush=True)
        print(f"EVS T={T:2d} sum of stages {tot * 1e3:.1f} us", flush=True)
        # alternatives
        alt = {}
        alt["multiply(xn, kn)  (no w)"] = lambda: ttnn.multiply(xn, kn)
        alt["sum(-1) of bf16? (typecast bf16 + sum)"] = lambda: ttnn.sum(
            ttnn.typecast(p_, ttnn.bfloat16), dim=-1, keepdim=True
        )
        alt["matmul p @ ones[D,32] fp32 HiFi4"] = lambda: ttnn.matmul(
            p_, eng._ones_dbg, compute_kernel_config=ck, dtype=ttnn.float32, core_grid=ttnn.CoreGrid(y=2, x=8)
        )
        eng._ones_dbg = (
            up(torch.ones(1, 1, D, 32)[:, :, :, :], ttnn.float32)
            if False
            else ttnn.from_torch(
                torch.ones(1, 1, D, 32),
                device=md,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
        )
        alt["rms_norm x with weight (single op ref)"] = lambda: ttnn.rms_norm(
            xr, epsilon=1e-6, weight=eng._w_scaled_row0, compute_kernel_config=ck
        )
        eng._w_scaled_row0 = ttnn.from_torch(
            torch.ones(1, 1, 1, D),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        alt["rms_norm x in bf16 out (timing only)"] = (
            lambda: ttnn.rms_norm(xr, epsilon=1e-6, compute_kernel_config=ck, dtype=ttnn.bfloat16)
            if False
            else ttnn.rms_norm(xr, epsilon=1e-6, compute_kernel_config=ck)
        )
        for n, fn in alt.items():
            try:
                fn()
                print(f"EVA T={T:2d} {n:40s} {chain_ms(md, fn) * 1e3:8.1f} us", flush=True)
            except Exception as e:
                print(f"EVA T={T:2d} {n:40s} FAILED {str(e)[:120]}", flush=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_engram_kv_matmul_shapes(mesh_device):
    """Observation only (the kv matmul is not changed by forward_v2): rows [T,1,1,Kin] makes the matmul T batches of M=1, which
    re-streams the column-sharded weight T times; rows as [1,1,T,Kin] is one M=T matmul."""
    md = mesh_device
    rows_, cols_ = tuple(md.shape)
    sh = _Shards()
    cfg, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    eng = DSV41DeviceEngram(md, 1, sh, mesh_config=cfg, ccl=ccl)
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows_, cols_))
    for T in (4, 16, 32):
        r = ttnn.from_torch(
            torch.randn(rows_ * T, 1, 1, eng.kin).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        mm = lambda a: ttnn.matmul(
            a, eng.wkv_T, compute_kernel_config=eng.ckc, dtype=ttnn.bfloat16, core_grid=ttnn.CoreGrid(y=8, x=8)
        )
        r2 = ttnn.reshape(r, [1, 1, T, eng.kin])
        print(
            f"EVK T={T:2d} matmul batched [T,1,1,Kin]: {chain_ms(md, lambda: mm(r)) * 1e3:7.1f} us   M=T [1,1,T,Kin]: {chain_ms(md, lambda: mm(r2)) * 1e3:7.1f} us   reshape rows: {chain_ms(md, lambda: ttnn.reshape(r, [1, 1, T, eng.kin])) * 1e3:6.1f} us",
            flush=True,
        )
        a = ttnn.to_torch(ttnn.get_device_tensors(mm(r))[0]).float().reshape(T, -1)
        b = ttnn.to_torch(ttnn.get_device_tensors(mm(r2))[0]).float().reshape(T, -1)
        print(f"EVK T={T:2d} outputs identical: {bool(torch.equal(a, b))}  PCC {R.pcc(a, b):.8f}", flush=True)
