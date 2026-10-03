# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Time each stage of the exact router (tt/router.py _forward_exact), inside a long trace, layer 2 gate weights."""

import time

import pytest
import torch

import ttnn
from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import CONFIG_PATH
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer
from models.demos.blackhole.deepseek_v41_flash.tt.router import DSV41Gate

T = 4


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
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_router_stages(mesh_device):
    md = mesh_device
    text = (
        CONFIG_PATH.read_text()
        .replace("num_shared_experts: 1", "num_shared_experts: 0")
        .replace("  shared_expert_ids_to_devices: fully_replicated\n", "")
    )
    cfg = TTMoEGateConfig.from_yaml(text).model_copy(update={"batch_per_device": T})
    w = load_moe_layer(2, experts=[0])
    gate = DSV41Gate(md, cfg, torch_gate_weight=w["gate_weight"], torch_gate_bias=w["gate_bias"], bias_shift=0.0)
    h = ttnn.from_torch(
        torch.randn(1, 1, T, 5120).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    ckc = gate._ckc_exact
    k = gate.k
    print(
        "SHAPES gate_weight",
        gate.tt_gate_weight.shape,
        gate.tt_gate_weight.dtype,
        "padded",
        gate._padded_experts,
        flush=True,
    )
    st = {}
    st["matmul"] = lambda: ttnn.matmul(h, gate.tt_gate_weight, compute_kernel_config=ckc, dtype=ttnn.float32)
    logits = st["matmul"]()
    st["softplus"] = lambda: ttnn.softplus(logits, beta=1.0, threshold=20.0)
    sp = st["softplus"]()
    st["sqrt"] = lambda: ttnn.sqrt(sp)
    score = st["sqrt"]()
    st["add bias"] = lambda: ttnn.add(score, gate._tt_rank_bias)
    rank = st["add bias"]()
    st["topk"] = lambda: ttnn.topk(rank, k=k, dim=-1, largest=True, sorted=True)
    _, idx = st["topk"]()
    st["typecast+to_layout idx32"] = lambda: ttnn.to_layout(ttnn.typecast(idx, ttnn.uint32), ttnn.TILE_LAYOUT)
    idx32 = st["typecast+to_layout idx32"]()
    st["gather"] = lambda: ttnn.gather(score, 3, index=idx32)
    sel = st["gather"]()
    st["sum+add"] = lambda: ttnn.add(ttnn.sum(sel, dim=3, keepdim=True), 1e-20)
    den = st["sum+add"]()
    st["div+mul+typecast"] = lambda: ttnn.typecast(ttnn.multiply(ttnn.div(sel, den), 1.5), ttnn.bfloat16)
    st["output layout/typecasts"] = lambda: (
        ttnn.to_layout(ttnn.typecast(idx, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT),
        ttnn.to_layout(ttnn.typecast(sel, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT),
    )
    tot = 0.0
    for name, fn in st.items():
        ms = chain_ms(md, fn)
        tot += ms
        print(f"RST {name:30s} {ms * 1e3:7.1f} us", flush=True)
    print(
        f"RST sum of stages {tot * 1e3:.1f} us;  whole _forward_exact {chain_ms(md, lambda: gate._forward_exact(h)) * 1e3:.1f} us",
        flush=True,
    )

    # ---- variants of the three expensive stages
    print("RST ---- variants", flush=True)

    def var(name, fn):
        try:
            out = fn()
            print(f"RSTV {name:46s} {chain_ms(md, fn) * 1e3:7.1f} us", flush=True)
            return out
        except Exception as e:
            print(f"RSTV {name:46s} FAILED {str(e)[:110]}", flush=True)

    for gy, gx in ((8, 8), (4, 8), (2, 8)):
        var(
            f"matmul core_grid {gy}x{gx}",
            lambda: ttnn.matmul(
                h,
                gate.tt_gate_weight,
                compute_kernel_config=ckc,
                dtype=ttnn.float32,
                core_grid=ttnn.CoreGrid(y=gy, x=gx),
            ),
        )
    ckc2 = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    var(
        "matmul core_grid 8x8 HiFi2",
        lambda: ttnn.matmul(
            h, gate.tt_gate_weight, compute_kernel_config=ckc2, dtype=ttnn.float32, core_grid=ttnn.CoreGrid(y=8, x=8)
        ),
    )
    var("topk fp32 [T,512] (current)", lambda: ttnn.topk(rank, k=k, dim=-1, largest=True, sorted=True))
    var("topk fp32 slice to [T,384]", lambda: ttnn.topk(rank[:, :, :, :384], k=k, dim=-1, largest=True, sorted=True))
    var("topk fp32 [T,512] sorted=False", lambda: ttnn.topk(rank, k=k, dim=-1, largest=True, sorted=False))
    rank_bf = ttnn.typecast(rank, ttnn.bfloat16)
    var("topk bf16 [T,512] (inexact; timing only)", lambda: ttnn.topk(rank_bf, k=k, dim=-1, largest=True, sorted=True))
    ar = ttnn.from_torch(
        torch.arange(512, dtype=torch.float32).reshape(1, 1, 1, 512),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    idxf = ttnn.typecast(idx, ttnn.float32)  # [T,1,1,k]? shape printed below
    print("RST idx shape", idx.shape, "score shape", score.shape, flush=True)

    E = 384
    score384 = score[:, :, :, :E]
    ar384 = ttnn.from_torch(
        torch.arange(E, dtype=torch.float32).reshape(1, 1, 1, E),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    ref_sel = ttnn.to_torch(ttnn.get_device_tensors(ttnn.gather(score, 3, index=idx32))[0]).float()

    def onehot():
        idf = ttnn.reshape(ttnn.typecast(idx, ttnn.float32), [T, 1, idx.shape[-1], 1])  # [T,1,k,1]
        oh = ttnn.eq(idf, ar384)  # [T,1,k,E] one-hot rows
        sc = ttnn.reshape(score384, [T, 1, 1, E])
        return ttnn.sum(ttnn.multiply(oh, sc), dim=-1, keepdim=True)  # [T,1,k,1]

    out = var("one-hot select (eq, multiply, sum)", onehot)
    if out is not None:
        got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(T, -1)
        print("RST one-hot select max|diff| vs gather:", (got - ref_sel.reshape(T, -1)).abs().max().item(), flush=True)
    var("gather on sliced [T,384] score", lambda: ttnn.gather(score384, 3, index=idx32))

    # ---- the fast exact router vs the reference exact router: identical experts, same weights, timing
    wa, ia = gate._forward_exact(h)
    wb, ib = gate._forward_exact_fast(h)
    d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(T, -1)
    print(
        "RSTF experts identical:",
        bool(torch.equal(d0(ia).long(), d0(ib).long())),
        " weights max|diff|",
        (d0(wa) - d0(wb)).abs().max().item(),
        flush=True,
    )
    print(
        f"RSTF exact_ref {chain_ms(md, lambda: gate._forward_exact(h)) * 1e3:.1f} us   exact_fast {chain_ms(md, lambda: gate._forward_exact_fast(h)) * 1e3:.1f} us",
        flush=True,
    )
