# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSV41_ROUTER=exact2 (_forward_exact_fast2) vs the current exact router (_forward_exact_fast): exactness on >=10k random
rows + real router inputs, and chain timing (per stage and total) at T = 4/8/16/32 tokens per device row."""

import os
import time

import pytest
import torch

import ttnn
from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import CONFIG_PATH
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer
from models.demos.blackhole.deepseek_v41_flash.tt.router import DSV41Gate

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-e")
PARAMS = [
    pytest.param(
        {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000},
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


def make_gate(md, layer, T, shift=0.0):
    text = (
        CONFIG_PATH.read_text()
        .replace("num_shared_experts: 1", "num_shared_experts: 0")
        .replace("  shared_expert_ids_to_devices: fully_replicated\n", "")
    )
    cfg = TTMoEGateConfig.from_yaml(text).model_copy(update={"batch_per_device": T})
    w = load_moe_layer(layer, experts=[0])
    return DSV41Gate(md, cfg, torch_gate_weight=w["gate_weight"], torch_gate_bias=w["gate_bias"], bias_shift=shift), w


def up(md, x):
    T = x.shape[0]
    return ttnn.from_torch(
        x.reshape(1, 1, T, -1).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def d0(t, T):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(T, -1)


def bf16_ulps(a, b):
    """difference in units of the bf16 ulp of the larger magnitude"""
    a, b = a.float(), b.float()
    m = torch.maximum(a.abs(), b.abs()).clamp(min=1e-30)
    ulp = 2.0 ** (torch.floor(torch.log2(m)) - 7)
    return (a - b).abs() / ulp


class Acc:
    def __init__(self):
        self.rows = self.id_eq = self.set_eq = 0
        self.wmax = 0.0
        self.ulpmax = 0.0
        self.nulp = 0

    def add(self, ia, wa, ib, wb):
        self.rows += ia.shape[0]
        self.id_eq += int((ia == ib).all(-1).sum())
        self.set_eq += sum(set(a.tolist()) == set(b.tolist()) for a, b in zip(ia, ib))
        d = (wa.float() - wb.float()).abs()
        self.wmax = max(self.wmax, d.max().item())
        u = bf16_ulps(wa, wb)
        self.ulpmax = max(self.ulpmax, u.max().item())
        self.nulp += int((u > 0.5).sum())

    def __str__(self):
        return (
            f"rows {self.rows} ids identical (ordered) {self.id_eq} same set {self.set_eq} | weights max|diff| {self.wmax:.3e} "
            f"max {self.ulpmax:.2f} bf16-ulp, #elements differing {self.nulp}"
        )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_router_v2_exact(mesh_device):
    md = mesh_device
    gate, w = make_gate(md, 2, 32)
    torch.manual_seed(0)
    tot = Acc()
    for T, iters in ((32, 330), (16, 40), (8, 40), (4, 40)):
        acc, acc_x = Acc(), Acc()
        for i in range(iters):
            scale = (0.25, 0.5, 1.0, 2.0, 4.0)[i % 5]
            x = torch.randn(T, 5120) * scale
            if i % 7 == 0:
                x = x * (torch.rand(1, 5120) < 0.05) * 8 + x * 0.3  # sparse outlier channels like real hidden states
            h = up(md, x)
            wa, ia = gate._forward_exact_fast(h)
            wb, ib = gate._forward_exact_fast2(h)
            acc.add(d0(ia, T).long(), d0(wa, T), d0(ib, T).long(), d0(wb, T))
            if i % 4 == 0:
                wc, ic = gate._forward_exact_fast2(h, exact_select=True)
                acc_x.add(d0(ia, T).long(), d0(wa, T), d0(ic, T).long(), d0(wc, T))
            ttnn.deallocate(h)
        print(f"V2EXACT random T={T}: {acc}\nV2EXACT random T={T} exact_select=True: {acc_x}", flush=True)
        tot.rows += acc.rows
        tot.id_eq += acc.id_eq
        tot.set_eq += acc.set_eq
        tot.wmax = max(tot.wmax, acc.wmax)
        tot.ulpmax = max(tot.ulpmax, acc.ulpmax)
        tot.nulp += acc.nulp
    print(f"V2EXACT random TOTAL: {tot}", flush=True)
    assert tot.id_eq == tot.rows
    # vs. the old reference path (gather based) once, T=32
    acc = Acc()
    for i in range(20):
        h = up(md, torch.randn(32, 5120) * (0.5 + i % 4))
        wa, ia = gate._forward_exact(h)
        wb, ib = gate._forward_exact_fast2(h)
        acc.add(d0(ia, 32).long(), d0(wa, 32), d0(ib, 32).long(), d0(wb, 32))
    print(f"V2EXACT vs exact_ref (gather, padded 512): {acc}", flush=True)
    # real router inputs (16 tokens per layer), with the layer's calibrated shift like the real model
    for L in (1, 2, 8, 14, 20):
        p = os.path.join(CHAIN, f"ffn_inputs_{L}.pt")
        if not os.path.exists(p):
            continue
        d = torch.load(p)
        x = d["x"].reshape(16, 5120)
        wl = load_moe_layer(L, experts=[0])
        W, b = wl["gate_weight"].float(), wl["gate_bias"].float()
        rk = torch.nn.functional.softplus(x.float() @ (W.T if W.shape[0] == 384 else W)).sqrt() + b
        for shift in (0.0, float(rk.topk(7, -1).values[:, 5].mean())):
            g, _ = make_gate(md, L, 16, shift)
            h = up(md, x)
            wa, ia = g._forward_exact_fast(h)
            wb, ib = g._forward_exact_fast2(h)
            acc = Acc()
            acc.add(d0(ia, 16).long(), d0(wa, 16), d0(ib, 16).long(), d0(wb, 16))
            ref_ok = sum(
                set(a.tolist()) == set(c.tolist()) for a, c in zip(d0(ib, 16).long(), d["idx"].reshape(16, -1).long())
            )
            print(
                f"V2EXACT real layer {L} shift {shift:.3f}: {acc} | same set as checkpoint gate {ref_ok}/16", flush=True
            )
            assert acc.id_eq == 16


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_router_v2_timing(mesh_device):
    md = mesh_device
    U, UT = ttnn.UnaryWithParam, ttnn.UnaryOpType
    L1 = ttnn.L1_MEMORY_CONFIG
    for T in (4, 8, 16, 32):
        gate, _ = make_gate(md, 2, T)
        h = up(md, torch.randn(T, 5120))
        p = lambda n, fn: print(f"V2T T={T:2d} {n:46s} {chain_ms(md, fn) * 1e3:7.1f} us", flush=True)
        p("CURRENT _forward_exact_fast", lambda: gate._forward_exact_fast(h))
        p("NEW _forward_exact_fast2 (default)", lambda: gate._forward_exact_fast2(h))
        p("NEW exact_select=True (bit-identical weights)", lambda: gate._forward_exact_fast2(h, exact_select=True))
        p("NEW l1=False", lambda: gate._forward_exact_fast2(h, l1=False))
        p("NEW grid 1x8", lambda: gate._forward_exact_fast2(h, grid=(1, 8)))
        # logits of the other grids are bit-identical to the 2x8 ones?
        mm = lambda g: ttnn.to_torch(
            ttnn.get_device_tensors(
                ttnn.matmul(
                    h,
                    gate._w_exact,
                    compute_kernel_config=gate._ckc_exact,
                    dtype=ttnn.float32,
                    core_grid=ttnn.CoreGrid(y=g[0], x=g[1]),
                )
            )[0]
        )
        print(f"V2T T={T:2d} logits 1x8 vs 2x8 bit-identical: {bool(torch.equal(mm((1, 8)), mm((2, 8))))}", flush=True)
        # per-stage (dependent chain: each stage is timed alone, repeated, so tiny ops read low; the totals above are what counts)
        k, E = gate.k, gate._E
        acts = [U(UT.SOFTPLUS, 1.0, 20.0), U(UT.SQRT)]
        st = {}
        st["matmul 2x8 -> L1"] = lambda: ttnn.matmul(
            h,
            gate._w_exact,
            compute_kernel_config=gate._ckc_exact,
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=2, x=8),
            memory_config=L1,
        )
        logits = st["matmul 2x8 -> L1"]()
        st["score = add(logits, 0, [softplus, sqrt])"] = lambda: ttnn.add(
            logits, gate._zero_exact, input_tensor_a_activations=acts, memory_config=L1
        )
        score = st["score = add(logits, 0, [softplus, sqrt])"]()
        st["rank = add(score, bias)"] = lambda: ttnn.add(score, gate._bias_exact, memory_config=L1)
        rank = st["rank = add(score, bias)"]()
        st["topk fp32"] = lambda: ttnn.topk(rank, k=k, dim=-1, largest=True, sorted=True, memory_config=L1)
        _, idx = st["topk fp32"]()
        st["typecast idx f32 + reshape + eq one-hot"] = lambda: ttnn.eq(
            gate._arange_col,
            ttnn.reshape(ttnn.typecast(idx, ttnn.float32, memory_config=L1), [T, 1, 1, k]),
            memory_config=L1,
        )
        oh = st["typecast idx f32 + reshape + eq one-hot"]()
        st["reshape score + select matmul"] = lambda: ttnn.matmul(
            ttnn.reshape(score, [T, 1, 1, E]),
            oh,
            compute_kernel_config=gate._ckc_exact,
            dtype=ttnn.float32,
            memory_config=L1,
        )
        sel = st["reshape score + select matmul"]()
        st["den matmul"] = lambda: ttnn.matmul(
            sel, gate._ones_k, compute_kernel_config=gate._ckc_exact, dtype=ttnn.float32, memory_config=L1
        )
        den = st["den matmul"]()
        st["fused div (+eps, *scale, bf16)"] = lambda: ttnn.div(
            sel,
            den,
            input_tensor_b_activations=[U(UT.ADD_UNARY_SFPU, 1e-20)],
            activations=[U(UT.MUL_UNARY_SFPU, 1.5)],
            dtype=ttnn.bfloat16,
            memory_config=L1,
        )
        w = st["fused div (+eps, *scale, bf16)"]()
        st["outputs (to_layout w, typecast+to_layout idx)"] = lambda: (
            ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT),
            ttnn.to_layout(ttnn.typecast(idx, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT),
        )
        tot = 0
        for n, fn in st.items():
            ms = chain_ms(md, fn)
            tot += ms
            print(f"V2S T={T:2d} {n:46s} {ms * 1e3:7.1f} us", flush=True)
        print(f"V2S T={T:2d} sum of stages {tot * 1e3:.1f} us", flush=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_router_v2_micro(mesh_device):
    """Exploration: cost of the individual pieces and their alternatives (what did / did not help)."""
    md = mesh_device
    U, UT = ttnn.UnaryWithParam, ttnn.UnaryOpType
    L1 = ttnn.L1_MEMORY_CONFIG
    rep = ttnn.ReplicateTensorToMesh(md)
    for T in (4, 32):
        gate, _ = make_gate(md, 2, T)
        k, E = gate.k, gate._E
        h = up(md, torch.randn(T, 5120))
        mm = lambda **kw: ttnn.matmul(h, gate._w_exact, compute_kernel_config=gate._ckc_exact, dtype=ttnn.float32, **kw)

        def var(name, fn):
            try:
                fn()
                print(f"V2M T={T:2d} {name:56s} {chain_ms(md, fn) * 1e3:7.1f} us", flush=True)
            except Exception as e:
                print(f"V2M T={T:2d} {name:56s} FAILED {str(e)[:100]}", flush=True)

        for gy, gx in ((2, 8), (1, 8), (1, 12), (2, 6), (4, 4), (3, 8), (2, 12)):
            var(f"matmul core_grid {gy}x{gx} L1 out", lambda: mm(core_grid=ttnn.CoreGrid(y=gy, x=gx), memory_config=L1))
        logits = mm(core_grid=ttnn.CoreGrid(y=2, x=8), memory_config=L1)
        zero = ttnn.from_torch(
            torch.zeros(1, 1, 1, E),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        acts = [U(UT.SOFTPLUS, 1.0, 20.0), U(UT.SQRT)]
        var(
            "score = add(logits, 0, [softplus, sqrt]) fused",
            lambda: ttnn.add(logits, zero, input_tensor_a_activations=acts, memory_config=L1),
        )
        var(
            "score = sqrt(softplus(logits)) unary x2",
            lambda: ttnn.sqrt(ttnn.softplus(logits, beta=1.0, threshold=20.0, memory_config=L1), memory_config=L1),
        )
        var(
            "score = multiply(logits, 1, [softplus, sqrt]) scalar",
            lambda: ttnn.multiply(logits, 1.0, input_tensor_a_activations=acts, memory_config=L1),
        )
        score = ttnn.add(logits, zero, input_tensor_a_activations=acts, memory_config=L1)
        rank = ttnn.add(score, gate._bias_exact, memory_config=L1)
        var("rank = add(score, bias) L1", lambda: ttnn.add(score, gate._bias_exact, memory_config=L1))
        var("rank = add(score, bias) post-act none, DRAM", lambda: ttnn.add(score, gate._bias_exact))
        var("topk L1 out", lambda: ttnn.topk(rank, k=k, dim=-1, largest=True, sorted=True, memory_config=L1))
        var("topk DRAM out", lambda: ttnn.topk(rank, k=k, dim=-1, largest=True, sorted=True))
        _, idx = ttnn.topk(rank, k=k, dim=-1, largest=True, sorted=True, memory_config=L1)
        print("V2M idx dtype", idx.dtype, idx.shape, idx.layout, flush=True)
        # --- index plumbing
        var(
            "A typecast f32 + reshape [T,1,k,1]",
            lambda: ttnn.reshape(ttnn.typecast(idx, ttnn.float32, memory_config=L1), [T, 1, k, 1]),
        )
        var("A' reshape [T,1,k,1] only (uint)", lambda: ttnn.reshape(idx, [T, 1, k, 1]))
        var("B to_layout RM (uint)", lambda: ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT))
        idx_rm = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT)
        var("B' RM idx: typecast f32 RM", lambda: ttnn.typecast(idx_rm, ttnn.float32))
        var(
            "B'' RM idx: view [T,1,k,1] + to_layout TILE",
            lambda: ttnn.to_layout(ttnn.reshape(idx_rm, [T, 1, k, 1]), ttnn.TILE_LAYOUT),
        )
        var("C to_layout(idx, RM, dtype=uint16)", lambda: ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16))
        var(
            "C' typecast uint16 + to_layout RM (current)",
            lambda: ttnn.to_layout(ttnn.typecast(idx, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT),
        )
        var("C'' RM idx -> typecast uint16", lambda: ttnn.typecast(idx_rm, ttnn.uint16))
        idsel = ttnn.reshape(ttnn.typecast(idx, ttnn.float32, memory_config=L1), [T, 1, k, 1])
        # --- select
        var("eq one-hot [T,1,k,E]", lambda: ttnn.eq(idsel, gate._arange, memory_config=L1))
        onehot = ttnn.eq(idsel, gate._arange, memory_config=L1)
        s3 = ttnn.reshape(score, [T, 1, 1, E])
        var("reshape score [T,1,1,E]", lambda: ttnn.reshape(score, [T, 1, 1, E]))
        var("multiply one-hot*score", lambda: ttnn.multiply(onehot, s3, memory_config=L1))
        prod = ttnn.multiply(onehot, s3, memory_config=L1)
        var("sum dim=-1 keepdim", lambda: ttnn.sum(prod, dim=-1, keepdim=True, memory_config=L1))
        var("sum dim=-1 no keepdim", lambda: ttnn.sum(prod, dim=-1, memory_config=L1))
        var("fused eq*score: where(onehot, score, 0)?", lambda: ttnn.where(onehot, s3, 0.0))
        var(
            "matmul onehot[T,1,k,E] @ score^T [T,1,E,1] fp32",
            lambda: ttnn.matmul(
                onehot,
                ttnn.reshape(score, [T, 1, E, 1]),
                compute_kernel_config=gate._ckc_exact,
                dtype=ttnn.float32,
                memory_config=L1,
            ),
        )
        sel = ttnn.sum(prod, dim=-1, keepdim=True, memory_config=L1)  # [T,1,k,1]
        # --- den / tail
        var("den sum dim=2 keepdim", lambda: ttnn.sum(sel, dim=2, keepdim=True, memory_config=L1))
        var("den sum dim=2 (no keepdim)", lambda: ttnn.sum(sel, dim=2, memory_config=L1))
        var("den mean dim=2 keepdim", lambda: ttnn.mean(sel, dim=2, keepdim=True, memory_config=L1))
        ones = ttnn.from_torch(
            torch.ones(1, 1, k, k),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        selr = ttnn.reshape(sel, [T, 1, 1, k])
        var(
            "den matmul sel[T,1,1,k] @ ones[k,k] fp32",
            lambda: ttnn.matmul(
                selr, ones, compute_kernel_config=gate._ckc_exact, dtype=ttnn.float32, memory_config=L1
            ),
        )
        var("den sum dim=-1 on [T,1,1,k]", lambda: ttnn.sum(selr, dim=-1, keepdim=True, memory_config=L1))
        var("reshape sel -> [T,1,1,k]", lambda: ttnn.reshape(sel, [T, 1, 1, k]))
        den = ttnn.sum(sel, dim=2, keepdim=True, memory_config=L1)
        var(
            "div fused",
            lambda: ttnn.div(
                sel,
                den,
                input_tensor_b_activations=[U(UT.ADD_UNARY_SFPU, 1e-20)],
                activations=[U(UT.MUL_UNARY_SFPU, 1.5)],
                dtype=ttnn.bfloat16,
                memory_config=L1,
            ),
        )
        w = ttnn.div(
            sel,
            den,
            input_tensor_b_activations=[U(UT.ADD_UNARY_SFPU, 1e-20)],
            activations=[U(UT.MUL_UNARY_SFPU, 1.5)],
            dtype=ttnn.bfloat16,
            memory_config=L1,
        )
        var("w to_layout RM", lambda: ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT))


def _variants(gate, md):
    """whole-method variants (exploration): name -> fn(h) -> (weights, indices)"""
    U, UT = ttnn.UnaryWithParam, ttnn.UnaryOpType
    L1 = ttnn.L1_MEMORY_CONFIG
    k, E = gate.k, gate._E
    rep = ttnn.ReplicateTensorToMesh(md)
    zero = ttnn.from_torch(
        torch.zeros(1, 1, 1, E),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    ones_k = ttnn.from_torch(
        torch.ones(1, 1, k, k),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    acol = ttnn.from_torch(
        torch.arange(E, dtype=torch.float32).reshape(1, 1, E, 1),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    acts = [U(UT.SOFTPLUS, 1.0, 20.0), U(UT.SQRT)]
    scale = gate.scaling_factor
    ck = gate._ckc_exact

    def front(h, full_score):
        logits = ttnn.matmul(
            h,
            gate._w_exact,
            compute_kernel_config=ck,
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=2, x=8),
            memory_config=L1,
        )
        if full_score:
            score = ttnn.add(logits, zero, input_tensor_a_activations=acts, memory_config=L1)
            rank = ttnn.add(score, gate._bias_exact, memory_config=L1)
        else:
            score = None
            rank = ttnn.add(logits, gate._bias_exact, input_tensor_a_activations=acts, memory_config=L1)
        _, idx = ttnn.topk(rank, k=k, dim=-1, largest=True, sorted=True, memory_config=L1)
        return logits, score, idx

    def idx_out(idx, T):
        return ttnn.view(ttnn.to_layout(ttnn.typecast(idx, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT), (T, 1, 1, k))

    def tail_T(sel, T, idx):  # sel [T,1,k,1] fp32 score
        den = ttnn.sum(sel, dim=2, keepdim=True, memory_config=L1)
        w = ttnn.div(
            sel,
            den,
            input_tensor_b_activations=[U(UT.ADD_UNARY_SFPU, 1e-20)],
            activations=[U(UT.MUL_UNARY_SFPU, scale)],
            dtype=ttnn.bfloat16,
            memory_config=L1,
        )
        return ttnn.reshape(ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT), (T, 1, 1, k)), idx_out(idx, T)

    def vA(h):  # orientation T, full score fused (no softplus/sqrt on small)
        T = h.shape[2]
        _, score, idx = front(h, True)
        oh = ttnn.eq(
            ttnn.reshape(ttnn.typecast(idx, ttnn.float32, memory_config=L1), [T, 1, k, 1]),
            gate._arange,
            memory_config=L1,
        )
        sel = ttnn.sum(
            ttnn.multiply(oh, ttnn.reshape(score, [T, 1, 1, E]), memory_config=L1),
            dim=-1,
            keepdim=True,
            memory_config=L1,
        )
        return tail_T(sel, T, idx)

    def vB(h):  # orientation k: onehot [1,k,T,E], no score reshape
        T = h.shape[2]
        _, score, idx = front(h, True)
        idk = ttnn.permute(
            ttnn.typecast(idx, ttnn.float32, memory_config=L1), (0, 3, 2, 1), memory_config=L1
        )  # [1,k,T,1]
        oh = ttnn.eq(idk, gate._arange, memory_config=L1)  # [1,k,T,E]
        sel = ttnn.sum(ttnn.multiply(oh, score, memory_config=L1), dim=-1, keepdim=True, memory_config=L1)  # [1,k,T,1]
        den = ttnn.sum(sel, dim=1, keepdim=True, memory_config=L1)  # [1,1,T,1]
        w = ttnn.div(
            sel,
            den,
            input_tensor_b_activations=[U(UT.ADD_UNARY_SFPU, 1e-20)],
            activations=[U(UT.MUL_UNARY_SFPU, scale)],
            dtype=ttnn.bfloat16,
            memory_config=L1,
        )
        wt = (
            ttnn.permute(w, (2, 3, 1, 0)) if False else ttnn.permute(w, (2, 1, 3, 0))
        )  # [T,k,1,1]?? placeholder, fixed below
        return wt, idx_out(idx, T)

    def vB2(h):  # like vB, weights via transpose to [1,1,T,k] then RM view
        T = h.shape[2]
        _, score, idx = front(h, True)
        idk = ttnn.permute(ttnn.typecast(idx, ttnn.float32, memory_config=L1), (0, 3, 2, 1), memory_config=L1)
        oh = ttnn.eq(idk, gate._arange, memory_config=L1)
        sel = ttnn.sum(ttnn.multiply(oh, score, memory_config=L1), dim=-1, keepdim=True, memory_config=L1)  # [1,k,T,1]
        den = ttnn.sum(sel, dim=1, keepdim=True, memory_config=L1)
        w = ttnn.div(
            sel,
            den,
            input_tensor_b_activations=[U(UT.ADD_UNARY_SFPU, 1e-20)],
            activations=[U(UT.MUL_UNARY_SFPU, scale)],
            dtype=ttnn.bfloat16,
            memory_config=L1,
        )
        wt = ttnn.permute(w, (0, 3, 2, 1), memory_config=L1)  # [1,1,T,k]
        return ttnn.view(ttnn.to_layout(wt, ttnn.ROW_MAJOR_LAYOUT), (T, 1, 1, k)), idx_out(idx, T)

    def vC(h):  # orientation T, matmul select: score[T,1,1,E] @ onehotT[T,1,E,k]
        T = h.shape[2]
        _, score, idx = front(h, True)
        idr = ttnn.reshape(ttnn.typecast(idx, ttnn.float32, memory_config=L1), [T, 1, 1, k])
        oh = ttnn.eq(acol, idr, memory_config=L1)  # [T,1,E,k]
        sel = ttnn.matmul(
            ttnn.reshape(score, [T, 1, 1, E]), oh, compute_kernel_config=ck, dtype=ttnn.float32, memory_config=L1
        )  # [T,1,1,k]
        den = ttnn.matmul(sel, ones_k, compute_kernel_config=ck, dtype=ttnn.float32, memory_config=L1)
        w = ttnn.div(
            sel,
            den,
            input_tensor_b_activations=[U(UT.ADD_UNARY_SFPU, 1e-20)],
            activations=[U(UT.MUL_UNARY_SFPU, scale)],
            dtype=ttnn.bfloat16,
            memory_config=L1,
        )
        return ttnn.reshape(ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT), (T, 1, 1, k)), idx_out(idx, T)

    def vD(h):  # orientation T, mult+sum select but on [T,1,E,k] layout, sum over dim 2 -> [T,1,1,k], den = sum(dim=-1)
        T = h.shape[2]
        _, score, idx = front(h, True)
        idr = ttnn.reshape(ttnn.typecast(idx, ttnn.float32, memory_config=L1), [T, 1, 1, k])
        oh = ttnn.eq(acol, idr, memory_config=L1)
        sel = ttnn.sum(
            ttnn.multiply(oh, ttnn.reshape(score, [T, 1, E, 1]), memory_config=L1),
            dim=2,
            keepdim=True,
            memory_config=L1,
        )
        den = ttnn.sum(sel, dim=-1, keepdim=True, memory_config=L1)
        w = ttnn.div(
            sel,
            den,
            input_tensor_b_activations=[U(UT.ADD_UNARY_SFPU, 1e-20)],
            activations=[U(UT.MUL_UNARY_SFPU, scale)],
            dtype=ttnn.bfloat16,
            memory_config=L1,
        )
        return ttnn.reshape(ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT), (T, 1, 1, k)), idx_out(idx, T)

    acol_u = ttnn.from_torch(
        torch.arange(E, dtype=torch.int32).reshape(1, 1, E, 1),
        device=md,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )

    def vE(h):  # vC with the uint32 index compared directly (no typecast to fp32), eq output fp32
        T = h.shape[2]
        _, score, idx = front(h, True)
        oh = ttnn.eq(acol_u, ttnn.reshape(idx, [T, 1, 1, k]), dtype=ttnn.float32, memory_config=L1)
        sel = ttnn.matmul(
            ttnn.reshape(score, [T, 1, 1, E]), oh, compute_kernel_config=ck, dtype=ttnn.float32, memory_config=L1
        )
        den = ttnn.sum(sel, dim=-1, keepdim=True, memory_config=L1)
        w = ttnn.div(
            sel,
            den,
            input_tensor_b_activations=[U(UT.ADD_UNARY_SFPU, 1e-20)],
            activations=[U(UT.MUL_UNARY_SFPU, scale)],
            dtype=ttnn.bfloat16,
            memory_config=L1,
        )
        return ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT), idx_out(idx, T)

    def vF(h):  # vC with den = sum(dim=-1)
        T = h.shape[2]
        _, score, idx = front(h, True)
        idr = ttnn.reshape(ttnn.typecast(idx, ttnn.float32, memory_config=L1), [T, 1, 1, k])
        oh = ttnn.eq(acol, idr, memory_config=L1)
        sel = ttnn.matmul(
            ttnn.reshape(score, [T, 1, 1, E]), oh, compute_kernel_config=ck, dtype=ttnn.float32, memory_config=L1
        )
        den = ttnn.sum(sel, dim=-1, keepdim=True, memory_config=L1)
        w = ttnn.div(
            sel,
            den,
            input_tensor_b_activations=[U(UT.ADD_UNARY_SFPU, 1e-20)],
            activations=[U(UT.MUL_UNARY_SFPU, scale)],
            dtype=ttnn.bfloat16,
            memory_config=L1,
        )
        return ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT), idx_out(idx, T)

    return {
        "vA full-score T-orient": vA,
        "vB2 k-orient": vB2,
        "vC matmul select": vC,
        "vE u32 eq + sum den": vE,
        "vF den sum": vF,
    }


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_router_v2_variants(mesh_device):
    md = mesh_device
    torch.manual_seed(1)
    for T in (4, 8, 16, 32):
        gate, _ = make_gate(md, 2, T)
        h = up(md, torch.randn(T, 5120))
        wa, ia = gate._forward_exact_fast(h)
        print(
            f"V2V T={T:2d} {'CURRENT':28s} {chain_ms(md, lambda: gate._forward_exact_fast(h)) * 1e3:7.1f} us",
            flush=True,
        )
        print(
            f"V2V T={T:2d} {'method default':28s} {chain_ms(md, lambda: gate._forward_exact_fast2(h)) * 1e3:7.1f} us",
            flush=True,
        )
        for name, fn in _variants(gate, md).items():
            try:
                wb, ib = fn(h)
                acc = Acc()
                acc.add(d0(ia, T).long(), d0(wa, T), d0(ib, T).long(), d0(wb, T))
                print(f"V2V T={T:2d} {name:28s} {chain_ms(md, lambda: fn(h)) * 1e3:7.1f} us   {acc}", flush=True)
            except Exception as e:
                print(f"V2V T={T:2d} {name:28s} FAILED {str(e)[:300]}", flush=True)
