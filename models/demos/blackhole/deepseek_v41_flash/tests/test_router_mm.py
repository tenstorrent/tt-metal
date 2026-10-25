# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""router_mm (split-K matmul + score + bias, one JIT program) vs the ttnn chain of DSV41Gate._forward_fused.

test_router_mm_exact: rank / score bit-compare and expert-id comparison (through router_select) per S on random rows,
identical rows, and the real FFN inputs of layers 1/2/8/14/20 (T = 4, 8 padded, 16)."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_router_v2 import CHAIN, chain_ms, d0, make_gate, up
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer
from models.demos.blackhole.deepseek_v41_flash.tt.router_mm import router_fused
from models.demos.blackhole.deepseek_v41_flash.tt.router_select import router_select

SS = [int(v) for v in os.environ.get("RMM_S", "1,2,4,5,8").split(",")]
PARAMS = [pytest.param({"l1_small_size": 16384, "trace_region_size": 200_000_000}, id="plain")]


def old_chain(gate, h):
    """the ttnn front end of _forward_fused: (rank, score) fp32 TILE"""
    L1 = ttnn.L1_MEMORY_CONFIG
    U, UT = ttnn.UnaryWithParam, ttnn.UnaryOpType
    logits = ttnn.matmul(
        h,
        gate._w_exact,
        compute_kernel_config=gate._ckc_exact,
        dtype=ttnn.float32,
        core_grid=ttnn.CoreGrid(y=1, x=8),
        memory_config=L1,
    )
    score = ttnn.add(
        logits, gate._zero_exact, input_tensor_a_activations=[U(UT.SOFTPLUS, 1.0, 20.0), U(UT.SQRT)], memory_config=L1
    )
    rank = ttnn.add(score, gate._bias_exact, memory_config=L1)
    return rank, score


def rows(t, T):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1, 384)[:T].float()


def bits_eq(a, b):
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


class Stat:
    def __init__(self):
        self.rows = self.rank_bad = self.score_bad = self.id_bad = self.set_bad = self.w_bad = 0
        self.rank_absmax = 0.0

    def add(self, T, ro, so, rn, sn, io, wo, i_n, wn):
        self.rows += T
        self.rank_bad += bits_eq(ro, rn)
        self.score_bad += bits_eq(so, sn)
        self.rank_absmax = max(self.rank_absmax, float((ro - rn).abs().max()))
        self.w_bad += int((wo.float().view(torch.int32) != wn.float().view(torch.int32)).sum())
        io, i_n = io.long(), i_n.long()
        self.id_bad += int((io != i_n).any(-1).sum())
        self.set_bad += sum(set(a.tolist()) != set(b.tolist()) for a, b in zip(io, i_n))

    def __str__(self):
        return (
            f"rows {self.rows}: rank elems differing {self.rank_bad} (max|d| {self.rank_absmax:.2e}), score elems "
            f"differing {self.score_bad}, weights differing {self.w_bad}, rows with different ordered ids {self.id_bad}, different SETS {self.set_bad}"
        )


def run_pair(gate, h, S, T, st):
    wk, bt = gate._mm_consts(S)
    ro, so = old_chain(gate, h)
    wo, io = router_select(ro, so, gate.k, 1e-20, gate.scaling_factor)
    wn, i_n, rn, sn = router_fused(h, wk, bt, S, gate.k, 1e-20, gate.scaling_factor, dump=True)
    st.add(T, rows(ro, T), rows(so, T), rows(rn, T), rows(sn, T), d0(io, T), d0(wo, T), d0(i_n, T), d0(wn, T))
    for t in (ro, so, rn, sn, wo, io, wn, i_n):
        ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_router_mm_exact(mesh_device):
    md = mesh_device
    torch.manual_seed(3)
    layers = [int(v) for v in os.environ.get("RMM_LAYERS", "2,14").split(",")]
    n_rand = int(os.environ.get("RMM_NRAND", "60"))
    for S in SS:
        tot = Stat()
        for L in layers:
            gate, _ = make_gate(md, L, 32)
            for T in (4, 8, 16, 32):
                st = Stat()
                for i in range(n_rand):
                    x = torch.randn(T, 5120) * (0.25, 0.5, 1.0, 2.0, 4.0)[i % 5]
                    if i % 7 == 0:
                        x = x * (torch.rand(1, 5120) < 0.05) * 8 + x * 0.3
                    if i % 11 == 0:
                        x[T // 2] = x[0]
                    h = up(md, x)
                    run_pair(gate, h, S, T, st)
                    ttnn.deallocate(h)
                print(f"RMM S={S} layer {L} random T={T}: {st}", flush=True)
                for a in ("rows", "rank_bad", "score_bad", "id_bad", "set_bad", "w_bad"):
                    setattr(tot, a, getattr(tot, a) + getattr(st, a))
                tot.rank_absmax = max(tot.rank_absmax, st.rank_absmax)
        print(f"RMM S={S} RANDOM TOTAL: {tot}", flush=True)
        # real FFN inputs
        real = Stat()
        for L in (1, 2, 8, 14, 20):
            p = os.path.join(CHAIN, f"ffn_inputs_{L}.pt")
            if not os.path.exists(p):
                continue
            x = torch.load(p)["x"].reshape(16, 5120)
            wl = load_moe_layer(L, experts=[0])
            gate, _ = make_gate(
                md,
                L,
                16,
                float(
                    (
                        torch.nn.functional.softplus(x.float() @ wl["gate_weight"].float()).sqrt()
                        + wl["gate_bias"].float()
                    )
                    .topk(7, -1)
                    .values[:, 5]
                    .mean()
                ),
            )
            for T in (4, 8, 16):
                xx = x[:T]
                h = up(md, xx)
                run_pair(gate, h, S, T, real)
                ttnn.deallocate(h)
        print(f"RMM S={S} REAL TOTAL: {real}", flush=True)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_router_mm_timing(mesh_device):
    md = mesh_device
    gate, _ = make_gate(md, 2, 8)
    us = lambda fn: chain_ms(md, fn) * 1e3
    for T in (4, 8, 16):
        h = up(md, torch.randn(T, 5120))
        print(f"RMMT T={T:2d} old front end (mm+2 add) {us(lambda: old_chain(gate, h)):7.1f} us", flush=True)
        print(f"RMMT T={T:2d} old full (fused)         {us(lambda: gate._forward_fused(h)):7.1f} us", flush=True)
        for S in SS:
            wk, bt = gate._mm_consts(S)
            fu = lambda l1: router_fused(h, wk, bt, S, 6, 1e-20, 1.5, out_l1=l1)
            print(f"RMMT T={T:2d} S={S} FUSED (dram out)   {us(lambda: fu(False)):7.1f} us", flush=True)
            print(f"RMMT T={T:2d} S={S} FUSED (L1 out)     {us(lambda: fu(True)):7.1f} us", flush=True)
