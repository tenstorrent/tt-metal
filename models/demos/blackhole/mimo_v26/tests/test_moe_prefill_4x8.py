# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase B (prefill style): dispatch -> unified_routed_expert_moe (or moe_fused_swiglu) -> combine -> post_combine_reduce at MiMo dims
(256 experts top-8, 8/chip, K=4096, N=2048) with bfp8 weights on the 4x8 mesh, against a dense torch MoE (bf16 weights, fp32 math, plain SiLU, no clamp).
Inputs are synthetic (random hidden states, random gate): this tests the data path and the expert/chip mapping per expert, NOT routing quality or skew
(see test_router_agreement.py for real tokens). Env: MIMO_UM_N tokens per mesh row (512), MIMO_UM_FID (hifi2|lofi), MIMO_UM_OP (unified|fused), MIMO_UM_CF (4),
MIMO_UM_SKEW (0|1: route a fraction of tokens to one fixed expert set to stress capacity).
"""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tt.moe_prefill import (
    COLS,
    EPC,
    HIDDEN,
    INTER,
    NUM_ROUTED,
    ROWS,
    TOPK,
    MiMoPrefillMoE,
    build_expert_weights,
    global_expert,
    make_ckc,
)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_moe_prefill_4x8(mesh_device):
    md = mesh_device
    NS = [int(v) for v in os.environ.get("MIMO_UM_NS", os.environ.get("MIMO_UM_N", "512")).split(",")]
    RUNS = int(os.environ.get("MIMO_UM_RUNS", "2"))
    fid = os.environ.get("MIMO_UM_FID", "hifi2")
    op = os.environ.get("MIMO_UM_OP", "unified")
    cf = int(os.environ.get("MIMO_UM_CF", "4"))
    skew = os.environ.get("MIMO_UM_SKEW", "0") == "1"
    torch.manual_seed(0)
    W = {}

    def w_of(e):
        if e not in W:
            g = torch.Generator().manual_seed(1000 + e)
            W[e] = (
                (torch.randn(HIDDEN, INTER, generator=g) * 0.02).bfloat16(),
                (torch.randn(HIDDEN, INTER, generator=g) * 0.02).bfloat16(),
                (torch.randn(INTER, HIDDEN, generator=g) * 0.02).bfloat16(),
            )
        return W[e]

    def dram_alloc_mb():
        try:
            mv = ttnn.get_memory_view(md.get_devices()[0], ttnn.BufferType.DRAM)
            return (
                mv.total_bytes_allocated_per_bank * mv.num_banks / 2**20
                if hasattr(mv, "num_banks")
                else mv.total_bytes_allocated_per_bank / 2**20
            )
        except Exception as ex:  # API differs between builds
            return float("nan") if not str(ex) else -1.0

    m0 = dram_alloc_mb()
    t0 = time.time()
    weights = build_expert_weights(md, w_of)
    print(
        f"RESULT dram weights of 8 experts/chip, 1 layer: {dram_alloc_mb() - m0:.1f} MB per chip (alloc view, may be per-bank dependent)",
        flush=True,
    )
    print(f"RESULT weights built {time.time() - t0:.1f}s ({NUM_ROUTED} experts, bfp8)", flush=True)
    um = MiMoPrefillMoE(md, weights, ckc=make_ckc(fid, packer_l1_acc=(op != "fused")), expert_op=op, cf=cf)

    for N in NS:
        x = torch.randn(ROWS, N, HIDDEN).bfloat16()
        gw = torch.randn(NUM_ROUTED, HIDDEN) * 0.02
        bias = torch.randn(NUM_ROUTED) * 0.03 + 2.0
        s = (x.float().reshape(-1, HIDDEN) @ gw.T).sigmoid()
        rank = s + bias
        if skew:  # a third of the tokens prefer experts 0..7 (all on column 0): stress per-expert capacity
            rank[torch.arange(rank.shape[0]) % 3 == 0, :8] += 10.0
        idx = torch.topk(rank, TOPK, dim=-1).indices
        sel = s.gather(1, idx)
        wgt = (sel / (sel.sum(-1, keepdim=True) + 1e-20)).bfloat16()
        idx_r, wgt_r = idx.reshape(ROWS, N, TOPK), wgt.reshape(ROWS, N, TOPK)
        print(
            f"RESULT max tokens/expert (all rows) {int(torch.bincount(idx.flatten(), minlength=NUM_ROUTED).max())} of {ROWS * N}",
            flush=True,
        )

        m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
        xd = ttnn.from_torch(
            x.reshape(ROWS, 1, N, HIDDEN),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=m2,
        )
        x_rm = ttnn.reshape(xd, [1, N, HIDDEN])
        id_ = ttnn.from_torch(
            idx_r.reshape(ROWS, N, 1, 1, TOPK).to(torch.int32),
            device=md,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=m2,
        )
        wd = ttnn.from_torch(
            wgt_r.reshape(ROWS, N, 1, 1, TOPK),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=m2,
        )
        id_ = ttnn.reshape(id_, [N, 1, 1, TOPK])
        wd = ttnn.reshape(wd, [N, 1, 1, TOPK])

        def run():
            return um.forward(x_rm, wd, id_, keep_counts=True)

        outs = []
        for it in range(RUNS):  # run 0 compiles; RUNS bit-identical runs = determinism gate
            part = run()
            ttnn.synchronize_device(md)
            devs = ttnn.get_device_tensors(part)
            host = torch.stack(
                [
                    torch.stack([ttnn.to_torch(devs[r * COLS + c]).reshape(N, HIDDEN).float() for c in range(COLS)])
                    for r in range(ROWS)
                ]
            )
            outs.append(host)
            counts = ttnn.get_device_tensors(um.last_counts)
            sh = um.shared(N)
            worst = 0
            for r in range(ROWS):
                for c in range(COLS):
                    cnt = ttnn.to_torch(counts[r * COLS + c]).reshape(-1)[:NUM_ROUTED].long()
                    load = sum(-(-int(cnt[global_expert(r, c, l)]) // 32) * 32 for l in range(EPC))
                    worst = max(worst, load)
            print(
                f"RESULT capacity: worst chip load {worst} rows of {sh.max_buf} ({'OVERFLOW' if worst > sh.max_buf else 'ok'}) cf={cf}",
                flush=True,
            )
            assert worst <= sh.max_buf, "dispatch capacity overflow"
            um.last_counts = None
        ts = []
        for _ in range(6):
            t0 = time.time()
            p_ = run()
            ttnn.synchronize_device(md)
            ts.append((time.time() - t0) * 1e3)
            ttnn.deallocate(p_)
            ttnn.deallocate(um.last_counts)
        print(
            f"RESULT time hint (eager, host-timed, not a perf claim) op={op} N={N}: step0 {ts[0]:.1f} ms, steps1-5 median {sorted(ts[1:])[2]:.1f} ms",
            flush=True,
        )
        dev_out = outs[0].sum(dim=1)  # sum the 8 column partials -> [ROWS, N, D]
        same = [bool(torch.equal(outs[0], o)) for o in outs[1:]]
        ident_rows = [all(bool(torch.equal(outs[0][r], o[r])) for o in outs[1:]) for r in range(ROWS)]
        print(
            f"RESULT determinism N={N}: {RUNS} runs, all bit-identical to run 0: {all(same)}; per mesh row {ident_rows}",
            flush=True,
        )
        assert all(same), "non-deterministic output"

        # dense reference per expert
        ref = torch.zeros(ROWS, N, HIDDEN)
        xf = x.float()
        flat_idx, flat_w = idx_r.reshape(-1, TOPK), wgt_r.reshape(-1, TOPK).float()
        xs = xf.reshape(-1, HIDDEN)
        refflat = torch.zeros(ROWS * N, HIDDEN)
        for e in range(NUM_ROUTED):
            tok, slot = (flat_idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            g, u, d = (t.float() for t in w_of(e))
            h = torch.nn.functional.silu(xs[tok] @ g) * (xs[tok] @ u)
            refflat.index_add_(0, tok, (h @ d) * flat_w[tok, slot].unsqueeze(1))
        ref = refflat.reshape(ROWS, N, HIDDEN)
        per_row = [pcc(dev_out[r], ref[r]) for r in range(ROWS)]
        # per-expert check: tokens that use only a given expert cannot be isolated; instead check the per-token PCC distribution
        tok_pcc = torch.tensor(
            [pcc(dev_out[r, t], ref[r, t]) for r in range(ROWS) for t in range(0, N, max(1, N // 64))]
        )
        print(
            f"RESULT op={op} fid={fid} N={N} cf={cf} skew={skew}: PCC per mesh row {['%.5f' % p for p in per_row]} "
            f"per-token PCC min {tok_pcc.min():.5f} mean {tok_pcc.mean():.5f} ; max|diff| {(dev_out - ref).abs().max():.4f} ref absmax {ref.abs().max():.3f}",
            flush=True,
        )
        # 32-row corruption check: per-token PCC for ALL tokens, grouped in 32-row blocks, per mesh row (a corrupted block shows as low PCC)
        a_, b_ = dev_out.double(), ref.double()
        a_, b_ = a_ - a_.mean(-1, keepdim=True), b_ - b_.mean(-1, keepdim=True)
        tp = (a_ * b_).sum(-1) / (a_.norm(dim=-1) * b_.norm(dim=-1) + 1e-30)  # [ROWS, N]
        blk = tp.reshape(ROWS, N // 32, 32)
        bad_blocks = [(blk[r].min(dim=1).values < 0.99).nonzero().flatten().tolist() for r in range(ROWS)]
        print(
            f"RESULT corruption N={N}: per-token PCC min per mesh row {[round(float(tp[r].min()), 5) for r in range(ROWS)]}; 32-row blocks with PCC<0.99 per row {bad_blocks}",
            flush=True,
        )
        assert not any(bad_blocks), f"corrupted 32-row blocks: {bad_blocks}"
        assert min(per_row) >= 0.999, f"layer PCC below 0.999: {per_row}"
