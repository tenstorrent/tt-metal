# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Scale-bias (alpha) decomposition of the device MoE vs an fp32 oracle, per stage. PCC is scale-invariant, so every number here is alpha = <dev,ref>/<ref,ref> and
rel = ||dev-ref|| / ||ref|| (plus cosine of the delta to the reference).
Real layer MIMO_LAYER (5; 1; or 16+ from the device capture dump MIMO_DUMP=mimo-device-s128b16), real weights bfp8, dump ffn_in and dump routing (idx, w).
Oracle (CPU, fp32 math, same bf16 ffn_in, bf16-exact dequantised weights, dump routing) on the first MIMO_ORACLE_TOK tokens of every mesh row (default 128).
Per op variant (MIMO_UM_OPS=unified,mm; env MOE_UNI_* and MOE_UNI_FIDELITY select the patched unified-op behaviour):
  A  expert outputs per (token, slot) read from `combine` (upto=4) vs oracle per-slot expert outputs      -> experts + dispatch + combine
  B  device final (post_combine_reduce, all columns summed) vs fp32 weighted sum of the SAME device expert outputs (bf16 weights) -> reduce stage
  C  device final vs oracle                                                                                -> total
  floor: bf16-chain dump ffn_out vs oracle (CPU bf16 chain vs fp32 math).  Also: sum of routing weights, device-read weights vs dequantised bf16 weights (mean ratio).
"""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_prefill_4x8 import pcc
from models.demos.blackhole.mimo_v26.tt.moe_prefill import (
    COLS,
    HIDDEN,
    ROWS,
    TOPK,
    MiMoPrefillMoE,
    build_expert_weights,
    global_expert,
    make_ckc,
)
from models.demos.blackhole.mimo_v26.tt.moe_weights import load_layer_experts

DUMP = os.environ.get("MIMO_DUMP", "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16")
LAYER = int(os.environ.get("MIMO_LAYER", "5"))
OPS = os.environ.get("MIMO_UM_OPS", "unified").split(",")
NT = int(os.environ.get("MIMO_ORACLE_TOK", "128"))


def stats(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    alpha = float((a @ b) / (b @ b))
    rel = float((a - b).norm() / b.norm())
    d = a - b
    cos = float((d @ b) / (d.norm() * b.norm() + 1e-30))
    return f"alpha={alpha:.4f} rel={rel:.4f} cos(delta,ref)={cos:+.2f}"


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_moe_scale_alpha(mesh_device):
    md = mesh_device
    a = torch.load(f"{DUMP}/layer_{LAYER}.pt", mmap=True)["prefill"]
    N = 512
    x = a["ffn_in"].reshape(ROWS, N, HIDDEN)
    idx_d = a["routing_idx"].reshape(ROWS, N, TOPK)
    w_d = a["routing_w"].reshape(ROWS, N, TOPK)
    ref_dump = a["ffn_out"].reshape(ROWS, N, HIDDEN).float()
    print(
        f"RESULT L{LAYER}: routing weight sum mean {float(w_d.sum(-1).mean()):.5f} min {float(w_d.sum(-1).min()):.5f} max {float(w_d.sum(-1).max()):.5f}; env "
        + " ".join(f"{k}={os.environ[k]}" for k in os.environ if k.startswith("MOE_UNI")),
        flush=True,
    )
    t0 = time.time()
    W = load_layer_experts(LAYER)
    # ---- fp32 oracle per slot (first NT tokens of each mesh row)
    tok = torch.arange(NT)
    O = torch.zeros(ROWS, NT, TOPK, HIDDEN)
    for r in range(ROWS):
        xr = x[r, :NT].float()
        ir = idx_d[r, :NT]
        for e in ir.unique().tolist():
            t_, s_ = (ir == e).nonzero(as_tuple=True)
            g, u, d = (m.float() for m in W[e])
            h = torch.nn.functional.silu(xr[t_] @ g) * (xr[t_] @ u)
            O[r, t_, s_] = h @ d
    wf = w_d[:, :NT].float()
    ref_fp32 = (O * wf.unsqueeze(-1)).sum(2)  # fp32 oracle, fp32 weights
    print(
        f"RESULT oracle built in {time.time() - t0:.0f}s; floor bf16-chain dump ffn_out vs fp32 oracle: {stats(ref_dump[:, :NT], ref_fp32)}",
        flush=True,
    )
    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    xt = ttnn.from_torch(
        x.reshape(ROWS, 1, N, HIDDEN),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=m2,
    )
    ixd = ttnn.reshape(
        ttnn.from_torch(
            idx_d.reshape(ROWS, N, 1, 1, TOPK).to(torch.int32),
            device=md,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=m2,
        ),
        [N, 1, 1, TOPK],
    )
    wtd = ttnn.reshape(
        ttnn.from_torch(
            w_d.reshape(ROWS, N, 1, 1, TOPK),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=m2,
        ),
        [N, 1, 1, TOPK],
    )
    w_bf = w_d.bfloat16().float()[:, :NT]
    for opname in OPS:
        # op names: unified | mm | mm_f32int (fp32 gate/up/activated) | mm_bf16w (bf16 weights instead of bfp8) | mm_hifi4 | mm_bf16w_f32int
        op = "mm" if opname.startswith("mm") else opname
        wdt = ttnn.bfloat16 if "bf16w" in opname else None
        wts = build_expert_weights(md, lambda e: W[e], nd_sharded=(op not in ("mm", "loop")), dtype=wdt)
        print(f"RESULT variant {opname}", flush=True)
        # (d) device-read weights vs dequantised bf16: slot 0 gate on a few devices
        ratios = []
        for r, c in ((0, 0), (1, 3), (3, 7)):
            e = global_expert(r, c, 0)
            dev = ttnn.to_torch(ttnn.get_device_tensors(wts[0][0])[r * COLS + c]).float().reshape(-1)
            ref = W[e][0].float().reshape(-1)
            n = min(dev.numel(), ref.numel())
            ratios.append(
                f"e{e}: mean ratio {float(dev[:n].abs().sum() / ref[:n].abs().sum()):.5f} maxdiff {float((dev[:n] - ref[:n]).abs().max()):.2e}"
            )
        print(f"RESULT op={opname} (d) device weights vs dequantised bf16 (gate, slot 0): {ratios}", flush=True)
        um = MiMoPrefillMoE(
            md, wts, ckc=make_ckc("hifi2", fp32acc=True, packer_l1_acc=(op != "fused")), expert_op=op, cf=4
        )
        x_rm = lambda: ttnn.reshape(ttnn.to_layout(xt, ttnn.ROW_MAJOR_LAYOUT), [1, N, HIDDEN])
        # A: expert outputs per slot from combine
        comb = um.forward(x_rm(), wtd, ixd, upto=4)
        ttnn.synchronize_device(md)
        S1 = torch.zeros(ROWS, NT, TOPK, HIDDEN)
        for r in range(ROWS):
            ir = idx_d[r, :NT]
            col_of = ir // 32
            for c in range(COLS):
                msk = col_of == c
                if not msk.any():
                    continue
                dc = ttnn.to_torch(ttnn.get_device_tensors(comb)[r * COLS + c])
                dc = dc.reshape(N, TOPK, HIDDEN)[:NT].float()
                S1[r][msk] = dc[msk]
        ttnn.deallocate(comb)
        print(
            f"RESULT L{LAYER} op={opname} A experts+dispatch+combine vs oracle per-slot outputs: {stats(S1, O)}",
            flush=True,
        )
        red_host = (S1 * w_bf.unsqueeze(-1)).sum(2)  # fp32 weighted sum of device expert outputs (bf16 weights)
        # B/C: final
        part = um.forward(x_rm(), wtd, ixd)
        ttnn.synchronize_device(md)
        devs = ttnn.get_device_tensors(part)
        out = torch.stack(
            [
                torch.stack(
                    [ttnn.to_torch(devs[r * COLS + c]).reshape(N, HIDDEN)[:NT].float() for c in range(COLS)]
                ).sum(0)
                for r in range(ROWS)
            ]
        )
        ttnn.deallocate(part)
        print(
            f"RESULT L{LAYER} op={opname} B reduce stage: device final vs fp32 weighted sum of device expert outputs: {stats(out, red_host)}",
            flush=True,
        )
        print(
            f"RESULT L{LAYER} op={opname} C total: device final vs fp32 oracle: {stats(out, ref_fp32)} ; PCC {pcc(out, ref_fp32):.5f}; (fp32 sum of device expert outputs with fp32 weights vs oracle: {stats((S1 * wf.unsqueeze(-1)).sum(2), ref_fp32)})",
            flush=True,
        )
        for lst in wts:
            for t in lst:
                ttnn.deallocate(t)
