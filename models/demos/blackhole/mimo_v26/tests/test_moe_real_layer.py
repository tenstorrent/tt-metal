# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-weight MoE layer on 4x8 vs the track-0 dump ffn_out: all 256 REAL experts of one layer (fp4 -> bf16 -> bfp8), real gate + bias, real ffn_in
(2048 tokens = 16 users x 128, 4 mesh rows x 512 tokens). Two device runs: (A) the dump's own routing (idx, w) fed to dispatch -> isolates the expert path,
(B) device `exact` router -> end to end. Reports PCC per mesh row vs bf16-chain dump ffn_out and vs fp32-chain ffn_out (noise floor: bf16 dump vs fp32 dump),
real per-expert skew and capacity. Env: MIMO_LAYER (1), MIMO_UM_CF (4).
"""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tests.test_moe_prefill_4x8 import pcc
from models.demos.blackhole.mimo_v26.tt.moe_prefill import (
    COLS,
    EPC,
    HIDDEN,
    NUM_ROUTED,
    ROWS,
    TOPK,
    MiMoPrefillMoE,
    build_expert_weights,
    global_expert,
    make_ckc,
)
from models.demos.blackhole.mimo_v26.tt.moe_weights import load_layer_experts
from models.demos.blackhole.mimo_v26.tt.router import MiMoGate

DUMP = os.environ.get("MIMO_DUMP", "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16")
DUMP32 = os.environ.get("MIMO_DUMP_F32", "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16-f32")
LAYER = int(os.environ.get("MIMO_LAYER", "1"))


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_moe_real_layer(mesh_device):
    md = mesh_device
    a = torch.load(f"{DUMP}/layer_{LAYER}.pt", mmap=True)["prefill"]
    f = torch.load(f"{DUMP32}/layer_{LAYER}.pt", mmap=True)["prefill"]
    N = a["ffn_in"].shape[0] * a["ffn_in"].shape[1] // ROWS  # 512 tokens per mesh row
    x = a["ffn_in"].reshape(ROWS, N, HIDDEN)
    t0 = time.time()
    W = load_layer_experts(LAYER)
    print(f"RESULT layer {LAYER}: loaded+dequantised 256 real experts in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    weights = build_expert_weights(md, lambda e: W[e])
    print(f"RESULT bfp8 weights on device in {time.time() - t0:.0f}s", flush=True)
    cf = int(os.environ.get("MIMO_UM_CF", "4"))
    um = MiMoPrefillMoE(md, weights, ckc=make_ckc("hifi2"), expert_op="unified", cf=cf)

    # skew from the real routing (all 2048 tokens)
    idx_all = a["routing_idx"]
    cnt = torch.bincount(idx_all.flatten(), minlength=NUM_ROUTED)
    print(
        f"RESULT real skew L{LAYER}: per-expert count mean {cnt.float().mean():.1f} max {int(cnt.max())} (expert {int(cnt.argmax())}) min {int(cnt.min())} zero-count experts {int((cnt == 0).sum())}; max/mean {float(cnt.max() / cnt.float().mean()):.2f}",
        flush=True,
    )
    load = {
        (r, c): sum(-(-int(cnt[global_expert(r, c, l)]) // 32) * 32 for l in range(EPC))
        for r in range(ROWS)
        for c in range(COLS)
    }
    wl = max(load.values())
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants

    for cfx in (1, 2, 3, 4):
        mb = compute_constants(N, NUM_ROUTED, TOPK, ROWS * COLS, ROWS, cfx)[2]
        print(
            f"RESULT capacity L{LAYER} cf={cfx}: worst chip load {wl} rows vs max_buf {mb}: {'OK' if wl <= mb else 'OVERFLOW'}; per-mesh-row worst {[max(load[(r, c)] for c in range(COLS)) for r in range(ROWS)]}",
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

    def run(wt, ix):
        x_rm = ttnn.reshape(ttnn.to_layout(xt, ttnn.ROW_MAJOR_LAYOUT), [1, N, HIDDEN])
        part = um.forward(x_rm, wt, ix)
        ttnn.synchronize_device(md)
        devs = ttnn.get_device_tensors(part)
        return torch.stack(
            [
                torch.stack([ttnn.to_torch(devs[r * COLS + c]).reshape(N, HIDDEN).float() for c in range(COLS)]).sum(0)
                for r in range(ROWS)
            ]
        )

    ref_b = a["ffn_out"].reshape(ROWS, N, HIDDEN).float()
    ref_f = f["ffn_out"].reshape(ROWS, N, HIDDEN).float()
    print(
        "RESULT floor: bf16-chain ffn_out vs fp32-chain ffn_out PCC per mesh row %s"
        % [round(pcc(ref_b[r], ref_f[r]), 5) for r in range(ROWS)],
        flush=True,
    )

    # (A) dump routing
    ixd = ttnn.from_torch(
        idx_all.reshape(ROWS, N, 1, 1, TOPK).to(torch.int32),
        device=md,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=m2,
    )
    wtd = ttnn.from_torch(
        a["routing_w"].reshape(ROWS, N, 1, 1, TOPK),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=m2,
    )
    outA = run(ttnn.reshape(wtd, [N, 1, 1, TOPK]), ttnn.reshape(ixd, [N, 1, 1, TOPK]))
    print(
        "RESULT layer %d (A) dump routing -> device experts bfp8: PCC vs bf16 dump ffn_out per row %s ; vs fp32 dump %s"
        % (
            LAYER,
            [round(pcc(outA[r], ref_b[r]), 5) for r in range(ROWS)],
            [round(pcc(outA[r], ref_f[r]), 5) for r in range(ROWS)],
        ),
        flush=True,
    )
    # (B) device exact router
    gw, gb = load_gate(LAYER)
    gate = MiMoGate(md, gw.to(torch.bfloat16), gb, bias_shift=0.0, batch_per_device=N)
    wtB, ixB = gate.route(xt, "exact")
    ixh = torch.stack(
        [ttnn.to_torch(ttnn.get_device_tensors(ixB)[r * COLS]).reshape(N, TOPK).long() for r in range(ROWS)]
    ).reshape(-1, TOPK)
    eq = (ixh.sort(-1).values == idx_all.sort(-1).values).all(-1)
    print(
        f"RESULT layer {LAYER} (B) device exact router vs bf16 dump idx: set match {float(eq.float().mean()):.4f}",
        flush=True,
    )
    outB = run(wtB, ixB)
    print(
        "RESULT layer %d (B) device exact router + experts: PCC vs bf16 dump ffn_out per row %s ; vs fp32 dump %s"
        % (
            LAYER,
            [round(pcc(outB[r], ref_b[r]), 5) for r in range(ROWS)],
            [round(pcc(outB[r], ref_f[r]), 5) for r in range(ROWS)],
        ),
        flush=True,
    )
    outB2 = run(wtB, ixB)
    print(
        "RESULT layer %d determinism (B) 2 runs bit-identical: %s" % (LAYER, bool(torch.equal(outB, outB2))), flush=True
    )
    assert min(pcc(outA[r], ref_b[r]) for r in range(ROWS)) >= 0.999
