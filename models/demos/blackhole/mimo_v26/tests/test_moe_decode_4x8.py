# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode-style MoE on 4x8: 32 users per mesh row, RAGGED live tokens per row, REAL gate weights + correction biases of layers 1/20/46
(MIMO_LAYERS), MiMoGate mode exact vs hilo (+calibrated shift) feeding the same dispatch/unified-expert/combine pipeline (bfp8 experts).
Experts and hidden states are SYNTHETIC here (random, seeded): this validates routing-to-combine plumbing, ragged handling, determinism; the router
agreement numbers on random hidden states are NOT a routing-quality claim (needs track-0 real hidden dumps).
Checks per layer and mode: (a) set agreement of device routing vs fp32 CPU routing (info), (b) device-routed MoE output vs dense torch MoE using the
SAME device indices/weights, live tokens only, PCC >= 0.999 per mesh row, (c) 3 runs bit-identical, (d) dead (padded) users do not disturb live ones.
"""
import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_prefill_4x8 import pcc
from models.demos.blackhole.mimo_v26.tt.moe_prefill import (
    COLS,
    HIDDEN,
    INTER,
    NUM_ROUTED,
    ROWS,
    TOPK,
    MiMoPrefillMoE,
    build_expert_weights,
    make_ckc,
)
from models.demos.blackhole.mimo_v26.tt.router import MiMoGate, calibrate_shift, cpu_route

CKPT = os.environ.get("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL")
LAYERS = [int(v) for v in os.environ.get("MIMO_LAYERS", "1,20,46").split(",")]
USERS = int(os.environ.get("MIMO_USERS", "32"))
LIVE = [int(v) for v in os.environ.get("MIMO_LIVE", "32,20,7,1").split(",")]  # live users per mesh row (ragged)


def load_gate(layer):
    from safetensors import safe_open

    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    out = []
    for n in ("weight", "e_score_correction_bias"):
        name = f"model.layers.{layer}.mlp.gate.{n}"
        with safe_open(os.path.join(CKPT, idx[name]), "pt") as f:
            out.append(f.get_tensor(name).float())
    return out


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_moe_decode_4x8(mesh_device):
    md = mesh_device
    W = {}

    def w_of(e):
        if e not in W:
            g = torch.Generator().manual_seed(1000 + e)
            W[e] = tuple(
                (torch.randn(a, b, generator=g) * 0.02).bfloat16()
                for a, b in ((HIDDEN, INTER), (HIDDEN, INTER), (INTER, HIDDEN))
            )
        return W[e]

    um = MiMoPrefillMoE(
        md,
        build_expert_weights(md, w_of),
        ckc=make_ckc("hifi2"),
        expert_op="unified",
        cf=int(os.environ.get("MIMO_UM_CF", "4")),
    )
    m2 = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    N = max(
        64, -(-USERS // 64) * 64
    )  # masked_bincount needs tokens % 64 == 0: 32 users/row are padded with dead rows to 64
    failures = []
    for layer in LAYERS:
        gw, gb = load_gate(layer)  # [256,4096] fp32, [256]
        print(
            f"RESULT layer={layer} gate bias mean={gb.mean():.3f} std={gb.std():.4f} min={gb.min():.3f} max={gb.max():.3f} W dtype-> fp32 absmax {gw.abs().max():.3f}",
            flush=True,
        )
        torch.manual_seed(layer)
        x = torch.randn(ROWS, N, HIDDEN)
        x = (x / x.pow(2).mean(-1, keepdim=True).sqrt()).bfloat16()  # unit-rms like a normed hidden
        live = torch.zeros(ROWS, N, dtype=torch.bool)
        for r in range(ROWS):
            live[r, : LIVE[r]] = True
        x = torch.where(live.unsqueeze(-1), x, torch.zeros_like(x))  # dead users: zero hidden
        xs = x.reshape(-1, HIDDEN).float()
        ref_idx, _ = cpu_route(xs, gw, gb)
        shift = calibrate_shift(xs[live.flatten()], gw, gb)
        xd = ttnn.from_torch(
            x.reshape(ROWS, 1, N, HIDDEN),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=m2,
        )
        for mode, sh in (("exact", 0.0), ("hilo", shift)):
            gate = MiMoGate(md, gw.to(torch.bfloat16), gb, bias_shift=sh, batch_per_device=N)
            outs = []
            for it in range(3):
                wt, ix = gate.route(xd, mode)
                if it == 0:
                    ixh = torch.stack(
                        [
                            ttnn.to_torch(ttnn.get_device_tensors(ix)[r * COLS]).reshape(N, TOPK).long()
                            for r in range(ROWS)
                        ]
                    )
                    wth = torch.stack(
                        [
                            ttnn.to_torch(ttnn.get_device_tensors(wt)[r * COLS]).reshape(N, TOPK).float()
                            for r in range(ROWS)
                        ]
                    )
                x_rm = ttnn.reshape(ttnn.to_layout(xd, ttnn.ROW_MAJOR_LAYOUT), [1, N, HIDDEN])
                part = um.forward(x_rm, wt, ix)
                ttnn.synchronize_device(md)
                devs = ttnn.get_device_tensors(part)
                outs.append(
                    torch.stack(
                        [
                            torch.stack(
                                [ttnn.to_torch(devs[r * COLS + c]).reshape(N, HIDDEN).float() for c in range(COLS)]
                            )
                            for r in range(ROWS)
                        ]
                    )
                )
            ident_all = all(bool(torch.equal(outs[0], o)) for o in outs[1:])
            lvm = live.reshape(ROWS, 1, N, 1)
            ident = all(
                bool(torch.equal(outs[0].masked_select(lvm.expand_as(outs[0])), o.masked_select(lvm.expand_as(o))))
                for o in outs[1:]
            )
            out = outs[0].sum(1)  # [ROWS, N, D]
            lv = live.flatten()
            agree = float(
                torch.tensor(
                    [set(a.tolist()) == set(b.tolist()) for a, b in zip(ixh.reshape(-1, TOPK)[lv], ref_idx[lv])]
                )
                .float()
                .mean()
            )
            # dense reference with the device's own routing
            ref = torch.zeros(ROWS * N, HIDDEN)
            fi, fw = ixh.reshape(-1, TOPK), wth.reshape(-1, TOPK)
            for e in range(NUM_ROUTED):
                tok, slot = ((fi == e) & lv.unsqueeze(1)).nonzero(as_tuple=True)
                if tok.numel() == 0:
                    continue
                g, u, d = (t.float() for t in w_of(e))
                h = torch.nn.functional.silu(xs[tok] @ g) * (xs[tok] @ u)
                ref.index_add_(0, tok, (h @ d) * fw[tok, slot].unsqueeze(1))
            ref = ref.reshape(ROWS, N, HIDDEN)
            pr = [pcc(out[r][live[r]], ref[r][live[r]]) for r in range(ROWS)]
            print(
                f"RESULT layer={layer} mode={mode} shift={sh:.3f} users/row={N} live/row={LIVE}: set_agree_vs_fp32_cpu(live,synthetic x)={agree:.4f} "
                f"MoE PCC per row (live tokens) {['%.5f' % p for p in pr]} 3-runs-identical(live tokens)={ident} (incl. dead pad tokens: {ident_all})",
                flush=True,
            )
            if not ident or min(pr) < 0.999:
                failures.append((layer, mode, pr, ident))
            for t in (wt, ix):
                ttnn.deallocate(t)
            del gate
    assert not failures, failures
