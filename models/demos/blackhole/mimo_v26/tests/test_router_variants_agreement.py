# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Expert-set agreement of the fast router variant (router_select kernel + 1x8 matmul grid, T=32 rows per call) vs the exact router and vs fp32 CPU routing,
on real ffn_in: layers 1..5 from the bf16 prefill dump (bf16 chain) and layers 6..47 from the device-captured dump (mimo-device-s128b16), real gate/bias.
Per layer: set match vs fp32 CPU (same input), flips with fp32 gap < 0.01, identical sets/weights vs the exact router, weights PCC, 2-run bit identity.
Env: MIMO_LAYERS (1..47), MIMO_VARIANT (sel_mm1x8)."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tt.router import TOPK, MiMoGate, cpu_route

D1 = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"
D2 = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-device-s128b16"
LAYERS = [int(v) for v in os.environ.get("MIMO_LAYERS", ",".join(str(i) for i in range(1, 48))).split(",")]
VAR = os.environ.get("MIMO_VARIANT", "sel_mm1x8")
T = 32


def sets_eq(a, b):
    return (a.sort(-1).values == b.sort(-1).values).all(-1)


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_router_variants_agreement(mesh_device, mesh_shape):
    md = mesh_device
    tot = {"n": 0, "fast_ok": 0, "exact_ok": 0, "same": 0}
    for L in LAYERS:
        src = D1 if L <= 5 else D2
        x = torch.load(f"{src}/layer_{L}.pt", mmap=True)["prefill"]["ffn_in"].reshape(-1, 4096)
        gw, gb = load_gate(L)
        ref, _ = cpu_route(x.float(), gw, gb)
        s = (x.float() @ gw.T).sigmoid() + gb
        top = torch.topk(s, TOPK + 1, dim=-1).values
        gap = top[:, TOPK - 1] - top[:, TOPK]
        gate = MiMoGate(md, gw.to(torch.bfloat16), gb, bias_shift=0.0, batch_per_device=None)
        for n in ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices"):
            ttnn.deallocate(getattr(gate, n))

        def run(fn):
            idx, wt = [], []
            for st in range(0, x.shape[0], T):
                xd = ttnn.from_torch(
                    x[st : st + T].reshape(1, 1, T, -1), device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                w, i = fn(xd)
                idx.append(ttnn.to_torch(i).reshape(T, TOPK).long())
                wt.append(ttnn.to_torch(w).reshape(T, TOPK).float())
            return torch.cat(idx), torch.cat(wt)

        fi, fw = run(lambda xd: gate.route_v(xd, VAR))
        fi2, fw2 = run(lambda xd: gate.route_v(xd, VAR))
        ei, ew = run(lambda xd: gate.route(xd, "exact"))
        eq_f, eq_e = sets_eq(fi, ref), sets_eq(ei, ref)
        flips = ~eq_f
        g = gap[flips]
        # weights compared per expert id
        full = lambda i, w: torch.zeros(i.shape[0], 256).scatter_(1, i, w)
        wp = float(torch.corrcoef(torch.stack([full(fi, fw).flatten(), full(ei, ew).flatten()]))[0, 1])
        same_exact = float(sets_eq(fi, ei).float().mean())
        det = bool(torch.equal(fi, fi2) and torch.equal(fw, fw2))
        print(
            f"RESULT L{L} {VAR}: set match vs fp32-CPU {float(eq_f.float().mean()):.4f} (exact router {float(eq_e.float().mean()):.4f}); flips {int(flips.sum())} with gap<0.01: "
            f"{float((g < 0.01).float().mean()) if g.numel() else 1.0:.2f}; sets identical to exact router {same_exact:.4f}; weights PCC vs exact {wp:.6f}; 2 runs identical {det}",
            flush=True,
        )
        tot["n"] += x.shape[0]
        tot["fast_ok"] += int(eq_f.sum())
        tot["exact_ok"] += int(eq_e.sum())
        tot["same"] += int(sets_eq(fi, ei).sum())
        del gate
    print(
        f"RESULT TOTAL {VAR}: set match vs fp32-CPU {tot['fast_ok'] / tot['n']:.4f}; exact router {tot['exact_ok'] / tot['n']:.4f}; identical to exact {tot['same'] / tot['n']:.4f}",
        flush=True,
    )
    assert tot["fast_ok"] >= tot["exact_ok"] - 0.002 * tot["n"], "fast router agrees less than the exact router"
