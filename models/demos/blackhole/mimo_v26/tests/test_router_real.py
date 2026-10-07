# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Router agreement on REAL ffn_in of track-0 dumps (bf16 chain, B=16 S=128, layers 1..5) with the real gate weight/bias.
Reference: fp32 CPU routing of the SAME bf16 ffn_in (pure router numerics) -> flips, share of flips with gap (8th vs 9th score+bias, fp32) < 0.01;
also vs the fp32-chain oracle dump and the floor (bf16-chain dump routing_idx vs fp32-chain dump routing_idx). Modes: plain (TTMoEGate bf16 op path),
exact (fp32 matmul + sigmoid + ttnn.topk), hilo (kernel + calibrated shift). Env: MIMO_DUMP, MIMO_DUMP_F32, MIMO_LAYERS (1,2,3,4,5), MIMO_ROUTER_T (64).
Also a hilo stage-diagnostic (same input 5 times: which stage differs).
"""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tt.router import TOPK, MiMoGate, calibrate_shift, cpu_route

DUMP = os.environ.get("MIMO_DUMP", "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16")
DUMP32 = os.environ.get("MIMO_DUMP_F32", "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16-f32")
LAYERS = [int(v) for v in os.environ.get("MIMO_LAYERS", "1,2,3,4,5").split(",")]
T = int(os.environ.get("MIMO_ROUTER_T", "64"))


def sets_equal(a, b):
    return (a.sort(-1).values == b.sort(-1).values).all(-1)


def flip_report(tag, eq, gap):
    n = eq.numel()
    fl = ~eq
    g = gap[fl]
    return (
        f"{tag}: set_match={float(eq.float().mean()):.4f} flips={int(fl.sum())}/{n} "
        f"flips_with_gap<0.01={float((g < 0.01).float().mean()) if g.numel() else float('nan'):.3f} "
        f"flip gap max={float(g.max()) if g.numel() else 0:.4f} (all-token share gap<0.01: {float((gap < 0.01).float().mean()):.3f})"
    )


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_router_real(mesh_device, mesh_shape):
    for layer in LAYERS:
        a = torch.load(f"{DUMP}/layer_{layer}.pt", mmap=True)["prefill"]
        f = torch.load(f"{DUMP32}/layer_{layer}.pt", mmap=True)["prefill"]
        x = a["ffn_in"].reshape(-1, 4096)  # bf16 [2048, 4096]
        gw, gb = load_gate(layer)
        ref_idx, _ = cpu_route(x.float(), gw, gb)
        s = (x.float() @ gw.T).sigmoid() + gb
        top = torch.topk(s, TOPK + 1, dim=-1).values
        gap = top[:, TOPK - 1] - top[:, TOPK]  # 8th - 9th, fp32 on the device input
        print(
            f"RESULT layer={layer} bias mean={gb.mean():.3f} std={gb.std():.4f}; gap<0.01 share (fp32, device input) {float((gap < 0.01).float().mean()):.3f}",
            flush=True,
        )
        eq_floor = sets_equal(a["routing_idx"], f["routing_idx"])
        print(
            "RESULT layer=%d FLOOR bf16-chain dump vs fp32-chain dump: %s"
            % (layer, flip_report("floor", eq_floor, f["route_gap"])),
            flush=True,
        )
        print(
            "RESULT layer=%d CPU-fp32-route(bf16 ffn_in) vs bf16 dump idx: set_match=%.4f ; vs fp32 dump idx: set_match=%.4f"
            % (
                layer,
                float(sets_equal(ref_idx, a["routing_idx"]).float().mean()),
                float(sets_equal(ref_idx, f["routing_idx"]).float().mean()),
            ),
            flush=True,
        )
        shift = calibrate_shift(x.float(), gw, gb)
        for mode, sh in (("plain", 0.0), ("exact", 0.0), ("hilo", shift)):
            gate = MiMoGate(mesh_device, gw.to(torch.bfloat16), gb, bias_shift=sh, batch_per_device=T)
            idxs = []
            for st in range(0, x.shape[0], T):
                tt_x = ttnn.from_torch(
                    x[st : st + T].reshape(1, 1, T, -1),
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                wt, ix = gate.route(tt_x, mode)
                idxs.append(ttnn.to_torch(ix).reshape(T, TOPK).long())
            idx = torch.cat(idxs)
            print(
                "RESULT layer=%d mode=%s vs fp32-CPU(same input): %s"
                % (layer, mode, flip_report(mode, sets_equal(idx, ref_idx), gap)),
                flush=True,
            )
            print(
                "RESULT layer=%d mode=%s vs fp32-chain oracle dump: set_match=%.4f ; vs bf16-chain dump: set_match=%.4f"
                % (
                    layer,
                    mode,
                    float(sets_equal(idx, f["routing_idx"]).float().mean()),
                    float(sets_equal(idx, a["routing_idx"]).float().mean()),
                ),
                flush=True,
            )
            if mode == "hilo" and layer == LAYERS[-1]:
                tt_x = ttnn.from_torch(
                    x[:T].reshape(1, 1, T, -1), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                runs = [gate.hilo_stages(tt_x) for _ in range(5)]
                for k_ in runs[0]:
                    same = [bool(torch.equal(runs[0][k_], r[k_])) for r in runs[1:]]
                    print(f"RESULT hilo stage {k_}: identical to run0 across 5 runs: {same}", flush=True)
                if not all(torch.equal(runs[0]["kernel_idx"], r["kernel_idx"]) for r in runs[1:]):
                    d = [
                        (runs[0]["kernel_idx"].sort(-1).values != r["kernel_idx"].sort(-1).values).any(-1).sum().item()
                        for r in runs[1:]
                    ]
                    print(f"RESULT hilo kernel idx rows (as sets) differing from run0: {d} of {T}", flush=True)
