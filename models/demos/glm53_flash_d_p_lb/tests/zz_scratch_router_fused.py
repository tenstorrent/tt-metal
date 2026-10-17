# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scratch: GLM router tail (sigmoid, bias, top-8, gather, normalize, x2.5) as today's 7 ttnn ops vs one
ttnn.experimental.deepseek_prefill.moe_grouped_topk (n_groups 1), on the real layer-RF_LAYER gate weights and
RF_ROWS random rms-normalized rows (RF_SEEDS seeds). Both against a float64 torch reference: expert-set mismatches,
weight error, device time."""

import json
import os
import struct
import time

import pytest
import torch

import ttnn

HF = os.environ.get("BRINGUP_HF", "/localdev/mstaletovic/bringup/glm53_flash_d_p/hf")
LAYER = int(os.environ.get("RF_LAYER", "4"))
ROWS = int(os.environ.get("RF_ROWS", "640"))
SEEDS = int(os.environ.get("RF_SEEDS", "8"))


def _load(name):
    idx = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]
    f = f"{HF}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))[name]
        fh.seek(8 + n + h["data_offsets"][0])
        raw = fh.read(h["data_offsets"][1] - h["data_offsets"][0])
    dt = {"BF16": torch.bfloat16, "F32": torch.float32}[h["dtype"]]
    return torch.frombuffer(bytearray(raw), dtype=dt).reshape(h["shape"])


def _dev(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _timed(device, f, n=20):
    f()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / n * 1e6


def test_router_fused(device):
    p = f"model.language_model.layers.{LAYER}."
    w = _load(p + "mlp.gate.weight").float()  # [E, H]
    bias = _load(p + "mlp.gate.e_score_correction_bias").float()
    bias = bias - bias.mean()  # the model's recentring (top-k is shift invariant)
    E, H = w.shape
    w_t = _dev(w.T.contiguous().reshape(1, 1, H, E), device, ttnn.float32)
    b_row = _dev(bias.reshape(1, 1, 1, E), device, ttnn.float32)
    b_full = _dev(bias.reshape(1, 1, 1, E).expand(1, 1, ROWS, E).contiguous(), device, ttnn.float32)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    mism = {"ops": 0, "fused": 0, "fp32": 0}
    werr = {"ops": 0.0, "fused": 0.0, "fp32": 0.0}
    times = {}
    stats = {}
    bf_off = {"ops": 0, "fused": 0, "fp32": 0}
    sbias = {"ops": 0.0, "fused": 0.0, "fp32": 0.0}
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        x = torch.randn(ROWS, H)
        x = (x / x.pow(2).mean(-1, keepdim=True).sqrt()).bfloat16()
        xd = _dev(x.reshape(1, 1, ROWS, H), device, ttnn.bfloat16)
        logits = ttnn.linear(xd, w_t, dtype=ttnn.float32, compute_kernel_config=cfg)

        # float64 reference on the device's own logits (isolates the tail)
        lg = ttnn.to_torch(logits).double().reshape(ROWS, E)
        sc = torch.sigmoid(lg)
        ref_i = torch.topk(sc + bias.double(), 8, dim=-1).indices
        ref_w = sc.gather(-1, ref_i)
        ref_w = ref_w / ref_w.sum(-1, keepdim=True) * 2.5

        def ops():
            s = ttnn.sigmoid(logits)
            ch = ttnn.add(s, b_row)
            _, i = ttnn.topk(ch, k=8, dim=-1, largest=True, sorted=True)
            top = ttnn.gather(s, dim=-1, index=i)
            ts = ttnn.multiply(ttnn.sum(top, dim=-1, keepdim=True), 1 / 2.5)
            return ttnn.div(top, ts), i

        def fused(fp32=False):
            wt, i = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
                logits,
                b_full,
                n_groups=1,
                summed_experts_per_group=1,
                topk_groups=1,
                n_activated_experts=8,
                route_scale=2.5,
                epsilon=1e-20,
                stable_sort=True,
                score_func="sigmoid",
                fp32_selection=fp32,
            )
            return wt, i

        for name, f in (("ops", ops), ("fused", fused), ("fp32", lambda: fused(True))):
            wt, i = f()
            gi = ttnn.to_torch(i).long().reshape(ROWS, -1)[:, :8]
            gw = ttnn.to_torch(wt).double().reshape(ROWS, -1)[:, :8]
            chs = (sc + bias.double()).sort(-1, descending=True).values
            for r in range(ROWS):
                if set(gi[r].tolist()) != set(ref_i[r].tolist()):
                    mism[name] += 1
                    nd = len(set(gi[r].tolist()) - set(ref_i[r].tolist()))
                    gap = (chs[r, 7] - chs[r, 8]).item()
                    worst = (chs[r, 7] - (sc[r] + bias.double())[list(set(gi[r].tolist()) - set(ref_i[r].tolist()))].min()).item()
                    stats.setdefault(name, []).append((nd, gap, worst))
                else:  # weights in the reference's expert order
                    pos = {e: k for k, e in enumerate(gi[r].tolist())}
                    got = torch.stack([gw[r, pos[e]] for e in ref_i[r].tolist()])
                    werr[name] = max(werr[name], (got - ref_w[r]).abs().max().item())
                    bf_off[name] += int((got.float().bfloat16() != ref_w[r].float().bfloat16()).sum())
                    sbias[name] += float((got - ref_w[r]).sum())
            if seed == 0:
                times[name] = _timed(device, f)
        ttnn.deallocate(logits)
        ttnn.deallocate(xd)
    n = ROWS * SEEDS
    for name, st in stats.items():
        nd = torch.tensor([a for a, _, _ in st]); gap = torch.tensor([b for _, b, _ in st]); worst = torch.tensor([c for _, _, c in st])
        print(f"ROUTER_FUSED {name}: experts wrong per bad row {dict(zip(*[v.tolist() for v in nd.unique(return_counts=True)]))}; "
              f"ref 8th-9th gap median {gap.median():.2e} max {gap.max():.2e}; picked-but-wrong expert below ref 8th by median "
              f"{worst.median():.2e} max {worst.max():.2e}", flush=True)
    for name in ("ops", "fused", "fp32"):
        print(
            f"ROUTER_FUSED {name:5s}: expert-set mismatches vs fp64 {mism[name]} / {n} rows, max weight err "
            f"{werr[name]:.2e} (matching rows), weights != bf16(fp64) {bf_off[name]} / {8 * n}, mean signed err "
            f"{sbias[name] / (8 * n):.2e}, {times[name]:.1f} us/call (host-timed, sync per 20)",
            flush=True,
        )
