# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Robust per-shape bf16 Conv3d blocking sweep for the ACE-Step Oobleck VAE (device-kernel time).

For each REAL VAE conv shape (in, out, k) at its REAL T, try every candidate blocking wrapped in
try/except so an L1 OOM (TT_THROW) on one candidate never aborts the sweep — we simply skip it and
keep going. Every surviving candidate is (a) checked for PCC vs the (32,32,1) fallback (blocking only
changes tiling, not math) and (b) timed by the Tracy device profiler (KERNEL time, not wall-clock).
Prints the fastest fitting blocking per shape as a ready-to-paste _VAE_BF16_BLACKHOLE table.

    python models/experimental/acestep/perf/sweep_vae_bf16.py

No trace capture (kept separate from Tracy). Isolated single convs (not the whole VAE) so each shape
is measured independently and a failure is contained.
"""

import json
import os

import pytest
import torch

import ttnn

from models.tt_dit.layers.audio_ops import Conv1dViaConv3d
from models.tt_dit.utils import conv3d as c3d

# Real Oobleck decoder conv shapes at T'=128 latent (in, out, k, T_in). T grows through upsampling.
# Only the compute-dominant shapes are swept at their real (large) T; small ones use a modest T.
SHAPES = [
    (64, 2048, 7, 256),
    (2048, 1024, 20, 2579),
    (1024, 1024, 7, 2560),
    (1024, 512, 12, 15371),
    (512, 512, 7, 15360),
    (512, 256, 8, 61447),
    (256, 256, 7, 61440),
    (256, 128, 8, 245767),
    (128, 128, 7, 491520),
    (128, 128, 4, 491523),
]

# The k20 ConvTranspose (2048->1024) has a huge L1 footprint (k=20); most big blockings OOM, and the
# fallback (32,32,1) is what it's stuck on. Sweep it with a SMALLER C grid so some candidate fits.
# (Kept in SHAPES above too; the candidate filter handles the OOMs.)
SHAPE_IDS = [f"{i}_{o}_k{k}_T{t}" for (i, o, k, t) in SHAPES]

# Candidate grid. C_in_block <= in_channels; big products raise TT_THROW and are skipped by try/except.
# T_out extended to 64/128: the first sweep's best blockings all hit the T_out=32 ceiling -> headroom.
C_IN_CANDS = [32, 64, 128, 256, 512]
C_OUT_CANDS = [32, 64, 128, 256]
T_OUT_CANDS = [8, 16, 32, 64, 128]

PROFILER_DUMP_EVERY = 8
SUBDIR = "acestep_vae_bf16_sweep"
DT = ttnn.bfloat16


def _candidates(in_c):
    for ci in C_IN_CANDS:
        if ci > in_c:
            continue
        for co in C_OUT_CANDS:
            for to in T_OUT_CANDS:
                yield (ci, co, to, 1, 1)


def _pcc(a, b):
    a = a.flatten().float() - a.flatten().float().mean()
    b = b.flatten().float() - b.flatten().float().mean()
    d = (a.norm() * b.norm()).item()
    return 1.0 if d == 0 else torch.dot(a, b).item() / d


def _build_and_run(device, in_c, out_c, k, T, blocking):
    """Register `blocking` (bf16), build a fresh conv, run one forward. Returns out tensor or raises."""
    key = (in_c, out_c, (k, 1, 1))
    saved = c3d._DEFAULT_BLOCKINGS.get(key)
    if blocking is not None:
        c3d._DEFAULT_BLOCKINGS[key] = blocking
    elif key in c3d._DEFAULT_BLOCKINGS:
        del c3d._DEFAULT_BLOCKINGS[key]
    try:
        conv = Conv1dViaConv3d(in_channels=in_c, out_channels=out_c, kernel_size=k, mesh_device=device, dtype=DT)
        g = torch.Generator().manual_seed(in_c * 100003 + out_c * 101 + k)
        w = torch.randn(out_c, in_c, k, generator=g)
        b = torch.randn(out_c, generator=g)
        conv.load_torch_state_dict({"conv.weight": w, "conv.bias": b})
        x = ttnn.from_torch(
            torch.randn(1, T, in_c, generator=g), device=device, dtype=DT, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        return conv.forward(x)
    finally:
        if saved is not None:
            c3d._DEFAULT_BLOCKINGS[key] = saved
        elif blocking is not None and key in c3d._DEFAULT_BLOCKINGS:
            del c3d._DEFAULT_BLOCKINGS[key]


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_sweep_bf16(device, shape):
    from tracy import signpost

    in_c, out_c, k, T = shape

    # Reference: the (32,32,1) fallback output (known-good bf16 baseline). Wrapped defensively.
    try:
        ref = ttnn.to_torch(_build_and_run(device, in_c, out_c, k, T, (32, 32, 1, 1, 1))).float()
    except Exception as ex:  # noqa: BLE001
        print(f"REF_FAIL {in_c}_{out_c}_k{k}_T{T}: {type(ex).__name__}")
        return

    valid = []
    for blk in _candidates(in_c):
        if blk == (32, 32, 1, 1, 1):
            valid.append(blk)  # baseline always included
            continue
        try:
            out = ttnn.to_torch(_build_and_run(device, in_c, out_c, k, T, blk)).float()
        except Exception:  # noqa: BLE001 — L1 OOM / unsupported blocking; skip, keep sweeping.
            continue
        try:
            if _pcc(ref, out) >= 0.999:
                valid.append(blk)
        except Exception:  # noqa: BLE001
            continue

    combo_file = os.environ.get("VAE_SWEEP_COMBO_FILE")
    if combo_file:
        with open(combo_file, "w") as f:
            json.dump({"shape": list(shape), "combos": [list(b) for b in valid]}, f)

    # Warmup (compile) all valid combos, flushing periodically.
    for i, blk in enumerate(valid):
        try:
            _build_and_run(device, in_c, out_c, k, T, blk)
        except Exception:  # noqa: BLE001
            pass
        if (i + 1) % PROFILER_DUMP_EVERY == 0:
            ttnn.ReadDeviceProfiler(device)
    ttnn.ReadDeviceProfiler(device)

    # Measured: one forward per valid combo, in order, between signposts.
    signpost("start")
    for i, blk in enumerate(valid):
        try:
            _build_and_run(device, in_c, out_c, k, T, blk)
            ttnn.synchronize_device(device)
        except Exception:  # noqa: BLE001
            pass
        if (i + 1) % PROFILER_DUMP_EVERY == 0:
            ttnn.ReadDeviceProfiler(device)
    signpost("stop")
    ttnn.ReadDeviceProfiler(device)


def _orchestrate():
    import pandas as pd
    from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

    best_table = {}
    for shape, sid in zip(SHAPES, SHAPE_IDS):
        combo_file = f"/tmp/vae_bf16_combos_{sid}.json"
        os.environ["VAE_SWEEP_COMBO_FILE"] = combo_file
        cmd = f"pytest models/experimental/acestep/perf/sweep_vae_bf16.py::test_sweep_bf16 -k '{sid}' -q -s"
        try:
            run_device_profiler(
                cmd,
                SUBDIR,
                check_test_return_code=False,
                device_analysis_types=["device_kernel_duration"],
                op_support_count=50000,
            )
            df = pd.read_csv(get_latest_ops_log_filename(SUBDIR))
            sp = df[df["OP TYPE"] == "signpost"]["OP CODE"]
            s, e = sp[sp == "start"].index[0], sp[sp == "stop"].index[0]
            seg = df.iloc[s + 1 : e]
            seg = seg[seg["OP CODE"] == "Conv3dDeviceOperation"]
            durs = pd.to_numeric(seg["DEVICE KERNEL DURATION [ns]"], errors="coerce").dropna().tolist()
            combos = json.load(open(combo_file))["combos"]
        except Exception as ex:  # noqa: BLE001 — a shape blew up entirely; record + continue.
            print(f"{sid}: SWEEP_FAIL {type(ex).__name__}: {ex}")
            continue
        if not durs or not combos:
            print(f"{sid}: no data (durs={len(durs)} combos={len(combos)})")
            continue
        n = min(len(durs), len(combos))
        pairs = sorted(zip(durs[:n], combos[:n]), key=lambda x: x[0])
        best_ns, best_blk = pairs[0]
        base = next((d for d, b in zip(durs[:n], combos[:n]) if tuple(b) == (32, 32, 1, 1, 1)), None)
        in_c, out_c, k, T = shape
        spd = f"{base / best_ns:.2f}x vs (32,32,1)" if base else ""
        print(f"{sid}: best={best_ns/1e3:.1f}us {tuple(best_blk)}  {spd}  ({n} fitting combos)")
        best_table[(in_c, out_c, (k, 1, 1))] = tuple(best_blk)

    print("\n# Paste into vae_conv_config._VAE_BF16_BLACKHOLE:")
    for (i, o, k), blk in best_table.items():
        print(f"    ({i}, {o}, {k}): {blk},")
    json.dump({f"{i}_{o}_{k[0]}": list(v) for (i, o, k), v in best_table.items()}, open("/tmp/vae_bf16_best.json", "w"))


if __name__ == "__main__":
    _orchestrate()
