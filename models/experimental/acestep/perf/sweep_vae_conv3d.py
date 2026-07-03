# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Brute-force sweep of Conv3dConfig blockings for the ACE-Step Oobleck VAE convs, timed by the
device profiler (KERNEL time, not host wall-clock).

The VAE is 99% of pipeline device time (Tracy). Every VAE conv currently hits the tt_dit conv3d
FALLBACK blocking (C_out_block=32, T_out_block=1) because none of our (in,out,kernel) shapes are in
tt_dit's tuned registry. This sweep finds, per shape, the fastest blocking (C_in_block, C_out_block,
T_out_block) by DEVICE KERNEL DURATION, and prints a table to register EXTERNALLY via
`register_conv3d_configs(...)` from an acestep config module — no tt_dit edits, no hardcoding in the
VAE forward.

Mirrors tt_dit's utils/sweep_mm_block_sizes.py: a worker test opens the mesh once, sweeps every
candidate under signpost("start")/("stop") with periodic ttnn.ReadDeviceProfiler flushes
(TT_METAL_PROFILER_MID_RUN_DUMP=1), and an orchestrator runs it under run_device_profiler and parses
the ops log for Conv3dDeviceOperation kernel durations. No trace capture (kept separate from Tracy).

    # Orchestrator (profiled):
    python models/experimental/acestep/perf/sweep_vae_conv3d.py

Correctness: a blocking only changes tiling, not math. The worker also runs each candidate once
eagerly and checks PCC vs the fallback output; candidates that diverge/err are dropped before timing.
"""

import json
import os

import pytest
import torch

import ttnn

from models.tt_dit.layers.audio_ops import Conv1dViaConv3d
from models.tt_dit.utils import conv3d as c3d

# Distinct (in, out, kernel, T_in) conv shapes in the Oobleck decoder at T'=128. T grows through the
# upsample stages. Kept to the compute-dominant ones (large T or large C).
SHAPES = [
    (64, 2048, 7, 256),
    (2048, 2048, 7, 2560),
    (2048, 2048, 1, 2560),
    (1024, 1024, 7, 15360),
    (512, 512, 7, 61440),
    (128, 2, 7, 491520),
]
SHAPE_IDS = [f"{i}_{o}_k{k}_T{t}" for (i, o, k, t) in SHAPES]

C_IN_CANDS = [32, 64, 128, 256]
C_OUT_CANDS = [32, 64, 128, 256]
T_OUT_CANDS = [1, 2, 4, 8]

PROFILER_DUMP_EVERY = 8
SUBDIR = "acestep_vae_conv_sweep"
RESULTS_JSON = os.environ.get("VAE_SWEEP_RESULTS", "/tmp/vae_conv_sweep_results.json")


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
    """Register `blocking` for this shape, build a fresh conv, run one forward. Returns out tensor."""
    key = (in_c, out_c, c3d._ntuple(k, 3))
    saved = c3d._DEFAULT_BLOCKINGS.get(key)
    if blocking is not None:
        c3d._DEFAULT_BLOCKINGS[key] = blocking
    elif key in c3d._DEFAULT_BLOCKINGS:
        del c3d._DEFAULT_BLOCKINGS[key]
    try:
        conv = Conv1dViaConv3d(
            in_channels=in_c, out_channels=out_c, kernel_size=k, mesh_device=device, dtype=ttnn.float32
        )
        # Populate real (random) weights — a fresh conv is META until loaded. Same seed per shape so
        # every blocking convolves identical weights/input -> PCC vs fallback is a true equivalence.
        g = torch.Generator().manual_seed(in_c * 100003 + out_c * 101 + k)
        w = torch.randn(out_c, in_c, k, generator=g)
        b = torch.randn(out_c, generator=g)
        conv.load_torch_state_dict({"conv.weight": w, "conv.bias": b})
        x = ttnn.from_torch(
            torch.randn(1, T, in_c, generator=g), device=device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        out = conv.forward(x)
        return out
    finally:
        if saved is not None:
            c3d._DEFAULT_BLOCKINGS[key] = saved
        elif blocking is not None and key in c3d._DEFAULT_BLOCKINGS:
            del c3d._DEFAULT_BLOCKINGS[key]


@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_vae_conv_sweep_worker(device, shape):
    """Sweep all blockings for one conv shape under start/stop signposts (profiled externally)."""
    from tracy import signpost

    in_c, out_c, k, T = shape
    # Correctness reference: fallback output.
    ref = ttnn.to_torch(_build_and_run(device, in_c, out_c, k, T, None)).float()

    valid = []
    for blk in _candidates(in_c):
        try:
            out = ttnn.to_torch(_build_and_run(device, in_c, out_c, k, T, blk)).float()
        except Exception:
            continue
        if _pcc(ref, out) >= 0.999:
            valid.append(blk)

    # Record the combo order so the orchestrator can line durations up with blockings.
    combo_file = os.environ.get("VAE_SWEEP_COMBO_FILE")
    if combo_file:
        with open(combo_file, "w") as f:
            json.dump({"shape": list(shape), "combos": [list(b) for b in valid]}, f)

    # Warmup all valid combos (compile), flushing periodically.
    for i, blk in enumerate(valid):
        _build_and_run(device, in_c, out_c, k, T, blk)
        if (i + 1) % PROFILER_DUMP_EVERY == 0:
            ttnn.ReadDeviceProfiler(device)
    ttnn.ReadDeviceProfiler(device)

    # Measured: one forward per combo, in order, between start/stop.
    signpost("start")
    for i, blk in enumerate(valid):
        out = _build_and_run(device, in_c, out_c, k, T, blk)
        ttnn.synchronize_device(device)
        if (i + 1) % PROFILER_DUMP_EVERY == 0:
            ttnn.ReadDeviceProfiler(device)
    signpost("stop")
    ttnn.ReadDeviceProfiler(device)


def _orchestrate():
    """Run the worker under the device profiler per shape; parse Conv3d kernel durations; print best."""
    from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler
    import pandas as pd

    best_table = {}
    for shape, sid in zip(SHAPES, SHAPE_IDS):
        combo_file = f"/tmp/vae_sweep_combos_{sid}.json"
        os.environ["VAE_SWEEP_COMBO_FILE"] = combo_file
        cmd = (
            f"pytest models/experimental/acestep/perf/sweep_vae_conv3d.py::test_vae_conv_sweep_worker "
            f"-k '{sid}' -q -s"
        )
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
        if not durs or not combos:
            print(f"{sid}: no data (durs={len(durs)} combos={len(combos)})")
            continue
        n = min(len(durs), len(combos))
        pairs = sorted(zip(durs[:n], combos[:n]), key=lambda x: x[0])
        best_ns, best_blk = pairs[0]
        worst_ns = pairs[-1][0]
        in_c, out_c, k, T = shape
        print(
            f"{sid}: best={best_ns/1e3:.1f}us {tuple(best_blk)}  worst={worst_ns/1e3:.1f}us  ({worst_ns/best_ns:.2f}x span, {n} combos)"
        )
        best_table[(in_c, out_c, k)] = tuple(best_blk)

    print("\n# External registration (acestep) — register_conv3d_configs(VAE_CONV3D_BLOCKINGS):")
    print("VAE_CONV3D_BLOCKINGS = {")
    for (i, o, k), blk in best_table.items():
        print(f"    ({i}, {o}, {k}): {blk},")
    print("}")
    json.dump({f"{i}_{o}_{k}": list(v) for (i, o, k), v in best_table.items()}, open(RESULTS_JSON, "w"))


if __name__ == "__main__":
    _orchestrate()
