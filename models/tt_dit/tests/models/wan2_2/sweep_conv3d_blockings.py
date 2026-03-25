#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Mesh-aware conv3d blocking sweep for Wan 2.2 VAE decoder.

Staged sweep strategy (spatial -> C_out -> C_in) with subprocess isolation
per config so device hangs don't kill the entire sweep.

Usage:
    source python_env/bin/activate && export PYTHONPATH=$(pwd)
    python models/tt_dit/tests/models/wan2_2/sweep_conv3d_blockings.py --mesh wh_4x8 --physical-mesh auto

    # Resume after crash (skips configs with existing JSON):
    python models/tt_dit/tests/models/wan2_2/sweep_conv3d_blockings.py --mesh wh_4x8 --physical-mesh auto

    # Force re-run all configs:
    python models/tt_dit/tests/models/wan2_2/sweep_conv3d_blockings.py --mesh wh_4x8 --physical-mesh auto --force
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path

WARMUP = 2
RUNS = 4

# (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)
Blocking = namedtuple("Blocking", ["cin", "cout", "t", "h", "w"])
RESULTS_DIR = Path(".cache/wan_conv3d_blocking_sweeps")
CONFIG_TIMEOUT = 600  # seconds per config subprocess

# ---------------------------------------------------------------------------
# Mesh configs: h_factor = mesh_shape[tp_axis], w_factor = mesh_shape[sp_axis]
# ---------------------------------------------------------------------------

MESH_CONFIGS = {
    "bh_2x2": dict(h_factor=2, w_factor=2, physical=(2, 2), desc="BH QB sp0tp1"),
    "bh_2x4": dict(h_factor=2, w_factor=4, physical=(2, 4), desc="BH LB sp1tp0"),
    "bh_4x8": dict(h_factor=4, w_factor=8, physical=(4, 8), desc="BH Galaxy sp1tp0"),
    "bh_4x32": dict(h_factor=4, w_factor=32, physical=(4, 32), desc="BH Galaxy 6U sp1tp0"),
    "wh_4x8": dict(h_factor=4, w_factor=8, physical=(4, 8), desc="WH Galaxy sp1tp0"),
}

RESOLUTIONS = {
    "480p": (4, 128, (480, 832)),
    "720p": (4, 128, (720, 1280)),
}

MESH_RESOLUTIONS = {
    "bh_2x2": ["480p"],
    "bh_2x4": ["480p"],
    "bh_4x8": ["480p", "720p"],
    "bh_4x32": ["480p", "720p"],
    "wh_4x8": ["480p", "720p"],
}

# Power-of-2 spatial candidates (non-pow2 values cause device hangs on some hardware)
HW_VARIATIONS = [(h, w) for h in [1, 2, 4, 8, 16, 32] for w in [1, 2, 4, 8, 16, 32] if h * w <= 256]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def valid_cin(c, padded_C_in, kernel_size):
    kT, kH, kW = kernel_size
    kernel_vol = kT * kH * kW
    return c >= 32 and c <= padded_C_in and padded_C_in % c == 0 and (kernel_vol * c) % 32 == 0


def valid_cout(c, C_out):
    from models.tt_dit.utils.conv3d import aligned_channels

    padded = aligned_channels(C_out)
    return c >= 32 and c <= padded and padded % c == 0 and c % 32 == 0


# ---------------------------------------------------------------------------
# Build per-device conv3d shapes (external padding included, matching production)
# ---------------------------------------------------------------------------


def build_conv_configs(h_factor, w_factor, resolution_key, cached=False):
    T0_uncached, T0_cached, (H_out, W_out) = RESOLUTIONS[resolution_key]
    T0 = T0_cached if cached else T0_uncached
    H0 = math.ceil(H_out // h_factor / 8) * 8
    W0 = W_out // w_factor
    tag = f"{'cached' if cached else 'uncached'}_{resolution_key}"

    # (case_id, T, H, W, C_in, kernel_size, C_out, Blocking baseline)
    B = Blocking
    return [
        # Latent resolution layers
        (f"384x384_k333_{tag}", 3, H0 // 8 + 2, W0 // 8 + 2, 384, (3, 3, 3), 384, B(128, 128, 1, 8, 2)),
        (f"384x32_k333_{tag}", 3, H0 // 8 + 2, W0 // 8 + 2, 384, (3, 3, 3), 32, B(384, 32, 1, 4, 4)),
        (f"32x384_k333_lat_{tag}", T0 // 4 + 2, H0 // 8 + 2, W0 // 8 + 2, 32, (3, 3, 3), 384, B(32, 384, 1, 8, 8)),
        # Time convs (no spatial padding for 3x1x1)
        (f"384x768_k311_{tag}", T0 // 4 + 2, H0 // 8, W0 // 8, 384, (3, 1, 1), 768, B(32, 32, 1, 1, 1)),
        (f"384x768_k311_up_{tag}", T0 // 2 + 2, H0 // 4, W0 // 4, 384, (3, 1, 1), 768, B(32, 32, 1, 1, 1)),
        (f"192x384_k311_{tag}", T0 // 2 + 2, H0 // 4, W0 // 4, 192, (3, 1, 1), 384, B(192, 128, 1, 1, 32)),
        # After first spatial upsample
        (f"192x384_k333_{tag}", 3, H0 // 4 + 2, W0 // 4 + 2, 192, (3, 3, 3), 384, B(96, 128, 1, 32, 1)),
        (f"384x384_k333_up_{tag}", 3, H0 // 4 + 2, W0 // 4 + 2, 384, (3, 3, 3), 384, B(128, 128, 1, 8, 2)),
        # After second upsample
        (f"96x192_k333_{tag}", 3, H0 // 2 + 2, W0 // 2 + 2, 96, (3, 3, 3), 192, B(96, 192, 1, 4, 4)),
        (f"192x192_k333_{tag}", 3, H0 // 2 + 2, W0 // 2 + 2, 192, (3, 3, 3), 192, B(96, 96, 1, 8, 4)),
        # 1x3x3 upsample convs
        (f"384x192_k133_{tag}", 3, H0 // 4 + 2, W0 // 4 + 2, 384, (1, 3, 3), 192, B(128, 64, 1, 32, 4)),
        (f"192x96_k133_{tag}", 3, H0 // 2 + 2, W0 // 2 + 2, 192, (1, 3, 3), 96, B(192, 96, 1, 8, 4)),
        # Output resolution layers
        (f"32x96_k333_{tag}", 3, H0 + 2, W0 + 2, 32, (3, 3, 3), 96, B(32, 96, 1, 4, 4)),
        (f"96x96_k333_{tag}", 3, H0 + 2, W0 + 2, 96, (3, 3, 3), 96, B(96, 96, 1, 8, 8)),
        (f"96x32_k333_{tag}", 3, H0 + 2, W0 + 2, 96, (3, 3, 3), 32, B(96, 32, 1, 16, 8)),
        (f"96x3_k333_{tag}", 3, H0 + 2, W0 + 2, 96, (3, 3, 3), 3, B(96, 32, 1, 16, 8)),
        # Cached temporal variants
        (f"384x384_k333_t_{tag}", T0 // 4 + 2, H0 // 8 + 2, W0 // 8 + 2, 384, (3, 3, 3), 384, B(128, 128, 1, 8, 2)),
        (f"192x384_k333_t_{tag}", T0 // 2 + 2, H0 // 4 + 2, W0 // 4 + 2, 192, (3, 3, 3), 384, B(96, 128, 1, 32, 1)),
        (f"384x384_k333_t2_{tag}", T0 // 2 + 2, H0 // 4 + 2, W0 // 4 + 2, 384, (3, 3, 3), 384, B(128, 128, 1, 8, 2)),
        (f"192x192_k333_t_{tag}", T0 + 2, H0 // 2 + 2, W0 // 2 + 2, 192, (3, 3, 3), 192, B(96, 96, 1, 8, 4)),
        (f"96x96_k333_t_{tag}", T0 + 2, H0 + 2, W0 + 2, 96, (3, 3, 3), 96, B(96, 96, 1, 8, 8)),
        (f"96x32_k333_t_{tag}", T0 + 2, H0 + 2, W0 + 2, 96, (3, 3, 3), 32, B(96, 32, 1, 16, 8)),
    ]


# ---------------------------------------------------------------------------
# Subprocess worker: sweeps one config, saves JSON, exits
# ---------------------------------------------------------------------------


def run_config_worker(config_json, physical_shape, result_path):
    """Runs in a subprocess. Opens device, sweeps all blockings for one config, saves result."""
    import torch

    import ttnn
    from models.tt_dit.utils.conv3d import aligned_channels

    cfg = json.loads(config_json)
    case_id = cfg["case_id"]
    T, H, W, C_in, C_out = cfg["T"], cfg["H"], cfg["W"], cfg["C_in"], cfg["C_out"]
    kernel_size = tuple(cfg["kernel_size"])
    baseline = Blocking(*cfg["baseline"])
    padded_C_in = aligned_channels(C_in)

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*physical_shape))
    grid_size = mesh_device.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    print(f"\n{'='*90}")
    print(f"  {case_id}: C_in={C_in}(pad={padded_C_in}) C_out={C_out} kernel={kernel_size}")
    print(f"  Shape: T={T} H={H} W={W}  Baseline: {baseline}")
    print(f"{'='*90}", flush=True)

    # Create tensors
    torch.manual_seed(42)
    tt_input = ttnn.from_torch(
        torch.randn(1, T, H, W, padded_C_in, dtype=torch.float32),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w = torch.randn(C_out, C_in, *kernel_size, dtype=torch.float32).permute(2, 3, 4, 1, 0)
    if padded_C_in != C_in:
        w = torch.nn.functional.pad(w, (0, 0, 0, padded_C_in - C_in))
    tt_weight = ttnn.from_torch(
        w.reshape(-1, C_out), device=mesh_device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, pad_value=0
    )
    tt_bias = ttnn.from_torch(
        torch.randn(1, C_out, dtype=torch.float32),
        device=mesh_device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        pad_value=0,
    )

    valid_cins = [c for c in range(32, padded_C_in + 1, 32) if valid_cin(c, padded_C_in, kernel_size)]
    valid_couts = [c for c in range(32, aligned_channels(C_out) + 1, 32) if valid_cout(c, C_out)]

    all_results = []
    best_us = float("inf")
    best_blocking = baseline
    baseline_us = None

    def run_conv3d(conv_config):
        out = ttnn.experimental.conv3d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            bias_tensor=tt_bias,
            config=conv_config,
            output_channels=C_out,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            padding_mode="zeros",
            dtype=ttnn.bfloat16,
            compute_kernel_config=ckc,
        )
        ttnn.synchronize_device(mesh_device)
        return out

    def try_blocking(blk, stage_name, is_baseline=False):
        nonlocal best_us, best_blocking, baseline_us
        label = f"Cin={blk.cin:3d} Cout={blk.cout:3d} T={blk.t} H={blk.h:2d} W={blk.w:2d}"
        try:
            conv_config = ttnn.Conv3dConfig(
                weights_dtype=ttnn.bfloat16,
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
                T_out_block=blk.t,
                W_out_block=blk.w,
                H_out_block=blk.h,
                C_out_block=blk.cout,
                C_in_block=blk.cin,
                compute_with_storage_grid_size=grid_size,
            )
            for _ in range(WARMUP):
                ttnn.deallocate(run_conv3d(conv_config))

            times = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                out = run_conv3d(conv_config)
                times.append((time.perf_counter() - t0) * 1e6)
                ttnn.deallocate(out)

            wall_us = statistics.median(times)
            tag = ""
            if wall_us < best_us:
                best_us = wall_us
                best_blocking = blk
                tag = " ** NEW BEST"
            if is_baseline:
                baseline_us = wall_us
                tag += " [baseline]"
            print(f"    {stage_name:8s} {label}  {wall_us:10.0f}us{tag}", flush=True)
            all_results.append(dict(stage=stage_name, blocking=list(blk), wall_us=wall_us, status="ok"))
        except Exception as e:
            err = str(e).split("\n")[0][:60]
            print(f"    {stage_name:8s} {label}  FAILED: {err}", flush=True)
            all_results.append(dict(stage=stage_name, blocking=list(blk), wall_us=float("inf"), status="error"))

    # Stage 0: baseline
    try_blocking(baseline, "baseline", is_baseline=True)

    # Stage 1: spatial sweep (skip h/w larger than actual dims)
    hw_candidates = [(h_val, w_val) for h_val, w_val in HW_VARIATIONS if h_val <= H and w_val <= W]
    print(
        f"  Stage 1: spatial sweep ({len(hw_candidates)} combos, Cin={baseline.cin}, Cout={baseline.cout})", flush=True
    )
    for h_val, w_val in hw_candidates:
        blk = Blocking(baseline.cin, baseline.cout, baseline.t, h_val, w_val)
        if blk != baseline:
            try_blocking(blk, "spatial")

    # Stage 2: C_out sweep with best spatial
    print(f"  Stage 2: C_out sweep ({len(valid_couts)} values, H={best_blocking.h}, W={best_blocking.w})", flush=True)
    for c_out in valid_couts:
        blk = Blocking(baseline.cin, c_out, baseline.t, best_blocking.h, best_blocking.w)
        if blk not in (best_blocking, baseline):
            try_blocking(blk, "c_out")

    # Stage 3: C_in sweep with best C_out + spatial
    print(f"  Stage 3: C_in sweep ({len(valid_cins)} values, Cout={best_blocking.cout})", flush=True)
    for c_in in valid_cins:
        blk = Blocking(c_in, best_blocking.cout, baseline.t, best_blocking.h, best_blocking.w)
        if blk not in (best_blocking, baseline):
            try_blocking(blk, "c_in")

    # Cleanup and save
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_weight)
    ttnn.deallocate(tt_bias)

    print(f"\n  >>> BEST: {best_blocking}  {best_us:.0f}us", end="")
    if baseline_us:
        print(f"  (vs baseline {baseline_us:.0f}us = {baseline_us / best_us:.2f}x)")
    else:
        print()

    result = dict(
        case_id=case_id,
        C_in=C_in,
        C_out=C_out,
        kernel=list(kernel_size),
        T=T,
        H=H,
        W=W,
        baseline=list(baseline),
        baseline_us=baseline_us,
        best_blocking=list(best_blocking),
        best_us=best_us,
        results=all_results,
    )
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())

    os._exit(0)  # bypass ttnn teardown which can hang


# ---------------------------------------------------------------------------
# Orchestrator: spawns one subprocess per config, handles timeouts
# ---------------------------------------------------------------------------


def reset_devices():
    """Reset all TT devices after a hang."""
    import ttnn

    num = ttnn.get_num_devices()
    ids = ",".join(str(i) for i in range(num))
    print(f"  Resetting {num} devices...", flush=True)
    subprocess.run(["tt-smi", "-r", ids], capture_output=True, timeout=120)
    print("  Devices reset.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Conv3d blocking sweep with subprocess isolation")
    parser.add_argument("--mesh", default="bh_4x8", choices=list(MESH_CONFIGS.keys()))
    parser.add_argument(
        "--physical-mesh",
        default=None,
        help="Physical mesh shape: '2x4', '4x8', or 'auto' (opens target mesh). Default: 1x1",
    )
    parser.add_argument("--force", action="store_true", help="Re-run configs even if JSON exists")
    parser.add_argument("--timeout", type=int, default=CONFIG_TIMEOUT, help="Timeout per config (seconds)")
    # Internal flag: run a single config in-process (used by subprocess)
    parser.add_argument("--_run-config", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_physical-shape", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_result-path", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Subprocess mode: run single config and exit
    if args._run_config:
        shape = tuple(int(x) for x in args._physical_shape.split("x"))
        run_config_worker(args._run_config, shape, args._result_path)
        return

    # Orchestrator mode
    mesh_cfg = MESH_CONFIGS[args.mesh]
    h_factor, w_factor = mesh_cfg["h_factor"], mesh_cfg["w_factor"]
    resolutions = MESH_RESOLUTIONS[args.mesh]

    if args.physical_mesh is None:
        physical_shape = (1, 1)
    elif args.physical_mesh == "auto":
        physical_shape = mesh_cfg["physical"]
    else:
        physical_shape = tuple(int(x) for x in args.physical_mesh.split("x"))

    print(f"Target mesh: {args.mesh} ({mesh_cfg['desc']}), h_factor={h_factor}, w_factor={w_factor}")
    print(f"Physical mesh: {physical_shape[0]}x{physical_shape[1]}")
    print(f"Resolutions: {resolutions}")
    print(f"Timeout per config: {args.timeout}s")

    results_dir = RESULTS_DIR / args.mesh
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build and dedup configs
    all_configs = []
    for res in resolutions:
        all_configs.extend(build_conv_configs(h_factor, w_factor, res, cached=False))
        all_configs.extend(build_conv_configs(h_factor, w_factor, res, cached=True))

    seen = {}
    deduped = []
    for cfg in all_configs:
        case_id, T, H, W, C_in, kernel, C_out, baseline = cfg
        key = (C_in, C_out, kernel, T, H, W)
        if key not in seen:
            seen[key] = case_id
            deduped.append(cfg)
    print(f"Total unique configs: {len(deduped)} (from {len(all_configs)} before dedup)")

    summary = {}
    script = os.path.abspath(__file__)
    shape_str = f"{physical_shape[0]}x{physical_shape[1]}"

    for i, (case_id, T, H, W, C_in, kernel_size, C_out, baseline_tuple) in enumerate(deduped):
        result_file = results_dir / f"{case_id}.json"

        if result_file.exists() and not args.force:
            with open(result_file) as f:
                summary[case_id] = json.load(f)
            print(f"  [{i+1}/{len(deduped)}] Skipping {case_id} (already done)")
            continue

        print(f"\n  [{i+1}/{len(deduped)}] Spawning subprocess for {case_id}...")

        config_json = json.dumps(
            dict(
                case_id=case_id,
                T=T,
                H=H,
                W=W,
                C_in=C_in,
                C_out=C_out,
                kernel_size=list(kernel_size),
                baseline=list(baseline_tuple),
            )
        )

        cmd = [
            sys.executable,
            script,
            "--_run-config",
            config_json,
            "--_physical-shape",
            shape_str,
            "--_result-path",
            str(result_file),
        ]

        try:
            proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
            proc.wait(timeout=args.timeout)
            if proc.returncode == 0 and result_file.exists():
                with open(result_file) as f:
                    summary[case_id] = json.load(f)
            else:
                print(f"  FAILED: subprocess exited with code {proc.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {args.timeout}s — killing subprocess")
            proc.kill()
            proc.wait()
            try:
                reset_devices()
            except Exception as e:
                print(f"  Device reset failed: {e}")

    # Final summary
    print(f"\n{'='*90}")
    print(f"  SUMMARY — {args.mesh}")
    print(f"{'='*90}")
    for case_id, s in summary.items():
        if isinstance(s, dict) and "best_blocking" in s:
            bu = s.get("baseline_us")
            be = s.get("best_us", 0)
            bu_str = f"{bu:.0f}us" if bu else "N/A"
            be_str = f"{be:.0f}us" if be else "N/A"
            sp = f"{bu/be:.2f}x" if bu and be and be > 0 else "N/A"
            print(f"  {case_id:40s} baseline={bu_str:>8s}  best={be_str:>8s}  {sp:>6s}  {s['best_blocking']}")


if __name__ == "__main__":
    main()
