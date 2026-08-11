# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tracy benchmark: cost of `load_sub_device_manager` + `clear_loaded_sub_device_manager`.

Single iteration with three phases marked by the same signposts in both modes:
    phase1_pre:    2 ops on full device
    phase2_middle: 2 ops — on full device (no_sd) OR on shared sub-device with
                   load/clear bracketing it (with_sd)
    phase3_post:   2 ops on full device

Each pytest invocation runs 1 warmup iteration (no signposts, not measured —
populates the program cache) followed by 10 measured iterations of the same
body. Compare the duration of `phase2_middle_start`→`phase2_middle_end` between
the two captures to see the load+clear overhead.

Run as a script: drives `python -m tracy ... -m pytest <self>` and prints stats
parsed from the resulting tracy_ops_data.csv.

    python models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_sub_device_load_clear_timing.py            # no_sd
    python models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_sub_device_load_clear_timing.py --with-sd  # with_sd

Or invoke pytest directly (no stats):
    pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_sub_device_load_clear_timing.py -k "no_sd"
"""

import os
import pathlib
import re
import sys

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import profiler

# Path the test writes its host-profiler samples to so the wrapper can include
# them in the final stats summary alongside the tracy-derived numbers.
HOST_PROFILER_FILE = pathlib.Path("/tmp/.tt_sub_device_host_profiler.txt")


def _iters_from_argv():
    """Wrapper encodes --iters into the pytest -k filter as `iters_<N>`. Read it
    back from sys.argv at module import time so the value flows into parametrize
    without needing env vars, conftest, or sidecar files."""
    for a in sys.argv:
        m = re.search(r"iters_(\d+)", a)
        if m:
            return int(m.group(1))
    return 10


_ITERS = _iters_from_argv()


def _two_ops_full(mesh_device, x, w, ckc):
    out = ttnn.matmul(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc)
    out2 = ttnn.matmul(out, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    return out2


def _two_ops_subdevice(mesh_device, x, w, sd_id, core_grid, ckc):
    out = ttnn.matmul(
        x,
        w,
        core_grid=core_grid,
        sub_device_id=sd_id,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=ckc,
    )
    out2 = ttnn.matmul(
        out,
        w,
        core_grid=core_grid,
        sub_device_id=sd_id,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=ckc,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    return out2


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (4, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("dispatch_sd_rows", [2])
@pytest.mark.parametrize("with_sd", [False, True], ids=["no_sd", "with_sd"])
@pytest.mark.parametrize("bench_iters", [_ITERS], ids=[f"iters_{_ITERS}"])
def test_sub_device_load_clear_tracy(
    mesh_device, device_params, dispatch_sd_rows, with_sd, bench_iters, is_ci_env, is_ci_v2_env
):
    if is_ci_env or is_ci_v2_env:
        pytest.skip("Skip tracy benchmark in CI")

    mesh_device.enable_program_cache()
    torch.manual_seed(0)

    grid = mesh_device.compute_with_storage_grid_size()
    grid_x, grid_y = grid.x, grid.y
    assert 0 < dispatch_sd_rows < grid_y

    dispatch_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, dispatch_sd_rows - 1))}
    )
    shared_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, dispatch_sd_rows), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
    )
    sd_manager_id = mesh_device.create_sub_device_manager(
        [ttnn.SubDevice([dispatch_cores]), ttnn.SubDevice([shared_cores])], 0
    )
    shared_sd_id = ttnn.SubDeviceId(1)
    shared_grid = ttnn.CoreGrid(x=grid_x, y=grid_y - dispatch_sd_rows)

    M, K, N = 512, 1024, 1024
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    x = ttnn.from_torch(
        torch.randn(1, 1, M, K, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, K, N, dtype=torch.bfloat16) * 0.02,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )

    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    logger.info(
        f"Mesh={tuple(mesh_device.shape)}, full grid={grid_x}x{grid_y}, "
        f"shared_sd grid={shared_grid.x}x{shared_grid.y}, with_sd={with_sd}"
    )

    # Case suffix tags every signpost so the perf report identifies which case
    # this capture belongs to (handy when comparing reports for no_sd vs with_sd).
    tag = "with_sd" if with_sd else "no_sd"
    profiler_key = f"phase2_middle_{tag}"
    profiler.clear()

    # 1 warmup iteration (no signposts -> not included in stats) + bench_iters measured.
    for i in range(1 + bench_iters):
        record = i > 0
        if record:
            signpost(f"phase1_pre_start_{tag}")
        out = _two_ops_full(mesh_device, x, w, ckc)
        ttnn.deallocate(out)
        if record:
            signpost(f"phase1_pre_end_{tag}")

        if record:
            ttnn.synchronize_device(mesh_device)
            signpost(f"phase2_middle_start_{tag}")
            profiler.start(profiler_key)
        if with_sd:
            mesh_device.load_sub_device_manager(sd_manager_id)
            out = _two_ops_subdevice(mesh_device, x, w, shared_sd_id, shared_grid, ckc)
            ttnn.deallocate(out)
            mesh_device.clear_loaded_sub_device_manager()
        else:
            out = _two_ops_full(mesh_device, x, w, ckc)
            ttnn.deallocate(out)
        if record:
            ttnn.synchronize_device(mesh_device)
            profiler.end(profiler_key)
            signpost(f"phase2_middle_end_{tag}")

        if record:
            signpost(f"phase3_post_start_{tag}")
        out = _two_ops_full(mesh_device, x, w, ckc)
        ttnn.deallocate(out)
        if record:
            signpost(f"phase3_post_end_{tag}")

    # Write per-iteration host-side durations (us) so the wrapper can build stats
    # alongside the tracy-derived ones at the end.
    samples_us = [d * 1e6 for d in profiler.times.get(profiler_key, [])]
    HOST_PROFILER_FILE.write_text("\n".join(f"{v:.3f}" for v in samples_us))


# -------------------------------------------------------------------------
# Script entrypoint: run tracy + pytest on this file, then parse CSV stats.
# -------------------------------------------------------------------------

TRACY_CSV = os.path.join(os.environ.get("TT_METAL_HOME", ""), "generated/profiler/.logs/tracy_ops_data.csv")


def _filter_outliers(values):
    """Drop samples outside Tukey's fence: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    import statistics

    if len(values) < 4:
        return values, []
    q1, _, q3 = statistics.quantiles(values, n=4)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    kept = [v for v in values if lo <= v <= hi]
    dropped = [v for v in values if v < lo or v > hi]
    return kept, dropped


def _parse_phase2_durations():
    """Read the most-recent tracy CSV and return the list of phase2_middle durations (us)."""
    import csv
    from pathlib import Path

    p = Path(TRACY_CSV)
    if not p.exists():
        raise SystemExit(f"tracy CSV not found: {p}")

    starts, ends = [], []
    with p.open() as f:
        next(f)  # header
        for row in csv.reader(f, delimiter=";"):
            if len(row) < 2:
                continue
            name = row[0].strip("`")
            ts = int(row[1])
            if "TT_SIGNPOST: phase2_middle_start" in name:
                starts.append(ts)
            elif "TT_SIGNPOST: phase2_middle_end" in name:
                ends.append(ts)

    if len(starts) != len(ends) or not starts:
        raise SystemExit(f"start/end mismatch: starts={len(starts)} ends={len(ends)}")

    return [(e - s) / 1000.0 for s, e in zip(starts, ends)]


def _print_stats_block(label, durations_us):
    import statistics

    kept, dropped = _filter_outliers(durations_us)
    print(f"  [{label}] samples={len(durations_us)} kept={len(kept)} dropped={len(dropped)}")
    if dropped:
        print(f"  [{label}] dropped (us): {[f'{d:.1f}' for d in dropped]}")
    if len(kept) < 2:
        print(f"  [{label}] (not enough samples after filtering for stats)")
        return
    print(
        f"  [{label}] mean={statistics.mean(kept):.2f} median={statistics.median(kept):.2f} "
        f"stdev={statistics.stdev(kept):.2f} min={min(kept):.2f} max={max(kept):.2f}"
    )


def _print_stats_for(case, tracy_durations_us, host_durations_us):
    print()
    print(f"=== phase2_middle stats ({case}) ===")
    _print_stats_block("tracy", tracy_durations_us)
    if host_durations_us:
        _print_stats_block("host ", host_durations_us)


def _run_case(case, iters):
    """Run tracy+pytest for one case and return (tracy_durations_us, host_durations_us)."""
    import subprocess

    print(f"Running tracy capture for case={case}, iters={iters} ...")
    nodeid = f"{__file__}::test_sub_device_load_clear_tracy"
    # Tracy's wrapper does `" ".join(args)` and runs via shell=True, so we embed
    # quotes into the -k argument here to keep "<case> and iters_<N>" as a single
    # token after the shell re-parses it.
    cmd = [
        sys.executable,
        "-m",
        "tracy",
        "-r",
        "-p",
        "-m",
        "pytest",
        nodeid,
        "-k",
        f'"{case} and iters_{iters}"',
    ]
    HOST_PROFILER_FILE.unlink(missing_ok=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(rc)
    tracy_durations = _parse_phase2_durations()
    host_durations = []
    if HOST_PROFILER_FILE.exists():
        host_durations = [float(x) for x in HOST_PROFILER_FILE.read_text().split() if x.strip()]
        HOST_PROFILER_FILE.unlink(missing_ok=True)
    return tracy_durations, host_durations


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--no-sd", action="store_true", help="run the no_sd case (default)")
    mode.add_argument("--with-sd", action="store_true", help="run the with_sd case")
    mode.add_argument("--both", action="store_true", help="run both no_sd and with_sd")
    ap.add_argument("--iters", type=int, default=10, help="measured iterations per case (default: 10)")
    args = ap.parse_args()

    cases = ["no_sd", "with_sd"] if args.both else (["with_sd"] if args.with_sd else ["no_sd"])
    results = {case: _run_case(case, args.iters) for case in cases}
    for case in cases:
        tracy_d, host_d = results[case]
        _print_stats_for(case, tracy_d, host_d)
