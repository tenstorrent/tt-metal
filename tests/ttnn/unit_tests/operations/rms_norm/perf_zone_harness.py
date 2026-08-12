# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage (MaybeDeviceZoneScope) + whole-op measurement harness for rms_norm.

Companion to `test_rms_norm_perf.py`, which measures the whole-op number only.
This one exists because the per-stage breakdown needs the RAW device profiler log
(`generated/profiler/.logs/profile_log_device.csv`), which `ttnn.ReadDeviceProfiler`
dumps in-process — no Tracy wrapper, one device session per shape set.

Run it (env vars must be set BEFORE device init, hence on the command line):

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
    scripts/tt-probe.sh rms_norm <<'EOF'
    from tests.ttnn.unit_tests.operations.rms_norm.perf_zone_harness import main
    main(["decode7168"])
    EOF

Everything is pinned to the perf loose cases' config: bf16 / TILE / gamma bf16
TILE / fp32_dest_acc_en=False / MathFidelity.HiFi2 (feature_spec `_PERF_BASE` +
`extras.math_fidelity`). Correctness is NOT asserted here — this file is also the
harness for ABLATED kernels, whose output is wrong by design.
"""

from __future__ import annotations

import os
import shutil
import time

import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm

_ML = ttnn.TensorMemoryLayout

# name -> (rows, hidden, memory_layout, shard_shape, core_grid)
CASES = {
    # ---- interleaved DRAM (the perf loose cases) ----
    "decode1024": (32, 1024, _ML.INTERLEAVED, None, None),
    "decode2304": (32, 2304, _ML.INTERLEAVED, None, None),
    "decode5120": (32, 5120, _ML.INTERLEAVED, None, None),
    # THE FOCUS SHAPE: the perf-flagged case (minimum_expected_speedup = 7.0).
    "decode7168": (32, 7168, _ML.INTERLEAVED, None, None),
    "prefill1024": (8192, 1024, _ML.INTERLEAVED, None, None),
    "prefill2304": (8192, 2304, _ML.INTERLEAVED, None, None),
    "prefill5120": (8192, 5120, _ML.INTERLEAVED, None, None),
    "prefill7168": (8192, 7168, _ML.INTERLEAVED, None, None),
    # ---- the pinned sharded geometries ----
    "wshard1024": (32, 1024, _ML.WIDTH_SHARDED, [32, 128], (8, 1)),
    "wshard2304": (32, 2304, _ML.WIDTH_SHARDED, [32, 256], (9, 1)),
    "wshard5120": (32, 5120, _ML.WIDTH_SHARDED, [32, 160], (8, 4)),
    "wshard7168": (32, 7168, _ML.WIDTH_SHARDED, [32, 256], (7, 4)),
    "bshard1024": (8192, 1024, _ML.BLOCK_SHARDED, [1024, 128], (8, 8)),
    # ---- guard-set representatives (distinct kernel paths) ----
    "rm_interleaved": (128, 512, _ML.INTERLEAVED, None, None),  # ROW_MAJOR legs
    "hshard512": (256, 512, _ML.HEIGHT_SHARDED, [32, 512], (1, 8)),
    "wtail": (32, 2047, _ML.INTERLEAVED, None, None),  # ragged hidden tile
}

RM_CASES = {"rm_interleaved"}


def _tensors(device, rows, hidden, memory_layout, shard_shape, core_grid, row_major=False):
    torch.manual_seed(0)
    shape = (1, 1, rows, hidden)
    layout = ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, hidden), dtype=torch.float32).to(torch.bfloat16)
    memory_config = None
    if memory_layout != _ML.INTERLEAVED:
        from eval.sharding import shard_config

        memory_config = shard_config(
            shard_shape, core_grid, memory_layout, layout=layout, dtype=ttnn.bfloat16, device=device
        )
    kw = {} if memory_config is None else {"memory_config": memory_config}
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, **kw)
    tt_gamma = ttnn.from_torch(torch_gamma, dtype=ttnn.bfloat16, layout=layout, device=device)
    return tt_input, tt_gamma


def measure(device, name, row_major=False):
    """One fresh run of `name`; returns DEVICE KERNEL DURATION [ns] (max over programs)."""
    rows, hidden, memory_layout, shard_shape, core_grid = CASES[name]
    tt_input, tt_gamma = _tensors(
        device, rows, hidden, memory_layout, shard_shape, core_grid, row_major=row_major or name in RM_CASES
    )
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    kw = {} if memory_layout == _ML.INTERLEAVED else {"memory_config": tt_input.memory_config()}

    ttnn.ReadDeviceProfiler(device)  # flush prep (from_torch / interleaved_to_sharded)
    out = rms_norm(tt_input, gamma=tt_gamma, epsilon=1e-6, compute_kernel_config=cfg, **kw)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    # Same extraction as eval/profiling.py: the per-program analysis dict, keyed
    # on "DEVICE KERNEL DURATION [ns]"; take the DOMINANT program (the op).
    ns = None
    for programs in per_chip.values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get("DEVICE KERNEL DURATION [ns]")
            if entry is None:
                continue
            d = float(entry.duration)
            ns = d if ns is None else max(ns, d)
    del out
    return ns, per_chip


def main(names, keep_log=True, row_major=False):
    device = ttnn.open_device(device_id=0)
    logdir = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", ".logs")
    try:
        for name in names:
            ns, data = measure(device, name, row_major=row_major)
            print(f"ZONE_HARNESS {name}: device_kernel_ns={ns} programs={len(data) if data else 0}")
            if keep_log:
                src = os.path.join(logdir, "profile_log_device.csv")
                if os.path.exists(src):
                    dst = os.path.join(logdir, f"zones_{name}.csv")
                    shutil.copyfile(src, dst)
                    print(f"ZONE_HARNESS {name}: zones -> {dst} ({os.path.getsize(dst)} bytes)")
            time.sleep(0.2)
    finally:
        ttnn.close_device(device)
