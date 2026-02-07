# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Reduce scatter async stress test -- runs 50,000 iterations in a single process.

Reproduces the exact parameters from the failing CI job:
  [WH-T3K][v1] Multi-process reduce scatter test model

The failure was a non-deterministic hang in run_reduce_scatter_impl() with:
  RuntimeError: TT_THROW @ tt_metal/impl/dispatch/system_memory_manager.cpp:627
  TIMEOUT: device timeout, potential hang detected, the device is unrecoverable

Parameters matched from models/vllm_test_utils/t3000_multiproc_test/test_model.py:
  - mesh: 2x4, submesh: 1x4
  - rs_input_shape: [1, 1, 8, 7168]
  - dim: 3, cluster_axis: 1
  - topology: Linear (FABRIC_1D)
  - enable_trace: False, num_iters: 3
  - use_barrier: True, use_persistent_buffers: False
  - bfloat16, DRAM interleaved, TILE layout

The device is opened ONCE and the reduce scatter op is looped inside the process.
On failure, the script exits with code 1 so the bash wrapper can reset and restart.
"""

import os
import sys
import time
import datetime

TOTAL_ITERATIONS = int(os.environ.get("STRESS_TOTAL_ITERATIONS", 50000))
SUMMARY_LOG = os.environ.get("STRESS_SUMMARY_LOG", "/dev/null")
START_ITERATION = int(os.environ.get("STRESS_START_ITERATION", 1))

print("Importing ttnn...", flush=True)
import ttnn

print("Importing test module...", flush=True)
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl


def log_summary(msg):
    """Write to summary log file (appending)."""
    with open(SUMMARY_LOG, "a") as f:
        f.write(msg + "\n")
        f.flush()


def log_both(msg):
    """Write to stdout and summary log."""
    print(msg, flush=True)
    log_summary(msg)


def main():
    # --- Device setup (done once) ---
    print("Setting up fabric...", flush=True)
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )

    print("Opening 2x4 mesh device...", flush=True)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(2, 4),
        trace_region_size=1171456,
    )

    print("Creating 1x4 submesh...", flush=True)
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 4))
    num_devices = submesh_device.get_num_devices()

    log_both(f"Device ready. Starting stress test from iteration {START_ITERATION} to {TOTAL_ITERATIONS}.")

    # --- Stress loop ---
    pass_count = 0
    fail_iteration = None
    iter_times = []

    for i in range(START_ITERATION, TOTAL_ITERATIONS + 1):
        t0 = time.time()

        try:
            run_reduce_scatter_impl(
                submesh_device,
                num_devices,
                rs_input_shape=[1, 1, 8, 7168],
                dim=3,
                num_links=1,
                rs_input_dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mem_config_input=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
                mem_config_rs=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
                rs_topology=ttnn.Topology.Linear,
                enable_trace=False,
                num_iters=3,
                ones_tensor=False,
                use_barrier=True,
                use_persistent_buffers=False,
                cluster_axis=1,
            )
        except Exception as e:
            elapsed = time.time() - t0
            log_both(f"!!! FAILURE on iteration {i} after {elapsed:.1f}s !!!")
            log_both(f"Error: {e}")
            fail_iteration = i
            break

        elapsed = time.time() - t0
        iter_times.append(elapsed)
        pass_count += 1

        # Per-iteration output (goes to terminal + full log via stdbuf | tee)
        print(f"PASS iteration {i}/{TOTAL_ITERATIONS} ({elapsed:.2f}s)", flush=True)

        # Periodic summary every 100 iterations
        if i % 100 == 0:
            avg_time = sum(iter_times[-100:]) / len(iter_times[-100:])
            remaining = TOTAL_ITERATIONS - i
            eta_seconds = remaining * avg_time
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            log_both(f"=== SUMMARY at iteration {i}: {pass_count} passed, avg {avg_time:.2f}s/iter, ETA: {eta_str} ===")

    # --- Teardown ---
    print("Closing devices...", flush=True)
    try:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception as e:
        print(f"Warning: cleanup error (expected after hang): {e}", flush=True)

    if fail_iteration is not None:
        log_both(f"STOPPED: failed on iteration {fail_iteration} ({pass_count} passed before failure)")
        sys.exit(1)
    else:
        log_both(f"COMPLETED: all {pass_count} iterations passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
