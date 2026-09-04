# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Analytically-known workloads for the perf-counter utilization sanity gate
(Phase 0 of the Nsight-counters plan).

Two regimes with closed-form ideal cost:
  * a square matmul (M=K=N) at a fixed math fidelity -> known FLOPs, runs
    compute-bound on the full grid.
  * a large element-wise add over DRAM-resident tensors -> known bytes
    moved (2 reads + 1 write), runs bandwidth-bound.

Run as a subprocess under ``python -m tracy`` so the device profiler captures
the ops. The pytest parent (``test_counter_utilization_sanity.py``) sets
``RUN_COUNTER_SANITY_WORKLOAD=1`` and parses the resulting CSV against the
manifest this module writes to ``COUNTER_SANITY_MANIFEST``.

Each op runs one untimed warmup (program-cache fill) followed by
``COUNTER_SANITY_ITERS`` profiled iterations; the parent ranks the warm
window, never the cold first iteration.
"""

import json
import os

import pytest
import torch

import ttnn

_WORKLOAD_ENABLED = os.environ.get("RUN_COUNTER_SANITY_WORKLOAD") == "1"

# Square matmul side and eltwise side. Both chosen large enough that the op
# saturates its bound (compute / bandwidth) rather than dispatch overhead.
MATMUL_SIDE = int(os.environ.get("COUNTER_SANITY_MATMUL_SIDE", "4096"))
ELTWISE_SIDE = int(os.environ.get("COUNTER_SANITY_ELTWISE_SIDE", "8192"))
ITERS = int(os.environ.get("COUNTER_SANITY_ITERS", "5"))

# Op codes as they appear in the ops_perf_results CSV OP CODE column.
MATMUL_OP_CODE = "Matmul"
ELTWISE_OP_CODE = "BinaryDeviceOperation"


def _manifest_path():
    return os.environ.get("COUNTER_SANITY_MANIFEST")


def _write_manifest(device):
    path = _manifest_path()
    if not path:
        return
    arch = str(device.arch()).split(".")[-1].lower()
    manifest = {
        "arch": arch,
        "math_fidelity": "HiFi2",
        "matmul": {
            "op_code": MATMUL_OP_CODE,
            "M": MATMUL_SIDE,
            "K": MATMUL_SIDE,
            "N": MATMUL_SIDE,
            # 2 FLOP per multiply-accumulate.
            "flops": 2 * MATMUL_SIDE * MATMUL_SIDE * MATMUL_SIDE,
        },
        "eltwise": {
            "op_code": ELTWISE_OP_CODE,
            "side": ELTWISE_SIDE,
            # bf16 = 2 bytes; add reads two operands and writes one result.
            "bytes_moved": 3 * ELTWISE_SIDE * ELTWISE_SIDE * 2,
        },
        "iters": ITERS,
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def _hifi2_compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


@pytest.mark.skipif(not _WORKLOAD_ENABLED, reason="driven only by test_counter_utilization_sanity under tracy")
def test_compute_bound_matmul():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
    try:
        _write_manifest(device)
        torch.manual_seed(0)
        a = ttnn.from_torch(
            torch.randn(MATMUL_SIDE, MATMUL_SIDE, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = ttnn.from_torch(
            torch.randn(MATMUL_SIDE, MATMUL_SIDE, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        compute_config = _hifi2_compute_config()

        def run():
            return ttnn.matmul(a, b, compute_kernel_config=compute_config)

        # Untimed warmup populates the program cache before the profiled window.
        ttnn.synchronize_device(device)
        run()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)

        for _ in range(ITERS):
            run()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    finally:
        ttnn.close_mesh_device(device)


@pytest.mark.skipif(not _WORKLOAD_ENABLED, reason="driven only by test_counter_utilization_sanity under tracy")
def test_bandwidth_bound_eltwise():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
    try:
        _write_manifest(device)
        torch.manual_seed(0)
        a = ttnn.from_torch(
            torch.randn(ELTWISE_SIDE, ELTWISE_SIDE, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = ttnn.from_torch(
            torch.randn(ELTWISE_SIDE, ELTWISE_SIDE, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def run():
            return ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ttnn.synchronize_device(device)
        run()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)

        for _ in range(ITERS):
            run()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    finally:
        ttnn.close_mesh_device(device)
