#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Minimal multi-rank smoke test for verifying tracy profiling via tt-run.

Each rank opens a single device, runs a few TTNN ops, and exits.
The workload is intentionally simple: the goal is to verify that tracy
wrapping, per-rank output dirs, and per-rank ports all work correctly.

Usage (T3K, 2 ranks, with tracy):
    tt-run --bare --mpi-args "--allow-run-as-root" \
           --rank-binding tests/ttnn/distributed/config/t3k_tracy_smoke_rank_bindings.yaml \
           --tracy "-r" \
           python3 tests/ttnn/distributed/test_tracy_multi_host_smoke.py
"""

import os
import sys
from pathlib import Path

T3K_MIN_PCIE_DEVICES = 4


def require_t3k():
    """Exit early if not running on a T3K (requires at least 4 PCIe devices)."""
    tenstorrent_devs = list(Path("/dev/tenstorrent").iterdir()) if Path("/dev/tenstorrent").is_dir() else []
    if len(tenstorrent_devs) < T3K_MIN_PCIE_DEVICES:
        print(f"Skipping: T3K required ({T3K_MIN_PCIE_DEVICES} PCIe devices) but found {len(tenstorrent_devs)}")
        sys.exit(0)


def main():
    require_t3k()

    import torch
    import ttnn

    rank = os.environ.get("OMPI_COMM_WORLD_RANK", "?")
    mesh_id = os.environ.get("TT_MESH_ID", "?")
    tracy_port = os.environ.get("TRACY_PORT", "N/A")
    profiler_dir = os.environ.get("TT_METAL_PROFILER_DIR", "N/A")

    print(f"[rank {rank}] mesh_id={mesh_id}  tracy_port={tracy_port}  profiler_dir={profiler_dir}")

    device = ttnn.open_device(device_id=0)

    torch.manual_seed(42)
    torch_input = torch.randn(1, 1, 256, 256, dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    x = ttnn.relu(ttnn_input)
    x = ttnn.exp(x)
    x = ttnn.gelu(x)

    ttnn.synchronize_device(device)

    result = ttnn.to_torch(ttnn.from_device(x))
    print(f"[rank {rank}] output shape={result.shape}  sample={result[0, 0, 0, :4].tolist()}")

    ttnn.close_device(device)
    print(f"[rank {rank}] done")


if __name__ == "__main__":
    main()
