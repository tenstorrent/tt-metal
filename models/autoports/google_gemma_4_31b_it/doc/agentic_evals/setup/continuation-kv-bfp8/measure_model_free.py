# SPDX-License-Identifier: Apache-2.0
"""Measure the normal model allocation before any serving KV pool exists."""

import argparse
import json
import time
from pathlib import Path

import torch
from readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device

import ttnn
from models.autoports.google_gemma_4_31b_it.tt.model import Gemma4Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    mesh = open_readiness_mesh_device("P300x2", "FABRIC_1D_RING")
    started = time.time()
    try:
        model = Gemma4Model(mesh, precision_config=str(args.precision))
        assert len(model.layers) == 60
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        allocator = {
            key: int(getattr(view, key))
            for key in (
                "num_banks",
                "total_bytes_per_bank",
                "total_bytes_allocated_per_bank",
                "total_bytes_free_per_bank",
                "largest_contiguous_bytes_free_per_bank",
            )
        }
        result = {
            "measured_unix": time.time(),
            "load_wall_s": time.time() - started,
            "precision": model.precision_config,
            "device_ids": list(mesh.get_device_ids()),
            "allocator": allocator,
            "model_allocations": model.allocation_bytes,
            "scope": "Normal Gemma4Model construction, same path as adapter; no KV allocation, generator or device profiler. Shared mesh allocator applies uniformly; not independent device samples. Trace region already excluded. Serving must revalidate its actual launch after reserves.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        close_readiness_mesh_device(mesh)


if __name__ == "__main__":
    main()
