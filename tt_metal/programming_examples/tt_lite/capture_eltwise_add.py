# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Python capture script for eltwise-add trace.
Equivalent to capture_and_dump.cpp but using ttnn Python API.
Validates the Python export pipeline: ttnn trace → get_trace_data → trace_binary.py → .ttb

Usage:
    python capture_eltwise_add.py [output.ttb]
"""

import sys
import os

import torch

import ttnn
from trace_binary import export_trace


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "eltwise_add.ttb"

    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

    num_tiles = 4
    shape = (1, 1, 32, 32 * num_tiles)

    torch_a = torch.full(shape, 5.0, dtype=torch.bfloat16)
    torch_b = torch.full(shape, 3.0, dtype=torch.bfloat16)

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Compile run
    tt_out = ttnn.add(tt_a, tt_b)
    result = ttnn.to_torch(tt_out)
    expected = 8.0
    if torch.allclose(result, torch.full_like(result, expected)):
        print(f"Compile run: PASS (all values = {expected})")
    else:
        print("Compile run: FAIL")
        ttnn.close_mesh_device(device)
        return 1

    # Deallocate compile output so addresses match during trace
    ttnn.deallocate(tt_out, force=True)

    # Capture trace
    print("Capturing trace...")
    trace_id = ttnn.begin_trace_capture(device)
    tt_out = ttnn.add(tt_a, tt_b)
    ttnn.end_trace_capture(device, trace_id)
    print("Trace captured.")

    # Verify trace replay
    ttnn.execute_trace(device, trace_id, blocking=True)
    result = ttnn.to_torch(tt_out)
    if torch.allclose(result, torch.full_like(result, expected)):
        print(f"Trace replay verify: PASS (all values = {expected})")
    else:
        print("Trace replay verify: FAIL")
        ttnn.release_trace(device, trace_id)
        ttnn.close_mesh_device(device)
        return 1

    # Export to .ttb
    io_tensors = {
        "input_a": tt_a,
        "input_b": tt_b,
        "output_c": tt_out,
    }
    export_trace(device, trace_id, output_path, io_tensors)

    ttnn.release_trace(device, trace_id)
    ttnn.close_mesh_device(device)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
