# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Workload for test_kernel_from_long_source_string.

Runs a program whose only kernel is created from a source string longer than
64 KiB. Meant to be run under Tracy capture by the outer test.

Exit codes:
  0  success
  5  real-time profiler not active on this configuration
"""

import sys

import torch

import ttnn

# The outer test checks that this text does not reach the Tracy capture.
PADDING_LINE = "// Padding that makes this kernel source longer than 64 KiB.\n"

KERNEL_SOURCE = '#include "api/dataflow/dataflow_api.h"\n\nvoid kernel_main() {}\n' + PADDING_LINE * 1200


def main():
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )

    if not ttnn.device.IsProgramRealtimeProfilerActive():
        print("SKIP: real-time profiler is not active on this configuration", file=sys.stderr)
        ttnn.close_mesh_device(mesh_device)
        sys.exit(5)

    try:
        core = ttnn.CoreCoord(0, 0)
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL_SOURCE,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
                    config=ttnn.ReaderConfigDescriptor(),
                )
            ]
        )
        inp, out = [
            ttnn.from_torch(torch.zeros(32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=mesh_device)
            for _ in range(2)
        ]
        for _ in range(3):
            ttnn.generic_op([inp, out], program)
        ttnn.synchronize_device(mesh_device)
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
