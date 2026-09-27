# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Profile only two real layers plus the actual full-model terminal path."""
import torch
from tracy import signpost

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator


def main():
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    try:
        gen = build_generator(None, mesh, max_seq_len=8192, layer_indices=(0, 5))
        gen.generate([2] + [100] * 4095, 4, stop_on_eos=False)
        ttnn.synchronize_device(mesh)
        signpost("PERF_DECODE")
        gen._replay()
        ttnn.synchronize_device(mesh)
        signpost("PERF_DECODE_END")
        ttnn.ReadDeviceProfiler(mesh)
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
