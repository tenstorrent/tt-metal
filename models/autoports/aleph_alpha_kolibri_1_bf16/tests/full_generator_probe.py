# SPDX-License-Identifier: Apache-2.0
import time

import torch

import ttnn

from ..tt.generator import build_generator


def main():
    torch.set_num_threads(8)
    start = time.monotonic()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    try:
        gen = build_generator(None, mesh, layer_indices=[0, 4], capacity=1024)
        print("GEN_SETUP", time.monotonic() - start, flush=True)
        out = gen.generate([42] * 33, 4)
        print("GEN_OUTPUT", out, dict(gen.counters), time.monotonic() - start, flush=True)
        second = gen.generate([42] * 33, 4)
        assert second == out, (out, second)
        print("GEN_DETERMINISTIC", flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
