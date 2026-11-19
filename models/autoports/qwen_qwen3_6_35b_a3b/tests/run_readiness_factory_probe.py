# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exercise the packaged standalone-module discovery with real reduced layers."""

import argparse
import json
from pathlib import Path

import torch
from readiness_check.run_teacher_forcing import _import_build_generator

import ttnn


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-sampling", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    generator = None
    try:
        factory = _import_build_generator(root)
        generator = factory(
            model_dir=root, mesh_device=mesh, layer_indices=[0, 3], max_seq_len=256, host_sampling=args.host_sampling
        )
        tokens = generator.generate([1, 2, 3, 4], 4)
        assert tokens == [28433, 150929, 11927, 66275], tokens
        generator.release_request(0)
        result = {
            "status": "PASS",
            "scope": "reduced standalone discovery",
            "host_sampling": args.host_sampling,
            "tokens": tokens,
            "counters": dict(generator.counters),
        }
        name = "host_sampling_probe.json" if args.host_sampling else "readiness_factory_probe.json"
        (root / "doc/full_model" / name).write_text(json.dumps(result, indent=2) + "\n")
        print("READINESS_FACTORY_PASS", flush=True)
    finally:
        if generator is not None:
            generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
