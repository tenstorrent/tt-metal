# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision propagation and non-aligned prompt smoke on both real layer kinds."""
import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[31, 32, 33, 1023, 1024, 1025, 4097])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    report = {"scope": "reduced real layers 0 and 5; propagation and shape safety only", "cases": []}
    try:
        gen = build_generator(None, mesh, max_seq_len=8192, layer_indices=(0, 5), precision_config=args.config)
        report["runtime_policy"] = gen.model.precision_summary()
        for length in args.lengths:
            tokens = gen.generate([2] + [100] * (length - 1), 3, stop_on_eos=False)
            position = ttnn.to_torch(ttnn.get_device_tensors(gen.cache_positions)[0]).item()
            assert len(tokens) == 3 and all(0 <= t < gen.model.config.vocab_size for t in tokens)
            assert position == length + 2
            report["cases"].append(
                {
                    "prompt_length": length,
                    "final_position": position,
                    "tokens": tokens,
                    "passed": True,
                    "metrics": dict(gen.metrics),
                }
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["runtime_policy"] = gen.model.precision_summary()
        report["status"] = "pass"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
