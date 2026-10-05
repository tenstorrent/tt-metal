"""Exact host and device-read composition controls for the K2 TP4 logits."""

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn

from ..tt.generator import K2Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(71938)
    result = dict(
        completed=False,
        scope="Setup correctness and host conversion timing; not serving performance",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        records=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    owner = SimpleNamespace(mesh=mesh, counters=Counter(), model=SimpleNamespace(vocab_size=250624))
    try:
        for batch in (1, 2, 16, 32):
            source = torch.randn(1, 1, batch, 262144).bfloat16()
            for on_device in (False, True):
                tensor = ttnn.from_torch(
                    source,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                    **({"device": mesh} if on_device else {}),
                )

                def old():
                    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))[..., :250624]

                def new():
                    return K2Generator._read_logits(owner, tensor)

                expected, actual = old(), new()
                assert torch.equal(expected, actual) and torch.equal(actual, source[..., :250624])
                row = dict(batch=batch, on_device=on_device, exact=True)
                if not on_device:
                    for name, function in (("composer", old), ("current", new)):
                        samples = []
                        for _ in range(5):
                            begin = time.perf_counter()
                            function()
                            samples.append((time.perf_counter() - begin) * 1000)
                        row[name + "_ms"] = statistics.median(samples)
                print(json.dumps(row), flush=True)
                result["records"].append(row)
                if on_device:
                    tensor.deallocate(True)
        result["completed"] = True
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
