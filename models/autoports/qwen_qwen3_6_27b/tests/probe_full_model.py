# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced full-model debug probe: real GDN + attention, embedding/norm/head/cache."""
import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--length", type=int, default=33)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    started = time.perf_counter()
    try:
        model = Model(mesh, layer_indices=None if args.all_layers else [0, 3])
        cache = model.create_cache(torch.arange(32, dtype=torch.int32)[None])
        ids = torch.arange(args.length, dtype=torch.int32)[None] + 100
        x = model.io._upload(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        p = model.io._upload(torch.arange(args.length, dtype=torch.int32)[None], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        out = model.prefill(x, p, cache=cache)
        host = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
        tokens = model.io._upload(torch.full((1, 1, 1, 32), 123, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        positions = model.io._upload(torch.tensor([args.length], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        out_decode = model.decode(tokens, positions, cache=cache, batch=1)
        host_decode = ttnn.to_torch(out_decode, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
        result = dict(
            layers=model.layer_indices,
            length=args.length,
            prefill_shape=list(host.shape),
            decode_shape=list(host_decode.shape),
            finite=bool(torch.isfinite(host).all() and torch.isfinite(host_decode).all()),
            loaded_memory=model.loaded_memory,
            with_cache_memory=model.memory_view(),
            elapsed_seconds=time.perf_counter() - started,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
