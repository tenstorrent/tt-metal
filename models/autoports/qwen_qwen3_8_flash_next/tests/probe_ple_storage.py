# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Discriminate PLE storage candidates; this is not decoder acceptance evidence.

Run with TT_VISIBLE_DEVICES=0 and the single-chip TT_MESH_GRAPH_DESC_PATH.
Only setup and comparison boundaries use torch. The small real-weight slice
tests operation support without attempting a known oversized allocation.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    with safe_open(args.snapshot / index[key], framework="pt") as shard:
        weight = shard.get_slice(key)[:4096, :].contiguous()
    ids = torch.tensor([[0, 1, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 2047, 2048, 4095] * 2])
    expected = weight[ids.long()].float()
    report = {"weight_key": key, "sample_shape": list(weight.shape), "candidates": []}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        report["capacity"] = {
            "num_banks": view.num_banks,
            "free_bytes_per_bank": view.total_bytes_free_per_bank,
            "total_free_bytes": view.num_banks * view.total_bytes_free_per_bank,
        }
        tt_ids = ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, device=mesh)
        for name, dtype in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b), ("bfp4", ttnn.bfloat4_b)):
            entry = {"dtype": name}
            w = ttnn.from_torch(weight, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh)
            restored = ttnn.to_torch(w).float()
            entry["storage_roundtrip_pcc"] = torch.corrcoef(
                torch.stack((weight.float().flatten(), restored.flatten()))
            )[0, 1].item()
            try:
                result = ttnn.embedding(tt_ids, w, layout=ttnn.TILE_LAYOUT)
                actual = ttnn.to_torch(result).reshape(expected.shape).float()
                entry["embedding_pcc"] = torch.corrcoef(torch.stack((expected.flatten(), actual.flatten())))[
                    0, 1
                ].item()
                ttnn.deallocate(result)
            except RuntimeError as exc:
                entry["embedding_error"] = str(exc).split("backtrace:")[0]
            try:
                gather_ids = ids.flatten()[:, None].expand(-1, 160).contiguous().to(torch.int32)
                tt_gather_ids = ttnn.from_torch(gather_ids, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh)
                result = ttnn.gather(w, 0, tt_gather_ids)
                actual = ttnn.to_torch(result).reshape(expected.shape).float()
                entry["gather_pcc"] = torch.corrcoef(torch.stack((expected.flatten(), actual.flatten())))[0, 1].item()
                ttnn.deallocate(result)
                ttnn.deallocate(tt_gather_ids)
            except RuntimeError as exc:
                entry["gather_error"] = str(exc).split("backtrace:")[0]
            report["candidates"].append(entry)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(entry), flush=True)
            ttnn.deallocate(w)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
