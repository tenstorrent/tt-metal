# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-weight wrapper probe; hardware jobs are externally timeout-bounded."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-layers", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    start = time.perf_counter()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    try:
        model = OrnithModel(mesh, layer_indices=None if args.all_layers else [0, 3])
        cache = model.allocate_cache(batch=1, context=262144)
        table = model.upload(
            torch.arange(8192, dtype=torch.int32)[None, :], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ids = torch.arange(33, dtype=torch.int64)[None, :] + 100
        tokens = model.upload(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        print("PREFILL", flush=True)
        out = model.host_logits(model.prefill(tokens, cache=cache, slot=0, start=0, page_table=table))
        print("PREFILL_DONE", out.shape, out[0, 0, -1].topk(5).indices, flush=True)
        token = model.upload(torch.tensor([[104]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos = model.upload(torch.tensor([33], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        print("DECODE", flush=True)
        decoded = model.host_logits(model.decode(token, cache=cache, slots=[0], positions=[pos], page_tables=[table]))
        print("DECODE_DONE", decoded.shape, decoded[0, 0, 0].topk(5).indices, flush=True)
        result = {
            "layers": model.layer_indices,
            "prefill_shape": list(out.shape),
            "decode_shape": list(decoded.shape),
            "finite": bool(torch.isfinite(out).all() and torch.isfinite(decoded).all()),
            "weights_memory": model.memory_after_weights,
            "final_memory": model.memory_view(),
            "elapsed_seconds": time.perf_counter() - start,
        }
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
