# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy harness for TtEmbedding at the PRODUCTION operating points.

The embedding has TWO production shapes (ocr_model.py):

- decode (hot path, inside the metal-traced token step): the persistent id
  buffer is [1, 1, 1, 32] uint32 replicated; lookup on the hidden-dim-sharded
  [1, 1, 151936, 1536] table -> [1, 1, 32, 1536] -> all_gather(dim=3) back to
  a replicated row (Topology.Linear, FABRIC_1D). Profile with --seq 32
  --traced (the step runs under metal trace in production).
- prefill (run once per image): real prompts are ~2.8k tokens, padded to a
  tile multiple -> ids [1, 1, 1, 2816]. Profile with --seq 2816.

Run under tracy:
    python -m tracy -p -v -r --op-support-count 20000 \
        models/demos/rednote_hilab_dots.ocr/tt/profile_embedding.py \
        --traced --seq 32
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

import ttnn

_TT_DIR = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("dots_ocr_tt_embedding", _TT_DIR / "embedding.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtEmbedding = _mod.TtEmbedding

SEQ = 32  # production decode posture: persistent [1, 1, 1, 32] id buffer


def _load_embed_weight():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download("rednote-hilab/dots.ocr", allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    key = "model.embed_tokens.weight"
    with safe_open(snap / idx[key], framework="pt") as f:
        return f.get_tensor(key).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    parser.add_argument("--seq", type=int, default=SEQ, help="id-row width: 32 decode posture, 2816 prefill")
    parser.add_argument("--num-links", type=int, default=None, help="override TtEmbedding all_gather links (A/B)")
    parser.add_argument(
        "--placement", choices=["replicate", "shard"], default=None, help="table placement A/B (default: block default)"
    )
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=32768,
        trace_region_size=25_000_000,
    )
    try:
        kwargs = {}
        if args.num_links is not None:
            kwargs["num_links"] = args.num_links
        if args.placement is not None:
            kwargs["placement"] = args.placement
        block = TtEmbedding(mesh_device, {"weight": _load_embed_weight()}, **kwargs)

        torch.manual_seed(0)
        ids = torch.randint(0, 151936, (1, 1, 1, args.seq), dtype=torch.int32)
        # Persistent input buffer: stable address for trace replay.
        ids_tt = ttnn.from_torch(
            ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Warmup: compile every kernel into the program cache.
        for _ in range(3):
            out = block.forward(ids_tt)
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)

        if args.traced:
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = block.forward(ids_tt)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            # One warmup replay, then the profiled replay.
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.release_trace(mesh_device, tid)
        else:
            out = block.forward(ids_tt)
            ttnn.synchronize_device(mesh_device)
        print("profiled iteration complete (traced=%s)" % args.traced)
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
