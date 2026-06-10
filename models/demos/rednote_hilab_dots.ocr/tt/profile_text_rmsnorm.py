# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy harness for TtTextRMSNorm at the PRODUCTION operating point.

Production context (tt/decoder_layer.py): the fp32 text decoder calls
text_rmsnorm twice per layer (input_layernorm / post_attention_layernorm)
on a REPLICATED [1, 1, 128, 1536] fp32 residual stream on the 1x4 mesh
(the replicated ``forward`` path; the distributed pre/post+all_gather path
is not on the decoder_layer hot path). fp32 gamma uses TILE [1, 1, 1, dim].

Run under tracy:
    python -m tracy -p -v -r --op-support-count 20000 \
        models/demos/rednote_hilab_dots.ocr/tt/profile_text_rmsnorm.py --traced
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

import ttnn

_TT_DIR = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("dots_ocr_tt_text_rmsnorm", _TT_DIR / "text_rmsnorm.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtTextRMSNorm = _mod.TtTextRMSNorm

SEQ = 128  # decoder_layer golden/production seq bucket
HIDDEN = 1536


def _load_weight():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download("rednote-hilab/dots.ocr", allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    full = "model.layers.0.input_layernorm.weight"
    with safe_open(snap / idx[full], framework="pt") as f:
        return f.get_tensor(full).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=32768,
        trace_region_size=25_000_000,
    )
    try:
        # Production dtype: the text decoder layer runs fp32 (attention path
        # is fp32-mandatory; the norm inherits the layer dtype).
        block = TtTextRMSNorm(mesh_device, {"weight": _load_weight()}, dtype=ttnn.float32, eps=1e-6)

        torch.manual_seed(0)
        x = torch.randn(1, 1, SEQ, HIDDEN)
        # Persistent input buffer: stable address for trace replay.
        x_tt = ttnn.from_torch(
            x,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Warmup: compile every kernel into the program cache.
        for _ in range(3):
            out = block.forward(x_tt)
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)

        if args.traced:
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = block.forward(x_tt)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            # One warmup replay, then the profiled replay.
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.release_trace(mesh_device, tid)
        else:
            out = block.forward(x_tt)
            ttnn.synchronize_device(mesh_device)
        print("profiled iteration complete (traced=%s)" % args.traced)
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
