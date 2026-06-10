# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy harness for TtTextAttention at the PRODUCTION operating point.

Production context (tt/decoder_layer.py): the fp32 text decoder calls
text_attention once per layer on a REPLICATED [1, 1, 128, 1536] fp32
residual stream on the 1x4 mesh (3 Q heads/chip, kv_replication=2,
row-parallel o_proj + sync all_gather all-reduce). cos/sin and the causal
mask come from the golden (real rope theta=1e6 tables), real
model.layers.0.self_attn weights.

Run under tracy:
    python -m tracy -p -v -r --op-support-count 20000 \
        models/demos/rednote_hilab_dots.ocr/tt/profile_text_attention.py --traced
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

import ttnn

_TT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _TT_DIR.parent

_spec = importlib.util.spec_from_file_location("dots_ocr_tt_text_attention", _TT_DIR / "text_attention.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtTextAttention = _mod.TtTextAttention

REPO = "rednote-hilab/dots.ocr"
HIDDEN = 1536


def _load_weights(prefix, keys):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for k in keys:
        full = f"{prefix}.{k}"
        with safe_open(snap / idx[full], framework="pt") as f:
            out[k] = f.get_tensor(full).float()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    parser.add_argument("--iters", type=int, default=1, help="profiled replay count")
    args = parser.parse_args()

    golden = torch.load(_MODEL_DIR / "reference" / "golden" / "text_attention.pt")
    x, cos, sin = golden["input"], golden["cos"], golden["sin"]
    _, seq, dim = x.shape
    assert dim == HIDDEN

    sd = _load_weights(
        "model.layers.0.self_attn",
        [
            "q_proj.weight",
            "q_proj.bias",
            "k_proj.weight",
            "k_proj.bias",
            "v_proj.weight",
            "v_proj.bias",
            "o_proj.weight",
        ],
    )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=32768,
        trace_region_size=50_000_000,
    )
    try:
        block = TtTextAttention(mesh_device, sd, num_heads=12, num_kv_heads=2)
        rot_mats = block.prepare_rope(cos, sin)
        causal_mask = block.prepare_causal_mask(seq)

        # Persistent input buffer: stable address for trace replay.
        x_tt = ttnn.from_torch(
            x.reshape(1, 1, seq, dim).float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Warmup: compile every kernel into the program cache.
        for _ in range(3):
            out = block.forward(x_tt, rot_mats, causal_mask)
            ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)

        if args.traced:
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = block.forward(x_tt, rot_mats, causal_mask)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            # One warmup replay, then the profiled replays.
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            for _ in range(max(1, args.iters)):
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            ttnn.release_trace(mesh_device, tid)
        else:
            out = block.forward(x_tt, rot_mats, causal_mask)
            ttnn.synchronize_device(mesh_device)
        print("profiled iteration complete (traced=%s, seq=%d)" % (args.traced, seq))
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
