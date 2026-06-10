# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy harness for TtVisionPatchEmbed at the PRODUCTION operating point.

Production context (tests/test_vision_transformer.py): the fp32 vision tower
calls patch_embed once per image with a replicated [1, 1, 896, 588] fp32
input on the 1x4 mesh (784 patches padded to 896 rows, C*P*P = 588).

Run under tracy:
    python -m tracy -p -v -r --op-support-count 20000 \
        models/demos/rednote_hilab_dots.ocr/tt/profile_vision_patch_embed.py --traced
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

import ttnn

_TT_DIR = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("dots_ocr_tt_vision_patch_embed", _TT_DIR / "vision_patch_embed.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
TtVisionPatchEmbed = _mod.TtVisionPatchEmbed

PADDED_SEQ = 896  # 784 real patches padded to a multiple of 128 (host-side)
PATCH_DIM = 588  # C * P * P = 3 * 14 * 14


def _load_weights():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = Path(snapshot_download("rednote-hilab/dots.ocr", allow_patterns=["*.json", "*.safetensors"]))
    idx = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for k in ["proj.weight", "proj.bias", "norm.weight"]:
        full = f"vision_tower.patch_embed.patchifier.{k}"
        with safe_open(snap / idx[full], framework="pt") as f:
            out[k] = f.get_tensor(full).float()
    return out


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
        # Production dtype: the vision tower runs fp32 end-to-end.
        block = TtVisionPatchEmbed(mesh_device, _load_weights(), dtype=ttnn.float32)

        torch.manual_seed(0)
        x = torch.randn(1, 1, PADDED_SEQ, PATCH_DIM)
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
