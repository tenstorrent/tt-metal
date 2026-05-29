# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling harness for the dots.ocr vision tower (DotsVisionTransformer).

Profiles :class:`TtVisionTower` -- the final vision-side assembly
(N x TtVisionBlock -> post-trunk TtVisionRMSNorm -> TtVisionPatchMerger) in
isolation under metal trace so the CSV reflects device-kernel time rather than
host dispatch. The patch_embed Conv2d+RMSNorm is the documented host-resident
boundary and runs on host; its [num_patches, embed_dim] output is the device
input to the traced tower body.

Production-representative shapes come from the seed-0 golden: grid_thw 1x4x4
(16 patches), embed_dim 1536, 12 heads, head_dim 128, spatial_merge_size 2,
reduced num_layers (2) standing in for the full 42-layer trunk (the per-block
device-kernel profile scales linearly with layer count, so 2 layers is a
representative-per-layer sample and keeps the trace small).

vision_tower COMPOSES already-optimized leaf modules: TtVisionBlock carries the
attention (-23.8%) + mlp (-13.8%) + residual L1 wins, and TtVisionPatchMerger is
at-ceiling. So it inherits those wins. This harness exists to check the
COMPOSITE boundaries that only appear at the assembly level: any reshard/transpose
between blocks landing DRAM, the post-trunk norm -> patch_merger handoff, and the
RoPE cos/sin threading.

Run under tracy::

    python3 -m tracy -p -v -r --op-support-count 50000 \
      models/demos/rednote_hilab_dots.ocr/tt/profile_vision_tower.py --traced

The ops CSV lands in generated/profiler/reports/<TIMESTAMP>/ops_perf_results_*.csv.
"""
import argparse
import importlib.util
import os

import torch

import ttnn

_TT_DIR = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("dots_tt_vision_tower_profile", os.path.join(_TT_DIR, "vision_tower.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtVisionTower = _mod.TtVisionTower

_GOLDEN_PATH = os.path.normpath(os.path.join(_TT_DIR, "..", "reference", "golden", "vision_tower.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    golden = torch.load(_GOLDEN_PATH, map_location="cpu", weights_only=False)
    pixel_values = golden["input"].to(torch.float32)
    grid_thw = torch.as_tensor(golden["grid_thw"])
    state_dict = {k: v.to(torch.float32) for k, v in golden["state_dict"].items()}
    cfg = golden["config"]

    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
    try:
        tower = TtVisionTower(
            device=device,
            state_dict=state_dict,
            grid_thw=grid_thw,
            num_layers=int(cfg["num_layers"]),
            embed_dim=int(cfg["embed_dim"]),
            num_heads=int(cfg["num_heads"]),
            num_channels=int(cfg["num_channels"]),
            temporal_patch_size=int(cfg["temporal_patch_size"]),
            patch_size=int(cfg["patch_size"]),
            spatial_merge_size=int(cfg["spatial_merge_size"]),
            rms_norm_eps=float(cfg["rms_norm_eps"]),
            ln_eps=float(cfg["ln_eps"]),
            post_norm=bool(cfg["post_norm"]),
        )

        # Host-resident patch_embed (documented boundary) -> device input tokens.
        hidden_states = tower.patch_embed(pixel_values)
        x = ttnn.from_torch(
            hidden_states,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Warmup: compile the kernels into the program cache.
        for _ in range(3):
            out = tower(x)
        ttnn.synchronize_device(device)

        if args.traced:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = tower(x)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            ttnn.release_trace(device, tid)
        else:
            out = tower(x)
            ttnn.synchronize_device(device)

        print("profile_vision_tower done; out shape", tuple(out.shape))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
