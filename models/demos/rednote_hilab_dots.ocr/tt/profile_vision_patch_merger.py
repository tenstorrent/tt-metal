# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling harness for the dots.ocr vision PatchMerger leaf block.

Profiles :class:`TtVisionPatchMerger` (LayerNorm eps 1e-6 with bias -> reshape
group 2x2 patches 1536->6144 -> Linear -> GELU -> Linear, all biased) in
isolation under metal trace so the CSV reflects device-kernel time rather than
host dispatch. Uses the production-representative shape from the vision trunk:
seq_length=256 patch tokens (the convention shared with profile_vision_block /
profile_vision_mlp), context_dim=1536, spatial_merge_size=2 -> hidden_size=6144,
merged output 64 tokens of out_dim=1536.

Run under tracy::

    python3 -m tracy -p -v -r --op-support-count 50000 \
      models/demos/rednote_hilab_dots.ocr/tt/profile_vision_patch_merger.py --traced

The ops CSV lands in generated/profiler/reports/<TIMESTAMP>/ops_perf_results_*.csv.
"""
import argparse
import importlib.util
import os

import torch

import ttnn

_TT_DIR = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "dots_tt_vision_patch_merger_profile", os.path.join(_TT_DIR, "vision_patch_merger.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtVisionPatchMerger = _mod.TtVisionPatchMerger

# Production-representative shapes (shared SEQ convention with sibling harnesses).
SEQ = 256
CONTEXT_DIM = 1536
SPATIAL_MERGE = 2
HIDDEN = CONTEXT_DIM * (SPATIAL_MERGE**2)  # 6144
OUT_DIM = 1536


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
    try:
        ln_weight = torch.randn(CONTEXT_DIM, dtype=torch.float32)
        ln_bias = torch.randn(CONTEXT_DIM, dtype=torch.float32)
        fc1_weight = torch.randn(HIDDEN, HIDDEN, dtype=torch.float32) * 0.02
        fc1_bias = torch.randn(HIDDEN, dtype=torch.float32) * 0.02
        fc2_weight = torch.randn(OUT_DIM, HIDDEN, dtype=torch.float32) * 0.02
        fc2_bias = torch.randn(OUT_DIM, dtype=torch.float32) * 0.02

        merger = TtVisionPatchMerger(
            device=device,
            ln_weight=ln_weight,
            ln_bias=ln_bias,
            fc1_weight=fc1_weight,
            fc1_bias=fc1_bias,
            fc2_weight=fc2_weight,
            fc2_bias=fc2_bias,
            context_dim=CONTEXT_DIM,
            spatial_merge_size=SPATIAL_MERGE,
        )

        host_in = torch.randn(SEQ, CONTEXT_DIM, dtype=torch.float32)
        x = ttnn.from_torch(
            host_in,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Warmup: compile the kernels into the program cache.
        for _ in range(3):
            out = merger(x)
        ttnn.synchronize_device(device)

        if args.traced:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = merger(x)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            ttnn.release_trace(device, tid)
        else:
            out = merger(x)
            ttnn.synchronize_device(device)

        print("profile_vision_patch_merger done; out shape", tuple(out.shape))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
