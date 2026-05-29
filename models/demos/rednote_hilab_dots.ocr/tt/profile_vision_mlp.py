# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling harness for the dots.ocr vision MLP (SwiGLU FFN) leaf block.

Profiles :class:`TtVisionMLP` (fused gate/up linear -> split -> silu(gate)*up ->
down linear, no bias) in isolation under metal trace so the CSV reflects
device-kernel time rather than host dispatch. Uses the production-representative
shapes from the seed-0 golden: seq_length=256, embed_dim=1536,
intermediate_size=4224.

Run under tracy::

    python3 -m tracy -p -v -r --op-support-count 50000 \
      models/demos/rednote_hilab_dots.ocr/tt/profile_vision_mlp.py --traced

The ops CSV lands in generated/profiler/reports/<TIMESTAMP>/ops_perf_results_*.csv.
"""
import argparse
import importlib.util
import os

import torch

import ttnn

_TT_DIR = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("dots_tt_vision_mlp_profile", os.path.join(_TT_DIR, "vision_mlp.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtVisionMLP = _mod.TtVisionMLP

# Production-representative shapes from the seed-0 golden.
SEQ = 256
DIM = 1536
INTERMEDIATE = 4224


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
    try:
        fc1_weight = torch.randn(INTERMEDIATE, DIM, dtype=torch.float32) * 0.02
        fc3_weight = torch.randn(INTERMEDIATE, DIM, dtype=torch.float32) * 0.02
        fc2_weight = torch.randn(DIM, INTERMEDIATE, dtype=torch.float32) * 0.02

        mlp = TtVisionMLP(
            device=device,
            fc1_weight=fc1_weight,
            fc3_weight=fc3_weight,
            fc2_weight=fc2_weight,
        )

        host_in = torch.randn(SEQ, DIM, dtype=torch.float32)
        x = ttnn.from_torch(
            host_in,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Warmup: compile the kernels into the program cache.
        for _ in range(3):
            out = mlp(x)
        ttnn.synchronize_device(device)

        if args.traced:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = mlp(x)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            ttnn.release_trace(device, tid)
        else:
            out = mlp(x)
            ttnn.synchronize_device(device)

        print("profile_vision_mlp done; out shape", tuple(out.shape))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
