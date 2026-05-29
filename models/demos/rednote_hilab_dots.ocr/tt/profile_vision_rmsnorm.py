# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy profiling harness for the dots.ocr vision RMSNorm leaf block.

Profiles :class:`TtVisionRMSNorm` (a single ttnn.rms_norm, eps=1e-5, dim=1536)
in isolation under metal trace so the CSV reflects device-kernel time rather
than host dispatch. The vision tower runs this norm per-patch inside the 42-layer
trunk; a representative production tile of patches is used for the shape.

Run under tracy::

    python3 -m tracy -p -v -r --op-support-count 50000 \
      models/demos/rednote_hilab_dots.ocr/tt/profile_vision_rmsnorm.py --traced

The ops CSV lands in generated/profiler/reports/<TIMESTAMP>/ops_perf_results_*.csv
(and a cpp_device_perf_report.csv under generated/profiler/.logs/).
"""
import argparse
import importlib.util
import os

import torch

import ttnn

_TT_DIR = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "dots_tt_vision_rmsnorm_profile", os.path.join(_TT_DIR, "vision_rmsnorm.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtVisionRMSNorm = _mod.TtVisionRMSNorm

# Production-representative shape: a tile of vision patches at hidden 1536.
# (Reduced PCC golden is [256,1536]; production grids push many more patches
# through the same per-token norm, so 1024 rows is a representative tile.)
N_PATCHES = 1024
DIM = 1536
EPS = 1e-5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traced", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
    try:
        weight = torch.randn(DIM, dtype=torch.float32)
        norm = TtVisionRMSNorm(device=device, dim=DIM, weight=weight, eps=EPS)

        host_in = torch.randn(N_PATCHES, DIM, dtype=torch.float32)
        x = ttnn.from_torch(
            host_in,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Warmup: compile the kernel into the program cache.
        for _ in range(3):
            out = norm(x)
        ttnn.synchronize_device(device)

        if args.traced:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = norm(x)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            # One profiled replay.
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            ttnn.release_trace(device, tid)
        else:
            out = norm(x)
            ttnn.synchronize_device(device)

        print("profile_vision_rmsnorm done; out shape", tuple(out.shape))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
