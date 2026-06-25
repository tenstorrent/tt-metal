# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill-only bench: run 18 VLM blocks (TP=8, seq=768) and capture CCL
device-kernel durations via tracy.

Run:
  python -m tracy -p -r -n prefill_tp8_ccl -o /tmp/tracy_prefill_ccl \\
    python_env/bin/python \\
    models/experimental/pi0_5/tests/perf/_bench_prefill_tp8_ccl.py

Parse ReduceScatterDeviceOperation / AllGatherDeviceOperation rows in
ops_perf_results CSV — compare against isolated CCL unit test numbers.

Uses chips 24-31 (same as test_ccl_prefill_tp8_perf.py) via TT_VISIBLE_DEVICES.
Requires PI05_CHECKPOINT_DIR to point to a valid pi0.5 checkpoint directory.
"""

import os

# Mirror the exact env setup of test_perf_tt_bh_glx_1x8.py (setdefault block at
# module top) so the 1-layer bench uses the same kernel paths as the production
# test. Note: the test's _apply_production_env_defaults() looks in the wrong path
# (models/_bench_runs/) so those flags are also absent here — consistent.
for _k, _v in {
    "PI0_TP": "8",
    "PI0_TP4_ATTN_HEADPAR": "1",
    "PI0_MLP_BS": "1",
    "PI0_MLP_FUSED_RS": "0",
    "TT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",
}.items():
    os.environ.setdefault(_k, _v)

os.environ.setdefault("PI0_SIGLIP_USE_FOLD", "1")
os.environ.setdefault("QWEN_NLP_CREATE_HEADS_HEAD_SPLIT", "1")
os.environ.setdefault("QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT", "1")

# 2-cam
os.environ["PI0_NUM_CAMERAS"] = "2"
os.environ["PI0_VLM_CHUNK_SIZE"] = "768"

os.environ.setdefault(
    "PI05_CHECKPOINT_DIR",
    "/home/tt-admin/ssinghal/qwen36/p150x4/tt-metal/models/experimental/pi0_5/weights/pi05_libero_upstream",
)

import torch
import ttnn

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh
from models.experimental.pi0_5.tt.tt_bh_glx.stage_prefill_tp4 import StagePrefillTP4

SEQ = 768  # 2-cam: 2×256 SigLIP + 256 lang
N_LAYERS = int(os.environ.get("BENCH_N_LAYERS", "1"))


def main():
    ckpt = os.environ["PI05_CHECKPOINT_DIR"]
    print(f"Loading weights from {ckpt} ...", flush=True)
    loader = Pi0_5WeightLoader(ckpt)
    weights = loader.categorized_weights

    cfg = Pi0_5ModelConfig()
    vlm_w = cfg.vlm_config.width  # 2048

    print(f"Opening 1×8 mesh (chips {os.environ['TT_VISIBLE_DEVICES']}, {N_LAYERS} layer(s)) ...", flush=True)
    with open_prefill_tp4_mesh(tp=8, l1_small_size=24576, trace_region_size=128 * 1024 * 1024) as mesh:
        stage = StagePrefillTP4(cfg, weights, mesh)
        stage.blocks = stage.blocks[:N_LAYERS]

        x = ttnn.from_torch(
            torch.randn(1, 1, SEQ, vlm_w).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Eager warmup — JIT-compiles all kernels
        print("Eager warmup ...", flush=True)
        hidden, _ = stage.run(x)
        ttnn.synchronize_device(mesh)

        # Trace capture — flush profiler after so only the replay is captured
        print("Capturing trace ...", flush=True)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        hidden, _ = stage.run(x)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)  # flush trace-capture pass; only replay is profiled

        # Single traced replay — tracy captures only this
        print(f"Traced replay ({N_LAYERS} layer(s), seq={SEQ}) ...", flush=True)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.ReadDeviceProfiler(mesh)
        ttnn.release_trace(mesh, tid)

        print("Done. Check RS/AG DK duration in tracy CSV.", flush=True)


if __name__ == "__main__":
    main()
