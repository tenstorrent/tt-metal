# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Denoise-loop latency benchmark for the ACE-Step DiT (batch 1, T'=128, p150).

Emits `METRIC denoise_ms=<per-step ms>` (median of several runs, back-to-back forwards with a
single device sync — the honest steady-state per-step latency). This is the optimization target
when the goal is DiT compute speed. Also prints total-loop time for context.

Not a PCC test — correctness is gated separately by the PCC suite. Uses fixed random inputs at the
10.24 s target length so runs are comparable.

    export ACESTEP_PIPELINE_DIR=/path/to/acestep_pipeline
    python models/experimental/acestep/perf/bench_denoise.py
"""

import statistics
import time

import torch
import ttnn

from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline

T = 256  # T'=128 target (10.24 s)
STEPS = 30
RUNS = 5
WARMUP = 3


def main():
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device, with_encoders=False)
        hch = pipe.args.audio_acoustic_hidden_dim
        torch.manual_seed(0)
        xt = ttnn.from_torch(torch.randn(1, 1, T, hch), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ctx = ttnn.from_torch(
            torch.randn(1, 1, T, hch * 2), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        enc = ttnn.from_torch(torch.randn(1, 1, 128, 2048), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tp = T // 2
        cos, sin = pipe._rope_tables(tp)
        mask = pipe._sliding_mask(tp)
        ts = ttnn.from_torch(
            torch.tensor([[[[0.7]]]], dtype=torch.float32), device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
        )

        def step():
            return pipe.dit.forward(xt, ctx, ts, ts, cos, sin, enc, sliding_mask=mask)

        for _ in range(WARMUP):
            ttnn.deallocate(step())

        per_step = []
        for _ in range(RUNS):
            t0 = time.time()
            outs = [step() for _ in range(STEPS)]
            ttnn.synchronize_device(device)
            dt = time.time() - t0
            for o in outs:
                ttnn.deallocate(o)
            per_step.append(dt / STEPS * 1000.0)

        ms = statistics.median(per_step)
        loop_s = ms * STEPS / 1000.0
        print(f"METRIC denoise_ms={ms:.3f}")
        print(f"denoise: {ms:.2f} ms/step (median of {RUNS}), {loop_s:.3f} s for {STEPS} steps at T'={tp}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
