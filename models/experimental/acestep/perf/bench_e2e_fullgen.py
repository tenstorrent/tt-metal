# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end SINGLE FULL GENERATION wall-clock latency (trace capture, warmup excluded).

One real prompt->audio generation via generate_song(use_trace=True): tokenizer -> text encoder ->
condition encoder -> 30-step DiT flow-matching denoise (traced, replayed per step) -> Oobleck VAE
decode -> 48 kHz stereo waveform. Measured with HOST WALL-CLOCK time (not Tracy), the way a user
experiences latency.

Protocol:
  1. Build the pipeline.
  2. WARMUP: one full generate_song(use_trace=True) — compiles kernels, captures the DiT trace,
     populates program/conv caches. NOT counted.
  3. MEASURED: N full generate_song(use_trace=True) runs, each wrapped start->synchronize->stop.
     Report per-run wall time + mean/std. Trace replay makes runs steady-state.

    export ACESTEP_PIPELINE_DIR=/path/to/acestep_pipeline   # or rely on HF cache
    python models/experimental/acestep/perf/bench_e2e_fullgen.py
"""

import statistics
import time

import ttnn

from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline

SECONDS = 10.24  # T'=128 latent frames (~10.24 s of audio)
INFER_STEPS = 30  # HF generate_audio default
PROMPT = "upbeat synthwave, driving bass, warm analog pads, nostalgic 80s energy"
LYRICS = "neon lights over the city tonight, we ride the endless skyline"
RUNS = 5


def _full_gen(pipe):
    """One full prompt->audio generation (traced denoise), synced. Returns wall seconds."""
    t0 = time.time()
    wav = pipe.generate_song(
        PROMPT, lyrics=LYRICS, seconds=SECONDS, infer_steps=INFER_STEPS, seed=0, use_trace=True
    )
    ttnn.synchronize_device(pipe.mesh_device)
    dt = time.time() - t0
    return dt, wav


def main():
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        t0 = time.time()
        pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device)
        seq = pipe._latent_len(SECONDS)
        print(f"[bench] pipeline built in {time.time()-t0:.1f}s | {SECONDS}s audio (T'={seq//2}), {INFER_STEPS} steps, trace=ON")

        # --- WARMUP (compile + trace capture + caches), NOT counted ---
        wdt, wav = _full_gen(pipe)
        print(f"[bench] warmup (excluded): {wdt:.3f}s | audio {tuple(wav.shape)} @ {pipe.SAMPLE_RATE}Hz")

        # --- MEASURED full generations ---
        times = []
        for i in range(1, RUNS + 1):
            dt, _ = _full_gen(pipe)
            times.append(dt)
            print(f"[bench] run {i}: {dt:.3f}s")

        mean = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        audio_s = wav.shape[-1] / pipe.SAMPLE_RATE
        print("\n" + "=" * 60)
        print(f"E2E FULL GENERATION (batch 1, {SECONDS}s audio, {INFER_STEPS} steps, trace=ON)")
        print("-" * 60)
        print(f"  wall time   : mean {mean:.3f}s  std {std:.3f}s  (min {min(times):.3f}s, n={RUNS})")
        print(f"  audio out   : {audio_s:.2f}s @ {pipe.SAMPLE_RATE}Hz stereo")
        print(f"  real-time   : {audio_s/mean:.2f}x faster than real-time")
        print("=" * 60)
        print(f"METRIC e2e_fullgen_s={mean:.4f}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
