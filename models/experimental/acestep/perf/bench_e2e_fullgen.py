# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end SINGLE FULL GENERATION wall-clock latency (warmup excluded).

One real prompt->audio generation via generate_song: tokenizer -> text encoder -> condition encoder
-> 30-step DiT flow-matching denoise -> Oobleck VAE decode -> 48 kHz stereo waveform. Measured with
HOST WALL-CLOCK time (not Tracy), the way a user experiences latency.

Three modes are measured:
  * CFG traced (guidance_scale=7.0, trace=ON) — the DEFAULT / correct text2music mode AND the
    deployment path. Classifier-free guidance runs the DiT TWICE per step (conditional + null
    context) and combines via APG. The whole two-pass + APG velocity step is captured as a ttnn
    trace and replayed per ODE step (numerically identical to eager, verified PCC 1.0), removing
    per-step host dispatch. This is the latency a user actually sees for prompt-faithful music.
  * CFG eager (guidance_scale=7.0, trace=OFF) — same math, no trace, to show the trace speedup.
  * no-CFG traced (guidance_scale=1.0, trace=ON) — the single-pass baseline, to show the CFG
    overhead (the extra DiT pass).

Protocol (per mode):
  1. WARMUP: one full generate_song — compiles kernels, captures the trace, populates program/conv
     caches. NOT counted.
  2. MEASURED: N runs, each wrapped start->synchronize->stop. Report per-run wall time + mean/std.

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
GUIDANCE_SCALE = 7.0  # HF text2music default (CFG). >1.0 => two DiT passes/step + APG.
SHIFT = 1.0  # HF base-model generate_audio default timestep shift (modeling_acestep_v15_base.py:1810).
PROMPT = "upbeat synthwave, driving bass, warm analog pads, nostalgic 80s energy"
LYRICS = "neon lights over the city tonight, we ride the endless skyline"
RUNS = 5


def _full_gen(pipe, *, guidance_scale, use_trace):
    """One full prompt->audio generation, synced. Returns (wall seconds, waveform)."""
    t0 = time.time()
    wav = pipe.generate_song(
        PROMPT,
        lyrics=LYRICS,
        seconds=SECONDS,
        infer_steps=INFER_STEPS,
        seed=0,
        guidance_scale=guidance_scale,
        shift=SHIFT,
        use_trace=use_trace,
    )
    ttnn.synchronize_device(pipe.mesh_device)
    dt = time.time() - t0
    return dt, wav


def _measure_mode(pipe, label, *, guidance_scale, use_trace):
    """Warmup (excluded) + RUNS measured full generations for one mode. Returns (mean, std, min, wav)."""
    wdt, wav = _full_gen(pipe, guidance_scale=guidance_scale, use_trace=use_trace)
    print(f"[bench] {label} warmup (excluded): {wdt:.3f}s | audio {tuple(wav.shape)} @ {pipe.SAMPLE_RATE}Hz")
    times = []
    for i in range(1, RUNS + 1):
        dt, _ = _full_gen(pipe, guidance_scale=guidance_scale, use_trace=use_trace)
        times.append(dt)
        print(f"[bench] {label} run {i}: {dt:.3f}s")
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std, min(times), wav


def main():
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        t0 = time.time()
        pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device)
        seq = pipe._latent_len(SECONDS)
        print(f"[bench] pipeline built in {time.time()-t0:.1f}s | {SECONDS}s audio (T'={seq//2}), {INFER_STEPS} steps")

        # --- CFG traced (guidance_scale=7.0, trace=ON): the DEFAULT text2music + deployment mode ---
        cfgt_mean, cfgt_std, cfgt_min, wav = _measure_mode(
            pipe, "CFG-traced", guidance_scale=GUIDANCE_SCALE, use_trace=True
        )
        # --- CFG eager (guidance_scale=7.0, trace=OFF): shows the trace speedup ---
        cfge_mean, cfge_std, cfge_min, _ = _measure_mode(
            pipe, "CFG-eager", guidance_scale=GUIDANCE_SCALE, use_trace=False
        )
        # --- no-CFG traced (guidance_scale=1.0, single pass, trace=ON): shows CFG overhead ---
        nc_mean, nc_std, nc_min, _ = _measure_mode(
            pipe, "no-CFG", guidance_scale=1.0, use_trace=True
        )

        audio_s = wav.shape[-1] / pipe.SAMPLE_RATE
        print("\n" + "=" * 70)
        print(f"E2E FULL GENERATION (batch 1, {SECONDS}s audio, {INFER_STEPS} steps)")
        print("-" * 70)
        print(f"  CFG traced (gs={GUIDANCE_SCALE}, 2 DiT/step): mean {cfgt_mean:.3f}s std {cfgt_std:.3f}s (min {cfgt_min:.3f}s)")
        print(f"  CFG eager  (gs={GUIDANCE_SCALE}, 2 DiT/step): mean {cfge_mean:.3f}s std {cfge_std:.3f}s (min {cfge_min:.3f}s)")
        print(f"  no-CFG trc (gs=1.0, 1 DiT/step)  : mean {nc_mean:.3f}s std {nc_std:.3f}s (min {nc_min:.3f}s)")
        print(f"  trace speedup (CFG): {cfge_mean/cfgt_mean:.2f}x  |  CFG overhead vs no-CFG: {cfgt_mean/nc_mean:.2f}x")
        print(f"  audio out   : {audio_s:.2f}s @ {pipe.SAMPLE_RATE}Hz stereo")
        print(f"  real-time   : CFG-traced {audio_s/cfgt_mean:.2f}x faster than real-time")
        print("=" * 70)
        # Primary metric = the CFG-traced (default text2music, deployment) full-generation wall time.
        print(f"METRIC e2e_fullgen_cfg_traced_s={cfgt_mean:.4f}")
        print(f"METRIC e2e_fullgen_cfg_eager_s={cfge_mean:.4f}")
        print(f"METRIC e2e_fullgen_nocfg_s={nc_mean:.4f}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
