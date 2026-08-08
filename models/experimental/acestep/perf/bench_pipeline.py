# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end stage timing for the ACE-Step text-to-music pipeline (batch 1, p150).

Breaks `generate_song` into its stages (prompt-encode / DiT denoise-loop / VAE decode / total) and
reports mean/std over a few runs, plus per-step denoise latency and throughput. This is the
baseline harness to compare against once trace capture + external optimization configs are applied.

Reuses `models.perf.benchmarking_utils.BenchmarkProfiler` (the tt_dit perf-test harness). Not a PCC
test — correctness is covered by tests/. Uses the fixed target length (T'=128 ~= 10.24 s) by default.

    export ACESTEP_PIPELINE_DIR=/path/to/acestep_pipeline
    python models/experimental/acestep/perf/bench_pipeline.py
"""

import statistics
import time

import ttnn

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline

# Fixed target: T'=128 -> T=256 latent frames -> 10.24 s @ 25 Hz. The overhead-bound edge; the
# regime we optimize first (trace capture removes the host path here).
SECONDS = 10.24
INFER_STEPS = 30
PROMPT = "upbeat synthwave, driving bass, warm analog pads, nostalgic 80s energy"
LYRICS = "neon lights over the city tonight, we ride the endless skyline"
RUNS = 3


def _instrumented_generate(pipe, profiler, iteration):
    """Run generate_song's stages with per-stage timing (mirrors generate_song internals)."""
    with profiler("encode", iteration=iteration):
        context = pipe.encode_prompt(PROMPT, LYRICS)
        ttnn.synchronize_device(pipe.mesh_device)

    # Build noise + context latents exactly as generate_song does, then time the denoise loop + VAE.
    import torch

    seq_len = pipe._latent_len(SECONDS)
    torch.manual_seed(0)
    noise = torch.randn(1, 1, seq_len, pipe.args.audio_acoustic_hidden_dim)
    hidden_noise = ttnn.from_torch(noise, device=pipe.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ctx_lat = ttnn.from_torch(
        torch.zeros(1, 1, seq_len, pipe.args.audio_acoustic_hidden_dim * 2),
        device=pipe.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    with profiler("denoise", iteration=iteration):
        latents = pipe.generate(hidden_noise, ctx_lat, context, infer_steps=INFER_STEPS)
        ttnn.synchronize_device(pipe.mesh_device)

    with profiler("vae", iteration=iteration):
        wav = pipe.decode(latents)
        ttnn.synchronize_device(pipe.mesh_device)
    return wav


def main():
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        t0 = time.time()
        pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device)
        print(
            f"[bench] pipeline built in {time.time() - t0:.1f}s | target {SECONDS}s (T'={pipe._latent_len(SECONDS)//2}), {INFER_STEPS} steps"
        )

        profiler = BenchmarkProfiler()
        # Warmup (also the future trace-capture run).
        with profiler("total", iteration=0):
            _instrumented_generate(pipe, profiler, 0)
        print(f"[bench] warmup total {profiler.get_duration('total', 0):.2f}s")

        for i in range(1, RUNS + 1):
            with profiler("total", iteration=i):
                _instrumented_generate(pipe, profiler, i)
            print(f"[bench] run {i} total {profiler.get_duration('total', i):.2f}s")

        def stat(name):
            xs = [profiler.get_duration(name, i) for i in range(1, RUNS + 1)]
            return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)

        print("\n" + "=" * 64)
        print(f"ACE-Step pipeline perf (batch 1, {SECONDS}s, {INFER_STEPS} steps)")
        print("-" * 64)
        for name in ("encode", "denoise", "vae", "total"):
            m, s = stat(name)
            print(f"{name:10} mean {m:7.3f}s  std {s:6.3f}s")
        dm, _ = stat("denoise")
        print(f"{'per-step':10} {dm / INFER_STEPS * 1000:7.1f}ms  ({INFER_STEPS / dm:.2f} steps/s)")
        print("=" * 64)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
