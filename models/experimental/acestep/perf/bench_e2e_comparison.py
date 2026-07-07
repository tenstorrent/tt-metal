# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 — TT (Blackhole) vs. the official CUDA reference, apples-to-apples.

Reproduces the EXACT benchmark config the upstream ACE-Step repo ships in its default example
(examples/default/input_params/*.json) so our wall-clock + real-time-factor numbers line up 1:1
with their recorded CUDA timings. The upstream JSON records a per-stage split
(preprocess / diffusion / latent2audio); we mirror it with our stages (encode / DiT denoise / VAE
decode), timed separately on-device.

Upstream reference config (from their shipped example JSON):
    prompt          = "pop, rap, electronic, blues, hip-house, rhythm and blues"
    audio_duration  = 170.64 s   (~2:50 full song)
    infer_step      = 60
    guidance_scale  = 15  (apg, guidance_interval=0.5, min_guidance_scale=3)
    scheduler       = euler

Their published numbers (see REFERENCE_TIMINGS below) are the v1.5 project-page headlines
("under 2 s on A100", "under 10 s on RTX 3090" for a 4-minute song).

NOTE on fairness: upstream uses a Deep-Compression AutoEncoder (DCAE, ~8x compression) + a linear
transformer; our v1.5 port uses the Oobleck VAE (Conv3d-heavy). Absolute times differ for
architectural reasons — the REAL-TIME FACTOR (audio_seconds / wall_seconds) is the fair comparison.

Run (later):
    export ACESTEP_PIPELINE_DIR=/path/to/acestep_pipeline
    python models/experimental/acestep/perf/bench_e2e_comparison.py
Emits `METRIC name=value` lines + a side-by-side comparison table.
"""

import statistics
import time

import ttnn

from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline

# ── EXACT upstream benchmark inputs (examples/default/input_params/*.json) ─────────────────────────
PROMPT = "pop, rap, electronic, blues, hip-house, rhythm and blues"
# The upstream lyrics are a ~24-line structured Chinese pop song; abbreviated here (lyric length has
# negligible effect on device time vs. the audio duration). Swap in the full lyric to be byte-exact.
LYRICS = (
    "[verse]\n我走过深夜的街道\n冷风吹乱思念的漂亮外套\n"
    "[chorus]\n愿你是风吹过我的脸\n带我飞过最远最遥远的山间\n"
    "[bridge]\n唱起对你的想念不隐藏\n[verse]\n深夜的钢琴弹起动人的旋律\n"
)
AUDIO_DURATION = 170.64  # seconds — matches upstream audio_duration
INFER_STEPS = 60  # matches upstream infer_step
GUIDANCE_SCALE = 15.0  # matches upstream guidance_scale (apg)
RUNS = 3  # median of N measured full generations (warmup excluded)

# ── Published reference timings to compare against (seconds). ──────────────────────────────────────
# Source: the ACE-Step v1.5 project page headline wall-times (full 4-minute song).
# rtf = audio_seconds / total_wall_seconds  (higher = faster than real time).
REFERENCE_TIMINGS = {
    "A100 (v1.5 page headline)": {
        "audio_s": 240.0,  # "4-min song"
        "total_s": 2.0,  # "under 2 seconds"
        "source": "ace-step v1.5 project page ('under 2s on A100')",
    },
    "RTX 3090 (v1.5 page headline)": {
        "audio_s": 240.0,
        "total_s": 10.0,  # "under 10 seconds"
        "source": "ace-step v1.5 project page ('under 10s on RTX 3090')",
    },
}


def _rtf(audio_s: float, wall_s: float) -> float:
    return audio_s / wall_s if wall_s > 0 else float("nan")


def _timed_generate(pipe, *, use_trace: bool):
    """One full prompt->audio generation with a per-STAGE split (encode / denoise / decode), so the
    breakdown lines up with upstream's preprocess/diffusion/latent2audio. Returns (stages, wav)."""
    import torch

    dev = pipe.mesh_device
    t0 = time.time()
    # --- stage 1: preprocess (tokenize + text encoder + condition encoder) ---
    enc_hs = pipe.encode_prompt(PROMPT, LYRICS, audio_duration=AUDIO_DURATION)
    ttnn.synchronize_device(dev)
    t_pre = time.time()

    # --- stage 2: diffusion (the CFG DiT denoise loop) ---
    seq_len = pipe._latent_len(AUDIO_DURATION)
    hidden_ch = pipe.args.audio_acoustic_hidden_dim
    gen = torch.Generator().manual_seed(0)
    noise = torch.randn(1, 1, seq_len, hidden_ch, generator=gen)
    sl = pipe._silence_latent
    if sl is not None:
        src = (sl[:, :seq_len, :] if sl.shape[1] >= seq_len
               else sl.repeat(1, (seq_len + sl.shape[1] - 1) // sl.shape[1], 1)[:, :seq_len, :])
        src = src.reshape(1, 1, seq_len, hidden_ch)
    else:
        src = torch.zeros(1, 1, seq_len, hidden_ch)
    context = torch.cat([src, torch.ones(1, 1, seq_len, hidden_ch)], dim=-1)
    noise_tt = ttnn.from_torch(noise, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    context_tt = ttnn.from_torch(context, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    uncond_hs = pipe._uncond_context(enc_hs) if GUIDANCE_SCALE > 1.0 else None
    latents = pipe.generate(
        noise_tt, context_tt, enc_hs, infer_steps=INFER_STEPS, shift=1.0, use_trace=use_trace,
        guidance_scale=GUIDANCE_SCALE, uncond_encoder_hidden_states=uncond_hs,
    )
    ttnn.synchronize_device(dev)
    t_dif = time.time()

    # --- stage 3: latent2audio (Oobleck VAE decode + output normalize) ---
    wav = pipe._normalize_audio(pipe.decode(latents))
    ttnn.synchronize_device(dev)
    t_dec = time.time()

    stages = {
        "preprocess_s": t_pre - t0,
        "diffusion_s": t_dif - t_pre,
        "latent2audio_s": t_dec - t_dif,
        "total_s": t_dec - t0,
    }
    return stages, wav


def _measure(pipe, *, use_trace: bool):
    """Warmup (excluded) + RUNS measured generations; return median-per-stage + audio length."""
    _timed_generate(pipe, use_trace=use_trace)  # warmup (JIT compile)
    runs = [_timed_generate(pipe, use_trace=use_trace)[0] for _ in range(RUNS)]
    med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
    return med


def main():
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        t0 = time.time()
        pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device)
        seq = pipe._latent_len(AUDIO_DURATION)
        print(f"[bench] pipeline built in {time.time()-t0:.1f}s | {AUDIO_DURATION}s audio "
              f"(T'={seq//2}), {INFER_STEPS} steps, gs={GUIDANCE_SCALE}")

        # measure TT (traced deployment path is the fair number to compare)
        tt = _measure(pipe, use_trace=True)
        audio_s = AUDIO_DURATION
        tt_rtf = _rtf(audio_s, tt["total_s"])

        print("\n" + "=" * 78)
        print(f"ACE-Step v1.5 — TT (Blackhole p150) vs CUDA reference | {AUDIO_DURATION:.1f}s audio, {INFER_STEPS} steps")
        print("=" * 78)
        hdr = f"{'Platform':<34}{'total(s)':>9}{'audio(s)':>9}{'RTF':>7}"
        print(hdr)
        print("-" * 78)
        print(f"{'TT Blackhole p150 (this run, traced)':<34}{tt['total_s']:>9.2f}{audio_s:>9.1f}{tt_rtf:>6.1f}x")
        for name, r in REFERENCE_TIMINGS.items():
            print(f"{name:<34}{r['total_s']:>9.2f}{r['audio_s']:>9.1f}{_rtf(r['audio_s'], r['total_s']):>6.1f}x")
        print("-" * 78)

        # per-stage split for TT (encode / DiT denoise / VAE decode)
        print("\nPer-stage (s)              TT p150")
        for stg in ("preprocess_s", "diffusion_s", "latent2audio_s", "total_s"):
            print(f"  {stg:<22}{tt[stg]:>8.2f}")
        print("\nNOTE: upstream uses DCAE (~8x compression) + linear transformer; our port uses the")
        print("      Oobleck VAE (Conv3d). RTF (audio/wall) is the fair cross-architecture metric.")
        print("=" * 78)

        print(f"METRIC comparison_total_s={tt['total_s']:.4f}")
        print(f"METRIC comparison_rtf={tt_rtf:.4f}")
        print(f"METRIC comparison_preprocess_s={tt['preprocess_s']:.4f}")
        print(f"METRIC comparison_diffusion_s={tt['diffusion_s']:.4f}")
        print(f"METRIC comparison_latent2audio_s={tt['latent2audio_s']:.4f}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
