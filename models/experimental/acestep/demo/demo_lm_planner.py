# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 — end-to-end LM planner + text-to-music demo on a Tenstorrent Blackhole p150.

The FULL hybrid pipeline in one script, all on the device:

    plain user query
        -> 5Hz LM planner ("Songwriter", 28-layer Qwen3-1.7B)  : Chain-of-Thought song blueprint
                                                                  (caption, bpm, key, duration, ...)
        -> Qwen3 text encoder + condition encoder               : DiT cross-attn conditioning
        -> 24-layer DiT flow-matching denoise (CFG, guidance 7) : audio latents
        -> Oobleck VAE decode                                   : 48 kHz stereo waveform
        -> .wav

This shows ACE-Step's hybrid architecture: the LM planner turns a simple query into a comprehensive
blueprint, which drives the diffusion model. Everything runs on the Tenstorrent device.

    python models/experimental/acestep/demo/demo_lm_planner.py

Prerequisites:
  - The pipeline checkpoints + the LM planner. Download once:
        python -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/Ace-Step1.5')"
    (the base DiT/VAE come from ACE-Step/acestep-v15-base, resolved from the HF cache or
     ACESTEP_PIPELINE_DIR).
  - soundfile for writing the .wav:  uv pip install soundfile

This is a usage sample for evaluators (not a test). Edit QUERY to try your own.
"""

import time

import ttnn

from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline

# --- what to generate: just a plain-English request. The LM planner writes the blueprint. ---
QUERY = "Write an upbeat synthwave song about driving through a neon city at night"
# Optional: give the singer words. Leave "" to let the track be instrumental / model-driven.
LYRICS = """[Verse]
Neon lights over the city tonight
We ride the endless skyline
[Chorus]
We are electric, we are alive
Racing the moon through the night
"""
OUT_PATH = "acestep_lm_planner_song.wav"


def main():
    print(f"[demo] user query : {QUERY!r}")

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        # Build the full stack INCLUDING the LM planner (with_lm=True).
        t0 = time.time()
        pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device, with_lm=True)
        print(f"[demo] pipeline + LM planner ready in {time.time() - t0:.1f}s")

        # 1) LM planner: query -> song blueprint (on-device Chain-of-Thought).
        t1 = time.time()
        plan = pipe.plan_song(QUERY, max_new_tokens=160)
        print(f"[demo] LM planner wrote the blueprint in {time.time() - t1:.1f}s:")
        print("       caption      :", (plan.get("caption") or "")[:100])
        print("       bpm          :", plan.get("bpm"))
        print("       keyscale     :", plan.get("keyscale"))
        print("       timesignature:", plan.get("timesignature"))
        print("       duration     :", plan.get("seconds"))

        # 2) Feed the blueprint into the diffusion text-to-music pipeline.
        caption = plan.get("caption") or QUERY
        seconds = plan.get("seconds") or 20.0
        seconds = max(10.0, min(float(seconds), 30.0))  # keep the demo runtime reasonable
        t2 = time.time()
        wav = pipe.generate_song(
            caption,
            lyrics=LYRICS,
            seconds=seconds,
            infer_steps=60,
            seed=0,
            guidance_scale=7.0,
            bpm=plan.get("bpm"),
            keyscale=plan.get("keyscale") or "",
            timesignature=plan.get("timesignature") or "",
        )
        dur = wav.shape[-1] / pipe.SAMPLE_RATE
        print(f"[demo] generated {dur:.2f}s of audio in {time.time() - t2:.1f}s")

        # 3) Save the song.
        audio = wav.squeeze(0).transpose(0, 1).clamp(-1, 1).float().numpy()  # [samples, 2]
        try:
            import soundfile as sf

            sf.write(OUT_PATH, audio, pipe.SAMPLE_RATE)
            print(f"[demo] wrote {OUT_PATH}  ({wav.shape[-1]} samples, {pipe.SAMPLE_RATE} Hz stereo)")
        except ImportError:
            import numpy as np

            np.save(OUT_PATH + ".npy", audio)
            print(f"[demo] soundfile not installed; saved {OUT_PATH}.npy (uv pip install soundfile for .wav)")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
