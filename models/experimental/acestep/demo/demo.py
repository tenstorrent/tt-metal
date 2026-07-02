# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 — text-to-music demo on a Tenstorrent Blackhole p150.

A short, readable example: a prompt + lyrics go in, a 48 kHz stereo `.wav` song comes out, using
the one-call `generate_song` API. The whole generation stack (tokenizer -> Qwen3 text encoder ->
condition encoder -> 24-layer DiT flow-matching denoise -> Oobleck VAE decode) runs on the device.

This is a usage sample for evaluators (not a test, not committed). Just run it:

    python models/experimental/acestep/demo/demo.py

Prerequisites:
  - The pipeline checkpoints (DiT + VAE + text encoder). Download once (resolved from the HF cache,
    or set ACESTEP_PIPELINE_DIR):
        python -c "from huggingface_hub import snapshot_download; snapshot_download('ACE-Step/Ace-Step1.5')"
  - soundfile for writing the .wav:  uv pip install soundfile

Honest scope: the on-device stack does the real generation. The LM planner (optional prompt
expander) is not used — text-to-music runs directly from the prompt/lyrics via the text encoder.
"""

import time

import ttnn

from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline

# --- what to generate ---
PROMPT = "upbeat synthwave, driving bass, warm analog pads, nostalgic 80s energy"
LYRICS = "neon lights over the city tonight, we ride the endless skyline"
SECONDS = 8.0
STEPS = 30  # HF generate_audio default
OUT_PATH = "acestep_demo_song.wav"


def main():
    print(f"[demo] prompt : {PROMPT!r}")
    print(f"[demo] lyrics : {LYRICS!r}")
    print(f"[demo] length : {SECONDS}s  |  steps: {STEPS}")

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        # Build the full text-to-music pipeline once (DiT + VAE + text/condition encoders + tokenizer).
        t0 = time.time()
        pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device)
        print(f"[demo] pipeline ready in {time.time() - t0:.1f}s")

        # One call: prompt/lyrics -> 48 kHz stereo waveform on the device.
        t1 = time.time()
        wav = pipe.generate_song(PROMPT, lyrics=LYRICS, seconds=SECONDS, infer_steps=STEPS, seed=0)
        print(f"[demo] generated {wav.shape[-1] / pipe.SAMPLE_RATE:.2f}s of audio in {time.time() - t1:.1f}s")

        # Save the song.
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
