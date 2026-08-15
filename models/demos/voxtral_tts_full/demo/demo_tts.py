# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 1 -- text-to-speech demo: real text in, a playable 24 kHz .wav out.

    ./python_env/bin/python -m models.demos.voxtral_tts_full.demo.demo_tts \
        --text "Hello from Tenstorrent." --voice neutral_male --out /tmp/voxtral_tt.wav

The chained forward pass is NOT duplicated here -- this entry point calls the same
`tt/pipeline.py::build_pipeline` + `run_tts` that `tests/e2e/test_e2e_tts.py` asserts on, so a
green test is a statement about this demo's code path.

`--compare` additionally runs the HF reference and prints the e2e PCC.
"""
from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import torch
import ttnn

from models.demos.voxtral_tts_full.tt.pipeline import (
    TRACE_REGION_SIZE,
    build_pipeline,
    load_hf_model,
    pcc,
)

SAMPLING_RATE = 24000


def save_wav(waveform: torch.Tensor, path: str, sample_rate: int = SAMPLING_RATE) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcm = (waveform.detach().reshape(-1).float().clamp(-1, 1).numpy() * 32767).astype("<i2")
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())
    return path


def main():
    ap = argparse.ArgumentParser(description="Voxtral-TTS text-to-speech on Tenstorrent")
    ap.add_argument("--text", default=None, help="text to speak (default: the config's shipped prompt)")
    ap.add_argument("--voice", default=None, help="one of the 20 presets (default: config.default_voice)")
    ap.add_argument("--max-frames", type=int, default=8, help="safety cap; 12.5 frames = 1 s of audio")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the flow ODE's N(0,1) initial condition, as real inference draws it. "
                         "Same seed -> same audio. --zero-x0 integrates from zero instead (degenerate).")
    ap.add_argument("--zero-x0", action="store_true",
                    help="integrate the flow ODE from zero rather than a Gaussian draw (the "
                         "per-component PCC harness's initial condition)")
    ap.add_argument("--layers", type=int, default=None, help="cap the depth built (None = every layer)")
    ap.add_argument("--out", default="/tmp/voxtral_tt.wav")
    ap.add_argument("--compare", action="store_true", help="also run the HF reference and print the e2e PCC")
    args = ap.parse_args()

    hf = load_hf_model()
    device = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION_SIZE)
    try:
        t0 = time.time()
        pipe = build_pipeline(device, model=hf, layers=args.layers)
        print(f"[demo] built in {time.time() - t0:.1f}s | stages={pipe.stages} | depths={pipe.depths}")

        enc = pipe.encode_inputs(
            text=args.text, voice=args.voice, max_frames=args.max_frames,
            seed=None if args.zero_x0 else args.seed,
        )
        print(f"[demo] prompt {enc['prompt_len']} ids | voice {enc['voice']!r} | max_frames {enc['max_frames']}")

        t0 = time.time()
        out = pipe.run_tts(enc)
        elapsed = time.time() - t0
        if out["waveform"] is None:
            print("[demo] the model emitted [END_AUDIO] on the first frame -- nothing to decode")
            return

        waveform = ttnn.to_torch(out["waveform"]).float()
        seconds = waveform.shape[-1] / SAMPLING_RATE
        path = save_wav(waveform, args.out)
        print(
            f"[demo] {out['n_frames']} frames -> {tuple(waveform.shape)} = {seconds:.2f}s @ {SAMPLING_RATE} Hz "
            f"| peak |x| {waveform.abs().max():.4f} | {elapsed:.1f}s"
        )
        print(f"[demo] graduated stubs invoked: {pipe.invocations}")
        print(f"[demo] -> {path}")

        if args.compare:
            from models.demos.voxtral_tts_full.tests.e2e.reference import golden

            ref, _ = golden(
                lambda: hf, enc["ids"], enc["voice"], enc["max_frames"],
                enc["x0_bank_host"], int(pipe.config.end_audio_id),
            )
            n = min(out["n_frames"], ref["n_frames"])
            got_c = ttnn.to_torch(out["frames"]).long()[:n]
            want_c = ref["frames"][:n]
            print(f"[demo] semantic codes exact: {int((got_c[:,0]==want_c[:,0]).sum())}/{n} frames")
            print(f"[demo] acoustic code flips per frame (of 36): "
                  f"{[int((got_c[i,1:]!=want_c[i,1:]).sum()) for i in range(n)]}")
            # The gate horizon and the full rollout, so this never reads as disagreeing with
            # tests/e2e/test_e2e_tts.py -- see README "Why Gate 3 is measured over a 2-frame horizon".
            gate = pcc(waveform[..., : 2 * 1920], ref["waveform"][..., : 2 * 1920])
            print(f"[demo] gate-horizon (2 frames) PCC={gate:.6f}")
            achieved = pcc(waveform[..., : n * 1920], ref["waveform"][..., : n * 1920])
            print(f"e2e PCC={achieved}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
