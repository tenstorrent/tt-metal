# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable TTS demo for microsoft/VibeVoice-1.5B on Tenstorrent hardware.

Runs the SAME chained TTNN pipeline as the e2e test (`tt/pipeline.py::run_tts`):
script text + reference voice sample -> 24 kHz speech waveform, produced by the
19 graduated native TTNN stubs. Writes a .wav and prints the token schedule,
per-stage PCC, and final e2e PCC vs the HF golden.

Run (quality defaults: S=20 ddpm steps, 64-frame horizon so the utterance finishes naturally):
    ./python_env/bin/python -m models.demos.vibevoice_1_5b.demo.demo_tts \
        --text "Speaker 0: Hello there." --no-golden --out /tmp/vibevoice_tt.wav
"""

from __future__ import annotations

import argparse
import struct
import wave

import torch

import ttnn
from models.demos.vibevoice_1_5b.tt import pipeline as P
from models.demos.vibevoice_1_5b.tt._golden import reference as R

OUTPUT_SR = 24000


def _write_wav(path, wav, sr):
    wav = torch.as_tensor(wav).reshape(-1).float()
    wav = (wav / (wav.abs().max() + 1e-8) * 0.98 * 32767).clamp(-32768, 32767).short()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(s)) for s in wav.tolist()))


def main():
    ap = argparse.ArgumentParser(description="VibeVoice-1.5B TTS on Tenstorrent")
    ap.add_argument("--text", default="Speaker 0: Hello there, this is a test.")
    ap.add_argument(
        "--frames", type=int, default=64, help="max speech-diffusion frames (horizon N; LM stops earlier naturally)"
    )
    ap.add_argument("--ddpm-steps", type=int, default=20, help="DDPM inference steps S (20 = quality anchor)")
    ap.add_argument("--out", default="/tmp/vibevoice_tt.wav")
    ap.add_argument("--no-golden", action="store_true", help="skip HF golden PCC (faster)")
    ap.add_argument("--eager", action="store_true", help="use the eager path (default: traced+2CQ, real-time)")
    args = ap.parse_args()
    use_trace = not args.eager

    torch.manual_seed(0)
    model = R.load_reference_model()
    processor = R.build_processor()
    inputs = dict(R.make_inputs(processor, args.text, R.default_voice_sample()))
    inputs["noises"] = R.make_noises(args.frames + 2, int(model.config.acoustic_vae_dim))
    golden = None
    if not args.no_golden:
        golden = R.hf_reference_tts(model, processor, inputs, N=args.frames, S=args.ddpm_steps, noises=inputs["noises"])

    # The traced+2CQ path (default) needs a trace region and a 2nd command queue; the eager path
    # needs neither. Open the device accordingly so the demo runs the same code the customer ships.
    open_kwargs = dict(device_id=0, l1_small_size=24576)
    if use_trace:
        open_kwargs.update(trace_region_size=400000000, num_command_queues=2)
    device = ttnn.open_device(**open_kwargs)
    try:
        res = P.run_tts(
            device,
            model,
            processor,
            inputs=inputs,
            N=args.frames,
            S=args.ddpm_steps,
            golden=golden,
            use_trace=use_trace,
            two_cq=use_trace,
        )
        # Pull the device-resident waveform to host BEFORE closing the device.
        wav = P._th(res["waveform_tt"]).reshape(-1)
    finally:
        ttnn.close_device(device)
    print(f"path: {'traced+2CQ' if use_trace else 'eager'}")
    _write_wav(args.out, wav, OUTPUT_SR)
    print(f"token schedule: {res['tokens']}")
    print(f"wrote {wav.shape[0]} samples ({wav.shape[0]/OUTPUT_SR:.2f}s) to {args.out}")
    if golden is not None:
        print(f"e2e PCC={res['e2e_pcc']}")


if __name__ == "__main__":
    main()
