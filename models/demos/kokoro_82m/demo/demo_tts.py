# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Kokoro-82M text-to-speech demo (Call 1).

Runs the ONE shared TTNN pipeline (tt/pipeline.py::run_tts) — the same wiring
the e2e test asserts — on a real phoneme string + real voice, and writes the
24 kHz waveform to disk.

    python -m models.demos.kokoro_82m.demo.demo_tts \
        --phonemes "kˈOkəɹO ɪz ˈoʊpən sˈOɹs" --voice af_heart --out out.wav
"""
from __future__ import annotations

import argparse
import os
import sys

import ttnn
from models.demos.kokoro_82m.tt import pipeline as P


def main():
    ap = argparse.ArgumentParser(description="Kokoro-82M TTNN text->speech demo")
    ap.add_argument("--phonemes", default=P.DEFAULT_PHONEMES, help="Kokoro phoneme string")
    ap.add_argument("--voice", default="af_heart", help="Kokoro voice pack name")
    ap.add_argument("--out", default="kokoro_tt.wav", help="output wav path")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "pcc"))
    from _reference_loader import load_reference_model

    model = load_reference_model("hexgrad/Kokoro-82M").float().eval()

    input_ids, ref_s = P.build_input(model, phonemes=args.phonemes, voice=args.voice)
    gold, gold_dur = P.hf_reference_tts(model, input_ids, ref_s, speed=args.speed)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        pipe = P.build_pipeline(device, model=model)
        wav, pred_dur = P.run_tts(pipe, input_ids, ref_s, speed=args.speed)
    finally:
        ttnn.close_device(device)

    wav_pcc = P.comp_pcc_flat(gold, wav)
    spec_pcc = P.log_spectrogram_pcc(gold, wav)
    print(f"phonemes={args.phonemes!r} voice={args.voice} tokens={input_ids.shape[-1]}")
    print(f"pred_dur (TT) = {pred_dur.tolist()}")
    print(f"pred_dur matches HF: {bool((pred_dur == gold_dur).all())}")
    print(f"waveform samples: {wav.numel()}  ({wav.numel()/24000:.2f}s @ 24kHz)")
    print(f"e2e PCC={wav_pcc}")  # raw waveform (NSF phase-chaotic)
    print(f"e2e log-spectrogram PCC={spec_pcc}")  # phase-invariant fidelity

    try:
        import soundfile as sf

        sf.write(args.out, wav.numpy(), 24000)
        print(f"wrote {args.out}")
    except Exception:
        import wave

        w = wave.open(args.out, "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        pcm = (wav.clamp(-1, 1) * 32767).short().numpy().tobytes()
        w.writeframes(pcm)
        w.close()
        print(f"wrote {args.out} (16-bit PCM)")


if __name__ == "__main__":
    main()
