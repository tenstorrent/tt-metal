# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable TTS demo for coqui/XTTS-v2 on Tenstorrent hardware.

Runs the SAME chained TTNN pipeline as the e2e test (`tt/pipeline.py::run_tts`):
text + speaker reference -> 24 kHz speech waveform, produced entirely by the
graduated native TTNN stubs. Emits a .wav and prints the achieved PCC vs the
HF/Coqui reference.

Run:
    ./python_env/bin/python -m models.demos.xtts_v2.demo.demo_tts \
        --text "hello world." --language en --tokens 40 --out /tmp/xtts_tt.wav
"""

from __future__ import annotations

import argparse
import importlib.util as ilu
import os
import struct
import wave

import torch

import ttnn
from models.demos.xtts_v2.tt import pipeline as P

HF_MODEL_ID = "coqui/XTTS-v2"
OUTPUT_SR = 24000


def _load_reference():
    here = os.path.dirname(os.path.abspath(__file__))
    rl = os.path.normpath(os.path.join(here, "..", "tests", "pcc", "_reference_loader.py"))
    spec = ilu.spec_from_file_location("_reference_loader", rl)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_reference_model(HF_MODEL_ID)


def _write_wav(path, wav, sr):
    wav = torch.as_tensor(wav).reshape(-1).float()
    wav = (wav / (wav.abs().max() + 1e-8) * 0.98 * 32767).clamp(-32768, 32767).short()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % wav.numel(), *wav.tolist()))


def main():
    ap = argparse.ArgumentParser(description="XTTS-v2 TTNN text-to-speech demo")
    ap.add_argument("--text", default="hello world.")
    ap.add_argument("--language", default="en")
    ap.add_argument("--tokens", type=int, default=40, help="AR decode horizon N")
    ap.add_argument("--out", default="/tmp/xtts_tt.wav")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        model = _load_reference()
        res = P.run_tts(device, model, text=args.text, language=args.language, N=args.tokens)
        print(f"e2e PCC={res['e2e_pcc']}")
        _write_wav(args.out, res["wav_tt"], OUTPUT_SR)
        print(f"wrote TT waveform -> {args.out}  ({res['wav_tt'].numel()} samples @ {OUTPUT_SR} Hz)")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
