# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Write the per-request waveforms produced by `phase_tt_mesh.py` out as .wav files.

`phase_tt_mesh.py` saves one tensor per request (`vocoder_wav_tt_r{i}.pt`, Block 4 run on
device). This just decodes them to 24 kHz wavs and reports each one's duration, so the mesh
run's audio can be listened to without loading coqui at all — only torch + soundfile are
needed, so it runs in either venv.

    python mesh_write_wavs.py --dir /path/to/xtts_out --out /path/to/xtts_out/wav
"""

import argparse
import glob
import os
import re

import soundfile as sf
import torch

SR = 24000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir holding vocoder_wav_tt_r*.pt")
    ap.add_argument("--out", default=None, help="output dir for wavs (default: <dir>/wav)")
    ap.add_argument("--reqs", default=None, help="optional dir of phase-A request dirs, to label each wav's text")
    args = ap.parse_args()

    out = args.out or os.path.join(args.dir, "wav")
    os.makedirs(out, exist_ok=True)

    paths = sorted(
        glob.glob(os.path.join(args.dir, "vocoder_wav_tt_r*.pt")),
        key=lambda p: int(re.search(r"_r(\d+)\.pt$", p).group(1)),
    )
    if not paths:
        raise SystemExit(f"no vocoder_wav_tt_r*.pt in {args.dir}")

    total = 0.0
    for p in paths:
        i = int(re.search(r"_r(\d+)\.pt$", p).group(1))
        wav = torch.load(p).cpu().float().squeeze().numpy()
        dst = os.path.join(out, f"r{i:02d}.wav")
        sf.write(dst, wav, SR)
        secs = len(wav) / SR
        total += secs
        text = ""
        if args.reqs:
            tf = os.path.join(args.reqs, f"r{i}", "text.txt")
            if os.path.exists(tf):
                text = "  " + open(tf).read().strip()[:60]
        print(f"r{i:02d}: {secs:6.2f}s  {dst}{text}")
    print(f"\n{len(paths)} waveforms, {total:.1f}s of speech total, written to {out}")


if __name__ == "__main__":
    main()
