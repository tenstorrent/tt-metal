"""Build the Japanese reference voice for the quality harness.

Converts a Common Voice clip to 24 kHz mono, RMS-normalises it to match
``demo/jim_reference.wav``, and writes the ``.refcache.pt`` that
``server.encode_reference_audio`` loads; with the cache present it never shells
out to ffmpeg. CPU only; no device needed.

Source (CC0-1.0): fsicoli/common_voice_17_0, ja dev, common_voice_ja_28360668.mp3

    # from the repository root
    python models/demos/qwen3_tts/tests/jp_quality/make_reference.py --src /path/to/common_voice_ja_28360668.mp3
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

HERE = Path(__file__).resolve().parent
SR = 24000
TARGET_RMS = 0.0923  # RMS of demo/jim_reference.wav
TEXT = "パスタを茹でるときにお湯が少ないと驚くほど味が落ちる"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="common_voice_ja_28360668.mp3")
    ap.add_argument("--out", default=str(HERE / "jp_reference.wav"))
    a = ap.parse_args()

    x, sr = sf.read(a.src, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = soxr.resample(x, sr, SR, quality="VHQ").astype(np.float32)
    x *= TARGET_RMS / np.sqrt(np.mean(x**2))
    peak = float(np.abs(x).max())
    assert peak < 0.99, f"clipping after normalisation (peak {peak:.3f})"

    out = Path(a.out)
    sf.write(out, x, SR, subtype="PCM_16")
    out.with_suffix(".txt").write_text(TEXT + "\n", encoding="utf-8")

    # Re-read the 16-bit file so the cache matches what is on disk exactly.
    audio, _ = sf.read(out, dtype="float32")
    audio = torch.from_numpy(audio)

    from models.demos.qwen3_tts.reference.functional import speech_tokenizer_encoder_forward_mimi

    ref_codes = speech_tokenizer_encoder_forward_mimi(audio.unsqueeze(0)).squeeze(0).T  # [T, 16]
    cache = str(out.with_suffix("")) + ".refcache.pt"
    torch.save({"ref_codes": ref_codes, "audio_data": audio}, cache)

    rms = float(torch.sqrt((audio**2).mean()))
    print(f"wrote {out}  {len(audio)/SR:.2f}s  rms {rms:.4f}  peak {peak:.3f}")
    print(f"wrote {cache}  ref_codes {tuple(ref_codes.shape)}")


if __name__ == "__main__":
    main()
