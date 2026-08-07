"""Full PSNR matrix over the saved WAVs -- including the CPU decode's own PSNR against the source.

The device columns elsewhere are scored against the CPU decode, which answers "how faithfully does the
hardware reproduce the reference". It does not say how good the reference itself is. The VAE is lossy,
so the round trip source -> encode -> decode costs something on CPU alone, and that number is the
ceiling any device can reach. Scoring everything against the source puts the hardware error in
proportion to the codec error.

Reads only WAVs, so no device and no model are needed.
"""

import math
import os

import numpy as np
import soundfile as sf

CLIPS_DIR = "/data/rshirvani/audio_compare/clips"
CLIPS = ["voice_libri1", "voice_libri2", "music_trumpet", "music_brahms"]
VARIANTS = [("cpu", "1_cpu"), ("device_post", "2_device"), ("device_pre", "3_device_prefix")]


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    n = min(len(ref), len(test))
    mse = float(np.mean((ref[:n] - test[:n]) ** 2))
    if mse == 0.0:
        return float("inf")
    peak = float(np.abs(ref[:n]).max())
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


def rel_rmse(ref: np.ndarray, test: np.ndarray) -> float:
    n = min(len(ref), len(test))
    return float(np.sqrt(np.mean((test[:n] - ref[:n]) ** 2)) / np.std(ref[:n]))


def read(clip: str, suffix: str):
    path = os.path.join(CLIPS_DIR, f"{clip}_{suffix}.wav")
    data, _ = sf.read(path)
    return np.asarray(data, dtype=np.float64)


def main():
    print("=== PSNR vs the ORIGINAL SOURCE (what the listener actually loses) ===")
    print(f"{'clip':<15} {'CPU dB':>9} {'dev post dB':>12} {'dev pre dB':>11}   {'post-CPU':>9} {'pre-CPU':>8}")
    print("-" * 72)
    rows = []
    for clip in CLIPS:
        src = read(clip, "0_source")
        vals = {name: psnr(src, read(clip, sfx)) for name, sfx in VARIANTS}
        rows.append(vals)
        print(
            f"{clip:<15} {vals['cpu']:>9.2f} {vals['device_post']:>12.2f} {vals['device_pre']:>11.2f}   "
            f"{vals['device_post'] - vals['cpu']:>+9.2f} {vals['device_pre'] - vals['cpu']:>+8.2f}"
        )
    n = len(rows)
    print("-" * 72)
    print(
        f"{'mean':<15} {sum(r['cpu'] for r in rows) / n:>9.2f} "
        f"{sum(r['device_post'] for r in rows) / n:>12.2f} "
        f"{sum(r['device_pre'] for r in rows) / n:>11.2f}   "
        f"{sum(r['device_post'] - r['cpu'] for r in rows) / n:>+9.2f} "
        f"{sum(r['device_pre'] - r['cpu'] for r in rows) / n:>+8.2f}"
    )

    print("\n=== for reference: device vs the CPU decode (hardware error alone) ===")
    print(f"{'clip':<15} {'post dB':>9} {'pre dB':>9} {'post rel_rmse':>14} {'pre rel_rmse':>13}")
    print("-" * 66)
    post_l, pre_l = [], []
    for clip in CLIPS:
        cpu = read(clip, "1_cpu")
        post, pre = read(clip, "2_device"), read(clip, "3_device_prefix")
        p, q = psnr(cpu, post), psnr(cpu, pre)
        post_l.append(p)
        pre_l.append(q)
        print(f"{clip:<15} {p:>9.2f} {q:>9.2f} {rel_rmse(cpu, post):>14.3e} {rel_rmse(cpu, pre):>13.3e}")
    print("-" * 66)
    print(f"{'mean':<15} {sum(post_l) / len(post_l):>9.2f} {sum(pre_l) / len(pre_l):>9.2f}")


if __name__ == "__main__":
    main()
