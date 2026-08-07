"""Is 28.6 dB against the source a fair measure of quality, or an artefact of the metric?

`score_wavs.py` reports ~28.6 dB for the CPU decode against the original input, far below the ~49 dB
the device scores against that decode. The gap is expected -- different references -- but waveform
PSNR is a harsh metric for a GAN vocoder: BigVGAN is trained on adversarial and spectral objectives,
not L2, so it reconstructs perceptually without being sample-aligned. Any constant delay or phase
rotation craters PSNR while changing nothing audible.

So this separates "the codec really loses that much" from "the metric is measuring the wrong thing":

  best lag        cross-correlate decode against source; a non-zero lag means the whole comparison
                  was misaligned and every PSNR-vs-source figure is pessimistic
  PSNR at lag     re-score after removing it
  gain-corrected  allow a single scalar level difference, which PSNR also punishes
  log-spectral    magnitude-only distance -- ignores phase entirely, closer to what is heard
  per-band SNR    where in the spectrum the loss actually sits
"""

import math
import os

import numpy as np
import soundfile as sf

CLIPS_DIR = "/data/rshirvani/audio_compare/clips"
CLIPS = ["voice_libri1", "voice_libri2", "music_trumpet", "music_brahms"]
SR = 32000


def read(clip, suffix):
    data, _ = sf.read(os.path.join(CLIPS_DIR, f"{clip}_{suffix}.wav"))
    return np.asarray(data, dtype=np.float64)


def psnr(ref, test):
    n = min(len(ref), len(test))
    mse = float(np.mean((ref[:n] - test[:n]) ** 2))
    if mse == 0:
        return float("inf")
    return 20 * math.log10(float(np.abs(ref[:n]).max())) - 10 * math.log10(mse)


def best_lag(ref, test, max_lag=512):
    """Integer lag maximising cross-correlation, searched by FFT."""
    n = 1 << (len(ref) + len(test)).bit_length()
    R = np.fft.rfft(ref, n)
    T = np.fft.rfft(test, n)
    xc = np.fft.irfft(R * np.conj(T), n)
    cand = np.concatenate([xc[: max_lag + 1], xc[-max_lag:]])
    idx = int(np.argmax(cand))
    return idx if idx <= max_lag else idx - len(cand)


def shifted(test, lag, length):
    if lag > 0:
        out = np.concatenate([np.zeros(lag), test])[:length]
    elif lag < 0:
        out = test[-lag:]
    else:
        out = test[:length]
    if len(out) < length:
        out = np.concatenate([out, np.zeros(length - len(out))])
    return out[:length]


def log_spec_distance(a, b, n_fft=1024, hop=256):
    w = np.hanning(n_fft)

    def spec(x):
        frames = [np.abs(np.fft.rfft(x[i : i + n_fft] * w)) for i in range(0, len(x) - n_fft, hop)]
        return np.log(np.maximum(np.array(frames), 1e-5))

    n = min(len(a), len(b))
    return float(np.mean(np.abs(spec(a[:n]) - spec(b[:n]))))


def band_snr(ref, test, edges=(0, 500, 2000, 6000, 16000)):
    n = min(len(ref), len(test))
    R = np.fft.rfft(ref[:n])
    E = np.fft.rfft(test[:n] - ref[:n])
    freqs = np.fft.rfftfreq(n, 1 / SR)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        sig = float(np.sum(np.abs(R[m]) ** 2))
        err = float(np.sum(np.abs(E[m]) ** 2))
        out.append(10 * math.log10(sig / err) if err > 0 and sig > 0 else float("inf"))
    return out


def main():
    print("=== decode vs SOURCE: is the low PSNR real, or a misalignment artefact? ===")
    print(f"{'clip':<15} {'variant':<12} {'lag':>5} {'PSNR raw':>9} {'PSNR@lag':>9} {'+gain':>8} {'logspec':>8}")
    print("-" * 74)
    for clip in CLIPS:
        src = read(clip, "0_source")
        for name, sfx in (("cpu", "1_cpu"), ("device_post", "2_device")):
            dec = read(clip, sfx)
            n = min(len(src), len(dec))
            s, d = src[:n], dec[:n]
            lag = best_lag(s, d)
            da = shifted(d, lag, n)
            g = float(np.dot(s, da) / np.dot(da, da)) if np.dot(da, da) > 0 else 1.0
            print(
                f"{clip:<15} {name:<12} {lag:>5} {psnr(s, d):>9.2f} {psnr(s, da):>9.2f} "
                f"{psnr(s, g * da):>8.2f} {log_spec_distance(s, d):>8.4f}"
            )

    print("\n=== where the codec loss sits: per-band SNR of the CPU decode vs source (dB) ===")
    print(f"{'clip':<15} {'0-0.5k':>8} {'0.5-2k':>8} {'2-6k':>8} {'6-16k':>8}")
    print("-" * 52)
    for clip in CLIPS:
        src, cpu = read(clip, "0_source"), read(clip, "1_cpu")
        b = band_snr(src, cpu)
        print(f"{clip:<15} " + " ".join(f"{v:>8.2f}" for v in b))


if __name__ == "__main__":
    main()
