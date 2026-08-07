"""Objective PERCEPTUAL metrics — the checks 6.58 said were missing, minus the human raters.

6.58 concluded "what remains is listening". That was too strong: MOS needs humans, but the
standard objective proxies for TTS quality were never run here. This adds three, none of which
requires a download:

  1. MCD (mel-cepstral distortion) with DTW alignment, device vs the fp32 CPU reference,
     end to end. THE standard objective TTS distance. DTW is required because the two are
     autoregressive and diverge in length (case 1: 99 device frames vs 97 reference), so a
     sample-aligned metric like PESQ would be meaningless on this pair.
     Rule of thumb from the voice-conversion literature: MCD < 4 dB is "hard to distinguish",
     6-8 dB is "noticeably different".

  2. F0 CONTOUR correlation and voiced-frame agreement — a prosody proxy, which is precisely what
     WER cannot see and what a listening pass is for.

  3. CODEC-ONLY comparison, sample-aligned. Feeding the SAME codes through the device codec and
     the fp32 reference codec removes the trajectory divergence entirely, so log-spectral distance
     and a segmental SNR are valid without DTW. This isolates Block 3, which today is reported
     only as waveform PCC 0.999920 — a number that says nothing about audibility.

Everything compares against the fp32 CPU reference, which is the ground truth on this branch.
"""
import glob
import json
import os
import wave

import librosa
import numpy as np

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
GEN = os.path.join(HERE, "generated")
SR = 24000


def read(p):
    with wave.open(p, "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return a.astype(np.float64) / 32768.0


def _melcep(y, n=25):
    """Mel-cepstra from a NATURAL-log mel spectrum.

    librosa.feature.mfcc applies power_to_db (10*log10), so its output is already dB-scaled and
    the standard MCD constant (10/ln10)*sqrt(2) double-counts it -- that is what produced the
    impossible 181 dB in the first run of this probe. MCD is defined on natural-log cepstra.
    """
    from scipy.fftpack import dct
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=256, n_mels=80)
    # RELATIVE floor, 80 dB below peak. An ABSOLUTE 1e-10 floor puts near-silent mel bins at
    # log(1e-10) = -23, where 1e-4 of added noise moves them by ~5 nats -- the first calibrated
    # run of this probe scored MCD(x, x+1e-4 noise) at 24.9 dB because of exactly that, i.e. it
    # was measuring its own noise floor rather than the signal.
    S = np.maximum(S, S.max() * 1e-8)
    return dct(np.log(S), axis=0, type=2, norm="ortho")[:n]


def mcd_dtw(a, b, n=25):
    """Mel-cepstral distortion, DTW-aligned, excluding c0 (energy) as convention requires."""
    fa, fb = _melcep(a, n), _melcep(b, n)
    _, wp = librosa.sequence.dtw(X=fa[1:], Y=fb[1:], metric="euclidean")
    diff = fa[1:, wp[:, 0]] - fb[1:, wp[:, 1]]
    return (10.0 / np.log(10)) * np.sqrt(2.0) * np.mean(np.sqrt((diff ** 2).sum(axis=0)))


def f0_stats(a, b):
    """Correlation of the log-F0 contours over commonly-voiced frames, plus voicing agreement."""
    kw = dict(fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=SR,
              frame_length=1024)
    fa, va, _ = librosa.pyin(a, **kw)
    fb, vb, _ = librosa.pyin(b, **kw)
    n = min(len(fa), len(fb))
    fa, fb, va, vb = fa[:n], fb[:n], va[:n], vb[:n]
    both = va & vb & ~np.isnan(fa) & ~np.isnan(fb)
    agree = float((va == vb).mean()) * 100
    if both.sum() < 8:
        return float("nan"), agree, int(both.sum())
    r = np.corrcoef(np.log(fa[both]), np.log(fb[both]))[0, 1]
    return float(r), agree, int(both.sum())


def main():
    fx = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"]

    # ---- CALIBRATE THE INSTRUMENT FIRST. An uncalibrated MCD is how the first run of this
    # probe reported 181 dB without anything flagging it as impossible. ----
    print("=== 0. MCD self-test (must hold or nothing below means anything) ===")
    ref = read(os.path.join(GEN, f"case1_{fx[1]['voice']}_prg_s0.wav"))
    other = read(os.path.join(GEN, f"case2_{fx[2]['voice']}_prg_s0.wav"))
    noisy = ref + np.random.default_rng(0).normal(0, 1e-4, len(ref))
    print(f"  MCD(x, x)                       {mcd_dtw(ref, ref):>8.3f} dB   (must be 0)")
    print(f"  MCD(x, x + 1e-4 noise)          {mcd_dtw(ref, noisy):>8.3f} dB   (must be small)")
    print(f"  MCD(x, DIFFERENT utterance)     {mcd_dtw(ref, other):>8.3f} dB   (must be large)")

    print("\n=== 1b. CONTROL: the model's own seed-to-seed variation ===")
    print("  Device and reference diverge because generation is STOCHASTIC, not because one is")
    print("  worse. Two device seeds bound how much of any device-vs-reference MCD is just that.")
    for ci in (1, 6):
        v = fx[ci]["voice"]
        try:
            s0 = read(os.path.join(GEN, f"case{ci}_{v}_prg_s0.wav"))
            s1 = read(os.path.join(GEN, f"case{ci}_{v}_prg_s1.wav"))
            r01, ag01, _ = f0_stats(s0, s1)
            print(f"  case {ci} {v:<16} MCD(seed0,seed1) {mcd_dtw(s0, s1):>6.2f} dB   "
                  f"logF0 r {r01:>6.3f}   voicing {ag01:>5.1f}%")
        except Exception as e:
            print(f"  case {ci}: {type(e).__name__}: {e}")

    print("\n=== 1+2. END TO END: device vs fp32 CPU reference (DTW-aligned) ===")
    print(f"  {'case':<26} {'MCD dB':>8} {'logF0 r':>9} {'voicing':>9} {'dev s':>7} {'ref s':>7}")
    pairs = []
    for p in sorted(glob.glob(os.path.join(GEN, "*_FP32REF_s*.wav"))):
        base = os.path.basename(p)
        ci = int(base.split("_")[0].replace("case", ""))
        seed = base.split("_s")[-1].replace(".wav", "")
        dev = os.path.join(GEN, f"case{ci}_{fx[ci]['voice']}_prg_s{seed}.wav")
        if not os.path.exists(dev):
            continue
        a, b = read(dev), read(p)
        m = mcd_dtw(a, b)
        r, agree, nv = f0_stats(a, b)
        pairs.append((ci, m, r))
        print(f"  case {ci} {fx[ci]['voice']:<14} s{seed} {m:>8.2f} {r:>9.3f} {agree:>8.1f}% "
              f"{len(a)/SR:>7.2f} {len(b)/SR:>7.2f}")
    print("  MCD guide: <4 dB hard to distinguish, 6-8 dB noticeably different.")

    print("\n=== 3. CODEC ONLY, sample-aligned (same codes through both codecs) ===")
    import torch
    import ttnn
    from models.experimental.voxtral_tts.reference import voxtral_codec_ref as cref
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_codec import TtVoxtralCodecDecoder
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device

    frames = torch.load(os.path.join(HERE, "tests", "real_frames_fixture.pt")).long()
    dev_h = open_device()
    try:
        wc = cref.load_codec_state()
        codes = cref.strip_offset_and_trim(frames)
        ref_w = cref.reference_decode(codes, wc).reshape(-1).double().numpy()
        tt_w = TtVoxtralCodecDecoder(dev_h)(codes).reshape(-1).double().numpy()
        n = min(len(ref_w), len(tt_w))
        r_, t_ = ref_w[:n], tt_w[:n]
        err = t_ - r_
        seg_snr = 10 * np.log10((r_ ** 2).mean() / max((err ** 2).mean(), 1e-20))
        Sr = np.abs(librosa.stft(r_, n_fft=1024, hop_length=256)) + 1e-10
        St = np.abs(librosa.stft(t_, n_fft=1024, hop_length=256)) + 1e-10
        lsd = np.mean(np.sqrt(np.mean((20 * np.log10(Sr / St)) ** 2, axis=0)))
        print(f"  frames {frames.shape[0]}, {n/SR:.2f}s")
        print(f"  SNR of the device codec vs fp32 reference   {seg_snr:>8.2f} dB")
        print(f"  log-spectral distance                       {lsd:>8.3f} dB")
        print(f"  MCD (sample-aligned, no DTW needed)         {mcd_dtw(r_, t_):>8.3f} dB")
        print("  (>20 dB SNR / <1 dB LSD is transparent for speech coding)")
    finally:
        ttnn.close_device(dev_h)


if __name__ == "__main__":
    main()
