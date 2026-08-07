"""The complete decode comparison in one place: pre-fix, post-fix, CPU, timing and quality.

Quality is recomputed from the WAVs in /data/rshirvani/audio_compare/clips/ every run, so it is
reproducible without a device. Timings cannot be recovered from a WAV, so they are recorded constants
taken from the two runs that produced those files:

    post-fix   cpu_vs_device.py on rouzbeh/audio-decode-exact-fp32  (9922d6106fe)
    pre-fix    the same harness on a full build of cglagovich/minimax-h3 (3fdb75f55e5),
               in .claude/worktrees/h3-prefix, verified to load its own _ttnn.so

fp32 throughout, 207 latents = 5.17 s of 32 kHz per clip, batch 2, steady state (decoder built once,
warm-up excluded, best of 3).

Two references are reported and they answer different questions:
    vs CPU     how closely the hardware reproduces the reference -- isolates our error
    vs SOURCE  what the listener loses end to end -- dominated by the lossy autoencoder
"""

import math
import os

import numpy as np
import soundfile as sf

CLIPS_DIR = "/data/rshirvani/audio_compare/clips"
CLIPS = ["voice_libri1", "voice_libri2", "music_trumpet", "music_brahms"]

# seconds, from the two runs described above
TIMING = {
    #               cpu     pre     post
    "voice_libri1": (2.195, 1.264, 1.108),
    "voice_libri2": (1.365, 1.263, 1.107),
    "music_trumpet": (2.412, 1.262, 1.108),
    "music_brahms": (1.921, 1.261, 1.109),
}
# note: the pre-fix run measured CPU separately (2.365/1.857/2.312/1.533 s); CPU timing varies a lot
# run to run on identical shapes, which is itself the point -- see the CPU spread line below.
CPU_PREFIX_RUN = {"voice_libri1": 2.365, "voice_libri2": 1.857, "music_trumpet": 2.312, "music_brahms": 1.533}


def read(clip, suffix):
    data, _ = sf.read(os.path.join(CLIPS_DIR, f"{clip}_{suffix}.wav"))
    return np.asarray(data, dtype=np.float64)


def psnr(ref, test):
    n = min(len(ref), len(test))
    mse = float(np.mean((ref[:n] - test[:n]) ** 2))
    return float("inf") if mse == 0 else 20 * math.log10(float(np.abs(ref[:n]).max())) - 10 * math.log10(mse)


def rel_rmse(ref, test):
    n = min(len(ref), len(test))
    return float(np.sqrt(np.mean((test[:n] - ref[:n]) ** 2)) / np.std(ref[:n]))


def log_spec(a, b, n_fft=1024, hop=256):
    w = np.hanning(n_fft)

    def S(x):
        return np.log(
            np.maximum(
                np.array([np.abs(np.fft.rfft(x[i : i + n_fft] * w)) for i in range(0, len(x) - n_fft, hop)]), 1e-5
            )
        )

    n = min(len(a), len(b))
    return float(np.mean(np.abs(S(a[:n]) - S(b[:n]))))


def mean(xs):
    return sum(xs) / len(xs)


def main():
    data = {}
    for c in CLIPS:
        src, cpu = read(c, "0_source"), read(c, "1_cpu")
        post, pre = read(c, "2_device"), read(c, "3_device_prefix")
        tc, tpre, tpost = TIMING[c]
        data[c] = dict(
            t_cpu=tc,
            t_pre=tpre,
            t_post=tpost,
            psnr_post_cpu=psnr(cpu, post),
            psnr_pre_cpu=psnr(cpu, pre),
            rr_post_cpu=rel_rmse(cpu, post),
            rr_pre_cpu=rel_rmse(cpu, pre),
            ls_post_cpu=log_spec(cpu, post),
            ls_pre_cpu=log_spec(cpu, pre),
            psnr_cpu_src=psnr(src, cpu),
            psnr_post_src=psnr(src, post),
            psnr_pre_src=psnr(src, pre),
            ls_cpu_src=log_spec(src, cpu),
            ls_post_src=log_spec(src, post),
            ls_pre_src=log_spec(src, pre),
        )

    W = 96
    print("MiniMax-H3 audio decode -- fp32, 5.17 s of 32 kHz per clip, batch 2")
    print("=" * W)

    print("\n1. TIMING (seconds per 5.17 s of audio; device = steady state, best of 3)")
    print("-" * W)
    print(f"{'clip':<15} {'CPU':>8} {'dev pre':>9} {'dev post':>9} {'pre->post':>10} {'CPU/post':>9}")
    for c in CLIPS:
        d = data[c]
        print(
            f"{c:<15} {d['t_cpu']:>8.3f} {d['t_pre']:>9.3f} {d['t_post']:>9.3f} "
            f"{d['t_pre'] / d['t_post']:>9.2f}x {d['t_cpu'] / d['t_post']:>8.2f}x"
        )
    mc, mpre, mpost = (mean([data[c][k] for c in CLIPS]) for k in ("t_cpu", "t_pre", "t_post"))
    print("-" * W)
    print(f"{'mean':<15} {mc:>8.3f} {mpre:>9.3f} {mpost:>9.3f} {mpre / mpost:>9.2f}x {mc / mpost:>8.2f}x")
    pres = [data[c]["t_pre"] for c in CLIPS]
    posts = [data[c]["t_post"] for c in CLIPS]
    cpus = [data[c]["t_cpu"] for c in CLIPS] + list(CPU_PREFIX_RUN.values())
    print(
        f"\n   spread across content:  device pre {min(pres):.3f}-{max(pres):.3f} s "
        f"| device post {min(posts):.3f}-{max(posts):.3f} s | CPU {min(cpus):.3f}-{max(cpus):.3f} s"
    )
    print("   device cost is set by op count and tensor shape, not by audio content; CPU is not.")

    print("\n2. QUALITY vs the CPU DECODE  (isolates hardware error -- how well we track the reference)")
    print("-" * W)
    print(
        f"{'clip':<15} {'PSNR pre':>9} {'PSNR post':>10} {'gain':>7} {'rrmse pre':>10} {'rrmse post':>11} "
        f"{'lsd pre':>8} {'lsd post':>9}"
    )
    for c in CLIPS:
        d = data[c]
        print(
            f"{c:<15} {d['psnr_pre_cpu']:>9.2f} {d['psnr_post_cpu']:>10.2f} "
            f"{d['psnr_post_cpu'] - d['psnr_pre_cpu']:>+7.2f} {d['rr_pre_cpu']:>10.3e} {d['rr_post_cpu']:>11.3e} "
            f"{d['ls_pre_cpu']:>8.3f} {d['ls_post_cpu']:>9.3f}"
        )
    print("-" * W)
    mpp, mpo = mean([data[c]["psnr_pre_cpu"] for c in CLIPS]), mean([data[c]["psnr_post_cpu"] for c in CLIPS])
    print(
        f"{'mean':<15} {mpp:>9.2f} {mpo:>10.2f} {mpo - mpp:>+7.2f} "
        f"{mean([data[c]['rr_pre_cpu'] for c in CLIPS]):>10.3e} "
        f"{mean([data[c]['rr_post_cpu'] for c in CLIPS]):>11.3e} "
        f"{mean([data[c]['ls_pre_cpu'] for c in CLIPS]):>8.3f} "
        f"{mean([data[c]['ls_post_cpu'] for c in CLIPS]):>9.3f}"
    )

    print("\n3. QUALITY vs the ORIGINAL SOURCE  (end to end -- includes the lossy autoencoder)")
    print("-" * W)
    print(f"{'clip':<15} {'CPU':>8} {'dev pre':>9} {'dev post':>9}    {'lsd CPU':>8} {'lsd pre':>8} {'lsd post':>9}")
    for c in CLIPS:
        d = data[c]
        print(
            f"{c:<15} {d['psnr_cpu_src']:>8.2f} {d['psnr_pre_src']:>9.2f} {d['psnr_post_src']:>9.2f}    "
            f"{d['ls_cpu_src']:>8.3f} {d['ls_pre_src']:>8.3f} {d['ls_post_src']:>9.3f}"
        )
    print("-" * W)
    print(
        f"{'mean':<15} {mean([data[c]['psnr_cpu_src'] for c in CLIPS]):>8.2f} "
        f"{mean([data[c]['psnr_pre_src'] for c in CLIPS]):>9.2f} "
        f"{mean([data[c]['psnr_post_src'] for c in CLIPS]):>9.2f}    "
        f"{mean([data[c]['ls_cpu_src'] for c in CLIPS]):>8.3f} "
        f"{mean([data[c]['ls_pre_src'] for c in CLIPS]):>8.3f} "
        f"{mean([data[c]['ls_post_src'] for c in CLIPS]):>9.3f}"
    )

    print("\n" + "=" * W)
    print("READING IT")
    print("-" * W)
    print(
        f"  speed      pre-fix {mpre:.3f} s -> post-fix {mpost:.3f} s  = {mpre / mpost:.2f}x; "
        f"CPU {mc:.3f} s = {mc / mpost:.2f}x slower than post-fix"
    )
    print(f"  vs CPU     {mpp:.2f} -> {mpo:.2f} dB = {mpo - mpp:+.2f} dB. Real, and uniform across speech and music.")
    print("  vs source  the autoencoder alone costs ~28.6 dB, ~20 dB more error than the hardware adds,")
    print("             so by PSNR pre and post are indistinguishable end to end (gap is noise).")
    print("  caveat     log-spectral distance disagrees: vs source it goes 1.17 (CPU) -> 2.00 (device),")
    print("             ~1.7x, so on that metric hardware error is comparable to codec error. PSNR is")
    print("             energy-weighted and hides broadband low-level error; lsd weights quiet bins")
    print("             equally. Masking sits between -- listening decides. WAVs are in the clips folder.")


if __name__ == "__main__":
    main()
