"""Score the WAVs produced by gen_audio.py: PSNR against the CPU reference, and against each other."""

import math
import os

import torch

OUT_DIR = "/data/rshirvani/audio_compare"
SR = 32000


def psnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    mse = torch.mean((ref.float() - test.float()) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(ref.abs().max().item()) - 10.0 * math.log10(mse)


def rel_rmse(ref: torch.Tensor, test: torch.Tensor) -> float:
    return float((test.double() - ref.double()).pow(2).mean().sqrt() / ref.double().std())


def log_spec_distance(a: torch.Tensor, b: torch.Tensor, n_fft: int = 1024, hop: int = 256) -> float:
    window = torch.hann_window(n_fft)
    out = []
    for sig in (a, b):
        flat = sig.reshape(-1, sig.shape[-1]).float()
        out.append(
            torch.log(
                torch.stft(flat, n_fft=n_fft, hop_length=hop, window=window, return_complex=True).abs().clamp_min(1e-5)
            )
        )
    return (out[0] - out[1]).abs().mean().item()


def main():
    ref_blob = torch.load(os.path.join(OUT_DIR, "reference.pt"))
    ref = ref_blob["wav"]
    print(f"CPU reference decode: {ref_blob['seconds']:.3f} s   ({ref.shape[-1] / SR:.2f} s of audio)\n")

    tags = [t for t in ("prefix", "postfix") if os.path.exists(os.path.join(OUT_DIR, f"{t}.pt"))]
    blobs = {t: torch.load(os.path.join(OUT_DIR, f"{t}.pt")) for t in tags}

    print("=== each device decode vs the CPU reference ===")
    print(f"{'run':<10} {'seconds':>9} {'vs CPU':>9} {'PSNR dB':>9} {'rel_rmse':>11} {'log-spec':>9}")
    print("-" * 62)
    for t in tags:
        b = blobs[t]
        print(
            f"{t:<10} {b['seconds']:>9.3f} {ref_blob['seconds'] / b['seconds']:>8.2f}x "
            f"{psnr(ref, b['wav']):>9.2f} {rel_rmse(ref, b['wav']):>11.3e} {log_spec_distance(ref, b['wav']):>9.4f}"
        )

    if len(tags) == 2:
        a, c = blobs["prefix"]["wav"], blobs["postfix"]["wav"]
        print("\n=== the two device decodes against each other ===")
        print(f"  PSNR(prefix, postfix)  {psnr(a, c):>9.2f} dB")
        print(f"  max abs difference     {float((a - c).abs().max()):>9.3e}")
        print(f"  identical              {torch.equal(a, c)}")
        print("\n=== speed ===")
        sp, so = blobs["prefix"]["seconds"], blobs["postfix"]["seconds"]
        print(f"  prefix  {sp:.3f} s")
        print(f"  postfix {so:.3f} s")
        print(f"  speedup {sp / so:.2f}x")
        dp, do = psnr(ref, a), psnr(ref, c)
        print(f"\n  PSNR change vs CPU reference: {dp:.2f} -> {do:.2f} dB  ({do - dp:+.2f} dB)")

    print(f"\nWAVs in {OUT_DIR}: source.wav, reference.wav, " + ", ".join(f"{t}.wav" for t in tags))


if __name__ == "__main__":
    main()
