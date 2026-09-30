#!/usr/bin/env python3
"""First-pass check: PCC and PSNR of a candidate against a reference.

  compare.py REF CAND [--seeds 0,1] [--json out.json] [--strict]

REF / CAND are both baseline run dirs (as written by test_fasth3_baseline_minimax_h3.py: seedN.mp4,
seedN_latents.pt, seedN_frames_u8_every16.npy, seedN.wav, seedN_timings.json) or both single files of
the same kind (.mp4, .npy uint8 frames, .pt latents, .wav). Pixels come from the lossless every-16th
frame npy when both sides have it, otherwise from the mp4 (lossy, all frames); --mp4 adds the mp4
pass even when the npy exists.
"""

import os

# Threaded BLAS on 1-D dot products spends its time spinning; one thread is ~1.5x faster end to end.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import wave  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import iter_mp4, load_timings, seed_stems, video_path  # noqa: E402


class Stats:
    """Streams sums in float64 so full 1080p clips never sit in memory as floats."""

    def __init__(self):
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = self.sse = 0.0
        self.frame_psnr = []

    def add(self, x: np.ndarray, y: np.ndarray, per_frame: bool = False):
        x = x.astype(np.float64).ravel()
        y = y.astype(np.float64).ravel()
        diff = x - y
        sse = float(diff @ diff)
        self.n += x.size
        self.sx += x.sum()
        self.sy += y.sum()
        self.sxx += x @ x
        self.syy += y @ y
        self.sxy += x @ y
        self.sse += sse
        if per_frame:
            self.frame_psnr.append(psnr(sse / x.size))

    def pcc(self) -> float:
        n = self.n
        cov = self.sxy - self.sx * self.sy / n
        var = (self.sxx - self.sx**2 / n) * (self.syy - self.sy**2 / n)
        if var <= 0:
            return 1.0 if self.sse == 0 else 0.0
        return float(cov / math.sqrt(var))

    def summary(self) -> dict:
        out = dict(pcc=self.pcc(), psnr=psnr(self.sse / self.n))
        if self.frame_psnr:
            worst = int(np.argmin(self.frame_psnr))
            out.update(frames=len(self.frame_psnr), min_frame_psnr=self.frame_psnr[worst], worst_frame=worst)
        return out


def psnr(mse: float, peak: float = 255.0) -> float:
    return float("inf") if mse == 0 else 10.0 * math.log10(peak * peak / mse)


def compare_frames(ref: np.ndarray, cand: np.ndarray, stride: int = 1) -> dict:
    count = min(len(ref), len(cand))
    stats = Stats()
    for i in range(count):
        stats.add(ref[i], cand[i], per_frame=True)
    out = stats.summary()
    out["worst_frame"] = out["worst_frame"] * stride
    if len(ref) != len(cand):
        out["frame_count_mismatch"] = [len(ref), len(cand)]
    return out


def compare_mp4(ref: Path, cand: Path) -> dict:
    stats = Stats()
    for (_, a), (_, b) in zip(iter_mp4(ref), iter_mp4(cand)):
        if a.shape != b.shape:
            return dict(error=f"shape mismatch {a.shape} vs {b.shape}")
        stats.add(a, b, per_frame=True)
    out = stats.summary()
    ref_count, cand_count = _count(ref), _count(cand)
    if ref_count != cand_count:
        out["frame_count_mismatch"] = [ref_count, cand_count]
    return out


def _count(path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def compare_latents(ref: Path, cand: Path) -> dict:
    import torch

    a = torch.load(ref, map_location="cpu", weights_only=False)
    b = torch.load(cand, map_location="cpu", weights_only=False)
    if not isinstance(a, dict):
        a, b = {"latents": a}, {"latents": b}
    out = {}
    for key in a:
        if key not in b or a[key] is None or b[key] is None:
            continue
        x, y = a[key].float().numpy(), b[key].float().numpy()
        if x.shape != y.shape:
            out[key] = dict(error=f"shape mismatch {list(x.shape)} vs {list(y.shape)}")
            continue
        stats = Stats()
        stats.add(x, y)
        peak = float(np.abs(x).max()) or 1.0
        out[key] = dict(
            pcc=stats.pcc(),
            max_abs_diff=float(np.abs(x - y).max()),
            psnr_vs_ref_peak=psnr(stats.sse / stats.n, peak),
            shape=list(x.shape),
        )
    return out


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return data.astype(np.float64) / 32767.0


def compare_wav(ref: Path, cand: Path) -> dict:
    a, b = read_wav(ref), read_wav(cand)
    count = min(len(a), len(b))
    stats = Stats()
    stats.add(a[:count], b[:count])
    signal = float(a[:count] @ a[:count])
    snr = float("inf") if stats.sse == 0 else 10.0 * math.log10(max(signal, 1e-12) / stats.sse)
    return dict(pcc=stats.pcc(), snr_db=snr)


def compare_files(ref: Path, cand: Path) -> dict:
    suffix = ref.suffix.lower()
    if suffix == ".mp4":
        return {"pixels_mp4": compare_mp4(ref, cand)}
    if suffix == ".npy":
        return {"pixels_npy": compare_frames(np.load(ref), np.load(cand))}
    if suffix == ".pt":
        return {"latents": compare_latents(ref, cand)}
    if suffix == ".wav":
        return {"audio": compare_wav(ref, cand)}
    raise SystemExit(f"unsupported file type {suffix}")


def compare_seed(ref_dir: Path, cand_dir: Path, stem: str, force_mp4: bool) -> dict:
    out = {}
    lat = f"{stem}_latents.pt"
    if (ref_dir / lat).is_file() and (cand_dir / lat).is_file():
        out["latents"] = compare_latents(ref_dir / lat, cand_dir / lat)
    npy = f"{stem}_frames_u8_every16.npy"
    have_npy = (ref_dir / npy).is_file() and (cand_dir / npy).is_file()
    if have_npy:
        out["pixels_npy"] = compare_frames(np.load(ref_dir / npy), np.load(cand_dir / npy), stride=16)
    ref_mp4, cand_mp4 = video_path(ref_dir, stem), video_path(cand_dir, stem)
    if (force_mp4 or not have_npy) and ref_mp4 and cand_mp4:
        out["pixels_mp4"] = compare_mp4(ref_mp4, cand_mp4)
    wav = f"{stem}.wav"
    if (ref_dir / wav).is_file() and (cand_dir / wav).is_file():
        out["audio"] = compare_wav(ref_dir / wav, cand_dir / wav)
    ref_t, cand_t = load_timings(ref_dir, stem), load_timings(cand_dir, stem)
    if ref_t and cand_t:
        out["timing"] = dict(
            ref_wall_s=ref_t.get("call_wall_s"),
            cand_wall_s=cand_t.get("call_wall_s"),
            speedup=(ref_t["call_wall_s"] / cand_t["call_wall_s"]) if cand_t.get("call_wall_s") else None,
        )
    return out


def verdict(result: dict, args) -> list[str]:
    fails = []
    for key in ("pixels_npy", "pixels_mp4"):
        pix = result.get(key)
        if not pix or "error" in pix:
            continue
        if pix["pcc"] < args.min_pcc:
            fails.append(f"{key}.pcc {pix['pcc']:.4f} < {args.min_pcc}")
        if pix["psnr"] < args.min_psnr:
            fails.append(f"{key}.psnr {pix['psnr']:.2f} < {args.min_psnr}")
    for key, lat in result.get("latents", {}).items():
        if "pcc" in lat and lat["pcc"] < args.min_latent_pcc:
            fails.append(f"latents.{key}.pcc {lat['pcc']:.4f} < {args.min_latent_pcc}")
    return fails


def fmt(value) -> str:
    if isinstance(value, float):
        return "inf" if math.isinf(value) else f"{value:.4f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ref")
    parser.add_argument("cand")
    parser.add_argument("--seeds", help="comma list; default = seeds present on both sides")
    parser.add_argument("--mp4", action="store_true", help="also compare every mp4 frame (slower, lossy)")
    parser.add_argument("--min-pcc", type=float, default=0.99)
    parser.add_argument("--min-psnr", type=float, default=30.0)
    parser.add_argument("--min-latent-pcc", type=float, default=0.99)
    parser.add_argument("--json", help="write full results here")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any threshold fails")
    args = parser.parse_args()

    ref, cand = Path(args.ref), Path(args.cand)
    t0 = time.time()
    results = {}
    if ref.is_dir():
        stems = sorted(set(seed_stems(ref)) & set(seed_stems(cand)), key=lambda s: int(s[4:]))
        if args.seeds:
            stems = [f"seed{s}" for s in args.seeds.split(",") if f"seed{s}" in stems]
        if not stems:
            raise SystemExit(f"no common seedN artifacts in {ref} and {cand}")
        for stem in stems:
            results[stem] = compare_seed(ref, cand, stem, args.mp4)
    else:
        results[cand.name] = compare_files(ref, cand)

    any_fail = False
    for name, result in results.items():
        fails = verdict(result, args)
        result["fails"] = fails
        any_fail |= bool(fails)
        print(f"== {name}  {'FAIL' if fails else 'PASS'}")
        for key, value in result.items():
            if key == "fails":
                continue
            if key == "latents":
                for lk, lv in value.items():
                    print(f"  latents.{lk:<11} " + "  ".join(f"{k}={fmt(v)}" for k, v in lv.items() if k != "shape"))
            else:
                print(f"  {key:<19} " + "  ".join(f"{k}={fmt(v)}" for k, v in value.items()))
        for fail in fails:
            print(f"  FAIL {fail}")
    elapsed = time.time() - t0
    print(f"compare wall {elapsed:.1f}s")
    if args.json:
        Path(args.json).write_text(
            json.dumps(dict(ref=str(ref), cand=str(cand), wall_s=elapsed, results=results), indent=2)
        )
    sys.exit(1 if (args.strict and any_fail) else 0)


if __name__ == "__main__":
    main()
