# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LTX quality harness: PCC/PSNR against a reference, a partial VBench subset, and still frames.

    # one clip; --ref is optional (without it only VBench and stills run)
    python -m models.tt_dit.tests.models.ltx.tools.ltx_eval video --cand new.mp4 --ref ref.mp4 --out q/
    # five seeds: pairs <cand-dir>/X.mp4 with <ref-dir>/X.mp4, scores clips in parallel, reports means
    python -m models.tt_dit.tests.models.ltx.tools.ltx_eval batch --cand-dir new/ --ref-dir ref/ --out q/
    # VAE decoder (or any) tensors saved with torch.save / np.save
    python -m models.tt_dit.tests.models.ltx.tools.ltx_eval tensor --cand new.pt --ref ref.pt

Common flags: --vbench none|<dim,...> (default: subject/background consistency, motion smoothness,
imaging quality), --vbench-ref to also score the reference, --prompt, --pcc-min, --psnr-min.
Writes <out>/report.json (batch: one per clip plus summary.json) and PNG stills
(<name>_f<idx>.png, and <name>_cmp_f<idx>.png = reference | candidate | 4x abs diff).
Prints one verdict line per run: QUALITY OK|FAIL ... . A FAIL is a flag to look at the
stills and seeds, not proof the video is worse.

A clip may carry a sidecar <stem>.json (written by test_pipeline_ltx_distilled run()) with the
prompt and seed it was made from. When candidate and reference both have one and their prompt
or seed differ, nothing is scored: QUALITY MISMATCH prompt|seed ... and exit 3, unless
--allow-mismatch. Without a sidecar on either side the clips are scored as before, with a warning.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DEFAULT_VBENCH = ["subject_consistency", "background_consistency", "motion_smoothness", "imaging_quality"]
# Dimensions that VBench scores on optical flow / frame interpolation; they run on a
# width-reduced lossless copy, same policy as the LTX CI gate (models/tt_dit/utils/vbench.md).
TEMPORAL_VBENCH = {"motion_smoothness", "dynamic_degree"}
PSNR_IDENTICAL = 99.0
# Scores between clips of different prompts or seeds are meaningless (PCC 0.27 for a correct clip).
PAIRING_KEYS = ("prompt", "seed")
EXIT_MISMATCH = 3


def sidecar_path(video):
    return Path(video).with_suffix(".json")


def write_sidecar(video, **meta):
    sidecar_path(video).write_text(json.dumps(meta, indent=2) + "\n")


def read_sidecar(video):
    path = sidecar_path(video)
    return json.loads(path.read_text()) if path.exists() else None


def pairing_mismatches(cand, ref):
    """Mismatch lines for a cand/ref pair, or None when either clip has no sidecar."""
    cand_meta, ref_meta = read_sidecar(cand), read_sidecar(ref)
    if cand_meta is None or ref_meta is None:
        return None
    return [
        f"QUALITY MISMATCH {key} clip={Path(cand).stem} cand={cand_meta.get(key)!r} ref={ref_meta.get(key)!r}"
        for key in PAIRING_KEYS
        if cand_meta.get(key) != ref_meta.get(key)
    ]


def check_pairing(pairs, allow_mismatch):
    """Print mismatches and a missing-sidecar warning; True when scoring may go ahead."""
    lines, unchecked = [], []
    for cand, ref in pairs:
        mismatches = pairing_mismatches(cand, ref)
        if mismatches is None:
            unchecked.append(Path(cand).stem)
        else:
            lines += mismatches
    for line in lines:
        print(line + (" (scored anyway: --allow-mismatch)" if allow_mismatch else ""), flush=True)
    if unchecked:
        print(f"WARNING: no sidecar on cand or ref, prompt/seed not checked: {','.join(unchecked)}", file=sys.stderr)
    return allow_mismatch or not lines


def iter_frames(path):
    """Yield RGB uint8 frames (H, W, 3) one at a time so two 1080p clips never sit in memory."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # A few threads decode 1080p faster than we score it; "AUTO" spawns one per core on a shared host.
        stream.thread_type, stream.thread_count = "AUTO", 4
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")


def video_meta(path):
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {
            "width": stream.width,
            "height": stream.height,
            "fps": float(stream.average_rate) if stream.average_rate else None,
            "frames": stream.frames,
        }


def psnr_from_mse(mse, data_range):
    return PSNR_IDENTICAL if mse == 0 else 20 * math.log10(data_range) - 10 * math.log10(mse)


def psnr(ref, cand, data_range):
    return psnr_from_mse(float(np.mean((ref.astype(np.float64) - cand.astype(np.float64)) ** 2)), data_range)


class Moments:
    """Running sums for a global Pearson correlation over arbitrarily many samples."""

    def __init__(self):
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0

    def add(self, x, y):
        # Integer pixels accumulate exactly (Python ints); float @ would also fan out BLAS threads.
        dtype = np.int64 if np.issubdtype(x.dtype, np.integer) and np.issubdtype(y.dtype, np.integer) else np.float64
        x, y = x.astype(dtype).ravel(), y.astype(dtype).ravel()
        self.n += x.size
        for name, value in (("sx", x.sum()), ("sy", y.sum()), ("sxx", np.dot(x, x)), ("syy", np.dot(y, y))):
            setattr(self, name, getattr(self, name) + value.item())
        self.sxy += np.dot(x, y).item()

    def pcc(self):
        n = self.n
        cov = n * self.sxy - self.sx * self.sy
        vx = n * self.sxx - self.sx**2
        vy = n * self.syy - self.sy**2
        if vx <= 0 or vy <= 0:
            return 1.0 if vx == vy and self.sx == self.sy else float("nan")
        return float(cov / math.sqrt(vx * vy))


def pcc(ref, cand):
    if np.issubdtype(ref.dtype, np.integer) and np.issubdtype(cand.dtype, np.integer):
        moments = Moments()
        moments.add(ref, cand)
        return moments.pcc()
    # Centre first: raw moments of float tensors with a large mean lose the variance to cancellation.
    x = np.asarray(ref, dtype=np.float64).ravel()
    y = np.asarray(cand, dtype=np.float64).ravel()
    x, y = x - x.mean(), y - y.mean()
    den = math.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / den) if den > 0 else (1.0 if np.array_equal(x, y) else float("nan"))


def frame_indices(count, spec):
    if spec:
        picks = [int(item) for item in spec.split(",")]
        return sorted({index if index >= 0 else count + index for index in picks})
    return sorted({0, count // 2, count - 1})


def save_png(path, rgb):
    from PIL import Image

    Image.fromarray(rgb).save(path)


def compare_videos(ref_path, cand_path, out_dir, *, name, stills):
    """Stream both clips in lockstep; return per-frame and global PCC/PSNR and write stills."""
    ref_meta, cand_meta = video_meta(ref_path), video_meta(cand_path)
    keep = set(frame_indices(ref_meta["frames"] or cand_meta["frames"], stills))
    moments = Moments()
    frame_psnr, frame_pcc = [], []
    sq_err, count = 0, 0
    for index, (ref, cand) in enumerate(zip(iter_frames(ref_path), iter_frames(cand_path))):
        if ref.shape != cand.shape:
            raise ValueError(f"Frame shape differs: ref {ref.shape} vs cand {cand.shape}")
        diff = (ref.astype(np.int64) - cand.astype(np.int64)).ravel()
        frame_sq_err = np.dot(diff, diff).item()
        frame_psnr.append(psnr_from_mse(frame_sq_err / diff.size, 255.0))
        frame_pcc.append(pcc(ref, cand))
        moments.add(ref, cand)
        sq_err += frame_sq_err
        count += diff.size
        if index in keep:
            heat = np.clip(np.abs(ref.astype(np.int16) - cand.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
            save_png(out_dir / f"{name}_cmp_f{index:03d}.png", np.concatenate([ref, cand, heat], axis=1))
    decoded = len(frame_psnr)
    # zip() stops at the shorter clip; a frame-count mismatch must not pass as parity.
    if decoded != (ref_meta["frames"] or decoded) or decoded != (cand_meta["frames"] or decoded):
        raise ValueError(f"Frame count differs: ref {ref_meta['frames']} vs cand {cand_meta['frames']}")
    return {
        "frames": decoded,
        "psnr": psnr_from_mse(sq_err / count, 255.0),
        "psnr_min": min(frame_psnr),
        "psnr_min_frame": int(np.argmin(frame_psnr)),
        "psnr_mean": float(np.mean(frame_psnr)),
        "pcc": moments.pcc(),
        "pcc_min": float(np.nanmin(frame_pcc)),
        "pcc_min_frame": int(np.nanargmin(frame_pcc)),
        "psnr_per_frame": [round(value, 3) for value in frame_psnr],
    }


def export_stills(path, out_dir, *, name, stills):
    keep = set(frame_indices(video_meta(path)["frames"], stills))
    written = []
    for index, frame in enumerate(iter_frames(path)):
        if index in keep:
            target = out_dir / f"{name}_f{index:03d}.png"
            save_png(target, frame)
            written.append(str(target))
    return written


def vbench_scores(path, dimensions, *, prompt, temporal_width):
    import torch

    repo = Path(__file__).resolve().parents[6]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from models.tt_dit.utils.vbench import score_vbench
    from models.tt_dit.utils.vbench_cpu import dynamic_degree, temporal_video

    torch.set_num_threads(int(os.environ.get("LTX_EVAL_THREADS", "8")))
    appearance = [dim for dim in dimensions if dim not in TEMPORAL_VBENCH]
    temporal = [dim for dim in dimensions if dim in TEMPORAL_VBENCH]
    scores = score_vbench(str(path), prompt=prompt, dimensions=appearance) if appearance else {}
    if temporal:
        with temporal_video(path, temporal_width) as reduced:
            if "motion_smoothness" in temporal:
                scores.update(score_vbench(str(reduced), prompt=prompt, dimensions=["motion_smoothness"]))
            if "dynamic_degree" in temporal:
                scores["dynamic_degree"] = dynamic_degree(reduced)
    return scores


def evaluate_clip(cand, ref, out_dir, *, name, opts):
    """Full report for one clip. Module-level so batch mode can run it in spawned workers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"name": name, "cand": str(cand), "ref": str(ref) if ref else None, "cand_meta": video_meta(cand)}
    started = time.perf_counter()
    report["stills"] = export_stills(cand, out_dir, name=name, stills=opts["stills"])
    if ref:
        report["ref_meta"] = video_meta(ref)
        report["parity"] = compare_videos(ref, cand, out_dir, name=name, stills=opts["stills"])
    report["parity_s"] = round(time.perf_counter() - started, 1)
    if opts["vbench"]:
        started = time.perf_counter()
        kwargs = dict(prompt=opts["prompt"], temporal_width=opts["temporal_width"])
        report["vbench"] = vbench_scores(cand, opts["vbench"], **kwargs)
        if ref and opts["vbench_ref"]:
            report["vbench_ref"] = vbench_scores(ref, opts["vbench"], **kwargs)
        report["vbench_s"] = round(time.perf_counter() - started, 1)
    report["verdict"] = verdict(report, opts)
    (out_dir / f"{name}_report.json").write_text(json.dumps(report, indent=2))
    return report


def verdict(report, opts):
    failures = []
    parity = report.get("parity")
    if parity:
        if parity["pcc"] < opts["pcc_min"]:
            failures.append(f"pcc {parity['pcc']:.5f} < {opts['pcc_min']}")
        if parity["psnr_min"] < opts["psnr_min"]:
            failures.append(f"psnr_min {parity['psnr_min']:.2f} < {opts['psnr_min']}")
    vb, vb_ref = report.get("vbench", {}), report.get("vbench_ref", {})
    for dim, score in vb.items():
        if not math.isfinite(score):
            failures.append(f"{dim} not finite")
        elif dim in vb_ref and score < vb_ref[dim] - opts["vbench_tol"]:
            failures.append(f"{dim} {score:.4f} < ref {vb_ref[dim]:.4f} - {opts['vbench_tol']}")
    return {"ok": not failures, "failures": failures}


def verdict_line(report):
    parts = ["QUALITY OK" if report["verdict"]["ok"] else "QUALITY FAIL", f"clip={report['name']}"]
    parity = report.get("parity")
    if parity:
        parts += [
            f"pcc={parity['pcc']:.5f}",
            f"pcc_min={parity['pcc_min']:.5f}",
            f"psnr={parity['psnr']:.2f}",
            f"psnr_min={parity['psnr_min']:.2f}@{parity['psnr_min_frame']}",
            f"frames={parity['frames']}",
        ]
    parts += [f"{dim}={score:.4f}" for dim, score in report.get("vbench", {}).items()]
    parts += [f"fail=[{'; '.join(report['verdict']['failures'])}]"] if report["verdict"]["failures"] else []
    return " ".join(parts)


def summarize(reports, opts):
    """Means over seeds; the batch passes only if every seed passes."""
    summary = {"clips": [r["name"] for r in reports], "ok": all(r["verdict"]["ok"] for r in reports)}
    parities = [r["parity"] for r in reports if "parity" in r]
    if parities:
        for key in ("pcc", "pcc_min", "psnr", "psnr_min", "psnr_mean"):
            summary[f"{key}_mean"] = float(np.mean([p[key] for p in parities]))
        summary["pcc_worst"] = min(p["pcc_min"] for p in parities)
        summary["psnr_worst"] = min(p["psnr_min"] for p in parities)
    for field in ("vbench", "vbench_ref"):
        rows = [r[field] for r in reports if field in r]
        if rows:
            summary[f"{field}_mean"] = {dim: float(np.mean([row[dim] for row in rows])) for dim in rows[0]}
    if len(reports) != opts["seeds"]:
        summary["warning"] = f"{len(reports)} clips, expected {opts['seeds']}"
    return summary


def load_tensor(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(f"{path}: expected one tensor, found keys {sorted(value)}")
        value = next(iter(value.values()))
    return value.float().numpy()


def compare_tensors(ref, cand):
    ref, cand = np.asarray(ref, dtype=np.float64), np.asarray(cand, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(f"Tensor shape differs: ref {ref.shape} vs cand {cand.shape}")
    span = float(ref.max() - ref.min()) or 1.0
    diff = np.abs(ref - cand)
    return {
        "shape": list(ref.shape),
        "pcc": pcc(ref, cand),
        "psnr": psnr(ref, cand, span),
        "data_range": span,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "finite": bool(np.isfinite(cand).all()),
    }


def parse_opts(args):
    dims = [] if args.vbench == "none" else [dim for dim in args.vbench.split(",") if dim]
    return {
        "vbench": dims,
        "vbench_ref": args.vbench_ref,
        "vbench_tol": args.vbench_tol,
        "prompt": args.prompt,
        "temporal_width": args.temporal_width,
        "stills": args.stills,
        "pcc_min": args.pcc_min,
        "psnr_min": args.psnr_min,
        "seeds": args.seeds,
    }


def run_batch(args, opts):
    cand_dir = Path(args.cand_dir)
    clips = sorted(cand_dir.glob("*.mp4"))
    if not clips:
        raise ValueError(f"No mp4 in {cand_dir}")
    pairs = []
    for clip in clips:
        ref = Path(args.ref_dir) / clip.name if args.ref_dir else None
        if ref is not None and not ref.exists():
            raise ValueError(f"No reference for {clip.name} in {args.ref_dir}")
        pairs.append((clip, ref))
    if args.ref_dir and not check_pairing(pairs, args.allow_mismatch):
        return EXIT_MISMATCH
    out = Path(args.out)
    workers = max(1, min(args.jobs, len(pairs)))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(evaluate_clip, cand, ref, out, name=cand.stem, opts=opts) for cand, ref in pairs]
        reports = [future.result() for future in futures]
    for report in reports:
        print(verdict_line(report), flush=True)
    summary = summarize(reports, opts)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(("BATCH OK " if summary["ok"] else "BATCH FAIL ") + json.dumps(summary), flush=True)
    return 0 if summary["ok"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    video = sub.add_parser("video")
    video.add_argument("--cand", required=True)
    video.add_argument("--ref")
    batch = sub.add_parser("batch")
    batch.add_argument("--cand-dir", required=True)
    batch.add_argument("--ref-dir")
    batch.add_argument("--jobs", type=int, default=5, help="parallel clips (each uses LTX_EVAL_THREADS)")
    batch.add_argument("--seeds", type=int, default=5, help="expected clip count; a mismatch is warned")
    for mode in (video, batch):
        mode.add_argument("--out", required=True)
        mode.add_argument("--vbench", default=",".join(DEFAULT_VBENCH), help="'none' or comma-separated dims")
        mode.add_argument("--vbench-ref", action="store_true", help="also score the reference and compare")
        mode.add_argument("--vbench-tol", type=float, default=0.01, help="allowed drop below the reference score")
        mode.add_argument("--prompt")
        mode.add_argument("--temporal-width", type=int, default=960, help="width for motion_smoothness input")
        mode.add_argument("--stills", help="comma-separated frame indices (negative from end); default first,mid,last")
        mode.add_argument("--pcc-min", type=float, default=0.99)
        mode.add_argument("--psnr-min", type=float, default=30.0, help="per-frame minimum, dB")
        mode.add_argument("--allow-mismatch", action="store_true", help="score even if sidecar prompt/seed differ")
    video.set_defaults(seeds=1)
    tensor = sub.add_parser("tensor")
    tensor.add_argument("--cand", required=True)
    tensor.add_argument("--ref", required=True)
    tensor.add_argument("--pcc-min", type=float, default=0.999)
    args = parser.parse_args(argv)

    if args.mode == "tensor":
        result = compare_tensors(load_tensor(args.ref), load_tensor(args.cand))
        ok = result["finite"] and result["pcc"] >= args.pcc_min
        print(("QUALITY OK " if ok else "QUALITY FAIL ") + json.dumps(result), flush=True)
        return 0 if ok else 1
    opts = parse_opts(args)
    if args.mode == "batch":
        return run_batch(args, opts)
    if args.ref and not check_pairing([(args.cand, args.ref)], args.allow_mismatch):
        return EXIT_MISMATCH
    report = evaluate_clip(args.cand, args.ref, args.out, name=Path(args.cand).stem, opts=opts)
    print(verdict_line(report), flush=True)
    return 0 if report["verdict"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
