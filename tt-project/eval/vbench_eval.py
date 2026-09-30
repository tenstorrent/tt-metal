#!/usr/bin/env python3
"""VBench on CPU: fast partial set by default, --full for every dimension VBench scores on custom videos.

  vbench_eval.py VIDEO_OR_RUN_DIR... [--full | --dims a,b] [--short-side 512] [--out DIR] [--ref-json J]

Partial = subject_consistency, motion_smoothness, aesthetic_quality, imaging_quality.
Full adds background_consistency, temporal_flickering, dynamic_degree, overall_consistency (needs the
prompt; read from seedN_timings.json or --prompt). Videos are re-encoded to --short-side first
(0 = native); VBench's CLIP/DINO/MUSIQ inputs are 224-512 px anyway, only AMT/RAFT see more pixels.
--frame-stride K (default 4) scores every Kth frame for the two per-frame-mean dims.
Scores at different --short-side / --frame-stride values are not comparable; compare at one setting.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import expand_videos, ffmpeg, load_timings, mp4_info  # noqa: E402

PARTIAL = ["subject_consistency", "motion_smoothness", "aesthetic_quality", "imaging_quality"]
FULL = PARTIAL + ["background_consistency", "temporal_flickering", "dynamic_degree", "overall_consistency"]
NEEDS_PROMPT = {"overall_consistency", "temporal_style"}


def stage(video: Path, staging: Path, short_side: int) -> Path:
    """Stable per-source name, so seedN from two run dirs never collide and reruns reuse the encode."""
    name = f"{video.parent.name}__{video.stem}.mp4"
    out = staging / name
    if out.is_file() and out.stat().st_mtime >= video.stat().st_mtime:
        return out
    if short_side <= 0:
        if out.exists() or out.is_symlink():
            out.unlink()
        out.symlink_to(video.resolve())
        return out
    info = mp4_info(video)
    scale = f"-2:{short_side}" if info["height"] <= info["width"] else f"{short_side}:-2"
    subprocess.run(
        [
            ffmpeg(),
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-an",
            "-vf",
            f"scale={scale}:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
    )
    return out


def subsample(stride: int):
    """Aesthetic and imaging quality are per-frame means, so a strided subset estimates the same value.
    Subject consistency and motion smoothness score adjacent frames and must see every frame."""
    import vbench.aesthetic_quality as aesthetic
    import vbench.imaging_quality as imaging

    for module in (aesthetic, imaging):
        load = module.load_video
        module.load_video = lambda *a, _load=load, **k: _load(*a, **k)[::stride]


def prompt_for(video: Path, override: str | None) -> str:
    if override:
        return override
    stem = video.stem.removesuffix("_silent")
    timings = load_timings(video.parent, stem)
    return (timings or {}).get("prompt", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true")
    group.add_argument("--dims", help="comma list of VBench dimensions")
    parser.add_argument("--short-side", type=int, default=512)
    parser.add_argument("--prompt", help="prompt for prompt-based dims when no timings json is present")
    parser.add_argument("--out", default="/tmp/fasth3_vbench", help="staging + raw VBench output")
    parser.add_argument("--json", help="summary json (default OUT/summary.json)")
    parser.add_argument("--ref-json", help="earlier summary.json to print per-dimension deltas against")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=4,
        help="score every Kth frame for aesthetic/imaging quality (1 = stock VBench)",
    )
    parser.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    from vbench import VBench

    if args.frame_stride > 1:
        subsample(args.frame_stride)

    dims = args.dims.split(",") if args.dims else (FULL if args.full else PARTIAL)
    videos = expand_videos(args.inputs)
    if not videos:
        raise SystemExit("no videos found")
    out = Path(args.out)
    staging = out / f"staged_{args.short_side}"
    staging.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    staged = {}
    for video in videos:
        staged[stage(video, staging, args.short_side)] = video
    stage_s = time.time() - t0
    print(f"staged {len(videos)} videos at short side {args.short_side} in {stage_s:.1f}s")

    # A private dir holding only this run's files, since custom_input mode scores everything in videos_path.
    run_dir = out / f"run_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for path in staged:
        (run_dir / path.name).symlink_to(path.resolve())
    prompts = {path.name: prompt_for(src, args.prompt) for path, src in staged.items()}
    by_name = {path.name: src for path, src in staged.items()}

    info_json = str(Path(__import__("vbench").__file__).parent / "VBench_full_info.json")
    bench = VBench("cpu", info_json, str(out / "raw"))
    summary = dict(
        short_side=args.short_side,
        frame_stride=args.frame_stride,
        videos=[str(v) for v in videos],
        dims={},
        per_video={},
        wall_s={},
    )
    for dim in dims:
        prompt_list = prompts if dim in NEEDS_PROMPT else []
        if dim in NEEDS_PROMPT and not all(prompts.values()):
            print(f"skip {dim}: no prompt for some videos (pass --prompt)")
            continue
        t1 = time.time()
        bench.evaluate(
            videos_path=str(run_dir),
            name=f"{run_dir.name}_{dim}",
            prompt_list=prompt_list,
            dimension_list=[dim],
            mode="custom_input",
        )
        summary["wall_s"][dim] = time.time() - t1
        raw = json.loads((out / "raw" / f"{run_dir.name}_{dim}_eval_results.json").read_text())[dim]
        summary["dims"][dim] = raw[0]
        for entry in raw[1]:
            src = by_name.get(Path(entry["video_path"]).name, entry["video_path"])
            summary["per_video"].setdefault(str(src), {})[dim] = entry["video_results"]
        print(f"{dim:<24} {raw[0]:.4f}   ({summary['wall_s'][dim]:.1f}s)")
    summary["wall_s"]["stage"] = stage_s
    summary["wall_s"]["total"] = time.time() - t0
    for path in run_dir.iterdir():
        path.unlink()
    run_dir.rmdir()

    json_path = Path(args.json) if args.json else out / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"total {summary['wall_s']['total']:.1f}s for {len(videos)} videos -> {json_path}")

    if args.ref_json:
        ref = json.loads(Path(args.ref_json).read_text())
        for key in ("short_side", "frame_stride"):
            if ref.get(key) != summary[key]:
                print(f"WARNING: ref scored at {key}={ref.get(key)}, this run at {summary[key]}")
        print(f"{'dimension':<24} {'ref':>8} {'cand':>8} {'delta':>8}")
        for dim, score in summary["dims"].items():
            if dim in ref.get("dims", {}):
                print(f"{dim:<24} {ref['dims'][dim]:8.4f} {score:8.4f} {score - ref['dims'][dim]:+8.4f}")


if __name__ == "__main__":
    main()
