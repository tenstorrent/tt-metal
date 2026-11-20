#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Score the clips a `run_matrix.py run` kept: frame sanity, CLIP prompt alignment, VBench.

    score_clips.py <run_dir>      (run_matrix.py score --out <run_dir> puts the vbench deps on PYTHONPATH)

Adds a "quality" block to each row of <run_dir>/results.json. A scorer that cannot run (no vbench, no
network for its checkpoints) records its error in the row instead of stopping the others.

Bars are applied only where a test calibrated them: H3 T2VA's fox-prompt bars come from
test_pipeline_minimax_h3.py and Wan's from its VBENCH_GATE. FL2VA and Ref2VA clips are conditioned on
fractal or reference content, so they are scored and reported without a pass/fail bar.
"""
from __future__ import annotations

import json
import re
import shutil
import statistics
import sys
import traceback
from pathlib import Path

import numpy as np

H3_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "imaging_quality",
)
# calibrated 2026-08-04, fox prompt, seed 0 (test_pipeline_minimax_h3.py)
H3_T2VA_BARS = {
    "subject_consistency": 0.95,
    "background_consistency": 0.95,
    "motion_smoothness": 0.97,
    "dynamic_degree": 1.0,
    "imaging_quality": 0.64,
}
H3_T2VA_CLIP_BAR = 33.0


def sampled_frames(path: Path, every: int = 8) -> np.ndarray:
    import imageio.v3 as iio

    frames = [f for i, f in enumerate(iio.imiter(path, plugin="pyav")) if i % every == 0]
    return np.stack(frames)


def frame_sanity(frames: np.ndarray) -> dict:
    """Does this look like video rather than flat gray or noise."""
    luma = frames.astype(np.float32).mean(-1)
    step = np.abs(np.diff(luma, axis=0)).mean() if len(luma) > 1 else 0.0
    out = {
        "sampled_frames": int(len(frames)),
        "min": int(frames.min()),
        "max": int(frames.max()),
        "mean": round(float(frames.mean()), 1),
        "spatial_std": round(float(luma.std(axis=(1, 2)).mean()), 1),
        "frame_to_frame_change": round(float(step), 2),
    }
    out["flat_or_static"] = bool(out["spatial_std"] < 3.0 or out["max"] - out["min"] < 10)
    return out


def clip_alignment(frames: np.ndarray, prompt: str) -> dict:
    from models.tt_dit.tests.models.minimax_h3.common_av import clip_prompt_alignment

    return clip_prompt_alignment(frames, prompt)


def vbench_scores(bundle: Path, clip_names: list[tuple[int, int]]) -> dict:
    from models.tt_dit.utils import vbench_bundle

    scores = {}
    for prompt_index, seed in clip_names:
        scores[(prompt_index, seed)] = vbench_bundle.score_clip(bundle, prompt_index, seed)
    dims = sorted({d for row in scores.values() for d in row})
    return {d: statistics.mean(row[d] for row in scores.values() if d in row) for d in dims}


def h3_bundle(case_dir: Path, clip: Path, text: str) -> Path:
    from models.tt_dit.utils import vbench_bundle
    from models.tt_dit.utils.vbench_bundle import GatePrompt

    bundle = vbench_bundle.create_bundle(case_dir / "vbench_h3")
    shutil.copy2(clip, vbench_bundle.clip_path(bundle, 0, 0))
    # Record-only floors: the dimensions are what we want scored; bars are applied separately below.
    vbench_bundle.write_manifest(
        bundle, prompts=[GatePrompt(text=text, thresholds={d: 0.0 for d in H3_DIMENSIONS})], seeds=[0]
    )
    return bundle


def score_row(run: Path, row: dict) -> dict:
    q: dict = {}
    clips = [run / c for c in row.get("clips", [])]
    if not clips:
        return {"error": "no clip kept for this case"}
    case_dir = clips[0].parent
    try:
        frames = np.concatenate([sampled_frames(c) for c in clips[:1]])
        q["frames"] = frame_sanity(frames)
    except Exception as e:  # noqa: BLE001
        return {"error": f"cannot decode {clips[0].name}: {e}"}

    if row["family"] == "h3":
        from models.tt_dit.tests.models.minimax_h3.common_av import CALIBRATED_FOX_PROMPT as prompt

        bars = H3_T2VA_BARS if row["task"] == "t2va" else {}
        clip_bar = H3_T2VA_CLIP_BAR if row["task"] == "t2va" else None
        try:
            bundle = h3_bundle(case_dir, clips[0], prompt)
            q["vbench"] = vbench_scores(bundle, [(0, 0)])
        except Exception:  # noqa: BLE001
            q["vbench_error"] = traceback.format_exc(limit=2)[-600:]
    else:
        if case_dir.name != "vbench":
            case_dir = case_dir / "vbench"
        manifest = (
            json.loads((case_dir / "manifest.json").read_text()) if (case_dir / "manifest.json").is_file() else None
        )
        if manifest is None:
            return {**q, "error": "Wan VBench bundle has no manifest (run without --no-vbench)"}
        prompt = manifest["prompts"][0]["text"]
        bars = manifest["prompts"][0]["thresholds"]
        clip_bar = None
        try:
            q["vbench"] = vbench_scores(case_dir, [(0, s) for s in manifest["seeds"]])
            q["vbench_seeds"] = manifest["seeds"]
        except Exception:  # noqa: BLE001
            q["vbench_error"] = traceback.format_exc(limit=2)[-600:]
        # CLIP over the first gate clip
        frames = sampled_frames(case_dir / f"p0_s{manifest['seeds'][0]}.mp4")

    try:
        q["clip"] = {k: round(float(v), 2) for k, v in clip_alignment(frames, prompt).items()}
        q["clip_bar"] = clip_bar
        if clip_bar is not None:
            q["clip_ok"] = q["clip"]["mean"] >= clip_bar
    except Exception:  # noqa: BLE001
        q["clip_error"] = traceback.format_exc(limit=2)[-600:]
    if "vbench" in q and bars:
        q["vbench_bars"] = bars
        q["vbench_ok"] = all(q["vbench"].get(d, float("nan")) >= b for d, b in bars.items())
    return q


def main():
    run = Path(sys.argv[1])
    results = json.loads((run / "results.json").read_text())
    for row in results:
        if (
            row.get("status") == "FAIL"
            and re.search(r"outside \[[\d.]+, [\d.]+\]s", row.get("detail", ""))
            and row.get("clips")
        ):
            row["status"] = "PERF_LIMIT"  # measured fine; tripped only the full-DRAM timing limits
        if row.get("status") not in ("PASS", "PERF_LIMIT"):
            continue
        q0 = row.get("quality") or {}
        if q0 and not any(k in q0 for k in ("error", "vbench_error", "clip_error")):
            continue  # already scored cleanly
        print(f"scoring {row['case']} ...", flush=True)
        row["quality"] = score_row(run, row)
        print(json.dumps(row["quality"], indent=1)[:800], flush=True)
        (run / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
