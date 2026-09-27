# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Export five equal-length LTX clips, score on the same host, and enforce their mean quality."""

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

METRICS = {"subject_consistency", "background_consistency", "motion_smoothness", "dynamic_degree", "imaging_quality"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def export_bundle(source, destination, *, prompt, thresholds, temporal_width=0):
    import cv2

    source, destination = Path(source), Path(destination)
    clips = sorted(source.glob("*.mp4"))
    if len(clips) != 5 or set(thresholds) != METRICS:
        raise ValueError("LTX CI requires all five seeds and all five VBench metrics")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"prompt": prompt, "thresholds": thresholds, "temporal_width": temporal_width, "clips": []}
    for index, clip in enumerate(clips):
        cap = cv2.VideoCapture(str(clip))
        try:
            shape = [
                int(cap.get(prop))
                for prop in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH)
            ]
        finally:
            cap.release()
        # Equal frame counts make the mean of per-clip scores identical to VBench's
        # directory aggregation (subject/background consistency weight by frame).
        if shape != [145, 1088, 1920]:
            raise ValueError(f"Unexpected VBench clip shape: {clip}: {shape}")
        name = f"seed_{index}.mp4"
        shutil.copyfile(clip, destination / name)
        manifest["clips"].append({"name": name, "sha256": digest(destination / name), "shape": shape})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_manifest(bundle):
    manifest = json.loads((Path(bundle) / "manifest.json").read_text())
    width = manifest.get("temporal_width", 0)
    if type(width) is not int or width < 0 or width > 1920 or width % 2:
        raise ValueError("Invalid VBench temporal width")
    if len(manifest["clips"]) != 5 or set(manifest["thresholds"]) != METRICS:
        raise ValueError("Incomplete VBench manifest")
    for index, clip in enumerate(manifest["clips"]):
        if clip["name"] != f"seed_{index}.mp4" or clip["shape"] != [145, 1088, 1920]:
            raise ValueError("Invalid VBench clip identity or shape")
    return manifest


def score_clip(bundle, index, output):
    import torch

    from models.tt_dit.utils.vbench import score_vbench

    torch.set_num_threads(4)
    bundle = Path(bundle)
    manifest = read_manifest(bundle)
    clip = manifest["clips"][index]
    path = bundle / clip["name"]
    if digest(path) != clip["sha256"]:
        raise ValueError(f"VBench clip checksum mismatch: {path}")
    if manifest.get("temporal_width", 0):
        from models.tt_dit.utils.vbench_cpu import score_clip_ci

        scores = score_clip_ci(path.resolve(), prompt=manifest["prompt"], temporal_width=manifest["temporal_width"])
    else:
        scores = score_vbench(str(path.resolve()), prompt=manifest["prompt"], dimensions=list(manifest["thresholds"]))
    Path(output).write_text(
        json.dumps({"index": index, "manifest_sha256": digest(bundle / "manifest.json"), "scores": scores})
    )


def aggregate(bundle, results):
    from models.tt_dit.utils.vbench import assert_scores

    bundle = Path(bundle)
    manifest = read_manifest(bundle)
    manifest_hash = digest(bundle / "manifest.json")
    rows = [json.loads(path.read_text()) for path in Path(results).glob("score_*.json")]
    if len(rows) != 5 or sorted(row["index"] for row in rows) != list(range(5)):
        raise ValueError("Missing or duplicate VBench seed results; all five are required")
    for row in rows:
        if row["manifest_sha256"] != manifest_hash or set(row["scores"]) != METRICS:
            raise ValueError("VBench results do not match the complete exported gate")
        if not all(math.isfinite(value) for value in row["scores"].values()):
            raise ValueError("Non-finite VBench score")
    means = {metric: sum(row["scores"][metric] for row in rows) / 5 for metric in METRICS}
    print(json.dumps(means, indent=2))
    assert_scores(means, manifest["thresholds"])
    return means


def evaluate(bundle):
    """Run all five seeds concurrently, propagate worker failures, then enforce the mean."""
    bundle = Path(bundle).resolve()
    manifest = read_manifest(bundle)
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
    workers = min(5, max(1, available // 4))
    print(f"VBench: five seeds, {workers} CPU workers, temporal width={manifest.get('temporal_width', 0)}", flush=True)
    with tempfile.TemporaryDirectory(prefix="vbench-results-") as results:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [
                pool.submit(score_clip, bundle, index, Path(results) / f"score_{index}.json") for index in range(5)
            ]
            for future in futures:
                future.result()
        return aggregate(bundle, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["score", "aggregate", "evaluate"])
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--index", type=int, choices=range(5))
    parser.add_argument("--output")
    parser.add_argument("--results")
    args = parser.parse_args()
    if args.mode == "evaluate":
        evaluate(args.bundle)
    elif args.mode == "score":
        if args.index is None or not args.output:
            parser.error("score requires --index and --output")
        score_clip(args.bundle, args.index, args.output)
    else:
        if not args.results:
            parser.error("aggregate requires --results")
        aggregate(args.bundle, args.results)
