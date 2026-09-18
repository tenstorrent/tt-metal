# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Export five equal-length LTX clips, score on CPU workers, and enforce their mean quality."""

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

METRICS = {"subject_consistency", "background_consistency", "motion_smoothness", "dynamic_degree", "imaging_quality"}
WEIGHTS = (
    "clip_model/ViT-B-32.pt",
    "raft_model/models/raft-things.pth",
    "amt_model/amt-s.pth",
    "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
    "dino_model/dino_vitbase16_pretrain.pth",
)
DINO_REPO = "dino_model/facebookresearch_dino_main"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def stage_models(destination):
    """Carry the existing evaluator weights to runners without model-cache mounts.

    The CPU runner's proxy blocks some upstream weight URLs. Use VBench's public
    local mode with only the five required weights and DINO source, not the entire
    shared cache. Missing assets fail at export, before costly CPU evaluation.
    """
    destination = Path(destination)
    cache = Path(os.environ.get("VBENCH_CACHE_DIR", Path.home() / ".cache/vbench"))
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))
    alternatives = {
        WEIGHTS[0]: Path.home() / ".cache/clip/ViT-B-32.pt",
        WEIGHTS[4]: torch_home / "hub/checkpoints/dino_vitbase16_pretrain.pth",
        DINO_REPO: torch_home / "hub/facebookresearch_dino_main",
    }
    for relative in (*WEIGHTS, DINO_REPO):
        source = cache / relative
        if not source.exists():
            source = alternatives.get(relative, source)
        if not source.exists():
            raise FileNotFoundError(f"Missing staged VBench asset: {relative} (cache: {cache})")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == DINO_REPO:
            shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "__pycache__", ".*"))
        else:
            shutil.copyfile(source, target)
    return {
        str(path.relative_to(destination)): digest(path) for path in sorted(destination.rglob("*")) if path.is_file()
    }


def verify_models(bundle, assets):
    cache = Path(bundle) / "models"
    if not set((*WEIGHTS, f"{DINO_REPO}/hubconf.py")).issubset(assets):
        raise ValueError("Incomplete VBench model assets")
    for relative, checksum in assets.items():
        path = cache / relative
        if not path.resolve().is_relative_to(cache.resolve()) or digest(path) != checksum:
            raise ValueError(f"VBench model checksum mismatch: {relative}")
    return cache.resolve()


def export_bundle(source, destination, *, prompt, thresholds):
    import cv2

    source, destination = Path(source), Path(destination)
    clips = sorted(source.glob("*.mp4"))
    if len(clips) != 5 or set(thresholds) != METRICS:
        raise ValueError("LTX CI requires all five seeds and all five VBench metrics")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"prompt": prompt, "thresholds": thresholds, "clips": []}
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
    manifest["assets"] = stage_models(destination / "models")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_manifest(bundle):
    manifest = json.loads((Path(bundle) / "manifest.json").read_text())
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
    # VBench reads this environment variable at import time. Configure it before
    # score_vbench imports VBench, then use its public local=True evaluator path.
    cache = verify_models(bundle, manifest["assets"])
    os.environ["VBENCH_CACHE_DIR"] = str(cache)
    # Upstream DINO's hubconf still calls load_state_dict_from_url in local mode.
    # Populate its expected Torch cache path with the verified, exported weight.
    torch.hub.set_dir(str(cache / "torch_hub"))
    checkpoint = cache / "torch_hub/checkpoints/dino_vitbase16_pretrain.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        checkpoint.symlink_to(cache / WEIGHTS[4])
    clip = manifest["clips"][index]
    path = bundle / clip["name"]
    if digest(path) != clip["sha256"]:
        raise ValueError(f"VBench clip checksum mismatch: {path}")
    scores = score_vbench(
        str(path.resolve()), prompt=manifest["prompt"], dimensions=list(manifest["thresholds"]), local=True
    )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["score", "aggregate"])
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--index", type=int, choices=range(5))
    parser.add_argument("--output")
    parser.add_argument("--results")
    args = parser.parse_args()
    if args.mode == "score":
        if args.index is None or not args.output:
            parser.error("score requires --index and --output")
        score_clip(args.bundle, args.index, args.output)
    else:
        if not args.results:
            parser.error("aggregate requires --results")
        aggregate(args.bundle, args.results)
