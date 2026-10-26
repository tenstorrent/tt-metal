# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VBench gate over N prompts x M seeds: each prompt's mean over its seeds must clear that prompt's floors.

A single clip at a single seed is too noisy to gate on (a numeric perturbation reshuffles the whole
video, and dynamic_degree is binary per clip), so a test renders every (prompt, seed) clip into a
fresh bundle directory with ``clip_path`` and then calls ``write_manifest``. The bundle is scored by
``evaluate``: in-process for a local run, or in CI by a separate step after pytest has closed the
devices::

    python3 -m models.tt_dit.utils.vbench_bundle evaluate --bundle DIR

The manifest names the scoring function, so a model with its own scoring policy (e.g. reduced-
resolution temporal metrics) sets it from its test and the CI command stays the same.
"""

import argparse
import importlib
import json
import math
import multiprocessing
import os
import socket
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_SCORER = "models.tt_dit.utils.vbench:score_vbench"


@dataclass(frozen=True)
class GatePrompt:
    """One gate prompt: its text and the floor each scored metric's mean over the seeds must clear."""

    text: str
    thresholds: dict[str, float]

    def __post_init__(self):
        if not self.text or not self.thresholds:
            raise ValueError("VBench prompt needs text and thresholds")


@dataclass(frozen=True)
class Manifest:
    """What a bundle holds: every (prompt, seed) clip, and how to score it."""

    prompts: tuple[GatePrompt, ...]
    seeds: tuple[int, ...]
    scorer: str = DEFAULT_SCORER
    scorer_kwargs: dict = field(default_factory=dict)
    host: str = field(default_factory=socket.gethostname)

    def __post_init__(self):
        if not self.prompts or not self.seeds:
            raise ValueError("VBench gate needs at least one prompt and one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("VBench gate seeds must be distinct")
        if not self.scorer.startswith("models.tt_dit."):
            raise ValueError(f"VBench scorer must live under models.tt_dit: {self.scorer}")

    @classmethod
    def from_json(cls, text):
        """Strict: an unknown or missing field is an error, not a default."""
        data = json.loads(text)
        try:
            prompts = tuple(GatePrompt(**prompt) for prompt in data.pop("prompts"))
            return cls(prompts=prompts, seeds=tuple(data.pop("seeds")), **data)
        except (KeyError, TypeError) as error:
            raise ValueError(f"Malformed VBench manifest: {error}") from error

    def check_clips(self, bundle):
        for index in range(len(self.prompts)):
            for seed in self.seeds:
                if not clip_path(bundle, index, seed).is_file():
                    raise ValueError(f"Missing VBench clip: {clip_path(bundle, index, seed)}")


def create_bundle(bundle):
    """Create an empty bundle directory; an existing one is refused so stale clips can't be scored."""
    bundle = Path(bundle)
    bundle.mkdir(parents=True, exist_ok=False)
    return bundle


def clip_path(bundle, prompt_index, seed):
    return Path(bundle) / f"p{prompt_index}_s{seed}.mp4"


def write_manifest(bundle, *, prompts, seeds, scorer=DEFAULT_SCORER, scorer_kwargs=None):
    """Record the gate once every clip is rendered. Each prompt is scored on its own threshold metrics."""
    manifest = Manifest(tuple(prompts), tuple(seeds), scorer, scorer_kwargs or {})
    manifest.check_clips(bundle)
    (Path(bundle) / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))


def read_manifest(bundle):
    manifest = Manifest.from_json((Path(bundle) / "manifest.json").read_text())
    manifest.check_clips(bundle)
    return manifest


def load_scorer(name):
    module, function = name.split(":")
    return getattr(importlib.import_module(module), function)


def score_clip(bundle, prompt_index, seed):
    import torch

    torch.set_num_threads(4)
    manifest = read_manifest(bundle)
    prompt = manifest.prompts[prompt_index]
    score = load_scorer(manifest.scorer)
    return score(
        str(clip_path(bundle, prompt_index, seed).resolve()),
        prompt=prompt.text,
        dimensions=list(prompt.thresholds),
        **manifest.scorer_kwargs,
    )


def aggregate(manifest, scores):
    """Average each prompt's scores over the seeds and enforce that prompt's floors.

    ``scores`` maps ``(prompt_index, seed)`` to a ``{metric: score}`` dict. A missing or non-finite
    score makes that prompt's mean non-finite, which fails the gate.
    """
    from models.tt_dit.utils.vbench import assert_scores

    seeds = manifest.seeds
    means = []
    failures = []
    for index, prompt in enumerate(manifest.prompts):
        thresholds = prompt.thresholds
        for seed in seeds:
            row = scores[(index, seed)]
            print(f"prompt {index} seed {seed}: " + ", ".join(f"{m}={row.get(m, math.nan):.4f}" for m in thresholds))
        mean = {m: sum(scores[(index, seed)].get(m, math.nan) for seed in seeds) / len(seeds) for m in thresholds}
        print(f"prompt {index} mean over {len(seeds)} seeds: " + ", ".join(f"{m}={v:.4f}" for m, v in mean.items()))
        means.append(mean)
        try:
            assert_scores(mean, thresholds)
        except AssertionError as error:
            failures.append(f"prompt {index} ({prompt.text[:60]!r}): {error}")

    if failures:
        raise AssertionError("\n".join(failures))
    return means


def prefetch_models(manifest):
    """Download, in this process, the models the workers would otherwise all fetch at once.

    VBench (local=False) loads background_consistency's CLIP "ViT-B/32" through the clip package, which
    downloads it on first use into ~/.cache/clip, empty in a fresh CI container. Parallel workers all
    start that 338 MB download into the same file, and the overlapping writes fail its SHA256 check.
    """
    if any("background_consistency" in prompt.thresholds for prompt in manifest.prompts):
        import clip

        clip.load("ViT-B/32", device="cpu")


def evaluate(bundle, *, workers=None):
    """Score every clip in parallel CPU processes, then enforce each prompt's mean."""
    bundle = Path(bundle).resolve()
    manifest = read_manifest(bundle)
    clips = [(index, seed) for index in range(len(manifest.prompts)) for seed in manifest.seeds]
    if workers is None:
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
        workers = min(len(clips), max(1, available // 4))
    print(
        f"VBench: {len(manifest.prompts)} prompts x {len(manifest.seeds)} seeds rendered on "
        f"{manifest.host}, {workers} CPU workers, scorer {manifest.scorer} {manifest.scorer_kwargs}",
        flush=True,
    )
    prefetch_models(manifest)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {clip: pool.submit(score_clip, bundle, *clip) for clip in clips}
        scores = {clip: future.result() for clip, future in futures.items()}
    return aggregate(manifest, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["evaluate"])
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    evaluate(args.bundle)
