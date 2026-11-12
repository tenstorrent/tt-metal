# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from models.tt_dit.utils import vbench_bundle
from models.tt_dit.utils.vbench import assert_scores
from models.tt_dit.utils.vbench_bundle import GatePrompt

SEEDS = [42, 43, 44, 45]
PROMPTS = [
    GatePrompt("cats", {"imaging_quality": 0.6, "dynamic_degree": 0.75}),
    GatePrompt("alley", {"subject_consistency": 0.8}),
]


@pytest.fixture
def bundle(tmp_path):
    bundle = vbench_bundle.create_bundle(tmp_path / "vbench")
    for index in range(len(PROMPTS)):
        for seed in SEEDS:
            vbench_bundle.clip_path(bundle, index, seed).touch()
    vbench_bundle.write_manifest(bundle, prompts=PROMPTS, seeds=SEEDS)
    return bundle


def passing_scores():
    # Prompt 0 moves on 3 of 4 seeds: exactly the 0.75 dynamic_degree floor.
    scores = {(0, seed): {"imaging_quality": 0.65, "dynamic_degree": float(seed != 42)} for seed in SEEDS}
    scores.update({(1, seed): {"subject_consistency": 0.9} for seed in SEEDS})
    return scores


def test_each_prompt_is_gated_on_its_mean_over_seeds(bundle):
    means = vbench_bundle.aggregate(vbench_bundle.read_manifest(bundle), passing_scores())
    assert means == [{"imaging_quality": 0.65, "dynamic_degree": 0.75}, {"subject_consistency": 0.9}]


@pytest.mark.parametrize("failure", ["below_floor", "missing_metric", "nan"])
def test_one_bad_prompt_fails_the_gate(bundle, failure, expect_error):
    scores = passing_scores()
    if failure == "below_floor":
        scores[(1, 45)]["subject_consistency"] = 0.0
    elif failure == "missing_metric":
        del scores[(1, 45)]["subject_consistency"]
    else:
        scores[(1, 45)]["subject_consistency"] = float("nan")
    with expect_error(AssertionError, "(?s)prompt 1 .*subject_consistency"):
        vbench_bundle.aggregate(vbench_bundle.read_manifest(bundle), scores)


@pytest.mark.parametrize(
    "failure", ["missing_clip", "duplicate_seed", "foreign_scorer", "no_thresholds", "unknown_field", "missing_field"]
)
def test_incomplete_or_bad_gate_is_rejected(bundle, failure, expect_error):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if failure == "missing_clip":
        vbench_bundle.clip_path(bundle, 1, 45).unlink()
    elif failure == "duplicate_seed":
        manifest["seeds"] = [42, 42]
    elif failure == "foreign_scorer":
        manifest["scorer"] = "os:system"
    elif failure == "no_thresholds":
        manifest["prompts"][0]["thresholds"] = {}
    elif failure == "unknown_field":
        manifest["prompts"][0]["weight"] = 2
    else:
        del manifest["seeds"]
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with expect_error(ValueError, "VBench"):
        vbench_bundle.read_manifest(bundle)


def test_rerun_into_a_bundle_removes_only_its_own_files(bundle):
    unrelated = [bundle / "notes.txt", bundle / "pizza_s1.mp4"]
    for path in unrelated:
        path.write_text("keep")
    vbench_bundle.create_bundle(bundle)
    assert not (bundle / "manifest.json").exists()
    assert not any(vbench_bundle.clip_path(bundle, i, seed).exists() for i in range(len(PROMPTS)) for seed in SEEDS)
    assert all(path.read_text() == "keep" for path in unrelated)


class InlinePool:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, fn, *args):
        try:
            result = fn(*args)
        except Exception as error:
            failure = error

            def result_fn():
                raise failure

            return SimpleNamespace(result=result_fn)
        return SimpleNamespace(result=lambda: result)


def test_evaluate_scores_every_clip_with_the_manifest_scorer(bundle, monkeypatch):
    calls = []
    scores = passing_scores()

    def score(path, *, prompt, dimensions, **kwargs):
        index = next(i for i, p in enumerate(PROMPTS) if p.text == prompt)
        seed = int(path.rsplit("_s", 1)[1].removesuffix(".mp4"))
        calls.append((index, seed, dimensions))
        return scores[(index, seed)]

    monkeypatch.setattr(vbench_bundle, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(vbench_bundle, "load_scorer", lambda name: score)
    vbench_bundle.evaluate(bundle)
    assert sorted((i, s) for i, s, _ in calls) == sorted(scores)
    assert all(dimensions == list(PROMPTS[i].thresholds) for i, _, dimensions in calls)


def test_shared_models_are_fetched_once_before_the_workers_start(tmp_path, monkeypatch):
    bundle = vbench_bundle.create_bundle(tmp_path / "vbench")
    prompts = [GatePrompt("cats", {"background_consistency": 0.8})]
    vbench_bundle.clip_path(bundle, 0, 42).touch()
    vbench_bundle.write_manifest(bundle, prompts=prompts, seeds=[42])
    events = []

    class Pool(InlinePool):
        def submit(self, fn, *args):
            events.append("worker")
            return SimpleNamespace(result=lambda: {"background_consistency": 0.9})

    monkeypatch.setattr(vbench_bundle, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(vbench_bundle, "prefetch_models", lambda manifest: events.append("prefetch"))
    vbench_bundle.evaluate(bundle)
    assert events == ["prefetch", "worker"]


def test_worker_failure_fails_the_gate(bundle, monkeypatch, expect_error):
    def score(path, **kwargs):
        raise RuntimeError("scoring worker failed")

    monkeypatch.setattr(vbench_bundle, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(vbench_bundle, "load_scorer", lambda name: score)
    with expect_error(RuntimeError, "scoring worker failed"):
        vbench_bundle.evaluate(bundle)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1])
def test_invalid_scores_fail(score, expect_error):
    with expect_error(AssertionError, "VBench quality gate failed"):
        assert_scores({"quality": score}, {"quality": 0.8})
