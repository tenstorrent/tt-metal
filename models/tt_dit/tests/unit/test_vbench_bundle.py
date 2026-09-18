# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from models.tt_dit.utils.vbench import assert_scores
from models.tt_dit.utils.vbench_bundle import METRICS, aggregate, digest


@pytest.fixture
def gate(tmp_path):
    bundle = tmp_path / "input"
    bundle.mkdir()
    manifest = {
        "prompt": "test",
        "thresholds": {metric: 0.8 for metric in METRICS},
        "clips": [{"name": f"seed_{i}.mp4", "shape": [145, 1088, 1920], "sha256": "unused"} for i in range(5)],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    results = tmp_path / "scores"
    results.mkdir()
    for i in range(5):
        scores = {metric: 0.9 for metric in METRICS}
        scores["dynamic_degree"] = int(i != 0)
        row = {"index": i, "manifest_sha256": digest(bundle / "manifest.json"), "scores": scores}
        (results / f"score_{i}.json").write_text(json.dumps(row))
    return bundle, results


def test_gate_averages_all_five_seeds(gate):
    assert aggregate(*gate)["dynamic_degree"] == 0.8


@pytest.mark.parametrize("failure", ["missing", "duplicate", "metric", "nan", "wrong_input", "below_floor"])
def test_incomplete_or_bad_quality_cannot_pass(gate, failure, expect_error):
    bundle, results = gate
    path = results / "score_4.json"
    row = json.loads(path.read_text())
    if failure == "missing":
        path.unlink()
    else:
        if failure == "duplicate":
            row["index"] = 0
        elif failure == "metric":
            del row["scores"]["motion_smoothness"]
        elif failure == "nan":
            row["scores"]["motion_smoothness"] = float("nan")
        elif failure == "wrong_input":
            row["manifest_sha256"] = "different build"
        else:
            row["scores"]["dynamic_degree"] = 0
        path.write_text(json.dumps(row))
    with expect_error((ValueError, AssertionError), "VBench"):
        aggregate(bundle, results)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1])
def test_invalid_scores_fail(score, expect_error):
    with expect_error(AssertionError, "VBench quality gate failed"):
        assert_scores({"quality": score}, {"quality": 0.8})


@pytest.fixture
def model_cache(tmp_path, monkeypatch):
    from models.tt_dit.utils.vbench_bundle import DINO_REPO, WEIGHTS

    cache = tmp_path / "cache"
    for relative in (*WEIGHTS, f"{DINO_REPO}/hubconf.py"):
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    (cache / "unrelated-model").write_bytes(b"do not export")
    monkeypatch.setenv("VBENCH_CACHE_DIR", str(cache))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return cache


def test_stage_only_required_weights_and_verify_checksums(model_cache, tmp_path, expect_error):
    from models.tt_dit.utils.vbench_bundle import WEIGHTS, stage_models, verify_models

    bundle = tmp_path / "bundle"
    assets = stage_models(bundle / "models")
    assert len(assets) == 6
    assert "unrelated-model" not in assets
    assert verify_models(bundle, assets) == (bundle / "models").resolve()
    (bundle / "models" / WEIGHTS[0]).write_bytes(b"corrupted")
    with expect_error(ValueError, "checksum mismatch"):
        verify_models(bundle, assets)


def test_missing_weight_fails_export(model_cache, tmp_path, expect_error):
    from models.tt_dit.utils.vbench_bundle import WEIGHTS, stage_models

    (model_cache / WEIGHTS[1]).unlink()
    with expect_error(FileNotFoundError, "Missing staged VBench asset"):
        stage_models(tmp_path / "bundle/models")


def test_existing_clip_and_torch_caches_are_supported(model_cache, tmp_path):
    from models.tt_dit.utils.vbench_bundle import DINO_REPO, WEIGHTS, stage_models, verify_models

    alternatives = {
        WEIGHTS[0]: tmp_path / "home/.cache/clip/ViT-B-32.pt",
        WEIGHTS[4]: tmp_path / "torch/hub/checkpoints/dino_vitbase16_pretrain.pth",
        DINO_REPO: tmp_path / "torch/hub/facebookresearch_dino_main",
    }
    for relative, path in alternatives.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        (model_cache / relative).rename(path)
    bundle = tmp_path / "bundle"
    assets = stage_models(bundle / "models")
    verify_models(bundle, assets)
