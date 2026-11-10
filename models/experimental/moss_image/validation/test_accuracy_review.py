# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for accuracy metric normalization and failure handling."""
import json

import pytest
import torch

from models.experimental.moss_image.validation.accuracy_review import error_metrics, review, sha256


def test_pcc_does_not_hide_gain_and_offset_error():
    reference = torch.tensor([1.0, 2.0, 3.0])
    actual = 2 * reference + 5
    result = error_metrics(actual, reference)
    assert result["pcc"] == pytest.approx(1)
    assert result["relative_rms"] > 3
    assert result["std_normalized_rmse"] > result["relative_rms"]
    assert result["fitted_actual_gain_vs_reference"] == pytest.approx(2)
    assert result["fitted_actual_offset_vs_reference"] == pytest.approx(5)
    assert result["rmse_after_fitting_reference_gain_offset"] == pytest.approx(0)


def test_population_standard_deviation_is_used():
    result = error_metrics(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 2.0]))
    assert result["rmse"] == pytest.approx(1)
    assert result["reference_population_std"] == pytest.approx(1)
    assert result["std_normalized_rmse"] == pytest.approx(1)
    assert result["reference_sample_std"] == pytest.approx(2**0.5)
    assert result["std_normalized_rmse_upstream"] == pytest.approx(2**-0.5)
    assert result["relative_rms"] == pytest.approx(2**-0.5)


def test_zero_denominator_is_explicit_and_json_finite():
    result = error_metrics(torch.ones(3), torch.zeros(3))
    assert result["pcc"] is None
    assert result["relative_rms"] is None
    assert result["std_normalized_rmse"] is None
    assert result["rmse"] == 1
    json.dumps(result, allow_nan=False)
    equal = error_metrics(torch.zeros(3), torch.zeros(3))
    assert equal["relative_rms"] == 0
    assert equal["std_normalized_rmse"] == 0
    assert equal["bitwise_equal"]


def test_sample_standard_deviation_is_undefined_for_one_element():
    result = error_metrics(torch.ones(1), torch.ones(1))
    assert result["reference_sample_std"] is None
    assert result["std_normalized_rmse_upstream"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "actual,reference",
    [
        (torch.ones(2), torch.ones(1, 2)),
        (torch.tensor([float("nan")]), torch.zeros(1)),
        (torch.zeros(1), torch.tensor([float("inf")])),
        (torch.empty(0), torch.empty(0)),
    ],
)
def test_invalid_metrics_fail(actual, reference):
    with pytest.raises(ValueError):
        error_metrics(actual, reference)


def review_fixture(tmp_path):
    """Synthetic artifact protocol test only; never pretends to execute TT."""
    actual, reference = tmp_path / "actual", tmp_path / "reference"
    actual.mkdir()
    reference.mkdir()
    artifacts = {}

    def save(root, name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
        if root == reference:
            artifacts[name.removesuffix(".pt")] = {"path": name, "sha256": sha256(path)}

    values = torch.arange(4).float().reshape(1, 1, 2, 2)
    context = {"hidden_states": torch.arange(8).float().reshape(1, 2, 4)}
    schedule = {"timesteps": torch.tensor([1000.0]), "sigmas": torch.tensor([1.0, 0.0])}
    save(actual, "initial_noise.pt", values)
    save(reference, "steps/000/input_latents.pt", values)
    for root in [actual, reference]:
        save(root, "scheduler.pt", schedule)
        for branch in ["positive", "negative"]:
            save(root, f"conditioning_{branch}.pt" if root == actual else f"conditioning/{branch}.pt", context)
            save(
                root, f"first_{branch}_prediction.pt" if root == actual else f"steps/000/{branch}_prediction.pt", values
            )
        save(root, "latent_001.pt" if root == actual else "steps/000/updated_latents.pt", values)
        # Reference is NCHW, saved actual image is NHWC.
        image = torch.linspace(-0.9, 0.9, 12).reshape(1, 3, 2, 2)
        save(
            root,
            "decoded_nhwc.pt" if root == actual else "vae/raw_output.pt",
            image.permute(0, 2, 3, 1) if root == actual else image,
        )
        (root / "image.png").write_bytes(b"host-only-fixture-image")
    arguments = dict(
        variant="SFT",
        prompt="fixture",
        negative_prompt="",
        height=32,
        width=32,
        steps=1,
        seed=42,
        guidance_scale=3,
        generator_device="cpu",
    )
    checkpoint = {"files": {"example": {"sha256": "fixture-checkpoint-identity"}}}
    status = dict(
        status="complete",
        activation_injection=False,
        cuda_used=False,
        arguments=arguments,
        checkpoint=checkpoint,
        steps=[{"index": 0}],
        timings={},
    )
    oracle = dict(
        status="complete",
        arguments=arguments,
        guidance_scale=3,
        checkpoint=checkpoint,
        artifacts=artifacts,
        steps=[{"index": 0}],
        image={"sha256": sha256(reference / "image.png")},
    )
    (actual / "status.json").write_text(json.dumps(status))
    (reference / "manifest.json").write_text(json.dumps(oracle))
    return actual, reference


def test_independent_review_checks_artifact_hashes(tmp_path):
    actual, reference = review_fixture(tmp_path)
    manifest_before = sha256(reference / "manifest.json")
    result = review(actual, reference, expected_manifest_sha256=manifest_before)
    assert result["actual_steps"] == 1
    assert result["checkpoint_files_matched_in_status"]
    assert result["image_accuracy"]["pcc_threshold_met"]
    assert result["metrics"]["decoded_pixels"]["std_normalized_rmse"] == 0
    assert result["reference_manifest_sha256"] == manifest_before
    assert sha256(reference / "manifest.json") == manifest_before
    torch.save({"hidden_states": torch.ones(1, 2, 4)}, reference / "conditioning/positive.pt")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        review(actual, reference, expected_manifest_sha256=manifest_before)


def test_review_rejects_configuration_drift(tmp_path):
    actual, reference = review_fixture(tmp_path)
    manifest_pin = sha256(reference / "manifest.json")
    path = actual / "status.json"
    status = json.loads(path.read_text())
    status["arguments"]["seed"] += 1
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="Configuration differs: seed"):
        review(actual, reference, expected_manifest_sha256=manifest_pin)


def test_review_rejects_replaced_reference_manifest(tmp_path):
    actual, reference = review_fixture(tmp_path)
    path = reference / "manifest.json"
    original_pin = sha256(path)
    content = json.loads(path.read_text())
    content["replacement"] = True
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="reference manifest pin mismatch"):
        review(actual, reference, expected_manifest_sha256=original_pin)
