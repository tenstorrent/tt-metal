# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read-only accuracy review of independent runs against unchanged CUDA captures.

This module never opens an accelerator or supplies reference activations to one.
Population and sample-standard-deviation-normalized RMSE supplement, rather than
replace, the original reference-RMS-normalized error metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    """Float64 metrics; undefined constant-reference metrics are JSON null."""
    if actual.shape != reference.shape or actual.numel() == 0:
        raise ValueError("Metric tensors must have equal, nonempty shapes")
    a, b = actual.double().flatten(), reference.double().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Metric tensors must be finite")
    ac, bc = a - a.mean(), b - b.mean()
    difference = a - b
    rmse = float(difference.square().mean().sqrt())
    reference_rms = float(b.square().mean().sqrt())
    reference_std = float(bc.square().mean().sqrt())
    reference_sample_std = float((bc.square().sum() / (b.numel() - 1)).sqrt()) if b.numel() > 1 else None

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else (0.0 if numerator == 0 else None)

    pcc_scale = float(ac.norm() * bc.norm())
    pcc = float((ac.dot(bc) / pcc_scale).clamp(-1, 1)) if pcc_scale else None
    gain = float(ac.dot(bc) / bc.dot(bc)) if reference_std else None
    offset = float(a.mean()) - gain * float(b.mean()) if gain is not None else None
    residual = float((a - (gain * b + offset)).square().mean().sqrt()) if gain is not None else None
    return {
        "pcc": pcc,
        "rmse": rmse,
        "relative_rms": ratio(rmse, reference_rms),
        "std_normalized_rmse": ratio(rmse, reference_std),
        "std_normalized_rmse_upstream": (
            ratio(rmse, reference_sample_std) if reference_sample_std is not None else None
        ),
        "max_absolute_error": float(difference.abs().max()),
        "mean_error": float(difference.mean()),
        "reference_rms": reference_rms,
        "reference_population_std": reference_std,
        "reference_sample_std": reference_sample_std,
        "actual_rms": float(a.square().mean().sqrt()),
        "fitted_actual_gain_vs_reference": gain,
        "fitted_actual_offset_vs_reference": offset,
        "rmse_after_fitting_reference_gain_offset": residual,
        "std_normalized_rmse_after_fitting_reference_gain_offset": (
            ratio(residual, reference_std) if residual is not None else None
        ),
        "bitwise_equal": torch.equal(actual, reference),
    }


def review(
    actual: Path,
    reference: Path,
    *,
    image_pcc_threshold: float = 0.99,
    expected_manifest_sha256: str | None = None,
) -> dict:
    actual, reference = actual.resolve(), reference.resolve()
    status = json.loads((actual / "status.json").read_text())
    oracle = json.loads((reference / "manifest.json").read_text())
    if status["status"] != "complete" or oracle["status"] != "complete":
        raise ValueError("Both independent runs must be complete")
    if status.get("activation_injection") is not False or status.get("cuda_used") is not False:
        raise ValueError("Actual run must declare no activation injection and no CUDA use")
    a, b = status["arguments"], oracle["arguments"]
    manifest_hash = sha256(reference / "manifest.json")
    if expected_manifest_sha256 is None:
        pins = json.loads((Path(__file__).parents[1] / "tests/capture_pins.json").read_text())
        expected_manifest_sha256 = pins[b["variant"]]["manifest_sha256"]
    if manifest_hash != expected_manifest_sha256:
        raise ValueError("Original CUDA reference manifest pin mismatch")
    for key in ["variant", "prompt", "negative_prompt", "height", "width", "steps", "seed"]:
        if a[key] != b[key]:
            raise ValueError(f"Configuration differs: {key}")
    if a["guidance_scale"] != oracle["guidance_scale"] or b["generator_device"] != "cpu":
        raise ValueError("Guidance or initial-noise generator differs")
    consumed = status.get("checkpoint")
    if consumed and consumed["files"] != oracle["checkpoint"]["files"]:
        raise ValueError("Consumed TT and CUDA checkpoint files differ")
    report = {
        "status": "complete",
        "scope": "Post-hoc numerical review of complete independent native runs",
        "activation_injection": False,
        "configuration": {key: a[key] for key in ["variant", "prompt", "height", "width", "steps", "seed"]},
        "actual_status_sha256": sha256(actual / "status.json"),
        "reference_manifest_sha256": manifest_hash,
        "original_reference_manifest_pin_verified": True,
        "checkpoint_files_matched_in_status": bool(consumed),
        "environment": status.get("environment"),
        "placement": status.get("placement"),
        "source_files": status.get("source_files"),
        "timing": status["timings"],
        "metrics": {},
        "compared_actual_artifacts_sha256": {},
        "definitions": {
            "relative_rms": "RMSE(actual-reference) / RMS(reference)",
            "std_normalized_rmse": "RMSE(actual-reference) / population_std(reference), correction=0",
            "std_normalized_rmse_upstream": (
                "RMSE(actual-reference) / sample_std(reference), correction=1; "
                "matches tt_dit/utils/check.py with the reference as its first argument"
            ),
            "gain_offset": "Least-squares actual = gain*reference+offset, diagnostic only; no correction applied",
            "undefined_values": (
                "null denotes an undefined metric; no epsilon is silently inserted. "
                "Zero denominator plus exact equality records zero error, except sample std is always undefined for N=1."
            ),
        },
    }

    def tensor(root, name, *, verify_reference=False):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact escapes its declared run directory")
        digest = sha256(path)
        if verify_reference:
            record = oracle["artifacts"][name.removesuffix(".pt")]
            if digest != record["sha256"]:
                raise ValueError(f"CUDA artifact hash mismatch: {name}")
        else:
            report["compared_actual_artifacts_sha256"][name] = digest
        return torch.load(path, map_location="cpu", weights_only=True)

    def compare(name, actual_name, reference_name, transform=lambda x: x):
        report["metrics"][name] = error_metrics(
            tensor(actual, actual_name), transform(tensor(reference, reference_name, verify_reference=True))
        )

    noise = tensor(actual, "initial_noise.pt")
    report["independent_noise_bitwise_equal"] = torch.equal(
        noise, tensor(reference, "steps/000/input_latents.pt", verify_reference=True)
    )
    schedule = tensor(actual, "scheduler.pt")
    reference_schedule = tensor(reference, "scheduler.pt", verify_reference=True)
    report["schedule_bitwise_equal"] = all(
        torch.equal(schedule[key], reference_schedule[key]) for key in ["timesteps", "sigmas"]
    )
    if not report["independent_noise_bitwise_equal"] or not report["schedule_bitwise_equal"]:
        raise ValueError("Independent initial noise or schedule differs")
    for branch in ["positive", "negative"]:
        report["metrics"]["encoder_" + branch] = error_metrics(
            tensor(actual, f"conditioning_{branch}.pt")["hidden_states"],
            tensor(reference, f"conditioning/{branch}.pt", verify_reference=True)["hidden_states"],
        )
        compare("first_" + branch + "_prediction", f"first_{branch}_prediction.pt", f"steps/000/{branch}_prediction.pt")
    count = len(status["steps"])
    if count != len(schedule["timesteps"]) or count != len(oracle["steps"]):
        raise ValueError("Step counts differ")
    report["actual_steps"] = count
    for index in range(count):
        compare(f"latent_{index+1:03d}", f"latent_{index+1:03d}.pt", f"steps/{index:03d}/updated_latents.pt")
    compare(
        "decoded_pixels", "decoded_nhwc.pt", "vae/raw_output.pt", lambda x: x.float().clamp(-1, 1).permute(0, 2, 3, 1)
    )
    pixels = report["metrics"]["decoded_pixels"]
    report["image_accuracy"] = {
        "pcc_threshold": image_pcc_threshold,
        "pcc_threshold_met": pixels["pcc"] is not None and pixels["pcc"] >= image_pcc_threshold,
        "note": "PCC threshold alone does not bound relative RMS or establish semantic prompt compliance.",
    }
    report["actual_image_sha256"] = sha256(actual / "image.png")
    if sha256(reference / "image.png") != oracle["image"]["sha256"]:
        raise ValueError("CUDA reference image hash differs from its manifest")
    if sha256(reference / "manifest.json") != manifest_hash:
        raise ValueError("Reference manifest changed during comparison")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.actual, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["metrics"]["decoded_pixels"], indent=2))


if __name__ == "__main__":
    main()
