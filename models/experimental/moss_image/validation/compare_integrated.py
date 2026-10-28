# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Post-hoc comparison: never supplies a reference activation to TT generation."""
import argparse
import json
from pathlib import Path
import torch
from models.experimental.moss_image.validation.dit_validate import readings
from models.experimental.moss_image.tt.checkpoint import sha256


def compare(actual: Path, reference: Path):
    def load(root, name):
        return torch.load(root / name, map_location="cpu", weights_only=True)

    status = json.loads((actual / "status.json").read_text())
    oracle = json.loads((reference / "manifest.json").read_text())
    if status["status"] != "complete" or oracle["status"] != "complete":
        raise ValueError("Both runs must be complete")
    a, b = status["arguments"], oracle["arguments"]
    for key in ["prompt", "negative_prompt", "height", "width", "steps", "seed", "variant"]:
        if a[key] != b[key]:
            raise ValueError(f"Integrated comparison configuration mismatch: {key}")
    if b["generator_device"] != "cpu":
        raise ValueError("Reference noise generator must be CPU for matching independent RNG")
    report = {
        "status": "complete",
        "activation_injection": False,
        "actual_run": str(actual),
        "reference_run": str(reference),
        "configuration": {k: a[k] for k in ["variant", "prompt", "height", "width", "steps", "seed"]},
        "timing": status["timings"],
        "results": {},
        "note": "Measurements of independent complete runs. Final-image metrics are reported, without a tuned visual-quality threshold.",
    }
    if a["guidance_scale"] != oracle["guidance_scale"]:
        raise ValueError("Guidance scale differs between integrated runs")
    from models.experimental.moss_image.tt.checkpoint import checkpoint_manifest

    actual_weights = checkpoint_manifest(Path(a["checkpoint"]).expanduser().resolve(), a["variant"])
    if actual_weights["files"] != oracle["checkpoint"]["files"]:
        raise ValueError("Consumed checkpoint files differ from CUDA reference")
    report["checkpoint_files"] = actual_weights["files"]
    noise = load(actual, "initial_noise.pt")
    ref = load(reference, "steps/000/input_latents.pt")
    report["independent_noise_bitwise_equal"] = torch.equal(noise, ref)
    for branch in ["positive", "negative"]:
        x = load(actual, f"conditioning_{branch}.pt")["hidden_states"]
        ref = load(reference, f"conditioning/{branch}.pt")["hidden_states"]
        report["results"]["encoder_" + branch] = readings(x, ref)
        report["results"]["step0_" + branch + "_prediction"] = readings(
            load(actual, f"first_{branch}_prediction.pt"), load(reference, f"steps/000/{branch}_prediction.pt")
        )
    schedule = load(actual, "scheduler.pt")
    reference_schedule = load(reference, "scheduler.pt")
    report["schedule_bitwise_equal"] = all(
        torch.equal(schedule[k], reference_schedule[k]) for k in ["timesteps", "sigmas"]
    )
    actual_step_count = len(status["steps"])
    if actual_step_count != len(schedule["timesteps"]) or actual_step_count != len(oracle["steps"]):
        raise ValueError("Actual denoising step counts differ from the saved schedule/reference")
    report["actual_steps"] = actual_step_count
    for index in range(actual_step_count):
        report["results"][f"latent_{index+1:03d}"] = readings(
            load(actual, f"latent_{index+1:03d}.pt"), load(reference, f"steps/{index:03d}/updated_latents.pt")
        )
    image = load(actual, "decoded_nhwc.pt")
    ref = load(reference, "vae/raw_output.pt").float().clamp(-1, 1).permute(0, 2, 3, 1).contiguous()
    report["results"]["decoded_pixels"] = readings(image, ref)
    report["actual_image_sha256"] = sha256(actual / "image.png")
    report["reference_image_sha256"] = oracle["image"]["sha256"]
    report["passed_prerequisites"] = bool(
        report["independent_noise_bitwise_equal"]
        and report["schedule_bitwise_equal"]
        and all(
            report["results"]["encoder_" + x]["pcc"] >= 0.99
            and report["results"]["encoder_" + x]["relative_rms"] <= 0.03
            for x in ["positive", "negative"]
        )
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actual", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = compare(Path(args.actual), Path(args.reference))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"status": report["status"], "decoded_pixels": report["results"]["decoded_pixels"]}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
