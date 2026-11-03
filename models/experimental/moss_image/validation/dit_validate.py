# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CUDA-capture-driven component qualification; deliberately not integrated T2I."""
from __future__ import annotations
import argparse
import os
import json
import importlib.metadata
import platform
from pathlib import Path
import time
import torch
from models.experimental.moss_image.tt.dit import TTMossDiT
from models.experimental.moss_image.tt.checkpoint import sha256, PIPELINE_SHA256, verify_checkpoint_manifest


def readings(actual, reference):
    a, b = actual.double().reshape(-1), reference.double().reshape(-1)
    if a.shape != b.shape or not torch.isfinite(a).all():
        raise ValueError(f"Nonfinite output or shape mismatch: {actual.shape} vs {reference.shape}")
    if not torch.isfinite(b).all():
        raise ValueError("Reference contains nonfinite values")
    ac, bc = a - a.mean(), b - b.mean()
    scale = torch.linalg.vector_norm(ac) * torch.linalg.vector_norm(bc)
    return {
        "pcc": float((torch.dot(ac, bc) / scale).clamp(-1, 1)) if scale else float(torch.equal(a, b)),
        "relative_rms": float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-30)),
        "max_absolute_error": float((a - b).abs().max()),
    }


def validate_steps(checkpoint, capture, mesh_device, output, step_indices=(0, 10, 25, 40, 49)):
    """Teacher-force native CUDA latents through one reusable TT DiT instance.

    This diagnoses local step error independently of accumulated trajectory drift.
    The caller owns the mesh/fabric. No captured reference is used as TT output.
    """
    import ttnn as t

    checkpoint, capture, output = Path(checkpoint), Path(capture), Path(output)
    oracle = json.loads((capture / "manifest.json").read_text())
    if oracle.get("pipeline_sha256") != PIPELINE_SHA256:
        raise ValueError("Capture does not identify the pinned official pipeline")
    verified = verify_checkpoint_manifest(checkpoint, oracle, components=["transformer"])
    report = {
        "activation_injection": True,
        "integrated_test": False,
        "scope": "Teacher-forced complete DiT forward at selected CUDA trajectory steps",
        "checkpoint": str(checkpoint),
        "capture": str(capture),
        "cards": mesh_device.get_num_devices(),
        "verified_checkpoint_files": verified,
        "results": {},
        "status": "running",
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    def load(name):
        record = oracle["artifacts"][name]
        path = (capture / record["path"]).resolve()
        if not path.is_relative_to(capture.resolve()) or sha256(path) != record["sha256"]:
            raise ValueError(f"Capture artifact integrity failure: {name}")
        return torch.load(path, map_location="cpu", weights_only=True)

    save()
    try:
        model = TTMossDiT(checkpoint, mesh_device)
        for index in step_indices:
            if index < 0:
                raise ValueError("Step index must be nonnegative")
            latent = load(f"steps/{index:03d}/input_latents")
            _, _, height, width = latent.shape
            for branch in ("positive", "negative"):
                cond = load("conditioning/" + branch)
                context = cond["hidden_states"][:, cond["attention_mask"][0].bool()]
                stamp = time.perf_counter()
                actual = model(
                    model.upload(latent.flatten(2).transpose(1, 2), dtype=t.float32),
                    float(load(f"steps/{index:03d}/timestep").reshape(-1)[0]),
                    model.upload(context),
                    height,
                    width,
                    skip_layers=cond["skip_layers"],
                )
                actual = model.host(actual).transpose(1, 2).reshape_as(latent)
                metrics = readings(actual, load(f"steps/{index:03d}/{branch}_prediction"))
                metrics["seconds_including_weight_upload_and_comparison"] = time.perf_counter() - stamp
                report["results"][f"step_{index:03d}/{branch}"] = metrics
                save()
                print(f"Teacher-forced step {index} {branch}: {json.dumps(metrics)}", flush=True)
        report["status"] = "complete"
        save()
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        save()
        raise
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--capture", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device-ids", type=int, nargs="+", required=True)
    p.add_argument("--case", choices=["blocks", "step"], default="blocks")
    p.add_argument("--step-index", type=int, default=0, help="Teacher-forced step from the native CUDA trajectory")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 21, 41])
    p.add_argument("--min-pcc", type=float, default=0.99)
    p.add_argument("--max-relative-rms", type=float, default=0.05)
    args = p.parse_args()
    if args.step_index < 0 or (args.case == "blocks" and args.step_index != 0):
        p.error("Block captures exist only at step 0; step index must be nonnegative")
    if not os.environ.get("TT_VISIBLE_DEVICES"):
        p.error("Set TT_VISIBLE_DEVICES to the reserved PCI BDF(s) before hardware initialization")
    import ttnn as t

    capture = Path(args.capture)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "activation_injection": True,
        "integrated_test": False,
        "cards": len(args.device_ids),
        "step_index": args.step_index,
        "checkpoint": args.checkpoint,
        "capture": str(capture),
        "results": {},
        "status": "running",
    }
    report["environment"] = {
        "python": platform.python_version(),
        "tt_visible_devices": os.environ["TT_VISIBLE_DEVICES"],
        "torch": importlib.metadata.version("torch"),
        "ttnn": importlib.metadata.version("ttnn"),
        "weights": "BF16",
        "math_fidelity": "HiFi4",
        "fp32_destination_accumulation": True,
    }
    source_root = Path(__file__).resolve().parents[1]
    report["source_files"] = {name: sha256(source_root / name) for name in ("tt/dit.py", "validation/dit_validate.py")}

    def save():
        out.write_text(json.dumps(report, indent=2) + "\n")

    def load(name):
        path = capture / (name + ".pt")
        if sha256(path) != oracle["artifacts"][name]["sha256"]:
            raise ValueError(f"Capture artifact checksum mismatch: {name}")
        return torch.load(path, map_location="cpu", weights_only=True)

    save()
    oracle = json.loads((capture / "manifest.json").read_text())
    if oracle.get("pipeline_sha256") != PIPELINE_SHA256:
        raise ValueError("Capture does not identify the pinned official pipeline")
    verified = verify_checkpoint_manifest(Path(args.checkpoint), oracle, components=["transformer"])
    report["verified_checkpoint_files"] = verified
    report["oracle_variant"] = oracle["checkpoint"]["declared_variant"]
    save()
    mesh = None
    try:
        if len(args.device_ids) == 2:
            t.set_fabric_config(t.FabricConfig.FABRIC_1D)
        mesh = t.open_mesh_device(
            mesh_shape=t.MeshShape(1, len(args.device_ids)),
            physical_device_ids=args.device_ids,
            l1_small_size=131072,
            trace_region_size=0,
        )
        model = TTMossDiT(args.checkpoint, mesh)

        def record(name, value, ref, seconds):
            result = readings(value, ref)
            result["seconds_including_weight_upload"] = seconds
            result["passed"] = result["pcc"] >= args.min_pcc and result["relative_rms"] <= args.max_relative_rms
            report["results"][name] = result
            save()
            print(name, json.dumps(result), flush=True)

        for branch in ["positive", "negative"]:
            if args.case == "blocks":
                for layer in args.layers:
                    values = load(f"blocks/{branch}/{layer:02d}/inputs")
                    x = model.upload(values["x"])
                    mod = values["modulation"].unbind(1)
                    mod = tuple(model.upload(v.unsqueeze(1)) for v in mod)
                    start = time.perf_counter()
                    y = model.block(x, layer, values["rope"], mod)
                    host = model.host(y)
                    record(
                        f"{branch}/block_{layer:02d}",
                        host,
                        load(f"blocks/{branch}/{layer:02d}/output"),
                        time.perf_counter() - start,
                    )
            else:
                latent = load(f"steps/{args.step_index:03d}/input_latents")
                _, _, h, w = latent.shape
                cond = load("conditioning/" + branch)
                ctx = cond["hidden_states"][:, cond["attention_mask"][0].bool()]
                x = model.upload(latent.flatten(2).transpose(1, 2), dtype=t.float32)
                start = time.perf_counter()

                def observe(name, value):
                    if args.step_index == 0 and name.startswith("block_") and int(name[-2:]) in args.layers:
                        ref = load(f"blocks/{branch}/{int(name[-2:]):02d}/output")
                        report["results"][branch + "/" + name] = readings(model.host(value), ref)
                        save()

                y = model(
                    x,
                    float(load(f"steps/{args.step_index:03d}/timestep").reshape(-1)[0]),
                    model.upload(ctx),
                    h,
                    w,
                    skip_layers=cond["skip_layers"],
                    observe=observe,
                )
                ref = load(f"steps/{args.step_index:03d}/{branch}_prediction").flatten(2).transpose(1, 2)
                record(branch + "/velocity", model.host(y), ref, time.perf_counter() - start)
        report["status"] = (
            "passed" if all(v.get("passed", True) for v in report["results"].values()) else "failed_accuracy"
        )
        save()
        if report["status"] != "passed":
            raise RuntimeError("DiT accuracy qualification failed; see report")
    except BaseException as exc:
        report["error"] = repr(exc)
        if report["status"] == "running":
            report["status"] = "failed_runtime"
        save()
        raise
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
        if len(args.device_ids) == 2:
            t.set_fabric_config(t.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
