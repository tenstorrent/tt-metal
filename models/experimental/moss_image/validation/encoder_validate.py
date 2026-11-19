# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-device Qwen3.5 all-token PCC against pinned native CUDA captures."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
from pathlib import Path
import time

import torch

from models.experimental.moss_image.tt.checkpoint import sha256, verify_checkpoint_manifest
from models.experimental.moss_image.tt.text_encoder import TTTextEncoder


def readings(actual, reference):
    a, b = actual.double().flatten(), reference.double().flatten()
    if a.shape != b.shape or not torch.isfinite(a).all():
        raise ValueError("Invalid encoder output shape or nonfinite values")
    ac, bc = a - a.mean(), b - b.mean()
    scale = ac.norm() * bc.norm()
    return {
        "pcc": float((ac @ bc) / scale) if scale else float(torch.equal(a, b)),
        "relative_rms": float((a - b).norm() / b.norm().clamp_min(1e-30)),
        "max_abs": float((a - b).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--branch", choices=["positive", "negative", "both"], default="both")
    parser.add_argument("--dump-dir", type=Path, help="Save observed TT tensors for numerical diagnosis")
    parser.add_argument("--min-pcc", type=float, default=0.99)
    parser.add_argument("--max-relative-rms", type=float, default=0.03)
    args = parser.parse_args()
    visible = os.environ.get("TT_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", visible) or args.device_id != 0:
        raise ValueError("Set TT_VISIBLE_DEVICES to the single reserved PCI BDF and use --device-id 0")
    import ttnn as t

    manifest = json.loads((args.capture / "manifest.json").read_text())
    verify_checkpoint_manifest(args.checkpoint, manifest, components=["text_encoder", "tokenizer"])

    def load(key):
        record = manifest["artifacts"][key]
        path = (args.capture / record["path"]).resolve()
        if not path.is_relative_to(args.capture.resolve()) or sha256(path) != record["sha256"]:
            raise ValueError(f"Capture integrity failure: {key}")
        return torch.load(path, map_location="cpu", weights_only=True)

    report = {
        "status": "running",
        "activation_injection": False,
        "integrated_test": False,
        "scope": "TT text encoder from native token IDs; CUDA tensors are comparisons only",
        "checkpoint": str(args.checkpoint),
        "capture": str(args.capture),
        "cuda_manifest_sha256": sha256(args.capture / "manifest.json"),
        "thresholds": {"min_pcc": args.min_pcc, "max_relative_rms": args.max_relative_rms},
        "environment": {
            "torch": torch.__version__,
            "ttnn": importlib.metadata.version("ttnn"),
            "ttnn_module": t.__file__,
            "device_id": args.device_id,
            "visible_pci_device": visible,
            "weights": "BF16",
            "fidelity": "HiFi4",
            "fp32_dest_acc": True,
        },
        "results": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    mesh = t.open_mesh_device(
        mesh_shape=t.MeshShape(1, 1), physical_device_ids=[args.device_id], l1_small_size=32768, trace_region_size=0
    )
    try:
        encoder = TTTextEncoder(args.checkpoint, mesh)
        for branch in ["positive", "negative"] if args.branch == "both" else [args.branch]:
            inputs = load(f"text/{branch}/inputs")
            length = inputs["input_ids"].shape[1]

            def record(name, value):
                if args.dump_dir is not None:
                    folder = args.dump_dir / branch
                    folder.mkdir(parents=True, exist_ok=True)
                    torch.save(t.to_torch(value)[:, :length].contiguous(), folder / (name + ".pt"))
                if name.startswith("layer_") and name.endswith(".output"):
                    layer = int(name.split(".")[0].split("_")[1])
                    expected = load(f"text/{branch}/layers/language_model.layers.{layer}")["output"]
                elif name == "last_hidden_state":
                    expected = load(f"text/{branch}/output")
                else:
                    return
                actual = t.to_torch(value).float()[:, :length]
                metrics = readings(actual, expected)
                metrics["passed"] = metrics["pcc"] >= args.min_pcc and metrics["relative_rms"] <= args.max_relative_rms
                report["results"][branch + "/" + name] = metrics
                save()
                print(branch, name, json.dumps(metrics), flush=True)

            start = time.perf_counter()
            result = encoder.encode(inputs["input_ids"], observe=record, max_layers=args.max_layers)
            t.synchronize_device(mesh)
            report[branch + "_seconds_including_upload_and_comparison"] = time.perf_counter() - start
            t.deallocate(result)
        report["status"] = "passed" if all(v["passed"] for v in report["results"].values()) else "failed_accuracy"
        save()
        if report["status"] != "passed":
            raise RuntimeError("Encoder qualification failed its accuracy gates")
    except BaseException as error:
        if report["status"] == "running":
            report["status"] = "failed_runtime"
        report["error"] = repr(error)
        save()
        raise
    finally:
        t.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
