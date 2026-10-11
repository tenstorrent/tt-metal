# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, bounded TP4 vision-only probe. Importing this module opens no device."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def verify_manifest(source, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Source manifest must contain file hashes")
    for name, expected in manifest["files"].items():
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"Source manifest path escapes its root: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source manifest mismatch: {name}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-aligned-case", action="store_true")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify-only", action="store_true")
    action.add_argument("--run-device", action="store_true")
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parents[4]
    manifest = verify_manifest(source, args.manifest)
    if args.verify_only:
        print(json.dumps({"source_files_verified": len(manifest["files"]), "device_opened": False}))
        return
    if args.output is None:
        parser.error("--run-device requires a new --output directory")

    import fcntl
    from types import SimpleNamespace

    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5

    import ttnn
    from models.demos.qwen38_27b_qb2.tt.generator import configure_fabric
    from models.demos.qwen38_27b_qb2.tt.vision import Qwen38VisionEncoder
    from models.tt_transformers.tt.ccl import TT_CCL

    # Same exclusive lock as profiling/serving. Fail immediately if occupied.
    with open("/tmp/tt-device.lock", "a") as device_lock:
        fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        report = {
            "state": "loading",
            "vision_only": True,
            "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "checkpoint": str(args.checkpoint.resolve()),
            "pcc_threshold": 0.99,
            "normalized_rms_threshold": 0.10,
            "cases": [],
            "device_closed": False,
        }

        def save():
            temporary = args.output / "result.json.tmp"
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.output / "result.json")

        def forbid_language_model(*unused_args, **unused_kwargs):
            raise AssertionError("The vision probe must never construct a language model")

        modeling_qwen3_5.Qwen3_5ForConditionalGeneration.__init__ = forbid_language_model
        modeling_qwen3_5.Qwen3_5TextModel.__init__ = forbid_language_model
        torch.set_num_threads(4)
        config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
        configure_fabric(topology=ttnn.Topology.Linear)
        mesh = None
        save()
        try:
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
            model = SimpleNamespace(mesh=mesh, config=config.text_config, snapshot=args.checkpoint, ccl=TT_CCL(mesh))
            started = time.perf_counter()
            encoder = Qwen38VisionEncoder(model, config)
            ttnn.synchronize_device(mesh)
            report["load_seconds"] = time.perf_counter() - started
            cases = [
                ("ragged-image", [[1, 6, 10]]),
                ("multiple-images", [[1, 4, 8], [1, 6, 10]]),
                ("video-frames", [[2, 6, 10]]),
            ]
            if args.include_aligned_case:
                cases.append(("aligned-2048", [[1, 32, 64]]))
            width = (
                config.vision_config.in_channels
                * config.vision_config.temporal_patch_size
                * config.vision_config.patch_size**2
            )
            generator = torch.Generator().manual_seed(617)
            report["state"] = "running"
            save()
            for name, grid_values in cases:
                grid = torch.tensor(grid_values, dtype=torch.int64)
                patches = int(grid.prod(-1).sum())
                pixels = torch.randn(patches, width, generator=generator) * 0.2
                with torch.inference_mode():
                    started = time.perf_counter()
                    expected = encoder.reference(pixels, grid).pooler_output.float()
                    reference_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    actual = encoder(pixels, grid).float()
                    device_seconds = time.perf_counter() - started
                if tuple(actual.shape) != tuple(expected.shape) or not torch.isfinite(actual).all():
                    raise AssertionError(f"Invalid native vision output for {name}")
                difference = actual - expected
                pcc = torch.corrcoef(torch.stack((expected.flatten(), actual.flatten())).double())[0, 1].item()
                nrms = (difference.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)).item()
                row = dict(
                    name=name,
                    grid=grid_values,
                    raw_patches=patches,
                    output_shape=list(actual.shape),
                    pcc=pcc,
                    normalized_rms=nrms,
                    max_abs_error=difference.abs().max().item(),
                    reference_seconds=reference_seconds,
                    device_seconds=device_seconds,
                    passed=pcc >= report["pcc_threshold"] and nrms <= report["normalized_rms_threshold"],
                )
                if name == "video-frames":
                    # A non-tile-aligned frame boundary must exclude the other
                    # frame at every attention layer, including padded inputs.
                    pixels[60:] += 10
                    changed = encoder(pixels, grid).float()
                    first_frame = 60 // config.vision_config.spatial_merge_size**2
                    error = (actual[:first_frame] - changed[:first_frame]).abs().max().item()
                    row["cross_frame_isolation_max_abs"] = error
                    row["passed"] = row["passed"] and error <= 1e-3
                report["cases"].append(row)
                save()
                print(json.dumps(row), flush=True)
                if not row["passed"]:
                    raise AssertionError(f"Native vision reference gate failed: {name}")
            report["state"] = "passed"
        except BaseException as error:
            report["state"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
                report["device_closed"] = True
            save()


if __name__ == "__main__":
    main()
