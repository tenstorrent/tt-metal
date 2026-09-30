# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare text-only Qwen3-VL prompt encoding on real TT against a CUDA capture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from models.experimental.qwen_image_2_1.validation.tt_vae_decode import metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-bdf", required=True)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument(
        "--captured-input-ids",
        action="store_true",
        help="diagnostic input; default tokenizes raw prompt locally",
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.cuda_dir / "manifest.json").read_text())
    os.environ["TT_VISIBLE_DEVICES"] = args.device_bdf
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)
    import ttnn

    from models.experimental.qwen_image_2_1.tt.tt_text_encoder import QwenImage21TextEncoder

    reports = []

    def progress(status, stage=None, error=None):
        temporary = args.output_dir / "progress.json.tmp"
        temporary.write_text(json.dumps({"status": status, "stage": stage, "error": error}, indent=2) + "\n")
        temporary.replace(args.output_dir / "progress.json")

    def compare(name, value):
        progress("running", name)
        expected_path = (
            args.cuda_dir / f"{name}/output.pt"
            if name.startswith("layer_") and "/" not in name
            else args.cuda_dir / f"{name}.pt"
        )
        if not expected_path.is_file():
            return
        expected = torch.load(expected_path, weights_only=True, map_location="cpu")
        actual = ttnn.to_torch(value).reshape(expected.shape).contiguous()
        row = {"stage": name, **metrics(actual, expected)}
        reports.append(row)
        torch.save(actual, args.output_dir / f"{name.replace('/', '_')}.pt")
        (args.output_dir / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    device = None
    try:
        progress("starting")
        device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 1),
            physical_device_ids=[0],
            l1_small_size=32768,
        )
        encoder = QwenImage21TextEncoder(args.checkpoint, device)
        captured_ids = torch.load(args.cuda_dir / "input_ids.pt", map_location="cpu", weights_only=True)
        from models.experimental.qwen_image_2_1.tt.text_prompt import tokenize_prompt

        ids, drop_idx = tokenize_prompt(args.checkpoint, manifest["prompt"])
        if not torch.equal(ids, captured_ids) or drop_idx != manifest["drop_idx"]:
            raise AssertionError("raw prompt tokenization differs from CUDA reference")
        if args.captured_input_ids:
            ids = captured_ids
        output = encoder.encode(ids, manifest["drop_idx"], compare, args.max_layers)
        actual = ttnn.to_torch(output).contiguous()
        torch.save(actual, args.output_dir / "prompt_embeds.pt")
        if args.max_layers is None or args.max_layers == 36:
            compare("prompt_embeds", output)
            if reports[-1]["stage"] != "prompt_embeds" or not reports[-1]["pcc"] >= 0.99:
                raise AssertionError("final prompt embedding PCC below 0.99 or missing comparison")
        progress("complete" if args.max_layers in (None, 36) else "partial", "prompt_embeds")
    except Exception as error:
        progress("failed", error=str(error))
        raise
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
