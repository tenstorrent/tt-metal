# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file

from models.demos.unet_3d.demo.utils import configure_logging

logger = configure_logging()

# LINK = "https://oc.embl.de/index.php/s/61s67Mg5VQy7dh9/download?path=%2FLateral-Root-Primordia%2Funet_bce_dice_nuclei_ds1x&files=best_checkpoint.pytorch"

LINK = "https://oc.embl.de/index.php/s/61s67Mg5VQy7dh9/download?path=%2FArabidopsis-Ovules%2Funet_bce_dice_ds2x&files=best_checkpoint.pytorch"


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    with urllib.request.urlopen(url) as response, open(output_path, "wb") as f:
        f.write(response.read())


def extract_state_dict(checkpoint_obj: object) -> dict:
    if isinstance(checkpoint_obj, dict):
        if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
            return checkpoint_obj["state_dict"]
        if "model_state_dict" in checkpoint_obj and isinstance(checkpoint_obj["model_state_dict"], dict):
            return checkpoint_obj["model_state_dict"]
        if "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], dict):
            return checkpoint_obj["model"]
        if all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError("Unsupported checkpoint structure; no tensor state_dict found.")


def sanitize_state_dict(state_dict: dict) -> dict:
    tensors = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            tensors[key] = value.detach().cpu()
    if not tensors:
        raise ValueError("No tensor entries found in state_dict.")
    return tensors


def remap_keys_for_torch_impl(tensors: dict) -> dict:
    remapped = {}
    unmapped = []

    for key, value in tensors.items():
        if key.startswith("encoders.") or key.startswith("decoders."):
            parts = key.split(".")
            if len(parts) >= 6 and parts[2] == "basic_module" and parts[3].startswith("SingleConv"):
                block_suffix = parts[3][-1]
                block_idx = "1" if block_suffix == "1" else "2"
                submodule = parts[4]
                param = parts[5]
                if parts[0] == "encoders" and parts[1] == "3":
                    prefix = "bottleneck"
                else:
                    prefix = f"{parts[0]}.{parts[1]}"
                if submodule == "conv":
                    new_key = f"{prefix}.conv_block_{block_idx}.conv.{param}"
                elif submodule == "groupnorm":
                    new_key = f"{prefix}.conv_block_{block_idx}.norm.{param}"
                else:
                    unmapped.append(key)
                    continue
                remapped[new_key] = value
                continue
        if key.startswith("final_conv."):
            remapped[key] = value
        else:
            unmapped.append(key)

    if unmapped:
        logger.warning("Unmapped keys were skipped:")
        for key in sorted(unmapped):
            logger.warning(key)

    # Add missing conv biases expected by torch_impl Conv3d layers.
    for key in list(remapped.keys()):
        if key.endswith(".conv.weight"):
            bias_key = key.replace(".conv.weight", ".conv.bias")
            if bias_key not in remapped:
                weight = remapped[key]
                remapped[bias_key] = torch.zeros(weight.shape[0], dtype=weight.dtype)

    return remapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UNet3D checkpoint and convert to safetensors.")
    parser.add_argument("--url", default=LINK, help="Checkpoint URL to download.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "models"),
        help="Directory to store the downloaded checkpoint and safetensors file.",
    )
    parser.add_argument(
        "--safetensors-name",
        default="confocal_boundary.safetensors",
        help="Filename for the output safetensors file.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "best_checkpoint.pytorch"
    safetensors_path = output_dir / args.safetensors_name

    download_file(args.url, checkpoint_path)
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint_obj)

    logger.info("Extracted %s tensors from checkpoint:", len(state_dict))
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if torch.is_tensor(tensor):
            logger.info("%s: shape=%s, dtype=%s", key, tuple(tensor.shape), tensor.dtype)
        else:
            logger.info("%s: NOT A TENSOR", key)
    tensors = sanitize_state_dict(state_dict)
    tensors = remap_keys_for_torch_impl(tensors)

    logger.info("Found %s tensors after remap:", len(tensors))
    for key in sorted(tensors.keys()):
        logger.info(key)

    save_file(tensors, str(safetensors_path))
    logger.info("Saved safetensors to: %s", safetensors_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint file: %s", checkpoint_path)


if __name__ == "__main__":
    main()
