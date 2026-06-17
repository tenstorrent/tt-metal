#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Download CosyVoice pretrained weights.

Downloads the CosyVoice-300M model weights from HuggingFace
and converts them to the format expected by the TTNN pipeline.

Usage:
    python models/demos/wormhole/cosyvoice/scripts/download_weights.py
    python models/demos/wormhole/cosyvoice/scripts/download_weights.py --model_version CosyVoice-300M
    python models/demos/wormhole/cosyvoice/scripts/download_weights.py --output /tmp/tt-metal-weights/cosyvoice
"""

import argparse
import os
import sys

from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Download CosyVoice weights")
    parser.add_argument("--model_version", type=str, default="CosyVoice-300M",
                        choices=["CosyVoice-300M", "CosyVoice-300M-Safe", "CosyVoice-300M-25Hz"],
                        help="CosyVoice model version")
    parser.add_argument("--output", type=str, default="/tmp/tt-metal-weights/cosyvoice",
                        help="Output directory for weights")
    parser.add_argument("--skip_if_exists", action="store_true",
                        help="Skip download if output already exists")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output
    weight_path = os.path.join(output_dir, "cosyvoice.pt")

    # Check if weights already exist
    if args.skip_if_exists and os.path.exists(weight_path):
        logger.info(f"Weights already exist at {weight_path}, skipping download")
        return

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Downloading CosyVoice ({args.model_version}) weights...")

    if args.model_version == "CosyVoice-300M":
        repo_id = "FunAudioLLM/CosyVoice-300M"
    elif args.model_version == "CosyVoice-300M-Safe":
        repo_id = "FunAudioLLM/CosyVoice-300M-Safe"
    elif args.model_version == "CosyVoice-300M-25Hz":
        repo_id = "FunAudioLLM/CosyVoice-300M-25Hz"
    else:
        raise ValueError(f"Unknown model version: {args.model_version}")

    try:
        from huggingface_hub import snapshot_download

        logger.info(f"Downloading from HuggingFace: {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )

        # Merge component checkpoints into a single state dict
        logger.info("Merging component checkpoints...")
        _merge_checkpoints(output_dir, weight_path)

        logger.info(f"Weights saved to {weight_path}")
        logger.info(f"Total size: {sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fns in os.walk(output_dir) for f in fns) / 1e9:.2f}GB")

    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to download weights: {e}")
        logger.info("To use the model without pretrained weights, run the demo with random initialization")
        logger.info("The model will still demonstrate the correct architecture and pipeline flow.")
        sys.exit(1)


def _merge_checkpoints(input_dir: str, output_path: str):
    """Merge individual component checkpoints into a single state dict.

    CosyVoice distributes weights as separate files for LLM, flow, and HiFi-GAN.
    This function merges them into one state dict for the TTNN pipeline.
    """
    import torch

    state_dict = {}

    # Expected component checkpoint files
    component_files = {
        "llm": os.path.join(input_dir, "llm.pt"),
        "flow": os.path.join(input_dir, "flow.pt"),
        "hifigan": os.path.join(input_dir, "hifigan.pt"),
    }

    for component_name, filepath in component_files.items():
        if os.path.exists(filepath):
            logger.info(f"Loading {component_name} from {filepath}")
            component_sd = torch.load(filepath, map_location="cpu", weights_only=True)
            for key, value in component_sd.items():
                state_dict[f"{component_name}.{key}"] = value
        else:
            logger.warning(f"{component_name} checkpoint not found at {filepath}")

    # Also try loading as a single combined checkpoint
    combined_path = os.path.join(input_dir, "cosyvoice.pt")
    if os.path.exists(combined_path):
        logger.info(f"Loading combined checkpoint from {combined_path}")
        combined_sd = torch.load(combined_path, map_location="cpu", weights_only=True)
        state_dict.update(combined_sd)

    if state_dict:
        torch.save(state_dict, output_path)
        logger.info(f"Merged state dict with {len(state_dict)} keys saved to {output_path}")
    else:
        logger.warning("No checkpoint files found to merge. Creating placeholder.")
        logger.info("Place the pretrained weights in the following locations:")
        logger.info(f"  {component_files['llm']}")
        logger.info(f"  {component_files['flow']}")
        logger.info(f"  {component_files['hifigan']}")
        logger.info("Or download from: https://huggingface.co/FunAudioLLM")


if __name__ == "__main__":
    main()
