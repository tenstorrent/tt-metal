# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Export one DiT block's checkpoint tensors without loading the full pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def load_block_weights(checkpoint_dir: Path, layer: int) -> dict[str, torch.Tensor]:
    transformer = checkpoint_dir / "transformer"
    mapping = json.loads((transformer / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"transformer_blocks.{layer}."
    selected = {name: shard for name, shard in mapping.items() if name.startswith(prefix)}
    if not selected:
        raise ValueError(f"no weights for layer {layer}")
    weights = {}
    for shard in sorted(set(selected.values())):
        with safe_open(transformer / shard, framework="pt", device="cpu") as source:
            for name, filename in selected.items():
                if filename == shard:
                    weights[name.removeprefix(prefix)] = source.get_tensor(name)
    if len(weights) != len(selected):
        raise RuntimeError("checkpoint index and exported tensors disagree")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="resolved HF snapshot directory")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    weights = load_block_weights(args.checkpoint, args.layer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, args.output)
    print(f"exported {len(weights)} tensors for layer {args.layer} to {args.output}")


if __name__ == "__main__":
    main()
