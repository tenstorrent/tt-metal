# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stream exact checkpoint tensor statistics for the three decoder kinds."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefixes = tuple(f"model.language_model.layers.{i}." for i in (0, 1, 3))
    selected = sorted(key for key in index if key.startswith(prefixes))
    report = {"revision": args.snapshot.name, "std_correction": 0, "tensors": []}
    for key in selected:
        with safe_open(args.snapshot / index[key], framework="pt") as f:
            value = f.get_tensor(key)
            flat = value.reshape(-1)
            total = torch.zeros((), dtype=torch.float64)
            squares = torch.zeros((), dtype=torch.float64)
            for start in range(0, flat.numel(), 1048576):
                chunk = flat[start : start + 1048576].double()
                total += chunk.sum()
                squares += chunk.square().sum()
            mean = total / flat.numel()
            std = (squares / flat.numel() - mean.square()).clamp_min(0).sqrt()
            report["tensors"].append(
                {
                    "name": key,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "count": value.numel(),
                    "mean": mean.item(),
                    "std": std.item(),
                    "checkpoint_shard": index[key],
                }
            )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("WEIGHT_STATS_DONE", key, flush=True)
    print("WEIGHT_STATS_COMPLETE", len(selected), flush=True)


if __name__ == "__main__":
    main()
