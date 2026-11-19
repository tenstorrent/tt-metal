# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Train a CPU-only group10/four-bit + fixed eight-bit-scale hypothesis.

The 256-entry scale reconstruction table and 16-entry weight codebook are
trained on checkpoint rows disjoint from all long-context fixture IDs. Neither
activations nor reference outputs participate in training. This is not an
implemented device format or an active-cache modification.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder_quantization import fit_codebook
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_quantization import SelectedRows


def fitted_scales(weight, codebook, group_size):
    values = weight.float().reshape(*weight.shape[:-1], 160 // group_size, group_size)
    thresholds = (codebook[:-1] + codebook[1:]) / 2
    scale = values.abs().amax(-1, keepdim=True).clamp_min(1e-20)
    for _ in range(5):
        levels = codebook[torch.bucketize(values / scale, thresholds)]
        scale = ((values * levels).sum(-1, keepdim=True) / levels.square().sum(-1, keepdim=True)).clamp_min(1e-20)
    return values, scale


def reconstruct_scale8(exact, spec):
    codebook = torch.tensor(spec["codebook_values"])
    values, scale = fitted_scales(exact.reshape(-1, 160), codebook, spec["group_size"])
    table = torch.tensor(spec["scale_codebook_values"])
    indices = torch.bucketize(scale, (table[:-1] + table[1:]) / 2)
    reconstructed_scale = table[indices]
    codes = torch.bucketize(values / reconstructed_scale.clamp_min(1e-30), (codebook[:-1] + codebook[1:]) / 2)
    result = (codebook[codes] * reconstructed_scale).bfloat16().reshape_as(exact)
    clipped = ((scale < table[0]) | (scale > table[-1])).sum().item()
    return result, clipped


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--reserve-zero", action="store_true", help="Reserve scale ID0 for exact zero groups")
    parser.add_argument("--lower-expansion", type=float, default=2.0)
    parser.add_argument("--upper-expansion", type=float, default=2.0)
    args = parser.parse_args()
    torch.set_num_threads(8)
    fixture = torch.load(args.fixture, weights_only=True, mmap=True)
    excluded = np.unique(fixture["outputs"]["ngram_ids"].numpy())
    generator = torch.Generator().manual_seed(43191)
    selected = []
    for shard in range(128):
        candidates = shard * 2500012 + torch.randperm(2500012, generator=generator)[:96]
        valid = candidates[torch.from_numpy(~np.isin(candidates.numpy(), excluded))]
        assert len(valid) >= 64
        selected.extend(valid[:64].tolist())
    snapshot = Path(fixture["identity"]["snapshot"])
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    rows = SelectedRows(snapshot, index, "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.", "absmax")
    weight = rows(torch.tensor(selected))
    codebook = fit_codebook(weight, args.group_size, False)
    _, scales = fitted_scales(weight, codebook, args.group_size)
    # Extend beyond observed extrema; quantization risk replay records clipping.
    assert args.lower_expansion >= 1 and args.upper_expansion >= 1
    low, high = float(scales.min()) / args.lower_expansion, float(scales.max()) * args.upper_expansion
    table = torch.exp(torch.linspace(np.log(low), np.log(high), 255 if args.reserve_zero else 256)).bfloat16().float()
    if args.reserve_zero:
        table = torch.cat((torch.zeros(1), table))
    assert bool((table[1:] > table[:-1]).all())
    spec = {
        "scope": __doc__,
        "quantization": "lsq",
        "lsq_iterations": 5,
        "group_size": args.group_size,
        "scales_per_row": 160 // args.group_size,
        "codebook_values": codebook.tolist(),
        "codebook_sha256": hashlib.sha256(struct.pack("<16f", *codebook.tolist())).hexdigest(),
        "scale_encoding": "uint8_fixed_log_lut",
        "scale_codebook_values": table.tolist(),
        "scale_codebook_sha256": hashlib.sha256(struct.pack("<256f", *table.tolist())).hexdigest(),
        "training_seed": 43191,
        "training_rows": selected,
        "training_rows_disjoint_from_all_fixture_hash_ids": True,
        "training_scale_min_max": [float(scales.min()), float(scales.max())],
        "scale_range_expansion": [args.lower_expansion, args.upper_expansion],
        "full_table_rows": 320001536,
        "full_table_payload_bytes": 320001536 * (80 + 160 // args.group_size),
        "codebook_payload_bytes": 16 * 4 + 256 * 2,
        "device_implementation": False,
    }
    reconstructed, clipped = reconstruct_scale8(weight, spec)
    spec["training_scale_clipped_groups"] = clipped
    spec["training_reconstruction_mse"] = (weight.float() - reconstructed.float()).square().mean().item()
    comparison = {}
    uniform = torch.linspace(low, high, 255 if args.reserve_zero else 256).bfloat16().float()
    if args.reserve_zero:
        uniform = torch.cat((torch.zeros(1), uniform))
    for name, scale_table in (("logarithmic", table), ("uniform", uniform)):
        candidate = dict(spec, scale_codebook_values=scale_table.tolist())
        values, count = reconstruct_scale8(weight, candidate)
        reconstructed_scales = scale_table[torch.bucketize(scales, (scale_table[:-1] + scale_table[1:]) / 2)]
        relative = (reconstructed_scales / scales - 1).abs()
        comparison[name] = {
            "weight_reconstruction_mse": (weight.float() - values.float()).square().mean().item(),
            "mean_relative_scale_error": relative.mean().item(),
            "maximum_relative_scale_error": relative.max().item(),
            "scale_clipped_groups": count,
        }
    spec["independent_training_scale_encoding_comparison"] = comparison
    assert "ttnn" not in sys.modules
    args.output.write_text(json.dumps(spec, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "training_rows": len(selected),
                "training_scale_min_max": spec["training_scale_min_max"],
                "full_table_payload_bytes": spec["full_table_payload_bytes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
