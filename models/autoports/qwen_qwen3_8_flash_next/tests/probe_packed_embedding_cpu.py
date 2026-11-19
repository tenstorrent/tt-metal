# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact packed/grouped-codebook parity and optional full preparation.

Does not import TTNN or open devices. --prepare-dir is an explicit expensive
full-domain setup operation; no cache is created by a parity-only invocation.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding_cpu import (
    _quantize_rows,
    prepare_checkpoint_cache,
    quantization_spec,
)


def independent_quantize(weight, spec):
    """Nearest-level reference, independent of bucketization and packed words."""
    values = weight.float().reshape(weight.shape[0], spec["scales_per_row"], spec["group_size"])
    codebook = torch.tensor(spec["codebook_values"])
    scale = values.abs().amax(-1, keepdim=True).clamp_min(1e-30 if spec["quantization"] == "absmax" else 1e-20)

    def nearest():
        divisor = scale.clamp_min(1e-30) if "scale_codebook_values" in spec else scale
        return codebook[((values / divisor)[..., None] - codebook).abs().argmin(-1)]

    if spec["quantization"] == "lsq":
        for _ in range(5):
            levels = nearest()
            scale = (
                (values * levels).sum(-1, keepdim=True) / levels.square().sum(-1, keepdim=True).clamp_min(1e-20)
            ).clamp_min(1e-20)
    if "scale_codebook_values" in spec:
        scale_table = torch.tensor(spec["scale_codebook_values"])
        scale = scale_table[(scale[..., None] - scale_table).abs().argmin(-1)]
    else:
        scale = scale.bfloat16().float()
    return (nearest() * scale).reshape(weight.shape).bfloat16()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--codebook-report", type=Path)
    parser.add_argument("--prepare-dir", type=Path)
    parser.add_argument(
        "--verify-cache-dir", type=Path, help="Read-only exact samples of a completed full cache across all128 shards"
    )
    parser.add_argument("--group-size", type=int, default=40)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    args = parser.parse_args()
    torch.set_num_threads(8)
    generator = torch.Generator().manual_seed(3031)
    weight = torch.cat((torch.zeros(1, 160), torch.randn(63, 160, generator=generator) * 0.02)).bfloat16()
    report = {"scope": "CPU packed representation parity; no device correctness claim", "cases": []}
    if args.snapshot:
        index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        real = []
        sample_locals = (0, 1, 7, 8, 31, 32, 65535, 65536, 2500011) if args.verify_cache_dir else (1017, 1018)
        for shard in range(128):
            key = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
            with safe_open(args.snapshot / index[key], framework="pt") as f:
                source = f.get_slice(key)
                real.append(torch.cat([source[position : position + 1, :].clone() for position in sample_locals]))
        weight = torch.cat((weight, *real))
        report["real_checkpoint_rows"] = 128 * len(sample_locals)
        report["real_checkpoint_shards"] = 128
    books = json.loads(args.codebook_report.read_text())["training"]["fixed_codebooks"] if args.codebook_report else {}
    specs = [quantization_spec()]
    for size in (80, 40, 32, 20, 16):
        for method in ("absmax", "lsq"):
            specs.append(quantization_spec(quantization=method, group_size=size))
            if f"group{size}_affineFalse" in books:
                specs.append(
                    quantization_spec(
                        quantization=method, group_size=size, codebook_values=books[f"group{size}_affineFalse"]
                    )
                )
    for spec in specs:
        options = {key: spec[key] for key in ("quantization", "group_size", "codebook_values")}
        packed, scales = _quantize_rows(weight, **options)
        codes = torch.stack([(packed.to(torch.int64) >> (4 * lane)) & 15 for lane in range(8)], -1).reshape(-1, 160)
        reconstructed = (
            torch.tensor(spec["codebook_values"])[codes]
            * scales.reshape(-1, spec["scales_per_row"]).repeat_interleave(spec["group_size"], -1).float()
        ).bfloat16()
        expected = independent_quantize(weight, spec)
        # Verify compact scale striping, including the final physical row of
        # the complete checkpoint without allocating its full scale buffer.
        selected_rows = torch.tensor([0, 1, 7, 8, 31, 32, 2500011, 2500012, 320001535], dtype=torch.int64)
        linear = selected_rows[:, None] * spec["scales_per_row"] + torch.arange(spec["scales_per_row"])
        address_ok = bool(((linear >> 5) * 32 + (linear & 31) == linear).all() and (linear <= 0xFFFFFFFF).all())
        entry = dict(
            spec,
            exact_equal=torch.equal(expected, reconstructed),
            max_abs_error=(expected.float() - reconstructed.float()).abs().max().item(),
            scale_address_exact=address_ok,
        )
        report["cases"].append(entry)
        assert entry["exact_equal"] and address_ok, entry
    report["passed"] = all(row["exact_equal"] and row["scale_address_exact"] for row in report["cases"])
    if args.verify_cache_dir:
        import numpy as np

        if args.snapshot is None:
            parser.error("--verify-cache-dir requires --snapshot")
        manifest = json.loads((args.verify_cache_dir / "manifest.json").read_text())
        spec = quantization_spec(
            quantization=manifest.get("quantization", "absmax"),
            group_size=manifest.get("group_size", 160),
            codebook_values=manifest.get("codebook_values"),
        )
        options = {key: spec[key] for key in ("quantization", "group_size", "codebook_values")}
        # Completed-cache reuse checks source/config identity and exact file
        # byte counts without rewriting anything.
        checked = prepare_checkpoint_cache(args.snapshot, args.verify_cache_dir, **options)
        global_rows = torch.tensor([shard * 2500012 + local for shard in range(128) for local in sample_locals])
        expected_words, expected_scales = _quantize_rows(torch.cat(real), **options)
        words = np.memmap(
            args.verify_cache_dir / "words.bin", mode="r", dtype=np.uint32, shape=(32, manifest["words_columns"])
        )
        scales = np.memmap(
            args.verify_cache_dir / "scales.bin", mode="r", dtype=np.uint16, shape=(32, manifest["scales_columns"])
        )
        linear_words = (global_rows[:, None] * 20 + torch.arange(20)).numpy()
        linear_scales = (global_rows[:, None] * spec["scales_per_row"] + torch.arange(spec["scales_per_row"])).numpy()
        observed_words = torch.from_numpy(words[linear_words % 32, linear_words // 32])
        observed_scales = torch.from_numpy(scales[linear_scales % 32, linear_scales // 32])
        verification = {
            "sample_rows": global_rows.numel(),
            "shards": 128,
            "local_row_boundaries": list(sample_locals),
            "words_exact": torch.equal(observed_words.flatten(), expected_words),
            "scales_exact": torch.equal(observed_scales.flatten(), expected_scales.view(torch.uint16)),
            "manifest": checked,
        }
        report["completed_cache_verification"] = verification
        report["passed"] &= verification["words_exact"] and verification["scales_exact"]
        assert report["passed"], verification
    if args.prepare_dir:
        if args.snapshot is None or f"group{args.group_size}_affineFalse" not in books:
            parser.error("--prepare-dir requires --snapshot and a matching learned --codebook-report")
        report["prepared_manifest"] = prepare_checkpoint_cache(
            args.snapshot,
            args.prepare_dir,
            chunk_rows=args.chunk_rows,
            quantization="lsq",
            group_size=args.group_size,
            codebook_values=books[f"group{args.group_size}_affineFalse"],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
