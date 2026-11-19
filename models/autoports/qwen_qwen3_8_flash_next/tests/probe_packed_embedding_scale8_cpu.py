# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact group10 scale-byte packing parity and full-cache preparation."""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_packed_embedding_cpu import independent_quantize
from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding_cpu import (
    _quantize_rows,
    _round32,
    prepare_checkpoint_cache,
    quantization_spec,
)


def stripe(values):
    padded = torch.zeros(32 * _round32((values.numel() + 31) // 32), dtype=torch.uint32)
    padded[: values.numel()] = values
    return padded.reshape(-1, 32).T.contiguous().numpy()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-dir", type=Path)
    parser.add_argument("--verify-cache-dir", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    args = parser.parse_args()
    torch.set_num_threads(8)
    selected = json.loads(args.spec.read_text())
    options = {key: selected[key] for key in ("quantization", "group_size", "codebook_values", "scale_codebook_values")}
    spec = quantization_spec(**options)
    generator = torch.Generator().manual_seed(3031)
    weight = torch.cat((torch.zeros(1, 160), torch.randn(63, 160, generator=generator) * 0.02)).bfloat16()
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    real, global_ids = [], []
    local_ids = (0, 1, 7, 8, 31, 32, 65535, 65536, 2500011)
    for shard in range(128):
        key = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        with safe_open(args.snapshot / index[key], framework="pt") as source:
            view = source.get_slice(key)
            real.append(torch.cat([view[local : local + 1].clone() for local in local_ids]))
        global_ids.extend(shard * 2500012 + local for local in local_ids)
    weight = torch.cat((weight, *real))
    statistics = {}
    words, scales = _quantize_rows(weight, **options, statistics=statistics)
    codes = torch.stack([(words.long() >> (lane * 4)) & 15 for lane in range(8)], -1).reshape(-1, 160)
    scale_ids = torch.stack([(scales.long() >> (lane * 8)) & 255 for lane in range(4)], -1).reshape(-1, 16)
    reconstructed = (
        torch.tensor(spec["codebook_values"])[codes]
        * torch.tensor(spec["scale_codebook_values"])[scale_ids].repeat_interleave(10, -1)
    ).bfloat16()
    expected = independent_quantize(weight, spec)
    assert torch.equal(expected, reconstructed)
    assert torch.equal(reconstructed[0], weight[0]), "Zero scale entry must preserve zero weights"
    addressing = torch.tensor([0, 1, 7, 8, 31, 32, 2500011, 2500012, 268435455, 268435456, 320001535])
    scale_linear = addressing[:, None] * 4 + torch.arange(4)
    assert (scale_linear <= 0xFFFFFFFF).all()
    assert torch.equal((scale_linear >> 5) * 32 + (scale_linear & 31), scale_linear)
    small_cache_equal = {}
    with tempfile.TemporaryDirectory(prefix="ple_scale8_cpu_") as root:
        for name, values, per_row in (("words", words, 20), ("scales", scales, 4)):
            striped = stripe(values)
            path = Path(root) / f"{name}.bin"
            striped.tofile(path)
            saved = np.memmap(path, mode="r", dtype=np.uint32, shape=striped.shape)
            offsets = np.arange(weight.shape[0])[:, None] * per_row + np.arange(per_row)
            restored = torch.from_numpy(saved[offsets % 32, offsets // 32])
            small_cache_equal[name] = torch.equal(restored.flatten(), values)
            assert small_cache_equal[name]
    report = {
        "scope": __doc__,
        "quantizer": spec,
        "real_rows": len(global_ids),
        "shards": 128,
        "local_boundaries": list(local_ids),
        "independent_reconstruction_exact": True,
        "zero_row_exact": True,
        "scale_address_maximum": int(scale_linear.max()),
        "scale_address_exact": True,
        "small_serialized_cache_exact": small_cache_equal,
        "sample_quantization_statistics": statistics,
        "device_executed": False,
    }
    if args.verify_cache_dir:
        manifest = prepare_checkpoint_cache(args.snapshot, args.verify_cache_dir, **options)
        from transformers import AutoConfig
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

        config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
        with torch.device("meta"):
            embedding = Qwen4ExpTextNGramEmbedding(config, 2560, 1, 0)
        valid_rows = embedding.total_vocab_size
        padding_stats = {}
        if valid_rows < manifest["rows"]:
            shard, first = divmod(valid_rows, 2500012)
            assert shard == 127, "Padding-domain audit expects padding confined to the final shard"
            key = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
            with safe_open(args.snapshot / index[key], framework="pt") as source:
                padding = source.get_slice(key)[first:, :].clone()
            _quantize_rows(padding, **options, statistics=padding_stats)
        total_stats = manifest["quantization_statistics"]
        report["hash_domain_scale_range_audit"] = {
            "valid_hash_domain_rows": valid_rows,
            "padded_checkpoint_rows": manifest["rows"],
            "padding_statistics": padding_stats,
            "below_range_groups_in_valid_hash_domain": (
                total_stats["below_positive_scale_lut_minimum"]
                - padding_stats.get("below_positive_scale_lut_minimum", 0)
            ),
            "above_range_groups_in_valid_hash_domain": (
                total_stats["above_scale_lut_maximum"] - padding_stats.get("above_scale_lut_maximum", 0)
            ),
        }
        hashes = {}
        for name, values, per_row in (("words", words[64 * 20 :], 20), ("scales", scales[64 * 4 :], 4)):
            saved = np.memmap(
                args.verify_cache_dir / f"{name}.bin",
                mode="r",
                dtype=np.uint32,
                shape=(32, manifest[f"{name}_columns"]),
            )
            offsets = np.array(global_ids)[:, None] * per_row + np.arange(per_row)
            actual = torch.from_numpy(saved[offsets % 32, offsets // 32])
            assert torch.equal(values, actual.flatten()), name
            with (args.verify_cache_dir / f"{name}.bin").open("rb") as source:
                hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
        changed = list(options["scale_codebook_values"])
        changed[128] = float(torch.tensor((changed[128] + changed[129]) / 2).bfloat16())
        assert options["scale_codebook_values"][128] < changed[128] < changed[129]
        try:
            prepare_checkpoint_cache(
                args.snapshot, args.verify_cache_dir, **dict(options, scale_codebook_values=changed)
            )
        except ValueError as error:
            assert "Cache identity mismatch" in str(error)
        else:
            raise AssertionError("A different scale LUT reused the completed cache")
        report["full_cache_verified_rows"] = len(global_ids)
        report["full_cache_payload_sha256"] = hashes
        report["manifest_sha256"] = hashlib.sha256((args.verify_cache_dir / "manifest.json").read_bytes()).hexdigest()
        report["changed_scale_lut_rejected"] = True
        report["verified_manifest"] = manifest
    if args.prepare_dir:
        report["prepared_manifest"] = prepare_checkpoint_cache(
            args.snapshot, args.prepare_dir, chunk_rows=args.chunk_rows, **options
        )
    assert "ttnn" not in sys.modules
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
