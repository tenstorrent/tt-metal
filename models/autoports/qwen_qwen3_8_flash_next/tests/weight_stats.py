# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Streaming exact real-weight statistics and deterministic full-shape CI weights.

No TTNN import/device access. Example:
python_env/bin/python models/autoports/qwen_qwen3_8_flash_next/tests/weight_stats.py \
  --layers 0 3 1 --output-dir models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/weight_stats
"""

import argparse
import hashlib
import json
import math
import os
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_CHECKPOINT = (
    "/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
)


def tensor_statistics(handle, key, *, chunk_bytes=16 * 1024 * 1024):
    """Population mean/std from every element, using FP64 parallel Welford.

    Integer buffers are exact model constants, not random trainable weights;
    preserve their values for the synthetic mapping.
    """
    source = handle.get_slice(key)
    shape = source.get_shape()
    dtype = source.get_dtype()
    n = math.prod(shape)
    if dtype not in ("BF16", "F16", "F32", "F64"):
        value = handle.get_tensor(key)
        return {
            "shape": shape,
            "dtype": dtype,
            "numel": n,
            "mean": value.double().mean().item(),
            "std": value.double().std(correction=0).item(),
            "exact_values": value.tolist(),
        }
    row_elements = math.prod(shape[1:]) if shape else 1
    rows = max(1, chunk_bytes // (8 * row_elements))
    count, mean, m2 = 0, 0.0, 0.0
    for start in range(0, shape[0] if shape else 1, rows):
        x = (source[start : min(start + rows, shape[0])] if shape else handle.get_tensor(key)).double()
        variance, local_mean = torch.var_mean(x, correction=0)
        size, local_mean = x.numel(), local_mean.item()
        new_count = count + size
        delta = local_mean - mean
        m2 += variance.item() * size + delta * delta * count * size / new_count
        mean += delta * size / new_count
        count = new_count
    assert count == n
    return {"shape": shape, "dtype": dtype, "numel": n, "mean": mean, "std": math.sqrt(m2 / n)}


def collect_layer(checkpoint, layer_idx, output):
    checkpoint = Path(checkpoint)
    prefix = f"model.language_model.layers.{layer_idx}."
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    keys = {k: v for k, v in index.items() if k.startswith(prefix)}
    assert keys, prefix
    result = {
        "model": "Qwen/Qwen3.8-Flash-Next",
        "revision": checkpoint.name,
        "layer_idx": layer_idx,
        "method": "all elements; FP64 streaming Welford; population std",
        "text_config": json.loads((checkpoint / "config.json").read_text())["text_config"],
        "tensors": {},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for filename in sorted(set(keys.values())):
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as handle:
            for key in sorted(k for k, f in keys.items() if f == filename):
                record = tensor_statistics(handle, key)
                record["checkpoint_key"] = key
                result["tensors"][key[len(prefix) :]] = record
                print(json.dumps({"layer": layer_idx, "key": key, **record}), flush=True)
    result["complete"] = len(result["tensors"]) == len(keys)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


class SyntheticState(Mapping):
    """Lazy deterministic full target shapes from measured weight moments.

    Each key has an independent seed; PLE tables additionally seed each 32-row
    block so sparse reference lookup and full TT upload produce identical data.
    Access order cannot change values. Normal weights use recorded mean/std and
    integer buffers retain exact values. This is diagnostic CI input, never a
    substitute for real-weight PCC.
    """

    def __init__(self, stats_path, seed=0):
        self.stats = json.loads(Path(stats_path).read_text())
        if not self.stats.get("complete"):
            raise ValueError("Synthetic weights require complete real-weight statistics")
        self.records = self.stats["tensors"]
        self.seed = seed
        self._row_blocks = OrderedDict()

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, key):
        record = self.records[key]
        dtype = self._dtype(record)
        if "exact_values" in record:
            return torch.tensor(record["exact_values"], dtype=dtype).reshape(record["shape"])
        if "ngram_embedding.shard_" in key:
            # The full TT setup tensor and sparse HF oracle reads share exactly
            # the same independently seeded 32-row blocks. Allocating a shard
            # does not require an additional full FP32 copy.
            output = torch.empty(record["shape"], dtype=dtype)
            for start in range(0, record["shape"][0], 32):
                stop = min(start + 32, record["shape"][0])
                output[start:stop] = self._block(key, start // 32, cache=False)
            return output
        digest = hashlib.sha256(f"{self.seed}/{key}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
        x = torch.randn(record["shape"], generator=generator, dtype=torch.float32)
        return x.mul_(record["std"]).add_(record["mean"]).to(dtype)

    @staticmethod
    def _dtype(record):
        return {
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "F32": torch.float32,
            "F64": torch.float64,
            "I64": torch.int64,
            "I32": torch.int32,
            "U32": torch.uint32,
            "U64": torch.uint64,
        }[record["dtype"]]

    def _block(self, key, block, *, cache=True):
        record = self.records[key]
        block_key = (key, block)
        if cache and block_key in self._row_blocks:
            self._row_blocks.move_to_end(block_key)
            return self._row_blocks[block_key]
        start = block * 32
        rows = min(32, record["shape"][0] - start)
        digest = hashlib.sha256(f"{self.seed}/{key}/block/{block}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
        result = torch.randn((rows, *record["shape"][1:]), generator=generator, dtype=torch.float32)
        result = result.mul_(record["std"]).add_(record["mean"]).to(self._dtype(record))
        if cache:
            self._row_blocks[block_key] = result
            if len(self._row_blocks) > 256:
                self._row_blocks.popitem(last=False)
        return result

    def rows(self, key, start, stop):
        """Fetch exact synthetic PLE rows in O(number of touched 32-row blocks).

        This never regenerates the entire shard. The bounded 256-block host
        cache is test-only; TT setup still uploads every shard from __getitem__.
        """
        if "ngram_embedding.shard_" not in key:
            return self[key][start:stop]
        record = self.records[key]
        if not 0 <= start < stop <= record["shape"][0]:
            raise ValueError("Invalid synthetic table row interval")
        parts = []
        for block in range(start // 32, (stop - 1) // 32 + 1):
            block_start = block * 32
            part = self._block(key, block)
            parts.append(part[max(start - block_start, 0) : min(stop - block_start, 32)])
        return torch.cat(parts, dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN38_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    for layer in args.layers:
        collect_layer(args.checkpoint, layer, args.output_dir / f"layer_{layer}.json")
