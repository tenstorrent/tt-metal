#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

FP8_DTYPES: tuple[torch.dtype, ...] = (torch.float8_e4m3fn,)
if hasattr(torch, "float8_e5m2"):
    FP8_DTYPES = FP8_DTYPES + (torch.float8_e5m2,)


def format_bytes(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    return f"{value:.2f} {unit}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Download (optional), dequantize, and rewrite a DeepSeek-style sharded " "safetensors checkpoint.")
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face repo id (use this OR --input-dir).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Local checkpoint directory containing model.safetensors.index.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the dequantized safetensors checkpoint.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token for gated/private repos.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional HF cache directory used when --repo-id is provided.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Output dtype for dequantized tensors only.",
    )
    parser.add_argument(
        "--keep-scale-inv",
        action="store_true",
        help="Keep *_scale_inv tensors in output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    parser.add_argument(
        "--max-output-shard-size-mb",
        type=int,
        default=5120,
        help=(
            "Maximum size in MiB for each output safetensors shard. "
            "Lower values reduce peak host memory usage during conversion."
        ),
    )
    return parser.parse_args()


def parse_torch_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype]


def validate_args(args: argparse.Namespace) -> None:
    has_repo = args.repo_id is not None
    has_input = args.input_dir is not None
    if has_repo == has_input:
        raise ValueError("Provide exactly one of --repo-id or --input-dir.")
    if args.max_output_shard_size_mb <= 0:
        raise ValueError("--max-output-shard-size-mb must be > 0.")


def resolve_input_dir(args: argparse.Namespace) -> Path:
    if args.input_dir is not None:
        input_dir = args.input_dir.expanduser().resolve()
        if not input_dir.is_dir():
            raise FileNotFoundError(f"--input-dir does not exist or is not a directory: {input_dir}")
        logger.info(f"Using local input checkpoint: {input_dir}")
        return input_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required when using --repo-id. " "Install it or pass --input-dir instead."
        ) from exc

    logger.info(f"Downloading checkpoint from HF repo: {args.repo_id}")
    input_dir = Path(
        snapshot_download(
            repo_id=args.repo_id,
            token=args.hf_token,
            cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
            ignore_patterns="original/*",
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
        )
    ).resolve()
    logger.info(f"Downloaded to: {input_dir}")
    return input_dir


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite to allow writing into an existing non-empty directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")
    return output_dir


def load_index(index_path: Path) -> dict[str, Any]:
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        index_obj = json.load(f)
    if "weight_map" not in index_obj or not isinstance(index_obj["weight_map"], dict):
        raise ValueError(f"Invalid index file (missing dict weight_map): {index_path}")
    logger.info("Loaded checkpoint index: " f"{len(index_obj['weight_map'])} tensor entries from {index_path}")
    return index_obj


def load_block_shape(config_path: Path) -> tuple[int, ...]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    try:
        block_shape = tuple(config["quantization_config"]["weight_block_size"])
    except (TypeError, KeyError) as exc:
        raise ValueError("config.json missing quantization_config.weight_block_size") from exc
    if not block_shape or any((not isinstance(v, int) or v <= 0) for v in block_shape):
        raise ValueError(f"Invalid block shape: {block_shape}")
    logger.info(f"Using quantization block shape: {block_shape}")
    return block_shape


def build_keys_by_file(weight_map: dict[str, str]) -> dict[str, list[str]]:
    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"Invalid key in weight_map: {key!r}")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Invalid shard filename in weight_map for key '{key}': {filename!r}")
        keys_by_file[filename].append(key)
    return keys_by_file


def validate_input_output_paths(input_dir: Path, output_dir: Path) -> None:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError(f"Input and output directories must be different. Got: {input_dir}")


def preflight_validate_checkpoint_structure(
    model_dir: Path,
    weight_map: dict[str, str],
    keys_by_file: dict[str, list[str]],
) -> None:
    if not weight_map:
        raise ValueError("Checkpoint index has no tensor entries in weight_map.")

    logger.info("Running checkpoint preflight validation")

    index_key_set = set(weight_map.keys())
    # Require every scale tensor to have a corresponding base tensor.
    for key in index_key_set:
        if key.endswith("_scale_inv"):
            base_key = key[: -len("_scale_inv")]
            if base_key not in index_key_set:
                raise ValueError(
                    f"Found scale tensor without matching base tensor: '{key}' "
                    f"(expected '{base_key}' in weight_map)"
                )

    missing_shards: list[str] = []
    missing_keys_in_shard: list[tuple[str, str]] = []
    unindexed_key_count = 0
    for shard_name, shard_keys in sorted(keys_by_file.items()):
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            missing_shards.append(shard_name)
            continue
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for key in shard_keys:
                if key not in available:
                    missing_keys_in_shard.append((shard_name, key))
            # Not fatal but useful: report keys physically present but not indexed.
            unindexed_key_count += len(available - index_key_set)

    if missing_shards:
        raise FileNotFoundError("Index references missing shard files: " + ", ".join(missing_shards))
    if missing_keys_in_shard:
        examples = ", ".join(f"{k} @ {s}" for s, k in missing_keys_in_shard[:5])
        raise KeyError(f"Index references keys that are missing in the referenced shard(s). Examples: {examples}")
    if unindexed_key_count > 0:
        logger.warning(
            f"Detected {unindexed_key_count} tensor key(s) present in shard files but absent from index weight_map"
        )
    logger.info("Checkpoint preflight validation passed")


class ShardTensorReader:
    def __init__(self, model_dir: Path, weight_map: dict[str, str]):
        self._model_dir = model_dir
        self._weight_map = weight_map
        self._handles: dict[str, Any] = {}

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self._weight_map:
            raise KeyError(f"Tensor key missing from weight_map: {key}")
        shard_name = self._weight_map[key]
        shard_path = self._model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Shard file not found for key '{key}': {shard_path}")
        handle = self._handles.get(shard_name)
        if handle is None:
            logger.info(f"Opening source shard: {shard_name}")
            handle = safe_open(shard_path, framework="pt", device="cpu")
            self._handles[shard_name] = handle
        return handle.get_tensor(key)

    def close(self) -> None:
        for handle in self._handles.values():
            close_fn = getattr(handle, "close", None)
            if callable(close_fn):
                close_fn()
        self._handles.clear()


def dequantize_tensor(tensor: torch.Tensor, inv_scale: torch.Tensor, block_shape: tuple[int, ...]) -> torch.Tensor:
    # Keep this equivalent to DeepSeek helper logic while avoiding ttnn imports.
    if tensor.ndim != inv_scale.ndim:
        raise ValueError(f"Tensor and inverse scale must have same ndim, got {tensor.ndim} and {inv_scale.ndim}")
    if len(block_shape) != tensor.ndim:
        raise ValueError(
            f"Block shape rank mismatch, got len(block_shape)={len(block_shape)} and tensor.ndim={tensor.ndim}"
        )
    if any(inv_scale.shape[i] * block_shape[i] < tensor.shape[i] for i in range(tensor.ndim)):
        raise ValueError(
            "Inverse scale shape does not cover tensor shape: "
            f"tensor={tuple(tensor.shape)}, inv_scale={tuple(inv_scale.shape)}, block_shape={block_shape}"
        )

    expanded = inv_scale
    for i, block_dim in enumerate(block_shape):
        expanded = expanded.repeat_interleave(block_dim, dim=i)

    slices = tuple(slice(0, size) for size in tensor.shape)
    return tensor.float() * expanded[slices].float()


def copy_auxiliary_files(input_dir: Path, output_dir: Path) -> None:
    copied_files = 0
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            continue
        if path.name == "model.safetensors.index.json":
            continue
        shutil.copy2(path, output_dir / path.name)
        copied_files += 1
    logger.info(f"Copied {copied_files} auxiliary files to output directory")


class BufferedShardWriter:
    """
    Incremental output writer that bounds memory by flushing tensor batches
    once a byte budget is reached.
    """

    def __init__(self, output_dir: Path, max_shard_size_bytes: int):
        self.output_dir = output_dir
        self.max_shard_size_bytes = max_shard_size_bytes
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffer_bytes = 0
        self._tmp_shard_names: list[str] = []
        self._tmp_shard_keys: list[list[str]] = []
        self._next_tmp_idx = 1
        self.total_size_bytes = 0
        logger.info("Configured output shard flushing: " f"max_output_shard_size={format_bytes(max_shard_size_bytes)}")

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def add_tensor(self, key: str, tensor: torch.Tensor) -> None:
        tensor_bytes = self._tensor_nbytes(tensor)

        if self._buffer and self._buffer_bytes + tensor_bytes > self.max_shard_size_bytes:
            self._flush()

        self._buffer[key] = tensor
        self._buffer_bytes += tensor_bytes
        self.total_size_bytes += tensor_bytes

        if self._buffer_bytes >= self.max_shard_size_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return

        tmp_name = f".tmp-model-{self._next_tmp_idx:05d}.safetensors"
        flushed_tensors = len(self._buffer)
        flushed_bytes = self._buffer_bytes
        save_file(self._buffer, str(self.output_dir / tmp_name))
        self._tmp_shard_names.append(tmp_name)
        self._tmp_shard_keys.append(list(self._buffer.keys()))
        self._next_tmp_idx += 1
        self._buffer.clear()
        self._buffer_bytes = 0
        logger.info(
            f"Flushed temporary output shard {tmp_name}: " f"{flushed_tensors} tensors, {format_bytes(flushed_bytes)}"
        )

    def finalize(self) -> dict[str, str]:
        self._flush()

        num_shards = len(self._tmp_shard_names)
        if num_shards == 0:
            logger.info("No output tensors were produced.")
            return {}

        output_weight_map: dict[str, str] = {}
        for idx, (tmp_name, keys) in enumerate(zip(self._tmp_shard_names, self._tmp_shard_keys), start=1):
            final_name = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
            (self.output_dir / tmp_name).replace(self.output_dir / final_name)
            logger.info(f"Finalized output shard: {final_name} ({len(keys)} tensors)")
            for key in keys:
                output_weight_map[key] = final_name

        return output_weight_map


def convert_checkpoint(
    input_dir: Path,
    output_dir: Path,
    out_dtype: torch.dtype,
    keep_scale_inv: bool,
    max_output_shard_size_bytes: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_input_output_paths(input_dir, output_dir)

    index_path = input_dir / "model.safetensors.index.json"
    config_path = input_dir / "config.json"
    index_obj = load_index(index_path)
    block_shape = load_block_shape(config_path)
    weight_map: dict[str, str] = index_obj["weight_map"]
    keys_by_file = build_keys_by_file(weight_map)
    preflight_validate_checkpoint_structure(input_dir, weight_map, keys_by_file)
    logger.info(f"Discovered {len(keys_by_file)} source shard files")

    reader = ShardTensorReader(input_dir, weight_map)
    writer = BufferedShardWriter(output_dir=output_dir, max_shard_size_bytes=max_output_shard_size_bytes)
    total_seen = 0
    total_dequantized = 0
    total_scales_kept = 0
    total_passthrough = 0

    try:
        shard_names = sorted(keys_by_file.keys())
        for shard_idx, shard_name in enumerate(shard_names, start=1):
            keys = keys_by_file[shard_name]
            logger.info(f"[{shard_idx}/{len(shard_names)}] Processing {shard_name} ({len(keys)} tensors)")
            shard_seen = 0
            shard_dequantized = 0
            shard_scales_kept = 0
            shard_passthrough = 0

            for key in keys:
                shard_seen += 1
                total_seen += 1
                if key.endswith("_scale_inv"):
                    if keep_scale_inv:
                        scale_tensor = reader.get_tensor(key)
                        writer.add_tensor(key, scale_tensor)
                        shard_scales_kept += 1
                        total_scales_kept += 1
                    continue

                tensor = reader.get_tensor(key)
                scale_key = f"{key}_scale_inv"

                if scale_key in weight_map:
                    if tensor.dtype not in FP8_DTYPES:
                        raise ValueError(
                            f"Tensor '{key}' has a scale tensor '{scale_key}' but dtype is {tensor.dtype}; "
                            "expected FP8 payload for dequantization."
                        )
                    inv_scale = reader.get_tensor(scale_key)
                    if not inv_scale.dtype.is_floating_point:
                        raise ValueError(f"Scale tensor '{scale_key}' must be floating point, got {inv_scale.dtype}")
                    tensor_out = dequantize_tensor(tensor, inv_scale, block_shape).to(out_dtype)
                    shard_dequantized += 1
                    total_dequantized += 1
                else:
                    is_fp8 = tensor.dtype in FP8_DTYPES
                    if is_fp8:
                        logger.error(f"Missing inverse-scale for FP8 tensor: {key} (expected {scale_key})")
                        raise KeyError(f"Missing inverse-scale tensor '{scale_key}' for float8 tensor '{key}'.")
                    tensor_out = tensor
                    shard_passthrough += 1
                    total_passthrough += 1

                writer.add_tensor(key, tensor_out)
            logger.info(
                f"Completed {shard_name}: seen={shard_seen}, "
                f"dequantized={shard_dequantized}, passthrough={shard_passthrough}, "
                f"scales_kept={shard_scales_kept}"
            )
    finally:
        reader.close()

    output_weight_map = writer.finalize()
    output_total_size = writer.total_size_bytes

    output_index = dict(index_obj)
    output_index["weight_map"] = output_weight_map
    metadata = dict(index_obj.get("metadata", {}))
    metadata["total_size"] = output_total_size
    output_index["metadata"] = metadata

    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(output_index, f, indent=2)
        f.write("\n")

    copy_auxiliary_files(input_dir, output_dir)

    logger.info(
        "Conversion completed: "
        f"written={len(output_weight_map)} tensors, total_size={format_bytes(output_total_size)} "
        f"(raw={output_total_size} bytes), output={output_dir}"
    )
    logger.info(
        "Tensor accounting summary: "
        f"seen={total_seen}, dequantized={total_dequantized}, "
        f"passthrough={total_passthrough}, scales_kept={total_scales_kept}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    input_dir = resolve_input_dir(args)
    output_dir = prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    out_dtype = parse_torch_dtype(args.dtype)
    logger.info(
        "Starting checkpoint conversion with options: "
        f"dtype={args.dtype}, keep_scale_inv={args.keep_scale_inv}, "
        f"max_output_shard_size_mb={args.max_output_shard_size_mb}"
    )

    convert_checkpoint(
        input_dir=input_dir,
        output_dir=output_dir,
        out_dtype=out_dtype,
        keep_scale_inv=args.keep_scale_inv,
        max_output_shard_size_bytes=args.max_output_shard_size_mb * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
