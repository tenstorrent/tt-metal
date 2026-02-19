#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor

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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel source-shard conversion workers. Use 1 for sequential mode.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse completed temporary shard outputs from a previous interrupted run when compatible "
            "per-shard manifests are present in --output-dir."
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
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be > 0.")


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

    total_shards = len(keys_by_file)
    logger.info(f"Preflight: validating {total_shards} shard file(s)")

    missing_shards: list[str] = []
    missing_keys_in_shard: list[tuple[str, str]] = []
    unindexed_key_count = 0
    preflight_started = time.monotonic()
    last_progress_log = preflight_started
    for shard_idx, (shard_name, shard_keys) in enumerate(sorted(keys_by_file.items()), start=1):
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

        now = time.monotonic()
        if now - last_progress_log >= 10.0:
            elapsed = now - preflight_started
            logger.info(
                f"Preflight progress: {shard_idx}/{total_shards} shards checked "
                f"({(100.0 * shard_idx / total_shards):.1f}%), elapsed={elapsed:.1f}s"
            )
            last_progress_log = now

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
    def __init__(self, model_dir: Path, weight_map: dict[str, str], max_open_handles: int = 32):
        if max_open_handles <= 0:
            raise ValueError("max_open_handles must be > 0.")
        self._model_dir = model_dir
        self._weight_map = weight_map
        self._max_open_handles = max_open_handles
        self._handles: OrderedDict[str, Any] = OrderedDict()

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self._weight_map:
            raise KeyError(f"Tensor key missing from weight_map: {key}")
        shard_name = self._weight_map[key]
        shard_path = self._model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Shard file not found for key '{key}': {shard_path}")
        handle = self._handles.get(shard_name)
        if handle is None:
            while len(self._handles) >= self._max_open_handles:
                evicted_shard, evicted_handle = self._handles.popitem(last=False)
                close_fn = getattr(evicted_handle, "close", None)
                if callable(close_fn):
                    close_fn()
                logger.debug(f"Evicted cached source shard handle: {evicted_shard}")
            logger.info(f"Opening source shard: {shard_name}")
            handle = safe_open(shard_path, framework="pt", device="cpu")
            self._handles[shard_name] = handle
        else:
            self._handles.move_to_end(shard_name)
        return handle.get_tensor(key)

    def close(self) -> None:
        for handle in self._handles.values():
            close_fn = getattr(handle, "close", None)
            if callable(close_fn):
                close_fn()
        self._handles.clear()


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

    def __init__(self, output_dir: Path, max_shard_size_bytes: int, temp_prefix: str = ".tmp-model"):
        self.output_dir = output_dir
        self.max_shard_size_bytes = max_shard_size_bytes
        self.temp_prefix = temp_prefix
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

        tmp_name = f"{self.temp_prefix}-{self._next_tmp_idx:05d}.safetensors"
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

    def collect_tmp_entries(self) -> list[tuple[str, list[str]]]:
        self._flush()
        return list(zip(self._tmp_shard_names, self._tmp_shard_keys))

    def finalize(self) -> dict[str, str]:
        return finalize_output_shards(self.output_dir, self.collect_tmp_entries())


def _source_keys_digest(keys: list[str]) -> str:
    h = hashlib.sha256()
    for key in keys:
        h.update(key.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def source_shard_manifest_path(output_dir: Path, shard_idx: int) -> Path:
    return output_dir / f".tmp-s{shard_idx:05d}.manifest.json"


def expected_output_keys_for_source_shard(source_keys: list[str], keep_scale_inv: bool) -> list[str]:
    output_keys: list[str] = []
    for key in source_keys:
        if key.endswith("_scale_inv"):
            if keep_scale_inv:
                output_keys.append(key)
        else:
            output_keys.append(key)
    return output_keys


def _safetensors_dtype_nbytes(dtype_name: str) -> int:
    mapping = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3FN": 1,
        "F8_E5M2": 1,
        "F16": 2,
        "BF16": 2,
        "U16": 2,
        "I16": 2,
        "F32": 4,
        "U32": 4,
        "I32": 4,
        "F64": 8,
        "U64": 8,
        "I64": 8,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported safetensors dtype in legacy tmp recovery: {dtype_name}")
    return mapping[dtype_name]


def maybe_recover_source_shard_from_legacy_tmp_files(
    output_dir: Path,
    shard_idx: int,
    shard_name: str,
    source_keys: list[str],
    weight_map: dict[str, str],
    keep_scale_inv: bool,
) -> dict[str, Any] | None:
    pattern = f".tmp-s{shard_idx:05d}-*.safetensors"
    tmp_paths = sorted(output_dir.glob(pattern))
    if not tmp_paths:
        return None

    expected_keys = expected_output_keys_for_source_shard(source_keys, keep_scale_inv)
    expected_key_set = set(expected_keys)
    observed_key_set: set[str] = set()
    tmp_entries: list[tuple[str, list[str]]] = []
    total_size_bytes = 0

    for tmp_path in tmp_paths:
        try:
            with safe_open(tmp_path, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                for key in keys:
                    tensor_slice = handle.get_slice(key)
                    shape = tensor_slice.get_shape()
                    dtype_name = str(tensor_slice.get_dtype())
                    numel = 1
                    for dim in shape:
                        numel *= dim
                    total_size_bytes += numel * _safetensors_dtype_nbytes(dtype_name)
        except Exception as exc:
            logger.warning(
                f"Failed to read legacy tmp shard {tmp_path.name} for source shard {shard_name}: {exc}. "
                "Reprocessing shard."
            )
            return None
        if not keys:
            logger.warning(f"Legacy tmp shard {tmp_path.name} has no tensors for source shard {shard_name}")
            return None
        for key in keys:
            if key in observed_key_set:
                logger.warning(
                    f"Legacy tmp shards contain duplicate key '{key}' for source shard {shard_name}; reprocessing."
                )
                return None
            observed_key_set.add(key)
        tmp_entries.append((tmp_path.name, keys))

    if observed_key_set != expected_key_set:
        logger.warning(
            f"Legacy tmp shard coverage mismatch for source shard {shard_name}: "
            f"expected={len(expected_key_set)} key(s), observed={len(observed_key_set)} key(s). Reprocessing."
        )
        return None

    logger.info(
        f"Recovered source shard {shard_idx} ({shard_name}) from legacy tmp files "
        f"({len(tmp_entries)} file(s), {len(observed_key_set)} key(s))"
    )
    dequantized = sum(1 for key in source_keys if not key.endswith("_scale_inv") and f"{key}_scale_inv" in weight_map)
    passthrough = sum(
        1 for key in source_keys if not key.endswith("_scale_inv") and f"{key}_scale_inv" not in weight_map
    )
    scales_kept = sum(1 for key in source_keys if key.endswith("_scale_inv") and keep_scale_inv)
    return {
        "shard_idx": shard_idx,
        "shard_name": shard_name,
        "tmp_entries": tmp_entries,
        "total_size_bytes": total_size_bytes,
        "seen": len(source_keys),
        "dequantized": dequantized,
        "passthrough": passthrough,
        "scales_kept": scales_kept,
    }


def write_source_shard_manifest(
    output_dir: Path,
    result: dict[str, Any],
    source_keys: list[str],
    out_dtype: torch.dtype,
    keep_scale_inv: bool,
) -> None:
    manifest_path = source_shard_manifest_path(output_dir, result["shard_idx"])
    manifest = {
        "version": 1,
        "shard_idx": result["shard_idx"],
        "shard_name": result["shard_name"],
        "source_key_count": len(source_keys),
        "source_key_digest": _source_keys_digest(source_keys),
        "out_dtype": str(out_dtype),
        "keep_scale_inv": bool(keep_scale_inv),
        "tmp_entries": [{"tmp_name": n, "keys": ks} for n, ks in result["tmp_entries"]],
        "total_size_bytes": int(result["total_size_bytes"]),
        "seen": int(result["seen"]),
        "dequantized": int(result["dequantized"]),
        "passthrough": int(result["passthrough"]),
        "scales_kept": int(result["scales_kept"]),
    }

    tmp_manifest_path = Path(str(manifest_path) + ".tmp")
    with tmp_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    tmp_manifest_path.replace(manifest_path)


def maybe_load_source_shard_manifest(
    output_dir: Path,
    shard_idx: int,
    shard_name: str,
    source_keys: list[str],
    weight_map: dict[str, str],
    out_dtype: torch.dtype,
    keep_scale_inv: bool,
) -> dict[str, Any] | None:
    manifest_path = source_shard_manifest_path(output_dir, shard_idx)
    if not manifest_path.is_file():
        return maybe_recover_source_shard_from_legacy_tmp_files(
            output_dir=output_dir,
            shard_idx=shard_idx,
            shard_name=shard_name,
            source_keys=source_keys,
            weight_map=weight_map,
            keep_scale_inv=keep_scale_inv,
        )

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to parse resume manifest {manifest_path}: {exc}. Reprocessing shard.")
        return None

    if manifest.get("version") != 1:
        logger.warning(f"Unsupported resume manifest version in {manifest_path}. Reprocessing shard.")
        return None
    if manifest.get("shard_idx") != shard_idx or manifest.get("shard_name") != shard_name:
        logger.warning(f"Resume manifest mismatch for shard {shard_name} in {manifest_path}. Reprocessing shard.")
        return None
    if manifest.get("source_key_count") != len(source_keys):
        logger.warning(f"Resume manifest key count mismatch for shard {shard_name}. Reprocessing shard.")
        return None
    if manifest.get("source_key_digest") != _source_keys_digest(source_keys):
        logger.warning(f"Resume manifest key digest mismatch for shard {shard_name}. Reprocessing shard.")
        return None
    if manifest.get("out_dtype") != str(out_dtype) or manifest.get("keep_scale_inv") != bool(keep_scale_inv):
        logger.warning(f"Resume manifest conversion options mismatch for shard {shard_name}. Reprocessing shard.")
        return None

    tmp_entries_obj = manifest.get("tmp_entries")
    if not isinstance(tmp_entries_obj, list):
        logger.warning(f"Resume manifest tmp_entries missing/invalid for shard {shard_name}. Reprocessing shard.")
        return None

    tmp_entries: list[tuple[str, list[str]]] = []
    for entry in tmp_entries_obj:
        if not isinstance(entry, dict):
            logger.warning(f"Resume manifest entry malformed for shard {shard_name}. Reprocessing shard.")
            return None
        tmp_name = entry.get("tmp_name")
        keys = entry.get("keys")
        if not isinstance(tmp_name, str) or not isinstance(keys, list) or any(not isinstance(k, str) for k in keys):
            logger.warning(f"Resume manifest entry malformed for shard {shard_name}. Reprocessing shard.")
            return None
        if not (output_dir / tmp_name).is_file():
            logger.warning(
                f"Resume manifest references missing temporary shard {tmp_name} for {shard_name}. Reprocessing shard."
            )
            return None
        tmp_entries.append((tmp_name, keys))

    return {
        "shard_idx": shard_idx,
        "shard_name": shard_name,
        "tmp_entries": tmp_entries,
        "total_size_bytes": int(manifest.get("total_size_bytes", 0)),
        "seen": int(manifest.get("seen", 0)),
        "dequantized": int(manifest.get("dequantized", 0)),
        "passthrough": int(manifest.get("passthrough", 0)),
        "scales_kept": int(manifest.get("scales_kept", 0)),
    }


def cleanup_source_shard_manifests(output_dir: Path, shard_indices: list[int]) -> None:
    cleaned = 0
    for shard_idx in shard_indices:
        path = source_shard_manifest_path(output_dir, shard_idx)
        if path.exists():
            path.unlink()
            cleaned += 1
    if cleaned:
        logger.info(f"Removed {cleaned} resume manifest file(s)")


def finalize_output_shards(output_dir: Path, tmp_entries: list[tuple[str, list[str]]]) -> dict[str, str]:
    num_shards = len(tmp_entries)
    if num_shards == 0:
        logger.info("No output tensors were produced.")
        return {}

    output_weight_map: dict[str, str] = {}
    for idx, (tmp_name, keys) in enumerate(tmp_entries, start=1):
        final_name = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        (output_dir / tmp_name).replace(output_dir / final_name)
        logger.info(f"Finalized output shard: {final_name} ({len(keys)} tensors)")
        for key in keys:
            output_weight_map[key] = final_name
    return output_weight_map


def process_source_shard(
    shard_idx: int,
    shard_name: str,
    keys: list[str],
    input_dir: Path,
    output_dir: Path,
    weight_map: dict[str, str],
    block_shape: tuple[int, ...],
    out_dtype: torch.dtype,
    keep_scale_inv: bool,
    max_output_shard_size_bytes: int,
) -> dict[str, Any]:
    logger.info(f"[worker {shard_idx}] start {shard_name} ({len(keys)} tensor entries)")
    reader = ShardTensorReader(input_dir, weight_map)
    writer = BufferedShardWriter(
        output_dir=output_dir,
        max_shard_size_bytes=max_output_shard_size_bytes,
        temp_prefix=f".tmp-s{shard_idx:05d}",
    )

    shard_seen = 0
    shard_dequantized = 0
    shard_scales_kept = 0
    shard_passthrough = 0
    started = time.monotonic()
    last_progress_log = started
    total_keys = len(keys)
    try:
        for key in keys:
            shard_seen += 1
            if key.endswith("_scale_inv"):
                if keep_scale_inv:
                    scale_tensor = reader.get_tensor(key)
                    writer.add_tensor(key, scale_tensor)
                    shard_scales_kept += 1
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
                del inv_scale
            else:
                is_fp8 = tensor.dtype in FP8_DTYPES
                if is_fp8:
                    logger.error(f"Missing inverse-scale for FP8 tensor: {key} (expected {scale_key})")
                    raise KeyError(f"Missing inverse-scale tensor '{scale_key}' for float8 tensor '{key}'.")
                tensor_out = tensor
                shard_passthrough += 1

            writer.add_tensor(key, tensor_out)
            del tensor
            del tensor_out

            now = time.monotonic()
            if now - last_progress_log >= 10.0:
                elapsed = now - started
                logger.info(
                    f"[worker {shard_idx}] progress {shard_seen}/{total_keys} "
                    f"({(100.0 * shard_seen / total_keys):.1f}%), "
                    f"dequantized={shard_dequantized}, passthrough={shard_passthrough}, "
                    f"scales_kept={shard_scales_kept}, elapsed={elapsed:.1f}s"
                )
                last_progress_log = now
    finally:
        reader.close()

    tmp_entries = writer.collect_tmp_entries()
    logger.info(
        f"[worker {shard_idx}] completed {shard_name}: seen={shard_seen}, "
        f"dequantized={shard_dequantized}, passthrough={shard_passthrough}, "
        f"scales_kept={shard_scales_kept}, tmp_shards={len(tmp_entries)}"
    )
    return {
        "shard_idx": shard_idx,
        "shard_name": shard_name,
        "tmp_entries": tmp_entries,
        "total_size_bytes": writer.total_size_bytes,
        "seen": shard_seen,
        "dequantized": shard_dequantized,
        "passthrough": shard_passthrough,
        "scales_kept": shard_scales_kept,
    }


def convert_checkpoint(
    input_dir: Path,
    output_dir: Path,
    out_dtype: torch.dtype,
    keep_scale_inv: bool,
    max_output_shard_size_bytes: int,
    num_workers: int = 1,
    resume: bool = False,
) -> None:
    if num_workers <= 0:
        raise ValueError("num_workers must be > 0.")
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
    logger.info(f"Conversion worker mode: num_workers={num_workers}")
    per_worker_output_buffer_bytes = max(1, max_output_shard_size_bytes // num_workers)
    logger.info(
        "Effective per-worker output buffer cap: "
        f"{format_bytes(per_worker_output_buffer_bytes)} "
        f"(global target shard size {format_bytes(max_output_shard_size_bytes)})"
    )
    shard_names = sorted(keys_by_file.keys())
    work_items = [(idx, name, keys_by_file[name]) for idx, name in enumerate(shard_names, start=1)]
    logger.info(f"Number of work items: {len(work_items)}")

    results: list[dict[str, Any]] = []
    pending_work_items: list[tuple[int, str, list[str]]] = []
    if resume:
        resumed = 0
        for idx, name, keys in work_items:
            resumed_result = maybe_load_source_shard_manifest(
                output_dir=output_dir,
                shard_idx=idx,
                shard_name=name,
                source_keys=keys,
                weight_map=weight_map,
                out_dtype=out_dtype,
                keep_scale_inv=keep_scale_inv,
            )
            if resumed_result is None:
                pending_work_items.append((idx, name, keys))
            else:
                results.append(resumed_result)
                resumed += 1
                logger.info(f"Resuming shard {idx}/{len(work_items)} from existing tmp outputs: {name}")
        logger.info(f"Resume scan complete: resumed={resumed}, pending={len(pending_work_items)}")
    else:
        pending_work_items = work_items

    if num_workers == 1:
        for idx, name, keys in pending_work_items:
            result = process_source_shard(
                shard_idx=idx,
                shard_name=name,
                keys=keys,
                input_dir=input_dir,
                output_dir=output_dir,
                weight_map=weight_map,
                block_shape=block_shape,
                out_dtype=out_dtype,
                keep_scale_inv=keep_scale_inv,
                max_output_shard_size_bytes=per_worker_output_buffer_bytes,
            )
            write_source_shard_manifest(
                output_dir=output_dir,
                result=result,
                source_keys=keys,
                out_dtype=out_dtype,
                keep_scale_inv=keep_scale_inv,
            )
            results.append(result)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures: dict[concurrent.futures.Future[dict[str, Any]], tuple[int, list[str]]] = {}
            for idx, name, keys in pending_work_items:
                future = executor.submit(
                    process_source_shard,
                    idx,
                    name,
                    keys,
                    input_dir,
                    output_dir,
                    weight_map,
                    block_shape,
                    out_dtype,
                    keep_scale_inv,
                    per_worker_output_buffer_bytes,
                )
                futures[future] = (idx, keys)

            for future in concurrent.futures.as_completed(futures):
                idx, keys = futures[future]
                result = future.result()
                write_source_shard_manifest(
                    output_dir=output_dir,
                    result=result,
                    source_keys=keys,
                    out_dtype=out_dtype,
                    keep_scale_inv=keep_scale_inv,
                )
                results.append(result)

    if len(results) != len(work_items):
        raise RuntimeError(
            f"Internal error: expected {len(work_items)} shard result(s), got {len(results)}. "
            "Some source shards were not converted or resumed."
        )

    results.sort(key=lambda r: r["shard_idx"])
    tmp_entries: list[tuple[str, list[str]]] = []
    total_seen = 0
    total_dequantized = 0
    total_scales_kept = 0
    total_passthrough = 0
    output_total_size = 0
    for result in results:
        tmp_entries.extend(result["tmp_entries"])
        total_seen += result["seen"]
        total_dequantized += result["dequantized"]
        total_passthrough += result["passthrough"]
        total_scales_kept += result["scales_kept"]
        output_total_size += result["total_size_bytes"]

    output_weight_map = finalize_output_shards(output_dir, tmp_entries)

    output_index = dict(index_obj)
    output_index["weight_map"] = output_weight_map
    metadata = dict(index_obj.get("metadata", {}))
    metadata["total_size"] = output_total_size
    output_index["metadata"] = metadata

    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(output_index, f, indent=2)
        f.write("\n")

    copy_auxiliary_files(input_dir, output_dir)
    cleanup_source_shard_manifests(output_dir, [r["shard_idx"] for r in results])

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
        f"max_output_shard_size_mb={args.max_output_shard_size_mb}, "
        f"num_workers={args.num_workers}, resume={args.resume}"
    )

    convert_checkpoint(
        input_dir=input_dir,
        output_dir=output_dir,
        out_dtype=out_dtype,
        keep_scale_inv=args.keep_scale_inv,
        max_output_shard_size_bytes=args.max_output_shard_size_mb * 1024 * 1024,
        num_workers=args.num_workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
