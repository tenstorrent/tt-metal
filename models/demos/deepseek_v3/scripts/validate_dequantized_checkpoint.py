#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from safetensors import safe_open

from models.demos.deepseek_v3.scripts.dequantize_hf_checkpoint import (
    build_keys_by_file,
    dequantize_tensor,
    load_block_shape,
    load_index,
)


def _safetensors_dtype_to_torch(dtype_str: str) -> torch.dtype:
    """Map safetensors dtype string (e.g. from get_slice(key).get_dtype()) to torch.dtype."""
    m = {
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": getattr(torch, "float8_e5m2", None),
    }
    out = m.get((dtype_str or "").upper())
    if out is None:
        raise ValueError(f"Unsupported safetensors dtype for validation: {dtype_str!r}")
    return out


def _is_fp8_safetensors_dtype(dtype_str: str) -> bool:
    s = (dtype_str or "").upper()
    return s in ("F8_E4M3", "F8_E5M2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a dequantized checkpoint against the original quantized checkpoint.",
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        required=True,
        help="Directory containing the original (quantized) checkpoint and model.safetensors.index.json.",
    )
    parser.add_argument(
        "--dequantized-dir",
        type=Path,
        required=True,
        help="Directory containing the dequantized checkpoint to validate.",
    )
    parser.add_argument(
        "--check-values",
        action="store_true",
        help="Load tensors and verify numerical correctness (passthrough equality, dequantized allclose).",
    )
    parser.add_argument(
        "--keep-scale-inv",
        action="store_true",
        help="Expect *_scale_inv tensors in the dequantized checkpoint (must match dequantize script flag).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for dequantized tensor comparison (default: 1e-2).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for dequantized tensor comparison (default: 1e-3).",
    )
    return parser.parse_args()


def _collect_metadata_from_checkpoint(
    model_dir: Path,
    weight_map: dict[str, str],
    keys_by_file: dict[str, list[str]],
) -> dict[str, tuple[tuple[int, ...], str]]:
    """For each key in weight_map, collect (shape, dtype_str) from shards using lazy get_slice."""
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    total = len(keys_by_file)
    started = time.monotonic()
    last_log = started
    for shard_idx, (shard_name, keys) in enumerate(sorted(keys_by_file.items()), start=1):
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Shard not found: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in keys:
                sl = handle.get_slice(key)
                shape = tuple(sl.get_shape())
                dtype_str = sl.get_dtype()
                result[key] = (shape, dtype_str)
        now = time.monotonic()
        if now - last_log >= 10.0:
            logger.info(
                f"Metadata progress: {shard_idx}/{total} shards "
                f"({100.0 * shard_idx / total:.1f}%), elapsed={now - started:.1f}s"
            )
            last_log = now
    return result


def _check_shard_files_exist(model_dir: Path, weight_map: dict[str, str]) -> list[str]:
    """Return list of missing shard filenames."""
    shard_names = set(weight_map.values())
    return [s for s in shard_names if not (model_dir / s).is_file()]


def run_metadata_checks(
    original_dir: Path,
    dequantized_dir: Path,
    keep_scale_inv: bool,
) -> dict[str, Any]:
    """Run tier-1 metadata-only checks. Returns a result dict with pass/fail and details."""
    index_path_orig = original_dir / "model.safetensors.index.json"
    index_path_deq = dequantized_dir / "model.safetensors.index.json"
    config_path_orig = original_dir / "config.json"

    result: dict[str, Any] = {
        "key_completeness": {"passed": False, "missing_keys": []},
        "scale_key_handling": {"passed": False, "message": ""},
        "no_extra_keys": {"passed": False, "extra_keys": []},
        "shape_preservation": {"passed": False, "mismatches": []},
        "dtype_correctness": {"passed": False, "mismatches": []},
        "shard_files_exist": {"passed": False, "missing_shards": []},
        "all_passed": False,
    }

    orig_index = load_index(index_path_orig)
    deq_index = load_index(index_path_deq)
    block_shape = load_block_shape(config_path_orig)
    orig_weight_map: dict[str, str] = orig_index["weight_map"]
    deq_weight_map: dict[str, str] = deq_index["weight_map"]

    orig_keys = set(orig_weight_map.keys())
    orig_base_keys = {k for k in orig_keys if not k.endswith("_scale_inv")}
    orig_scale_keys = orig_keys - orig_base_keys
    deq_keys = set(deq_weight_map.keys())

    # 1. Key completeness: every non-_scale_inv key from original exists in dequantized
    missing = sorted(orig_base_keys - deq_keys)
    result["key_completeness"]["missing_keys"] = missing
    result["key_completeness"]["passed"] = len(missing) == 0

    # 2. Scale key handling
    if keep_scale_inv:
        missing_scales = sorted(orig_scale_keys - deq_keys)
        if missing_scales:
            result["scale_key_handling"][
                "message"
            ] = f"Expected scale_inv keys in dequantized checkpoint but missing: {missing_scales[:5]}{'...' if len(missing_scales) > 5 else ''}"
        else:
            result["scale_key_handling"]["passed"] = True
            result["scale_key_handling"]["message"] = "All scale_inv keys present"
    else:
        present_scales = sorted(deq_keys & orig_scale_keys)
        if present_scales:
            result["scale_key_handling"][
                "message"
            ] = f"Dequantized checkpoint should not contain scale_inv keys but has: {present_scales[:5]}{'...' if len(present_scales) > 5 else ''}"
        else:
            result["scale_key_handling"]["passed"] = True
            result["scale_key_handling"]["message"] = "No scale_inv keys in dequantized (expected)"

    # 3. No extra keys
    allowed_deq = orig_base_keys | (orig_scale_keys if keep_scale_inv else set())
    extra = sorted(deq_keys - allowed_deq)
    result["no_extra_keys"]["extra_keys"] = extra
    result["no_extra_keys"]["passed"] = len(extra) == 0

    # 4 & 5. Shape and dtype
    keys_by_file_orig = build_keys_by_file(orig_weight_map)
    keys_by_file_deq = build_keys_by_file(deq_weight_map)
    orig_meta = _collect_metadata_from_checkpoint(original_dir, orig_weight_map, keys_by_file_orig)
    deq_meta = _collect_metadata_from_checkpoint(dequantized_dir, deq_weight_map, keys_by_file_deq)

    keys_to_compare = orig_base_keys & deq_keys
    shape_mismatches = []
    dtype_mismatches = []
    fp8_dtypes = (torch.float8_e4m3fn, getattr(torch, "float8_e5m2", None))
    for key in sorted(keys_to_compare):
        orig_shape, orig_dtype_str = orig_meta[key]
        deq_shape, deq_dtype_str = deq_meta[key]
        if orig_shape != deq_shape:
            shape_mismatches.append((key, orig_shape, deq_shape))
        orig_is_fp8 = _is_fp8_safetensors_dtype(orig_dtype_str)
        try:
            deq_torch_dtype = _safetensors_dtype_to_torch(deq_dtype_str)
        except ValueError:
            dtype_mismatches.append((key, orig_dtype_str, deq_dtype_str, "unsupported dtype"))
            continue
        if orig_is_fp8:
            if not (deq_torch_dtype.is_floating_point and deq_torch_dtype not in fp8_dtypes):
                dtype_mismatches.append((key, orig_dtype_str, deq_dtype_str, "expected float (dequantized)"))
        else:
            try:
                orig_torch = _safetensors_dtype_to_torch(orig_dtype_str)
                if deq_torch_dtype != orig_torch:
                    dtype_mismatches.append((key, orig_dtype_str, deq_dtype_str, "passthrough dtype must match"))
            except ValueError:
                pass

    result["shape_preservation"]["mismatches"] = shape_mismatches
    result["shape_preservation"]["passed"] = len(shape_mismatches) == 0
    result["dtype_correctness"]["mismatches"] = dtype_mismatches
    result["dtype_correctness"]["passed"] = len(dtype_mismatches) == 0

    # 6. Shard file existence (dequantized)
    missing_shards = _check_shard_files_exist(dequantized_dir, deq_weight_map)
    result["shard_files_exist"]["missing_shards"] = missing_shards
    result["shard_files_exist"]["passed"] = len(missing_shards) == 0

    result["all_passed"] = (
        result["key_completeness"]["passed"]
        and result["scale_key_handling"]["passed"]
        and result["no_extra_keys"]["passed"]
        and result["shape_preservation"]["passed"]
        and result["dtype_correctness"]["passed"]
        and result["shard_files_exist"]["passed"]
    )
    result["orig_meta"] = orig_meta
    result["deq_meta"] = deq_meta
    result["block_shape"] = block_shape
    result["orig_weight_map"] = orig_weight_map
    result["deq_weight_map"] = deq_weight_map
    result["orig_base_keys"] = orig_base_keys
    result["orig_scale_keys"] = orig_scale_keys
    return result


class _ShardReader:
    """Lazy reader that opens shards on demand and caches handles."""

    def __init__(self, model_dir: Path, weight_map: dict[str, str]):
        self._model_dir = model_dir
        self._weight_map = weight_map
        self._handles: dict[str, Any] = {}

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self._weight_map:
            raise KeyError(key)
        shard_name = self._weight_map[key]
        if shard_name not in self._handles:
            path = self._model_dir / shard_name
            self._handles[shard_name] = safe_open(path, framework="pt", device="cpu")
        return self._handles[shard_name].get_tensor(key)

    def close(self) -> None:
        for h in self._handles.values():
            close_fn = getattr(h, "close", None)
            if callable(close_fn):
                close_fn()
        self._handles.clear()


def run_value_checks(
    meta_result: dict[str, Any],
    original_dir: Path,
    dequantized_dir: Path,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Run tier-2 value checks. meta_result must contain orig_meta, deq_meta, block_shape, weight_maps, orig_base_keys."""
    value_result: dict[str, Any] = {
        "passthrough_ok": [],
        "passthrough_fail": [],
        "dequantized_ok": [],
        "dequantized_fail": [],
        "all_passed": False,
    }
    orig_reader = _ShardReader(original_dir, meta_result["orig_weight_map"])
    deq_reader = _ShardReader(dequantized_dir, meta_result["deq_weight_map"])
    block_shape = meta_result["block_shape"]
    orig_meta = meta_result["orig_meta"]
    deq_meta = meta_result["deq_meta"]
    orig_base_keys = meta_result["orig_base_keys"]
    keys_to_check = orig_base_keys & set(deq_meta.keys())
    total = len(keys_to_check)
    started = time.monotonic()
    last_log = started
    try:
        for idx, key in enumerate(sorted(keys_to_check)):
            orig_shape, orig_dtype_str = orig_meta[key]
            orig_is_fp8 = _is_fp8_safetensors_dtype(orig_dtype_str)
            orig_t = orig_reader.get_tensor(key)
            deq_t = deq_reader.get_tensor(key)
            if not orig_is_fp8:
                if torch.equal(orig_t, deq_t):
                    value_result["passthrough_ok"].append(key)
                else:
                    value_result["passthrough_fail"].append(key)
            else:
                scale_key = f"{key}_scale_inv"
                inv_scale = orig_reader.get_tensor(scale_key)
                expected = dequantize_tensor(orig_t, inv_scale, block_shape).to(deq_t.dtype)
                if torch.allclose(deq_t.float(), expected.float(), rtol=rtol, atol=atol):
                    value_result["dequantized_ok"].append(key)
                else:
                    value_result["dequantized_fail"].append(key)
            now = time.monotonic()
            if now - last_log >= 10.0:
                logger.info(
                    f"Value check progress: {idx + 1}/{total} "
                    f"({100.0 * (idx + 1) / total:.1f}%), elapsed={now - started:.1f}s"
                    f"passthrough_fail={len(value_result['passthrough_fail'])}, dequantized_fail={len(value_result['dequantized_fail'])}"
                )
                last_log = now
    finally:
        orig_reader.close()
        deq_reader.close()

    value_result["all_passed"] = (
        len(value_result["passthrough_fail"]) == 0 and len(value_result["dequantized_fail"]) == 0
    )
    return value_result


def print_summary(meta_result: dict[str, Any], value_result: dict[str, Any] | None) -> None:
    """Print structured summary to stderr/log."""
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    def section(name: str, passed: bool, details: str = ""):
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {name}")
        if details:
            logger.info(f"         {details}")

    section("Key completeness", meta_result["key_completeness"]["passed"])
    if meta_result["key_completeness"]["missing_keys"]:
        mk = meta_result["key_completeness"]["missing_keys"]
        logger.info(f"         Missing: {mk[:10]}{'...' if len(mk) > 10 else ''}")

    section(
        "Scale key handling", meta_result["scale_key_handling"]["passed"], meta_result["scale_key_handling"]["message"]
    )
    section("No extra keys", meta_result["no_extra_keys"]["passed"])
    if meta_result["no_extra_keys"]["extra_keys"]:
        ek = meta_result["no_extra_keys"]["extra_keys"]
        logger.info(f"         Extra: {ek[:10]}{'...' if len(ek) > 10 else ''}")

    section("Shape preservation", meta_result["shape_preservation"]["passed"])
    for key, o, d in meta_result["shape_preservation"]["mismatches"][:5]:
        logger.info(f"         {key}: orig{o} vs deq{d}")
    if len(meta_result["shape_preservation"]["mismatches"]) > 5:
        logger.info(f"         ... and {len(meta_result['shape_preservation']['mismatches']) - 5} more")

    section("Dtype correctness", meta_result["dtype_correctness"]["passed"])
    for tup in meta_result["dtype_correctness"]["mismatches"][:5]:
        logger.info(f"         {tup[0]}: orig={tup[1]} deq={tup[2]} ({tup[3]})")
    if len(meta_result["dtype_correctness"]["mismatches"]) > 5:
        logger.info(f"         ... and {len(meta_result['dtype_correctness']['mismatches']) - 5} more")

    section("Shard files exist", meta_result["shard_files_exist"]["passed"])
    if meta_result["shard_files_exist"]["missing_shards"]:
        logger.info(f"         Missing: {meta_result['shard_files_exist']['missing_shards']}")

    meta_ok = meta_result["all_passed"]
    logger.info("-" * 60)
    logger.info(f"  Metadata checks: {'PASS' if meta_ok else 'FAIL'}")

    if value_result is not None:
        section("Passthrough tensor equality", len(value_result["passthrough_fail"]) == 0)
        if value_result["passthrough_fail"]:
            pf = value_result["passthrough_fail"]
            logger.info(f"         Failed: {pf[:5]}{'...' if len(pf) > 5 else ''}")
        section("Dequantized tensor correctness", len(value_result["dequantized_fail"]) == 0)
        if value_result["dequantized_fail"]:
            df = value_result["dequantized_fail"]
            logger.info(f"         Failed: {df[:5]}{'...' if len(df) > 5 else ''}")
        logger.info(f"  Value checks: {'PASS' if value_result['all_passed'] else 'FAIL'}")
    logger.info("=" * 60)


def main() -> int:
    args = parse_args()
    original_dir = args.original_dir.expanduser().resolve()
    dequantized_dir = args.dequantized_dir.expanduser().resolve()
    if not original_dir.is_dir():
        logger.error(f"Original directory does not exist: {original_dir}")
        return 1
    if not dequantized_dir.is_dir():
        logger.error(f"Dequantized directory does not exist: {dequantized_dir}")
        return 1
    if original_dir == dequantized_dir:
        logger.error("Original and dequantized directories must be different")
        return 1

    logger.info(f"Original checkpoint: {original_dir}")
    logger.info(f"Dequantized checkpoint: {dequantized_dir}")
    logger.info(f"Keep scale_inv: {args.keep_scale_inv}, Check values: {args.check_values}")

    meta_result = run_metadata_checks(original_dir, dequantized_dir, args.keep_scale_inv)
    value_result = None
    if args.check_values and meta_result["all_passed"]:
        value_result = run_value_checks(meta_result, original_dir, dequantized_dir, args.rtol, args.atol)
    elif args.check_values and not meta_result["all_passed"]:
        logger.warning("Skipping value checks because metadata checks failed")

    print_summary(meta_result, value_result)

    if not meta_result["all_passed"]:
        return 1
    if value_result is not None and not value_result["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
