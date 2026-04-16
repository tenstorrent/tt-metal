#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Convert DeepSeek expert weights to a BSPM-pre-quantized tt-metal weight cache.

Runs on the target hardware (TG / T3K / N300) with a live mesh device.  For
each MoE layer the script applies BitSculpt BSPM tile assignments to the
dequantized expert weights before the standard bfloat4_b/bfloat8_b conversion,
producing a weight cache that reflects the mixed-precision allocation.

Usage
-----
    # Minimal — TG system, R1-0528, Variant B 3.5 b/e:
    MESH_DEVICE=TG \\
    python models/demos/deepseek_v3/scripts/convert_bspm_weights.py \\
        --model-path  /proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528 \\
        --output-path /path/to/output/bspm_B_3.5 \\
        --bspm-dir    /path/to/bit_sculpt/results \\
        --bspm-model  deepseek-r1-0528

    # Custom variant / budget:
        --bspm-variant A --bspm-budget 4.0

    # Dry-run: validate BSPM files exist and are loadable (no conversion):
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from safetensors import safe_open
from transformers import AutoConfig

# ---------------------------------------------------------------------------
# Helpers shared with test_bspm_demo.py
# ---------------------------------------------------------------------------

_TT_CODE_TO_MANT: dict[int, int | None] = {0: 7, 1: 3, 2: 1, 3: None}
_PROJ_IDX = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
_EXPERT_KEY_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.weight$")
_STACKED_EXPERT_KEY_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts_stacked\.(\w+)\.weight$")


class _BsprnStateDict(Mapping[str, torch.Tensor]):
    """Lazy state-dict wrapper that applies BSPM quantization on demand.

    Pre-computing all expert overrides for DeepSeek V3 (256 experts × 58 MoE
    layers × 3 projections) would require ~650 GB of host RAM before conversion
    starts.  Instead, quantization is computed on the fly inside __getitem__,
    keeping peak memory proportional to one tensor at a time.

    BSPM code arrays (~11 MB each) are cached per-layer after first access so
    each BSPM file is read at most once.  The quantized tensors themselves are
    never retained — they are handed directly to the caller and immediately freed.
    """

    def __init__(
        self,
        base,
        hf_config,
        bspm_dir: Path,
        bspm_model: str,
        variant: str,
        budget: float,
        *,
        _base_prefix: str = "",
        _codes_cache: dict[int, np.ndarray | None] | None = None,
        _load_bspm=None,
        _qdq=None,
        _first_k_dense: int | None = None,
        _expected_experts: int | None = None,
        _require_complete: bool = False,
    ):
        self._base = base
        self._base_prefix = _base_prefix
        self._first_k_dense = (
            _first_k_dense if _first_k_dense is not None else getattr(hf_config, "first_k_dense_replace", 3)
        )
        self._expected_experts = (
            _expected_experts if _expected_experts is not None else getattr(hf_config, "n_routed_experts", None)
        )
        self._require_complete = _require_complete
        self._bspm_dir = bspm_dir
        self._bspm_model = bspm_model
        self._variant = variant
        self._budget = budget
        self._codes_cache = {} if _codes_cache is None else _codes_cache

        if _load_bspm is None or _qdq is None:
            from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import load_bspm_for_layer
            from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import quantize_dequantize_bfp

            if _load_bspm is None:
                _load_bspm = load_bspm_for_layer
            if _qdq is None:
                _qdq = quantize_dequantize_bfp

        self._load_bspm = _load_bspm
        self._qdq = _qdq

    # ------------------------------------------------------------------
    # Public dict-like interface (delegates to base for non-expert keys)
    # ------------------------------------------------------------------

    def _full_key(self, key: str) -> str:
        return f"{self._base_prefix}{key}"

    def _get_stacked_weight(self, key: str) -> torch.Tensor:
        view_with_prefix = getattr(self._base, "view_with_prefix", None)
        if callable(view_with_prefix) and key.startswith("experts_stacked."):
            return view_with_prefix("experts_stacked.")[key[len("experts_stacked.") :]]
        return self._base[key]

    def _bspm_file_for_layer(self, layer_idx: int) -> Path:
        return (
            self._bspm_dir
            / self._bspm_model
            / f"layer_{layer_idx}"
            / "precision_eval"
            / f"precision_map_{self._variant}_{self._budget:.1f}.bspm"
        )

    def _missing_bspm_message(self, layer_idx: int) -> str:
        return f"Layer {layer_idx}: required BSPM file not found at {self._bspm_file_for_layer(layer_idx)}"

    def _partial_bspm_message(self, layer_idx: int, covered: int, expected: int) -> str:
        return f"Layer {layer_idx}: BSPM codes cover {covered}/{expected} experts"

    def __getitem__(self, key):
        full_key = self._full_key(key)

        m = _EXPERT_KEY_RE.match(full_key)
        if m:
            layer_idx, expert_idx, proj_name = int(m.group(1)), int(m.group(2)), m.group(3)
            if layer_idx >= self._first_k_dense and proj_name in _PROJ_IDX:
                codes = self._codes_for_layer(layer_idx)
                if codes is None:
                    if self._require_complete:
                        raise ValueError(self._missing_bspm_message(layer_idx))
                elif expert_idx < codes.shape[0]:
                    return self._apply_bspm(self._base[key], codes, expert_idx, proj_name)
                elif self._require_complete:
                    expected = self._expected_experts if self._expected_experts is not None else expert_idx + 1
                    raise ValueError(self._partial_bspm_message(layer_idx, codes.shape[0], expected))
        m = _STACKED_EXPERT_KEY_RE.match(full_key)
        if m:
            layer_idx, proj_name = int(m.group(1)), m.group(2)
            if layer_idx >= self._first_k_dense and proj_name in _PROJ_IDX:
                codes = self._codes_for_layer(layer_idx)
                if codes is None:
                    if self._require_complete:
                        raise ValueError(self._missing_bspm_message(layer_idx))
                else:
                    stacked_weight = self._get_stacked_weight(key)
                    if stacked_weight.ndim != 3:
                        raise ValueError(
                            f"Expected stacked expert tensor '{key}' to have rank 3, got {stacked_weight.ndim}"
                        )
                    if self._require_complete:
                        expected = stacked_weight.shape[0] if self._expected_experts is None else self._expected_experts
                        if codes.shape[0] != expected:
                            raise ValueError(self._partial_bspm_message(layer_idx, codes.shape[0], expected))
                    num_bspm_experts = min(stacked_weight.shape[0], codes.shape[0])
                    return torch.stack(
                        [
                            self._apply_bspm(stacked_weight[expert_idx], codes, expert_idx, proj_name)
                            if expert_idx < num_bspm_experts
                            else stacked_weight[expert_idx]
                            for expert_idx in range(stacked_weight.shape[0])
                        ]
                    ).contiguous()
        return self._base[key]

    def __contains__(self, key):
        return key in self._base

    def __iter__(self):
        return iter(self._base)

    def __len__(self):
        return len(self._base)

    def keys(self):
        return self._base.keys()

    def items(self):
        for k in self._base:
            yield k, self[k]

    def values(self):
        for k in self._base:
            yield self[k]

    def view_with_prefix(self, prefix: str, num_layers: int | None = None) -> "_BsprnStateDict":
        view_with_prefix = getattr(self._base, "view_with_prefix", None)
        if callable(view_with_prefix):
            base_view = view_with_prefix(prefix, num_layers)
        else:
            from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict

            base_view = sub_state_dict(self._base, prefix, num_layers)

        return _BsprnStateDict(
            base_view,
            hf_config=None,
            bspm_dir=self._bspm_dir,
            bspm_model=self._bspm_model,
            variant=self._variant,
            budget=self._budget,
            _base_prefix=self._full_key(prefix),
            _codes_cache=self._codes_cache,
            _load_bspm=self._load_bspm,
            _qdq=self._qdq,
            _first_k_dense=self._first_k_dense,
            _expected_experts=self._expected_experts,
            _require_complete=self._require_complete,
        )

    # ------------------------------------------------------------------
    # Dry-run helper: validate all BSPM files without quantizing tensors
    # ------------------------------------------------------------------

    def validate_bspm_files(
        self,
        hf_config,
        *,
        require_complete: bool = False,
        cache_results: bool = True,
    ) -> int:
        """Load and validate all per-layer BSPM files. Returns count of missing files."""
        missing_files: list[Path] = []
        partial_layers: list[tuple[int, int, int]] = []
        expected_experts = getattr(hf_config, "n_routed_experts", None)
        for layer_idx in range(self._first_k_dense, hf_config.num_hidden_layers):
            codes = self._load_codes_for_layer(layer_idx, cache_result=cache_results)
            if codes is None:
                missing_files.append(self._bspm_file_for_layer(layer_idx))
            else:
                if expected_experts is not None and codes.shape[0] != expected_experts:
                    if require_complete:
                        partial_layers.append((layer_idx, codes.shape[0], expected_experts))
                    else:
                        logger.warning(
                            f"Layer {layer_idx}: BSPM codes cover {codes.shape[0]} of {expected_experts} experts; "
                            "conversion will use only the overlapping expert rows."
                        )
                logger.debug(f"Layer {layer_idx}: BSPM codes shape {codes.shape} — OK")
        if require_complete and (missing_files or partial_layers):
            issues: list[str] = []
            if missing_files:
                sample_missing = ", ".join(str(path) for path in missing_files[:2])
                extra = "" if len(missing_files) <= 2 else f" and {len(missing_files) - 2} more"
                issues.append(f"Missing BSPM files required for export: {sample_missing}{extra}")
            if partial_layers:
                sample_partial = ", ".join(
                    f"layer {layer_idx} covers {covered}/{expected} experts"
                    for layer_idx, covered, expected in partial_layers[:2]
                )
                extra = "" if len(partial_layers) <= 2 else f" and {len(partial_layers) - 2} more layers"
                issues.append(f"Incomplete BSPM coverage: {sample_partial}{extra}")
            raise ValueError(". ".join(issues) + ".")
        return len(missing_files)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_codes_for_layer(self, layer_idx: int, *, cache_result: bool) -> np.ndarray | None:
        if cache_result and layer_idx in self._codes_cache:
            return self._codes_cache[layer_idx]

        bspm_file = self._bspm_file_for_layer(layer_idx)
        if not bspm_file.exists():
            logger.warning(f"Layer {layer_idx}: BSPM file not found, using raw weights — {bspm_file}")
            if cache_result:
                self._codes_cache[layer_idx] = None
            return None

        if self._expected_experts is None:
            bspm_data = self._load_bspm(str(bspm_file))
        else:
            bspm_data = self._load_bspm(str(bspm_file), expected_n_experts=self._expected_experts)
        codes = bspm_data["codes"]
        if cache_result:
            self._codes_cache[layer_idx] = codes
        return codes

    def _codes_for_layer(self, layer_idx: int) -> np.ndarray | None:
        return self._load_codes_for_layer(layer_idx, cache_result=True)

    def _apply_bspm(self, w_orig: torch.Tensor, codes: np.ndarray, expert_idx: int, proj_name: str) -> torch.Tensor:
        """Quantize-dequantize a single expert weight tensor according to BSPM codes."""
        orig_dtype = w_orig.dtype  # capture before .float() so output matches checkpoint dtype (bf16)
        proj_idx = _PROJ_IDX[proj_name]
        w_kn = w_orig.float().numpy().T  # HF layout (N, K) → compute in (K, N)
        K, N = w_kn.shape
        tiles_h, tiles_w = K // 32, N // 32

        expert_codes = codes[expert_idx, proj_idx, : tiles_h * tiles_w].reshape(tiles_h, tiles_w)
        unique_codes = np.unique(expert_codes)
        non_bfp4 = unique_codes[unique_codes != 1]

        if not len(non_bfp4):
            return w_orig  # all tiles already bfp4 — no-op

        # (tiles_h, 32, tiles_w, 32) → (tiles_h, tiles_w, 32, 32)
        w_tiled = w_kn.reshape(tiles_h, 32, tiles_w, 32).transpose(0, 2, 1, 3).copy()

        for code in non_bfp4:
            ri, ci = np.where(expert_codes == code)
            mant_bits = _TT_CODE_TO_MANT.get(int(code), 3)
            if mant_bits is None:
                w_tiled[ri, ci] = 0.0
            else:
                w_tiled[ri, ci] = self._qdq(w_tiled[ri, ci], mant_bits)

        # (tiles_h, tiles_w, 32, 32) → (K, N) → (N, K) to restore HF layout
        w_out = torch.from_numpy(w_tiled.transpose(0, 2, 1, 3).reshape(K, N).T.copy())
        return w_out.to(orig_dtype)


def _validate_stacked_dequantized_checkpoint(model_path: Path, hf_config, *, num_layers: int | None = None) -> None:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"Stacked dequantized DeepSeek checkpoint is missing index file {index_path}.")

    with index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    first_k_dense = getattr(hf_config, "first_k_dense_replace", 3)
    expected_experts = getattr(hf_config, "n_routed_experts", None)
    num_layers = hf_config.num_hidden_layers if num_layers is None else num_layers
    required_keys = [
        f"model.layers.{layer_idx}.mlp.experts_stacked.{proj_name}.weight"
        for layer_idx in range(first_k_dense, num_layers)
        for proj_name in ("gate_proj", "down_proj", "up_proj")
    ]
    missing_keys = [key for key in required_keys if key not in weight_map]
    if missing_keys:
        sample_missing = ", ".join(missing_keys[:3])
        extra = "" if len(missing_keys) <= 3 else f" and {len(missing_keys) - 3} more"
        raise ValueError(
            f"Checkpoint at {model_path} is not a complete stacked DeepSeek checkpoint; "
            f"missing stacked expert tensors such as {sample_missing}{extra}."
        )

    invalid_shapes: list[tuple[str, tuple[int, ...]]] = []
    keys_by_shard: dict[str, list[str]] = {}
    for key in required_keys:
        keys_by_shard.setdefault(weight_map[key], []).append(key)

    for shard_name, shard_keys in keys_by_shard.items():
        shard_path = model_path / shard_name
        if not shard_path.is_file():
            raise ValueError(f"Checkpoint at {model_path} references missing shard file {shard_path}.")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                shape = tuple(handle.get_slice(key).get_shape())
                if len(shape) != 3 or (expected_experts is not None and shape[0] != expected_experts):
                    invalid_shapes.append((key, shape))

    if invalid_shapes:
        sample_invalid = ", ".join(f"{key} has shape {shape}" for key, shape in invalid_shapes[:3])
        extra = "" if len(invalid_shapes) <= 3 else f" and {len(invalid_shapes) - 3} more"
        expert_msg = "" if expected_experts is None else f" and contain exactly {expected_experts} experts"
        raise ValueError(
            f"Checkpoint at {model_path} has invalid stacked expert tensors; "
            f"each tensor must be rank 3{expert_msg}. Examples: {sample_invalid}{extra}."
        )


def _resolve_stacked_dequantized_model_path(model_path: Path, hf_config) -> Path:
    from models.demos.deepseek_v3.utils.hf_model_utils import (
        _load_model_weight_map,
        default_stacked_dequantized_model_path,
    )

    explicit_path = model_path.expanduser()
    default_path = default_stacked_dequantized_model_path(model_path).expanduser()

    def _looks_like_quantized_source_checkpoint(path: Path) -> bool:
        try:
            weight_map, _ = _load_model_weight_map(path)
        except Exception:
            return False
        return any(key.endswith("_scale_inv") for key in weight_map)

    if explicit_path.exists():
        try:
            _validate_stacked_dequantized_checkpoint(explicit_path, hf_config)
            return explicit_path.resolve()
        except ValueError:
            if not _looks_like_quantized_source_checkpoint(explicit_path):
                raise

    if default_path != explicit_path and default_path.exists():
        _validate_stacked_dequantized_checkpoint(default_path, hf_config)
        return default_path.resolve()

    raise FileNotFoundError(
        f"Stacked dequantized DeepSeek checkpoint not found at {default_path}. "
        "Run scripts/dequantize_hf_checkpoint.py <source-model-path> --stack-experts first."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert DeepSeek expert weights to a BSPM-pre-quantized tt-metal weight cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the HF model dir or stacked dequantized checkpoint.",
    )
    parser.add_argument("--output-path", type=Path, required=True, help="Output weight cache directory.")
    parser.add_argument(
        "--bspm-dir", type=Path, required=True, help="BitSculpt results root (contains <bspm-model>/ subdirs)."
    )
    parser.add_argument(
        "--bspm-model", type=str, required=True, help="BSPM model sub-directory, e.g. deepseek-r1-0528."
    )
    parser.add_argument("--bspm-variant", type=str, default="B", help="BSPM variant letter.")
    parser.add_argument("--bspm-budget", type=float, default=3.5, help="Bits-per-element budget.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate BSPM files exist and are loadable (no device / no conversion).",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()

    # ── HF config ───────────────────────────────────────────────────────────
    hf_config = AutoConfig.from_pretrained(args.model_path.resolve(), trust_remote_code=True)
    logger.info(f"Model: {hf_config.num_hidden_layers} layers, {hf_config.n_routed_experts} experts/layer")

    # ── Resolve dequantized model path ──────────────────────────────────────
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict

    try:
        deq_path = _resolve_stacked_dequantized_model_path(args.model_path, hf_config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Loading dequantized weights from {deq_path}")
    state_dict = LazyStateDict(deq_path)

    # ── Build lazy BSPM state dict ──────────────────────────────────────────
    # Quantization is computed on demand in __getitem__; no preprocessing loop.
    logger.info(
        f"Building lazy BSPM state dict: variant={args.bspm_variant}, budget={args.bspm_budget} b/e, "
        f"layers {getattr(hf_config, 'first_k_dense_replace', 3)}–{hf_config.num_hidden_layers - 1}"
    )
    bspm_state_dict = _BsprnStateDict(
        state_dict,
        hf_config,
        args.bspm_dir,
        args.bspm_model,
        args.bspm_variant,
        args.bspm_budget,
        _require_complete=True,
    )

    if args.dry_run:
        logger.info("--dry-run: validating BSPM files …")
        t0 = time.time()
        try:
            missing = bspm_state_dict.validate_bspm_files(
                hf_config,
                require_complete=True,
                cache_results=False,
            )
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
        logger.info(
            f"Validation complete in {time.time() - t0:.1f}s — "
            f"{hf_config.num_hidden_layers - getattr(hf_config, 'first_k_dense_replace', 3) - missing} OK, "
            f"{missing} missing"
        )
        return

    try:
        bspm_state_dict.validate_bspm_files(
            hf_config,
            require_complete=True,
            cache_results=False,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Device setup ────────────────────────────────────────────────────────
    import ttnn
    from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel, get_fabric_config
    from models.demos.deepseek_v3.utils.weight_config import get_weight_config

    mesh_shape_env = os.environ.get("MESH_DEVICE", "TG")
    from models.demos.deepseek_v3.utils.test_utils import SYSTEM_NAME_TO_MESH_SHAPE

    mesh_shape = SYSTEM_NAME_TO_MESH_SHAPE.get(mesh_shape_env, (4, 8))
    logger.info(f"Opening mesh device {mesh_shape[0]}×{mesh_shape[1]} ({mesh_shape_env})")

    device_params = {"fabric_config": get_fabric_config()}
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
        **device_params,
    )

    try:
        output_path = args.output_path
        if output_path.exists() and not args.force:
            logger.error(f"Output path {output_path} already exists. Use --force to overwrite.")
            sys.exit(1)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting weights → {output_path}")
        t1 = time.time()
        get_weight_config(
            RowBatchedModel,
            hf_config,
            (bspm_state_dict,),
            output_path,
            mesh_device,
            force_recalculate=True,
            emit_weight_cache=True,
        )
        logger.info(f"Weight conversion done in {time.time() - t1:.1f}s")
        logger.info(f"BSPM weight cache written to {output_path}")
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
