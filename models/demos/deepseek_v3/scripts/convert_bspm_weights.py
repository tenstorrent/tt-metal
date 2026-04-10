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
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from transformers import AutoConfig

# ---------------------------------------------------------------------------
# Helpers shared with test_bspm_demo.py
# ---------------------------------------------------------------------------

_TT_CODE_TO_MANT: dict[int, int | None] = {0: 7, 1: 3, 2: 1, 3: None}
_PROJ_IDX = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
_EXPERT_KEY_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.weight$")


class _BsprnStateDict:
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
        bspm_root: Path,
    ):
        self._base = base
        self._first_k_dense = getattr(hf_config, "first_k_dense_replace", 3)
        self._bspm_dir = bspm_dir
        self._bspm_model = bspm_model
        self._variant = variant
        self._budget = budget
        self._codes_cache: dict[int, np.ndarray | None] = {}  # layer → codes or None

        sys.path.insert(0, str(bspm_root))
        # Import once here so failures surface immediately, not mid-conversion.
        from integration.ttnn.bspm_loader import load_bspm_for_layer

        from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import quantize_dequantize_bfp

        self._load_bspm = load_bspm_for_layer
        self._qdq = quantize_dequantize_bfp

    # ------------------------------------------------------------------
    # Public dict-like interface (delegates to base for non-expert keys)
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        m = _EXPERT_KEY_RE.match(key)
        if m:
            layer_idx, expert_idx, proj_name = int(m.group(1)), int(m.group(2)), m.group(3)
            if layer_idx >= self._first_k_dense and proj_name in _PROJ_IDX:
                codes = self._codes_for_layer(layer_idx)
                if codes is not None and expert_idx < codes.shape[0]:
                    return self._apply_bspm(self._base[key], codes, expert_idx, proj_name)
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

    # ------------------------------------------------------------------
    # Dry-run helper: validate all BSPM files without quantizing tensors
    # ------------------------------------------------------------------

    def validate_bspm_files(self, hf_config) -> int:
        """Load and validate all per-layer BSPM files.  Returns count of missing files."""
        missing = 0
        for layer_idx in range(self._first_k_dense, hf_config.num_hidden_layers):
            codes = self._codes_for_layer(layer_idx)
            if codes is None:
                missing += 1
            else:
                logger.debug(f"Layer {layer_idx}: BSPM codes shape {codes.shape} — OK")
        return missing

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _codes_for_layer(self, layer_idx: int) -> np.ndarray | None:
        if layer_idx not in self._codes_cache:
            bspm_file = (
                self._bspm_dir
                / self._bspm_model
                / f"layer_{layer_idx}"
                / "precision_eval"
                / f"precision_map_{self._variant}_{self._budget:.1f}.bspm"
            )
            if not bspm_file.exists():
                logger.warning(f"Layer {layer_idx}: BSPM file not found, using raw weights — {bspm_file}")
                self._codes_cache[layer_idx] = None
            else:
                self._codes_cache[layer_idx] = self._load_bspm(str(bspm_file))["codes"]
        return self._codes_cache[layer_idx]

    def _apply_bspm(self, w_orig: torch.Tensor, codes: np.ndarray, expert_idx: int, proj_name: str) -> torch.Tensor:
        """Quantize-dequantize a single expert weight tensor according to BSPM codes."""
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
        return w_out.to(w_orig.dtype)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert DeepSeek expert weights to a BSPM-pre-quantized tt-metal weight cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the HF model dir (FP8 or dequantized).")
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

    # ── Resolve dequantized model path ──────────────────────────────────────
    from models.demos.deepseek_v3.utils.hf_model_utils import default_dequantized_model_path
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict

    deq_path = default_dequantized_model_path(args.model_path)
    if not deq_path.exists():
        logger.error(
            f"Dequantized checkpoint not found at {deq_path}. " f"Run scripts/dequantize_hf_checkpoint.py first."
        )
        sys.exit(1)

    logger.info(f"Loading dequantized weights from {deq_path}")
    state_dict = LazyStateDict(deq_path)

    # ── HF config ───────────────────────────────────────────────────────────
    hf_config = AutoConfig.from_pretrained(args.model_path.resolve(), trust_remote_code=True)
    logger.info(f"Model: {hf_config.num_hidden_layers} layers, {hf_config.n_routed_experts} experts/layer")

    # ── Derive bspm_root from bspm_dir ──────────────────────────────────────
    bspm_root = args.bspm_dir.parent

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
        bspm_root,
    )

    if args.dry_run:
        logger.info("--dry-run: validating BSPM files …")
        t0 = time.time()
        missing = bspm_state_dict.validate_bspm_files(hf_config)
        logger.info(
            f"Validation complete in {time.time() - t0:.1f}s — "
            f"{hf_config.num_hidden_layers - getattr(hf_config, 'first_k_dense_replace', 3) - missing} OK, "
            f"{missing} missing"
        )
        return

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
        )
        logger.info(f"Weight conversion done in {time.time() - t1:.1f}s")
        logger.info(f"BSPM weight cache written to {output_path}")
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
