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
        --output-path /localdev/mtairum/deepseek_cache/bspm_B_3.5 \\
        --bspm-dir    /localdev/mtairum/bit_sculpt/results \\
        --bspm-model  deepseek-r1-0528

    # Custom variant / budget:
        --bspm-variant A --bspm-budget 4.0

    # Dry-run: preprocessing only, no device conversion (fast sanity check):
        --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig

# ---------------------------------------------------------------------------
# Helpers shared with test_bspm_demo.py
# ---------------------------------------------------------------------------

_TT_CODE_TO_MANT: dict[int, int | None] = {0: 7, 1: 3, 2: 1, 3: None}
_PROJ_IDX = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}


class _OverrideStateDict:
    """Lazy state-dict wrapper that returns per-key overrides and delegates
    everything else to the base mapping without materialising it."""

    def __init__(self, base, overrides: dict):
        self._base = base
        self._overrides = overrides

    def __getitem__(self, key):
        return self._overrides[key] if key in self._overrides else self._base[key]

    def __contains__(self, key):
        return key in self._overrides or key in self._base

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


def _preprocess_all_layers(
    state_dict,
    hf_config,
    bspm_dir: Path,
    bspm_model: str,
    variant: str,
    budget: float,
    bspm_root: Path,
    n_io_workers: int = 16,
) -> _OverrideStateDict:
    """Apply BSPM tile pre-quantization to all MoE layers in state_dict.

    Strategy: iterate by *projection* (not expert) so all 256 expert weights
    for a given projection are loaded in parallel and processed in one batched
    numpy call, rather than 256 sequential single-expert calls.

    Per layer: 3 projections × O(unique_codes) numpy calls
    vs original: 256 experts × 3 projections × O(unique_codes) calls

    Returns an _OverrideStateDict wrapping the original (lazy) state_dict so
    the full checkpoint is never materialised in memory.
    """
    sys.path.insert(0, str(bspm_root))
    from integration.ttnn.bspm_loader import load_bspm_for_layer

    from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import quantize_dequantize_bfp

    first_k_dense = getattr(hf_config, "first_k_dense_replace", 3)
    n_moe_layers = hf_config.num_hidden_layers - first_k_dense
    overrides: dict = {}

    layer_iter = tqdm(
        range(first_k_dense, hf_config.num_hidden_layers),
        desc="BSPM preprocessing layers",
        unit="layer",
        total=n_moe_layers,
    )
    for layer_idx in layer_iter:
        bspm_file = (
            bspm_dir
            / bspm_model
            / f"layer_{layer_idx}"
            / "precision_eval"
            / f"precision_map_{variant}_{budget:.1f}.bspm"
        )
        if not bspm_file.exists():
            logger.warning(f"Layer {layer_idx}: BSPM file not found, using raw weights — {bspm_file}")
            continue

        bspm_data = load_bspm_for_layer(str(bspm_file))
        bspm_codes = bspm_data["codes"]  # (n_experts, 3, tiles_per_proj)
        n_experts = min(hf_config.n_routed_experts, bspm_codes.shape[0])

        t_layer = time.time()

        for proj_name, proj_idx in _PROJ_IDX.items():
            keys = [f"model.layers.{layer_idx}.mlp.experts.{e}.{proj_name}.weight" for e in range(n_experts)]
            present = [(e, k) for e, k in enumerate(keys) if k in state_dict]
            if not present:
                continue

            expert_indices, present_keys = zip(*present)

            # ── Load all expert weights for this projection in parallel ──────
            # I/O-bound: threads hide NFS/disk latency effectively.
            def _load(k):
                return state_dict[k].float()

            with ThreadPoolExecutor(max_workers=n_io_workers) as pool:
                tensors = list(pool.map(_load, present_keys))

            # Stack: (n_e, N, K) → transpose to (n_e, K, N)
            w_nk = torch.stack(tensors).numpy()  # (n_e, N, K)
            w_kn = w_nk.transpose(0, 2, 1)  # (n_e, K, N) — view, no copy yet
            n_e, K, N = w_kn.shape
            tiles_h, tiles_w = K // 32, N // 32

            # codes: (n_e, tiles_h, tiles_w)
            codes_3d = bspm_codes[list(expert_indices), proj_idx, : tiles_h * tiles_w].reshape(n_e, tiles_h, tiles_w)

            unique_codes = np.unique(codes_3d)
            non_bfp4 = unique_codes[unique_codes != 1]
            if len(non_bfp4) == 0:
                continue  # all tiles bfp4 — no-op for all experts

            # (n_e, tiles_h, tiles_w, 32, 32) — contiguous copy needed for writes
            w_tiled = w_kn.reshape(n_e, tiles_h, 32, tiles_w, 32).transpose(0, 1, 3, 2, 4).copy()

            for code in non_bfp4:
                ei, ri, ci = np.where(codes_3d == code)
                mant_bits = _TT_CODE_TO_MANT.get(int(code), 3)
                if mant_bits is None:
                    w_tiled[ei, ri, ci] = 0.0
                else:
                    # Single batched call across ALL experts for this code
                    w_tiled[ei, ri, ci] = quantize_dequantize_bfp(w_tiled[ei, ri, ci], mant_bits)

            # Restore (n_e, K, N) → (n_e, N, K) and store overrides
            w_out = w_tiled.transpose(0, 1, 3, 2, 4).reshape(n_e, K, N).transpose(0, 2, 1)
            orig_dtype = tensors[0].dtype
            for i, key in enumerate(present_keys):
                overrides[key] = torch.from_numpy(w_out[i].copy()).to(orig_dtype)

            del tensors, w_nk, w_kn, w_tiled, w_out

        logger.debug(f"Layer {layer_idx} preprocessed in {time.time() - t_layer:.1f}s")

    logger.info(f"BSPM preprocessing complete: {len(overrides)} expert weight keys overridden")
    return _OverrideStateDict(state_dict, overrides)


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
        help="Run BSPM preprocessing only (no device / no conversion). Useful for timing and sanity checks.",
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

    # ── BSPM preprocessing ──────────────────────────────────────────────────
    t0 = time.time()
    logger.info(
        f"Applying BSPM pre-quantization: variant={args.bspm_variant}, budget={args.bspm_budget} b/e, "
        f"layers {getattr(hf_config, 'first_k_dense_replace', 3)}–{hf_config.num_hidden_layers - 1}"
    )
    bspm_state_dict = _preprocess_all_layers(
        state_dict,
        hf_config,
        args.bspm_dir,
        args.bspm_model,
        args.bspm_variant,
        args.bspm_budget,
        bspm_root,
    )
    logger.info(f"BSPM preprocessing done in {time.time() - t0:.1f}s")

    if args.dry_run:
        logger.info("--dry-run: skipping device conversion. Done.")
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
