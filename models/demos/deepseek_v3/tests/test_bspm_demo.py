# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""BSPM vs uniform-BFP4 output comparison for the deepseek_v3 demo (5 layers).

Runs the same decode forward pass twice through the deepseek_v3 RowBatchedModel:

  1. Baseline  — weights converted via the standard pipeline (uniform bfloat4_b).
  2. BSPM      — expert weights pre-quantized per-tile according to BitSculpt BSPM
                 assignments, then converted via the same standard pipeline.

The BSPM pre-quantization step simulates what CompressedTensor does on Blackhole
but without changing the kernel: each 32×32 expert-weight tile is quantized at its
BSPM-assigned precision (bfp8/bfp4/bfp2/zero) and dequantized back to float before
the standard bfloat4_b conversion runs.  The kernel thus sees a tensor whose values
are already at BSPM quality.

Environment variables
---------------------
DEEPSEEK_V3_HF_MODEL  : path to the local HF model dir (read by conftest).
BSPM_RESULTS_DIR      : BitSculpt results root, e.g. /localdev/.../bit_sculpt/results
BSPM_MODEL_NAME       : sub-directory under BSPM_RESULTS_DIR, e.g. deepseek-r1-0528
BSPM_VARIANT          : variant letter, default "B"
BSPM_BUDGET           : b/e budget as float, default 3.5

Run
---
    MESH_DEVICE=TG \
    DEEPSEEK_V3_HF_MODEL=/proj_sw/... \
    BSPM_RESULTS_DIR=/localdev/mtairum/bit_sculpt/results \
    BSPM_MODEL_NAME=deepseek-r1-0528 \
    pytest models/demos/deepseek_v3/tests/test_bspm_demo.py -v
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel, get_fabric_config
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.hf_model_utils import default_dequantized_model_path
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    get_model_config,
    get_rope_tensors,
    get_test_weight_config,
    paged_caches_from_torch,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_LAYERS = 5  # Override: run only 5 layers instead of 61
MAX_SEQ_LEN = 128  # Trim KV cache tables for faster setup

# PCC thresholds
PCC_UNIFORM_VS_BSPM = 0.93  # direct comparison: uniform bfp4 vs BSPM pre-quantized
PCC_BSPM_VS_REF = 0.95  # BSPM vs float reference (lower than baseline due to aggressive tiles)
PCC_BASELINE_VS_REF = 0.97  # baseline vs float reference (same as test_model.py)


# ---------------------------------------------------------------------------
# Lazy state-dict wrapper that allows per-key overrides without materializing
# the full underlying mapping
# ---------------------------------------------------------------------------


class _OverrideStateDict:
    """Thin wrapper around a base Mapping that returns overridden values for
    specific keys and delegates all other key accesses to the base mapping.

    This avoids materializing the full (potentially multi-GB) state dict when
    only a small subset of keys (MoE expert weights) need to be modified.
    """

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


# ---------------------------------------------------------------------------
# BSPM preprocessing helper
# ---------------------------------------------------------------------------


def _preprocess_experts_with_bspm(
    state_dict,
    hf_config,
    bspm_results_dir: Path,
    bspm_model_name: str,
    variant: str = "B",
    budget: float = 3.5,
    num_layers: int = NUM_LAYERS,
) -> dict:
    """Return a new state dict where each MoE expert-weight tile is pre-quantized
    according to its BSPM assignment and dequantized back to float32.

    Tile layout
    -----------
    HF expert weights are stored as (out_features, in_features), i.e. (N, K).
    BitSculpt produces BSPM codes in (K, N) row-major tile order (tile index =
    row * tiles_w + col, with the K dimension as rows).  We transpose to (K, N)
    before applying tiles and transpose back afterwards.

    BSPM code → mantissa bits mapping (tt-metal convention after remap)
    -------------------------------------------------------------------
    0 → bfp8  (7 mantissa bits, near-lossless)
    1 → bfp4  (3 mantissa bits, default allocation)
    2 → bfp2  (1 mantissa bit,  low-saliency tiles)
    3 → zero  (tile zeroed out, dead/pruned tiles)
    """
    try:
        from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import quantize_dequantize_bfp
    except ImportError as e:
        pytest.skip(f"Required module not importable for BSPM preprocessing: {e}")

    # Lazy import — derive the bit_sculpt root from BSPM_RESULTS_DIR (most reliable)
    # or fall back to a sibling-of-tt-metal guess.
    bspm_root = Path(os.environ.get("BSPM_RESULTS_DIR", "")).parent
    if not bspm_root.exists():
        bspm_root = Path(__file__).resolve().parents[4].parent / "bit_sculpt"
    sys.path.insert(0, str(bspm_root))
    try:
        from integration.ttnn.bspm_loader import load_bspm_for_layer
    except ImportError as e:
        pytest.skip(f"bspm_loader not importable: {e}")

    # tt-metal code → mantissa bits (None = zero tile)
    TT_CODE_TO_MANT: dict[int, int | None] = {0: 7, 1: 3, 2: 1, 3: None}
    # HF projection name → BSPM proj_idx (0=gate, 1=up, 2=down)
    PROJ_IDX = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}

    first_k_dense = getattr(hf_config, "first_k_dense_replace", 3)

    # Build an override dict; all other keys are lazily delegated to state_dict
    # (avoids materializing the full multi-GB weight mapping in memory)
    overrides: dict = {}

    layers_processed = 0
    for layer_idx in range(first_k_dense, num_layers):
        bspm_file = (
            bspm_results_dir
            / bspm_model_name
            / f"layer_{layer_idx}"
            / "precision_eval"
            / f"precision_map_{variant}_{budget:.1f}.bspm"
        )
        if not bspm_file.exists():
            logger.warning(f"BSPM file not found for layer {layer_idx}: {bspm_file}")
            continue

        # dict with "codes": (n_experts, 3, tiles_per_proj) uint8 in tt-metal code space
        bspm_data = load_bspm_for_layer(str(bspm_file))
        bspm_codes = bspm_data["codes"]  # np.ndarray (n_experts, 3, tiles_per_proj)
        n_experts_bspm = bspm_codes.shape[0]
        n_experts = min(hf_config.n_routed_experts, n_experts_bspm)

        from tqdm import tqdm

        for expert_idx in tqdm(range(n_experts), desc=f"layer {layer_idx} experts", unit="expert", leave=True):
            for proj_name, proj_idx in PROJ_IDX.items():
                key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.weight"
                if key not in state_dict:
                    continue

                # Load as float32 — state_dict must be dequantized (BF16) before calling this
                w_f32 = overrides[key].float() if key in overrides else state_dict[key].float()  # (N, K)

                # Transpose to (K, N) — BitSculpt's tile convention
                w_kn = w_f32.T.contiguous().numpy()  # (K, N)
                K, N = w_kn.shape
                tiles_h, tiles_w = K // 32, N // 32
                assert (
                    bspm_codes.shape[2] >= tiles_h * tiles_w
                ), f"BSPM has {bspm_codes.shape[2]} tiles but weight needs {tiles_h * tiles_w}"
                codes_1d = bspm_codes[expert_idx, proj_idx, : tiles_h * tiles_w]

                # Apply per-tile quantization — batched by code value.
                # code=1 (bfp4) is skipped: the standard pipeline encodes to bfp4
                # anyway, so it's a no-op.  Only non-bfp4 tiles need preprocessing.
                codes_2d = codes_1d.reshape(tiles_h, tiles_w)
                unique_codes = np.unique(codes_2d)
                non_bfp4_codes = unique_codes[unique_codes != 1]
                if len(non_bfp4_codes) == 0:
                    continue  # all tiles are bfp4 — skip weight entirely

                # (tiles_h, tiles_w, 32, 32) — easier to index by (r, c)
                w_tiled = w_kn.reshape(tiles_h, 32, tiles_w, 32).transpose(0, 2, 1, 3).copy()

                for code in non_bfp4_codes:
                    rs, cs = np.where(codes_2d == code)
                    tiles_batch = w_tiled[rs, cs]  # (n_tiles, 32, 32)
                    mant_bits = TT_CODE_TO_MANT.get(int(code), 3)
                    if mant_bits is None:
                        w_tiled[rs, cs] = 0.0
                    else:
                        w_tiled[rs, cs] = quantize_dequantize_bfp(tiles_batch, mant_bits)

                # Restore (tiles_h, 32, tiles_w, 32) → (K, N) → transpose to (N, K)
                w_out = w_tiled.transpose(0, 2, 1, 3).reshape(K, N)
                overrides[key] = torch.from_numpy(w_out).T.to(w_f32.dtype)

        layers_processed += 1

    if layers_processed == 0:
        pytest.skip(
            f"No BSPM files found under {bspm_results_dir / bspm_model_name} "
            f"for layers {first_k_dense}–{num_layers - 1}"
        )

    logger.info(f"BSPM pre-quantization applied to {layers_processed} MoE layers ({len(overrides)} keys overridden)")
    return _OverrideStateDict(state_dict, overrides)


# ---------------------------------------------------------------------------
# Shared decode-step runner
# ---------------------------------------------------------------------------


def _run_one_decode_step(
    hf_config,
    mesh_device,
    ccl,
    paged_config,
    weight_config,
    torch_input: torch.Tensor,  # (seq=1, batch)
    position_ids: torch.Tensor,  # (batch,)
    torch_page_table: torch.Tensor,
) -> torch.Tensor:
    """Run one RowBatchedModel decode forward step and return logits on CPU."""
    dp_factor = mesh_device.shape[1]
    batches_per_device = USERS_PER_ROW // dp_factor
    blocks_per_batch = paged_config.max_num_blocks // batches_per_device

    # Empty KV caches (position 0, no prior context)
    cache_dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
    batch_size = int(position_ids.shape[0])
    empty_caches = tuple(
        torch.zeros((batch_size, 1, 0, cache_dim), dtype=torch.bfloat16) for _ in range(hf_config.num_hidden_layers)
    )

    mapping = torch_page_table
    paged_input_caches, _ = paged_caches_from_torch(
        empty_caches,
        tuple(mesh_device.shape),
        paged_config,
        user_id=None,
        mappings=tuple(mapping for _ in range(hf_config.num_hidden_layers)),
    )

    tt_page_tables = tuple(
        MLA2D.create_page_table(page_table=mapping, paged_config=paged_config, mesh_device=mesh_device)
        for _ in range(hf_config.num_hidden_layers)
    )

    model_config = get_model_config(RowBatchedModel, "decode", hf_config, mesh_device)
    model_state = RowBatchedModel.create_state(hf_config, paged_config, mesh_device, ccl, paged_input_caches)
    model_shared_state = RowBatchedModel.create_shared_state(hf_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    position_ids_tt = ttnn.from_torch(
        position_ids,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        dtype=ttnn.int32,
    )
    rope_tensors = get_rope_tensors(hf_config, USERS_PER_ROW, 1, position_ids, mesh_device)

    tt_output = RowBatchedModel.forward_decode(tt_input, position_ids_tt, run_config, rope_tensors, tt_page_tables)
    ttnn.synchronize_device(mesh_device)

    logits = (
        ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
        )
        .cpu()
        .float()
    )

    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)
    ttnn.deallocate(position_ids_tt)
    return logits


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": get_fabric_config()}],
    indirect=True,
)
def test_bspm_vs_uniform_5layers_decode(
    device_params,
    mesh_device,
    hf_config,
    model_path,
    cache_path,
    ccl,
    force_recalculate_weight_config,
):
    """Run the deepseek_v3 demo (5 layers, 1 decode step) with uniform bfloat4_b
    weights and with BSPM pre-quantized weights; compare output logits.

    Both runs use identical random inputs and page tables so the comparison
    isolates the effect of BSPM-assigned per-tile precision.

    Expected result
    ---------------
    - Baseline PCC vs BSPM  > 0.93  (small quality degradation from mixed precision)
    - If BSPM is well-calibrated, most token predictions stay the same; heavily
      compressed expert tiles contribute minimal cross-attention weight so the
      output distribution shifts only slightly.
    """
    # ── 5-layer config ──────────────────────────────────────────────────────
    hf_config_5 = deepcopy(hf_config)
    hf_config_5.num_hidden_layers = NUM_LAYERS
    hf_config_5.max_seq_len = MAX_SEQ_LEN

    # Load from the dequantized (BF16) checkpoint — the raw HF checkpoint for
    # DeepSeek-R1-0528 uses float8_e4m3fn; convert_weights requires BF16 tensors.
    deq_path = default_dequantized_model_path(model_path)
    if not deq_path.exists():
        pytest.skip(
            f"Dequantized checkpoint not found at {deq_path}. "
            f"Run save_dequantized_hf_checkpoint() first to create it."
        )
    deq_state_dict_5 = sub_state_dict(LazyStateDict(deq_path), "", NUM_LAYERS)

    # ── Shared random decode inputs ─────────────────────────────────────────
    dp_factor = mesh_device.shape[1]
    batch_size = USERS_PER_ROW * mesh_device.shape[0]
    paged_config = MLA2D.get_valid_paged_config(hf_config_5.max_seq_len, USERS_PER_ROW, dp_factor)

    # Random decode input: position 0, one token per user
    position_ids = torch.zeros(batch_size, dtype=torch.long)
    torch_input = torch.randint(0, hf_config_5.vocab_size - 1, (batch_size, 1), dtype=torch.long).T  # (1, batch)

    # Shared page table (same structure for both runs)
    batches_per_device = USERS_PER_ROW // dp_factor
    blocks_per_batch = paged_config.max_num_blocks // batches_per_device
    torch_page_table = torch.arange(paged_config.max_num_blocks, dtype=torch.int32).reshape(
        batches_per_device, blocks_per_batch
    )

    # ── Run 1: uniform bfloat4_b (baseline) ─────────────────────────────────
    logger.info("=== Run 1: uniform bfloat4_b (baseline) ===")
    weight_cfg_uniform = get_test_weight_config(
        RowBatchedModel,
        hf_config_5,
        (deq_state_dict_5,),
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
        test_name="test_bspm_demo",
        layer_id=f"uniform_{NUM_LAYERS}layers",
        real_weights=True,
    )
    logits_uniform = _run_one_decode_step(
        hf_config_5,
        mesh_device,
        ccl,
        paged_config,
        weight_cfg_uniform,
        torch_input,
        position_ids,
        torch_page_table,
    )
    logger.info(f"Baseline logits shape: {logits_uniform.shape}")

    # ── Run 2: BSPM pre-quantized ────────────────────────────────────────────
    bspm_results_dir = Path(os.environ.get("BSPM_RESULTS_DIR", ""))
    bspm_model_name = os.environ.get("BSPM_MODEL_NAME", "")
    bspm_variant = os.environ.get("BSPM_VARIANT", "B")
    bspm_budget = float(os.environ.get("BSPM_BUDGET", "3.5"))

    if not bspm_results_dir or not bspm_results_dir.exists() or not bspm_model_name:
        pytest.skip(
            "BSPM_RESULTS_DIR and BSPM_MODEL_NAME must be set to compare BSPM weights. "
            "Baseline run completed successfully."
        )

    logger.info(f"=== Run 2: BSPM pre-quantized " f"({bspm_model_name}, variant {bspm_variant}, {bspm_budget} b/e) ===")
    bspm_state_dict = _preprocess_experts_with_bspm(
        deq_state_dict_5,
        hf_config_5,
        bspm_results_dir,
        bspm_model_name,
        variant=bspm_variant,
        budget=bspm_budget,
        num_layers=NUM_LAYERS,
    )
    weight_cfg_bspm = get_test_weight_config(
        RowBatchedModel,
        hf_config_5,
        (bspm_state_dict,),
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
        test_name="test_bspm_demo",
        layer_id=f"bspm_{NUM_LAYERS}layers_{bspm_model_name}_{bspm_variant}_{bspm_budget}",
        real_weights=True,
    )
    logits_bspm = _run_one_decode_step(
        hf_config_5,
        mesh_device,
        ccl,
        paged_config,
        weight_cfg_bspm,
        torch_input,
        position_ids,
        torch_page_table,
    )
    logger.info(f"BSPM logits shape: {logits_bspm.shape}")

    # ── Compare ──────────────────────────────────────────────────────────────
    passing, pcc_msg = comp_pcc(logits_uniform, logits_bspm, PCC_UNIFORM_VS_BSPM)
    logger.info(f"\n{'='*60}")
    logger.info(f"Uniform bfp4  vs  BSPM ({bspm_variant} {bspm_budget} b/e):  {pcc_msg}")

    # Token-level agreement (greedy argmax)
    tokens_uniform = logits_uniform.argmax(dim=-1)  # (1, batch)
    tokens_bspm = logits_bspm.argmax(dim=-1)
    match_rate = (tokens_uniform == tokens_bspm).float().mean().item()
    logger.info(f"Top-1 token match rate (greedy): {match_rate:.3f}")
    logger.info(f"{'='*60}")

    assert passing, (
        f"BSPM logits PCC too low vs uniform baseline: {pcc_msg}. "
        f"Expected PCC > {PCC_UNIFORM_VS_BSPM}. "
        f"This may indicate the BSPM allocation is too aggressive for tt-metal's quantizer "
        f"or that the tile orientation in _preprocess_experts_with_bspm() is incorrect."
    )
    logger.info(
        f"PASSED: BSPM ({bspm_variant} {bspm_budget} b/e) output matches uniform baseline "
        f"with PCC > {PCC_UNIFORM_VS_BSPM}. Top-1 token match: {match_rate:.3f}"
    )
