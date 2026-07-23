# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Shared PCC / ISL test helpers for HunyuanImage-3.0 PCC tests.

from __future__ import annotations

import csv
import os
from pathlib import Path

import torch

from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.model_config import (
    IMAGE_BASE_SIZE,
    PRODUCTION_IMAGE_TOKENS,
    PRODUCTION_LATENT_GRID,
    PRODUCTION_SEQ,
    PRODUCTION_TEXT_POST,
    PRODUCTION_TEXT_PRE,
    TILE_SIZE,
    load_config,
    production_image_infos,
    production_image_span,
)

ROOT = Path(__file__).resolve().parents[5]

PCC_STRICT = 0.999
PCC_BLOCK = 0.99
PCC_PIPELINE = 0.98
PCC_CHAINED = 0.86
PCC_DECODE_STACK = 0.96  # wte+N layers+ln_f at S=1 (single-token smoke)
# Per-layer teacher-forced PCC at S=1 (decode). A single token's top-k MoE routing
# can flip on a bf16 tie-break: the router input is the bf16 post-attention-layernorm
# output (~1e-2 off fp32), which is enough to reverse a near-tie between experts whose
# softmax gates differ by ~1e-4. That swaps ~1/topk of the token's MoE output. With
# only one row there is nothing to average it against, so the flip dominates PCC (~0.98);
# at S>=32 the same per-token error is diluted and the strict PCC_BLOCK gate holds.
PCC_PER_LAYER_DECODE = 0.96
# Teacher-forced 32L wte→layers→ln_f→lm_head at S=1 (free-running S=1 drifts ~0.59).
PCC_LOGIT_DECODE = 0.96
PCC_LOGIT_PREFILL = 0.85  # 32L chained last-token logits at production seq
PCC_LOGIT_MAX_CONTEXT = 0.85  # 32L last-token logits at tile-aligned max context

# Production ISL for submodule PCC gates (decode + prefill at full image layout).
_G = PRODUCTION_LATENT_GRID
_PRE = PRODUCTION_TEXT_PRE
_IMG = PRODUCTION_IMAGE_TOKENS
PRODUCTION_IMAGE_INFOS = production_image_infos()
PRODUCTION_IMAGE_SPAN = production_image_span()
PRODUCTION_PHASE_CASES = [
    (1, "decode S=1"),
    (PRODUCTION_SEQ, f"production prefill S={PRODUCTION_SEQ} ({_PRE}+{_G}×{_G}+{PRODUCTION_TEXT_POST})"),
]

# Submodule gates: decode, text prefill, and image-layout prefill @ production seq.
PRODUCTION_MODULE_CASES = [
    ("text", 1, None, "decode S=1"),
    ("text", PRODUCTION_SEQ, None, f"production prefill S={PRODUCTION_SEQ} text-only"),
    ("image", PRODUCTION_SEQ, PRODUCTION_IMAGE_INFOS, f"production prefill S={PRODUCTION_SEQ} {_G}×{_G} span"),
]

# Lean ISL sweep: decode, one tile, full image grid, production layout.
# Max context is covered by dedicated @pytest.mark.slow tests.
LEAN_ISL_CASES = [
    (1, 1, "decode S=1"),
    (1, TILE_SIZE, f"one tile S={TILE_SIZE}"),
    (1, _IMG, f"image grid S={_IMG} ({_G}×{_G} tokens)"),
    (1, PRODUCTION_SEQ, f"production S={PRODUCTION_SEQ} ({_PRE}+{_G}×{_G}+{PRODUCTION_TEXT_POST})"),
]

# One batch>1 smoke per linear/rope subsystem (attention stays B=1 on TT).
BATCH_CASE = (2, TILE_SIZE, f"batch=2 S={TILE_SIZE}")

IMAGE_MODE_CASES = [
    (1, _IMG, [[(slice(0, _IMG), (_G, _G))]], f"{_G}×{_G} full-seq image grid"),
    (
        1,
        PRODUCTION_SEQ,
        PRODUCTION_IMAGE_INFOS,
        f"{_G}×{_G} + text pre/post @ S={PRODUCTION_SEQ}",
    ),
]

# Merged text/image PCC cases for RoPE and attention.
ROPE_ATTN_PCC_CASES = [("text", batch, seq_len, None, label) for batch, seq_len, label in LEAN_ISL_CASES] + [
    ("image", batch, seq_len, image_infos, label) for batch, seq_len, image_infos, label in IMAGE_MODE_CASES
]

MASK_CASES = [
    (f"causal only         S={TILE_SIZE}", TILE_SIZE, [[]]),
    (f"one image span      S={TILE_SIZE}", TILE_SIZE, [[slice(4, 20)]]),
    (f"two image spans     S={TILE_SIZE * 2}", TILE_SIZE * 2, [[slice(3, 11), slice(40, 56)]]),
    (f"bsz=2 same spans    S={TILE_SIZE}", TILE_SIZE, [[slice(5, 12)], [slice(5, 12)]]),
    (f"bsz=2 diff spans    S={TILE_SIZE}", TILE_SIZE, [[slice(2, 8)], [slice(16, 28)]]),
    (f"production layout   S={PRODUCTION_SEQ}", PRODUCTION_SEQ, PRODUCTION_IMAGE_SPAN),
]


def model_dims(cfg: dict | None = None) -> tuple[int, int, int, int]:
    cfg = cfg or load_config()
    return (
        cfg["hidden_size"],
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["attention_head_dim"],
    )


def image_slices_from_infos(image_infos):
    if not image_infos:
        return None
    return [[span for span, _ in row] for row in image_infos]


# Merged text/image PCC cases for decoder layer (same ISL grid as attention).
DECODER_LAYER_PCC_CASES = ROPE_ATTN_PCC_CASES

# Latent spatial grids for patch_embed / final_layer (real checkpoint weights).
PATCH_GRID_FAST = [(8, "smoke GRID=8")]
PATCH_GRID_SLOW = [(_G, f"production GRID={_G} ({IMAGE_BASE_SIZE}² latent)")]

TIMESTEP_EMBEDDER_PREFIXES = ["timestep_emb", "time_embed", "time_embed_2"]
TIMESTEP_RTL = 0.01

# MoE block ISL (router + expert FFN + full MoE layer).
MOE_ISL_FAST = [(1, "decode S=1"), (TILE_SIZE, f"one tile S={TILE_SIZE}")]
MOE_ISL_SLOW = [(_IMG, f"image grid S={_IMG}")]
MOE_ISL_PRODUCTION = [
    (1, "decode S=1"),
    (PRODUCTION_SEQ, f"production prefill S={PRODUCTION_SEQ} ({_PRE}+{_G}×{_G}+{PRODUCTION_TEXT_POST})"),
]
# Max-context MoE (S=MAX_SEQ_TILE_ALIGNED) is gated in test_full_dim_moe_denoise.py.

MOE_SET_MATCH = 0.99
MOE_WEIGHT_PCC = 0.999
MOE_PARALLEL_PCC = 0.99
MOE_PARALLEL_PCC_BF8 = 0.97
MOE_TIE_EPS = 2e-3

# Denoise pipeline layouts: text_pre + grid² + text_post.
TEXT_PRE = PRODUCTION_TEXT_PRE
TEXT_POST = PRODUCTION_TEXT_POST


def pipeline_layout(grid: int, text_pre: int = TEXT_PRE, text_post: int = TEXT_POST) -> dict:
    n_img = grid * grid
    img_start = text_pre
    return {
        "grid": grid,
        "text_pre": text_pre,
        "text_post": text_post,
        "n_img": n_img,
        "seq_len": text_pre + n_img + text_post,
        "img_start": img_start,
        "img_slice": slice(img_start, img_start + n_img),
    }


PIPELINE_LAYOUT_FAST = pipeline_layout(8)
PIPELINE_LAYOUT_PROD = pipeline_layout(PRODUCTION_LATENT_GRID)


def rope_image_infos(image_infos, batch: int):
    if image_infos is not None:
        return image_infos
    return [None] * batch


def per_layer_pcc(seq_len: int) -> float:
    """Per-layer teacher-forced PCC gate for a given ISL.

    S=1 (decode) uses PCC_PER_LAYER_DECODE: a single-token sequence has no other rows
    to average out a bf16 MoE routing tie-break, so a near-tie expert flip dominates the
    metric. S>=32 keeps the strict PCC_BLOCK gate.
    """
    return PCC_PER_LAYER_DECODE if seq_len == 1 else PCC_BLOCK


def pcc_metrics(ref: torch.Tensor, tt_out: torch.Tensor, threshold: float) -> tuple[float, float]:
    _, p = comp_pcc(ref, tt_out, threshold)
    d = (ref.float() - tt_out.float()).abs().max().item()
    return p, d


def write_isl_csv(rows: list[dict], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def isl_csv_path(default_name: str = "hunyuan_isl_sweep.csv") -> Path | None:
    if env := os.environ.get("HY_PCC_CSV"):
        return Path(env)
    if os.environ.get("HY_PCC_CSV_DIR"):
        return Path(os.environ["HY_PCC_CSV_DIR"]) / default_name
    return None
