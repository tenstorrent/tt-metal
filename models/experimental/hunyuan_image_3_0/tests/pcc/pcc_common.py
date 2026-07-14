# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Shared PCC / ISL test helpers for HunyuanImage-3.0 PCC tests.

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import torch

from models.common.utility_functions import comp_pcc

ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH = ROOT / "models/experimental/hunyuan_image_3_0/ref/tokenizer/assets/config.json"

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
PCC_LOGIT_PREFILL = 0.85  # 32L chained last-token logits at S=4160 (conservative)
PCC_LOGIT_MAX_CONTEXT = 0.85  # 32L last-token logits at tile-aligned max context

PRODUCTION_SEQ = 4160  # 32 + 64×64 + 32

# Production ISL for submodule PCC gates (decode + prefill at full image layout).
PRODUCTION_PHASE_CASES = [
    (1, "decode S=1"),
    (PRODUCTION_SEQ, "production prefill S=4160 (32+64×64+32)"),
]

# 64×64 image token span in production T2I layout (32 text pre + 4096 image + 32 text post).
PRODUCTION_IMAGE_INFOS = [[(slice(32, 32 + 64 * 64), (64, 64))]]
PRODUCTION_IMAGE_SPAN = [[slice(32, 32 + 64 * 64)]]

# Submodule gates: decode, text prefill, and image-layout prefill @ S=4160.
PRODUCTION_MODULE_CASES = [
    ("text", 1, None, "decode S=1"),
    ("text", PRODUCTION_SEQ, None, "production prefill S=4160 text-only"),
    ("image", PRODUCTION_SEQ, PRODUCTION_IMAGE_INFOS, "production prefill S=4160 64×64 span"),
]

# Lean ISL sweep: decode, one tile, full image grid, production layout.
# Max context (S=22784) is covered by dedicated @pytest.mark.slow tests.
LEAN_ISL_CASES = [
    (1, 1, "decode S=1"),
    (1, 32, "one tile S=32"),
    (1, 4096, "image grid S=4096 (64×64 tokens)"),
    (1, PRODUCTION_SEQ, "production S=4160 (32+64×64+32)"),
]

# One batch>1 smoke per linear/rope subsystem (attention stays B=1 on TT).
BATCH_CASE = (2, 32, "batch=2 S=32")

IMAGE_MODE_CASES = [
    (1, 4096, [[(slice(0, 4096), (64, 64))]], "64×64 full-seq image grid"),
    (1, PRODUCTION_SEQ, [[(slice(32, 32 + 64 * 64), (64, 64))]], "64×64 + text pre/post @ S=4160"),
]

# Merged text/image PCC cases for RoPE and attention.
ROPE_ATTN_PCC_CASES = [("text", batch, seq_len, None, label) for batch, seq_len, label in LEAN_ISL_CASES] + [
    ("image", batch, seq_len, image_infos, label) for batch, seq_len, image_infos, label in IMAGE_MODE_CASES
]

MASK_CASES = [
    ("causal only         S=32", 32, [[]]),
    ("one image span      S=32", 32, [[slice(4, 20)]]),
    ("two image spans     S=64", 64, [[slice(3, 11), slice(40, 56)]]),
    ("bsz=2 same spans    S=32", 32, [[slice(5, 12)], [slice(5, 12)]]),
    ("bsz=2 diff spans    S=32", 32, [[slice(2, 8)], [slice(16, 28)]]),
    ("production layout   S=4160", PRODUCTION_SEQ, [[slice(32, 32 + 64 * 64)]]),
]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def max_seq_tile_aligned(cfg: dict | None = None) -> int:
    cfg = cfg or load_config()
    return (int(cfg["max_position_embeddings"]) // 32) * 32


def model_dims(cfg: dict | None = None) -> tuple[int, int, int, int]:
    cfg = cfg or load_config()
    return (
        cfg["hidden_size"],
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["attention_head_dim"],
    )


def _first_cfg_val(value):
    return value if isinstance(value, int) else value[0]


def transformer_cfg(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    h = cfg["hidden_size"]
    return {
        "H": h,
        "HEADS": cfg["num_attention_heads"],
        "KV": cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        "HD": cfg.get("attention_head_dim", h // cfg["num_attention_heads"]),
        "E": _first_cfg_val(cfg["num_experts"]),
        "K": _first_cfg_val(cfg["moe_topk"]),
        "MOE_INTER": _first_cfg_val(cfg["moe_intermediate_size"]),
        "NUM_SHARED": _first_cfg_val(cfg.get("num_shared_expert", 1)),
        "NORM_TOPK": cfg.get("norm_topk_prob", True),
        "MIXED": cfg.get("use_mixed_mlp_moe", True),
        "QKN": cfg.get("use_qk_norm", True),
        "EPS": cfg.get("rms_norm_eps", 1e-5),
        "MAX_SEQ": int(cfg["max_position_embeddings"]),
    }


def image_slices_from_infos(image_infos):
    if not image_infos:
        return None
    return [[span for span, _ in row] for row in image_infos]


# Merged text/image PCC cases for decoder layer (same ISL grid as attention).
DECODER_LAYER_PCC_CASES = ROPE_ATTN_PCC_CASES

# Latent spatial grids for patch_embed / final_layer (real checkpoint weights).
PATCH_GRID_FAST = [(8, "smoke GRID=8")]
PATCH_GRID_SLOW = [(64, "production GRID=64 (1024² latent)")]

TIMESTEP_EMBEDDER_PREFIXES = ["timestep_emb", "time_embed", "time_embed_2"]
TIMESTEP_RTL = 0.01

# MoE block ISL (router + expert FFN + full MoE layer).
MOE_ISL_FAST = [(1, "decode S=1"), (32, "one tile S=32")]
MOE_ISL_SLOW = [(4096, "image grid S=4096")]
MOE_ISL_PRODUCTION = [
    (1, "decode S=1"),
    (PRODUCTION_SEQ, "production prefill S=4160 (32+64×64+32)"),
]
# Max-context MoE (S≈22784) is gated in test_full_dim_moe_denoise.py.

MOE_SET_MATCH = 0.99
MOE_WEIGHT_PCC = 0.999
MOE_PARALLEL_PCC = 0.99
MOE_PARALLEL_PCC_BF8 = 0.97
MOE_TIE_EPS = 2e-3

# Denoise pipeline layouts: text_pre + grid² + text_post.
TEXT_PRE = 32
TEXT_POST = 32


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
PIPELINE_LAYOUT_PROD = pipeline_layout(64)


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
