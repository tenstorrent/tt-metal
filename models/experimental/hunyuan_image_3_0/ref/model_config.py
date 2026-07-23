# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Central HunyuanImage-3.0 model configuration.

Single source of truth for backbone / VAE / ViT dims: checkpoint or bundled
``ref/tokenizer/assets/config.json``. Callers should import helpers/constants
from here instead of hardcoding 4096 / 32 / 64 / scaling_factor / etc.

    Usage
    -----
    from models.experimental.hunyuan_image_3_0.ref.model_config import (
        load_config,
        transformer_cfg,
        backbone_kwargs,
        VAE_SCALING_FACTOR,
        VIT_CONFIG,
        ALIGNER_CONFIG,
        IMAGE_BASE_SIZE,
        PRODUCTION_LATENT_GRID,
        PRODUCTION_SEQ,
    )

    cfg = load_config()                    # bundled assets (tests / offline)
    cfg = load_config(model_dir)           # live checkpoint config.json
    dims = transformer_cfg(cfg)            # H, HEADS, KV, HD, E, K, ...
    kw = backbone_kwargs(cfg)              # HunyuanTtModel / DecoderLayer kwargs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_CONFIG_PATH = _PKG_ROOT / "ref" / "tokenizer" / "assets" / "config.json"

# Hardware tile size (Tenstorrent). Not a model hyperparam; used only to align
# production text padding / max-context to TILE multiples.
TILE_SIZE = 32

_cached_bundled: dict[str, Any] | None = None


def _first_cfg_val(value: Any) -> Any:
    return value if isinstance(value, (int, float, bool, str)) or value is None else value[0]


def load_config(model_dir: Path | str | None = None) -> dict[str, Any]:
    """Load HF-shaped ``config.json``.

    Prefer ``model_dir/config.json`` when given; otherwise the bundled assets
    copy (and cache it for subsequent calls).
    """
    if model_dir is not None:
        path = Path(model_dir) / "config.json"
        with open(path) as f:
            return json.load(f)

    global _cached_bundled
    if _cached_bundled is None:
        with open(BUNDLED_CONFIG_PATH) as f:
            _cached_bundled = json.load(f)
    return _cached_bundled


def transformer_cfg(cfg: dict | None = None) -> dict[str, Any]:
    """Normalized backbone dims used by tests / demos / TT constructors."""
    cfg = cfg or load_config()
    h = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    return {
        "H": h,
        "HEADS": heads,
        "KV": int(cfg.get("num_key_value_heads", heads)),
        "HD": int(cfg.get("attention_head_dim", cfg.get("head_dim", h // heads))),
        "E": int(_first_cfg_val(cfg["num_experts"])),
        "K": int(_first_cfg_val(cfg["moe_topk"])),
        "MOE_INTER": int(_first_cfg_val(cfg["moe_intermediate_size"])),
        "NUM_SHARED": int(_first_cfg_val(cfg.get("num_shared_expert", 1))),
        "NORM_TOPK": bool(cfg.get("norm_topk_prob", True)),
        # Alias kept for demos that historically used short key "NORM".
        "NORM": bool(cfg.get("norm_topk_prob", True)),
        "MIXED": bool(cfg.get("use_mixed_mlp_moe", True)),
        "QKN": bool(cfg.get("use_qk_norm", True)),
        "EPS": float(cfg.get("rms_norm_eps", 1e-5)),
        "MAX_SEQ": int(cfg["max_position_embeddings"]),
        "NUM_LAYERS": int(cfg["num_hidden_layers"]),
        "VOCAB": int(cfg["vocab_size"]),
        "ROPE_THETA": float(cfg.get("rope_theta", 10000.0)),
        "INTER": int(_first_cfg_val(cfg["moe_intermediate_size"])),
        "SHARED": int(_first_cfg_val(cfg.get("num_shared_expert", 1))),
    }


def backbone_kwargs(cfg: dict | None = None) -> dict[str, Any]:
    """Keyword args matching ``HunyuanTtModel`` / ``HunyuanTtDecoderLayer``."""
    d = transformer_cfg(cfg)
    return {
        "num_layers": d["NUM_LAYERS"],
        "hidden_size": d["H"],
        "num_heads": d["HEADS"],
        "num_kv_heads": d["KV"],
        "head_dim": d["HD"],
        "num_experts": d["E"],
        "moe_topk": d["K"],
        "use_qk_norm": d["QKN"],
        "use_mixed_mlp_moe": d["MIXED"],
        "norm_topk_prob": d["NORM_TOPK"],
        "rms_norm_eps": d["EPS"],
    }


def attention_kwargs(cfg: dict | None = None) -> dict[str, Any]:
    """Keyword args matching ``HunyuanTtAttention`` / ref attention config."""
    d = transformer_cfg(cfg)
    return {
        "hidden_size": d["H"],
        "num_heads": d["HEADS"],
        "num_kv_heads": d["KV"],
        "head_dim": d["HD"],
        "use_qk_norm": d["QKN"],
        "eps": d["EPS"],
    }


def vae_section(cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    return dict(cfg["vae"])


def vae_scaling_factor(cfg: dict | None = None) -> float:
    return float(vae_section(cfg)["scaling_factor"])


def vae_block_out_channels(cfg: dict | None = None) -> tuple[int, ...]:
    return tuple(int(c) for c in vae_section(cfg)["block_out_channels"])


def vae_mid_channels(cfg: dict | None = None) -> int:
    """VAE mid / deepest block width (= last ``block_out_channels`` entry)."""
    return vae_block_out_channels(cfg)[-1]


def vit_config(cfg: dict | None = None) -> dict[str, Any]:
    """SigLIP2 vision tower knobs (subset used by ref/tt ports)."""
    cfg = cfg or load_config()
    vit = cfg["vit"]
    keys = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_hidden_layers",
        "num_channels",
        "patch_size",
        "num_patches",
        "layer_norm_eps",
        "attention_dropout",
        "hidden_act",
    )
    return {k: vit[k] for k in keys}


def aligner_config(cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    al = cfg["vit_aligner"]
    return {
        "projector_type": al["projector_type"],
        "input_dim": al["input_dim"],
        "n_embed": al["n_embed"],
        "depth": al["depth"],
    }


def image_base_size(cfg: dict | None = None) -> int:
    """Canonical square generation size in pixels (demo / production default)."""
    cfg = cfg or load_config()
    if "image_base_size" in cfg:
        return int(cfg["image_base_size"])
    # Live HF checkpoints may omit the key; match the released demo default.
    return 1024


def latent_grid_from_pixels(pixels: int, cfg: dict | None = None) -> int:
    cfg = cfg or load_config()
    f = int(vae_section(cfg)["ffactor_spatial"])
    g = pixels // f
    if g * f != pixels:
        raise ValueError(f"pixels={pixels} not divisible by vae ffactor_spatial={f}")
    return g


def production_latent_grid(cfg: dict | None = None) -> int:
    """Latent H=W for ``image_base_size`` square generation."""
    return latent_grid_from_pixels(image_base_size(cfg), cfg)


def production_seq_len(
    cfg: dict | None = None,
    *,
    text_pre: int | None = None,
    text_post: int | None = None,
) -> int:
    """Production T2I layout length: text_pre + grid² + text_post."""
    g = production_latent_grid(cfg)
    pre = TILE_SIZE if text_pre is None else text_pre
    post = TILE_SIZE if text_post is None else text_post
    return pre + g * g + post


def production_image_infos(
    cfg: dict | None = None,
    *,
    text_pre: int | None = None,
) -> list:
    """RoPE image_infos for the production T2I layout (one 64×64 image span)."""
    g = production_latent_grid(cfg)
    pre = TILE_SIZE if text_pre is None else text_pre
    return [[(slice(pre, pre + g * g), (g, g))]]


def production_image_span(
    cfg: dict | None = None,
    *,
    text_pre: int | None = None,
) -> list:
    g = production_latent_grid(cfg)
    pre = TILE_SIZE if text_pre is None else text_pre
    return [[slice(pre, pre + g * g)]]


def max_seq_tile_aligned(cfg: dict | None = None) -> int:
    """Largest TILE-aligned seq_len ≤ ``max_position_embeddings``."""
    cfg = cfg or load_config()
    return (int(cfg["max_position_embeddings"]) // TILE_SIZE) * TILE_SIZE


def patch_embed_hidden_channels(cfg: dict | None = None) -> int:
    """UNetDown/Up mid width.

    Upstream ``patch_embed_hidden_dim`` is absent from released ``config.json``;
    it matches the VAE mid / deepest ``block_out_channels`` entry.
    """
    cfg = cfg or load_config()
    if "patch_embed_hidden_dim" in cfg:
        return int(cfg["patch_embed_hidden_dim"])
    return vae_mid_channels(cfg)


# ---------------------------------------------------------------------------
# Module-level defaults (bundled config) — import these instead of literals.
# ---------------------------------------------------------------------------
_CFG = load_config()
_TC = transformer_cfg(_CFG)
_VAE = vae_section(_CFG)

HIDDEN_SIZE = _TC["H"]
NUM_HIDDEN_LAYERS = _TC["NUM_LAYERS"]
NUM_ATTENTION_HEADS = _TC["HEADS"]
NUM_KEY_VALUE_HEADS = _TC["KV"]
ATTENTION_HEAD_DIM = _TC["HD"]
NUM_EXPERTS = _TC["E"]
MOE_TOPK = _TC["K"]
RMS_NORM_EPS = _TC["EPS"]
ROPE_THETA = _TC["ROPE_THETA"]
VOCAB_SIZE = _TC["VOCAB"]
MAX_POSITION_EMBEDDINGS = _TC["MAX_SEQ"]

VAE_SCALING_FACTOR = float(_VAE["scaling_factor"])
VAE_LATENT_CHANNELS = int(_VAE["latent_channels"])
VAE_FFACTOR_SPATIAL = int(_VAE["ffactor_spatial"])
VAE_FFACTOR_TEMPORAL = int(_VAE["ffactor_temporal"])
VAE_IN_CHANNELS = int(_VAE["in_channels"])
VAE_OUT_CHANNELS = int(_VAE["out_channels"])
VAE_BLOCK_OUT_CHANNELS = vae_block_out_channels(_CFG)
VAE_MID_CHANNELS = vae_mid_channels(_CFG)
# GroupNorm groups (upstream AutoencoderKLConv3D); not present in config.json.
VAE_NUM_GROUPS = 32
VAE_GN_EPS = 1e-6

# Patch-embed mid width (= VAE mid); see ``patch_embed_hidden_channels``.
PATCH_EMBED_HIDDEN_CHANNELS = patch_embed_hidden_channels(_CFG)

IMAGE_BASE_SIZE = image_base_size(_CFG)
PRODUCTION_LATENT_GRID = production_latent_grid(_CFG)
PRODUCTION_IMAGE_TOKENS = PRODUCTION_LATENT_GRID * PRODUCTION_LATENT_GRID
# Tile-aligned text pads around the image span (PCC / synthetic denoise layouts).
PRODUCTION_TEXT_PRE = TILE_SIZE
PRODUCTION_TEXT_POST = TILE_SIZE
PRODUCTION_SEQ = production_seq_len(_CFG)
MAX_SEQ_TILE_ALIGNED = max_seq_tile_aligned(_CFG)

VIT_CONFIG = vit_config(_CFG)
ALIGNER_CONFIG = aligner_config(_CFG)

BACKBONE_KWARGS = backbone_kwargs(_CFG)
ATTENTION_KWARGS = attention_kwargs(_CFG)
