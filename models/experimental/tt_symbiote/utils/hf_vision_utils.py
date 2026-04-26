# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Utility to swap vision-tower attention from FlashAttention2 to eager (CPU-safe).

Extracted from the dots_ocr reference hf_utils module so that
tt_symbiote tests can use it without depending on the dots_ocr demo package.
"""

from __future__ import annotations

import importlib

from transformers.utils import logging

logger = logging.get_logger(__name__)


def _install_eager_vision_attention(model) -> None:
    """
    Dots stores **vision** attention in ``config.vision_config.attn_implementation`` (default
    ``flash_attention_2``). The ``from_pretrained`` ``_attn_implementation=eager`` flag only sets the
    **text** LM attention; the vision tower is already built with ``VisionFlashAttention2``.

    Replace each block's attention with the remote code's **eager** ``VisionAttention`` (manual
    mask + matmul + softmax) using the same ``qkv``/``proj`` weights so HF reference matches
    PCC expectations without ``flash_attn`` at runtime.
    """
    vt = getattr(model, "vision_tower", None)
    if vt is None or not hasattr(vt, "blocks"):
        return
    cfg = getattr(getattr(model, "config", None), "vision_config", None)
    if cfg is None:
        return

    mod = importlib.import_module(vt.__class__.__module__)
    VisionAttention = getattr(mod, "VisionAttention", None)
    if VisionAttention is None:
        logger.warning(
            "eager vision swap: VisionAttention not found in %s; leaving vision attention as loaded.",
            vt.__class__.__module__,
        )
        return

    if getattr(cfg, "attn_implementation", None) == "eager" and all(
        isinstance(b.attn, VisionAttention) for b in vt.blocks
    ):
        return

    cfg.attn_implementation = "eager"
    for block in vt.blocks:
        old = block.attn
        if isinstance(old, VisionAttention):
            continue
        device = next(old.parameters()).device
        dtype = next(old.parameters()).dtype
        new_attn = VisionAttention(
            cfg,
            cfg.embed_dim,
            num_heads=cfg.num_attention_heads,
            bias=cfg.use_bias,
        )
        new_attn = new_attn.to(device=device, dtype=dtype)
        new_attn.load_state_dict(old.state_dict(), strict=True)
        block.attn = new_attn
