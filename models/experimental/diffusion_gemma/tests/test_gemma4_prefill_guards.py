# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Guards for the DiffusionGemma-local denoise attention + model.

These RoPE-offset / cache-slice guards used to live in the shared Gemma4 prefill
op; they now belong to DiffusionGemma so the backbone stays untouched. The
denoise pass is single-user, so ``validate_q_rope_offset`` only enforces tile
alignment (no batched-prefill case).
"""

import pytest
import torch
from types import SimpleNamespace

from models.experimental.diffusion_gemma.tt.diffusion_attention import _slice_rope_cache, validate_q_rope_offset
from models.experimental.diffusion_gemma.tt.model import DiffusionGemma4Model


def test_q_rope_offset_must_be_tile_aligned():
    validate_q_rope_offset(32)
    validate_q_rope_offset(0)
    with pytest.raises(ValueError, match="q_rope_offset must be a multiple of 32"):
        validate_q_rope_offset(1)
    with pytest.raises(ValueError, match="RoPE cache start must be a multiple of 32"):
        _slice_rope_cache(None, 1, 32)


def test_get_rope_mats_reaches_256k_and_rejects_overflow():
    cache_len = 262144
    model = SimpleNamespace(
        hf_config=SimpleNamespace(layer_types=["sliding_attention"]),
        rope_caches={
            "sliding_attention": (
                torch.zeros(1, 1, cache_len, 8),
                torch.zeros(1, 1, cache_len, 8),
            )
        },
    )

    cos, sin = DiffusionGemma4Model._get_rope_mats(model, 0, seq_len=cache_len)
    assert cos.shape[-2] == cache_len
    assert sin.shape[-2] == cache_len
    with pytest.raises(ValueError, match="requested RoPE seq_len 262176 exceeds cache length 262144"):
        DiffusionGemma4Model._get_rope_mats(model, 0, seq_len=cache_len + 32)


def test_slice_rope_cache_rejects_overflow():
    cache = SimpleNamespace(shape=[1, 1, 262144, 8])
    with pytest.raises(ValueError, match=r"RoPE cache slice \[262144, 262176\) exceeds cache length 262144"):
        _slice_rope_cache(cache, 262144, 32)
