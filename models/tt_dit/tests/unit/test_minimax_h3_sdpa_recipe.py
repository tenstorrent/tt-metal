# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: MiniMax-H3 exp ring recipe configs (no device).

The default recipe itself is covered by test_sdpa_dit_recipe_defaults.py."""

import pytest

import ttnn
from models.tt_dit.models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention
from models.tt_dit.utils.sdpa_recipe import recipe_config

ACCURATE, STANDARD = ttnn.SDPAPrecision.ACCURATE, ttnn.SDPAPrecision.STANDARD


def _bare_attention(precision=STANDARD, *, n_local_heads=14, grid=(12, 10), use_exp=True):
    # Bypass __init__, which needs a mesh device; set only what the config helpers read.
    attn = object.__new__(MiniMaxH3Attention)
    attn.sdpa_precision = precision
    attn.sdpa_kv_dtype = ttnn.bfloat16
    attn.head_dim = 128
    attn.n_local_heads = n_local_heads
    attn.full_grid = ttnn.CoreCoord(*grid)
    attn.sdpa_program_config = recipe_config(attn.full_grid)
    attn._exp_sdpa_program_configs = {}
    attn.exp_ring_max_passes = 3
    attn.use_exp_ring_sdpa = use_exp
    return attn


def _chunks(pc):
    return pc.q_chunk_size, pc.k_chunk_size


@pytest.mark.parametrize("seq_local", [2304, 3072, 1152, 4768 // 4])
def test_exp_ring_recipe_config_is_op_selected(seq_local):
    attn = _bare_attention(ACCURATE)
    pc = attn._exp_sdpa_program_config(seq_local)
    assert pc is not None and _chunks(pc) == (0, 0)
    grid = pc.compute_with_storage_grid_size
    assert (grid.x, grid.y) == (attn.full_grid.x, attn.full_grid.y)


def test_exp_ring_recipe_infeasible_falls_back_to_ring():
    attn = _bare_attention(ACCURATE)
    assert attn._exp_sdpa_program_config(20000) is None  # needs > 33 chunks of <= 320 rows
    assert attn._exp_sdpa_program_config(100) is None  # every usable q_chunk is below 128
    assert _bare_attention(ACCURATE, use_exp=False)._exp_sdpa_program_config(2304) is None
