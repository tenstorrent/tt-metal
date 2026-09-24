# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: MiniMax-H3 opt-in SDPA recipe wiring (program configs, kwargs, validation; no device)."""

import inspect

import pytest
import ttnn

from models.tt_dit.models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention

ACCURATE, COMPENSATED, LOW = (
    ttnn.SDPAPrecision.ACCURATE,
    ttnn.SDPAPrecision.COMPENSATED,
    ttnn.SDPAPrecision.LOW_PRECISION,
)


def _bare_attention(precision=None, *, n_local_heads=14, grid=(12, 10), use_exp=True):
    # Bypass __init__, which needs a mesh device; set only what the config helpers read.
    attn = object.__new__(MiniMaxH3Attention)
    attn.sdpa_precision = precision
    attn.sdpa_kv_dtype = ttnn.bfloat16
    attn.head_dim = 128
    attn.n_local_heads = n_local_heads
    attn.full_grid = ttnn.CoreCoord(*grid)
    attn.sdpa_worker_grid = (grid[0] - 1, grid[1])
    attn._sdpa_program_configs = {}
    attn._recipe_sdpa_program_configs = {}
    attn._exp_sdpa_program_configs = {}
    attn.exp_ring_max_passes = 3
    attn.exp_ring_max_k_chunk = 512
    attn.use_exp_ring_sdpa = use_exp
    return attn


def _chunks(pc):
    return pc.q_chunk_size, pc.k_chunk_size


def test_legacy_program_configs_unchanged():
    attn = _bare_attention(None)
    for seq in (4768, 9216, 1000):
        for ring in (True, False):
            assert attn._attn_program_config(seq, ring=ring) is attn._sdpa_program_config(seq, ring=ring)


def test_ring_recipe_keeps_supported_measured_chunks():
    attn = _bare_attention(ACCURATE)
    assert _chunks(attn._attn_program_config(4768, ring=True)) == (320, 384)  # 10 tiles: even
    assert _chunks(attn._attn_program_config(9216, ring=True)) == (256, 512)
    pc = attn._attn_program_config(9216, ring=True)
    assert (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y) == (11, 10)


def test_ring_recipe_even_tile_rule_and_fallback():
    attn = _bare_attention(COMPENSATED)
    attn.measured_sdpa_chunk_sizes = {5000: (288, 384), 6000: (224, 1024), 7000: (96, 128)}
    assert _chunks(attn._attn_program_config(5000, ring=True)) == (256, 384)  # 9 tiles: odd on ring
    assert _chunks(attn._attn_program_config(5000, ring=False)) == (288, 384)  # dense accepts odd tiles
    assert _chunks(attn._attn_program_config(6000, ring=False)) == (224, 512)
    assert _chunks(attn._attn_program_config(7000, ring=True)) == (256, 512)


def test_dense_recipe_short_refiner_sequence_falls_back():
    # The token refiner's short text stream: legacy clamps chunks to the sequence, the recipe cannot.
    attn = _bare_attention(ACCURATE, use_exp=False)
    assert _chunks(attn._sdpa_program_config(96, ring=False)) == (96, 96)
    assert _chunks(attn._attn_program_config(96, ring=False)) == (256, 512)
    pc = attn._attn_program_config(96, ring=False)
    assert (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y) == (12, 10)


@pytest.mark.parametrize("seq_local", [2304, 3072, 1152, 4768 // 4])
def test_exp_ring_recipe_config_is_recipe_supported(seq_local):
    attn = _bare_attention(ACCURATE)
    pc = attn._exp_sdpa_program_config(seq_local)
    assert pc is not None
    q, k = _chunks(pc)
    assert k == 512 and 128 <= q <= 320 and q % 32 == 0
    cols = pc.compute_with_storage_grid_size.x - 1
    segs = -(-seq_local // q) // cols
    assert -(-seq_local // q) == cols * segs  # a head's chunks fill whole rows, as the op requires
    assert -(-attn.n_local_heads * segs // attn.full_grid.y) <= 3


def test_exp_ring_recipe_infeasible_falls_back_to_ring():
    attn = _bare_attention(ACCURATE)
    assert attn._exp_sdpa_program_config(20000) is None  # needs > 33 chunks of <= 320 rows
    assert attn._exp_sdpa_program_config(100) is None  # every usable q_chunk is below 128
    assert _bare_attention(ACCURATE, use_exp=False)._exp_sdpa_program_config(2304) is None


def test_legacy_sdpa_kwargs_read_compute_config_at_call_time():
    attn = _bare_attention(None)
    attn.sdpa_compute_kernel_config = "initial"
    assert attn._sdpa_kwargs() == {"compute_kernel_config": "initial"}
    attn.sdpa_compute_kernel_config = "reassigned"
    assert attn._sdpa_kwargs() == {"compute_kernel_config": "reassigned"}


@pytest.mark.parametrize("precision", [ACCURATE, COMPENSATED, LOW])
def test_recipe_sdpa_kwargs_replace_compute_config(precision):
    attn = _bare_attention(precision)
    attn.sdpa_compute_kernel_config = "ignored"
    assert attn._sdpa_kwargs() == {"precision": precision, "inputs_prepared": precision == LOW}


@pytest.mark.parametrize(
    "precision, kv_dtype, head_dim",
    [
        (ACCURATE, None, 96),  # not a recipe head dim (64/128/256 are)
        (COMPENSATED, ttnn.bfloat8_b, 128),  # packed KV needs LOW_PRECISION
        (None, ttnn.bfloat8_b, 128),  # KV dtype without a recipe
        (LOW, ttnn.float32, 128),  # unsupported KV storage
    ],
)
def test_constructor_rejects_unsupported_recipe_args(precision, kv_dtype, head_dim):
    # Validation runs before the constructor touches the mesh, so no device is needed to reach it.
    with pytest.raises(ValueError):
        MiniMaxH3Attention(
            hidden_size=5376,
            num_heads=56,
            head_dim=head_dim,
            mesh_device=None,
            ccl_manager=None,
            parallel_config=None,
            sdpa_precision=precision,
            sdpa_kv_dtype=kv_dtype,
        )


def test_attention_has_no_mask_path():
    # MiniMax-H3 attention is unmasked everywhere (ring logical_n masks the pad tail), so there is no
    # mask argument a recipe would have to reject.
    params = inspect.signature(MiniMaxH3Attention.forward).parameters
    assert not any("mask" in name for name in params)
