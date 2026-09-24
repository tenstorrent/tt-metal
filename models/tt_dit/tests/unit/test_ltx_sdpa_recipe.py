# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: LTX-2 SDPA recipe wiring (SD3.5 recipe tests: test_sd35_sdpa_recipe.py).

LTXAttention instances are built with object.__new__ (no mesh device); only the small config /
kwargs helpers and the call-time mask handling are exercised. The default recipe itself is covered by
test_sdpa_dit_recipe_defaults.py.
"""

import pytest

import ttnn
from models.tt_dit.models.transformers.ltx.attention_ltx import LTXAttention
from models.tt_dit.utils.sdpa_recipe import recipe_config, validate_recipe_args

GRID = ttnn.CoreCoord(12, 10)
WORKER_GRID = ttnn.CoreCoord(11, 10)


def _pc(q, k, grid=GRID):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, q_chunk_size=q, k_chunk_size=k, exp_approx_mode=False
    )


def _bare_attention(precision=None):
    attention = object.__new__(LTXAttention)
    attention.sdpa_precision = precision
    attention.sdpa_kv_dtype = ttnn.bfloat16
    attention.sdpa_compute_kernel_config = "legacy"
    attention.is_self = True
    if precision is None:
        # Legacy (non-Blackhole) configs.
        attention.sdpa_program_config = _pc(256, 256)
        attention.ring_sdpa_program_config = _pc(256, 256, WORKER_GRID)
    else:
        attention.sdpa_program_config = recipe_config(GRID)
        attention.ring_sdpa_program_config = recipe_config(WORKER_GRID)
    attention.cross_ring_sdpa_program_config = attention.ring_sdpa_program_config
    return attention


def _chunks(pc):
    return pc.q_chunk_size, pc.k_chunk_size


def test_program_config_helpers_return_the_module_configs():
    for precision in (None, ttnn.SDPAPrecision.ACCURATE):
        attention = _bare_attention(precision)
        for n in (9728, 38912, 1234):
            assert attention._ring_program_config(n) is attention.ring_sdpa_program_config
        assert attention._dense_program_config() is attention.sdpa_program_config
        for shape in ((1216, 32), (4864, 32), (1, 2)):
            assert attention._cross_program_config(*shape) is attention.sdpa_program_config


def test_recipe_chunks_are_op_selected():
    attention = _bare_attention(ttnn.SDPAPrecision.ACCURATE)
    grid = lambda pc: (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y)
    for n in (9728, 38912, 1234):
        pc = attention._ring_program_config(n)
        assert _chunks(pc) == (0, 0) and grid(pc) == (11, 10)
    assert _chunks(attention._dense_program_config()) == (0, 0)
    for shape in ((1216, 32), (4864, 32), (1, 2)):
        assert _chunks(attention._cross_program_config(*shape)) == (0, 0)


def test_tuned_chunks_are_legacy_only():
    # The BH per-N ring, per-shape cross and V2A ring-cross tables are gone: recipes choose chunks.
    assert all(not key[0] for key in LTXAttention.sdpa_chunk_size_map)
    for name in ("ring_sdpa_chunk_by_n", "sdpa_chunk_by_shape", "cross_ring_sdpa_q_chunk_map"):
        assert not hasattr(LTXAttention, name)


def test_sdpa_kwargs_legacy_read_at_call_time_and_recipe_replaces_it():
    attention = _bare_attention()
    assert attention._sdpa_kwargs() == {"compute_kernel_config": "legacy"}
    attention.sdpa_compute_kernel_config = "quantized"  # quant presets swap it after construction
    assert attention._sdpa_kwargs() == {"compute_kernel_config": "quantized"}

    attention.sdpa_precision = ttnn.SDPAPrecision.ACCURATE
    assert attention._sdpa_kwargs() == {"precision": ttnn.SDPAPrecision.ACCURATE, "inputs_prepared": False}
    attention.sdpa_precision = ttnn.SDPAPrecision.LOW_PRECISION
    assert attention._sdpa_kwargs() == {"precision": ttnn.SDPAPrecision.LOW_PRECISION, "inputs_prepared": True}


def test_recipe_non_key_length_masks_pass_through():
    attention = _bare_attention(ttnn.SDPAPrecision.ACCURATE)
    # A mask without its logical key length (or any cross-attention mask) goes to the recipe as attn_mask.
    assert attention._recipe_mask_kv_len(object(), None) is None
    attention.is_self = False
    assert attention._recipe_mask_kv_len(object(), 200) is None


def test_recipe_key_length_mask_becomes_slice_length():
    attention = _bare_attention(ttnn.SDPAPrecision.FAST)
    assert attention._recipe_mask_kv_len(object(), 200) == 200
    assert attention._recipe_mask_kv_len(None, 200) is None  # unpadded audio: no slicing
    with pytest.raises(ValueError, match="positive"):
        attention._recipe_mask_kv_len(object(), 0)
    # Legacy ignores attn_kv_len entirely (the mask is passed to SDPA as before).
    legacy = _bare_attention()
    assert legacy._recipe_mask_kv_len(object(), 200) is None
    assert legacy._recipe_mask_kv_len(object(), None) is None


def test_v2a_ring_cross_and_gathered_audio_configs():
    legacy = _bare_attention()
    assert legacy._cross_ring_program_config(64) is legacy.cross_ring_sdpa_program_config
    assert legacy.cross_ring_sdpa_program_config is legacy.ring_sdpa_program_config
    assert legacy._gathered_program_config(128) is legacy.sdpa_program_config

    attention = _bare_attention(ttnn.SDPAPrecision.ACCURATE)
    # Tiny per-device audio Q shards: SDPA sizes the Q chunk itself.
    for q_len in (32, 64, 160, 4096):
        pc = attention._cross_ring_program_config(q_len)
        assert _chunks(pc) == (0, 0)
        assert (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y) == (11, 10)
        assert _chunks(attention._gathered_program_config(q_len)) == (0, 0)


def test_ltx_head_dims_video_and_audio_accepted():
    # Video attention is 4096 / 32 heads = D128; audio (and audio<->video cross) is 2048 / 32 = D64.
    for head_dim in (4096 // 32, 2048 // 32):
        kv = validate_recipe_args(ttnn.SDPAPrecision.ACCURATE, None, head_dim=head_dim, model="LTX-2")
        assert kv == ttnn.bfloat16
    assert (
        validate_recipe_args(ttnn.SDPAPrecision.LOW_PRECISION, ttnn.bfloat8_b, head_dim=64, model="LTX-2")
        == ttnn.bfloat8_b
    )
    with pytest.raises(ValueError, match="head_dim"):
        validate_recipe_args(ttnn.SDPAPrecision.ACCURATE, None, head_dim=96, model="LTX-2")
    with pytest.raises(ValueError, match="Blackhole"):
        validate_recipe_args(ttnn.SDPAPrecision.ACCURATE, None, head_dim=64, model="LTX-2", is_blackhole=False)


def test_ltx_low_precision_kv_dtype_validation():
    low = ttnn.SDPAPrecision.LOW_PRECISION
    assert validate_recipe_args(low, ttnn.bfloat8_b, head_dim=128, model="LTX-2") == ttnn.bfloat8_b
    with pytest.raises(ValueError, match="LOW_PRECISION"):
        validate_recipe_args(ttnn.SDPAPrecision.ACCURATE, ttnn.bfloat8_b, head_dim=128, model="LTX-2")
    with pytest.raises(ValueError, match="LOW_PRECISION"):
        validate_recipe_args(None, ttnn.bfloat4_b, head_dim=128, model="LTX-2")
