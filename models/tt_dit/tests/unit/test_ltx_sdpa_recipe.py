# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: LTX-2 opt-in SDPA recipe wiring, plus the SD3.5 / Ideogram4 recipe rejections.

LTXAttention instances are built with object.__new__ (no mesh device); only the small config /
kwargs helpers and the call-time mask rejection are exercised.
"""

import pytest

import ttnn
from models.tt_dit.models.transformers.ltx.attention_ltx import LTXAttention
from models.tt_dit.models.transformers.transformer_ideogram4 import Ideogram4Transformer
from models.tt_dit.models.transformers.transformer_sd35 import SD35Transformer2DModel
from models.tt_dit.utils.sdpa_recipe import validate_recipe_args

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
    attention.sdpa_program_config = _pc(256, 256)
    attention.ring_sdpa_program_config = _pc(128, 512, WORKER_GRID)
    # Tuned BH 4x8 ring chunks keyed by N (see LTXAttention.ring_sdpa_chunk_by_n).
    attention._ring_pc_by_n = {9728: _pc(96, 256, WORKER_GRID), 38912: _pc(192, 512, WORKER_GRID)}
    # Tuned BH video text / A2V cross chunks (see LTXAttention.sdpa_chunk_by_shape).
    attention._sdpa_pc_by_shape = {(1216, 32): _pc(128, 128), (4864, 32): _pc(192, 128)}
    return attention


def _chunks(pc):
    return pc.q_chunk_size, pc.k_chunk_size


def test_legacy_program_configs_are_the_tuned_objects():
    attention = _bare_attention()
    assert attention._ring_program_config(9728) is attention._ring_pc_by_n[9728]
    assert attention._ring_program_config(1234) is attention.ring_sdpa_program_config
    assert attention._dense_program_config() is attention.sdpa_program_config
    assert attention._cross_program_config(4864, 32) is attention._sdpa_pc_by_shape[(4864, 32)]
    assert attention._cross_program_config(1, 2) is attention.sdpa_program_config


def test_recipe_ring_chunks_keep_even_tiles_and_fall_back():
    attention = _bare_attention(ttnn.SDPAPrecision.ACCURATE)
    assert _chunks(attention._ring_program_config(9728)) == (256, 256)  # Q96 < 128 -> Q256; K256 kept
    assert _chunks(attention._ring_program_config(38912)) == (192, 512)  # 6 tiles, even -> kept
    assert _chunks(attention._ring_program_config(1234)) == (128, 512)
    pc = attention._ring_program_config(38912)
    assert (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y) == (11, 10)


def test_recipe_ring_rejects_odd_tile_q_chunk():
    attention = _bare_attention(ttnn.SDPAPrecision.ACCURATE)
    attention.ring_sdpa_program_config = _pc(224, 384, WORKER_GRID)  # 7 tiles: odd
    assert _chunks(attention._ring_program_config(1234)) == (256, 384)


def test_recipe_dense_and_cross_chunks():
    attention = _bare_attention(ttnn.SDPAPrecision.ACCURATE)
    assert _chunks(attention._dense_program_config()) == (256, 256)
    assert _chunks(attention._cross_program_config(1216, 32)) == (128, 512)  # K128 unsupported
    assert _chunks(attention._cross_program_config(4864, 32)) == (192, 512)
    attention.sdpa_program_config = _pc(224, 384)  # dense allows odd Q tiles
    assert _chunks(attention._cross_program_config(1, 2)) == (224, 384)


def test_sdpa_kwargs_legacy_read_at_call_time_and_recipe_replaces_it():
    attention = _bare_attention()
    assert attention._sdpa_kwargs() == {"compute_kernel_config": "legacy"}
    attention.sdpa_compute_kernel_config = "quantized"  # quant presets swap it after construction
    assert attention._sdpa_kwargs() == {"compute_kernel_config": "quantized"}

    attention.sdpa_precision = ttnn.SDPAPrecision.ACCURATE
    assert attention._sdpa_kwargs() == {"precision": ttnn.SDPAPrecision.ACCURATE, "inputs_prepared": False}
    attention.sdpa_precision = ttnn.SDPAPrecision.LOW_PRECISION
    assert attention._sdpa_kwargs() == {"precision": ttnn.SDPAPrecision.LOW_PRECISION, "inputs_prepared": True}


def test_recipe_rejects_masked_attention_at_call_time():
    attention = _bare_attention(ttnn.SDPAPrecision.ACCURATE)
    with pytest.raises(ValueError, match="unmasked"):
        attention.forward(spatial_1BND=None, N=0, attn_mask=object())


def test_ltx_head_dims_video_accepted_audio_rejected():
    # Video attention is 4096 / 32 heads = D128; audio (and audio<->video cross) is 2048 / 32 = D64.
    kv = validate_recipe_args(ttnn.SDPAPrecision.ACCURATE, None, head_dim=4096 // 32, model="LTX-2")
    assert kv == ttnn.bfloat16
    with pytest.raises(ValueError, match="D128"):
        validate_recipe_args(ttnn.SDPAPrecision.ACCURATE, None, head_dim=2048 // 32, model="LTX-2")
    with pytest.raises(ValueError, match="D128"):
        validate_recipe_args(ttnn.SDPAPrecision.ACCURATE, None, head_dim=128, model="LTX-2", is_blackhole=False)


def test_ltx_low_precision_kv_dtype_validation():
    low = ttnn.SDPAPrecision.LOW_PRECISION
    assert validate_recipe_args(low, ttnn.bfloat8_b, head_dim=128, model="LTX-2") == ttnn.bfloat8_b
    with pytest.raises(ValueError, match="LOW_PRECISION"):
        validate_recipe_args(ttnn.SDPAPrecision.ACCURATE, ttnn.bfloat8_b, head_dim=128, model="LTX-2")
    with pytest.raises(ValueError, match="LOW_PRECISION"):
        validate_recipe_args(None, ttnn.bfloat4_b, head_dim=128, model="LTX-2")


@pytest.mark.parametrize(
    "validate, head_dim",
    [(SD35Transformer2DModel.validate_sdpa_recipe, 64), (Ideogram4Transformer.validate_sdpa_recipe, 256)],
    ids=["sd35_d64", "ideogram4_d256"],
)
def test_unwired_models_reject_recipes(validate, head_dim):
    validate(None, None, head_dim=head_dim)  # legacy: no-op
    for precision in (ttnn.SDPAPrecision.ACCURATE, ttnn.SDPAPrecision.LOW_PRECISION):
        with pytest.raises(ValueError):
            validate(precision, None, head_dim=head_dim)
    # Even a D128 config of these models is rejected: the recipe is not wired there.
    with pytest.raises(ValueError, match="not wired"):
        validate(ttnn.SDPAPrecision.ACCURATE, None, head_dim=128)
    with pytest.raises(ValueError, match="LOW_PRECISION"):
        validate(None, ttnn.bfloat8_b, head_dim=head_dim)
