# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: Ideogram4 (D256) opt-in SDPA recipe wiring.

Ideogram4TransformerBlock instances are built with object.__new__ (no mesh device); only the small
program-config / kwargs helpers and the call-time mask rejection are exercised.
"""

import pytest

import ttnn
from models.tt_dit.models.transformers.transformer_ideogram4 import Ideogram4Transformer, Ideogram4TransformerBlock

GRID = ttnn.CoreCoord(12, 10)
WORKER_GRID = (12, 9)
HEAD_DIM = 256


def _pc(q, k, grid=GRID):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, q_chunk_size=q, k_chunk_size=k, exp_approx_mode=False
    )


def _bare_block(precision=None, kv_dtype=ttnn.bfloat16):
    block = object.__new__(Ideogram4TransformerBlock)
    block.sdpa_precision = precision
    block.sdpa_kv_dtype = kv_dtype
    block.sdpa_q_chunk_size = 128
    block.sdpa_k_chunk_size = 256
    block.sdpa_worker_grid = WORKER_GRID
    block._ring_sdpa_pc_cache = {}
    block.sdpa_program_config = _pc(128, 256)
    block.sdpa_compute_kernel_config = object()
    block.matmul_compute_kernel_config = object()
    block.rope_trans_mat = None
    return block


def _chunks(pc):
    return pc.q_chunk_size, pc.k_chunk_size


def test_legacy_program_configs_are_the_tuned_objects():
    block = _bare_block()
    assert block._recipe_sdpa_program_config(block.sdpa_program_config, ring=False) is block.sdpa_program_config
    ring_pc = block._get_ring_sdpa_program_config(512)
    assert block._recipe_sdpa_program_config(ring_pc, ring=True) is ring_pc
    assert _chunks(ring_pc) == (128, 256)


@pytest.mark.parametrize("precision", [ttnn.SDPAPrecision.FAST, ttnn.SDPAPrecision.ACCURATE])
def test_recipe_keeps_tuned_d256_q128_k256_dense_and_ring(precision):
    block = _bare_block(precision)
    dense = block._recipe_sdpa_program_config(block.sdpa_program_config, ring=False)
    ring = block._recipe_sdpa_program_config(block._get_ring_sdpa_program_config(512), ring=True)
    assert _chunks(dense) == (128, 256)
    assert _chunks(ring) == (128, 256)  # Q128 = 4 tiles (even): valid ring checkpoint
    grid = lambda pc: (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y)
    assert grid(dense) == (12, 10)
    assert grid(ring) == WORKER_GRID  # CCL row stays reserved
    assert not dense.exp_approx_mode and not ring.exp_approx_mode  # left unset for the recipe


def test_recipe_ring_rejects_odd_tile_q_chunk_dense_keeps_it():
    block = _bare_block(ttnn.SDPAPrecision.ACCURATE)
    assert _chunks(block._recipe_sdpa_program_config(_pc(160, 256), ring=True)) == (256, 256)
    assert _chunks(block._recipe_sdpa_program_config(_pc(160, 256), ring=False)) == (160, 256)
    assert _chunks(block._recipe_sdpa_program_config(_pc(64, 128), ring=False)) == (256, 512)


def test_sdpa_kwargs_legacy_read_at_call_time_and_recipe_replaces_it():
    block = _bare_block()
    config = object()
    block.sdpa_compute_kernel_config = config
    assert block._sdpa_kwargs() == {"compute_kernel_config": config}
    for precision, prepared in ((ttnn.SDPAPrecision.FAST, False), (ttnn.SDPAPrecision.LOW_PRECISION, True)):
        block = _bare_block(precision)
        assert block._sdpa_kwargs() == {"precision": precision, "inputs_prepared": prepared}


def test_recipe_rejects_segment_mask_at_call_time(monkeypatch):
    # Stub everything before the SDPA dispatch; the recipe must raise before any SDPA/CCL call.
    block = _bare_block(ttnn.SDPAPrecision.ACCURATE)
    block.n_local_heads = 1
    block._all_gather_hidden = lambda t: t
    block.qkv = lambda x, **_: (x, x, x)
    block.norm_q = block.norm_k = lambda t, **_: t
    monkeypatch.setattr(ttnn, "unsqueeze", lambda t, _dim: t)
    monkeypatch.setattr(ttnn.experimental, "nlp_create_qkv_heads", lambda t, **_: (t, None, None))
    for name in ("scaled_dot_product_attention", "ring_joint_scaled_dot_product_attention"):
        monkeypatch.setattr(ttnn.transformer, name, lambda *a, **k: pytest.fail("SDPA called with a mask"))
    for sp_factor in (1, 2):
        block.sp_factor = sp_factor
        with pytest.raises(ValueError, match="unmasked"):
            block._attention(object(), cos=None, sin=None, attn_mask=object(), spatial_sequence_length=512)


def test_d256_recipes_accepted_by_model_validation():
    validate = Ideogram4Transformer.validate_sdpa_recipe
    validate(None, None, head_dim=HEAD_DIM)  # legacy: no-op
    for precision in (ttnn.SDPAPrecision.FAST, ttnn.SDPAPrecision.ACCURATE, ttnn.SDPAPrecision.LOW_PRECISION):
        validate(precision, None, head_dim=HEAD_DIM)
    validate(ttnn.SDPAPrecision.LOW_PRECISION, ttnn.bfloat8_b, head_dim=HEAD_DIM)
    with pytest.raises(ValueError, match="head_dim"):
        validate(ttnn.SDPAPrecision.ACCURATE, None, head_dim=96)


def test_low_precision_kv_dtype_validation():
    validate = Ideogram4Transformer.validate_sdpa_recipe
    with pytest.raises(ValueError, match="LOW_PRECISION"):
        validate(ttnn.SDPAPrecision.ACCURATE, ttnn.bfloat8_b, head_dim=HEAD_DIM)
    with pytest.raises(ValueError, match="LOW_PRECISION"):
        validate(None, ttnn.bfloat8_b, head_dim=HEAD_DIM)
    with pytest.raises(ValueError, match="dtype"):
        validate(ttnn.SDPAPrecision.LOW_PRECISION, ttnn.float32, head_dim=HEAD_DIM)
