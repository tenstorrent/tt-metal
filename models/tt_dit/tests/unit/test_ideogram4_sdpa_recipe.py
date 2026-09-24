# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: Ideogram4 (D256) SDPA recipe wiring.

Ideogram4TransformerBlock instances are built with object.__new__ (no mesh device); only the small
kwargs helpers and the call-time masked dispatch are exercised. The default recipe itself is covered by
test_sdpa_dit_recipe_defaults.py.
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
    block.sdpa_worker_grid = WORKER_GRID
    block.sdpa_program_config = _pc(128, 256)
    block.ring_sdpa_program_config = _pc(128, 256, WORKER_GRID)
    block.sdpa_compute_kernel_config = object()
    block.matmul_compute_kernel_config = object()
    block.rope_trans_mat = None
    return block


def _chunks(pc):
    return pc.q_chunk_size, pc.k_chunk_size


def test_sdpa_kwargs_legacy_read_at_call_time_and_recipe_replaces_it():
    block = _bare_block()
    config = object()
    block.sdpa_compute_kernel_config = config
    assert block._sdpa_kwargs() == {"compute_kernel_config": config}
    for precision, prepared in ((ttnn.SDPAPrecision.FAST, False), (ttnn.SDPAPrecision.LOW_PRECISION, True)):
        block = _bare_block(precision)
        assert block._sdpa_kwargs() == {"precision": precision, "inputs_prepared": prepared}


def test_recipe_segment_mask_uses_dense_recipe_with_mask(monkeypatch):
    # Stub everything before the SDPA dispatch; a masked recipe call must reach the dense recipe SDPA
    # with attn_mask (never the unmasked ring recipe), gathering K/V first under SP.
    block = _bare_block(ttnn.SDPAPrecision.ACCURATE)
    block.n_local_heads = 1
    block.sp_axis = 1
    block._all_gather_hidden = lambda t: t
    block.qkv = lambda x, **_: (x, x, x)
    block.norm_q = block.norm_k = lambda t, **_: t
    block.o = lambda t, **_: t
    gathered = []

    class _Ccl:
        def all_gather_persistent_buffer(self, t, **_):
            gathered.append(t)
            return t

    block.ccl_manager = _Ccl()
    monkeypatch.setattr(ttnn, "unsqueeze", lambda t, _dim: t)
    monkeypatch.setattr(ttnn.experimental, "nlp_create_qkv_heads", lambda t, **_: (t, None, None))
    monkeypatch.setattr(ttnn.transformer, "concatenate_heads", lambda t: t)
    monkeypatch.setattr(
        ttnn.transformer, "ring_joint_scaled_dot_product_attention", lambda *a, **k: pytest.fail("ring with a mask")
    )
    calls = []

    def dense(q, k, v, **kwargs):
        calls.append(kwargs)
        return q

    monkeypatch.setattr(ttnn.transformer, "scaled_dot_product_attention", dense)
    mask = object()
    for sp_factor in (1, 2):
        block.sp_factor = sp_factor
        gathered.clear()
        block._attention(object(), cos=None, sin=None, attn_mask=mask, spatial_sequence_length=512)
        assert calls[-1]["attn_mask"] is mask
        assert calls[-1]["precision"] == ttnn.SDPAPrecision.ACCURATE
        assert len(gathered) == (2 if sp_factor > 1 else 0)


def test_d256_recipes_accepted_by_model_validation():
    validate = Ideogram4Transformer.validate_sdpa_recipe
    validate(None, None, head_dim=HEAD_DIM)  # the default recipe
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
