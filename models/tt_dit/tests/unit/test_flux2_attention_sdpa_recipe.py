# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: FLUX.2 (blocks/attention_opt.py) opt-in SDPA recipe wiring (no device)."""

from types import SimpleNamespace

import pytest
import torch
import ttnn

from models.tt_dit.blocks import attention_opt
from models.tt_dit.blocks.attention_opt import Attention

FAST, LOW = ttnn.SDPAPrecision.FAST, ttnn.SDPAPrecision.LOW_PRECISION


def _bare_attention(precision=None, *, sp=4, tp=8):
    # Bypass __init__, which needs a mesh device; only host-side helpers are under test.
    attention = object.__new__(Attention)
    attention.sdpa_precision = precision
    attention.sdpa_compute_kernel_config = "initial"
    attention.parallel_config = SimpleNamespace(
        sequence_parallel=SimpleNamespace(factor=sp), tensor_parallel=SimpleNamespace(factor=tp)
    )
    attention.ring_sdpa_program_config = {}
    attention.ring_sdpa_worker_grid = (12, 9)
    return attention


def _chunks(config):
    return config.q_chunk_size, config.k_chunk_size


@pytest.fixture
def blackhole(monkeypatch):
    monkeypatch.setattr(attention_opt, "is_blackhole", lambda: True)


def test_legacy_ring_program_config_unchanged(blackhole):
    attention = _bare_attention(None)
    config = attention.get_ring_sdpa_program_config(4096 * 16)
    assert _chunks(config) == (192, 512)
    assert config.exp_approx_mode is False


@pytest.mark.parametrize(
    "sp, tp, seq_len, expected",
    [
        (4, 8, 4096, (128, 256)),  # tuned 1024x1024 chunk kept
        (4, 8, 4096 * 16, (192, 512)),  # 6 Q tiles (even) kept
        (8, 4, 16384, (320, 384)),  # 10 Q tiles, K384 kept
        (2, 2, 1234, (128, 512)),  # default entry
    ],
)
def test_recipe_ring_program_config_keeps_tuned_chunks(blackhole, sp, tp, seq_len, expected):
    config = _bare_attention(FAST, sp=sp, tp=tp).get_ring_sdpa_program_config(seq_len)
    assert _chunks(config) == expected
    assert config.exp_approx_mode is None
    assert (config.compute_with_storage_grid_size.x, config.compute_with_storage_grid_size.y) == (12, 9)


def test_recipe_ring_program_config_falls_back(blackhole):
    attention = _bare_attention(FAST)
    attention.ring_sdpa_chunk_size_map = {(True, 4, 8): {-1: (224, 1024)}}
    # 7 Q tiles is odd (ring needs even), K1024 is unsupported.
    assert _chunks(attention.get_ring_sdpa_program_config(4096)) == (256, 512)


def test_recipe_dense_program_config():
    tuned = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(12, 9), q_chunk_size=224, k_chunk_size=512, exp_approx_mode=False
    )
    dense = Attention._recipe_program_config(tuned, ring=False)
    assert _chunks(dense) == (224, 512)  # joint SDPA accepts odd tile counts
    assert dense.exp_approx_mode is None
    assert _chunks(Attention._recipe_program_config(tuned, ring=True)) == (256, 512)


def test_legacy_kwargs_read_compute_config_at_call_time():
    attention = _bare_attention(None)
    assert attention._sdpa_kwargs() == {"compute_kernel_config": "initial"}
    attention.sdpa_compute_kernel_config = "reassigned"
    assert attention._sdpa_kwargs() == {"compute_kernel_config": "reassigned"}
    assert attention._ring_buffer_kwargs(None) == {}


def test_recipe_kwargs_replace_compute_config():
    assert _bare_attention(FAST)._sdpa_kwargs() == {"precision": FAST, "inputs_prepared": False}
    assert _bare_attention(LOW)._sdpa_kwargs() == {"precision": LOW, "inputs_prepared": True}


def test_recipe_rejects_device_tensor_logical_lengths():
    length = ttnn.from_torch(torch.zeros(1, dtype=torch.int32))
    _bare_attention(None)._check_recipe_logical_lengths(4096, length)  # legacy: untouched
    _bare_attention(FAST)._check_recipe_logical_lengths(4096, 512)
    with pytest.raises(ValueError):
        _bare_attention(FAST)._check_recipe_logical_lengths(4096, length)


def test_validate_head_dim_and_kv_dtype():
    assert Attention._validate_sdpa_recipe(None, None, head_dim=64, blackhole=False) == ttnn.bfloat16
    assert Attention._validate_sdpa_recipe(LOW, ttnn.bfloat8_b, head_dim=128, blackhole=True) == ttnn.bfloat8_b
    for precision, kv_dtype, head_dim, bh in [
        (FAST, None, 96, True),  # D96 is not a recipe head dim
        (FAST, None, 128, False),  # Wormhole unsupported
        (FAST, ttnn.bfloat8_b, 128, True),  # packed KV needs LOW_PRECISION
        (None, ttnn.bfloat8_b, 128, True),  # KV dtype without a recipe
    ]:
        with pytest.raises(ValueError):
            Attention._validate_sdpa_recipe(precision, kv_dtype, head_dim=head_dim, blackhole=bh)
