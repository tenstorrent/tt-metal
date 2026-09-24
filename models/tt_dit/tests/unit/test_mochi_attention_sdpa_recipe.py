# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: Mochi (attention_mochi.py) SDPA recipe wiring (no device).

The default recipe itself is covered by test_sdpa_dit_recipe_defaults.py."""

import pytest
import torch
import ttnn

from models.tt_dit.models.transformers.attention_mochi import MochiAttention

FAST, LOW = ttnn.SDPAPrecision.FAST, ttnn.SDPAPrecision.LOW_PRECISION


def _bare_attention(precision=None):
    # Bypass __init__, which needs a mesh device; only host-side helpers are under test.
    attention = object.__new__(MochiAttention)
    attention.sdpa_precision = precision
    attention.sdpa_compute_kernel_config = "initial"
    return attention


def test_tuned_chunks_are_legacy_only():
    # On Blackhole every call runs a recipe with op-selected chunks; the table serves other archs only.
    assert all(not key[0] for key in MochiAttention.sdpa_chunk_size_map)


def test_legacy_kwargs_read_compute_config_at_call_time():
    attention = _bare_attention(None)
    assert attention._sdpa_kwargs() == {"compute_kernel_config": "initial"}
    attention.sdpa_compute_kernel_config = "reassigned"
    assert attention._sdpa_kwargs() == {"compute_kernel_config": "reassigned"}
    assert attention._ring_buffer_kwargs(None) == {}


def test_recipe_kwargs_replace_compute_config():
    assert _bare_attention(FAST)._sdpa_kwargs() == {"precision": FAST, "inputs_prepared": False}
    assert _bare_attention(LOW)._sdpa_kwargs() == {"precision": LOW, "inputs_prepared": True}


def test_recipe_rejects_device_tensor_logical_length():
    length = ttnn.from_torch(torch.zeros(1, dtype=torch.int32))
    _bare_attention(None)._check_recipe_logical_lengths(length)  # legacy: untouched
    _bare_attention(FAST)._check_recipe_logical_lengths(44520)
    with pytest.raises(ValueError):
        _bare_attention(FAST)._check_recipe_logical_lengths(length)


def test_validate_head_dim_and_kv_dtype():
    assert MochiAttention._validate_sdpa_recipe(None, None, head_dim=64, blackhole=False) == ttnn.bfloat16
    assert MochiAttention._validate_sdpa_recipe(LOW, ttnn.bfloat4_b, head_dim=128, blackhole=True) == ttnn.bfloat4_b
    for precision, kv_dtype, head_dim, bh in [
        (FAST, None, 96, True),  # D96 is not a recipe head dim
        (FAST, None, 128, False),  # Wormhole unsupported
        (FAST, ttnn.bfloat8_b, 128, True),  # packed KV needs LOW_PRECISION
        (None, ttnn.bfloat8_b, 128, True),  # KV dtype without a recipe
    ]:
        with pytest.raises(ValueError):
            MochiAttention._validate_sdpa_recipe(precision, kv_dtype, head_dim=head_dim, blackhole=bh)
