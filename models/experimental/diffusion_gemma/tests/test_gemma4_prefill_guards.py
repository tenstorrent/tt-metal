import pytest
import torch
from types import SimpleNamespace

from models.demos.gemma4.tt.attention.prefill import _slice_rope_cache, _validate_q_rope_offset
from models.demos.gemma4.tt.model import Gemma4Model


def test_q_rope_offset_must_be_tile_aligned():
    _validate_q_rope_offset(32, batch_size=1)
    with pytest.raises(ValueError, match="q_rope_offset must be a multiple of 32"):
        _validate_q_rope_offset(1, batch_size=1)
    with pytest.raises(ValueError, match="RoPE cache start must be a multiple of 32"):
        _slice_rope_cache(None, 1, 32)


def test_batched_prefill_rejects_nonzero_q_rope_offset():
    _validate_q_rope_offset(0, batch_size=2)
    with pytest.raises(ValueError, match="nonzero q_rope_offset is only supported for single-user prefill"):
        _validate_q_rope_offset(32, batch_size=2)


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

    cos, sin = Gemma4Model._get_rope_mats(model, 0, seq_len=cache_len)
    assert cos.shape[-2] == cache_len
    assert sin.shape[-2] == cache_len
    with pytest.raises(ValueError, match="requested RoPE seq_len 262176 exceeds cache length 262144"):
        Gemma4Model._get_rope_mats(model, 0, seq_len=cache_len + 32)


def test_slice_rope_cache_rejects_overflow():
    cache = SimpleNamespace(shape=[1, 1, 262144, 8])
    with pytest.raises(ValueError, match=r"RoPE cache slice \[262144, 262176\) exceeds cache length 262144"):
        _slice_rope_cache(cache, 262144, 32)
