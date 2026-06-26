import pytest

from models.demos.gemma4.tt.attention.prefill import _slice_rope_cache, _validate_q_rope_offset


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
