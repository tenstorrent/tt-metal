import inspect

import pytest

from models.demos.gemma4.tt.attention import Gemma4Attention
from models.demos.gemma4.tt.attention.decode import decode_forward
from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase, coerce_kv_cache_phase
from models.demos.gemma4.tt.attention.prefill import prefill_forward
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.model import Gemma4Model


class _FakeHidden:
    shape = (1, 1, 32, 16)


def test_kv_phase_defaults_preserve_gemma4_write_paths():
    assert coerce_kv_cache_phase(None, is_decode=False) == KVCachePhase.PREFILL_WRITE
    assert coerce_kv_cache_phase(None, is_decode=True) == KVCachePhase.COMMIT_APPEND


def test_kv_phase_accepts_explicit_readonly_value():
    assert coerce_kv_cache_phase("denoise_readonly", is_decode=False) == KVCachePhase.DENOISE_READONLY


def test_kv_phase_rejects_decode_readonly_value():
    with pytest.raises(ValueError, match="DENOISE_READONLY is a prefill-only KV phase"):
        coerce_kv_cache_phase(KVCachePhase.DENOISE_READONLY, is_decode=True)
    with pytest.raises(ValueError, match="DENOISE_READONLY is a prefill-only KV phase"):
        coerce_kv_cache_phase("denoise_readonly", is_decode=True)


def test_kv_phase_rejects_mode_mismatches():
    with pytest.raises(ValueError, match="PREFILL_WRITE is a prefill-only KV phase"):
        coerce_kv_cache_phase(KVCachePhase.PREFILL_WRITE, is_decode=True)
    with pytest.raises(ValueError, match="PREFILL_WRITE is a prefill-only KV phase"):
        coerce_kv_cache_phase("prefill_write", is_decode=True)
    with pytest.raises(ValueError, match="COMMIT_APPEND is a decode-only KV phase"):
        coerce_kv_cache_phase(KVCachePhase.COMMIT_APPEND, is_decode=False)
    with pytest.raises(ValueError, match="COMMIT_APPEND is a decode-only KV phase"):
        coerce_kv_cache_phase("commit_append", is_decode=False)


def test_kv_phase_is_threaded_through_attention_call_chain():
    assert "kv_phase" in inspect.signature(Gemma4Model.__call__).parameters
    assert "attn_mask" in inspect.signature(Gemma4Model.__call__).parameters
    assert "kv_hidden_states" in inspect.signature(Gemma4Model.__call__).parameters
    assert "prefix_kv_by_layer" in inspect.signature(Gemma4Model.__call__).parameters
    assert "q_rope_offset" in inspect.signature(Gemma4Model.__call__).parameters
    assert "kv_phase" in inspect.signature(Gemma4Model.ttnn_prefill_forward).parameters
    assert "attn_mask" in inspect.signature(Gemma4Model.ttnn_prefill_forward).parameters
    assert "kv_hidden_states" in inspect.signature(Gemma4Model.ttnn_prefill_forward).parameters
    assert "prefix_kv_by_layer" in inspect.signature(Gemma4Model.ttnn_prefill_forward).parameters
    assert "q_rope_offset" in inspect.signature(Gemma4Model.ttnn_prefill_forward).parameters
    assert "kv_phase" in inspect.signature(Gemma4Model.ttnn_decode_forward).parameters
    assert "kv_phase" in inspect.signature(Gemma4Model.ttnn_verify_forward).parameters
    assert "kv_phase" in inspect.signature(Gemma4DecoderLayer.__call__).parameters
    assert "attn_mask" in inspect.signature(Gemma4DecoderLayer.__call__).parameters
    assert "kv_hidden_states" in inspect.signature(Gemma4DecoderLayer.__call__).parameters
    assert "prefix_kv" in inspect.signature(Gemma4DecoderLayer.__call__).parameters
    assert "q_rope_offset" in inspect.signature(Gemma4DecoderLayer.__call__).parameters
    assert "kv_phase" in inspect.signature(Gemma4Attention.__call__).parameters
    assert "attn_mask" in inspect.signature(Gemma4Attention.__call__).parameters
    assert "kv_hidden_states" in inspect.signature(Gemma4Attention.__call__).parameters
    assert "prefix_kv" in inspect.signature(Gemma4Attention.__call__).parameters
    assert "q_rope_offset" in inspect.signature(Gemma4Attention.__call__).parameters
    assert "write_kv_cache" in inspect.signature(prefill_forward).parameters
    assert "attn_mask" in inspect.signature(prefill_forward).parameters
    assert "kv_hidden_states" in inspect.signature(prefill_forward).parameters
    assert "prefix_kv" in inspect.signature(prefill_forward).parameters
    assert "q_rope_offset" in inspect.signature(prefill_forward).parameters
    assert "write_kv_cache" in inspect.signature(decode_forward).parameters


def test_model_rejects_prefix_kv_layer_count_mismatch_before_layer_forward():
    model = object.__new__(Gemma4Model)
    model.layers = [object(), object()]
    model.tt_kv_cache = [None, None]

    with pytest.raises(ValueError, match="prefix_kv_by_layer has 1 entries but model has 2 layers"):
        Gemma4Model.__call__(
            model,
            _FakeHidden(),
            prefix_kv_by_layer=[("k0", "v0")],
        )


def test_model_rejects_model_level_kv_hidden_states_for_multi_layer_forward():
    model = object.__new__(Gemma4Model)
    model.layers = [object(), object()]
    model.tt_kv_cache = [None, None]

    with pytest.raises(ValueError, match="model-level kv_hidden_states/q_rope_offset"):
        Gemma4Model.__call__(
            model,
            _FakeHidden(),
            kv_hidden_states=object(),
        )


def test_model_rejects_model_level_q_rope_offset_for_multi_layer_forward():
    model = object.__new__(Gemma4Model)
    model.layers = [object(), object()]
    model.tt_kv_cache = [None, None]

    with pytest.raises(ValueError, match="model-level kv_hidden_states/q_rope_offset"):
        Gemma4Model.__call__(
            model,
            _FakeHidden(),
            q_rope_offset=32,
        )
