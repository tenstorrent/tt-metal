import inspect

from models.demos.gemma4.tt.attention import Gemma4Attention
from models.demos.gemma4.tt.attention.decode import decode_forward
from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase, coerce_kv_cache_phase
from models.demos.gemma4.tt.attention.prefill import prefill_forward
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.model import Gemma4Model


def test_kv_phase_defaults_preserve_gemma4_write_paths():
    assert coerce_kv_cache_phase(None, is_decode=False) == KVCachePhase.PREFILL_WRITE
    assert coerce_kv_cache_phase(None, is_decode=True) == KVCachePhase.COMMIT_APPEND


def test_kv_phase_accepts_explicit_readonly_value():
    assert coerce_kv_cache_phase("denoise_readonly", is_decode=False) == KVCachePhase.DENOISE_READONLY


def test_kv_phase_is_threaded_through_attention_call_chain():
    assert "kv_phase" in inspect.signature(Gemma4Model.__call__).parameters
    assert "kv_phase" in inspect.signature(Gemma4Model.ttnn_prefill_forward).parameters
    assert "kv_phase" in inspect.signature(Gemma4Model.ttnn_decode_forward).parameters
    assert "kv_phase" in inspect.signature(Gemma4DecoderLayer.__call__).parameters
    assert "kv_phase" in inspect.signature(Gemma4Attention.__call__).parameters
    assert "write_kv_cache" in inspect.signature(prefill_forward).parameters
    assert "write_kv_cache" in inspect.signature(decode_forward).parameters
