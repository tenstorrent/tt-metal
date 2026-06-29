import inspect

import pytest

from models.demos.gemma4.tt.attention import Gemma4Attention
from models.demos.gemma4.tt.attention.prefill import prefill_forward
from models.demos.gemma4.tt.model import Gemma4Model
from models.experimental.diffusion_gemma.kv_phase import KVCachePhase, coerce_kv_cache_phase
from models.experimental.diffusion_gemma.tt.diffusion_attention import denoise_attention


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


def test_denoise_knobs_are_isolated_from_shared_gemma4_attention():
    """The bidirectional/prefix-KV/RoPE-offset knobs live ONLY in DiffusionGemma.

    The shared Gemma4 attention stack must stay on its stock causal signature so
    the backbone is untouched; the denoise-specific knobs belong to the
    diffusion-local ``denoise_attention`` helper instead.
    """
    diffusion_params = {"attn_mask", "kv_hidden_states", "prefix_kv", "q_rope_offset", "kv_phase"}
    for fn in (Gemma4Model.__call__, Gemma4Attention.__call__, prefill_forward):
        shared = set(inspect.signature(fn).parameters) & diffusion_params
        assert not shared, f"{fn.__qualname__} leaked diffusion-only kwargs: {shared}"

    denoise_sig = inspect.signature(denoise_attention).parameters
    for name in ("attn_mask", "kv_hidden_states", "prefix_kv", "q_rope_offset"):
        assert name in denoise_sig
