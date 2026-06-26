# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase
import pytest

from models.experimental.diffusion_gemma.tt.denoise_forward import (
    DenoiseLogitsAdapter,
    denoise_attention_forward,
    read_prompt_kv_cache_by_layer,
)


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape


class _FakeAttention:
    def __init__(self):
        self.kwargs = None

    def __call__(self, hidden_states, **kwargs):
        self.kwargs = kwargs
        return hidden_states


class _FakeLayer:
    def __init__(self):
        self.self_attn = _FakeAttention()


class _FakeModel:
    def __init__(self, num_layers=1):
        self.mesh_device = object()
        self.layers = [_FakeLayer() for _ in range(num_layers)]
        self.tt_kv_cache = [f"cache-{idx}" for idx in range(num_layers)]
        self.rope_requests = []

    def _get_rope_mats(self, layer_idx, seq_len):
        self.rope_requests.append((layer_idx, seq_len))
        return ("cos", "sin")


def test_denoise_attention_defaults_to_maskless_noncausal_prefix_kv():
    model = _FakeModel()
    prompt_kv = (_FakeTensor([1, 1, 64, 16]), _FakeTensor([1, 1, 64, 16]))
    canvas_hidden = _FakeTensor([1, 1, 256, 32])

    out = denoise_attention_forward(
        model,
        layer_idx=0,
        prompt_kv=prompt_kv,
        canvas_hidden=canvas_hidden,
    )

    assert out is canvas_hidden
    kwargs = model.layers[0].self_attn.kwargs
    assert kwargs["is_decode"] is False
    assert kwargs["is_causal"] is False
    assert kwargs["kv_phase"] is KVCachePhase.DENOISE_READONLY
    assert kwargs["attn_mask"] is None
    assert kwargs["prefix_kv"] is prompt_kv
    assert kwargs["q_rope_offset"] == 64
    assert model.rope_requests == [(0, 320)]


def test_denoise_attention_accepts_explicit_canvas_rope_offset_for_later_blocks():
    model = _FakeModel()
    prompt_kv = (_FakeTensor([1, 1, 64, 16]), _FakeTensor([1, 1, 64, 16]))
    canvas_hidden = _FakeTensor([1, 1, 256, 32])

    denoise_attention_forward(
        model,
        layer_idx=0,
        prompt_kv=prompt_kv,
        canvas_hidden=canvas_hidden,
        q_rope_offset=64 + 2 * 256,
    )

    kwargs = model.layers[0].self_attn.kwargs
    assert kwargs["q_rope_offset"] == 576
    assert model.rope_requests == [(0, 832)]


def test_denoise_logits_adapter_threads_canvas_rope_offset():
    calls = []

    def _fake_logits_from_tokens(tt_model, **kwargs):
        calls.append(kwargs)
        return "logits"

    adapter = DenoiseLogitsAdapter(
        object(),
        prompt_hidden_by_layer=["prompt"],
        q_rope_offset=576,
        logits_from_tokens=_fake_logits_from_tokens,
    )

    assert adapter("canvas", 0) == "logits"
    assert calls[0]["prompt_hidden_by_layer"] == ["prompt"]
    assert calls[0]["canvas_tokens"] == "canvas"
    assert calls[0]["q_rope_offset"] == 576


def test_read_prompt_kv_cache_by_layer_reads_every_model_layer():
    calls = []
    model = _FakeModel(num_layers=3)

    def fake_read(kv_cache, *, prompt_len, seq_len_start=0):
        calls.append((kv_cache, prompt_len, seq_len_start))
        return (f"k-{kv_cache}", f"v-{kv_cache}")

    out = read_prompt_kv_cache_by_layer(model, prompt_len=64, seq_len_start=32, read_fn=fake_read)

    assert out == [
        ("k-cache-0", "v-cache-0"),
        ("k-cache-1", "v-cache-1"),
        ("k-cache-2", "v-cache-2"),
    ]
    assert calls == [("cache-0", 64, 32), ("cache-1", 64, 32), ("cache-2", 64, 32)]


def test_read_prompt_kv_cache_by_layer_rejects_cache_layer_mismatch():
    model = _FakeModel(num_layers=2)
    model.tt_kv_cache = ["cache-0"]

    with pytest.raises(ValueError, match="tt_kv_cache has 1 layers"):
        read_prompt_kv_cache_by_layer(model, prompt_len=64, read_fn=lambda *args, **kwargs: None)
