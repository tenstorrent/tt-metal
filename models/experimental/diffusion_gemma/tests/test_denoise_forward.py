# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase
import pytest

from models.experimental.diffusion_gemma.tt.denoise_forward import (
    DenoiseLogitsAdapter,
    denoise_attention_forward,
    make_denoise_logits_adapter_from_kv_cache,
    make_denoise_logits_adapter_from_checkpoint_state,
    make_denoise_logits_adapter_from_remapped_state,
    make_generation_logits_fn_builder_from_remapped_state,
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


def test_make_denoise_logits_adapter_from_kv_cache_wires_prompt_kv_and_self_conditioning():
    calls = {}
    model = _FakeModel(num_layers=2)

    def fake_read(tt_model, *, prompt_len, seq_len_start=0):
        calls["read"] = (tt_model, prompt_len, seq_len_start)
        return ["kv0", "kv1"]

    class _FakeAdapter:
        def __init__(self, tt_model, **kwargs):
            calls["adapter"] = (tt_model, kwargs)

    out = make_denoise_logits_adapter_from_kv_cache(
        model,
        prompt_len=64,
        seq_len_start=32,
        self_conditioning="self-conditioning",
        self_conditioning_embedding_weight="embedding",
        self_conditioning_compute_kernel_config="kernel",
        read_prompt_kv_fn=fake_read,
        adapter_cls=_FakeAdapter,
    )

    assert isinstance(out, _FakeAdapter)
    assert calls["read"] == (model, 64, 32)
    tt_model, kwargs = calls["adapter"]
    assert tt_model is model
    assert kwargs["prompt_hidden_by_layer"] == ["kv0", "kv1"]
    assert kwargs["self_conditioning"] == "self-conditioning"
    assert kwargs["self_conditioning_embedding_weight"] == "embedding"
    assert kwargs["self_conditioning_compute_kernel_config"] == "kernel"
    assert kwargs["q_rope_offset"] == 64


def test_make_denoise_logits_adapter_from_kv_cache_accepts_explicit_rope_offset():
    calls = {}

    class _FakeAdapter:
        def __init__(self, tt_model, **kwargs):
            calls["q_rope_offset"] = kwargs["q_rope_offset"]

    make_denoise_logits_adapter_from_kv_cache(
        _FakeModel(),
        prompt_len=64,
        q_rope_offset=64 + 2 * 256,
        read_prompt_kv_fn=lambda *args, **kwargs: ["kv"],
        adapter_cls=_FakeAdapter,
    )

    assert calls["q_rope_offset"] == 576


def test_make_denoise_logits_adapter_from_checkpoint_state_builds_full_adapter_inputs():
    calls = {}
    model = _FakeModel(num_layers=2)
    config = {"hidden_size": 8, "intermediate_size": 6, "rms_norm_eps": 1e-5}

    def fake_self_conditioning_builder(device, state_dict, **kwargs):
        calls["self_conditioning"] = (device, state_dict, kwargs)
        return "self-conditioning"

    def fake_embedding_builder(device, embedding_weight, **kwargs):
        calls["embedding"] = (device, embedding_weight, kwargs)
        return "embedding-tt"

    def fake_adapter_builder(tt_model, **kwargs):
        calls["adapter"] = (tt_model, kwargs)
        return "adapter"

    out = make_denoise_logits_adapter_from_checkpoint_state(
        model,
        prompt_len=64,
        seq_len_start=32,
        self_conditioning_state={"weights": "state"},
        embedding_weight="embedding-weight",
        config=config,
        q_rope_offset=576,
        self_conditioning_dtype="dtype",
        self_conditioning_compute_kernel_config="kernel",
        self_conditioning_builder=fake_self_conditioning_builder,
        embedding_weight_builder=fake_embedding_builder,
        adapter_builder=fake_adapter_builder,
    )

    assert out == "adapter"
    assert calls["self_conditioning"] == (
        model.mesh_device,
        {"weights": "state"},
        {
            "config": config,
            "hidden_size": None,
            "intermediate_size": None,
            "eps": None,
            "dtype": "dtype",
        },
    )
    assert calls["embedding"] == (
        model.mesh_device,
        "embedding-weight",
        {"hidden_size": 8, "dtype": "dtype"},
    )
    assert calls["adapter"][0] is model
    assert calls["adapter"][1] == {
        "prompt_len": 64,
        "seq_len_start": 32,
        "self_conditioning": "self-conditioning",
        "self_conditioning_embedding_weight": "embedding-tt",
        "self_conditioning_compute_kernel_config": "kernel",
        "q_rope_offset": 576,
    }


def test_make_denoise_logits_adapter_from_remapped_state_uses_backbone_embedding_key():
    calls = {}

    def fake_checkpoint_builder(tt_model, **kwargs):
        calls["builder"] = (tt_model, kwargs)
        return "adapter"

    out = make_denoise_logits_adapter_from_remapped_state(
        "model",
        prompt_len=64,
        backbone_state={"model.language_model.embed_tokens.weight": "embedding-weight"},
        self_conditioning_state={"self": "conditioning"},
        config="config",
        checkpoint_adapter_builder=fake_checkpoint_builder,
    )

    assert out == "adapter"
    assert calls["builder"] == (
        "model",
        {
            "prompt_len": 64,
            "self_conditioning_state": {"self": "conditioning"},
            "embedding_weight": "embedding-weight",
            "config": "config",
        },
    )


def test_make_denoise_logits_adapter_from_remapped_state_rejects_missing_embedding():
    with pytest.raises(ValueError, match="missing tied embedding weight"):
        make_denoise_logits_adapter_from_remapped_state(
            "model",
            prompt_len=64,
            backbone_state={},
            self_conditioning_state={},
        )


def test_make_generation_logits_fn_builder_from_remapped_state_matches_generate_hook():
    calls = {}
    backbone_state = {"model.language_model.embed_tokens.weight": "embedding-weight"}
    self_conditioning_state = {"self": "conditioning"}

    def fake_adapter_builder(tt_model, **kwargs):
        calls["adapter"] = (tt_model, kwargs)
        return "adapter"

    builder = make_generation_logits_fn_builder_from_remapped_state(
        backbone_state=backbone_state,
        self_conditioning_state=self_conditioning_state,
        config="config",
        seq_len_start=32,
        adapter_builder=fake_adapter_builder,
    )

    out = builder(
        "model",
        prompt_tokens="prompt-tokens",
        prompt_len=64,
        page_table="page-table",
        page_tables_per_layer=["layer-pages"],
    )

    assert out == "adapter"
    assert calls["adapter"] == (
        "model",
        {
            "prompt_len": 64,
            "backbone_state": backbone_state,
            "self_conditioning_state": self_conditioning_state,
            "config": "config",
            "seq_len_start": 32,
        },
    )
