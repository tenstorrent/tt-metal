# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from models.experimental.diffusion_gemma.tt import denoise_forward as DF
from models.experimental.diffusion_gemma.tt.denoise_forward import (
    DenoiseLogitsAdapter,
    _build_denoise_attn_mask_for_layer,
    denoise_attention_forward,
    embed_canvas_tokens,
    make_denoise_logits_adapter_from_kv_cache,
    make_denoise_logits_adapter_from_checkpoint_state,
    make_denoise_logits_adapter_from_remapped_state,
    make_generation_logits_fn_builder_from_checkpoint_state,
    make_generation_logits_fn_builder_from_remapped_state,
    read_prompt_kv_cache_by_layer,
)


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape


class _FakeAttention:
    """Stand-in for a Gemma4Attention instance (identity marker)."""


class _FakeLayer:
    def __init__(self):
        self.self_attn = _FakeAttention()


class _RecordingDenoiseAttention:
    """Records the args ``denoise_attention`` is called with and echoes hidden."""

    def __init__(self):
        self.calls = []

    def __call__(self, attn, hidden_states, **kwargs):
        self.calls.append((attn, hidden_states, kwargs))
        return hidden_states


class _FakeModel:
    def __init__(self, num_layers=1, *, layer_types=None, sliding_window=1024):
        self.mesh_device = object()
        self.layers = [_FakeLayer() for _ in range(num_layers)]
        self.tt_kv_cache = [f"cache-{idx}" for idx in range(num_layers)]
        self.rope_requests = []
        if layer_types is not None:
            self.hf_config = SimpleNamespace(layer_types=layer_types, sliding_window=sliding_window)

    def _get_rope_mats(self, layer_idx, seq_len):
        self.rope_requests.append((layer_idx, seq_len))
        return ("cos", "sin")


def test_denoise_attention_defaults_to_maskless_noncausal_prefix_kv(monkeypatch):
    model = _FakeModel()
    recorder = _RecordingDenoiseAttention()
    monkeypatch.setattr(DF, "denoise_attention", recorder)
    prompt_kv = (_FakeTensor([1, 1, 64, 16]), _FakeTensor([1, 1, 64, 16]))
    canvas_hidden = _FakeTensor([1, 1, 256, 32])

    out = denoise_attention_forward(
        model,
        layer_idx=0,
        prompt_kv=prompt_kv,
        canvas_hidden=canvas_hidden,
    )

    assert out is canvas_hidden
    attn, hidden, kwargs = recorder.calls[0]
    assert attn is model.layers[0].self_attn
    assert hidden is canvas_hidden
    assert kwargs["attn_mask"] is None
    assert kwargs["kv_hidden_states"] is None
    assert kwargs["prefix_kv"] is prompt_kv
    assert kwargs["q_rope_offset"] == 64
    assert model.rope_requests == [(0, 320)]


def test_denoise_attention_accepts_explicit_canvas_rope_offset_for_later_blocks(monkeypatch):
    model = _FakeModel()
    recorder = _RecordingDenoiseAttention()
    monkeypatch.setattr(DF, "denoise_attention", recorder)
    prompt_kv = (_FakeTensor([1, 1, 64, 16]), _FakeTensor([1, 1, 64, 16]))
    canvas_hidden = _FakeTensor([1, 1, 256, 32])

    denoise_attention_forward(
        model,
        layer_idx=0,
        prompt_kv=prompt_kv,
        canvas_hidden=canvas_hidden,
        q_rope_offset=64 + 2 * 256,
    )

    _, _, kwargs = recorder.calls[0]
    assert kwargs["q_rope_offset"] == 576
    assert model.rope_requests == [(0, 832)]


def test_denoise_attn_mask_builder_skips_full_and_short_sliding_layers():
    calls = []
    model = _FakeModel(
        num_layers=2,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=1024,
    )

    def mask_builder(*args, **kwargs):
        calls.append((args, kwargs))
        return "mask"

    assert (
        _build_denoise_attn_mask_for_layer(
            model,
            0,
            prompt_len=2048,
            canvas_len=256,
            mask_builder=mask_builder,
        )
        is None
    )
    assert (
        _build_denoise_attn_mask_for_layer(
            model,
            1,
            prompt_len=64,
            canvas_len=256,
            mask_builder=mask_builder,
        )
        is None
    )
    assert calls == []


def test_denoise_attn_mask_builder_materializes_long_prompt_sliding_mask():
    calls = []
    model = _FakeModel(num_layers=1, layer_types=["sliding_attention"], sliding_window=4)

    def mask_builder(*args, **kwargs):
        calls.append((args, kwargs))
        return "mask"

    assert (
        _build_denoise_attn_mask_for_layer(
            model,
            0,
            prompt_len=10,
            canvas_len=6,
            mask_builder=mask_builder,
        )
        == "mask"
    )
    assert calls == [
        (
            (model.mesh_device,),
            {
                "prompt_len": 10,
                "canvas_len": 6,
                "layer_type": "sliding_attention",
                "sliding_window": 4,
            },
        )
    ]


def test_denoise_attn_mask_builder_rejects_missing_sliding_window():
    model = _FakeModel(num_layers=1, layer_types=["sliding_attention"], sliding_window=None)

    with pytest.raises(ValueError, match="requires a positive sliding_window"):
        _build_denoise_attn_mask_for_layer(
            model,
            0,
            prompt_len=10,
            canvas_len=6,
            mask_builder=lambda *args, **kwargs: "mask",
        )


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


def test_embed_canvas_tokens_rejects_batch_greater_than_one():
    with pytest.raises(ValueError, match="batch=1"):
        embed_canvas_tokens(object(), _FakeTensor([2, 256]))


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


def test_make_generation_logits_fn_builder_from_checkpoint_state_remaps_once():
    calls = {}
    dg_state = {"raw": "state"}

    def fake_remap(state_dict):
        calls["remap"] = state_dict
        return {"backbone": "state"}, {"self": "conditioning"}, ["ignored"]

    def fake_remapped_builder(**kwargs):
        calls["builder"] = kwargs
        return "builder"

    out = make_generation_logits_fn_builder_from_checkpoint_state(
        dg_state,
        config="config",
        seq_len_start=32,
        remap_fn=fake_remap,
        remapped_builder=fake_remapped_builder,
    )

    assert out == "builder"
    assert calls["remap"] is dg_state
    assert calls["builder"] == {
        "backbone_state": {"backbone": "state"},
        "self_conditioning_state": {"self": "conditioning"},
        "config": "config",
        "seq_len_start": 32,
    }
