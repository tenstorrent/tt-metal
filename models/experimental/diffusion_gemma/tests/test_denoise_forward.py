# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the DiffusionGemma denoise forward pass.

Covers the per-layer denoise forward (attention, norms, MoE dispatch, the logits adapter and its
builders), the per-layer sliding-window reveal masks and the bounded prefix read (#51080), and the
on-device full-canvas RMSNorm topology probes.
"""

import pathlib
from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.reference.attention_mask import build_canvas_reveal_denoise_mask
from models.experimental.diffusion_gemma.tt import denoise_forward as DF
from models.experimental.diffusion_gemma.tt.denoise_forward import (
    DenoiseLogitsAdapter,
    _build_denoise_attn_mask_for_layer,
    _chunked_norm_forward,
    _denoise_router_forward,
    denoise_attention_forward,
    embed_canvas_tokens,
    make_denoise_logits_adapter_from_kv_cache,
    make_denoise_logits_adapter_from_checkpoint_state,
    make_denoise_logits_adapter_from_remapped_state,
    make_generation_logits_fn_builder_from_checkpoint_state,
    make_generation_logits_fn_builder_from_remapped_state,
    read_prompt_kv_cache_by_layer,
    read_prompt_kv_cache_slice,
)

# --- test doubles ---------------------------------------------------------------------------


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.deallocated = False

    def deallocate(self, force):
        self.deallocated = force

    def device(self):
        # Real tensors have one, and the norm path asks for it to pick a compute kernel config. A
        # double that omits it only passes while nothing reaches that code -- the same way
        # _FakeModel without .layers passed until the bounded read stopped being opt-in.
        return "fake-device"


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


class _KVCache:
    def __init__(self, span, head_dim=16):
        self.shape = [1, 2, span, head_dim]


def _kv_cache_model(spans, head_dim=16):
    return SimpleNamespace(
        tt_kv_cache=[(_KVCache(span, head_dim), _KVCache(span, head_dim)) for span in spans],
        layers=[None] * len(spans),
    )


# --- adapter lifecycle ----------------------------------------------------------------------


def test_denoise_adapter_reset_releases_all_trace_persistent_buffers():
    tensors = {name: _FakeTensor((name,)) for name in ("prev", "signal-a", "signal-b", "cos", "sin")}
    released_prefix_windows = []
    adapter = object.__new__(DenoiseLogitsAdapter)
    adapter.prev_logits = tensors["prev"]
    adapter.signal_buf = tensors["signal-a"]
    adapter.signal_buf_b = tensors["signal-b"]
    adapter.trace_safe_self_conditioning = True
    adapter.signal_ping_pong = True
    adapter._canvas_rope_bufs = {"sliding_attention": (tensors["cos"], tensors["sin"])}
    adapter.use_canvas_rope = True
    adapter.prompt_hidden_by_layer = SimpleNamespace(
        release_window_buffers=lambda: released_prefix_windows.append(True)
    )

    adapter.reset()

    assert all(tensor.deallocated for tensor in tensors.values())
    assert released_prefix_windows == [True]
    assert adapter.prev_logits is None
    assert adapter.signal_buf is None
    assert adapter.signal_buf_b is None
    assert adapter.trace_safe_self_conditioning is False
    assert adapter.signal_ping_pong is False
    assert adapter._canvas_rope_bufs == {}
    assert adapter.use_canvas_rope is False


def test_denoise_adapter_advances_mutable_prefix_after_commit():
    seen = []
    adapter = object.__new__(DenoiseLogitsAdapter)
    adapter.prompt_hidden_by_layer = SimpleNamespace(set_prompt_len=seen.append)
    adapter.prompt_len = 32
    adapter.q_rope_offset = 32

    assert adapter.advance_prefix_after_commit(288) is True
    assert seen == [288]
    assert adapter.prompt_len == 288
    assert adapter.q_rope_offset == 288


# --- denoise attention ----------------------------------------------------------------------


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


# --- per-layer attention mask builder -------------------------------------------------------


@pytest.mark.parametrize(
    ("layer_types", "layer_idx", "prompt_len", "canvas_len", "sliding_window", "explicit"),
    [
        pytest.param(
            ["full_attention", "sliding_attention"],
            0,
            2048,
            256,
            1024,
            {"use_explicit_sliding_mask": True},
            id="full-attention-layer",
        ),
        pytest.param(
            ["full_attention", "sliding_attention"],
            1,
            64,
            256,
            1024,
            {"use_explicit_sliding_mask": True},
            id="sliding-layer-below-the-window",
        ),
        pytest.param(
            ["sliding_attention"],
            0,
            10,
            6,
            4,
            {},
            id="long-prompt-sliding-layer-defaults-to-maskless",
        ),
    ],
)
def test_denoise_attn_mask_builder_stays_maskless(
    layer_types, layer_idx, prompt_len, canvas_len, sliding_window, explicit
):
    calls = []
    model = _FakeModel(num_layers=len(layer_types), layer_types=layer_types, sliding_window=sliding_window)

    def mask_builder(*args, **kwargs):
        calls.append((args, kwargs))
        return "mask"

    assert (
        _build_denoise_attn_mask_for_layer(
            model,
            layer_idx,
            prompt_len=prompt_len,
            canvas_len=canvas_len,
            mask_builder=mask_builder,
            **explicit,
        )
        is None
    )
    assert calls == []


def test_denoise_attn_mask_builder_can_materialize_long_prompt_sliding_mask_for_ab_tests():
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
            use_explicit_sliding_mask=True,
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


def test_denoise_attn_mask_builder_rejects_missing_sliding_window(expect_error):
    model = _FakeModel(num_layers=1, layer_types=["sliding_attention"], sliding_window=None)

    with expect_error(ValueError, match="requires a positive sliding_window"):
        _build_denoise_attn_mask_for_layer(
            model,
            0,
            prompt_len=10,
            canvas_len=6,
            use_explicit_sliding_mask=True,
            mask_builder=lambda *args, **kwargs: "mask",
        )


# --- chunked norm ---------------------------------------------------------------------------


def test_chunked_norm_forward_norms_the_whole_canvas_in_one_op(monkeypatch):
    """Weighted norms go through the DG-owned sharded norm, the WHOLE canvas in one op.

    They used to call gemma4's `norm.forward` per 32-row slice. They no longer do, for two reasons:
    that entry point takes no compute_kernel_config and so could only accumulate the 88 per-core
    partials in bf16, and 8 ops where 1 will do costs -20.4% per block. See
    _norm_compute_kernel_config for why one op and eight are now bit-identical.
    """
    seen = []

    def fake_sharded(norm, hidden, rows):
        seen.append(rows)
        return _FakeTensor(hidden.shape)

    monkeypatch.setattr(DF, "_sharded_rms_norm_rows", fake_sharded)
    norm = SimpleNamespace(with_scale=True, tt_weight=object(), eps=1e-6)

    out = _chunked_norm_forward(norm, _FakeTensor([1, 1, 96, 2816]))

    assert out.shape == [1, 1, 96, 2816]
    assert seen == [96], "the whole canvas in one op -- DG_NORM_FULLCANVAS was deleted, not defaulted"


def test_chunked_norm_forward_falls_back_to_gemma4_when_no_sharded_config_fits(monkeypatch):
    """A width with no usable core grid must still norm, via the old per-chunk gemma4 path."""
    calls = []

    class _FakeTtnn:
        DRAM_MEMORY_CONFIG = "dram"

        @staticmethod
        def slice(tensor, starts, ends, *, memory_config):
            calls.append(("slice", starts, ends, memory_config))
            return _FakeTensor([tensor.shape[0], tensor.shape[1], ends[2] - starts[2], tensor.shape[3]])

        @staticmethod
        def concat(tensors, *, dim, memory_config):
            calls.append(("concat", [tensor.shape for tensor in tensors], dim, memory_config))
            return _FakeTensor([1, 1, sum(tensor.shape[2] for tensor in tensors), tensors[0].shape[3]])

    class _FakeNorm:
        with_scale = True
        tt_weight = object()
        eps = 1e-6

        def forward(self, tensor):
            calls.append(("norm", tensor.shape))
            return _FakeTensor(tensor.shape)

    monkeypatch.setattr(DF, "ttnn", _FakeTtnn)
    monkeypatch.setattr(DF, "_sharded_rms_norm_rows", lambda norm, hidden, rows: None)

    out = _chunked_norm_forward(_FakeNorm(), _FakeTensor([1, 1, 96, 2816]))

    assert out.shape == [1, 1, 96, 2816]
    assert [call[0] for call in calls] == ["slice", "norm", "slice", "norm", "slice", "norm", "concat"]


def test_chunked_norm_forward_uses_sharded_scaleless_norm(monkeypatch):
    calls = []

    def fake_rms_norm_dram(tensor, *, epsilon, chunk_size, compute_kernel_config=None):
        calls.append((tensor.shape, epsilon, chunk_size, compute_kernel_config))
        return _FakeTensor(tensor.shape)

    monkeypatch.setattr(DF, "_rms_norm_dram", fake_rms_norm_dram)
    monkeypatch.setattr(DF, "_norm_compute_kernel_config", lambda device: "fp32-acc")
    norm = SimpleNamespace(with_scale=False, tt_weight=None, eps=1e-6)

    out = _chunked_norm_forward(norm, _FakeTensor([1, 1, 96, 2816]))

    assert out.shape == [1, 1, 96, 2816]
    # The scaleless path must ALSO get fp32 accumulation -- it is a denoise norm like any other.
    assert calls == [([1, 1, 96, 2816], 1e-6, 32, "fp32-acc")]


def test_fullcanvas_must_not_capture_the_scaleless_router_norm(monkeypatch):
    """The full-canvas path must leave the weightless MoE-router norm on its own path.

    The scaleless test used to sit BELOW the full-canvas attempt, so with the flag on the router norm
    was routed into _fullcanvas_norm and never reached _rms_norm_dram -- silently swapping its
    reduction topology from 8 cores / block_w=11 / single-stage to 88 cores / block_w=1 / two-stage.
    A flag named "full canvas" re-sharding an unrelated norm by ordering alone is the bug; this pins
    the ordering. The previous test does not catch it because it never sets the flag.
    """
    calls = []

    def fake_rms_norm_dram(tensor, *, epsilon, chunk_size, compute_kernel_config=None):
        calls.append((tensor.shape, epsilon, chunk_size))
        return _FakeTensor(tensor.shape)

    def fail_sharded(norm, hidden_states, rows):
        raise AssertionError("a scaleless norm must not reach the weighted sharded path")

    monkeypatch.setattr(DF, "_rms_norm_dram", fake_rms_norm_dram)
    monkeypatch.setattr(DF, "_sharded_rms_norm_rows", fail_sharded)
    monkeypatch.setattr(DF, "_norm_compute_kernel_config", lambda device: "fp32-acc")
    norm = SimpleNamespace(with_scale=False, tt_weight=None, eps=1e-6)

    out = _chunked_norm_forward(norm, _FakeTensor([1, 1, 256, 2816]))

    assert out.shape == [1, 1, 256, 2816]
    assert calls == [([1, 1, 256, 2816], 1e-6, 32)]


def test_the_deleted_fullcanvas_flag_stays_deleted(monkeypatch):
    """DG_NORM_FULLCANVAS is gone; setting it either way must not change anything.

    It was default OFF for two reasons, both refuted by measurement. (1) The two shapes were not
    bit-identical -- true only because ttnn's rmsnorm accumulates partials in bf16 by default; under
    fp32 accumulation they agree on 0 of 69,206,016 elements. (2) It made answers 27% shorter -- a
    10-question artifact that shrank to -10% at 71 and vanished at 198, where the run was in fact
    LONGER (11,069 chars) and scored 71.21% vs 66.67% for the previous full run on the same
    questions. Reintroducing the selector is a deliberate act that should have to delete this test.
    """
    assert not hasattr(DF, "_norm_fullcanvas_enabled"), "the DG_NORM_FULLCANVAS gate is back"
    seen = []

    def fake_sharded(norm, hidden, rows):
        seen.append(rows)
        return _FakeTensor(hidden.shape)

    monkeypatch.setattr(DF, "_sharded_rms_norm_rows", fake_sharded)
    norm = SimpleNamespace(with_scale=True, tt_weight=object(), eps=1e-6)

    for value in ("0", "1"):
        monkeypatch.setenv("DG_NORM_FULLCANVAS", value)
        _chunked_norm_forward(norm, _FakeTensor([1, 1, 256, 2816]))

    assert seen == [256, 256], "the stale flag must be inert in both directions"


# --- denoise hidden forward -----------------------------------------------------------------


def test_denoise_hidden_forward_reads_and_deallocates_lazy_prompt_sources(monkeypatch):
    calls = []
    prompt_sources = [
        (_FakeTensor([1, 1, 32, 16]), _FakeTensor([1, 1, 32, 16])),
        _FakeTensor([1, 1, 32, 16]),
    ]
    canvas = _FakeTensor([1, 1, 256, 16])
    layer_hiddens = [_FakeTensor([1, 1, 256, 16]), _FakeTensor([1, 1, 256, 16])]
    final_hidden = _FakeTensor([1, 1, 256, 16])
    model = _FakeModel(num_layers=2)
    model.norm = SimpleNamespace()

    def prompt_source(layer_idx):
        calls.append(("read", layer_idx))
        return prompt_sources[layer_idx]

    def fake_layer_forward(
        tt_model, layer_idx, hidden_states, prompt_source, attn_mask, q_rope_offset, *, canvas_rope_provider=None
    ):
        assert canvas_rope_provider is None
        calls.append(("layer", layer_idx, prompt_source, q_rope_offset))
        return layer_hiddens[layer_idx]

    monkeypatch.setattr(DF, "_denoise_layer_forward", fake_layer_forward)
    monkeypatch.setattr(DF, "_chunked_norm_forward", lambda norm, hidden_states: final_hidden)

    out = DF.denoise_hidden_forward(
        model,
        prompt_hidden_by_layer=prompt_source,
        prompt_len=32,
        canvas_hidden=canvas,
    )

    assert out is final_hidden
    assert calls == [
        ("read", 0),
        ("layer", 0, prompt_sources[0], 32),
        ("read", 1),
        ("layer", 1, prompt_sources[1], 32),
    ]
    assert prompt_sources[0][0].deallocated is True
    assert prompt_sources[0][1].deallocated is True
    assert prompt_sources[1].deallocated is True


def test_denoise_hidden_forward_deallocates_final_norm_input(monkeypatch):
    prompt = _FakeTensor([1, 1, 32, 16])
    canvas = _FakeTensor([1, 1, 256, 16])
    layer_hidden = _FakeTensor([1, 1, 256, 16])
    final_hidden = _FakeTensor([1, 1, 256, 16])
    model = _FakeModel(num_layers=1)
    model.norm = SimpleNamespace()

    def fake_layer_forward(
        tt_model, layer_idx, hidden_states, prompt_source, attn_mask, q_rope_offset, *, canvas_rope_provider=None
    ):
        assert tt_model is model
        assert layer_idx == 0
        assert hidden_states is canvas
        assert prompt_source is prompt
        assert attn_mask is None
        assert q_rope_offset == 32
        assert canvas_rope_provider is None
        return layer_hidden

    def fake_chunked_norm(norm, hidden_states):
        assert norm is model.norm
        assert hidden_states is layer_hidden
        return final_hidden

    monkeypatch.setattr(DF, "_denoise_layer_forward", fake_layer_forward)
    monkeypatch.setattr(DF, "_chunked_norm_forward", fake_chunked_norm)
    monkeypatch.setattr(DF, "_build_denoise_attn_mask_for_layer", lambda *args, **kwargs: None)

    out = DF.denoise_hidden_forward(model, prompt_hidden_by_layer=[prompt], canvas_hidden=canvas)

    assert out is final_hidden
    assert layer_hidden.deallocated is True


@pytest.mark.parametrize(("owns_result", "expect_freed"), [(False, False), (True, True)])
def test_denoise_hidden_forward_honours_prompt_source_ownership(monkeypatch, owns_result, expect_freed):
    """The per-layer ``finally`` must skip the free when the source reports owns_result False.

    This drives the REAL ``denoise_hidden_forward`` (same harness as
    ``test_denoise_hidden_forward_reads_and_deallocates_lazy_prompt_sources``) rather than
    re-implementing the predicate, so deleting the ``owns_result`` guard from the production
    ``finally`` actually fails this test. That guard is what stops the borrowed fixed-span
    prefix read from deallocating the model-owned KV cache — a device-fatal
    ``TT_FATAL: Input Tensor is not allocated`` on the next block, which CPU CI can never catch
    directly, so the guard needs real coverage here.
    """
    freed = []
    monkeypatch.setattr(DF, "_deallocate_prompt_source", lambda src: freed.append(src))

    num_layers = 2
    prompt_sources = [(_FakeTensor([1, 1, 32, 16]), _FakeTensor([1, 1, 32, 16])) for _ in range(num_layers)]
    layer_hiddens = [_FakeTensor([1, 1, 256, 16]) for _ in range(num_layers)]
    final_hidden = _FakeTensor([1, 1, 256, 16])
    model = _FakeModel(num_layers=num_layers)
    model.norm = SimpleNamespace()

    class _Reader:
        """Callable prompt source mirroring MutablePrefixKVReader's ownership contract."""

        def __init__(self, owns):
            self.owns_result = owns

        def __call__(self, layer_idx):
            return prompt_sources[layer_idx]

    monkeypatch.setattr(
        DF,
        "_denoise_layer_forward",
        lambda tt_model, layer_idx, hidden_states, prompt_source, attn_mask, q_rope_offset, *, canvas_rope_provider=None: layer_hiddens[
            layer_idx
        ],
    )
    monkeypatch.setattr(DF, "_chunked_norm_forward", lambda norm, hidden_states: final_hidden)

    out = DF.denoise_hidden_forward(
        model,
        prompt_hidden_by_layer=_Reader(owns_result),
        prompt_len=32,
        canvas_hidden=_FakeTensor([1, 1, 256, 16]),
    )

    assert out is final_hidden
    if expect_freed:
        assert freed == prompt_sources, "an owning source's per-layer prefix must be freed"
    else:
        assert freed == [], "a BORROWED prefix must never be freed -- that would free the model KV cache"


# --- router and MoE dispatch ----------------------------------------------------------------


def test_denoise_router_forward_uses_chunked_norm(monkeypatch):
    calls = []

    class _Tensor(_FakeTensor):
        def __init__(self, name, shape=[1, 1, 96, 128]):
            super().__init__(shape)
            self.name = name

    class _FakeTtnn:
        @staticmethod
        def mul(lhs, rhs):
            calls.append(("mul", lhs.name, getattr(rhs, "name", rhs)))
            return _Tensor(f"mul({lhs.name})")

        @staticmethod
        def linear(lhs, rhs):
            calls.append(("linear", lhs.name, rhs.name))
            return _Tensor("scores", [1, 1, 96, 16])

        @staticmethod
        def softmax(tensor, *, dim):
            calls.append(("softmax", tensor.name, dim))
            return _Tensor("probs", tensor.shape)

        @staticmethod
        def topk(tensor, *, k, dim):
            calls.append(("topk", tensor.name, k, dim))
            return _Tensor("topk-values", [1, 1, 96, k]), _Tensor("topk-indices", [1, 1, 96, k])

        @staticmethod
        def sum(tensor, *, dim, keepdim):
            calls.append(("sum", tensor.name, dim, keepdim))
            return _Tensor("topk-sum", [1, 1, 96, 1])

        @staticmethod
        def div(lhs, rhs):
            calls.append(("div", lhs.name, rhs.name))
            return _Tensor("topk-normalized", lhs.shape)

        @staticmethod
        def zeros_like(tensor):
            calls.append(("zeros_like", tensor.name))
            return _Tensor("zeros", tensor.shape)

        @staticmethod
        def scatter(tensor, *, dim, index, src):
            calls.append(("scatter", tensor.name, dim, index.name, src.name))
            return _Tensor("dense-routing", tensor.shape)

    def fake_chunked_norm(norm, hidden_states):
        calls.append(("chunked_norm", norm.name, hidden_states.name))
        return _Tensor("normed", hidden_states.shape)

    monkeypatch.setattr(DF, "ttnn", _FakeTtnn)
    monkeypatch.setattr(DF, "_chunked_norm_forward", fake_chunked_norm)
    router = SimpleNamespace(
        norm=SimpleNamespace(name="router-norm"),
        scale=_Tensor("scale", [1, 1, 1, 128]),
        scalar_root_size=0.125,
        proj_weight=_Tensor("proj-weight", [1, 1, 128, 16]),
        top_k=2,
        per_expert_scale=_Tensor("per-expert-scale", [1, 1, 1, 16]),
    )

    out = _denoise_router_forward(router, _Tensor("hidden"))

    assert out.name == "mul(dense-routing)"
    assert calls[0] == ("chunked_norm", "router-norm", "hidden")
    assert ("linear", "mul(mul(normed))", "proj-weight") in calls
    assert ("scatter", "zeros", -1, "topk-indices", "topk-normalized") in calls


def test_denoise_moe_is_unconditionally_the_concat_path(monkeypatch):
    """No env var may route the denoise MoE anywhere but concat-experts.

    The token-gather capacity dispatch (``DG_SPARSE_MOE``) and the dense-128 reference behind it
    (``DG_ALLOW_DENSE_MOE``) were deleted on 2026-07-29 along with the ``DG_MOE_CONCAT`` selector,
    because the gather path leaves the denoise trajectory ~100x above the halt-entropy threshold —
    no block settles and roughly two thirds of requests end degenerate. The exact combination that
    used to select the dense reference is set here: it must now be inert, not a quiet route back.
    """
    monkeypatch.setenv("DG_MOE_CONCAT", "0")
    monkeypatch.setenv("DG_SPARSE_MOE", "0")
    monkeypatch.setenv("DG_ALLOW_DENSE_MOE", "1")

    dense_routing = _FakeTensor([1, 1, 256, 128])
    monkeypatch.setattr(DF, "_denoise_router_forward", lambda router, hidden: dense_routing)
    calls = []

    def fake_concat_experts_forward(experts, hidden, routing):
        calls.append((experts, hidden, routing))
        return "concat-output"

    monkeypatch.setattr(DF, "concat_experts_forward", fake_concat_experts_forward)
    experts = object()
    moe = SimpleNamespace(router=object(), experts=experts)
    expert_input = _FakeTensor([1, 1, 256, 2816])

    assert DF._denoise_moe_forward(moe, _FakeTensor([1, 1, 256, 2816]), expert_input) == "concat-output"
    assert calls == [(experts, expert_input, dense_routing)]
    assert dense_routing.deallocated is True


def test_expert_matmuls_opt_into_blackhole_fp32_full_dst_accumulation(monkeypatch):
    from models.experimental.diffusion_gemma.tt import concat_moe

    class _Device:
        def arch(self):
            return concat_moe.ttnn.Arch.BLACKHOLE

    tensor = SimpleNamespace(device=lambda: _Device())
    fallback = object()
    captured = {}
    concat_moe._EXPERT_FP32_FULL_SYNC_CFG_CACHE.clear()
    monkeypatch.setenv("DG_SPARSE_EXPERT_FP32_FULL_SYNC", "1")

    def fake_init(arch, **kwargs):
        captured.update(arch=arch, **kwargs)
        return "accurate-config"

    monkeypatch.setattr(concat_moe.ttnn, "init_device_compute_kernel_config", fake_init)
    monkeypatch.setattr(concat_moe.ttnn, "MathFidelity", SimpleNamespace(HiFi4="hifi4"))

    assert concat_moe.expert_compute_kernel_config(tensor, fallback) == "accurate-config"
    assert captured == {
        "arch": concat_moe.ttnn.Arch.BLACKHOLE,
        "math_fidelity": "hifi4",
        "math_approx_mode": False,
        "fp32_dest_acc_en": True,
        "packer_l1_acc": False,
        "dst_full_sync_en": True,
    }

    monkeypatch.setenv("DG_SPARSE_EXPERT_FP32_FULL_SYNC", "0")
    assert concat_moe.expert_compute_kernel_config(tensor, fallback) is fallback


def test_expert_matmuls_keep_wormhole_policy(monkeypatch):
    from models.experimental.diffusion_gemma.tt import concat_moe

    wormhole = SimpleNamespace(device=lambda: SimpleNamespace(arch=lambda: concat_moe.ttnn.Arch.WORMHOLE_B0))
    fallback = object()
    monkeypatch.setenv("DG_SPARSE_EXPERT_FP32_FULL_SYNC", "1")

    assert concat_moe.expert_compute_kernel_config(wormhole, fallback) is fallback


# --- logits adapter -------------------------------------------------------------------------


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


def test_denoise_logits_adapter_reports_retained_logits_ownership():
    def _fake_logits_from_tokens(tt_model, **kwargs):
        del tt_model, kwargs
        return "logits"

    adapter = DenoiseLogitsAdapter(
        object(),
        prompt_hidden_by_layer=["prompt"],
        logits_from_tokens=_fake_logits_from_tokens,
    )

    logits = adapter("canvas", 0)

    assert adapter.owns_logits(logits)
    assert not adapter.owns_logits("other")


def test_denoise_logits_adapter_temperatures_previous_logits_for_self_conditioning():
    calls = []

    def fake_logits_from_tokens(tt_model, **kwargs):
        del tt_model
        calls.append(kwargs["self_conditioning_temperature"])
        return _FakeTensor([1, 1, 256, 32])

    adapter = DenoiseLogitsAdapter(
        object(),
        prompt_hidden_by_layer=["prompt"],
        self_conditioning=object(),
        self_conditioning_embedding_weight="embedding",
        max_denoise_steps=8,
        temperature_start=0.8,
        temperature_end=0.4,
        logits_from_tokens=fake_logits_from_tokens,
    )

    adapter("canvas", 0)
    adapter("canvas", 1)
    adapter("canvas", 2)

    assert calls == [1.0, 0.8, 0.75]


def test_embed_canvas_tokens_rejects_batch_greater_than_one(expect_error):
    with expect_error(ValueError, match="batch=1"):
        embed_canvas_tokens(object(), _FakeTensor([2, 256]))


# --- prompt KV reads ------------------------------------------------------------------------


def test_read_prompt_kv_cache_slice_uses_dram_slice_outputs(monkeypatch):
    calls = []

    class _FakeCache:
        def __init__(self, name):
            self.name = name
            self.shape = [1, 2, 128, 16]

    class _FakeTtnn:
        TILE_SIZE = 32
        DRAM_MEMORY_CONFIG = "dram"

        @staticmethod
        def slice(cache, starts, ends, *, memory_config=None):
            calls.append((cache.name, starts, ends, memory_config))
            return f"slice-{cache.name}"

    monkeypatch.setattr(DF, "ttnn", _FakeTtnn)

    assert read_prompt_kv_cache_slice(
        (_FakeCache("k-cache"), _FakeCache("v-cache")), prompt_len=64, seq_len_start=32
    ) == (
        "slice-k-cache",
        "slice-v-cache",
    )
    assert calls == [
        ("k-cache", [0, 0, 32, 0], [1, 2, 96, 16], "dram"),
        ("v-cache", [0, 0, 32, 0], [1, 2, 96, 16], "dram"),
    ]


def test_full_span_read_clones_by_default_and_borrows_when_asked(monkeypatch):
    """The full-span read must never hand back an unowned alias unless asked to."""
    cloned = []

    class _FakeCache:
        def __init__(self, name):
            self.name = name
            self.shape = [1, 2, 128, 16]

    class _FakeTtnn:
        TILE_SIZE = 32
        DRAM_MEMORY_CONFIG = "dram"

        @staticmethod
        def clone(cache):
            cloned.append(cache.name)
            return f"clone-{cache.name}"

        @staticmethod
        def slice(cache, starts, ends, *, memory_config=None):
            raise AssertionError("a full-span read must not slice (it would alias the cache)")

    monkeypatch.setattr(DF, "ttnn", _FakeTtnn)
    k, v = _FakeCache("k-cache"), _FakeCache("v-cache")

    # Default: an owned clone, exactly as before.
    assert read_prompt_kv_cache_slice((k, v), prompt_len=128) == ("clone-k-cache", "clone-v-cache")
    assert cloned == ["k-cache", "v-cache"]

    # Opted in: the cache tensors themselves, no copy.
    cloned.clear()
    assert read_prompt_kv_cache_slice((k, v), prompt_len=128, borrow_full_span=True) == (k, v)
    assert cloned == []


@pytest.mark.parametrize(
    ("seq_len_start", "prompt_len", "capacity", "expected"),
    [
        pytest.param(0, 1024, 1024, ((0, 1024),), id="whole-window"),
        pytest.param(256, 512, 1024, ((256, 768),), id="one-segment"),
        pytest.param(768, 512, 1024, ((768, 1024), (0, 256)), id="wrap"),
        pytest.param(2048, 1024, 1024, ((0, 1024),), id="absolute-position-wrap"),
    ],
)
def test_hybrid_cache_sequence_segments(seq_len_start, prompt_len, capacity, expected):
    assert (
        DF.hybrid_cache_sequence_segments(
            seq_len_start=seq_len_start,
            prompt_len=prompt_len,
            capacity=capacity,
        )
        == expected
    )


def test_hybrid_cache_sequence_segments_rejects_reads_larger_than_window(expect_error):
    with expect_error(ValueError, match="exceeds physical capacity"):
        DF.hybrid_cache_sequence_segments(seq_len_start=0, prompt_len=1056, capacity=1024)


def test_reader_owns_result_is_true_unless_span_covers_whole_cache():
    # Borrowing not requested -> always owned.
    reader = DF.MutablePrefixKVReader(_kv_cache_model([128] * 3), prompt_len=64)
    reader.set_read_span(128)
    assert reader.owns_result is True

    # Requested and the fixed span covers the whole cache -> borrowed.
    reader = DF.MutablePrefixKVReader(_kv_cache_model([128] * 3), prompt_len=64, borrow_full_span=True)
    reader.set_read_span(128)
    assert reader.owns_result is False

    # Requested but the span is a strict prefix -> the slice is an independent copy, so owned.
    reader = DF.MutablePrefixKVReader(_kv_cache_model([128] * 3), prompt_len=64, borrow_full_span=True)
    reader.set_read_span(64)
    assert reader.owns_result is True

    # Non-uniform cache spans -> refuse to borrow (some layer would be a partial slice).
    reader = DF.MutablePrefixKVReader(_kv_cache_model([128, 128, 256]), prompt_len=64, borrow_full_span=True)
    reader.set_read_span(128)
    assert reader.owns_result is True

    # A nonzero start offset is a partial read by definition -> owned.
    reader = DF.MutablePrefixKVReader(
        _kv_cache_model([128] * 3), prompt_len=64, seq_len_start=32, borrow_full_span=True
    )
    reader.set_read_span(128)
    assert reader.owns_result is True


def test_hybrid_reader_borrows_full_layer_view_but_not_uncached_layers():
    full_view = (_FakeTensor([1, 1, 4096, 16]), _FakeTensor([1, 1, 4096, 16]))
    model = SimpleNamespace(
        _dg_model_owned_hybrid_kv=True,
        _dg_hybrid_sliding_layers=frozenset({0}),
        _dg_hybrid_full_cache_views={1: full_view},
    )
    reader = object.__new__(DF.MutablePrefixKVReader)
    reader.borrow_full_span = True
    reader.seq_len_start = 0
    reader.prompt_len = 64
    reader.read_span = 4096
    reader.tt_model = model
    reader._window_bufs = {0: ("k-window", "v-window")}

    assert reader.owns_result_for(0) is False
    assert reader.owns_result_for(1) is False
    assert reader.owns_result_for(2) is True


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


def test_read_prompt_kv_cache_by_layer_rejects_cache_layer_mismatch(expect_error):
    model = _FakeModel(num_layers=2)
    model.tt_kv_cache = ["cache-0"]

    with expect_error(ValueError, match="tt_kv_cache has 1 layers"):
        read_prompt_kv_cache_by_layer(model, prompt_len=64, read_fn=lambda *args, **kwargs: None)


# --- adapter builders -----------------------------------------------------------------------


def test_make_denoise_logits_adapter_from_kv_cache_reads_prompt_kv_lazily():
    calls = {"read": []}
    model = _FakeModel(num_layers=2)

    def fake_read(tt_model, *, prompt_len, seq_len_start=0, layer_idx=None):
        calls["read"].append((tt_model, prompt_len, seq_len_start, layer_idx))
        return (f"k{layer_idx}", f"v{layer_idx}")

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
    assert calls["read"] == []
    tt_model, kwargs = calls["adapter"]
    assert tt_model is model
    prompt_source = kwargs["prompt_hidden_by_layer"]
    assert callable(prompt_source)
    assert prompt_source(1) == ("k1", "v1")
    assert calls["read"] == [(model, 64, 32, 1)]
    prompt_source.set_prompt_len(320)
    assert prompt_source(0) == ("k0", "v0")
    assert calls["read"][-1] == (model, 320, 32, 0)
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
        # The prefill pad span rides down next to the PADDED prompt_len
        # (DG_DENOISE_HIDE_PREFILL_PADS); None here because this builder is called without the
        # unpadded prompt_tokens the true length would come from.
        "true_prompt_len": None,
        "seq_len_start": 32,
        "self_conditioning": "self-conditioning",
        "self_conditioning_embedding_weight": "embedding-tt",
        "self_conditioning_compute_kernel_config": "kernel",
        "q_rope_offset": 576,
        "max_denoise_steps": None,
        "temperature_start": 0.8,
        "temperature_end": 0.4,
    }


def test_make_denoise_logits_adapter_from_checkpoint_state_defaults_fp32_softmax_kernel():
    calls = {}
    model = _FakeModel()

    def fake_adapter_builder(tt_model, **kwargs):
        calls["adapter"] = kwargs
        return "adapter"

    out = make_denoise_logits_adapter_from_checkpoint_state(
        model,
        prompt_len=64,
        self_conditioning_state={"weights": "state"},
        embedding_weight="embedding-weight",
        hidden_size=8,
        intermediate_size=6,
        self_conditioning_builder=lambda *args, **kwargs: "self-conditioning",
        embedding_weight_builder=lambda *args, **kwargs: "embedding-tt",
        adapter_builder=fake_adapter_builder,
        default_compute_kernel_config_fn=lambda: "default-fp32-kernel",
    )

    assert out == "adapter"
    assert calls["adapter"]["self_conditioning_compute_kernel_config"] == "default-fp32-kernel"


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


def test_make_denoise_logits_adapter_from_remapped_state_rejects_missing_embedding(expect_error):
    with expect_error(ValueError, match="missing tied embedding weight"):
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
            "true_prompt_len": None,
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


# --- DG_SKIP ablation gate ------------------------------------------------------------------


def test_dg_skip_rejects_an_unknown_component(monkeypatch, expect_error):
    """A typo must abort, not silently measure the unablated step.

    An ignored token is the worst failure mode for a measurement tool: ``DG_SKIP=moe1`` would run
    the full MoE and the harness would report the number as an ablation, which looks plausible.
    """
    monkeypatch.setenv("DG_SKIP", "moe1")
    with expect_error(ValueError, match="DG_SKIP has unknown component"):
        DF._skip_components()


def test_every_dg_skip_token_has_a_consumer():
    """An advertised-but-unwired token is worse than no token at all.

    ``_SKIP_TOKENS`` used to list ``sc``/``cattn``/``cshared``/``cmoe``; no code ever read them, so
    ``DG_SKIP=cattn`` validated clean, ran the full commit body, and the harness reported the number
    as an ablation. The old test only checked the set against itself, which cannot catch that. Pin the
    seam instead: every accepted token must be branched on somewhere.
    """
    source = pathlib.Path(DF.__file__).read_text()
    unwired = [token for token in sorted(DF._SKIP_TOKENS) if f'"{token}" in skip' not in source]
    assert not unwired, f"DG_SKIP tokens accepted but never consumed: {unwired}"


# --- sliding-window reveal mask geometry (#51080 item 3) ------------------------------------
#
# HF's sliding layers retain only the last ``sliding_window - 1`` committed tokens, so TT's
# all-attend denoise used to attend keys HF does not have on 25 of 30 layers. Enforcing the
# retention gives each layer TYPE its own reveal-mask content while keeping ONE shape, so every
# captured trace stays valid. The regime split is what makes this gateable:
#
# * ``prompt_len <= sliding_window - 1`` — the window cannot bind, so the sliding mask is
#   IDENTICAL to the full mask. Enabling the flag there is bit-exact.
# * ``prompt_len > sliding_window - 1`` — the masks differ; that is the decision-changing regime
#   whose gate is a decision-agreement run against fp32 HF.

W = 8
CANVAS = 4
P_MAX = 32


def _attend(mask):
    return mask == 0


def _reveal(prompt_len, *, layer_type, enforce):
    return build_canvas_reveal_denoise_mask(
        prompt_len,
        CANVAS,
        P_MAX,
        layer_type=layer_type,
        sliding_window=W,
        enforce_sliding_window=enforce,
    )


def test_flag_defaults_on_and_zero_opts_out(monkeypatch):
    """Gated by the GPQA decision-agreement run, so retention is now the default.

    56 of 64 shipped-config collapses were at or after the block whose committed prefix crosses
    W-1; below that the mask is bit-identical, so the flip only changes the regime that was wrong.
    """
    monkeypatch.delenv("DG_DENOISE_SLIDING_WINDOW", raising=False)
    assert DF.denoise_sliding_window_enabled() is True
    monkeypatch.setenv("DG_DENOISE_SLIDING_WINDOW", "0")
    assert DF.denoise_sliding_window_enabled() is False, "the maskless path must stay reachable"
    monkeypatch.setenv("DG_DENOISE_SLIDING_WINDOW", "1")
    assert DF.denoise_sliding_window_enabled() is True


@pytest.mark.parametrize("prompt_len", [0, W // 2, W - 1])
def test_below_the_window_the_sliding_mask_equals_the_full_mask(prompt_len):
    """Bit-exact regime: nothing has been evicted yet, so enforcing changes nothing."""
    full = _reveal(prompt_len, layer_type="full_attention", enforce=False)
    sliding = _reveal(prompt_len, layer_type="sliding_attention", enforce=True)
    assert torch.equal(full, sliding), f"prompt_len={prompt_len} must be unchanged by the window"


@pytest.mark.parametrize("prompt_len", [W, W + 1, 3 * W])
def test_above_the_window_only_the_retained_committed_tail_is_attended(prompt_len):
    sliding = _attend(_reveal(prompt_len, layer_type="sliding_attention", enforce=True))
    full = _attend(_reveal(prompt_len, layer_type="full_attention", enforce=False))
    assert not torch.equal(sliding, full), "the window must bind above W-1"

    keep_from = prompt_len - (W - 1)
    for j in range(P_MAX):
        expected = keep_from <= j < prompt_len
        assert bool(sliding[0, j]) is expected, f"prefix col {j} (prompt_len={prompt_len})"
    # Canvas is always fully visible, and there is no query dependence.
    assert bool(sliding[:, P_MAX:].all())
    for row in range(1, CANVAS):
        assert torch.equal(sliding[row], sliding[0])


def test_full_attention_layers_are_unaffected_by_the_window():
    prompt_len = 3 * W
    a = _reveal(prompt_len, layer_type="full_attention", enforce=True)
    b = _reveal(prompt_len, layer_type="full_attention", enforce=False)
    assert torch.equal(a, b), "enforce_sliding_window must be inert on full_attention layers"


# --- reveal-mask buffers, per layer type ----------------------------------------------------


class _FakeBuf:
    def __init__(self, tag):
        self.tag = tag
        self.freed = False

    def deallocate(self, force=True):
        self.freed = True


def _adapter(layer_types, *, enforce):
    """Minimal adapter shell exercising only the reveal-mask buffer machinery."""
    adapter = object.__new__(DF.DenoiseLogitsAdapter)
    adapter.tt_model = SimpleNamespace(
        layers=[
            SimpleNamespace(self_attn=SimpleNamespace(config=SimpleNamespace(sliding_window=W))) for _ in layer_types
        ],
        hf_config=SimpleNamespace(layer_types=list(layer_types), sliding_window=W),
        mesh_device=None,
    )
    adapter._reveal_canvas_len = CANVAS
    adapter._reveal_p_max = P_MAX
    adapter._reveal_enforce_window = enforce
    adapter._reveal_mask_bufs = {}
    adapter._reveal_mask_buf = None
    adapter.use_reveal_mask = False
    return adapter


DG_LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention"]


def test_two_masks_when_enabled_and_dispatch_is_per_layer_type():
    adapter = _adapter(DG_LAYER_TYPES, enforce=True)
    assert set(adapter._reveal_mask_layer_types()) == {"sliding_attention", "full_attention"}

    adapter._reveal_mask_bufs = {"sliding_attention": _FakeBuf("slide"), "full_attention": _FakeBuf("full")}
    for layer_idx, layer_type in enumerate(DG_LAYER_TYPES):
        expected = "slide" if layer_type == "sliding_attention" else "full"
        assert adapter._reveal_mask_provider(layer_idx).tag == expected, f"layer {layer_idx}"


def test_one_mask_shared_by_every_layer_when_the_window_is_disabled():
    adapter = _adapter(DG_LAYER_TYPES, enforce=False)
    assert adapter._reveal_mask_layer_types() == ("full_attention",)

    only = _FakeBuf("full")
    adapter._reveal_mask_bufs = {"full_attention": only}
    for layer_idx in range(len(DG_LAYER_TYPES)):
        assert adapter._reveal_mask_provider(layer_idx) is only


def test_per_block_update_rebuilds_every_mask_at_the_new_committed_len(monkeypatch):
    """Both buffers must be refreshed per block, since the retained window slides with the prefix.

    A frozen sliding buffer would also suppress the late-block collapse, by over-restricting
    attention rather than matching HF, and an end-to-end run could not distinguish the two.
    """
    adapter = _adapter(DG_LAYER_TYPES, enforce=True)
    adapter.use_reveal_mask = True
    adapter._reveal_mask_bufs = {
        "sliding_attention": _FakeBuf("slide"),
        "full_attention": _FakeBuf("full"),
    }

    built = []

    def recording_build(self, prompt_len, layer_type="full_attention"):
        built.append((int(prompt_len), layer_type))
        return _FakeBuf(f"fresh-{layer_type}")

    monkeypatch.setattr(DF.DenoiseLogitsAdapter, "_build_reveal_mask_device", recording_build)
    monkeypatch.setattr(DF.ttnn, "copy", lambda fresh, buf: None)

    # update_reveal_mask_buffer enforces the product's 32-tile alignment on the committed length,
    # so this uses real tile multiples rather than the toy CANVAS used for mask geometry above.
    committed = [32 * n for n in (1, 2, 3, 4)]  # one tile-aligned commit per block
    for prompt_len in committed:
        adapter.update_reveal_mask_buffer(prompt_len)

    for prompt_len in committed:
        for layer_type in ("sliding_attention", "full_attention"):
            assert (prompt_len, layer_type) in built, f"{layer_type} not rebuilt at prompt_len={prompt_len}"
    assert len(built) == 2 * len(committed), f"expected one rebuild per buffer per block, got {built}"


def test_release_frees_every_mask_and_clears_state():
    adapter = _adapter(DG_LAYER_TYPES, enforce=True)
    bufs = {"sliding_attention": _FakeBuf("slide"), "full_attention": _FakeBuf("full")}
    adapter._reveal_mask_bufs = dict(bufs)
    adapter.use_reveal_mask = True

    adapter.release_reveal_mask_buffers()

    assert all(b.freed for b in bufs.values())
    assert adapter._reveal_mask_bufs == {}
    assert adapter._reveal_mask_buf is None
    assert adapter.use_reveal_mask is False


def test_sliding_layer_needs_mask_threshold_is_window_minus_one():
    """The old threshold came from the staircase and included canvas_len; it must not."""
    assert DF._sliding_layer_needs_denoise_mask(W - 1, CANVAS, W) is False
    assert DF._sliding_layer_needs_denoise_mask(W, CANVAS, W) is True
    # canvas_len must not influence the predicate
    for canvas in (1, 256, 4096):
        assert DF._sliding_layer_needs_denoise_mask(W - 1, canvas, W) is False


# --- bounded per-layer sliding span (#51080 item 3, perf half) ------------------------------

# Real magnitudes, not the toy W/P_MAX above: at W=8, P_MAX=32 the span rounds up to a whole tile
# and hits the "span >= p_max -> nothing to save" fallback, so a toy geometry cannot observe a
# bounded span at all.
W_REAL, P_MAX_REAL, CANVAS_REAL, PROMPT_REAL = 1024, 4096, 256, 2048


def _drive_fixed_reveal(monkeypatch, *, retention):
    """Run the REAL _prepare_fixed_reveal and capture the sliding_span it prepares.

    DG_DENOISE_SLIDING_SPAN is gone; the bounded read is now gated on the retention mask at its one
    decision point inside this function. A test on a flag reader could not have caught that gate
    regressing, and this module previously only ever monkeypatched the mask selector -- which is how a
    NotImplementedError on a live flag pair once survived in the sibling builder. So drive the real
    thing.
    """
    from models.experimental.diffusion_gemma.tt import traced_denoise as TD

    captured = {}

    class _Reader:
        owns_result = True
        borrow_full_span = False

        def set_read_span(self, p_max):
            captured["read_span"] = p_max

        def prepare_window_buffers(self, layers):
            captured["window_layers"] = dict(layers)

        def refresh_windows(self, prompt_len):
            captured["refreshed"] = prompt_len

    layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    adapter = SimpleNamespace(
        prompt_len=PROMPT_REAL,
        prompt_hidden_by_layer=_Reader(),
        tt_model=SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=SimpleNamespace(config=SimpleNamespace(sliding_window=W_REAL)))
                for _ in layer_types
            ],
            hf_config=SimpleNamespace(layer_types=list(layer_types), sliding_window=W_REAL),
            mesh_device=None,
        ),
        use_reveal_mask=True,
        _reveal_mask_bufs={},
        prepare_reveal_mask_buffers=lambda **kw: captured.update(kw),
        update_reveal_mask_buffer=lambda prompt_len: None,
    )

    monkeypatch.setenv("DG_DENOISE_SLIDING_WINDOW", "1" if retention else "0")
    monkeypatch.setattr(TD, "_resolve_reveal_pmax", lambda a: P_MAX_REAL)
    monkeypatch.setattr(TD, "prefix_borrow_enabled", lambda: False)
    TD._prepare_fixed_reveal(adapter, canvas_len=CANVAS_REAL)
    return captured


def test_bounded_read_is_on_by_default_when_retention_is_enforced(monkeypatch):
    """No flag any more: retention on (the default) is what engages the bounded read."""
    cap = _drive_fixed_reveal(monkeypatch, retention=True)
    assert cap["enforce_window"] is True
    assert cap["sliding_span"] == W_REAL, "sliding layers must read the bounded window"
    assert set(cap["window_layers"]) == {0, 1, 2, 3, 4}, "only the sliding layers are bounded"
    assert all(v == W_REAL for v in cap["window_layers"].values())
    assert cap["refreshed"] == PROMPT_REAL


def test_bounded_read_does_not_engage_without_the_retention_mask(monkeypatch):
    """A bounded read without the retention mask would CHANGE visibility, not implement it.

    This is the property the deleted DG_DENOISE_SLIDING_SPAN gate used to carry, and it is the one
    thing about the bounded read that is NOT bit-identical -- so it has to keep being tested.
    """
    cap = _drive_fixed_reveal(monkeypatch, retention=False)
    assert cap["enforce_window"] is False
    assert cap["sliding_span"] is None, "the bounded read must stay off without retention"
    assert "window_layers" not in cap, "no block-resident window buffers may be allocated"


def test_read_span_is_tile_aligned_and_capped_by_pmax():
    assert DF.sliding_read_span(1024, 4096) == 1024
    assert DF.sliding_read_span(1000, 4096) == 1024  # rounded up to a tile
    assert DF.sliding_read_span(1024, 512) == 512  # never exceed the reveal span
    assert DF.sliding_read_span(8, 4096) % 32 == 0


def test_read_offset_tracks_the_committed_tail_and_stays_tile_aligned():
    span, p_max = 1024, 4096
    assert DF.sliding_read_offset(0, span, p_max) == 0
    assert DF.sliding_read_offset(1024, span, p_max) == 0
    assert DF.sliding_read_offset(1056, span, p_max) == 32
    # Clamped so the window never runs past the reveal span.
    assert DF.sliding_read_offset(4096, span, p_max) == p_max - span
    for P in range(0, 4096, 32):
        assert DF.sliding_read_offset(P, span, p_max) % 32 == 0


@pytest.mark.parametrize("prompt_len", [0, W, 2 * W, 3 * W, P_MAX])  # all <= P_MAX
def test_bounded_window_mask_matches_the_full_span_mask_on_shared_columns(prompt_len):
    """The bounded read + its mask must expose EXACTLY the keys the full-span path exposes.

    This is the equivalence that makes the perf half safe once the fidelity half is in: the
    bounded read simply omits columns the full-span mask had already set to NEG.
    """
    from models.experimental.diffusion_gemma.reference.attention_mask import (
        build_canvas_reveal_denoise_window_mask,
    )

    span = W  # tile-aligned stand-in for 1024
    lo = max(0, prompt_len - span)
    full = _attend(_reveal(prompt_len, layer_type="sliding_attention", enforce=True))
    bounded = build_canvas_reveal_denoise_window_mask(prompt_len, CANVAS, span, lo, sliding_window=W) == 0

    # Absolute positions visible under each construction must be identical.
    full_visible = {j for j in range(P_MAX) if bool(full[0, j])}
    bounded_visible = {lo + r for r in range(span) if bool(bounded[0, r])}
    assert bounded_visible == full_visible, f"prompt_len={prompt_len}"
    # Canvas fully visible, no query dependence.
    assert bool(bounded[:, span:].all())
    for row in range(1, CANVAS):
        assert torch.equal(bounded[row], bounded[0])


def test_bounded_mask_is_prompt_independent_in_steady_state():
    """Once the window binds, the bounded mask stops changing between blocks."""
    from models.experimental.diffusion_gemma.reference.attention_mask import (
        build_canvas_reveal_denoise_window_mask,
    )

    span = W
    masks = []
    for prompt_len in (3 * W, 4 * W, 5 * W):
        lo = max(0, prompt_len - span)
        masks.append(build_canvas_reveal_denoise_window_mask(prompt_len, CANVAS, span, lo, sliding_window=W))
    assert all(torch.equal(masks[0], m) for m in masks[1:])


def test_bounded_window_refuses_a_nonzero_prefix_base(expect_error):
    """The bounded sliding read must FAIL LOUD on prefill-from-non-zero, not read the wrong rows.

    prepare_window_buffers/refresh_windows derive their absolute offset from prompt_len alone and pass
    it straight through as seq_len_start; they do NOT add self.seq_len_start the way the unbounded read
    at read_prompt_kv_cache_by_layer does. Every production construction passes 0 today, so this is
    dormant -- but prefill-from-non-zero is the declared keystone of the vLLM-native plan, and the
    bounded read went from opt-in to DEFAULT on 2026-07-29, so a silent 25-of-30-layers misread has to
    be a raise. Delete this test when the offset folds seq_len_start in properly.
    """
    reader = DF.MutablePrefixKVReader(_kv_cache_model([4096], head_dim=8), prompt_len=2048, seq_len_start=32)
    reader.set_read_span(4096)
    with expect_error(NotImplementedError, match="non-zero prefix base"):
        reader.prepare_window_buffers({0: 1024})


def test_bounded_window_is_a_noop_when_no_layers_are_listed():
    """An empty window_layers map must not trip the non-zero-base guard -- there is no bounded read."""
    reader = DF.MutablePrefixKVReader(_kv_cache_model([4096], head_dim=8), prompt_len=2048, seq_len_start=32)
    reader.set_read_span(4096)
    reader.prepare_window_buffers({})  # must not raise
    assert reader.window_layers == {}


def test_per_layer_ownership_never_frees_a_window_buffer():
    """A windowed layer hands back a persistent buffer; freeing it corrupts later blocks."""
    reader = object.__new__(DF.MutablePrefixKVReader)
    reader.borrow_full_span = False  # full layers would be OWNED clones
    reader.seq_len_start = 0
    reader.prompt_len = 64
    reader.read_span = P_MAX
    reader.tt_model = None
    reader._window_bufs = {0: ("k", "v"), 2: ("k", "v")}
    reader.window_layers = {0: W, 2: W}

    # Windowed layers: never owned, even though the reader's global flag says owned.
    assert reader.owns_result_for(0) is False
    assert reader.owns_result_for(2) is False
    # Non-windowed layer keeps the global contract (here: owned, because borrow is off).
    assert reader.owns_result_for(1) is True
    assert DF._prompt_source_is_owned(reader, 0) is False
    assert DF._prompt_source_is_owned(reader, 1) is True


def test_prompt_source_is_owned_falls_back_for_plain_sources():
    """Sources without per-layer resolution keep the historic contract."""
    assert DF._prompt_source_is_owned(lambda i: None, 3) is True
    assert DF._prompt_source_is_owned(SimpleNamespace(owns_result=False), 3) is False
    assert DF._prompt_source_is_owned(SimpleNamespace(owns_result=True), 3) is True


# --- canvas-tail workspace (#51080 item 4) --------------------------------------------------


def test_bounded_span_and_hidden_pads_compose_in_the_mask_selector(monkeypatch):
    """The flag combination that used to raise NotImplementedError must now build a mask.

    This calls the REAL ``_build_reveal_mask_device``. Every other test in this file monkeypatches
    it, which is exactly how a NotImplementedError on a live flag pair survived unnoticed: nothing
    ever executed the selector. Asserts the pad span reaches the bounded builder, since the bounded
    key axis carries absolute positions and needs no remapping.
    """
    adapter = _adapter(DG_LAYER_TYPES, enforce=True)
    adapter._reveal_sliding_span = P_MAX  # lo == 0, i.e. tile-aligned for this toy geometry
    adapter._reveal_pad_span = (P_MAX - 6, P_MAX)

    seen = {}

    def fake_window(mesh_device, **kw):
        seen.update(kw)
        return "window-mask"

    monkeypatch.setattr(DF, "build_device_canvas_reveal_window_mask", fake_window)

    out = adapter._build_reveal_mask_device(P_MAX, layer_type="sliding_attention")

    assert out == "window-mask", "the bounded branch must be taken, not the full-span one"
    assert seen["span"] == P_MAX
    assert seen["hidden_prefix_span"] == (P_MAX - 6, P_MAX), "the pad span must reach the bounded builder"


def test_full_attention_layers_still_get_the_absolute_pad_span(monkeypatch):
    """The 5 full-attention layers read the whole p_max prefix, so they need the pads hidden forever
    -- the bounded span's self-retiring behaviour must not leak into them."""
    adapter = _adapter(DG_LAYER_TYPES, enforce=True)
    adapter._reveal_sliding_span = P_MAX
    adapter._reveal_pad_span = (P_MAX - 6, P_MAX)

    seen = {}
    monkeypatch.setattr(DF, "build_device_canvas_reveal_mask", lambda mesh_device, **kw: seen.update(kw) or "full-mask")

    assert adapter._build_reveal_mask_device(P_MAX, layer_type="full_attention") == "full-mask"
    assert seen["hidden_prefix_span"] == (P_MAX - 6, P_MAX)
    assert "span" not in seen, "a full-attention layer must not take the bounded read"
