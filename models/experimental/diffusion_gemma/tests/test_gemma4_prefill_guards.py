# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Guards for the DiffusionGemma-local denoise attention + model.

These RoPE-offset / cache-slice guards used to live in the shared Gemma4 prefill
op; they now belong to DiffusionGemma so the backbone stays untouched. The
denoise pass is single-user, so ``validate_q_rope_offset`` only enforces tile
alignment (no batched-prefill case).
"""

import torch
from contextlib import contextmanager
from types import SimpleNamespace

from models.experimental.diffusion_gemma.tt import diffusion_attention as DA
from models.experimental.diffusion_gemma.tt.diffusion_attention import (
    _apply_rope_chunked,
    _sdpa_q_chunked,
    _slice_rope_cache,
    validate_q_rope_offset,
)
from models.experimental.diffusion_gemma.tt.model import DiffusionGemma4Model


def test_q_rope_offset_must_be_tile_aligned(expect_error):
    validate_q_rope_offset(32)
    validate_q_rope_offset(0)
    with expect_error(ValueError, "q_rope_offset must be a multiple of 32"):
        validate_q_rope_offset(1)
    with expect_error(ValueError, "RoPE cache start must be a multiple of 32"):
        _slice_rope_cache(None, 1, 32)


def test_get_rope_mats_reaches_256k_and_rejects_overflow(expect_error):
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

    cos, sin = DiffusionGemma4Model._get_rope_mats(model, 0, seq_len=cache_len)
    assert cos.shape[-2] == cache_len
    assert sin.shape[-2] == cache_len
    with expect_error(ValueError, "requested RoPE seq_len 262176 exceeds cache length 262144"):
        DiffusionGemma4Model._get_rope_mats(model, 0, seq_len=cache_len + 32)


def test_model_call_establishes_diffusion_activation_context(monkeypatch):
    from models.demos.gemma4.tt.model import Gemma4Model
    from models.experimental.diffusion_gemma.tt import prefill_moe

    events = []

    @contextmanager
    def fake_context(model):
        events.append(("enter", model))
        try:
            yield
        finally:
            events.append(("exit", model))

    monkeypatch.setattr(prefill_moe, "use_tuned_prefill_moe", fake_context)
    monkeypatch.setattr(Gemma4Model, "__call__", lambda self, *args, **kwargs: (args, kwargs))
    model = object.__new__(DiffusionGemma4Model)

    assert model("hidden", is_decode=False) == (("hidden",), {"is_decode": False})
    assert events == [("enter", model), ("exit", model)]


def test_slice_rope_cache_rejects_overflow(expect_error):
    cache = SimpleNamespace(shape=[1, 1, 262144, 8])
    with expect_error(ValueError, r"RoPE cache slice \[262144, 262176\) exceeds cache length 262144"):
        _slice_rope_cache(cache, 262144, 32)


def test_apply_rope_chunked_slices_tensor_and_cache(monkeypatch):
    calls = []
    monkeypatch.setenv("DG_ROPE_FULLCANVAS", "0")

    class _Tensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape
            self.deallocated = False

        def deallocate(self, force):
            self.deallocated = force

    class _FakeTtnn:
        DRAM_MEMORY_CONFIG = "dram"

        @staticmethod
        def slice(tensor, starts, ends, *, memory_config=None):
            calls.append(("slice", tensor.name, starts, ends, memory_config))
            return _Tensor(
                f"{tensor.name}[h{starts[1]}:{ends[1]},s{starts[2]}:{ends[2]},d{starts[3]}:{ends[3]}]",
                [ends[idx] - starts[idx] for idx in range(4)],
            )

        @staticmethod
        def concat(tensors, *, dim, memory_config):
            calls.append(("concat", [tensor.name for tensor in tensors], dim, memory_config))
            shape = list(tensors[0].shape)
            shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
            return _Tensor("rope-out", shape)

        @staticmethod
        def mul(lhs, rhs):
            calls.append(("mul", lhs.name, getattr(rhs, "name", rhs)))
            return _Tensor(f"mul({lhs.name})", lhs.shape)

        @staticmethod
        def add(lhs, rhs):
            calls.append(("add", lhs.name, rhs.name))
            return _Tensor(f"add({lhs.name})", lhs.shape)

    monkeypatch.setattr(DA, "ttnn", _FakeTtnn)

    out = _apply_rope_chunked(
        _Tensor("q", [1, 2, 64, 256]),
        _Tensor("cos", [1, 1, 512, 256]),
        _Tensor("sin", [1, 1, 512, 256]),
        start_offset=32,
    )

    assert out.shape == [1, 2, 64, 256]
    assert [call for call in calls if call[0] == "add"] == [
        ("add", "mul(q[h0:1,s0:32,d0:256])", "mul(rope-out)"),
        ("add", "mul(q[h1:2,s0:32,d0:256])", "mul(rope-out)"),
        ("add", "mul(q[h0:1,s32:64,d0:256])", "mul(rope-out)"),
        ("add", "mul(q[h1:2,s32:64,d0:256])", "mul(rope-out)"),
    ]
    cache_slices = [call for call in calls if call[0] == "slice" and call[1] in {"cos", "sin"}]
    assert {tuple(call[2]) for call in cache_slices} == {(0, 0, 32, 0), (0, 0, 64, 0)}
    assert calls[-1] == ("concat", ["rope-out", "rope-out"], 2, "dram")


def test_apply_rope_chunked_keeps_single_sequence_chunk_allocated(monkeypatch):
    calls = []
    monkeypatch.setenv("DG_ROPE_FULLCANVAS", "0")

    class _Tensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape
            self.deallocated = False

        def deallocate(self, force):
            self.deallocated = force

    class _FakeTtnn:
        DRAM_MEMORY_CONFIG = "dram"

        @staticmethod
        def slice(tensor, starts, ends, *, memory_config=None):
            calls.append(("slice", tensor.name, starts, ends, memory_config))
            return _Tensor(
                f"{tensor.name}[h{starts[1]}:{ends[1]},s{starts[2]}:{ends[2]},d{starts[3]}:{ends[3]}]",
                [ends[idx] - starts[idx] for idx in range(4)],
            )

        @staticmethod
        def concat(tensors, *, dim, memory_config):
            calls.append(("concat", [tensor.name for tensor in tensors], dim, memory_config))
            shape = list(tensors[0].shape)
            shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
            return _Tensor(f"concat-dim{dim}", shape)

        @staticmethod
        def mul(lhs, rhs):
            return _Tensor(f"mul({lhs.name})", lhs.shape)

        @staticmethod
        def add(lhs, rhs):
            return _Tensor(f"add({lhs.name})", lhs.shape)

    monkeypatch.setattr(DA, "ttnn", _FakeTtnn)
    source = _Tensor("k", [1, 2, 32, 256])

    out = _apply_rope_chunked(
        source, _Tensor("cos", [1, 1, 512, 256]), _Tensor("sin", [1, 1, 512, 256]), start_offset=32
    )

    assert out.shape == [1, 2, 32, 256]
    assert out.name == "concat-dim1"
    assert out.deallocated is False
    assert source.deallocated is True
    assert [call for call in calls if call[0] == "concat" and call[2] == 2] == []


def test_sdpa_q_chunked_slices_q_and_mask(monkeypatch):
    calls = []
    monkeypatch.setenv("DG_SDPA_FULLCANVAS", "0")

    class _Tensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape
            self.deallocated = False

        def deallocate(self, force):
            self.deallocated = force

    class _FakeTtnn:
        DRAM_MEMORY_CONFIG = "dram"

        @staticmethod
        def CoreCoord(x, y):
            return ("grid", x, y)

        @staticmethod
        def SDPAProgramConfig(**kwargs):
            calls.append(("program", kwargs))
            return "program"

        @staticmethod
        def slice(tensor, starts, ends, *, memory_config=None):
            calls.append(("slice", tensor.name, starts, ends, memory_config))
            return _Tensor(
                f"{tensor.name}[s{starts[2]}:{ends[2]}]",
                [ends[idx] - starts[idx] for idx in range(4)],
            )

        @staticmethod
        def concat(tensors, *, dim, memory_config):
            calls.append(("concat", [tensor.name for tensor in tensors], dim, memory_config))
            shape = list(tensors[0].shape)
            shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
            return _Tensor("sdpa-out", shape)

    class _FakeTransformer:
        @staticmethod
        def scaled_dot_product_attention(q, k, v, **kwargs):
            calls.append(("sdpa", q.name, k.name, v.name, getattr(kwargs.get("attn_mask"), "name", None), kwargs))
            return _Tensor(f"sdpa({q.name})", q.shape)

    _FakeTtnn.transformer = _FakeTransformer
    monkeypatch.setattr(DA, "ttnn", _FakeTtnn)

    out = _sdpa_q_chunked(
        _Tensor("q", [1, 4, 96, 256]),
        _Tensor("k", [1, 2, 128, 256]),
        _Tensor("v", [1, 2, 128, 256]),
        attn_mask=_Tensor("mask", [1, 1, 96, 128]),
        head_dim=256,
    )

    assert out.shape == [1, 4, 96, 256]
    assert [call[1] for call in calls if call[0] == "sdpa"] == ["q[s0:32]", "q[s32:64]", "q[s64:96]"]
    assert [call[4] for call in calls if call[0] == "sdpa"] == ["mask[s0:32]", "mask[s32:64]", "mask[s64:96]"]
    assert calls[-1] == ("concat", ["sdpa(q[s0:32])", "sdpa(q[s32:64])", "sdpa(q[s64:96])"], 2, "dram")


def test_sdpa_q_chunked_falls_back_to_manual_gqa_on_l1_clash(monkeypatch):
    calls = []

    class _Tensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape
            self.deallocated = False

        def deallocate(self, force):
            self.deallocated = force

    class _FakeTtnn:
        DRAM_MEMORY_CONFIG = "dram"

        @staticmethod
        def CoreCoord(x, y):
            return ("grid", x, y)

        @staticmethod
        def SDPAProgramConfig(**kwargs):
            calls.append(("program", kwargs))
            return "program"

        @staticmethod
        def slice(tensor, starts, ends, *, memory_config=None):
            calls.append(("slice", tensor.name, starts, ends, memory_config))
            return _Tensor(
                f"{tensor.name}[h{starts[1]}:{ends[1]},s{starts[2]}:{ends[2]}]",
                [ends[idx] - starts[idx] for idx in range(4)],
            )

        @staticmethod
        def clone(tensor, *, memory_config):
            calls.append(("clone", tensor.name, memory_config))
            return _Tensor(f"clone({tensor.name})", tensor.shape)

        @staticmethod
        def concat(tensors, *, dim, memory_config):
            calls.append(("concat", [tensor.name for tensor in tensors], dim, memory_config))
            shape = list(tensors[0].shape)
            shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
            return _Tensor(f"concat{dim}", shape)

        @staticmethod
        def permute(tensor, order, *, memory_config):
            calls.append(("permute", tensor.name, order, memory_config))
            shape = [tensor.shape[idx] for idx in order]
            return _Tensor(f"permute({tensor.name})", shape)

        @staticmethod
        def matmul(lhs, rhs, *, transpose_b=False, memory_config):
            calls.append(("matmul", lhs.name, rhs.name, transpose_b, memory_config))
            rhs_name = f"transpose({rhs.name})" if transpose_b else rhs.name
            out_width = rhs.shape[-2] if transpose_b else rhs.shape[-1]
            return _Tensor(f"matmul({lhs.name},{rhs_name})", [lhs.shape[0], lhs.shape[1], lhs.shape[2], out_width])

        @staticmethod
        def softmax(tensor, *, dim, numeric_stable):
            calls.append(("softmax", tensor.name, dim, numeric_stable))
            return _Tensor(f"softmax({tensor.name})", tensor.shape)

    class _FakeTransformer:
        @staticmethod
        def scaled_dot_product_attention(q, k, v, **kwargs):
            calls.append(("sdpa", q.name, k.name, v.name, kwargs))
            raise RuntimeError("Statically allocated circular buffers in program clash with L1 buffers")

    _FakeTtnn.transformer = _FakeTransformer
    monkeypatch.setattr(DA, "ttnn", _FakeTtnn)

    out = _sdpa_q_chunked(
        _Tensor("q", [1, 4, 32, 256]),
        _Tensor("k", [1, 2, 288, 256]),
        _Tensor("v", [1, 2, 288, 256]),
        head_dim=256,
    )

    assert out.shape == [1, 4, 32, 256]
    assert [call[0] for call in calls].count("sdpa") == 1
    assert [call for call in calls if call[0] == "softmax"] == [
        ("softmax", "matmul(q[h0:2,s0:32],transpose(concat1))", -1, True),
        ("softmax", "matmul(q[h2:4,s0:32],transpose(concat1))", -1, True),
    ]


def test_sdpa_q_chunked_warns_about_gqa_fallback_once(monkeypatch):
    warnings = []

    monkeypatch.setattr(DA, "_FALLBACK_WARNED", False)
    DA.reset_sdpa_fallback_counts()
    monkeypatch.setattr(DA.logger, "warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(DA, "_manual_gqa_attention", lambda q, k, v: "staged")

    def raising_sdpa(q, k, v, **kwargs):
        raise RuntimeError("Statically allocated circular buffers in program clash with L1 buffers")

    monkeypatch.setattr(DA.ttnn.transformer, "scaled_dot_product_attention", raising_sdpa)
    monkeypatch.setattr(DA, "_denoise_sdpa_program_config", lambda *args, **kwargs: "program")

    tt_q = SimpleNamespace(shape=[1, 4, 32, 256])
    tt_k = SimpleNamespace(shape=[1, 2, 288, 256])
    tt_v = SimpleNamespace(shape=[1, 2, 288, 256])

    first = _sdpa_q_chunked(tt_q, tt_k, tt_v, head_dim=256, layer_idx=5)
    second = _sdpa_q_chunked(tt_q, tt_k, tt_v, head_dim=256, layer_idx=5)

    assert first == "staged"
    assert second == "staged"
    assert len(warnings) == 1
    assert "staged GQA fallback" in warnings[0]
    assert DA.get_sdpa_fallback_counts() == {5: 2}
