# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the self-conditioning gated MLP (#47461/#47463).

Reconciled against transformers ``DiffusionGemmaSelfConditioning``: the module is
``post_norm(inputs_embeds + down(act(gate(pre_norm(signal))) * up(pre_norm(signal))))``
with a scaleless ``post_norm``, and the soft-embedding feeds it.

The device PCC checks validate the ttnn module against the pure-torch reference oracle
(`reference/self_conditioning.py`). They are checkpoint-free: random weights are generated
in the reference module and loaded verbatim into the device module, so they isolate the
module's compute (RMSNorm conventions + GeGLU) from weight loading (which
`tests/test_config.py` already validates against the real checkpoint).

Run on QB2:
  DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_self_conditioning.py
"""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.reference.self_conditioning import (
    DiffusionGemmaRMSNorm,
    SelfConditioning,
)
from models.experimental.diffusion_gemma.tt.self_conditioning import (
    build_self_conditioning_embedding_weight,
    build_self_conditioning,
    TtSelfConditioning,
    _dram_for_rms_norm,
    _width_sharded_rms_norm,
    _rms_norm_dram,
    self_conditioning_logits_l1_mode,
    validate_self_conditioning_state,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


# --- reference gated MLP ----------------------------------------------------


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _embedding(vocab, hidden, seed=0):
    return torch.randn(vocab, hidden, generator=_gen(seed))


def test_module_has_expected_params_and_scaleless_post_norm():
    mod = SelfConditioning(16, intermediate_size=40)
    names = dict(mod.named_parameters())
    # the 4 checkpoint weights: pre_norm + gate/up/down; post_norm is scaleless (no weight)
    assert "pre_norm.weight" in names
    assert {"gate_proj.weight", "up_proj.weight", "down_proj.weight"} <= set(names)
    assert not mod.post_norm.with_scale and not hasattr(mod.post_norm, "weight")
    assert names["gate_proj.weight"].shape == (40, 16)
    assert names["down_proj.weight"].shape == (16, 40)


def test_forward_shape_and_is_post_norm_of_sum():
    batch, length, hidden, inter = 2, 8, 16, 40
    mod = SelfConditioning(hidden, intermediate_size=inter)
    emb = torch.randn(batch, length, hidden, generator=_gen(1))
    signal = torch.randn(batch, length, hidden, generator=_gen(2))
    out = mod(emb, signal)
    assert out.shape == (batch, length, hidden)
    # reproduce the exact composition
    normed = mod.pre_norm(signal)
    sc = mod.down_proj(mod._act(mod.gate_proj(normed)) * mod.up_proj(normed))
    assert torch.allclose(out, mod.post_norm(emb + sc), atol=1e-6)


def test_zero_signal_is_post_norm_of_embeds_not_identity():
    # First denoise step / disabled: zero signal -> post_norm(inputs_embeds), NOT
    # inputs_embeds unchanged (the decoder always post-normalizes its embeddings).
    batch, length, vocab, hidden = 2, 8, 32, 16
    mod = SelfConditioning(hidden, intermediate_size=24)
    emb = torch.randn(batch, length, hidden, generator=_gen(3))
    out_disabled = mod.condition(emb, None, _embedding(vocab, hidden), enabled=False)
    assert torch.allclose(out_disabled, mod.post_norm(emb), atol=1e-6)
    assert torch.allclose(out_disabled, mod(emb, torch.zeros_like(emb)), atol=1e-6)
    # and it is NOT just the input embeddings (post_norm rescales)
    assert not torch.allclose(out_disabled, emb, atol=1e-3)


def test_activation_silu_supported_and_unknown_rejected(expect_error):
    SelfConditioning(8, activation="silu")  # ok
    with expect_error(ValueError):
        SelfConditioning(8, activation="relu6")._act(torch.zeros(1))


# --- reference soft embedding -----------------------------------------------


def test_soft_embedding_onehot_recovers_scaled_token_row():
    # Canonical applies embed_scale = hidden**0.5 -> a one-hot recovers scale * emb[row].
    vocab, hidden = 20, 12
    emb = _embedding(vocab, hidden, seed=3)
    logits = torch.full((1, 1, vocab), -1e4)
    logits[..., 7] = 1e4
    soft = SelfConditioning.soft_embedding(logits, emb)
    assert torch.allclose(soft[0, 0], emb[7] * (hidden**0.5), atol=1e-3)
    # explicit, independent check that the scale is present (guards against re-dropping it)
    assert not torch.allclose(soft[0, 0], emb[7], atol=1e-2)


def test_soft_embedding_is_scaled_convex_combination():
    # soft / embed_scale is the convex combination (lies in the embedding bounding box).
    vocab, hidden = 6, 4
    emb = _embedding(vocab, hidden, seed=4)
    soft = SelfConditioning.soft_embedding(torch.randn(1, 5, vocab, generator=_gen(5)), emb) / (hidden**0.5)
    assert torch.all(soft <= emb.max(dim=0).values + 1e-5)
    assert torch.all(soft >= emb.min(dim=0).values - 1e-5)


def test_soft_embedding_per_example_mask_zeroes_signal():
    vocab, hidden = 10, 8
    emb = _embedding(vocab, hidden, seed=7)
    logits = torch.randn(3, 5, vocab, generator=_gen(8))
    mask = torch.tensor([True, False, True])
    soft = SelfConditioning.soft_embedding(logits, emb, mask=mask)
    assert torch.all(soft[1] == 0)  # masked example -> zero signal
    assert torch.any(soft[0] != 0) and torch.any(soft[2] != 0)


# --- reference rmsnorm ------------------------------------------------------


def test_rmsnorm_matches_reference_formula():
    x = torch.randn(2, 3, 16, generator=_gen(9))
    n = DiffusionGemmaRMSNorm(16, with_scale=False)
    expected = x.float() * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
    assert torch.allclose(n(x), expected.type_as(x), atol=1e-6)


# --- ttnn state validation and construction ---------------------------------


def _state(hidden_size=8, intermediate_size=6):
    return {
        "pre_norm.weight": torch.ones(hidden_size),
        "gate_proj.weight": torch.ones(intermediate_size, hidden_size),
        "up_proj.weight": torch.ones(intermediate_size, hidden_size),
        "down_proj.weight": torch.ones(hidden_size, intermediate_size),
    }


def test_validate_self_conditioning_state_accepts_expected_shapes():
    validate_self_conditioning_state(_state(), hidden_size=8, intermediate_size=6)


@pytest.mark.parametrize(
    "key,replacement,match",
    [
        pytest.param("up_proj.weight", None, "missing self-conditioning weights", id="missing-weight"),
        pytest.param("down_proj.weight", torch.ones(6, 8), "down_proj.weight has shape", id="transposed-shape"),
    ],
)
def test_validate_self_conditioning_state_rejects_bad_state(key, replacement, match, expect_error):
    state = _state()
    if replacement is None:
        del state[key]
    else:
        state[key] = replacement

    with expect_error(ValueError, match=match):
        validate_self_conditioning_state(state, hidden_size=8, intermediate_size=6)


def test_build_self_conditioning_uses_config_and_forwards_constructor_args():
    calls = {}

    class _FakeSelfConditioning:
        def __init__(self, device, state_dict, **kwargs):
            calls["ctor"] = (device, state_dict, kwargs)

    config = SimpleNamespace(hidden_size=8, intermediate_size=6, rms_norm_eps=1e-5)

    out = build_self_conditioning(
        "device",
        _state(),
        config=config,
        dtype="dtype",
        module_cls=_FakeSelfConditioning,
    )

    assert isinstance(out, _FakeSelfConditioning)
    assert calls["ctor"][0] == "device"
    expected_state = _state()
    assert calls["ctor"][1].keys() == expected_state.keys()
    for key, expected in expected_state.items():
        assert torch.equal(calls["ctor"][1][key], expected)
    assert calls["ctor"][2] == {
        "hidden_size": 8,
        "intermediate_size": 6,
        "eps": 1e-5,
        "dtype": "dtype",
    }


def test_build_self_conditioning_requires_dimensions_without_config(expect_error):
    with expect_error(ValueError, match="hidden_size and intermediate_size"):
        build_self_conditioning("device", _state(), module_cls=object)


# --- ttnn embedding weight --------------------------------------------------


def test_build_self_conditioning_embedding_weight_uses_matmul_layout(monkeypatch):
    calls = {}
    monkeypatch.setenv("DG_SELFCOND_PRECHUNK_EMBED", "0")

    class _FakeTtnn:
        bfloat16 = "bf16"
        TILE_LAYOUT = "tile"
        DRAM_MEMORY_CONFIG = "dram"

        @staticmethod
        def as_tensor(value, **kwargs):
            calls["as_tensor"] = (value.clone(), kwargs)
            return "device-embedding"

    from models.experimental.diffusion_gemma.tt import self_conditioning as SC

    monkeypatch.setattr(SC, "ttnn", _FakeTtnn)
    embedding = torch.arange(24, dtype=torch.float32).reshape(3, 8)

    out = build_self_conditioning_embedding_weight(
        "device",
        embedding,
        hidden_size=8,
        dtype="bf16",
        tensor_fn=_FakeTtnn.as_tensor,
    )

    value, kwargs = calls["as_tensor"]
    assert out == "device-embedding"
    assert value.shape == (1, 1, 3, 8)
    assert torch.equal(value[0, 0], embedding)
    assert kwargs == {
        "device": "device",
        "dtype": "bf16",
        "layout": "tile",
        "memory_config": "dram",
    }


def test_build_self_conditioning_embedding_weight_can_prechunk_without_changing_values(monkeypatch):
    calls = []

    class _FakeTtnn:
        TILE_LAYOUT = "tile"
        DRAM_MEMORY_CONFIG = "dram"

        @staticmethod
        def as_tensor(value, **kwargs):
            calls.append((value.clone(), kwargs))
            return value.clone()

    from models.experimental.diffusion_gemma.tt import self_conditioning as SC

    monkeypatch.setattr(SC, "ttnn", _FakeTtnn)
    monkeypatch.setenv("DG_SELFCOND_PRECHUNK_EMBED", "1")
    embedding = torch.arange(16385 * 8, dtype=torch.float32).reshape(16385, 8)

    out = build_self_conditioning_embedding_weight(
        "device",
        embedding,
        hidden_size=8,
        dtype="bf16",
        tensor_fn=_FakeTtnn.as_tensor,
    )

    assert isinstance(out, SC.ChunkedEmbeddingWeight)
    assert out.shape == (1, 1, 16385, 8)
    assert out.chunk_size == 8192
    assert [tuple(chunk.shape) for chunk in out.chunks] == [(1, 1, 8192, 8), (1, 1, 8192, 8), (1, 1, 1, 8)]
    assert torch.equal(torch.cat([chunk[0, 0] for chunk in out.chunks]), embedding)
    assert all(
        kwargs
        == {
            "device": "device",
            "dtype": "bf16",
            "layout": "tile",
            "memory_config": "dram",
        }
        for _, kwargs in calls
    )


def test_build_self_conditioning_embedding_weight_defaults_to_prechunk(monkeypatch):
    class _FakeTtnn:
        TILE_LAYOUT = "tile"
        DRAM_MEMORY_CONFIG = "dram"

        @staticmethod
        def as_tensor(value, **kwargs):
            return value.clone()

    from models.experimental.diffusion_gemma.tt import self_conditioning as SC

    monkeypatch.setattr(SC, "ttnn", _FakeTtnn)
    monkeypatch.delenv("DG_SELFCOND_PRECHUNK_EMBED", raising=False)
    embedding = torch.zeros(8193, 8)

    out = build_self_conditioning_embedding_weight(
        "device",
        embedding,
        hidden_size=8,
        dtype="bf16",
        tensor_fn=_FakeTtnn.as_tensor,
    )

    assert isinstance(out, SC.ChunkedEmbeddingWeight)
    assert [tuple(chunk.shape) for chunk in out.chunks] == [(1, 1, 8192, 8), (1, 1, 1, 8)]


def test_build_self_conditioning_embedding_weight_rejects_hidden_mismatch(expect_error):
    with expect_error(ValueError, match="embedding hidden size"):
        build_self_conditioning_embedding_weight(
            "device", torch.ones(3, 8), hidden_size=16, tensor_fn=lambda *a, **k: None
        )


# --- ttnn rms-norm memory placement -----------------------------------------


def test_dram_for_rms_norm_moves_l1_or_sharded_inputs(monkeypatch):
    calls = []

    class _Mem:
        def __init__(self, buffer_type, *, sharded=False):
            self.buffer_type = buffer_type
            self._sharded = sharded

        def is_sharded(self):
            return self._sharded

    class _Tensor:
        def __init__(self, name, mem):
            self.name = name
            self._mem = mem
            self.shape = (1, 1, 32, 8)

        def memory_config(self):
            return self._mem

    class _FakeTtnn:
        class BufferType:
            DRAM = "dram"

        DRAM_MEMORY_CONFIG = "dram-memcfg"

        @staticmethod
        def to_memory_config(tensor, memory_config):
            calls.append((tensor, memory_config))
            return _Tensor(f"{tensor.name}-dram", _Mem("dram"))

    from models.experimental.diffusion_gemma.tt import self_conditioning as SC

    monkeypatch.setattr(SC, "ttnn", _FakeTtnn)

    dram = _Tensor("dram", _Mem("dram"))
    l1 = _Tensor("l1", _Mem("l1"))
    sharded_dram = _Tensor("sharded", _Mem("dram", sharded=True))

    assert _dram_for_rms_norm(dram) is dram
    assert _dram_for_rms_norm(l1).name == "l1-dram"
    assert _dram_for_rms_norm(sharded_dram).name == "sharded-dram"
    assert calls == [(l1, "dram-memcfg"), (sharded_dram, "dram-memcfg")]


def test_forward_requests_dram_rms_norm_outputs(monkeypatch):
    calls = []

    class _Mem:
        buffer_type = "dram"

        def is_sharded(self):
            return False

    class _Tensor:
        def __init__(self, name):
            self.name = name
            self.shape = (1, 1, 32, 8)
            self.deallocated = False

        def memory_config(self):
            return _Mem()

        def deallocate(self, force):
            self.deallocated = force

    class _FakeTtnn:
        class BufferType:
            DRAM = "dram"

        DRAM_MEMORY_CONFIG = "dram-memcfg"

        @staticmethod
        def rms_norm(tensor, **kwargs):
            calls.append(("rms_norm", tensor, kwargs))
            return _Tensor(f"norm({tensor.name})")

        @staticmethod
        def linear(tensor, weight):
            calls.append(("linear", tensor, weight))
            return _Tensor(f"linear({tensor.name})")

        @staticmethod
        def gelu(tensor, *, fast_and_approximate_mode):
            calls.append(("gelu", tensor, fast_and_approximate_mode))
            return tensor

        @staticmethod
        def mul(lhs, rhs):
            calls.append(("mul", lhs, rhs))
            return _Tensor("hidden")

        @staticmethod
        def add(lhs, rhs):
            calls.append(("add", lhs, rhs))
            return _Tensor("summed")

    from models.experimental.diffusion_gemma.tt import self_conditioning as SC

    monkeypatch.setattr(SC, "ttnn", _FakeTtnn)
    monkeypatch.setattr(SC, "apply_gelu", lambda tensor: _FakeTtnn.gelu(tensor, fast_and_approximate_mode=True))
    module = TtSelfConditioning.__new__(TtSelfConditioning)
    module.eps = 1e-6
    module.pre_norm_weight = "pre-weight"
    module.gate_proj = "gate"
    module.up_proj = "up"
    module.down_proj = "down"

    out = module.forward(_Tensor("embeds"), _Tensor("signal"))

    assert out.name == "norm(summed)"
    rms_calls = [call for call in calls if call[0] == "rms_norm"]
    assert len(rms_calls) == 2
    assert rms_calls[0][2]["memory_config"] == "dram-memcfg"
    assert rms_calls[0][2]["weight"] == "pre-weight"
    assert rms_calls[1][2]["memory_config"] == "dram-memcfg"
    assert "weight" not in rms_calls[1][2]


def test_condition_without_prev_logits_uses_post_norm_fast_path(monkeypatch):
    calls = []

    class _Mem:
        buffer_type = "dram"

        def is_sharded(self):
            return False

    class _Tensor:
        def __init__(self, name):
            self.name = name
            self.shape = (1, 1, 32, 8)

        def memory_config(self):
            return _Mem()

    class _FakeTtnn:
        class BufferType:
            DRAM = "dram"

        DRAM_MEMORY_CONFIG = "dram-memcfg"

        @staticmethod
        def rms_norm(tensor, **kwargs):
            calls.append(("rms_norm", tensor, kwargs))
            return _Tensor(f"norm({tensor.name})")

        @staticmethod
        def mul(*args, **kwargs):
            raise AssertionError("zero-signal MLP path should be skipped")

    from models.experimental.diffusion_gemma.tt import self_conditioning as SC

    monkeypatch.setattr(SC, "ttnn", _FakeTtnn)
    module = TtSelfConditioning.__new__(TtSelfConditioning)
    module.eps = 1e-6

    out = module.condition(_Tensor("embeds"), None, "embedding")

    assert out.name == "norm(embeds)"
    assert len(calls) == 1
    assert calls[0][0] == "rms_norm"
    assert calls[0][1].name == "embeds"
    assert calls[0][2] == {"epsilon": 1e-6, "memory_config": "dram-memcfg"}


def test_rms_norm_dram_chunks_long_sequences(monkeypatch):
    calls = []

    class _Mem:
        buffer_type = "dram"

        def is_sharded(self):
            return False

    class _Tensor:
        def __init__(self, name, shape=(1, 1, 96, 8)):
            self.name = name
            self.shape = shape
            self.deallocated = False

        def memory_config(self):
            return _Mem()

        def deallocate(self, force):
            self.deallocated = force

    class _FakeTtnn:
        class BufferType:
            DRAM = "dram"

        DRAM_MEMORY_CONFIG = "dram-memcfg"

        @staticmethod
        def slice(tensor, starts, ends, *, memory_config):
            calls.append(("slice", tensor.name, starts, ends, memory_config))
            return _Tensor(f"{tensor.name}[{starts[2]}:{ends[2]}]", (1, 1, ends[2] - starts[2], 8))

        @staticmethod
        def rms_norm(tensor, **kwargs):
            calls.append(("rms_norm", tensor.name, kwargs))
            return _Tensor(f"norm({tensor.name})", tensor.shape)

        @staticmethod
        def concat(tensors, *, dim, memory_config):
            calls.append(("concat", [tensor.name for tensor in tensors], dim, memory_config))
            return _Tensor("concat", (1, 1, sum(tensor.shape[2] for tensor in tensors), 8))

    from models.experimental.diffusion_gemma.tt import self_conditioning as SC

    monkeypatch.setattr(SC, "ttnn", _FakeTtnn)

    out = _rms_norm_dram(_Tensor("x"), epsilon=1e-6)

    assert out.name == "concat"
    assert [call[0] for call in calls] == ["slice", "rms_norm", "slice", "rms_norm", "slice", "rms_norm", "concat"]
    assert calls[-1] == (
        "concat",
        ["norm(x[0:32])", "norm(x[32:64])", "norm(x[64:96])"],
        2,
        "dram-memcfg",
    )


def test_width_sharded_rms_norm_uses_sharded_program_for_production_width(monkeypatch):
    calls = []

    class _Tensor:
        def __init__(self, name, shape=(1, 1, 32, 2816)):
            self.name = name
            self.shape = shape
            self.deallocated = False

        def deallocate(self, force):
            self.deallocated = force

    class _FakeTtnn:
        TILE_SIZE = 32
        DRAM_MEMORY_CONFIG = "dram"
        ShardStrategy = SimpleNamespace(WIDTH="width")
        ShardOrientation = SimpleNamespace(ROW_MAJOR="row-major")

        @staticmethod
        def CoreGrid(*, x, y):
            return ("grid", x, y)

        @staticmethod
        def create_sharded_memory_config(shape, **kwargs):
            calls.append(("create_mem", shape, kwargs))
            return "sharded-mem"

        @staticmethod
        def LayerNormShardedMultiCoreProgramConfig(**kwargs):
            calls.append(("program", kwargs))
            return "program"

        @staticmethod
        def to_memory_config(tensor, memory_config):
            calls.append(("to_mem", tensor.name, memory_config))
            return _Tensor(f"{tensor.name}:{memory_config}", tensor.shape)

        @staticmethod
        def rms_norm(tensor, **kwargs):
            calls.append(("rms_norm", tensor.name, kwargs))
            return _Tensor("sharded-out", tensor.shape)

        @staticmethod
        def sharded_to_interleaved(tensor, memory_config):
            calls.append(("to_interleaved", tensor.name, memory_config))
            return _Tensor("dram-out", tensor.shape)

    from models.experimental.diffusion_gemma.tt import self_conditioning as SC

    monkeypatch.setattr(SC, "ttnn", _FakeTtnn)

    out = _width_sharded_rms_norm(_Tensor("x"), weight=_Tensor("w"), epsilon=1e-6)

    assert out.name == "dram-out"
    assert calls[0] == (
        "create_mem",
        (32, 2816),
        {"core_grid": ("grid", 8, 1), "strategy": "width", "orientation": "row-major"},
    )
    assert calls[1] == (
        "program",
        {"compute_with_storage_grid_size": (8, 1), "subblock_w": 1, "block_h": 1, "block_w": 11, "inplace": False},
    )
    assert ("to_mem", "x", "sharded-mem") in calls
    assert ("to_mem", "w", "sharded-mem") in calls
    rms_call = next(call for call in calls if call[0] == "rms_norm")
    assert rms_call[2]["program_config"] == "program"
    assert rms_call[2]["memory_config"] == "sharded-mem"


# --- ttnn soft embedding chunking -------------------------------------------


def test_self_conditioning_logits_l1_defaults_to_chain(monkeypatch):
    monkeypatch.delenv("DG_SELFCOND_LOGITS_L1", raising=False)
    assert self_conditioning_logits_l1_mode() == "chain"


def test_soft_embedding_chunks_large_vocab_without_full_softmax(monkeypatch):
    calls = []
    monkeypatch.setenv("DG_SELFCOND_LOGITS_L1", "off")

    class _Tensor:
        def __init__(self, name, shape, memory_config="dram"):
            self.name = name
            self.shape = shape
            self.memory_config = memory_config
            self.deallocated = False

        def deallocate(self, force):
            self.deallocated = force

    class _FakeTtnn:
        DRAM_MEMORY_CONFIG = "dram"
        L1_MEMORY_CONFIG = "l1"

        @staticmethod
        def softmax(*args, **kwargs):
            raise AssertionError("chunked path must not materialize full softmax")

        @staticmethod
        def max(tensor, *, dim, keepdim):
            calls.append(("max", tensor.name, dim, keepdim))
            return _Tensor(f"max({tensor.name})", (1, 1, tensor.shape[2], 1))

        @staticmethod
        def slice(tensor, starts, ends, *, memory_config):
            calls.append(("slice", tensor.name, starts, ends, memory_config))
            shape = (
                ends[0] - starts[0],
                ends[1] - starts[1],
                ends[2] - starts[2],
                ends[3] - starts[3],
            )
            return _Tensor(
                f"{tensor.name}[{starts[2]}:{ends[2]},{starts[3]}:{ends[3]}]",
                shape,
                memory_config,
            )

        @staticmethod
        def subtract(a, b, *, memory_config=None):
            output_memory_config = memory_config or a.memory_config
            calls.append(("subtract", a.name, b.name, output_memory_config))
            return _Tensor(f"sub({a.name},{b.name})", a.shape, output_memory_config)

        @staticmethod
        def exp(tensor, *, memory_config=None):
            output_memory_config = memory_config or tensor.memory_config
            calls.append(("exp", tensor.name, output_memory_config))
            return _Tensor(f"exp({tensor.name})", tensor.shape, output_memory_config)

        @staticmethod
        def sum(tensor, *, dim, keepdim, memory_config=None):
            output_memory_config = memory_config or tensor.memory_config
            calls.append(("sum", tensor.name, dim, keepdim, output_memory_config))
            return _Tensor(
                f"sum({tensor.name})",
                (tensor.shape[0], tensor.shape[1], tensor.shape[2], 1),
                output_memory_config,
            )

        @staticmethod
        def matmul(a, b, *, memory_config):
            calls.append(("matmul", a.name, b.name, memory_config))
            return _Tensor(
                f"matmul({a.name},{b.name})",
                (a.shape[0], a.shape[1], a.shape[2], b.shape[-1]),
                memory_config,
            )

        @staticmethod
        def add(a, b, *, memory_config=None):
            output_memory_config = memory_config or a.memory_config
            calls.append(("add", a.name, b.name, output_memory_config))
            return _Tensor(f"add({a.name},{b.name})", a.shape, output_memory_config)

        @staticmethod
        def div(a, b, *, memory_config=None):
            output_memory_config = memory_config or a.memory_config
            calls.append(("div", a.name, b.name, output_memory_config))
            return _Tensor(f"div({a.name},{b.name})", a.shape, output_memory_config)

        @staticmethod
        def multiply(tensor, scalar):
            calls.append(("multiply", tensor.name, scalar))
            return _Tensor(f"mul({tensor.name})", tensor.shape, tensor.memory_config)

    from models.experimental.diffusion_gemma.tt import self_conditioning as SC

    monkeypatch.setattr(SC, "ttnn", _FakeTtnn)
    module = TtSelfConditioning.__new__(TtSelfConditioning)
    module.hidden_size = 8

    out = module._soft_embedding_chunked(
        _Tensor("logits", (1, 1, 32, 64)),
        _Tensor("embedding", (1, 1, 64, 8)),
        vocab_chunk_size=32,
    )

    assert out.name.startswith("mul(div(")
    assert [call[0] for call in calls].count("matmul") == 2
    assert [call for call in calls if call[0] == "slice"] == [
        ("slice", "logits", [0, 0, 0, 0], [1, 1, 32, 32], "dram"),
        ("slice", "embedding", [0, 0, 0, 0], [1, 1, 32, 8], "dram"),
        ("slice", "logits", [0, 0, 0, 32], [1, 1, 32, 64], "dram"),
        ("slice", "embedding", [0, 0, 32, 0], [1, 1, 64, 8], "dram"),
    ]
    assert [call[-1] for call in calls if call[0] == "sum"] == ["dram", "dram"]
    assert [call[-1] for call in calls if call[0] == "add"] == ["dram", "dram"]
    assert [call[-1] for call in calls if call[0] == "div"] == ["dram"]

    calls.clear()
    persistent_chunks = (
        _Tensor("embedding-chunk-0", (1, 1, 32, 8)),
        _Tensor("embedding-chunk-1", (1, 1, 32, 8)),
        _Tensor("embedding-chunk-2", (1, 1, 1, 8)),
    )
    chunked_weight = SC.ChunkedEmbeddingWeight(
        chunks=persistent_chunks,
        shape=(1, 1, 65, 8),
        chunk_size=32,
    )

    out = module._soft_embedding_chunked(
        _Tensor("logits", (1, 1, 32, 65)),
        chunked_weight,
        vocab_chunk_size=8192,
    )

    assert out.name.startswith("mul(div(")
    assert [call for call in calls if call[0] == "slice"] == [
        ("slice", "logits", [0, 0, 0, 0], [1, 1, 32, 32], "dram"),
        ("slice", "logits", [0, 0, 0, 32], [1, 1, 32, 64], "dram"),
        ("slice", "logits", [0, 0, 0, 64], [1, 1, 32, 65], "dram"),
    ]
    assert [call for call in calls if call[0] == "matmul"] == [
        ("matmul", "exp(sub(logits[0:32,0:32],max(logits)))", "embedding-chunk-0", "dram"),
        ("matmul", "exp(sub(logits[0:32,32:64],max(logits)))", "embedding-chunk-1", "dram"),
        ("matmul", "exp(sub(logits[0:32,64:65],max(logits)))", "embedding-chunk-2", "dram"),
    ]
    assert [call[-1] for call in calls if call[0] == "sum"] == ["dram", "dram", "dram"]
    assert [call[-1] for call in calls if call[0] == "add"] == ["dram", "dram", "dram", "dram"]
    assert [call[-1] for call in calls if call[0] == "div"] == ["dram"]
    assert all(not chunk.deallocated for chunk in persistent_chunks)

    monkeypatch.setenv("DG_SELFCOND_LOGITS_L1", "chain")
    calls.clear()
    out = module._soft_embedding_chunked(
        _Tensor("logits", (1, 1, 32, 65)),
        chunked_weight,
        vocab_chunk_size=8192,
    )

    assert out.name.startswith("mul(div(")
    assert [call[-1] for call in calls if call[0] == "slice"] == ["l1", "l1", "l1"]
    assert [call[-1] for call in calls if call[0] == "subtract"] == ["l1", "l1", "l1"]
    assert [call[-1] for call in calls if call[0] == "exp"] == ["l1", "l1", "l1"]
    assert [call[-1] for call in calls if call[0] == "sum"] == ["l1", "l1", "l1"]
    assert [call for call in calls if call[0] == "matmul"] == [
        ("matmul", "exp(sub(logits[0:32,0:32],max(logits)))", "embedding-chunk-0", "dram"),
        ("matmul", "exp(sub(logits[0:32,32:64],max(logits)))", "embedding-chunk-1", "dram"),
        ("matmul", "exp(sub(logits[0:32,64:65],max(logits)))", "embedding-chunk-2", "dram"),
    ]
    assert [call[-1] for call in calls if call[0] == "add"] == ["dram", "l1", "dram", "l1"]
    assert [call[-1] for call in calls if call[0] == "div"] == ["dram"]
    assert all(not chunk.deallocated for chunk in persistent_chunks)


# --- device PCC -------------------------------------------------------------

# 26B-A4B dims: hidden 2816, self-conditioning intermediate = dense intermediate 2112 (NOT moe 704).
HIDDEN, INTER, EPS = 2816, 2112, 1e-6


def _build(seed):
    """Reference module (random weights) + a device module loaded from its weights."""
    torch.manual_seed(seed)
    ref = SelfConditioning(HIDDEN, INTER, eps=EPS, activation="gelu_pytorch_tanh").eval()
    state = {
        "pre_norm.weight": ref.pre_norm.weight.data.clone(),
        "gate_proj.weight": ref.gate_proj.weight.data.clone(),
        "up_proj.weight": ref.up_proj.weight.data.clone(),
        "down_proj.weight": ref.down_proj.weight.data.clone(),
    }
    return ref, state


def _to_dev(t, device):
    return ttnn.from_torch(t.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # [1,B,L,H]


def _embed_to_dev(embed_w, device):
    """Tied embedding table [vocab, hidden] -> ttnn [1,1,vocab,hidden] TILE (matmul operand)."""
    return ttnn.from_torch(
        embed_w.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


# ``use_module_device`` gives one device open/teardown for the whole module — avoid the
# QB2 erisc-29-25 teardown hang from repeated CreateDevice (see test_attention).
@pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
)
@pytest.mark.use_module_device
@pytest.mark.parametrize("seq_len", [256])
def test_self_conditioning_pcc(device, seq_len):
    ref, state = _build(0)
    tt = TtSelfConditioning(device, state, hidden_size=HIDDEN, intermediate_size=INTER, eps=EPS)

    emb = torch.randn(1, seq_len, HIDDEN)
    signal = torch.randn(1, seq_len, HIDDEN)

    with torch.no_grad():
        golden = ref(emb, signal)  # [1, L, H]

    out = ttnn.to_torch(tt.forward(_to_dev(emb, device), _to_dev(signal, device)))[0]  # [1,B,L,H] -> [B,L,H]
    assert_with_pcc(golden, out, 0.99)


@pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
)
@pytest.mark.use_module_device
@pytest.mark.parametrize("seq_len", [256])
def test_zero_signal_is_post_norm_of_embeds(device, seq_len):
    """Zero signal -> post_norm(inputs_embeds), NOT inputs_embeds unchanged."""
    ref, state = _build(1)
    tt = TtSelfConditioning(device, state, hidden_size=HIDDEN, intermediate_size=INTER, eps=EPS)

    emb = torch.randn(1, seq_len, HIDDEN)
    signal = torch.zeros(1, seq_len, HIDDEN)

    with torch.no_grad():
        golden = ref(emb, signal)  # == ref.post_norm(emb)

    out = ttnn.to_torch(tt.forward(_to_dev(emb, device), _to_dev(signal, device)))[0]
    assert_with_pcc(golden, out, 0.99)
    # sanity: the module rescales the embeddings (post_norm), so it is not identity
    assert not torch.allclose(out.float(), emb, atol=1e-3)


@pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
)
@pytest.mark.use_module_device
@pytest.mark.parametrize("seq_len", [256])
@pytest.mark.parametrize("vocab", [256])
def test_condition_full_path_pcc(device, seq_len, vocab):
    """Full self-conditioning: soft-embed prev logits (softmax @ embed) THEN the
    gated MLP — the production decoder path, vs the reference condition()."""
    ref, state = _build(2)
    tt = TtSelfConditioning(device, state, hidden_size=HIDDEN, intermediate_size=INTER, eps=EPS)

    emb = torch.randn(1, seq_len, HIDDEN)
    prev_logits = torch.randn(1, seq_len, vocab)
    embed_w = torch.randn(vocab, HIDDEN)

    with torch.no_grad():
        golden = ref.condition(emb, prev_logits, embed_w, enabled=True)  # [1, L, H]

    out = ttnn.to_torch(
        tt.condition(_to_dev(emb, device), _to_dev(prev_logits, device), _embed_to_dev(embed_w, device))
    )[0]
    assert_with_pcc(golden, out, 0.99)


@pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
)
@pytest.mark.use_module_device
@pytest.mark.parametrize("seq_len", [256])
@pytest.mark.parametrize("vocab", [256])
def test_condition_none_logits_is_post_norm(device, seq_len, vocab):
    """prev_logits=None (first step / encoder) -> post_norm(inputs_embeds), device == ref."""
    ref, state = _build(3)
    tt = TtSelfConditioning(device, state, hidden_size=HIDDEN, intermediate_size=INTER, eps=EPS)

    emb = torch.randn(1, seq_len, HIDDEN)
    embed_w = torch.randn(vocab, HIDDEN)
    with torch.no_grad():
        golden = ref.condition(emb, None, embed_w, enabled=False)  # == post_norm(emb)

    out = ttnn.to_torch(tt.condition(_to_dev(emb, device), None, _embed_to_dev(embed_w, device)))[0]
    assert_with_pcc(golden, out, 0.99)
