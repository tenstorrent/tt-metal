# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.tt.self_conditioning import (
    build_self_conditioning_embedding_weight,
    build_self_conditioning,
    TtSelfConditioning,
    _dram_for_rms_norm,
    _rms_norm_dram,
    validate_self_conditioning_state,
)


def _state(hidden_size=8, intermediate_size=6):
    return {
        "pre_norm.weight": torch.ones(hidden_size),
        "gate_proj.weight": torch.ones(intermediate_size, hidden_size),
        "up_proj.weight": torch.ones(intermediate_size, hidden_size),
        "down_proj.weight": torch.ones(hidden_size, intermediate_size),
    }


def test_validate_self_conditioning_state_accepts_expected_shapes():
    validate_self_conditioning_state(_state(), hidden_size=8, intermediate_size=6)


def test_validate_self_conditioning_state_rejects_missing_weight():
    state = _state()
    del state["up_proj.weight"]

    with pytest.raises(ValueError, match="missing self-conditioning weights"):
        validate_self_conditioning_state(state, hidden_size=8, intermediate_size=6)


def test_validate_self_conditioning_state_rejects_wrong_shape():
    state = _state()
    state["down_proj.weight"] = torch.ones(6, 8)

    with pytest.raises(ValueError, match="down_proj.weight has shape"):
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


def test_build_self_conditioning_requires_dimensions_without_config():
    with pytest.raises(ValueError, match="hidden_size and intermediate_size"):
        build_self_conditioning("device", _state(), module_cls=object)


def test_build_self_conditioning_embedding_weight_uses_matmul_layout(monkeypatch):
    calls = {}

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


def test_build_self_conditioning_embedding_weight_rejects_hidden_mismatch():
    with pytest.raises(ValueError, match="embedding hidden size"):
        build_self_conditioning_embedding_weight(
            "device", torch.ones(3, 8), hidden_size=16, tensor_fn=lambda *a, **k: None
        )


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
