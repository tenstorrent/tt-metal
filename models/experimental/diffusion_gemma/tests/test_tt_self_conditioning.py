# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

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


def _state(hidden_size=8, intermediate_size=6):
    return {
        "pre_norm.weight": torch.ones(hidden_size),
        "gate_proj.weight": torch.ones(intermediate_size, hidden_size),
        "up_proj.weight": torch.ones(intermediate_size, hidden_size),
        "down_proj.weight": torch.ones(hidden_size, intermediate_size),
    }


def test_validate_self_conditioning_state_accepts_expected_shapes():
    validate_self_conditioning_state(_state(), hidden_size=8, intermediate_size=6)


def test_validate_self_conditioning_state_rejects_missing_weight(expect_error):
    state = _state()
    del state["up_proj.weight"]

    with expect_error(ValueError, match="missing self-conditioning weights"):
        validate_self_conditioning_state(state, hidden_size=8, intermediate_size=6)


def test_validate_self_conditioning_state_rejects_wrong_shape(expect_error):
    state = _state()
    state["down_proj.weight"] = torch.ones(6, 8)

    with expect_error(ValueError, match="down_proj.weight has shape"):
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


def test_self_conditioning_logits_l1_defaults_to_chain(monkeypatch):
    monkeypatch.delenv("DG_SELFCOND_LOGITS_L1", raising=False)
    assert self_conditioning_logits_l1_mode() == "chain"


def test_build_self_conditioning_embedding_weight_rejects_hidden_mismatch(expect_error):
    with expect_error(ValueError, match="embedding hidden size"):
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
