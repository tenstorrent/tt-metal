# SPDX-License-Identifier: Apache-2.0
"""Execute setup/dispatch methods with meta host tensors and no TTNN import."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "tt"


def source_class(file, name, methods, namespace):
    tree = ast.parse((ROOT / file).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    node.bases = []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / file), "exec"), namespace)
    return namespace[name]


class Tensor:
    def __init__(self, shape, dtype="bf16", layout="tile"):
        self.shape, self.dtype, self.layout = tuple(shape), dtype, layout


class FakeTT:
    bfloat16, bfloat8_b, bfloat4_b, float32, uint32 = "bf16", "bf8", "bf4", "fp32", "u32"
    TILE_LAYOUT, ROW_MAJOR_LAYOUT = "tile", "rm"
    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = "dram", "l1"
    MathFidelity = SimpleNamespace(HiFi4="hifi4", HiFi2="hifi2", LoFi="lofi")

    def __init__(self):
        self.allocations = []

    def empty(self, shape, *, dtype, layout, **kwargs):
        result = Tensor(shape, dtype, layout)
        self.allocations.append(result)
        return result

    def from_torch(self, value, *, dtype, layout, **kwargs):
        return self.empty(value.shape, dtype=dtype, layout=layout)

    def typecast(self, value, dtype):
        return self.empty(value.shape, dtype=dtype, layout=value.layout)

    def __getattr__(self, name):
        if name in (
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            "CoreRangeSet",
            "CoreRange",
            "CoreCoord",
            "init_device_compute_kernel_config",
            "SDPAProgramConfig",
        ):
            return lambda *args, **kwargs: SimpleNamespace(**kwargs)
        raise AttributeError(name)


class Ops:
    def __init__(self, tt):
        self.tt = tt
        self.mesh = SimpleNamespace(arch=lambda: "test", compute_with_storage_grid_size=lambda: (8, 8))
        self.calls = []

    def upload(self, value, dtype="bf16", layout="tile"):
        return self.tt.from_torch(value, dtype=dtype, layout=layout)

    def shards(self, parts, dtype="bf16", layout="tile"):
        assert len(parts) == 4
        return self.upload(parts[0], dtype, layout)

    def linear(self, x, weight, **kwargs):
        self.calls.append(("linear", x, weight, kwargs))
        return "local-result"

    def scatter_reduce(self, value):
        self.calls.append(("scatter", value))
        return "row-result"


def qsa_state():
    shapes = {
        "q_proj.weight": (12288, 2560),
        "k_proj.weight": (512, 2560),
        "v_proj.weight": (512, 2560),
        "o_proj.weight": (2560, 6144),
        "indexer.index_qk_proj.weight": (640, 2560),
        "q_norm.weight": (256,),
        "k_norm.weight": (256,),
        "indexer.q_layernorm.weight": (128,),
        "indexer.k_layernorm.weight": (128,),
    }
    return {key: torch.empty(shape) for key, shape in shapes.items()}


def qsa_config():
    return SimpleNamespace(
        hidden_size=2560,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        attention_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000},
        rms_norm_eps=1e-6,
        max_position_embeddings=262144,
    )


@pytest.mark.parametrize("allocate", [True, False])
def test_qsa_optional_block_ids_preserves_real_constructor_cache_shapes(allocate):
    tt = FakeTT()
    qsa = source_class("qsa.py", "QSA", ["from_state_dict"], {"ttnn": tt})
    with torch.device("meta"):
        obj = qsa.from_state_dict(
            qsa_state(),
            hf_config=qsa_config(),
            layer_idx=3,
            mesh_device=Ops(tt).mesh,
            batch_size=32,
            max_requests=64,
            allocate_kv_cache=False,
            attention_pool_tokens=279552,
            **({} if allocate else {"allocate_block_ids": False}),
        )
    assert obj.pooled_keys.shape == (1, 1, 65536, 128)
    assert obj.pooled_cache.shape == (1, 1, 69888, 128)
    assert obj.raw_key_cache.shape == (8736, 1, 32, 128)
    block_allocations = [x for x in tt.allocations if x.shape == (1, 1, 1, 65536) and x.dtype == "fp32"]
    assert len(block_allocations) == int(allocate)
    assert (obj.block_ids is not None) == allocate


@pytest.mark.parametrize("allocate", [True, False])
def test_projection_keeps_identical_batched_dispatch_without_column_weight(allocate, expect_error):
    tt, compute = FakeTT(), object()
    ops = Ops(tt)
    cls = source_class(
        "multichip_decoder.py", "AttentionOutputProjection", ["__init__", "__call__"], {"ttnn": tt, "math": math}
    )
    row = object()
    with torch.device("meta"):
        projection = cls(
            torch.empty(2560, 6144), row, ops, compute, **({} if allocate else {"allocate_column_weight": False})
        )
    assert len(tt.allocations) == int(allocate)
    assert (projection.column_weight is not None) == allocate
    for batch in (3, 32):
        for seq in (1, 2, 4, 8):
            value = Tensor((batch, 1, seq, 1536))
            assert projection(value) == "row-result"
            assert ops.calls[-2] == ("linear", value, row, {"compute_kernel_config": compute})
            assert ops.calls[-1] == ("scatter", "local-result")
    if not allocate:
        with expect_error(ValueError, "Singleton projection"):
            projection(Tensor((1, 1, 1, 1536)))


@pytest.mark.parametrize("batch,pool", [(1, None), (32, None), (1, 279552), (32, 279552)])
def test_parallel_qsa_propagates_serving_only_allocation_choices(batch, pool):
    tt, calls = FakeTT(), []
    ops = Ops(tt)

    class Parent:
        @classmethod
        def from_state_dict(cls, state, **kwargs):
            calls.append(kwargs)
            obj = cls()
            obj.paged_pooled, obj.max_context = kwargs["attention_pool_tokens"] is not None, 262144
            obj._physical_update_indices = obj._index_core = object()
            return obj

    def projection(*args, **kwargs):
        calls.append(kwargs)
        return object()

    cls = source_class(
        "multichip_decoder.py",
        "ParallelQSA",
        ["build"],
        {"ttnn": tt, "FusedQSA": Parent, "AttentionOutputProjection": projection},
    )
    cls.cache_dtype = "fp8"
    with torch.device("meta"):
        cls.build(qsa_state(), qsa_config(), ops, 3, batch, 2 * batch, pool)
    assert calls[0]["allocate_block_ids"] is (pool is None)
    assert calls[1]["allocate_column_weight"] is (pool is None or batch == 1)


@pytest.mark.parametrize("batch,pool", [(1, None), (32, None), (1, 279552), (32, 279552)])
def test_decoder_defers_only_serving_preparation_and_generator_buckets_prepare(batch, pool):
    tt, calls, ple_shapes = FakeTT(), [], []
    ops = Ops(tt)

    def gdn(*args, **kwargs):
        calls.append(kwargs)
        return object()

    ple = SimpleNamespace(
        make_cache=lambda n: object(),
        prepare=lambda **kwargs: ple_shapes.append((kwargs["batch"], kwargs["physical_seq_len"])),
    )
    ns = {
        "ttnn": tt,
        "MeshPolicy": lambda: None,
        "MeshOps": lambda *args: ops,
        "ParallelHyperConnection": lambda *args: object(),
        "ParallelMoE": lambda *args: object(),
        "ParallelGatedDeltaNet": gdn,
        "_PrefixView": lambda state, prefix: state,
        "ParallelPLE": lambda *args: ple,
    }
    cls = source_class("multichip_decoder.py", "MultichipDecoder", ["from_state_dict"], ns)
    base = source_class("functional_decoder.py", "FunctionalDecoder", ["prepare"], {"ttnn": tt})
    cls.prepare = base.prepare
    cfg = SimpleNamespace(
        hidden_size=2560,
        hc_count=4,
        num_experts=512,
        max_position_embeddings=262144,
        layer_types=["linear_attention"],
        ple_layer_ids=[1],
    )
    with torch.device("meta"):
        obj = cls.from_state_dict(
            {},
            hf_config=cfg,
            layer_idx=0,
            mesh_device=ops.mesh,
            batch_size=batch,
            chunk_size=128,
            attention_pool_tokens=pool,
        )
        assert set(obj.lengths) == ({1, 128} if pool is None else set())
        assert calls[0]["allocate_column_weight"] is (pool is None or batch == 1)
        obj.prepare([1, 2, 4, 8])
    assert set(obj.lengths) == ({1, 2, 4, 8, 128} if pool is None else {1, 2, 4, 8})
    assert len(ple_shapes) == len(obj.lengths)
    assert ((batch, 128) in ple_shapes) is (pool is None)
