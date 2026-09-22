# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for contract-owned target attention arithmetic.

The real attention configuration, RoPE helper, attention forward functions,
and adapter constructors execute with lower device operations stubbed. These
tests establish policy ownership and dispatch, not device numerical parity.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import adapter as _adapter_fixture

adapter = _adapter_fixture


class Tensor:
    def __init__(self, name, shape):
        self.name = name
        self.shape = self.padded_shape = tuple(shape)

    def __getitem__(self, indices):
        indices = list(indices if isinstance(indices, tuple) else (indices,))
        if Ellipsis in indices:
            index = indices.index(Ellipsis)
            indices[index : index + 1] = [slice(None)] * (len(self.shape) - len(indices) + 1)
        indices += [slice(None)] * (len(self.shape) - len(indices))
        shape = [len(range(*index.indices(size))) for size, index in zip(self.shape, indices)]
        return Tensor(self.name + ".slice", shape)

    def memory_config(self):
        return "dram"

    def device(self):
        return SimpleNamespace(get_num_devices=lambda: 8)

    def deallocate(self, force):
        assert force


class DeviceAPI(ModuleType):
    def __init__(self):
        super().__init__("ttnn")
        self.bfloat16 = "bfloat16"
        self.DRAM_MEMORY_CONFIG = "dram"
        self.L1_MEMORY_CONFIG = "l1"
        self.ROW_MAJOR_LAYOUT = "row_major"
        self.TILE_LAYOUT = "tile"
        self.MemoryConfig = object
        self.MathFidelity = SimpleNamespace(HiFi2="hifi2", HiFi4="hifi4")
        self.events = []
        self.configs = []
        self.sdpa_calls = []
        self.updates = []
        self.experimental = SimpleNamespace(
            rotary_embedding=self.rotary,
            paged_update_cache=lambda *args, **kwargs: self.updates.append((args, kwargs)),
            nlp_concat_heads=lambda value, **kwargs: value,
        )
        self.transformer = SimpleNamespace(
            paged_scaled_dot_product_attention_decode=lambda *args, **kwargs: self.sdpa("paged", *args, **kwargs),
            scaled_dot_product_attention_decode=lambda *args, **kwargs: self.sdpa("nonpaged", *args, **kwargs),
        )

    Shape = staticmethod(list)
    CoreCoord = staticmethod(lambda x, y: SimpleNamespace(x=x, y=y))
    PagedCacheGeometryOverride = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))
    to_memory_config = staticmethod(lambda value, *args, **kwargs: value)
    to_layout = staticmethod(lambda value, *args, **kwargs: value)
    deallocate = staticmethod(lambda value: None)
    reshape = staticmethod(lambda value, shape: Tensor(value.name, shape))
    permute = staticmethod(lambda value, order: Tensor(value.name, [value.shape[index] for index in order]))

    def transpose(self, value, first, second):
        order = list(range(len(value.shape)))
        order[first], order[second] = order[second], order[first]
        return self.permute(value, order)

    def repeat(self, value, shape):
        self.events.append(("repeat", value.name, tuple(shape)))
        return Tensor(value.name + ".repeat", [size * count for size, count in zip(value.shape, shape)])

    def neg(self, value):
        self.events.append(("neg", value.name))
        return Tensor(value.name + ".neg", value.shape)

    def concat(self, values, dim):
        self.events.append(("concat", tuple(value.name for value in values), dim))
        shape = list(values[0].shape)
        shape[dim] = sum(value.shape[dim] for value in values)
        return Tensor("rotated", shape)

    def mul(self, left, right, **kwargs):
        self.events.append(("mul", left.name, right.name, kwargs))
        return Tensor(left.name + ".mul", left.shape)

    def add(self, left, right):
        self.events.append(("add", left.name, right.name))
        return Tensor(left.name + ".add", left.shape)

    def rotary(self, value, cos, sin, token_index, **kwargs):
        self.events.append(("rotary", value.name, token_index, kwargs))
        return value

    def SDPAProgramConfig(self, **kwargs):
        result = SimpleNamespace(**{"max_cores_per_head_batch": 16, **kwargs})
        self.configs.append((kwargs, result))
        return result

    def sdpa(self, path, q, k, v, **kwargs):
        self.sdpa_calls.append((path, (q, k, v), kwargs))
        return Tensor("sdpa", q.shape)


@pytest.fixture
def attention(adapter, monkeypatch):
    """Import complete production modules with only lower dependencies replaced."""
    api = DeviceAPI()
    monkeypatch.setitem(sys.modules, "ttnn", api)
    monkeypatch.delenv("GEMMA4_PV_K_CHUNK", raising=False)
    monkeypatch.delenv("GEMMA4_PV_SDPA_HEAD_SPLITS", raising=False)
    root = Path(__file__).resolve().parents[5] / "models/demos/gemma4/tt/attention"
    name = "gemma4_contract_attention_policy_test"
    lower = {
        "models.demos.gemma4.config": {"MeshConfig": object, "Mode": object},
        "models.demos.gemma4.tt.compute_config": {
            "sdpa_fp32_dest_acc_en": lambda default=True: default,
            "sdpa_math_fidelity": lambda default: default,
        },
        name + ".weights": {"AttentionWeights": object, "load_attention_weights": None},
        name + ".kv_cache": {"init_kv_cache": None},
        name + ".prefill": {"flush_deferred_bounded_fills": None, "prefill_forward": None},
    }
    for module_name, members in lower.items():
        module = ModuleType(module_name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, module_name, module)
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    try:
        spec.loader.exec_module(module)
        yield SimpleNamespace(
            module=module,
            operations=sys.modules[name + ".operations"],
            decode=sys.modules[name + ".decode"],
            api=api,
        )
    finally:
        for suffix in ("operations", "decode"):
            sys.modules.pop(name + "." + suffix, None)


def config(attention, dim, selected=False):
    hf = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"],
        hidden_size=5376,
        num_attention_heads=32,
        rms_norm_eps=1e-6,
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        head_dim=256,
        global_head_dim=512,
        sliding_window=1024,
        rope_theta=10000,
        global_rope_theta=1000000,
        partial_rotary_factor=0.25,
    )
    result = attention.module.Gemma4AttentionConfig(hf, int(dim == 512))
    if selected:
        result.decode_rope_fast_and_approximate_mode = True
        result.decode_sdpa_max_cores_per_head_batch = 1
    return result


@pytest.mark.parametrize("heads,dim", [(1, 512), (2, 256), (4, 256), (4, 512)])
def test_rope_selection_changes_only_two_multiply_keywords(attention, heads, dim):
    api = attention.api
    values = (Tensor("input", [1, 32, heads, dim]), Tensor("cos", [1, 32, 1, dim]), Tensor("sin", [1, 32, 1, dim]))
    for selected in (False, True):
        api.events.clear()
        if selected:
            attention.operations.apply_rope_decode_peruser(*values, fast_and_approximate_mode=True)
        else:
            attention.operations.apply_rope_decode_peruser(*values)
        events = list(api.events)
        multiplications = [event for event in events if event[0] == "mul"]
        assert len(multiplications) == 2
        assert [event[-1] for event in multiplications] == (
            [{"fast_and_approximate_mode": True}] * 2 if selected else [{}, {}]
        )
        if selected:
            assert [event[:-1] if event[0] == "mul" else event for event in events] == [
                event[:-1] if event[0] == "mul" else event for event in ordinary
            ]
        else:
            ordinary = events


def prepare_forward(attention, monkeypatch, dim, batch, positions, packed=False):
    kv = 1 if dim == 512 else 2
    qshape = [1, 4, batch * positions, dim] if packed else [1, batch, 4, dim]
    kvshape = [1, kv, batch * positions, dim] if packed else [1, batch, kv, dim]
    q, k, v = Tensor("q", qshape), Tensor("k", kvshape), Tensor("v", kvshape)
    for name, value in {
        "apply_qkv_projection": lambda *args, **kwargs: Tensor("projected", [1, 1, batch * positions, 5376]),
        "split_qkv_heads_decode": lambda *args, **kwargs: (q, k, v),
        "split_qkv_heads_prefill": lambda *args, **kwargs: (q, k, v),
        "apply_per_head_norm": lambda value, *args, **kwargs: value,
        "effective_block_size": lambda *args: 64,
        "concat_heads": lambda value, **kwargs: value,
        "apply_output_projection": lambda value, *args: value,
        "apply_allreduce": lambda value, *args: value,
    }.items():
        monkeypatch.setattr(attention.decode, name, value)
    return dict(
        hidden_states=Tensor("hidden", [1, 1, batch * positions, 5376]),
        cos_cache=Tensor("cos", [1, 1, batch * positions, dim]),
        sin_cache=Tensor("sin", [1, 1, batch * positions, dim]),
        weights=SimpleNamespace(is_global=dim == 512, kv_replicated=kv == 1, q_norm_weight=None, k_norm_weight=None),
        kv_cache=(Tensor("cache-k", [8, kv, 64, dim]), Tensor("cache-v", [8, kv, 64, dim])),
        mesh_config=SimpleNamespace(tp=8),
        mesh_device=SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=12, y=10)),
        position_idx=Tensor("positions", [batch]),
        page_table=Tensor("pages", [batch, 64]),
    )


@pytest.mark.parametrize("dim", [256, 512])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize(
    "batch,shared,paged", [(1, False, True), (32, False, True), (32, True, True), (32, True, False)]
)
def test_ordinary_real_forward_selects_policy_without_changing_rope_route(
    attention, monkeypatch, dim, selected, batch, shared, paged
):
    inputs = prepare_forward(attention, monkeypatch, dim, batch, 1)
    inputs["config"] = config(attention, dim, selected)
    if not paged:
        inputs["page_table"] = None
    attention.module.decode_forward(**inputs, token_index=0, is_kv_shared=shared, rope_presliced=True)
    api = attention.api
    kwargs, pc = api.configs[0]
    assert ("max_cores_per_head_batch" in kwargs) == selected
    assert pc.max_cores_per_head_batch == (1 if selected else 16)
    assert pc.q_chunk_size == 32 and pc.k_chunk_size == 64 and pc.exp_approx_mode is False
    grid = pc.compute_with_storage_grid_size
    assert (grid.x, grid.y) == ((8, 4) if dim == 512 and batch == 1 else (12, 10))
    path, operands, sdpa_kwargs = api.sdpa_calls[0]
    assert path == ("paged" if paged else "nonpaged")
    assert sdpa_kwargs["program_config"] is pc
    assert sdpa_kwargs["cur_pos_tensor"] is inputs["position_idx"]
    assert operands[1:] == inputs["kv_cache"]
    assert len(api.updates) == (0 if shared else 2)
    muls = [event for event in api.events if event[0] == "mul"]
    rotations = [event for event in api.events if event[0] == "rotary"]
    if batch == 1:
        assert not muls and [event[2] for event in rotations] == [0, 0]
    else:
        assert not rotations
        assert len(muls) == (2 if shared else 4)
        assert all(event[-1] == ({"fast_and_approximate_mode": True} if selected else {}) for event in muls)


@pytest.mark.parametrize("dim", [256, 512])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("positions,default_cap", [(6, 16), (24, 8)])
def test_packed_real_forward_passes_selected_or_existing_cap_to_sdpa(
    attention, monkeypatch, dim, selected, positions, default_cap
):
    inputs = prepare_forward(attention, monkeypatch, dim, 1, positions, packed=True)
    inputs["config"] = config(attention, dim, selected)
    mask = Tensor("mask", [1, 1, 4 * positions, 3072])
    attention.module.packed_decode_forward(
        **inputs,
        kv_write_idxs=None,
        attn_mask=mask,
        packed_p=positions,
        is_kv_shared=True,
        rope_packed=(inputs["cos_cache"], inputs["sin_cache"]),
    )
    api = attention.api
    assert len(api.configs) == len(api.sdpa_calls) == 1
    kwargs, pc = api.configs[0]
    assert kwargs["max_cores_per_head_batch"] == (1 if selected else default_cap)
    assert pc.q_chunk_size == 32 and pc.k_chunk_size == 64 and pc.exp_approx_mode is False
    path, operands, sdpa_kwargs = api.sdpa_calls[0]
    assert path == "paged" and operands[0].shape == (1, 1, 4 * positions, dim)
    assert operands[1:] == inputs["kv_cache"]
    assert sdpa_kwargs["program_config"] is pc and sdpa_kwargs["page_table_tensor"] is inputs["page_table"]
    assert sdpa_kwargs["attn_mask"] is mask and sdpa_kwargs["is_causal"] is False
    assert sdpa_kwargs["compute_kernel_config"] is None
    assert sdpa_kwargs["paged_cache_geometry"].num_kv_heads == (1 if dim == 512 else 2)
    assert [event[2] for event in api.events if event[0] == "rotary"] == [None]
    assert not any(event[0] == "mul" for event in api.events)


@pytest.mark.parametrize("maximum", [1, 8, 32])
def test_contract_constructor_owns_policy_before_prepare_and_preserves_buckets(
    adapter, attention, monkeypatch, maximum
):
    monkeypatch.delenv("GEMMA4_DECODE_WARMUP_BATCHES", raising=False)
    monkeypatch.setenv("GEMMA4_DFLASH_WARMUP_DECODE", "1")
    monkeypatch.setenv("GEMMA4_DFLASH_WIDTH_SET", "1")
    targets = lambda: [
        SimpleNamespace(
            bounded_sliding_kv_cache=False,
            layers=[SimpleNamespace(self_attn=SimpleNamespace(config=config(attention, dim))) for dim in (256, 512)],
        )
        for _ in range(2)
    ]
    contract_targets, ordinary_targets, legacy_targets = targets(), targets(), targets()
    events = []

    def policy(models):
        return [
            (
                layer.self_attn.config.decode_rope_fast_and_approximate_mode,
                layer.self_attn.config.decode_sdpa_max_cores_per_head_batch,
            )
            for target in models
            for layer in target.layers
        ]

    def native_init(instance, model, model_args, mesh_device):
        assert policy(model) == [(False, None)] * 4
        instance.model, instance.model_args, instance.mesh_device = model, model_args, mesh_device
        events.append("super_init")

    def native_warmup(instance, **kwargs):
        assert policy(instance.model) == [(True, 1)] * 4
        assert instance._ct_warmup_depth == 1
        events.append(("ordinary", kwargs["enable_trace"], kwargs["max_batch_size"]))

    monkeypatch.setattr(adapter.HybridAttentionForCausalLM, "__init__", native_init)
    monkeypatch.setattr(adapter.HybridAttentionForCausalLM, "warmup_model_decode", native_warmup, raising=False)
    ordinary = adapter.Gemma4ForCausalLM(ordinary_targets, [], object())
    legacy = adapter.Gemma4DFlashForCausalLM(legacy_targets, [], object())
    contract = adapter.Gemma4DFlashContractForCausalLM(contract_targets, [], object())
    assert events == ["super_init", "super_init", "super_init"]
    assert policy(contract.model) == [(True, 1)] * 4
    assert policy(ordinary.model) == [(False, None)] * 4
    assert policy(legacy.model) == [(False, None)] * 4
    assert not contract.model_capabilities["supports_async_decode"]
    assert not contract.model_capabilities["supports_async_spec_decode"]

    def prepare_widths(*args, **kwargs):
        assert policy(contract.model) == [(True, 1)] * 4
        events.append(("widths", kwargs.get("prepare_only", False)))

    contract._spec_get_drafter = lambda: events.append("drafter")
    contract._spec_capture_width_set = prepare_widths
    events.clear()
    for traced in (False, True):
        contract.warmup_model_decode(
            kv_cache=object(), enable_trace=traced, max_batch_size=maximum, num_blocks=64, can_sample_on_device=True
        )
    buckets = sorted({1, maximum})
    assert contract.tt_supported_decode_batch_sizes == tuple(buckets)
    assert events == (
        [("ordinary", False, batch) for batch in buckets]
        + ["drafter", ("widths", True)]
        + [("ordinary", True, batch) for batch in buckets]
        + [("widths", False)]
    )
    assert policy(ordinary.model) == [(False, None)] * 4
    assert policy(legacy.model) == [(False, None)] * 4
