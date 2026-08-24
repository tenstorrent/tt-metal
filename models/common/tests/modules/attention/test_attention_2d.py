# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only contract tests for Attention2D; no TT device is opened."""

import inspect
from dataclasses import FrozenInstanceError, replace
from enum import Enum
from types import SimpleNamespace

import pytest

from models.common.modules.attention import attention_2d
from models.common.modules.attention.attention_2d import (
    Attention2D,
    Attention2DConfig,
    Attention2DLowLevelCallables,
    Attention2DSequenceConfig,
    DecodeMetadata,
    KVCacheBinding,
    PagedKVMetadata,
    PrefillAttentionMode,
    PrefillCollectiveMode,
    PrefillMetadata,
    PrefillRecipeIdentity,
    PrefillRowMode,
    resolve_attention2d_config,
)
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_2d import RMSNorm2DGeometry


class _Source:
    def __init__(self, shape):
        self.shape = shape


class _Arch(Enum):
    WORMHOLE_B0 = 1


class _Mesh:
    def __init__(self, shape=(8, 4), devices=32, arch=_Arch.WORMHOLE_B0):
        self.shape = shape
        self._devices = devices
        self._arch = arch

    def get_num_devices(self):
        return self._devices

    def arch(self):
        return self._arch


class _Tensor:
    def __init__(self, name, shape=(1, 1, 32, 128), dtype="act", placement="temp"):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self._placement = placement

    def memory_config(self):
        return self._placement

    def __getitem__(self, item):
        return _Tensor(f"{self.name}-view", dtype=self.dtype, placement=self._placement)

    def __repr__(self):
        return self.name


def _weight(shape, mesh, value, mapper, dtype):
    return LazyWeight(
        source=_Source(shape),
        device=mesh,
        mesh_mapper_config=mapper,
        memory_config="weight-mem",
        layout="tile",
        dtype=dtype,
        _value=value,
    )


def _identity(
    row=PrefillRowMode.SINGLE_ROW,
    collective=PrefillCollectiveMode.REGULAR,
    attention=PrefillAttentionMode.REGULAR,
    length=128,
):
    return PrefillRecipeIdentity(length, row, collective, attention)


def _sequence(identity=None):
    identity = identity or _identity()
    suffix = "/".join((identity.row_mode.value, identity.collective_mode.value, identity.attention_mode.value))
    return Attention2DSequenceConfig(
        identity=identity,
        qkv_program_config=f"qkv:{suffix}",
        sdpa_program_config=f"sdpa:{suffix}",
        wo_program_config=f"wo:{suffix}",
        qkv_output_memory_config="prefill-qkv",
        heads_memory_config="prefill-heads",
        kv_memory_config="prefill-kv",
        sdpa_output_memory_config="prefill-sdpa",
        concat_memory_config="prefill-concat",
        wo_output_memory_config="prefill-projected",
        qkv_kernel_config="prefill-qkv-kernel",
        sdpa_kernel_config="prefill-sdpa-kernel",
        wo_kernel_config="prefill-wo-kernel",
        activation_dtype="act",
    )


def _all_recipes():
    identities = (
        PrefillRecipeIdentity(128, row, collective, attention)
        for row in PrefillRowMode
        for collective in PrefillCollectiveMode
        for attention in PrefillAttentionMode
    )
    return {identity: _sequence(identity) for identity in identities}


def _low_level(events, *, fail_at=None):
    def rotary(q, k, rot_mats, **kwargs):
        events.append(("rotary", kwargs["mode"], kwargs.get("recipe")))
        return _Tensor("rot-q"), _Tensor("rot-k")

    def reduce_qkv(tensor, **kwargs):
        events.append(("reduce-qkv", kwargs["mode"], kwargs.get("recipe")))
        return _Tensor("reduced-qkv")

    def gather_heads(tensor, **kwargs):
        events.append(("gather-heads", kwargs["mode"], kwargs.get("recipe"), kwargs["prefix_bounds"]))
        if fail_at == "gather-heads":
            raise RuntimeError("injected gather failure")
        return _Tensor("gathered")

    def reduce_output(tensor, **kwargs):
        events.append(("reduce-output", kwargs["mode"], kwargs.get("recipe")))
        placement = "decode-out" if kwargs["mode"] == "decode" else "prefill-out"
        return _Tensor("output", dtype="act", placement=placement)

    return Attention2DLowLevelCallables(rotary, reduce_qkv, gather_heads, reduce_output)


def _config(*, events=None, mesh=None, bias=False, releases=None, runtime_releases=None, **overrides):
    events = [] if events is None else events
    releases = [] if releases is None else releases
    runtime_releases = [] if runtime_releases is None else runtime_releases
    mesh = mesh or _Mesh()
    wqkv_mapper, wo_mapper, bias_mapper = "wqkv-map", "wo-map", "bias-map"
    values = dict(
        wqkv=_weight((5120, 10240), mesh, _Tensor("WQKV"), wqkv_mapper, "wqkv-dtype"),
        # Qwen3-32B geometry: head_dim is decoupled from the hidden size, so WO
        # reduces n_heads * head_dim (8192) back to dim (5120).
        wo=_weight((8192, 5120), mesh, _Tensor("WO"), wo_mapper, "wo-dtype"),
        wqkv_bias=_weight((10240,), mesh, _Tensor("BIAS"), bias_mapper, "bias-dtype") if bias else None,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        max_batch_size=32,
        max_seq_len=2048,
        low_level=_low_level(events),
        runtime_tensor_factory=lambda offsets, lower, upper, device: (
            _Tensor("offsets", shape=(4,), dtype="uint32"),
            _Tensor("lower", shape=(4,), dtype="uint32"),
            _Tensor("upper", shape=(4,), dtype="uint32"),
        ),
        runtime_tensor_releaser=runtime_releases.append,
        intermediate_releaser=releases.append,
        mesh_device=mesh,
        wqkv_mesh_mapper_config=wqkv_mapper,
        wo_mesh_mapper_config=wo_mapper,
        bias_mesh_mapper_config=bias_mapper if bias else None,
        weight_memory_config="weight-mem",
        weight_layout="tile",
        wqkv_dtype="wqkv-dtype",
        wo_dtype="wo-dtype",
        bias_dtype="bias-dtype" if bias else None,
        decode_input_placement="decode-in",
        decode_output_placement="decode-out",
        prefill_input_placement="prefill-in",
        prefill_output_placement="prefill-out",
        decode_qkv_output_memory_config="decode-qkv",
        decode_heads_memory_config="decode-heads",
        decode_kv_memory_config="decode-kv",
        decode_sdpa_output_memory_config="decode-sdpa",
        decode_concat_memory_config="decode-concat",
        decode_wo_output_memory_config="decode-projected",
        decode_program_config="decode-qkv-program",
        decode_sdpa_program_config="decode-sdpa-program",
        decode_wo_program_config="decode-wo-program",
        decode_qkv_kernel_config="decode-qkv-kernel",
        decode_sdpa_kernel_config="decode-sdpa-kernel",
        decode_wo_kernel_config="decode-wo-kernel",
        decode_activation_dtype="act",
        prefill_sequence_configs=_all_recipes(),
    )
    values.update(overrides)
    return Attention2DConfig(**values)


def _paged_binding(model, **overrides):
    values = dict(block_size=32, max_num_blocks=64, cache_dtype="cache", page_table_dtype="uint32")
    values.update(overrides.pop("metadata", {}))
    metadata = PagedKVMetadata(**values)
    shape = overrides.pop("shape", (metadata.max_num_blocks, 1, metadata.block_size, 128))
    dtype = overrides.pop("dtype", "cache")
    return KVCacheBinding(
        _Tensor("keys", shape=shape, dtype=dtype, placement="cache"),
        _Tensor(
            "values",
            shape=overrides.pop("value_shape", shape),
            dtype=overrides.pop("value_dtype", dtype),
            placement="cache",
        ),
        overrides.pop("owner", object()),
        metadata,
        overrides.pop("mesh_device", model.config.mesh_device),
    )


def _page_table(rows=32, columns=64, dtype="uint32"):
    shape = tuple(rows) if isinstance(rows, tuple) else (rows, columns)
    return _Tensor("page-table", shape=shape, dtype=dtype, placement="dram")


@pytest.fixture
def host_ttnn(monkeypatch):
    events = []

    def linear(x, weight, **kwargs):
        stage = "qkv" if weight.name == "WQKV" else "wo"
        events.append((stage, kwargs))
        return _Tensor(stage, dtype=kwargs["dtype"], placement=kwargs["memory_config"])

    def heads_decode(x, **kwargs):
        events.append(("heads-decode", kwargs))
        return _Tensor("q"), _Tensor("k"), _Tensor("v")

    def heads_prefill(x, **kwargs):
        events.append(("heads-prefill", kwargs))
        return _Tensor("q"), _Tensor("k"), _Tensor("v")

    def rotary_result(name):
        def call(*args, **kwargs):
            events.append((name, kwargs))
            placement = kwargs.get("memory_config", "temp")
            return _Tensor(name, placement=placement)

        return call

    monkeypatch.setattr(attention_2d.ttnn, "linear", linear)

    def to_memory_config(tensor, memory_config, dtype=None):
        events.append(("to-memory", {"memory_config": memory_config, "dtype": dtype}))
        return _Tensor(f"placed-{tensor.name}", dtype=dtype or tensor.dtype, placement=memory_config)

    monkeypatch.setattr(attention_2d.ttnn, "to_memory_config", to_memory_config)

    def typecast(tensor, dtype):
        events.append(("typecast", {"dtype": dtype}))
        return _Tensor(f"cast-{tensor.name}", dtype=dtype, placement=tensor.memory_config())

    monkeypatch.setattr(attention_2d.ttnn, "typecast", typecast)
    monkeypatch.setattr(attention_2d.ttnn.experimental, "nlp_create_qkv_heads_decode", heads_decode)
    monkeypatch.setattr(attention_2d.ttnn.experimental, "nlp_create_qkv_heads", heads_prefill)
    for name in ("paged_update_cache", "paged_fill_cache"):
        monkeypatch.setattr(
            attention_2d.ttnn.experimental,
            name,
            lambda *args, _name=name, **kwargs: events.append((_name, kwargs)),
            raising=False,
        )
    monkeypatch.setattr(
        attention_2d.ttnn, "update_cache", lambda *args, **kwargs: events.append(("update_cache", kwargs))
    )
    monkeypatch.setattr(attention_2d.ttnn, "fill_cache", lambda *args, **kwargs: events.append(("fill_cache", kwargs)))
    monkeypatch.setattr(
        attention_2d.ttnn.transformer,
        "paged_scaled_dot_product_attention_decode",
        rotary_result("sdpa-decode-paged"),
    )
    monkeypatch.setattr(
        attention_2d.ttnn.transformer,
        "scaled_dot_product_attention_decode",
        rotary_result("sdpa-decode"),
    )
    monkeypatch.setattr(
        attention_2d.ttnn.transformer,
        "scaled_dot_product_attention",
        rotary_result("sdpa-prefill"),
    )
    monkeypatch.setattr(
        attention_2d.ttnn.transformer,
        "chunked_scaled_dot_product_attention",
        rotary_result("sdpa-chunked"),
    )
    monkeypatch.setattr(
        attention_2d.ttnn.experimental,
        "nlp_concat_heads_decode",
        rotary_result("concat-decode"),
    )
    monkeypatch.setattr(attention_2d.ttnn.experimental, "nlp_concat_heads", rotary_result("concat-prefill"))
    return events


def test_resolves_geometry_and_normalizes_architecture_enum():
    config = resolve_attention2d_config(_config(architecture=_Arch.WORMHOLE_B0))
    assert (config.dim, config.qkv_size, config.architecture) == (5120, 10240, "wormhole")
    assert config.scale == pytest.approx(128**-0.5)


def test_config_and_identity_recipe_mapping_are_frozen():
    config = resolve_attention2d_config(_config())
    with pytest.raises(FrozenInstanceError):
        config.max_seq_len = 4096
    with pytest.raises(TypeError):
        config.prefill_sequence_configs[_identity()] = _sequence()
    assert config.sequence_config(_identity()).identity == _identity()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"mesh_device": _Mesh(shape=(4, 8))}, "mesh shape"),
        ({"mesh_device": _Mesh(devices=31)}, "32 devices"),
        ({"mesh_device": _Mesh(arch="blackhole")}, "Wormhole only"),
        ({"users_per_column": 4}, "max_batch_size=32"),
        ({"max_batch_size": 16}, "max_batch_size=32"),
        ({"n_heads": 62}, "divisible"),
        ({"prefill_sequence_configs": {}}, "identity-keyed"),
        ({"runtime_tensor_factory": None}, "runtime_tensor_factory"),
        ({"decode_program_config": None}, "policy is incomplete"),
        ({"decode_input_placement": None}, "policy is incomplete"),
    ],
)
def test_static_policy_fails_closed(change, message):
    base = _config()
    if "mesh_device" in change:
        mesh = change["mesh_device"]
        change = {
            **change,
            "wqkv": replace(base.wqkv, device=mesh),
            "wo": replace(base.wo, device=mesh),
        }
    with pytest.raises((TypeError, ValueError), match=message):
        resolve_attention2d_config(replace(base, **change))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("wqkv_mesh_mapper_config", "wrong-map"),
        ("weight_memory_config", "wrong-memory"),
        ("weight_layout", "row-major"),
        ("wqkv_dtype", "wrong-dtype"),
    ],
)
def test_weight_placement_must_match_exactly(field, replacement):
    with pytest.raises(ValueError, match="exactly match"):
        resolve_attention2d_config(replace(_config(), **{field: replacement}))


def test_weight_shapes_and_atomic_qk_norm_fail_closed():
    config = _config()
    with pytest.raises(ValueError, match="wqkv source shape"):
        resolve_attention2d_config(replace(config, wqkv=replace(config.wqkv, source=_Source((5120, 999)))))
    with pytest.raises(ValueError, match="wo source shape"):
        resolve_attention2d_config(replace(config, wo=replace(config.wo, source=_Source((5120, 2560)))))
    with pytest.raises(ValueError, match="supplied together"):
        resolve_attention2d_config(replace(config, q_norm_config=object()))


def test_wo_shape_follows_the_attention_projection_width():
    config = resolve_attention2d_config(_config())

    assert tuple(config.wo.source.shape) == (8192, 5120)
    with pytest.raises(ValueError, match="wo source shape must be"):
        resolve_attention2d_config(replace(config, wo=replace(config.wo, source=_Source((5120, 5120)))))


def test_square_wo_resolves_when_attention_width_equals_hidden_size():
    mesh = _Mesh()
    resolved = resolve_attention2d_config(
        _config(
            mesh=mesh,
            wqkv=_weight((8192, 10240), mesh, _Tensor("WQKV"), "wqkv-map", "wqkv-dtype"),
            wo=_weight((8192, 8192), mesh, _Tensor("WO"), "wo-map", "wo-dtype"),
        )
    )

    assert (resolved.dim, resolved.qkv_size) == (8192, 10240)


def test_qk_norm_requires_explicit_head_local_geometry():
    config = _config()
    norm = SimpleNamespace(
        weight=_weight((128,), config.mesh_device, _Tensor("norm"), "norm-map", "norm-dtype"),
        geometry=RMSNorm2DGeometry.DISTRIBUTED,
    )
    with pytest.raises(ValueError, match="head-local"):
        resolve_attention2d_config(replace(config, q_norm_config=norm, k_norm_config=norm))


@pytest.mark.parametrize(
    ("field", "context", "message"),
    [
        ("decode_prefetch_context", SimpleNamespace(mesh_device=_Mesh()), "different mesh"),
        ("prefill_prefetch_context", SimpleNamespace(mode="decode"), "incompatible mode"),
    ],
)
def test_prefetch_context_selectors_fail_closed(field, context, message):
    with pytest.raises(ValueError, match=message):
        resolve_attention2d_config(replace(_config(), **{field: context}))


def test_recipe_key_and_recipe_policy_fail_closed():
    identity = _identity()
    wrong = replace(_sequence(identity), identity=replace(identity, collective_mode=PrefillCollectiveMode.RING))
    with pytest.raises(ValueError, match="exactly match"):
        resolve_attention2d_config(replace(_config(), prefill_sequence_configs={identity: wrong}))
    with pytest.raises(ValueError, match="policy is incomplete"):
        replace(_sequence(), qkv_program_config=None)


@pytest.mark.parametrize(
    ("binding_kwargs", "message"),
    [
        ({"shape": (63, 1, 32, 128), "metadata": {"max_num_blocks": 63}}, "capacity"),
        ({"shape": (64, 2, 32, 128)}, "paged KV cache shape"),
        ({"value_shape": (64, 1, 64, 128)}, "identical rank-4"),
        ({"dtype": "wrong"}, "dtype does not match metadata"),
        ({"value_dtype": "other"}, "dtypes must match"),
    ],
)
def test_kv_cache_shape_dtype_and_capacity_are_validated(binding_kwargs, message):
    model = Attention2D.from_config(_config())
    with pytest.raises(ValueError, match=message):
        model.bind_kv_cache(_paged_binding(model, **binding_kwargs))


def test_contiguous_cache_shape_and_foreign_mesh_are_rejected():
    model = Attention2D.from_config(_config())
    binding = KVCacheBinding(_Tensor("k", shape=(8, 1, 1024, 128)), _Tensor("v", shape=(8, 1, 1024, 128)), object())
    with pytest.raises(ValueError, match="contiguous KV cache shape"):
        model.bind_kv_cache(binding)
    with pytest.raises(ValueError, match="different mesh"):
        model.bind_kv_cache(_paged_binding(model, mesh_device=_Mesh()))


@pytest.mark.parametrize(
    ("table", "message"),
    [
        # Decode attends to one mesh column's users on each device, so the
        # device-local table carries users_per_column rows (or that batch once
        # per core). A 31-row table addresses neither.
        (_page_table(rows=31), "device-local rows"),
        (_page_table(rows=4), "device-local rows"),
        (_page_table(rows=(32, 64)), "rank-2"),
        (_page_table(columns=63), "capacity"),
        (_page_table(dtype="int16"), "dtype"),
    ],
)
def test_decode_page_table_contract_fails_before_compute(host_ttnn, table, message):
    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    with pytest.raises(ValueError, match=message):
        model.decode_forward(
            _Tensor("x", dtype="act", placement="decode-in"),
            "rot",
            DecodeMetadata(_Tensor("positions", shape=(32,), dtype="uint32"), table),
        )
    assert host_ttnn == []


@pytest.mark.parametrize("rows", [8, 16, 32])
def test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats(host_ttnn, rows):
    """8 rows is the interleaved table; a multiple is the L1-sharded repeat."""

    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    result = model.decode_forward(
        _Tensor("x", dtype="act", placement="decode-in"),
        "rot",
        DecodeMetadata(_Tensor("positions", shape=(32,), dtype="uint32"), _page_table(rows=rows)),
    )

    assert result.name == "output"
    assert "sdpa-decode-paged" in [event[0] for event in host_ttnn]


def test_prefill_page_table_must_reach_every_filled_user(host_ttnn):
    """``paged_fill_cache`` indexes the table by user, unlike decode."""

    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    with pytest.raises(ValueError, match="one row for every addressed user"):
        model.prefill_forward(
            _Tensor("x", dtype="act", placement="prefill-in"),
            "rot",
            PrefillMetadata(128, (31,), page_table=_page_table(rows=31)),
        )
    assert host_ttnn == []


def test_decode_direct_ttnn_recipe_is_straight_line_and_owns_stages(host_ttnn):
    low_events, releases = [], []
    model = Attention2D.from_config(_config(events=low_events, releases=releases, bias=True))
    model.bind_kv_cache(_paged_binding(model))
    result = model.decode_forward(
        _Tensor("x", dtype="act", placement="decode-in"),
        "rot",
        DecodeMetadata(_Tensor("positions", shape=(32,), dtype="uint32"), _page_table()),
    )

    assert result.name == "output"
    assert [event[0] for event in host_ttnn] == [
        "qkv",
        "heads-decode",
        "to-memory",
        "to-memory",
        "paged_update_cache",
        "paged_update_cache",
        "sdpa-decode-paged",
        "concat-decode",
        "wo",
    ]
    assert [event[0] for event in low_events] == ["reduce-qkv", "rotary", "gather-heads", "reduce-output"]
    assert host_ttnn[0][1]["bias"].name == "BIAS"
    lower, upper = low_events[2][-1]
    assert (lower.name, upper.name) == ("lower", "upper")
    assert len(releases) == len({id(tensor) for tensor in releases})
    assert result not in releases


def test_decode_uses_optional_fused_reduce_create_heads_hook(host_ttnn):
    low_events, releases = [], []
    base = _config(events=low_events, releases=releases)

    def fused(qkv, **kwargs):
        low_events.append(("fused-qkv-heads", qkv.name, kwargs["mode"]))
        return _Tensor("fused-q"), _Tensor("fused-k"), _Tensor("fused-v")

    low_level = replace(base.low_level, reduce_create_qkv_heads=fused)
    model = Attention2D.from_config(replace(base, low_level=low_level))
    model.bind_kv_cache(_paged_binding(model))

    model.decode_forward(
        _Tensor("x", dtype="act", placement="decode-in"),
        "rot",
        DecodeMetadata(_Tensor("positions", shape=(32,), dtype="uint32"), _page_table()),
    )

    assert "heads-decode" not in [event[0] for event in host_ttnn]
    assert "reduce-qkv" not in [event[0] for event in low_events]
    assert low_events[0] == ("fused-qkv-heads", "qkv", "decode")
    assert any(tensor.name == "qkv" for tensor in releases)


def test_decode_optionally_gathers_column_local_users_before_concat(host_ttnn):
    low_events, releases = [], []
    base = _config(events=low_events, releases=releases)

    def gather_users(attention, **kwargs):
        low_events.append(("gather-users", attention.name, kwargs["mode"]))
        return _Tensor("gathered-users")

    low_level = replace(base.low_level, gather_users=gather_users)
    model = Attention2D.from_config(replace(base, low_level=low_level))
    model.bind_kv_cache(_paged_binding(model))
    model.decode_forward(
        _Tensor("x", dtype="act", placement="decode-in"),
        "rot",
        DecodeMetadata(_Tensor("positions", shape=(32,), dtype="uint32"), _page_table()),
    )

    assert ("gather-users", "sdpa-decode-paged", "decode") in low_events
    assert any(tensor.name == "sdpa-decode-paged" for tensor in releases)


def test_decode_prefetch_context_is_passed_to_both_projections(host_ttnn):
    context = SimpleNamespace(mode="decode", global_cb="decode-cb", worker_sub_device_id="decode-worker")
    model = Attention2D.from_config(_config(decode_prefetch_context=context))
    model.bind_kv_cache(_paged_binding(model))
    model.decode_forward(
        _Tensor("x", dtype="act", placement="decode-in"),
        "rot",
        DecodeMetadata(_Tensor("positions", shape=(32,), dtype="uint32"), _page_table()),
    )
    projections = [kwargs for stage, kwargs in host_ttnn if stage in {"qkv", "wo"}]
    assert [(kwargs["global_cb"], kwargs["sub_device_id"]) for kwargs in projections] == [
        ("decode-cb", "decode-worker"),
        ("decode-cb", "decode-worker"),
    ]


def test_decode_contiguous_cache_selects_tensor_indexed_update_and_nonpaged_sdpa(host_ttnn):
    model = Attention2D.from_config(_config())
    shape = (8, 1, 2048, 128)
    model.bind_kv_cache(
        KVCacheBinding(
            _Tensor("keys", shape=shape, dtype="cache", placement="cache"),
            _Tensor("values", shape=shape, dtype="cache", placement="cache"),
            object(),
            mesh_device=model.config.mesh_device,
        )
    )
    positions = _Tensor("positions", shape=(32,), dtype="uint32")
    model.decode_forward(
        _Tensor("x", dtype="act", placement="decode-in"),
        "rot",
        DecodeMetadata(positions),
    )
    stages = [stage for stage, _ in host_ttnn]
    assert stages.count("paged_update_cache") == 2
    assert "sdpa-decode" in stages
    assert "sdpa-decode-paged" not in stages
    update_kwargs = [kwargs for stage, kwargs in host_ttnn if stage == "paged_update_cache"]
    assert [kwargs["update_idxs_tensor"] for kwargs in update_kwargs] == [positions, positions]
    assert all(set(kwargs) == {"update_idxs_tensor"} for kwargs in update_kwargs)


def test_qk_norm_runs_head_local_decode_and_prefill_paths(monkeypatch, host_ttnn):
    norm_events = []

    class FakeNorm:
        def __init__(self, name):
            self.name = name

        def decode_forward(self, value):
            norm_events.append(f"{self.name}-decode")
            return _Tensor(f"{self.name}-decode")

        def prefill_forward(self, value):
            norm_events.append(f"{self.name}-prefill")
            return _Tensor(f"{self.name}-prefill")

    norms = iter((FakeNorm("q"), FakeNorm("k")))
    monkeypatch.setattr(attention_2d.RMSNorm2D, "from_config", classmethod(lambda cls, config: next(norms)))
    base = _config()
    norm_config = SimpleNamespace(
        weight=_weight((128,), base.mesh_device, _Tensor("norm"), "norm-map", "norm-dtype"),
        geometry=RMSNorm2DGeometry.HEAD_LOCAL,
    )
    model = Attention2D.from_config(replace(base, q_norm_config=norm_config, k_norm_config=norm_config))
    model.bind_kv_cache(_paged_binding(model))

    model.decode_forward(
        _Tensor("decode-x", dtype="act", placement="decode-in"),
        "rot",
        DecodeMetadata(_Tensor("positions", shape=(32,), dtype="uint32"), _page_table()),
    )
    model.prefill_forward(
        _Tensor("prefill-x", dtype="act", placement="prefill-in"),
        "rot",
        PrefillMetadata(128, page_table=_page_table()),
    )
    assert norm_events == ["q-decode", "k-decode", "q-prefill", "k-prefill"]


@pytest.mark.parametrize("row_mode", list(PrefillRowMode))
@pytest.mark.parametrize("collective_mode", list(PrefillCollectiveMode))
@pytest.mark.parametrize("attention_mode", list(PrefillAttentionMode))
def test_prefill_selects_every_composite_recipe(host_ttnn, row_mode, collective_mode, attention_mode):
    low_events = []
    model = Attention2D.from_config(_config(events=low_events))
    model.bind_kv_cache(_paged_binding(model))
    users = (0,) if row_mode is PrefillRowMode.SINGLE_ROW else tuple(range(32))
    kwargs = {}
    if attention_mode is PrefillAttentionMode.PREFIX_CHUNKED:
        kwargs = {"prefix_user_id": users[0], "chunk_start": 128, "chunk_page_table": _page_table()}
    metadata = PrefillMetadata(
        128,
        users,
        collective_mode,
        page_table=_page_table(),
        **kwargs,
    )
    result = model.prefill_forward(_Tensor("x", dtype="act", placement="prefill-in"), "rot", metadata)
    expected = PrefillRecipeIdentity(128, row_mode, collective_mode, attention_mode)

    assert result.name == "output"
    selected = [event[2] for event in low_events if event[0] == "reduce-qkv"]
    assert selected == [expected]
    sdpa_name = "sdpa-chunked" if attention_mode is PrefillAttentionMode.PREFIX_CHUNKED else "sdpa-prefill"
    assert sdpa_name in [event[0] for event in host_ttnn]


def test_prefill_selects_2048_recipe_and_passes_prefetch_context(host_ttnn):
    identity = _identity(length=2048)
    context = SimpleNamespace(mode="prefill", global_cb="prefill-cb", worker_sub_device_id="prefill-worker")
    model = Attention2D.from_config(
        _config(prefill_sequence_configs={identity: _sequence(identity)}, prefill_prefetch_context=context)
    )
    model.bind_kv_cache(_paged_binding(model))
    model.prefill_forward(
        _Tensor("x", dtype="act", placement="prefill-in"),
        "rot",
        PrefillMetadata(2048, page_table=_page_table()),
    )
    projections = [kwargs for stage, kwargs in host_ttnn if stage in {"qkv", "wo"}]
    assert [kwargs["program_config"] for kwargs in projections] == [
        _sequence(identity).qkv_program_config,
        _sequence(identity).wo_program_config,
    ]
    assert [(kwargs["global_cb"], kwargs["sub_device_id"]) for kwargs in projections] == [
        ("prefill-cb", "prefill-worker"),
        ("prefill-cb", "prefill-worker"),
    ]


def test_concat32_contiguous_cache_fill_slices_each_physical_row(monkeypatch):
    fills = []
    monkeypatch.setattr(attention_2d.ttnn, "fill_cache", lambda cache, value, user: fills.append((cache, value, user)))
    model = Attention2D.from_config(_config())
    shape = (32, 1, 2048, 128)
    binding = KVCacheBinding(
        _Tensor("keys", shape=shape, dtype="cache", placement="cache"),
        _Tensor("values", shape=shape, dtype="cache", placement="cache"),
        object(),
        mesh_device=model.config.mesh_device,
    )
    metadata = PrefillMetadata(128, tuple(range(32)))
    model._fill_prefill_cache(binding, _Tensor("k"), _Tensor("v"), metadata)

    assert [user for _, _, user in fills] == [user for user in range(32) for _ in range(2)]
    assert all(value.name.endswith("-view") for _, value, _ in fills)
    assert model._intermediates == {}


def test_concat32_paged_cache_maps_source_rows_to_ordered_users(monkeypatch):
    slices, fills = [], []

    class TrackedTensor(_Tensor):
        def __getitem__(self, item):
            slices.append(item[0].start)
            return super().__getitem__(item)

    monkeypatch.setattr(
        attention_2d.ttnn.experimental,
        "paged_fill_cache",
        lambda cache, value, table, **kwargs: fills.append((value, table, kwargs["batch_idx"])),
        raising=False,
    )
    model = Attention2D.from_config(_config())
    binding = _paged_binding(model)
    users = (1, 0, *range(2, 32))
    model._fill_prefill_cache(
        binding,
        TrackedTensor("k"),
        TrackedTensor("v"),
        PrefillMetadata(128, users, page_table=TrackedTensor("table", shape=(32, 64), dtype="uint32")),
    )

    assert slices[:6] == [0, 0, 1, 1, 1, 0]
    assert all(batch_idx == 0 for _, _, batch_idx in fills)
    assert model._intermediates == {}


def test_chunked_sdpa_reads_only_the_addressed_users_page_table_row(host_ttnn):
    """``paged_fill_cache`` indexes the table by user; chunked SDPA cannot.

    Chunked SDPA requires the table's leading dimension to equal Q's batch,
    which is one for a single-row prefill, so the module slices the addressed
    row out of the same table the fill indexed.
    """

    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    model.prefill_forward(
        _Tensor("x", dtype="act", placement="prefill-in"),
        "rot",
        PrefillMetadata(
            128,
            (5,),
            page_table=_page_table(),
            chunk_page_table=_page_table(),
            chunk_start=128,
            prefix_user_id=5,
        ),
    )

    tables = [kwargs["page_table_tensor"] for stage, kwargs in host_ttnn if stage == "sdpa-chunked"]
    assert [table.name for table in tables] == ["page-table-view"]
    assert model._intermediates == {}


def test_concat32_chunked_sdpa_keeps_the_full_page_table():
    """A concatenated prefill's Q batch already matches the 32-row table."""

    model = Attention2D.from_config(_config())
    table = _page_table()
    metadata = PrefillMetadata(128, tuple(range(32)), page_table=table, chunk_start=128, prefix_user_id=0)

    assert model._sdpa_page_table(metadata) is table


@pytest.mark.parametrize(
    "metadata",
    [
        PrefillMetadata(128, (1, 1), page_table=_page_table()),
        PrefillMetadata(128, (32,), page_table=_page_table()),
        PrefillMetadata(128, (0, 1), page_table=_page_table()),
        PrefillMetadata(128, (0,), page_table=_page_table(), prefix_user_id=1),
        PrefillMetadata(128, (0,), page_table=_page_table(), chunk_start=1),
        PrefillMetadata(128, (0,), page_table=_page_table(), chunk_start=128, chunk_start_tensor=_Tensor("start")),
    ],
)
def test_prefill_invocation_policy_fails_before_compute(host_ttnn, metadata):
    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    with pytest.raises(ValueError):
        model.prefill_forward(_Tensor("x", dtype="act", placement="prefill-in"), "rot", metadata)
    assert host_ttnn == []


@pytest.mark.parametrize(
    ("mode", "tensor", "message"),
    [
        ("decode", _Tensor("x", dtype="act", placement="wrong"), "placement"),
        ("decode", _Tensor("x", dtype="wrong", placement="decode-in"), "dtype"),
        ("prefill", _Tensor("x", dtype="act", placement="wrong"), "placement"),
        ("prefill", _Tensor("x", dtype="wrong", placement="prefill-in"), "dtype"),
    ],
)
def test_activation_placement_and_dtype_fail_before_compute(host_ttnn, mode, tensor, message):
    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged_binding(model))
    if mode == "decode":
        call = lambda: model.decode_forward(
            tensor, "rot", DecodeMetadata(_Tensor("positions", shape=(32,)), _page_table())
        )
    else:
        call = lambda: model.prefill_forward(tensor, "rot", PrefillMetadata(128, page_table=_page_table()))
    with pytest.raises(ValueError, match=message):
        call()
    assert host_ttnn == []


def test_runtime_tensors_are_factory_owned_and_released_once():
    created, released = [], []

    def factory(offsets, lower, upper, mesh):
        created.append((offsets, lower, upper, mesh))
        return _Tensor("offsets"), _Tensor("lower"), _Tensor("upper")

    model = Attention2D.from_config(_config(runtime_tensor_factory=factory, runtime_tensor_releaser=released.append))
    first = model._ensure_runtime_tensors()
    assert model._ensure_runtime_tensors() is first
    model.close()
    model.close()
    assert len(created) == 1
    assert released == list(first)


def test_intermediates_are_drained_when_a_stage_raises(host_ttnn):
    events, released = [], []
    config = _config(events=events, releases=released)
    model = Attention2D.from_config(replace(config, low_level=_low_level(events, fail_at="gather-heads")))
    model.bind_kv_cache(_paged_binding(model))
    with pytest.raises(RuntimeError, match="injected gather"):
        model.decode_forward(
            _Tensor("x", dtype="act", placement="decode-in"),
            "rot",
            DecodeMetadata(_Tensor("positions", shape=(32,)), _page_table()),
        )
    assert model._intermediates == {}
    assert any(tensor.name == "concat-decode" for tensor in released)


def test_close_prevents_every_mutating_or_execution_entrypoint(host_ttnn):
    model = Attention2D.from_config(_config())
    binding = _paged_binding(model)
    owner = binding.owner
    model.bind_kv_cache(binding)
    model.close()
    calls = (
        lambda: model.bind_kv_cache(binding),
        lambda: model.unbind_kv_cache(owner),
        model.load_device_weights,
        model._ensure_runtime_tensors,
        lambda: model.decode_forward("x", "rot", DecodeMetadata("positions")),
        lambda: model.prefill_forward("x", "rot", PrefillMetadata(128)),
        lambda: model.forward("x", "rot", mode="decode", metadata=DecodeMetadata("positions")),
    )
    for call in calls:
        with pytest.raises(RuntimeError, match="closed"):
            call()
    assert host_ttnn == []


def test_cache_binding_is_borrowed_idempotent_and_owner_guarded():
    model = Attention2D.from_config(_config())
    binding = _paged_binding(model)
    model.bind_kv_cache(binding)
    model.bind_kv_cache(binding)
    with pytest.raises(RuntimeError, match="already bound"):
        model.bind_kv_cache(_paged_binding(model))
    with pytest.raises(PermissionError, match="binding owner"):
        model.unbind_kv_cache(object())
    assert model.unbind_kv_cache(binding.owner) is binding


def test_mode_specific_projection_weights_materialize_independently():
    config = _config()
    prefill_wqkv = _weight((5120, 10240), config.mesh_device, _Tensor("PREFILL-WQKV"), "prefill-qkv-map", "bf16")
    prefill_wo = _weight((5120, 5120), config.mesh_device, _Tensor("PREFILL-WO"), "prefill-wo-map", "bf16")
    model = Attention2D.from_config(replace(config, prefill_wqkv=prefill_wqkv, prefill_wo=prefill_wo))

    model.load_device_weights("decode")
    assert (model.wqkv.name, model.wo.name) == ("WQKV", "WO")
    assert model._loaded_weight_modes == {"decode"}

    model.load_device_weights("prefill")
    assert (model.prefill_wqkv.name, model.prefill_wo.name) == ("PREFILL-WQKV", "PREFILL-WO")
    assert model._loaded_weight_modes == {"decode", "prefill"}


def test_transition_does_not_claim_or_release_borrowed_collective_output():
    released = []
    model = object.__new__(Attention2D)
    model.config = SimpleNamespace(intermediate_releaser=released.append)
    model._intermediates = {}
    old, borrowed = _Tensor("old"), _Tensor("borrowed")
    model._own(old)

    assert model._transition(old, borrowed, borrowed=True) is borrowed
    model._release(borrowed)

    assert released == [old]
    assert model._intermediates == {}


def test_module_owns_direct_ttnn_stages_and_has_no_legacy_or_model_imports():
    source = inspect.getsource(attention_2d)
    for call in (
        "ttnn.linear",
        "nlp_create_qkv_heads_decode",
        "paged_update_cache",
        "paged_fill_cache",
        "scaled_dot_product_attention",
        "nlp_concat_heads",
    ):
        assert call in source
    assert "Attention2DOps" not in source
    assert "models.demos" not in source
    assert "models.common.models" not in source
