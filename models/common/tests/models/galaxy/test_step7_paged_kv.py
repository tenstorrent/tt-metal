# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step 7, area 1: paged KV.

Host-decidable half of the paged-KV gate. What is proved here, without a mesh:

* every active slot owns a pairwise-disjoint, contiguous run of blocks, and no
  inactive slot's sink block ever lands inside an active slot's run - this is
  the *mechanism* behind the "no cross-slot contamination" gate;
* the two page-table layouts really are two layouts, mapped two ways, and the
  decode one is deliberately unpadded;
* a paged capacity resolved *after* the model is constructed reaches every
  layer, and is refused while a cache is bound;
* binding is transactional: a bind that fails part-way leaves no layer bound,
  and only the owner may unbind.

What still needs silicon, and is therefore *not* claimed here: the PCC of a
paged fill/read against the contiguous path. That comparison lives in
``test_full_model_wh_galaxy.py`` for each model and has never been run.

The ``Attention2D`` host fakes are imported from the module suite rather than
copied; that suite is the single source of truth for what a legal 2D attention
config looks like, and duplicating it is how the two drift apart.
"""

from __future__ import annotations

import pytest
import torch

from models.common.models.galaxy.direct_runner import GalaxyDirectRunner
from models.common.models.galaxy.kv_contract import (
    GalaxyAttentionKVSpec,
    GalaxyPagedAttentionConfig,
    GalaxyPagedKVContract,
)
from models.common.models.llama33_70b_galaxy.model import Llama33_70BGalaxyTransformer2D
from models.common.models.qwen3_32b_galaxy.model import Qwen3_32BGalaxyTransformer2D
from models.common.modules.attention.attention_2d import Attention2D, DecodeMetadata, PrefillMetadata
from models.common.tests.models.galaxy.step7_harness import (
    BLOCK_SIZE,
    GALAXY_PHYSICAL_BATCH,
    GALAXY_USERS_PER_COLUMN,
    RecordingModel,
    ReplicateMapper,
    ShardMapper,
    patch_direct_runner,
)

# Shared host fakes. See the module docstring for why these are imported.
from models.common.tests.modules.attention.test_attention_2d import (  # noqa: F401 - host_ttnn is a fixture
    _config,
    _page_table,
    _Tensor,
    host_ttnn,
)

_MODEL_CLASSES = pytest.mark.parametrize(
    "model_class",
    [Llama33_70BGalaxyTransformer2D, Qwen3_32BGalaxyTransformer2D],
    ids=["llama", "qwen"],
)


# ---------------------------------------------------------------------------
# Block ownership: the mechanism behind "no cross-slot contamination"
# ---------------------------------------------------------------------------


def _runner(*, active_slots: int, max_seq_len: int = 2048) -> GalaxyDirectRunner:
    blocks_per_user = max_seq_len // BLOCK_SIZE
    sinks = GALAXY_PHYSICAL_BATCH - active_slots
    model = RecordingModel(
        max_seq_len=max_seq_len,
        max_num_blocks=blocks_per_user * active_slots + sinks,
    )
    return GalaxyDirectRunner(model, active_slots=active_slots)


@pytest.mark.parametrize("active_slots", [1, 8, 16, 31, 32])
def test_no_two_slots_can_address_the_same_block(active_slots):
    """The cross-slot contamination gate, reduced to its host-decidable core.

    A write for user *i* can only reach user *j*'s cache if some block appears
    in both rows of the page table. Block ownership is static, so that is a
    property of the table alone.
    """

    runner = _runner(active_slots=active_slots)
    rows = runner._page_table_rows()
    assert rows.shape == (GALAXY_PHYSICAL_BATCH, runner.blocks_per_user)

    owned = [set(int(block) for block in rows[slot].tolist()) for slot in range(GALAXY_PHYSICAL_BATCH)]
    for slot, blocks in enumerate(owned):
        for other in range(slot + 1, GALAXY_PHYSICAL_BATCH):
            shared = blocks & owned[other]
            assert not shared, f"slots {slot} and {other} share blocks {sorted(shared)}"


@pytest.mark.parametrize("active_slots", [1, 16, 31])
def test_an_idle_slots_sink_never_lands_in_an_active_slots_run(active_slots):
    """Padding rows still write KV; their blocks must be nobody else's."""

    runner = _runner(active_slots=active_slots)
    rows = runner._page_table_rows()
    active = set()
    for slot in range(active_slots):
        active |= set(int(block) for block in rows[slot].tolist())
    for slot in range(active_slots, GALAXY_PHYSICAL_BATCH):
        sinks = set(int(block) for block in rows[slot].tolist())
        assert len(sinks) == 1, f"idle slot {slot} owns {len(sinks)} blocks, expected one sink"
        assert not (sinks & active), f"idle slot {slot} sink {sinks} overlaps an active slot"


@pytest.mark.parametrize("active_slots", [1, 8, 16, 31, 32])
def test_every_addressed_block_is_inside_the_allocated_pool(active_slots):
    runner = _runner(active_slots=active_slots)
    rows = runner._page_table_rows()
    pool = runner.kv_spec.paged_attention_config.max_num_blocks
    assert int(rows.min()) >= 0
    assert int(rows.max()) < pool, f"page table addresses block {int(rows.max())} outside a {pool}-block pool"


def test_each_active_slot_owns_enough_blocks_for_the_whole_served_context():
    runner = _runner(active_slots=32)
    assert runner.blocks_per_user * runner.block_size >= runner.max_seq_len


# ---------------------------------------------------------------------------
# The two page-table layouts
# ---------------------------------------------------------------------------


def test_prefill_and_decode_stage_two_different_tables_two_different_ways(monkeypatch):
    """Prefill replicates a padded table; decode shards an unpadded one.

    ``paged_fill_cache`` indexes by ``batch_idx`` so every device needs every
    user's row; the paged decode SDPA derives its KV length from the row width,
    so padding it would claim context a slot does not own.
    """

    recorder = patch_direct_runner(monkeypatch)
    runner = _runner(active_slots=32)
    runner.open()
    try:
        tables = recorder.page_tables
        assert len(tables) == 2, f"expected exactly two staged page tables, got {len(tables)}"
        prefill, decode = tables

        assert isinstance(prefill.mapper, ReplicateMapper)
        assert isinstance(decode.mapper, ShardMapper)
        assert decode.mapper.dims == (None, 0)

        # Prefill is stick-aligned to eight int32 entries; decode is not padded.
        assert prefill.host.shape[1] % runner.page_table_column_alignment == 0
        assert decode.host.shape[1] == runner.blocks_per_user
        assert prefill.host.shape[0] == decode.host.shape[0] == GALAXY_PHYSICAL_BATCH

        # The device-local view is what every module validator reads.
        assert tuple(decode.mapper.device_local(decode.host).shape) == (
            GALAXY_USERS_PER_COLUMN,
            runner.blocks_per_user,
        )
        assert tuple(prefill.mapper.device_local(prefill.host).shape)[0] == GALAXY_PHYSICAL_BATCH
    finally:
        runner.close()


def test_padding_the_prefill_table_cannot_invent_a_block_a_slot_does_not_own(monkeypatch):
    """Stick alignment must pad with zeros only beyond every real entry."""

    recorder = patch_direct_runner(monkeypatch)
    # 2048/32 = 64 blocks per user is already aligned; force a ragged width.
    runner = _runner(active_slots=32, max_seq_len=2048)
    monkeypatch.setattr(runner, "blocks_per_user", 65)
    prefill = runner.prefill_page_table_rows()
    assert prefill.shape[1] == 72, "65 entries must pad up to the next multiple of eight"
    assert torch.equal(prefill[:, 65:], torch.zeros((GALAXY_PHYSICAL_BATCH, 7), dtype=torch.int32))
    assert not recorder.staged, "reading the table rows must not stage anything"


# ---------------------------------------------------------------------------
# Late capacity resolution and transactional binding
# ---------------------------------------------------------------------------


def _detached_model(model_class, *, n_layers: int = 4, paged: bool = True):
    """Build the KV-owning half of a transformer without touching a mesh.

    ``set_kv_cache`` and ``configure_paged_attention`` read only these
    attributes, so this exercises the real implementations of both.
    """

    config = _config()
    mesh = config.mesh_device
    layers = []
    for _ in range(n_layers):
        attention = Attention2D.from_config(_config(mesh=mesh))
        layers.append(_Layer(attention))
    spec = GalaxyAttentionKVSpec(
        n_local_kv_heads=1,
        head_dim=128,
        kv_cache_dtype="cache",
        page_table_dtype="uint32",
        paged_attention_config=GalaxyPagedAttentionConfig(block_size=32, max_num_blocks=64) if paged else None,
    )
    model = object.__new__(model_class)
    model._closed = False
    model._kv_bound = False
    model._kv_owner = object()
    model._kv_specs = (spec,) * n_layers
    model.layers = layers
    model.mesh_device = mesh
    return model


class _Layer:
    def __init__(self, attention):
        self.attention = attention
        self.kv_spec = None


def _cache(n_layers: int, *, shape=(64, 1, 32, 128), dtype="cache"):
    return [
        [_Tensor(f"k{index}", shape=shape, dtype=dtype), _Tensor(f"v{index}", shape=shape, dtype=dtype)]
        for index in range(n_layers)
    ]


@_MODEL_CLASSES
def test_a_paged_capacity_resolved_after_construction_reaches_every_layer(model_class):
    """Late capacity resolution: the cache is sized after the model exists."""

    model = _detached_model(model_class, paged=False)
    assert all(spec.paged_attention_config is None for spec in model._kv_specs)

    model.configure_paged_attention(block_size=32, max_num_blocks=128)

    for spec, layer in zip(model._kv_specs, model.layers):
        assert spec.paged_attention_config == GalaxyPagedAttentionConfig(block_size=32, max_num_blocks=128)
        assert layer.kv_spec is spec
        assert spec.local_cache_shape() == (128, 1, 32, 128)


@_MODEL_CLASSES
def test_paged_capacity_cannot_be_resolved_while_a_cache_is_bound(model_class):
    model = _detached_model(model_class)
    model.set_kv_cache(_cache(len(model.layers)))
    try:
        with pytest.raises(RuntimeError, match="cannot be reconfigured"):
            model.configure_paged_attention(block_size=32, max_num_blocks=256)
    finally:
        model.set_kv_cache(None)


@_MODEL_CLASSES
def test_binding_every_layer_is_all_or_nothing(model_class):
    """A bind that fails part-way must leave no layer bound.

    The third layer's cache carries the wrong dtype, which ``Attention2D``
    rejects. Layers 0 and 1 have already been bound when that happens.
    """

    model = _detached_model(model_class, n_layers=4)
    cache = _cache(4)
    cache[2] = [
        _Tensor("k2", shape=(64, 1, 32, 128), dtype="wrong"),
        _Tensor("v2", shape=(64, 1, 32, 128), dtype="wrong"),
    ]

    with pytest.raises(ValueError):
        model.set_kv_cache(cache)

    assert model._kv_bound is False
    for index, layer in enumerate(model.layers):
        assert layer.attention.kv_cache_binding is None, f"layer {index} stayed bound after a failed bind"


@_MODEL_CLASSES
def test_a_malformed_layer_entry_also_unwinds_every_earlier_layer(model_class):
    model = _detached_model(model_class, n_layers=3)
    cache = _cache(3)
    cache[1] = [_Tensor("k1", shape=(64, 1, 32, 128), dtype="cache")]  # one tensor, not two

    with pytest.raises(ValueError, match="exactly two K/V tensors"):
        model.set_kv_cache(cache)

    assert all(layer.attention.kv_cache_binding is None for layer in model.layers)


@_MODEL_CLASSES
def test_unbind_is_transactional_and_repeatable(model_class):
    model = _detached_model(model_class, n_layers=3)
    model.set_kv_cache(_cache(3))
    assert all(layer.attention.kv_cache_binding is not None for layer in model.layers)

    model.set_kv_cache(None)
    assert model._kv_bound is False
    assert all(layer.attention.kv_cache_binding is None for layer in model.layers)

    # Idempotent: unbinding an unbound model is not an error.
    model.set_kv_cache(None)
    assert all(layer.attention.kv_cache_binding is None for layer in model.layers)


@_MODEL_CLASSES
def test_rebinding_replaces_the_previous_binding_rather_than_stacking(model_class):
    model = _detached_model(model_class, n_layers=2)
    model.set_kv_cache(_cache(2))
    first = [layer.attention.kv_cache_binding for layer in model.layers]

    model.set_kv_cache(_cache(2))
    second = [layer.attention.kv_cache_binding for layer in model.layers]

    assert all(a is not b for a, b in zip(first, second))
    try:
        assert all(binding is not None for binding in second)
    finally:
        model.set_kv_cache(None)


@_MODEL_CLASSES
def test_a_layer_count_mismatch_is_refused_before_anything_is_bound(model_class):
    model = _detached_model(model_class, n_layers=3)
    with pytest.raises(ValueError, match="kv_cache has 2 entries"):
        model.set_kv_cache(_cache(2))
    assert all(layer.attention.kv_cache_binding is None for layer in model.layers)


def test_only_the_binding_owner_may_unbind():
    """A second owner cannot steal a bound cache."""

    model = _detached_model(Llama33_70BGalaxyTransformer2D, n_layers=1)
    model.set_kv_cache(_cache(1))
    try:
        with pytest.raises(PermissionError):
            model.layers[0].attention.unbind_kv_cache(object())
    finally:
        model.set_kv_cache(None)


# ---------------------------------------------------------------------------
# The KV contract the common paged manager reads
# ---------------------------------------------------------------------------


def test_the_kv_contract_presents_one_layer_view_per_spec_and_forwards_binding():
    model = RecordingModel(n_layers=5)
    contract = GalaxyPagedKVContract(model, model.kv_specs)

    assert contract.config.n_layers == 5
    assert contract.config.num_devices == 8  # KV heads shard over the eight mesh rows
    assert contract.config.mesh_device is model.mesh_device
    assert [view.attention_config for view in contract.config.block_configs] == list(model.kv_specs)

    contract.set_kv_cache("a-cache")
    assert model.bound_cache == "a-cache"


def test_the_kv_contract_refuses_a_model_with_no_resolved_mesh():
    model = RecordingModel()
    model.mesh_device = None
    with pytest.raises(ValueError, match="resolved mesh_device"):
        GalaxyPagedKVContract(model, model.kv_specs)


def test_a_contiguous_spec_has_no_paged_shape_and_no_binding_metadata():
    spec = GalaxyAttentionKVSpec(n_local_kv_heads=1, head_dim=128, kv_cache_dtype="cache")
    assert spec.paged_kv_metadata() is None
    with pytest.raises(ValueError, match="contiguous KV cache has no paged shape"):
        spec.local_cache_shape()


# ---------------------------------------------------------------------------
# Adversarial: the two layouts fed to the wrong side
# ---------------------------------------------------------------------------


def test_decode_rejects_a_table_that_cannot_carry_one_mesh_columns_users(host_ttnn):
    """A row count that is not a whole number of device-local batches."""

    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged(model))
    for rows in (4, 12, 31):
        with pytest.raises(ValueError, match="device-local rows"):
            model.decode_forward(
                _Tensor("x", dtype="act", placement="decode-in"),
                "rot",
                DecodeMetadata(_Tensor("positions", shape=(32,), dtype="uint32"), _page_table(rows=rows)),
            )
    assert host_ttnn == []


def test_prefill_rejects_a_decode_shaped_table_that_cannot_reach_the_addressed_user(host_ttnn):
    """The decode table's device-local view is eight rows; prefill fills 32.

    This is the *reverse* of the adversarial case below, and it does fail
    closed: ``paged_fill_cache`` indexes by ``batch_idx``, so a table that stops
    at row 7 cannot address user 8.
    """

    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged(model))
    with pytest.raises(ValueError, match="one row for every addressed user"):
        model.prefill_forward(
            _Tensor("x", dtype="act", placement="prefill-in"),
            "rot",
            PrefillMetadata(128, (8,), page_table=_page_table(rows=GALAXY_USERS_PER_COLUMN)),
        )
    assert host_ttnn == []


def test_decode_cannot_tell_the_prefill_layout_from_a_four_core_l1_repeat(host_ttnn):
    """KNOWN GAP - Milestone B defect D-C1. Recorded, not worked around.

    The step-7 gate asks that a prefill-shaped page table fed to decode be
    *rejected*. It is not. ``_validate_decode_page_table`` discriminates on row
    count alone, and accepts any positive multiple of ``users_per_column``
    because an L1-sharded table legitimately repeats the device-local batch once
    per core. The replicated prefill table's device-local view is 32 rows, and
    ``32 == 4 * 8``, so it passes every check and reaches ``paged_update_cache``.

    Shape cannot separate the two. The distinguishing fact is placement: the
    prefill table is DRAM-interleaved and replicated, whereas a legitimate
    repeat is L1 height-sharded over exactly ``rows / users_per_column`` cores.
    The validator never reads ``memory_config()``.

    This test pins the behaviour that exists so the gap cannot silently close or
    silently widen. It is deliberately *not* an assertion that the behaviour is
    correct; see ``tttv2_milestone_b_evidence/coverage/REPORT.md`` area 1.
    """

    model = Attention2D.from_config(_config())
    model.bind_kv_cache(_paged(model))
    prefill_shaped = _page_table(rows=GALAXY_PHYSICAL_BATCH, columns=64)

    result = model.decode_forward(
        _Tensor("x", dtype="act", placement="decode-in"),
        "rot",
        DecodeMetadata(_Tensor("positions", shape=(32,), dtype="uint32"), prefill_shaped),
    )

    assert result.name == "output"
    assert "sdpa-decode-paged" in [event[0] for event in host_ttnn]
    # The discriminator that would work is never consulted.
    assert not any("memory_config" in str(event) for event in host_ttnn if event[0] == "paged_update_cache")


def _paged(model):
    from models.common.modules.attention.attention_2d import KVCacheBinding, PagedKVMetadata

    metadata = PagedKVMetadata(block_size=32, max_num_blocks=64, cache_dtype="cache", page_table_dtype="uint32")
    shape = (64, 1, 32, 128)
    return KVCacheBinding(
        _Tensor("keys", shape=shape, dtype="cache", placement="cache"),
        _Tensor("values", shape=shape, dtype="cache", placement="cache"),
        object(),
        metadata,
        model.config.mesh_device,
    )
