# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lanes on the chain's shared pools, no device: the layouts drop the kv and compressed families (the pager and the
importer never touch the pools), admit_lane_kv_paged is a no-op, and the allocators pass the pools through."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.gdn import Qwen38TTNNGDNState
from models.demos.blackhole.qwen38_flash_next.ttnn.lanes import Qwen38LaneLayout
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38TTNNQSA


def _bare(cls, **attributes):
    instance = object.__new__(cls)
    for name, value in attributes.items():
        object.__setattr__(instance, name, value)
    return instance


def _qsa_lane(shared):
    return _bare(
        qsa_module.Qwen38TTNNQSALaneState,
        packed_kv_cache="kv",
        compressed_index_cache="comp",
        kv_staging="stage",
        raw_key_ring="ring",
        pools_shared=shared,
    )


def _gdn():
    return _bare(Qwen38TTNNGDNState, recurrent="rec", conv=("c0", "c1", "c2", "c3"))


def _lane_state(shared, paged=True):
    layers = [SimpleNamespace(attention=_gdn(), ple=None) for _ in range(model_module.BACKBONE_LAYERS)]
    layers[3] = SimpleNamespace(attention=_qsa_lane(shared), ple=None)
    layers[model_module.PLE_CHECKPOINT_LAYER].ple = SimpleNamespace(conv=("p",))
    return SimpleNamespace(
        lanes=32,
        is_view=False,
        layers=layers,
        paging=SimpleNamespace(page_size=128) if paged else None,
        page_table="pt",
    )


def _chain_state(paged=True):
    layers = [SimpleNamespace(attention=_gdn(), ple=None) for _ in range(model_module.BACKBONE_LAYERS)]
    layers[3] = SimpleNamespace(
        attention=_bare(
            qsa_module.Qwen38TTNNQSAGenericState,
            packed_kv_cache="gkv",
            compressed_index_cache="gcomp",
            kv_staging="gstage",
            raw_key_ring="gring",
        ),
        ple=None,
    )
    layers[model_module.PLE_CHECKPOINT_LAYER].ple = SimpleNamespace(conv=("p",))
    return SimpleNamespace(layers=layers, paging=SimpleNamespace(page_size=128) if paged else None)


@pytest.fixture
def model(monkeypatch):
    m = _bare(model_module.Qwen38TTNNTextModel, allocated_context=4096, mesh_contract=SimpleNamespace())
    m._validate_lane_state = lambda state: state.lanes
    m._validate_generic_state = lambda state: None
    monkeypatch.setattr(Qwen38LaneLayout, "validate", lambda self, contract: None)
    return m


def test_lane_layout_drops_kv_and_compressed_with_shared_pools(model):
    layout = model.lane_layout(_lane_state(shared=True))
    assert layout.kv == () and layout.compressed == () and layout.staging == ("stage",) and layout.ring == ("ring",)


def test_lane_layout_keeps_compressed_for_own_paged_pool_and_both_for_flat(model):
    paged = model.lane_layout(_lane_state(shared=False))
    assert paged.kv == () and paged.compressed == ("comp",)
    flat = model.lane_layout(_lane_state(shared=False, paged=False))
    assert flat.kv == ("kv",) and flat.compressed == ("comp",)


def test_generic_lane_layout_flags(model):
    chain = _chain_state()
    default = model.generic_lane_layout(chain)
    assert default.kv == ("gkv",) and default.compressed == ("gcomp",)
    both = model.generic_lane_layout(chain, exclude_kv=True, exclude_compressed=True)
    assert both.kv == () and both.compressed == () and both.staging == ("gstage",)
    only_kv = model.generic_lane_layout(chain, exclude_kv=True)
    assert only_kv.kv == () and only_kv.compressed == ("gcomp",)


def test_layout_validation_count_rule_accepts_empty_kv_and_compressed(expect_error):
    empty = dict(lanes=1, allocated_context=8192, recurrent=(), conv=(), ple=())
    ok = Qwen38LaneLayout(kv=(), compressed=(), staging=("s",), ring=("r",), **empty)
    # past the count rule the first failure is the tensor shape, not the family counts
    with expect_error(Exception, match="^(?!QSA lane families must hold)"):
        ok.validate(SimpleNamespace())
    bad = Qwen38LaneLayout(kv=(), compressed=("c", "c2"), staging=("s",), ring=("r",), **empty)
    with expect_error(RuntimeError, match="QSA lane families must hold one tensor"):
        bad.validate(SimpleNamespace())


def test_admit_lane_kv_paged_is_a_noop_with_shared_pools(model, monkeypatch):
    from models.demos.blackhole.qwen38_flash_next.ttnn.fused import kv_page_scatter as scatter_module

    calls = []
    monkeypatch.setattr(scatter_module, "kv_page_scatter", lambda *a, **k: calls.append(k))
    model._require_parent_lane_state = lambda state, op: None
    model.admit_lane_kv_paged(_lane_state(shared=True), _chain_state(), 0, 128)
    assert calls == []
    model.admit_lane_kv_paged(_lane_state(shared=False), _chain_state(), 0, 128)
    assert len(calls) == 1 and calls[0]["lane"] == 0 and calls[0]["page_size"] == 128


def _qsa_layer(recorder):
    attention = _bare(Qwen38TTNNQSA, mesh_device="mesh")
    attention.allocate_lane_state = lambda lanes, **kwargs: recorder.append(kwargs) or "attn"
    layer = _bare(
        layer_module.Qwen38TTNNDecoderLayer,
        namespace=layer_module.Qwen38TTNNLayerNamespace.BACKBONE,
        layer_index=3,
        attention=attention,
        ple=None,
    )
    layer._allocate_lane_moe = lambda lanes: "moe"
    layer._allocate_lane_attention_rows = lambda lanes: None
    layer._validate_lane_state = lambda state: state.lanes
    return layer


def test_layer_allocate_lane_state_passes_pools_through(monkeypatch, expect_error):
    seen = []
    layer = _qsa_layer(seen)
    pools = qsa_module.Qwen38TTNNQSAPagedPools(kv="k", index="i")
    layer.allocate_lane_state(32, pools=pools)
    layer.allocate_lane_state(32, pool_rows=256)
    layer.allocate_lane_state(32)
    assert seen == [{"pools": pools}, {"pool_rows": 256}, {}]
    with expect_error(ValueError, "pools replace"):
        layer.allocate_lane_state(32, pool_rows=256, pools=pools)


def _pools(num_pages, page_size):
    return qsa_module.Qwen38TTNNQSAPagedPools(
        kv=SimpleNamespace(shape=(1, 1, num_pages * page_size, 512)),
        index=SimpleNamespace(shape=(num_pages, 1, page_size // 4, 128)),
    )


def _alloc_model():
    layers = [
        SimpleNamespace(layer_index=i, layer_type=model_module.Qwen38TTNNLayerType.QSA if i == 3 else None)
        for i in range(4)
    ]
    return _bare(model_module.Qwen38TTNNTextModel, layers=layers, allocated_context=4096, _poisoned_error=None)


def test_model_allocate_lane_state_checks_pools_against_paging(expect_error):
    model = _alloc_model()
    paging = qsa_module.Qwen38LanePaging(page_size=128, num_pages=64, max_blocks=64)
    with expect_error(ValueError, match="require the lane paging"):
        model.allocate_lane_state(32, residue_free=True, pools={3: _pools(64, 128)})
    with expect_error(ValueError, match="must cover QSA layers"):
        model.allocate_lane_state(32, residue_free=True, paged=paging, pools={})
    with expect_error(ValueError, match="do not match the lane paging"):
        model.allocate_lane_state(32, residue_free=True, paged=paging, pools={3: _pools(32, 128)})
    with expect_error(ValueError, match="do not match the lane paging"):
        model.allocate_lane_state(32, residue_free=True, paged=paging, pools={3: _pools(64, 256)})
