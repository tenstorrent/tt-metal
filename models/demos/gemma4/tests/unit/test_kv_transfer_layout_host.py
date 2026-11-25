# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests for ``Gemma4ForCausalLM.get_kv_transfer_layout``, the KV layout the vllm-tt-plugin gives KV connectors.

A connector moves a layer's KV by vLLM layer name and block ID. Per layer name the layout lists the paged device
tensors, the bytes of one block on one chip, and the mesh devices (row-major) that hold each KV head shard. The
model state here is fake (no device): ``kv_caches`` holds objects with only ``shape``, ``dtype`` and ``layout``,
shaped as ``allocate_kv_cache_per_layer`` allocates them. The refused configurations keep KV whose block rows are
not vLLM blocks of one layer, so a layout would move wrong data.

    pytest models/demos/gemma4/tests/unit/test_kv_transfer_layout_host.py -p no:cacheprovider
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

SLIDING, FULL = "sliding_attention", "full_attention"
BLOCK_SIZE = 64
NUM_BLOCKS = 4096


@dataclass(frozen=True)
class Geometry:
    """A Gemma4 text config's attention geometry: five sliding layers, then one full-attention layer."""

    layers: int
    sliding_kv_heads: int
    full_kv_heads: int
    sliding_head_dim: int = 256
    full_head_dim: int = 512
    num_kv_shared_layers: int = 0

    @property
    def layer_types(self):
        return [FULL if i % 6 == 5 else SLIDING for i in range(self.layers)]


GEMMA4_31B = Geometry(layers=60, sliding_kv_heads=16, full_kv_heads=4)
GEMMA4_26B = Geometry(layers=30, sliding_kv_heads=8, full_kv_heads=2)


def _model(adapter, geometry, *, mesh_shape=(1, 8), tp_axis=1, hybrid=False, bounded=False, lanes=False):
    """A ``Gemma4ForCausalLM`` with only the state the layout reads: text config, mesh config and KV switches."""
    model = adapter.Gemma4ForCausalLM.__new__(adapter.Gemma4ForCausalLM)
    model._HYBRID_KV_CACHE_GROUPS_ENABLED = hybrid
    model._bounded_sliding_kv_cache = bounded
    text_config = SimpleNamespace(
        layer_types=geometry.layer_types,
        num_key_value_heads=geometry.sliding_kv_heads,
        head_dim=geometry.sliding_head_dim,
        num_global_key_value_heads=geometry.full_kv_heads,
        global_head_dim=geometry.full_head_dim,
        num_kv_shared_layers=geometry.num_kv_shared_layers,
        sliding_window=1024,
    )
    mesh_config = SimpleNamespace(
        mesh_shape=mesh_shape,
        tp_axis=tp_axis,
        tp=mesh_shape[tp_axis],
        lane_sharded=lanes,
        lanes=mesh_shape[1 - tp_axis],
    )
    model.model = [SimpleNamespace(hf_config=text_config, mesh_config=mesh_config, kv_shared_layer_map={})]
    return model


def _cache(adapter, blocks, heads, head_dim, dtype):
    return SimpleNamespace(shape=(blocks, heads, BLOCK_SIZE, head_dim), dtype=dtype, layout=adapter.ttnn.TILE_LAYOUT)


def _local_heads(kv_heads, tp):
    """KV heads one chip holds: a split when there are at least ``tp`` heads, else one GQA-assigned head."""
    return kv_heads // tp if kv_heads >= tp else 1


def _kv_caches(adapter, geometry, *, tp=8, dtype=None, sliding_blocks=NUM_BLOCKS):
    """``allocate_kv_cache_per_layer``'s result for one submesh: a ``[k, v]`` pair per layer, per-chip shapes."""
    dtype = dtype or adapter.ttnn.bfloat16
    layers = []
    for layer_type in geometry.layer_types:
        if layer_type == SLIDING:
            shape = (sliding_blocks, _local_heads(geometry.sliding_kv_heads, tp), geometry.sliding_head_dim)
        else:
            shape = (NUM_BLOCKS, _local_heads(geometry.full_kv_heads, tp), geometry.full_head_dim)
        layers.append([_cache(adapter, *shape, dtype), _cache(adapter, *shape, dtype)])
    return [layers]


def _config(names):
    return SimpleNamespace(num_blocks=NUM_BLOCKS, kv_cache_groups=[SimpleNamespace(layer_names=list(names))])


def _names(geometry):
    return [f"model.layers.{i}.self_attn" for i in range(geometry.layers)]


# -- the 31B on one 8-chip mesh ---------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_name, tile_bytes", [("bfloat16", 2048), ("bfloat8_b", 1088)], ids=["bf16", "bfp8"])
def test_31b_layout_splits_sliding_heads_and_pairs_chips_per_full_head(adapter, dtype_name, tile_bytes):
    model = _model(adapter, GEMMA4_31B)
    kv_caches = _kv_caches(adapter, GEMMA4_31B, dtype=getattr(adapter.ttnn, dtype_name))

    layout = model.get_kv_transfer_layout(kv_caches, _config(_names(GEMMA4_31B)))

    assert sorted(layout) == sorted(_names(GEMMA4_31B))
    for index, layer_type in enumerate(GEMMA4_31B.layer_types):
        name = f"model.layers.{index}.self_attn"
        (k, k_bytes, k_shards), (v, v_bytes, v_shards) = layout[name]
        assert k is kv_caches[0][index][0] and v is kv_caches[0][index][1]
        assert (k_bytes, k_shards) == (v_bytes, v_shards)
        if layer_type == SLIDING:
            # 2 of 16 heads per chip, chip c holds heads 2c and 2c + 1: one shard per chip
            assert k_bytes == 2 * (BLOCK_SIZE // 32) * (256 // 32) * tile_bytes
            assert [list(s) for s in k_shards] == [[c] for c in range(8)]
        else:
            # 4 heads on 8 chips: chip c holds head c // 2, so each head is one shard on a chip pair
            assert k_bytes == 1 * (BLOCK_SIZE // 32) * (512 // 32) * tile_bytes
            assert [list(s) for s in k_shards] == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_a_transferred_block_holds_every_kv_head_once(adapter):
    """Shards times per-chip bytes is the layer's whole KV block, so servers with other chip counts agree on it."""
    model = _model(adapter, GEMMA4_31B)
    layout = model.get_kv_transfer_layout(_kv_caches(adapter, GEMMA4_31B), _config(_names(GEMMA4_31B)))

    for name, layer_type in zip(_names(GEMMA4_31B), GEMMA4_31B.layer_types):
        if layer_type == SLIDING:
            kv_heads, head_dim = GEMMA4_31B.sliding_kv_heads, GEMMA4_31B.sliding_head_dim
        else:
            kv_heads, head_dim = GEMMA4_31B.full_kv_heads, GEMMA4_31B.full_head_dim
        for _, block_bytes, shards in layout[name]:
            assert len(shards) * block_bytes == kv_heads * (BLOCK_SIZE // 32) * (head_dim // 32) * 2048


def test_chips_off_the_tp_axis_hold_their_tp_peers_heads(adapter):
    """The 31B on an 8x4 mesh with heads over the rows (TP axis 0): each shard is every chip of its TP rows."""
    model = _model(adapter, GEMMA4_31B, mesh_shape=(8, 4), tp_axis=0)
    layout = model.get_kv_transfer_layout(_kv_caches(adapter, GEMMA4_31B), _config(_names(GEMMA4_31B)))

    _, _, sliding_shards = layout["model.layers.0.self_attn"][0]
    _, _, full_shards = layout["model.layers.5.self_attn"][0]
    assert [list(s) for s in sliding_shards] == [[4 * r + c for c in range(4)] for r in range(8)]
    assert [list(s) for s in full_shards] == [list(range(8 * h, 8 * h + 8)) for h in range(4)]


def test_only_the_layer_names_of_the_kv_cache_config_are_described(adapter):
    model = _model(adapter, GEMMA4_31B)
    names = ["model.layers.3.self_attn", "model.layers.11.self_attn"]

    layout = model.get_kv_transfer_layout(_kv_caches(adapter, GEMMA4_31B), _config(names))

    assert sorted(layout) == sorted(names)


# -- refused: a block ID does not name one layer's KV in its own buffer ---------------------------------------


def test_hybrid_kv_cache_groups_are_refused(adapter, expect_error):
    """26B with hybrid groups: layers 6t .. 6t+5 (five sliding, one full) share [k, v] pair t."""
    model = _model(adapter, GEMMA4_26B, hybrid=True)
    pairs = _kv_caches(adapter, GEMMA4_26B)[0]
    shared = [[pairs[i - i % 6][0], pairs[i - i % 6][1]] for i in range(GEMMA4_26B.layers)]

    with expect_error(ValueError, "GEMMA4_HYBRID_KV_CACHE_GROUPS"):
        model.get_kv_transfer_layout([shared], _config(_names(GEMMA4_26B)))


def test_bounded_sliding_rings_are_refused(adapter, expect_error):
    model = _model(adapter, GEMMA4_31B, bounded=True)
    kv_caches = _kv_caches(adapter, GEMMA4_31B, sliding_blocks=16 * 32)

    with expect_error(ValueError, "bounded sliding"):
        model.get_kv_transfer_layout(kv_caches, _config(_names(GEMMA4_31B)))


def test_kv_shared_layers_are_refused(adapter, expect_error):
    """E2B-like: the last layers read their source layer's buffer."""
    geometry = Geometry(layers=12, sliding_kv_heads=8, full_kv_heads=8, num_kv_shared_layers=6)
    model = _model(adapter, geometry)

    with expect_error(ValueError, "num_kv_shared_layers=6"):
        model.get_kv_transfer_layout(_kv_caches(adapter, geometry), _config(_names(geometry)))


def test_lane_sharding_is_refused(adapter, expect_error):
    model = _model(adapter, GEMMA4_31B, mesh_shape=(4, 8), lanes=True)

    with expect_error(ValueError, "lane"):
        model.get_kv_transfer_layout(_kv_caches(adapter, GEMMA4_31B), _config(_names(GEMMA4_31B)))


def test_several_submeshes_are_refused(adapter, expect_error):
    model = _model(adapter, GEMMA4_31B)
    kv_caches = _kv_caches(adapter, GEMMA4_31B) * 2

    with expect_error(ValueError, "2 submeshes"):
        model.get_kv_transfer_layout(kv_caches, _config(_names(GEMMA4_31B)))


# -- refused: a cache the model's geometry does not explain ---------------------------------------------------


def _alias_layer_6_to_0(adapter, caches):
    caches[0][6] = caches[0][0]


def _full_layer_with_two_heads(adapter, caches):
    caches[0][5][0] = _cache(adapter, NUM_BLOCKS, 2, 512, adapter.ttnn.bfloat16)


def _sliding_layer_short_of_blocks(adapter, caches):
    caches[0][0][1] = _cache(adapter, NUM_BLOCKS // 2, 2, 256, adapter.ttnn.bfloat16)


def _row_major_layer(adapter, caches):
    caches[0][1][0].layout = adapter.ttnn.ROW_MAJOR_LAYOUT


def _unknown_dtype_layer(adapter, caches):
    caches[0][2][0].dtype = adapter.ttnn.uint32


@pytest.mark.parametrize(
    "corrupt, layer",
    [
        (_alias_layer_6_to_0, "model.layers.6.self_attn"),
        (_full_layer_with_two_heads, "model.layers.5.self_attn"),
        (_sliding_layer_short_of_blocks, "model.layers.0.self_attn"),
        (_row_major_layer, "model.layers.1.self_attn"),
        (_unknown_dtype_layer, "model.layers.2.self_attn"),
    ],
    ids=["aliased-buffer", "head-count", "block-count", "layout", "dtype"],
)
def test_a_cache_the_geometry_does_not_explain_is_refused(adapter, expect_error, corrupt, layer):
    model = _model(adapter, GEMMA4_31B)
    kv_caches = _kv_caches(adapter, GEMMA4_31B)
    corrupt(adapter, kv_caches)

    with expect_error(ValueError, layer.replace(".", r"\.")):
        model.get_kv_transfer_layout(kv_caches, _config(_names(GEMMA4_31B)))


def test_a_layer_name_without_a_cache_is_refused(adapter, expect_error):
    model = _model(adapter, GEMMA4_31B)

    with expect_error(ValueError, r"model\.layers\.60\.self_attn"):
        model.get_kv_transfer_layout(
            _kv_caches(adapter, GEMMA4_31B), _config(_names(GEMMA4_31B) + ["model.layers.60.self_attn"])
        )
