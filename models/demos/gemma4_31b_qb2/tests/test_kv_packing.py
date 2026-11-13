# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""KV packing: full-attention blocks of 256 tokens packed into the shared wide rows.

A pool position holds one full-attention layer and five sliding layers that share one backing of wide rows (sliding
geometry [P, 4, 128, 256]); the scheduler gives every block ID to one group at a time. A sliding block (128 tokens) is one
wide row, a full-attention block (256 tokens) is the same row read as two native 128-token half-pages 2b and 2b + 1.
"""

from types import SimpleNamespace

import torch

import ttnn
from models.demos.gemma4_31b_qb2.tt.generator_vllm import Gemma4ForCausalLM

LAYERS = 12  # two pool positions of five sliding layers and one full-attention layer
FULL = frozenset(range(5, LAYERS, 6))


def hf_config():
    return SimpleNamespace(
        layer_types=["full_attention" if i in FULL else "sliding_attention" for i in range(LAYERS)],
        num_key_value_heads=16,
        head_dim=256,
        num_global_key_value_heads=4,
        global_head_dim=512,
        sliding_window=1024,
    )


def vllm_config(block_size=128):
    return SimpleNamespace(
        max_in_flight_tokens=8192,
        model_config=SimpleNamespace(hf_text_config=hf_config()),
        cache_config=SimpleNamespace(block_size=block_size),
    )


def test_full_attention_blocks_are_256_tokens_with_the_page_size_of_a_sliding_block():
    specs = Gemma4ForCausalLM.get_kv_cache_spec(vllm_config())
    sizes = {i: specs[f"model.layers.{i}.self_attn"].block_size for i in range(LAYERS)}
    assert {i: s for i, s in sizes.items() if i in FULL} == {i: 256 for i in FULL}
    assert {s for i, s in sizes.items() if i not in FULL} == {128}
    # One page size across the hybrid groups: one global ID space, one wide row per ID.
    assert {spec.page_size_bytes for spec in specs.values()} == {2 * 128 * 16 * 256 * 2}


def test_the_block_size_stays_128_tokens(expect_error):
    with expect_error(ValueError, "128-token cache pages"):
        Gemma4ForCausalLM.get_kv_cache_spec(vllm_config(block_size=256))


def test_half_pages_of_full_attention_blocks_and_the_null_block():
    blocks = torch.tensor([[3, 5, 0, 0], [1, 0, 0, 0]])
    pages = Gemma4ForCausalLM._half_pages(blocks, 8)
    assert pages.tolist() == [[6, 7, 10, 11, 0, 0, 0, 0], [2, 3, 0, 0, 0, 0, 0, 0]]


def test_half_pages_of_a_narrow_table_and_a_table_padded_to_the_page_width(expect_error):
    assert Gemma4ForCausalLM._half_pages(torch.tensor([[4]]), 8).tolist() == [[8, 9]]
    padded = torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0]])  # as wide as for 128-token blocks, live entries in the first half
    assert Gemma4ForCausalLM._half_pages(padded, 8).tolist() == [[4, 5, 0, 0, 0, 0, 0, 0]]
    with expect_error(ValueError, "past its 256-token capacity"):
        Gemma4ForCausalLM._half_pages(torch.tensor([[1, 0, 0, 0, 0, 0, 0, 2]]), 8)


def table_adapter(columns=1, batch=4, width=8):
    model = object.__new__(Gemma4ForCausalLM)
    model.sub_caches, model.verify_caches, model.full_layers = {}, {}, frozenset({1})
    model.cache = SimpleNamespace(
        batch=batch,
        kv=[None, None],
        page_tables={i: torch.zeros(batch, width, dtype=torch.int32) for i in range(2)},
        columns=columns,
    )
    return model


def test_tables_expand_only_the_full_attention_layers():
    model = table_adapter()
    sliding_source, full_source = torch.tensor([[7, 9], [3, 0]]), torch.tensor([[7, 9], [3, 0]])
    tables = model._tables(model.cache, [sliding_source, full_source], slots=[2, 0])
    assert tables[0][2, :2].tolist() == [7, 9] and tables[0][0, :2].tolist() == [3, 0]
    assert tables[1][2, :4].tolist() == [14, 15, 18, 19] and tables[1][0, :4].tolist() == [6, 7, 0, 0]
    assert tables[1][1].count_nonzero() == 0 and tables[1][3].count_nonzero() == 0


def test_tables_translate_padding_rows_and_replicate_draft_columns():
    model = table_adapter(columns=2, batch=4)
    source = torch.tensor([[11], [-1]])
    tables = model._tables(model.cache, [source, source])
    assert tables[1][0, :2].tolist() == [22, 23] and tables[1][1].count_nonzero() == 0
    assert tables[1][2:].equal(tables[1][:2]), "draft columns repeat the live rows"


def fake_tensor(shape, buffer_id):
    return SimpleNamespace(
        shape=shape,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        buffer_unique_id=lambda: buffer_id,
        buffer_address=lambda: 4096 * buffer_id,
        buffer_aligned_page_size=lambda: 1088,
    )


def allocation_adapter(monkeypatch):
    allocated = []

    def view(root, shape):
        assert tuple(root.shape)[1:] == (4, 128, 256) and shape == [2 * root.shape[0], 1, 128, 512]
        return fake_tensor(tuple(shape), root.buffer_unique_id())

    monkeypatch.setattr(ttnn.experimental, "view", view)

    def layer(i):
        sliding = i not in FULL

        def allocate_cache(*, physical_pages):
            allocated.append(physical_pages)
            return tuple(fake_tensor((physical_pages, 4, 128, 256), 10 * len(allocated) + k) for k in range(2))

        return SimpleNamespace(
            sliding=sliding,
            kv_heads=4 if sliding else 1,
            head_dim=256 if sliding else 512,
            allocate_cache=allocate_cache,
        )

    model = object.__new__(Gemma4ForCausalLM)
    model.generator = SimpleNamespace(model=SimpleNamespace(layers=[layer(i) for i in range(LAYERS)]))
    model.full_layers, model.rows, model.max_seq_len = FULL, 4, 1024
    model.verify_shapes, model.row_sizes = (), ()
    return model, allocated


def specs(blocks, bad_layer=None):
    out = []
    for i in range(LAYERS):
        shape = (blocks, 1, 256, 512) if i in FULL else (blocks, 4, 128, 256)
        if i == bad_layer:
            shape = (blocks, 4, 128, 512)
        out.append((shape, torch.bfloat16, i // 6))
    return out


def test_each_pool_position_allocates_one_wide_backing_and_views_it_for_its_full_layer(monkeypatch):
    model, allocated = allocation_adapter(monkeypatch)
    cache = model.allocate_kv_cache_per_layer(specs(40))
    assert allocated == [44, 44], "one backing per pool position, n blocks plus the scratch rows"
    for pool in (0, 1):
        sliding = [cache.kv[i] for i in range(6 * pool, 6 * pool + 5)]
        full = cache.kv[6 * pool + 5]
        assert all(s is sliding[0] for s in sliding)
        assert full[0].shape == (88, 1, 128, 512) and full[0].buffer_unique_id() == sliding[0][0].buffer_unique_id()
        assert full[1].buffer_unique_id() == sliding[0][1].buffer_unique_id()
    assert cache.scratch_pages[0] == [40, 41, 42, 43] and cache.scratch_pages[5] == [80, 82, 84, 86]
    assert cache.page_tables[5].shape == (4, 8)


def test_the_allocator_checks_the_geometry_and_the_pool_composition(monkeypatch, expect_error):
    model, _ = allocation_adapter(monkeypatch)
    with expect_error(ValueError, "Unexpected allocator geometry"):
        model.allocate_kv_cache_per_layer(specs(40, bad_layer=2))
    model, _ = allocation_adapter(monkeypatch)
    bad = specs(40)
    bad[5] = (bad[5][0], bad[5][1], 0)  # a second full-attention layer in pool 0 and none in pool 1
    bad[11] = (bad[11][0], bad[11][1], 0)
    with expect_error(ValueError, "one full-attention layer"):
        model.allocate_kv_cache_per_layer(bad)
