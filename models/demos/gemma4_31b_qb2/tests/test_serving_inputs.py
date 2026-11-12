# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Scheduler tables retain row ownership across padded prefill and decode."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.generator import CacheState, Gemma4Generator
from models.demos.gemma4_31b_qb2.tt.generator_vllm import Gemma4ForCausalLM


def adapter():
    model = object.__new__(Gemma4ForCausalLM)
    model.cache = SimpleNamespace(
        batch=4, kv=[None, None], page_tables={i: torch.zeros(4, 8, dtype=torch.int32) for i in range(2)}
    )
    return model


def test_prefill_scatters_live_prefix_and_ignores_padded_rows():
    model = adapter()
    source = torch.tensor([[11, 12], [21, 22], [91, 92], [93, 94]])
    tables = model._tables(model.cache, [source, source], slots=[3, 1])
    expected = torch.zeros(4, 8, dtype=torch.int32)
    expected[3, :2] = torch.tensor([11, 12])
    expected[1, :2] = torch.tensor([21, 22])
    assert torch.equal(tables[0], expected)
    assert tables[0] is tables[1]
    source.fill_(100)
    assert torch.equal(tables[0], expected), "Later scheduler writes must not mutate a submitted table"


def test_decode_preserves_slot_order_and_translates_null_pages():
    model = adapter()
    source = torch.tensor([[11, 12], [21, 22], [-1, -1], [31, 32]])
    tables = model._tables(model.cache, [source, source])
    assert torch.equal(tables[0][:, :2], torch.tensor([[11, 12], [21, 22], [0, 0], [31, 32]]))
    assert torch.count_nonzero(tables[0][:, 2:]) == 0


def test_prefill_rejects_missing_live_rows(expect_error):
    model = adapter()
    source = torch.zeros(1, 8, dtype=torch.int32)
    with expect_error(ValueError, "fewer rows"):
        model._tables(model.cache, [source, source], slots=[0, 1])


def private_sliding_cache(logical_pages):
    generator = object.__new__(Gemma4Generator)
    generator.mesh = object()
    generator.max_seq_len = 262144
    generator.model = SimpleNamespace(layers=[SimpleNamespace(kind="sliding_attention", kv_heads=4, head_dim=128)])
    pool = min(logical_pages, Decoder.SLIDING_WINDOW_PAGES)
    table = (torch.arange(2)[:, None] * pool + torch.arange(logical_pages)[None, :] % pool).int()
    kv = [
        SimpleNamespace(
            shape=(2 * pool + 2, 4, 128, 128),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=lambda: ttnn.DRAM_MEMORY_CONFIG,
            device=lambda: generator.mesh,
            buffer_unique_id=lambda index=index: index,
        )
        for index in range(2)
    ]
    state = CacheState(
        [kv],
        {"sliding_attention": table},
        2,
        logical_pages * 128,
        scratch_pages={"sliding_attention": [2 * pool, 2 * pool + 1]},
    )
    return generator, state, table


@pytest.mark.parametrize("logical_pages", [5, 12])
def test_private_sliding_cache_accepts_short_and_cyclic_tables(logical_pages):
    generator, state, table = private_sliding_cache(logical_pages)
    generator._validate_cache(state, state.page_tables)
    # Moving whole request slots preserves private ownership and the cycle.
    generator._validate_cache(state, {"sliding_attention": table.flip(0)})


@pytest.mark.parametrize("row,column,value", [(0, 1, 0), (1, 0, 0), (0, 9, 1)])
def test_private_sliding_cache_rejects_aliases_and_broken_cycles(row, column, value, expect_error):
    generator, state, table = private_sliding_cache(12)
    table[row, column] = value
    with expect_error(ValueError, "distinct|cyclic"):
        generator._validate_cache(state, state.page_tables)


def test_decoder_rejects_other_four_chip_blackhole_topologies(monkeypatch, expect_error):
    mesh = SimpleNamespace(get_num_devices=lambda: 4, shape=(1, 4), arch=lambda: ttnn.Arch.BLACKHOLE)
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: object())
    with expect_error(ValueError, "P300_X2"):
        Decoder.from_state_dict({}, hf_config=None, layer_idx=0, mesh_device=mesh)


def test_prefill_history_follows_slot_moves_and_request_release():
    model = object.__new__(Gemma4ForCausalLM)
    model.prefill_slot_keys = {0: 11, 1: 12}
    model.cache = SimpleNamespace(prefill_history={11: object(), 12: object()})
    model.note_state_slots_moved({0: 2, 1: 0})
    model.release_request(0)
    assert model.prefill_slot_keys == {2: 11}
    assert set(model.cache.prefill_history) == {11}
    model.release_request(1)  # Already empty: do not discard another request.
    assert set(model.cache.prefill_history) == {11}
    model.release_request(2)
    assert model.prefill_slot_keys == {}
    assert model.cache.prefill_history == {}


def test_persistent_capture_release_discards_all_prefill_history():
    closed = []
    model = object.__new__(Gemma4ForCausalLM)
    model.prefill_slot_keys = {0: 11}
    model.cache = SimpleNamespace(prefill_history={11: object()})
    model.generator = SimpleNamespace(close=lambda: closed.append(True))
    model.entry = object()
    model.release_persistent_capture()
    assert model.prefill_slot_keys == {}
    assert model.cache.prefill_history == {}
    assert model.entry is None
    assert closed == [True]


@pytest.mark.parametrize("start", [128, 1024, 1152, 8192])
def test_prefix_history_gathers_retained_pages_in_logical_order(monkeypatch, expect_error, start):
    generator = object.__new__(Gemma4Generator)
    generator.model = SimpleNamespace(
        layers=[SimpleNamespace(sliding=True, window=1024), SimpleNamespace(sliding=False)]
    )
    table = torch.arange(start // 128, 0, -1).reshape(1, -1)
    cache = torch.arange((start // 128 + 1) * 128).reshape(-1, 1, 128, 1).float()
    state = SimpleNamespace(kv=[(cache, cache + 1), None])
    monkeypatch.setattr(ttnn, "concat", torch.cat)
    monkeypatch.setattr(ttnn, "typecast", lambda tensor, dtype: tensor)
    # The page id travels as device tensors (host tensors here); the slice takes the page [start, end) along dim 0.
    generator._upload = lambda values, dtype: values
    monkeypatch.setattr(
        ttnn, "slice", lambda cache, begin, end, slice_dim, num_devices: cache[int(begin[0]) : int(end[0])]
    )
    histories = generator._restore_prefill_history(state, {0: table}, 0, start)
    expected = torch.cat([cache[p : p + 1] for p in table[0, -8:]], dim=2)
    assert torch.equal(histories[0][0], expected)
    assert torch.equal(histories[0][1], expected + 1)
    assert histories[1] is None
    table[0, -1] = 0
    with expect_error(ValueError, "Missing sliding prefix pages"):
        generator._restore_prefill_history(state, {0: table}, 0, start)


def test_concurrent_shared_prefixes_keep_private_history_and_follow_moves():
    model = adapter()
    model.prefill_slot_keys = {}
    model.next_prefill_key = 0
    model.cache.prefill_history = {}
    submitted = []
    model.generator = SimpleNamespace(prefill_forward=lambda *args, **kwargs: submitted.append(kwargs["history_keys"]))
    shared = torch.tensor([[11, 12], [11, 13]])
    tokens = torch.zeros(2, 256, dtype=torch.int32)
    model.prefill_forward(
        tokens,
        None,
        model.cache,
        [256, 256],
        page_tables_per_layer=[shared, shared],
        start_pos=[128, 128],
        empty_slots=[0, 1],
    )
    first, second = submitted[-1]
    assert first != second
    model.cache.prefill_history.update({first: object(), second: object()})
    model.note_state_slots_moved({0: 2, 1: 0})
    model.release_request(0)
    assert set(model.cache.prefill_history) == {first}
    model.prefill_forward(
        tokens[:1], None, model.cache, [256], page_tables_per_layer=[shared, shared], start_pos=[128], empty_slots=[2]
    )
    assert submitted[-1] == [first]
    model.prefill_forward(
        tokens[:1], None, model.cache, [256], page_tables_per_layer=[shared, shared], start_pos=[128], empty_slots=[0]
    )
    assert submitted[-1][0] not in (first, second)


@pytest.mark.parametrize("remap", [None, [1, 0, 2, 3]])
def test_decode_releases_only_its_slot_history_despite_page_id_collision(remap):
    model = adapter()
    model.max_batch_size = model.rows = 4
    model.row_sizes, model.sub_caches = (), {}
    model.prefill_slot_keys = {0: 1, 1: 2}
    model.cache.prefill_history = {1: object(), 2: object()}
    model.generator = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(sliding=False)]),
        prepare_decode=lambda *args, **kwargs: {"logits": torch.zeros(1, 1, 4, 10)},
        replay=lambda *args, **kwargs: None,
        _host_logits=lambda tensor: tensor,
    )
    # Physical page 2 belongs to decoding row 0. Private history key 2 belongs
    # to row 1, whose prefill is still in progress and must remain live.
    tables = torch.tensor([[2, 3], [2, 4], [0, 0], [0, 0]])
    model.decode_forward(
        torch.zeros(4, 1, dtype=torch.int32),
        torch.tensor([256, -1, -1, -1]),
        None,
        model.cache,
        page_tables_per_layer=[tables, tables],
        slot_remap=None if remap is None else torch.tensor(remap),
    )
    remaining_slot = 1 if remap is None else 0
    assert model.prefill_slot_keys == {remaining_slot: remaining_slot + 1}
    assert set(model.cache.prefill_history) == {remaining_slot + 1}
    if remap is not None:
        model.note_state_slots_moved({source: dest for dest, source in enumerate(remap)})
        assert model.prefill_slot_keys == {1: 1}


def spec_adapter(slots, spec_k):
    model = object.__new__(Gemma4ForCausalLM)
    model.spec_k = spec_k
    model.max_batch_size = slots
    model.rows = slots * (1 + spec_k)
    return model


@pytest.mark.parametrize(
    "slots,requested,effective",
    [(1, 7, 7), (4, 7, 7), (8, 7, 3), (8, 2, 2), (16, 7, 1), (16, 1, 1)],
)
def test_spec_plan_fits_the_decode_trace(slots, requested, effective):
    plan = Gemma4ForCausalLM.spec_plan(None, slots, requested)
    assert plan.effective_k == effective and slots * (1 + plan.effective_k) <= 32
    assert plan.accept_modes == ("argmax_ids",) and plan.supports_narrow_decode


@pytest.mark.parametrize("slots", [17, 32])
def test_spec_plan_refuses_slot_counts_without_room_for_drafts(slots):
    reject = Gemma4ForCausalLM.spec_plan(None, slots, 3)
    assert reject.supported_k == ()


def test_draft_columns_ride_in_rows_after_the_plain_decode_rows():
    slots, spec_k = 2, 2
    model = spec_adapter(slots, spec_k)
    model.cache = SimpleNamespace(
        batch=model.rows, kv=[None], page_tables={0: torch.zeros(model.rows, 4, dtype=torch.int32)}
    )
    source = torch.tensor([[5, 6], [7, 8]])
    tables = model._tables(model.cache, [source])[0]
    assert tables.tolist() == [[5, 6, 0, 0], [7, 8, 0, 0]] * (1 + spec_k)
    assert model._pad_rows(torch.tensor([[11], [12]]), 0).reshape(-1).tolist() == [11, 12, 0, 0, 0, 0]
    assert model._pad_rows(torch.tensor([3, 4]), -1).tolist() == [3, 4, -1, -1, -1, -1]


def test_verify_block_is_flattened_by_column_and_answers_per_slot():
    slots, spec_k = 2, 2
    model = spec_adapter(slots, spec_k)
    model.cache = SimpleNamespace(
        batch=model.rows, kv=[None], page_tables={0: torch.zeros(model.rows, 4, dtype=torch.int32)}
    )
    seen = {}

    def prepare_decode(tokens, positions, **kwargs):
        seen["tokens"], seen["positions"] = tokens.reshape(-1).tolist(), positions.tolist()
        return {"tokens": ttnn.from_torch(torch.arange(32, dtype=torch.int32).reshape(1, 1, 1, 32), dtype=ttnn.uint32)}

    model.generator = SimpleNamespace(
        prepare_decode=prepare_decode,
        configure_sampling=lambda **kw: seen.setdefault("batch", kw["batch"]),
        replay=lambda entry: None,
    )
    model.read_decode_output = lambda output: output
    block = torch.tensor([[10, 11, 12], [20, 999, 999]])
    positions = torch.tensor([[7, 8, 9], [3, -1, -1]])
    out = model._verify(block, positions, model.cache, [torch.tensor([[1, 2], [3, 4]])], None)
    assert seen["tokens"] == [10, 20, 11, 0, 12, 0] and seen["positions"] == [7, 3, 8, -1, 9, -1]
    assert seen["batch"] == 6
    # device row j * slots + u answers slot u, column j
    assert out.argmax_ids.tolist() == [[0, 2, 4], [1, 3, 5]]


def test_kv_pool_follows_max_model_len():
    """The pool is one request of the served context plus the sliding windows and in-flight prefill tokens."""
    full = Gemma4ForCausalLM.get_max_tokens_all_users(max_num_seqs=32, max_model_len=262144)
    half = Gemma4ForCausalLM.get_max_tokens_all_users(max_num_seqs=32, max_model_len=131072)
    assert full == Gemma4ForCausalLM.get_max_tokens_all_users(max_num_seqs=32)
    assert full - half == 131072
    assert full == 262144 + 5 * (
        (Decoder.SLIDING_WINDOW_PAGES * Decoder.PAGE_SIZE * 32 + (Decoder.SLIDING_WINDOW_PAGES - 1) * Decoder.PAGE_SIZE)
        + 2 * Gemma4ForCausalLM.MAX_PREFILL_TOKENS
    )
