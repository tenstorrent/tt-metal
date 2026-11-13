# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Scheduler tables retain row ownership across padded prefill and decode."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4_31b_qb2.tt.decode_rows import verify_shapes
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.generator import CacheState, Gemma4Generator
from models.demos.gemma4_31b_qb2.tt.generator_vllm import Gemma4ForCausalLM


def adapter():
    model = object.__new__(Gemma4ForCausalLM)
    model.sub_caches, model.verify_caches, model.row_sizes, model.verify_shapes, model.cur_state = {}, {}, (), (), None
    model.cache = SimpleNamespace(
        batch=4, kv=[None, None], page_tables={i: torch.zeros(4, 8, dtype=torch.int32) for i in range(2)}, columns=1
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
    model.max_batch_size = model.rows = slots
    model.sub_caches, model.verify_caches, model.row_sizes, model.cur_state = {}, {}, (), None
    model.verify_shapes = verify_shapes(slots, spec_k)
    model.prefill_slot_keys = {}
    model.cache = SimpleNamespace(prefill_history={})
    return model


@pytest.fixture
def spec_k_restored(monkeypatch):
    # spec_plan records the draft length on the class; restore it after the test.
    monkeypatch.setattr(Gemma4ForCausalLM, "spec_k", Gemma4ForCausalLM.spec_k)


@pytest.mark.parametrize(
    "slots,requested,effective,k_by_rows",
    [
        (1, 7, 5, ((1, 5),)),
        (4, 7, 5, ((1, 5), (4, 3))),
        (8, 7, 5, ((1, 5), (4, 3), (8, 0))),
        (8, 2, 2, ((1, 2), (4, 2), (8, 0))),
        (32, 7, 5, ((1, 5), (4, 3), (32, 0))),
        (32, 1, 1, ((1, 1), (4, 1), (32, 0))),
    ],
)
def test_spec_plan_follows_the_live_requests_and_fits_the_decode_trace(
    spec_k_restored, slots, requested, effective, k_by_rows
):
    plan = Gemma4ForCausalLM.spec_plan(None, slots, requested)
    assert plan.effective_k == effective and plan.k_by_rows == k_by_rows
    assert all(users * (1 + k) <= 32 for users, k in verify_shapes(slots, requested))
    assert plan.accept_modes == ("argmax_ids",) and plan.supports_narrow_decode


def test_spec_plan_refuses_a_draft_length_of_zero(spec_k_restored):
    assert Gemma4ForCausalLM.spec_plan(None, 8, 0).supported_k == ()


def test_draft_columns_ride_in_rows_after_the_plain_decode_rows():
    slots, spec_k = 2, 2
    model = spec_adapter(slots, spec_k)
    state = SimpleNamespace(
        batch=slots * (1 + spec_k),
        kv=[None],
        page_tables={0: torch.zeros(slots * (1 + spec_k), 4, dtype=torch.int32)},
        columns=1 + spec_k,
    )
    model.verify_caches = {slots: state}
    model.cache = SimpleNamespace(
        batch=slots, kv=[None], page_tables={0: torch.zeros(slots, 4, dtype=torch.int32)}, columns=1
    )
    source = torch.tensor([[5, 6], [7, 8]])
    tables = model._tables(state, [source])[0]
    assert tables.tolist() == [[5, 6, 0, 0], [7, 8, 0, 0]] * (1 + spec_k)
    assert model._pad_rows(torch.tensor([[11]]), 0).reshape(-1).tolist() == [11, 0]
    assert model._pad_rows(torch.tensor([3]), -1).tolist() == [3, -1]


def test_verify_block_is_flattened_by_column_and_answers_per_slot():
    slots, spec_k = 2, 2
    model = spec_adapter(slots, spec_k)
    state = SimpleNamespace(
        batch=slots * (1 + spec_k),
        kv=[None],
        page_tables={0: torch.zeros(slots * (1 + spec_k), 4, dtype=torch.int32)},
        columns=1 + spec_k,
    )
    model.verify_caches = {slots: state}
    model.cache = SimpleNamespace(
        batch=slots, kv=[None], page_tables={0: torch.zeros(slots, 4, dtype=torch.int32)}, columns=1
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
    # device row j * users + u answers slot u, column j
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


def stub_state(rows, columns, slots, table_width=4):
    return SimpleNamespace(
        batch=rows,
        kv=[None],
        page_tables={0: torch.zeros(rows, table_width, dtype=torch.int32)},
        columns=columns,
    )


def stub_generator(seen, ids):
    """Records each prepare_decode (tokens, positions, state) and answers every row with `ids` rows in order."""

    def prepare_decode(tokens, positions, *, page_table, kv_cache):
        seen.append({"tokens": tokens.reshape(-1).tolist(), "positions": positions.tolist(), "state": kv_cache})
        rows = len(positions)
        return {
            "tokens": ttnn.from_torch(torch.tensor(ids(rows), dtype=torch.int32).reshape(1, 1, 1, -1), dtype=ttnn.uint32),
            "logits": torch.zeros(1, 1, 32, 10),
        }

    return SimpleNamespace(
        prepare_decode=prepare_decode,
        configure_sampling=lambda **kw: None,
        replay=lambda entry, **kw: None,
        _host_logits=lambda tensor: tensor,
    )


def serving_model(slots, spec_k, row_sizes):
    """A model with the full set of decode and verify traces of a `slots`-slot server (stubbed device)."""
    model = spec_adapter(slots, spec_k)
    model.row_sizes = row_sizes
    model.cache = stub_state(slots, 1, slots)
    model.cache.prefill_history = {}
    model.sub_caches = {n: stub_state(n, 1, slots) for n in row_sizes}
    model.verify_caches = {users: stub_state(users * (1 + k), 1 + k, slots) for users, k in model.verify_shapes}
    model.read_decode_output = lambda output: output
    return model


def test_decode_runs_the_smallest_trace_that_covers_the_live_rows():
    model = serving_model(8, 0, (1, 4))
    model.verify_shapes = ()
    seen = []
    model.generator = stub_generator(seen, lambda rows: list(range(32)))
    tables = [torch.ones(8, 4, dtype=torch.int32)]
    for live, expected in [(1, 1), (3, 4), (4, 4), (5, 8)]:
        positions = torch.tensor([10] * live + [-1] * (8 - live))
        model.decode_forward(
            torch.ones(8, 1, dtype=torch.int32), positions, None, model.cache, page_tables_per_layer=tables
        )
        state = seen[-1]["state"]
        assert state.batch == expected and len(seen[-1]["positions"]) == expected
        assert state is (model.cache if expected == 8 else model.sub_caches[expected])


def test_verify_with_fewer_users_than_slots_uses_the_smallest_verify_trace():
    model = serving_model(8, 5, (1, 4))
    seen = []
    model.generator = stub_generator(seen, lambda rows: list(range(100, 100 + rows)))
    block = torch.arange(48).reshape(8, 6)
    positions = torch.full((8, 6), -1)
    positions[0] = torch.arange(7, 13)
    out = model._verify(block, positions, model.cache, [torch.ones(8, 4, dtype=torch.int32)], None)
    assert seen[-1]["state"] is model.verify_caches[1] and len(seen[-1]["positions"]) == 6
    # one user: row j answers column j; slots without a live request answer -1
    assert out.argmax_ids[0].tolist() == [100, 101, 102, 103, 104, 105]
    assert (out.argmax_ids[1:] == -1).all()


def test_two_to_four_users_use_the_sixteen_row_trace_with_three_drafts():
    model = serving_model(8, 5, (1, 4))
    seen = []
    model.generator = stub_generator(seen, lambda rows: list(range(rows)))
    block = torch.arange(48).reshape(8, 6)
    positions = torch.full((8, 6), -1)
    positions[:3, :4] = torch.arange(20, 24)
    out = model._verify(block, positions, model.cache, [torch.ones(8, 4, dtype=torch.int32)], None)
    state = seen[-1]["state"]
    assert state is model.verify_caches[4] and state.batch == 16
    # four users x four columns: row j * 4 + u answers column j of user u; no column beyond the three drafts
    assert out.argmax_ids[:4, :4].tolist() == [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
    assert (out.argmax_ids[:, 4:] == -1).all() and (out.argmax_ids[4:] == -1).all()


def test_a_draftless_block_above_the_verify_traces_runs_as_a_plain_decode():
    model = serving_model(8, 5, (1, 4))
    seen = []
    model.generator = stub_generator(seen, lambda rows: [1000 + row for row in range(32)])
    live = 5
    block = torch.zeros(8, 6, dtype=torch.long)
    block[:, 0] = torch.arange(8) + 50
    positions = torch.full((8, 6), -1)
    positions[:live, 0] = 30
    valid = torch.zeros(8, dtype=torch.int32)
    out = model._verify(block, positions, model.cache, [torch.ones(8, 4, dtype=torch.int32)], None, valid)
    # five live requests: the 8-row plain trace (the full one), column 0 only
    assert seen[-1]["state"] is model.cache
    assert seen[-1]["tokens"] == list(range(50, 58)) and seen[-1]["positions"] == [30] * 5 + [-1] * 3
    assert out.argmax_ids[:, 0].tolist() == [1000 + row for row in range(8)]
    assert (out.argmax_ids[:, 1:] == -1).all()


@pytest.mark.parametrize("valid", [None, torch.tensor([0, 0, 2, 0, 0, 0, 0, 0])])
def test_a_block_with_drafts_above_the_verify_traces_is_refused(valid):
    model = serving_model(8, 5, (1, 4))
    model.generator = stub_generator([], lambda rows: list(range(32)))
    positions = torch.full((8, 6), -1)
    positions[:5, 0] = 30
    with pytest.raises(ValueError, match="No verify trace covers 5"):
        model._verify(torch.zeros(8, 6, dtype=torch.long), positions, model.cache, [None], None, valid)


def test_plain_ids_pick_the_trace_by_live_rows_and_pad_inactive_rows():
    model = serving_model(8, 5, (1, 4))
    seen = []
    model.generator = stub_generator(seen, lambda rows: list(range(32)))
    ids = model._decode_plain_ids(
        torch.tensor([7, 8, 9, 0, 0, 0, 0, 0]),
        torch.tensor([3, 3, 3, -1, -1, -1, -1, -1]),
        model.cache,
        [torch.ones(8, 4, dtype=torch.int32)],
        None,
    )
    assert seen[-1]["state"] is model.sub_caches[4] and seen[-1]["tokens"] == [7, 8, 9, 0]
    assert ids.tolist() == list(range(8))


@pytest.mark.parametrize("remap", [None, [1, 0, 2, 3, 4, 5, 6, 7]])
@pytest.mark.parametrize("plain_block", [False, True])
def test_speculative_steps_release_the_prefill_tail_of_requests_entering_decode(remap, plain_block):
    model = serving_model(8, 5, (1, 4))
    model.generator = stub_generator([], lambda rows: list(range(32)))
    model.prefill_slot_keys = {0: 11, 1: 12}
    model.cache.prefill_history = {11: object(), 12: object()}
    block = torch.zeros(8, 6, dtype=torch.long)
    positions = torch.full((8, 6), -1)
    live = 5 if plain_block else 1
    positions[:live, 0] = 30
    valid = torch.zeros(8, dtype=torch.int32) if plain_block else None
    model.decode_forward(
        block,
        positions,
        None,
        model.cache,
        page_tables_per_layer=[torch.ones(8, 4, dtype=torch.int32)],
        spec_mode="argmax_ids",
        num_valid_drafts=valid,
        slot_remap=None if remap is None else torch.tensor(remap),
    )
    if plain_block:
        # rows 0-4 decode: both slots with a private tail are released
        assert model.prefill_slot_keys == {} and model.cache.prefill_history == {}
    else:
        # only row 0 decodes: its slot (0, or 1 after the remap) is released, the other slot's prefill continues
        kept = 1 if remap is None else 0
        assert model.prefill_slot_keys == {kept: 11 + kept} and set(model.cache.prefill_history) == {11 + kept}


def test_warm_up_captures_every_decode_and_verify_trace(monkeypatch):
    model = serving_model(8, 5, (1, 4))
    captured = []
    model.generator = SimpleNamespace(
        _tables=lambda state, tables: None,
        configure_sampling=lambda **kw: None,
        prepare_entries=lambda states: captured.extend(states) or [{"tokens": None}] * len(states),
    )
    model.mesh = None
    monkeypatch.setattr(ttnn, "get_memory_view", lambda *a: SimpleNamespace(total_bytes_allocated_per_bank=0))
    model.warmup_model_decode(model.cache, True)
    expected = [model.cache, *model.sub_caches.values(), *model.verify_caches.values()]
    assert len(captured) == len(expected) == 1 + 2 + 2 and all(any(c is e for c in captured) for e in expected)
