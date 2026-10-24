# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P1d no-device tests: the continuous-batching policy over a fake decoder, the decoder's host page/park logic and
``Qwen38TTNNDevicePositionRow.park``."""

import copy
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import (
    WARM_TOKEN_ID,
    Qwen38ContinuousBatch,
    Qwen38PageAllocator,
    Qwen38PlainLaneDecoder,
    Qwen38Request,
    lane_page_count,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MAX_LANES, Qwen38TTNNDevicePositionRow
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging


class FakeDecoder:
    """Host stand-in with the decoder's policy-facing interface; a step maps token t to t + 1 on every lane."""

    def __init__(self, lanes, page_size=0, num_pages=0, seed=0):
        self.lanes = lanes
        self.paging = None if not page_size else SimpleNamespace(page_size=page_size)
        self.paged = bool(page_size)
        self.page_size = page_size
        self.allocator = Qwen38PageAllocator(num_pages, seed) if page_size else None
        self.page_rows = [[] for _ in range(lanes)]
        self.dirty = False
        self.table_writes = 0
        self.table_at_step: list[list[list[int]]] = []
        self.admits: list[tuple[int, int, int | None]] = []  # (lane, prompt_len, page_count)
        self.parks: list[list[int]] = []
        self.steps: list[list[int]] = []
        self.released: list[int] = []
        self.last_admit = {"prefill_s": 0.5, "import_s": 0.25}
        self.on_step = None

    def free_pages(self):
        return len(self.allocator.free) if self.paged else 0

    def admit(self, lane, prompt_ids, page_count=None):
        assert not self.page_rows[lane], "admit into a lane that holds pages"
        self.admits.append((lane, len(prompt_ids), page_count))
        if self.paged:
            self.page_rows[lane] = self.allocator.allocate(page_count)
            self.table_writes += 1
            self.dirty = False
        return 1000 * (lane + 1)

    def grow_lane(self, lane):
        self.page_rows[lane] += self.allocator.allocate(1)
        self.dirty = True

    def release_lane(self, lane, write_table=True):
        self.released.append(lane)
        if self.paged:
            self.allocator.release(self.page_rows[lane])
            self.page_rows[lane] = []
            self.dirty = True

    def sync_page_table(self):
        if not self.dirty:
            return False
        self.table_writes += 1
        self.dirty = False
        return True

    def park_idle(self, lanes):
        self.parks.append(list(lanes))
        return False

    def step(self, tokens):
        assert len(tokens) == self.lanes
        self.steps.append(list(tokens))
        self.table_at_step.append(copy.deepcopy(self.page_rows))
        if self.on_step is not None:
            self.on_step(tokens)
        return [t + 1 for t in tokens]


def prompt(length):
    return list(range(length))


def run(decoder, specs):
    """specs: (id, prompt_len, max_new, arrival)."""

    batch = Qwen38ContinuousBatch(decoder, [Qwen38Request(i, prompt(p), n, a) for i, p, n, a in specs])
    return batch, batch.run()


def tokens_of(report, request_id):
    return next(r["tokens"] for r in report["requests"] if r["request_id"] == request_id)


def test_tokens_are_t0_plus_one_per_step_and_idle_lanes_get_the_warm_token():
    decoder = FakeDecoder(4)
    _, report = run(decoder, [(0, 10, 4, 0), (1, 20, 2, 0)])
    assert tokens_of(report, 0) == [1000, 1001, 1002, 1003]
    assert tokens_of(report, 1) == [2000, 2001]
    assert len(decoder.steps) == 3  # 4 tokens = t0 + 3 steps
    assert decoder.steps[0] == [1000, 2000, WARM_TOKEN_ID, WARM_TOKEN_ID]
    assert decoder.steps[1] == [1001] + [WARM_TOKEN_ID] * 3  # lane 1 finished (2 tokens) and was released
    assert decoder.steps[2] == [1002] + [WARM_TOKEN_ID] * 3


def test_admission_order_lowest_free_lane_and_waiting_for_a_lane():
    decoder = FakeDecoder(3)
    # r0 finishes first (2 tokens): its lane 0 goes to r3; r4 waits until r3 and r1 are done (step 3), lowest lane 0
    specs = [(0, 5, 2, 0), (1, 5, 4, 0), (2, 5, 8, 0), (3, 5, 3, 0), (4, 5, 3, 0)]
    batch, report = run(decoder, specs)
    assert batch.admission_order == [(0, 0), (1, 1), (2, 2), (3, 0), (4, 0)]
    by_id = {r["request_id"]: r for r in report["requests"]}
    assert by_id[3]["admit_step"] == 1 and by_id[4]["admit_step"] == 3
    assert all(len(by_id[i]["tokens"]) == n for i, _, n, _ in specs)


def test_lowest_free_lane_when_several_lanes_free_up():
    decoder = FakeDecoder(4)
    specs = [(0, 5, 3, 0), (1, 5, 2, 0), (2, 5, 3, 0), (3, 5, 9, 0), (4, 5, 2, 2), (5, 5, 2, 2)]
    batch, _ = run(decoder, specs)
    # at step 2 lanes 0, 1 and 2 are all free again: the new requests take lanes 0 and 1
    assert batch.admission_order[4:] == [(4, 0), (5, 1)]


def test_requests_wait_for_their_arrival_step_and_idle_batches_do_not_replay():
    decoder = FakeDecoder(2)
    batch, report = run(decoder, [(0, 5, 3, 7), (1, 5, 3, 7), (2, 5, 2, 20)])
    assert [s["step"] for s in report["step_log"]] == [7, 8, 20]  # no replay while nothing is active
    assert len(decoder.steps) == 3
    by_id = {r["request_id"]: r for r in report["requests"]}
    assert by_id[0]["admit_step"] == 7 and by_id[2]["admit_step"] == 20


def test_max_new_tokens_one_is_admitted_and_released_without_a_step():
    decoder = FakeDecoder(2, page_size=64, num_pages=20)
    batch, report = run(decoder, [(0, 100, 1, 0)])
    assert report["steps"] == 0 and decoder.steps == []
    assert tokens_of(report, 0) == [1000]
    assert decoder.released == [0] and decoder.free_pages() == 19
    # a one-token request frees its lane in the same iteration: the next request takes lane 0 again
    decoder = FakeDecoder(1)
    batch, report = run(decoder, [(0, 5, 1, 0), (1, 5, 1, 0), (2, 5, 3, 0)])
    assert batch.admission_order == [(0, 0), (1, 0), (2, 0)]
    assert len(decoder.steps) == 2 and tokens_of(report, 2) == [1000, 1001, 1002]


@pytest.mark.parametrize("page_size", [32, 64])
def test_pages_grow_on_demand_at_page_boundaries(page_size):
    prompt_len, max_new = page_size - 2, 2 * page_size
    decoder = FakeDecoder(2, page_size=page_size, num_pages=30)
    batch, report = run(decoder, [(0, prompt_len, max_new, 0)])
    assert decoder.admits == [(0, prompt_len, lane_page_count(prompt_len, 0, page_size))]
    assert decoder.admits[0][2] == 1  # prompt rows + the first decode row fit one page
    for j, rows in enumerate(decoder.table_at_step):
        row = prompt_len + j  # the row step j writes
        assert len(rows[0]) == row // page_size + 1, f"step {j} (row {row})"
        assert rows[1] == []
    assert len(decoder.table_at_step) == max_new - 1
    assert report["requests"][0]["pages_used"] == lane_page_count(prompt_len, max_new - 1, page_size)
    assert report["requests"][0]["pages_used"] <= report["requests"][0]["budget_pages"]
    assert decoder.free_pages() == 29  # all freed


def test_page_table_is_rewritten_only_when_a_row_changed():
    decoder = FakeDecoder(3, page_size=32, num_pages=40)
    specs = [(0, 10, 50, 0), (1, 20, 5, 0), (2, 15, 6, 3)]
    batch, report = run(decoder, specs)
    log = report["step_log"]
    # r1 (5 tokens) is released at step 4, r2 (6 tokens, admitted at step 3) at step 8, and r0 first writes a row
    # of its second page at step 22 (rows 10 + j): those are the only steps that rewrite the table
    assert {e["step"] for e in log if e["table_write"]} == {4, 8, 22}
    # admissions write the table themselves (admit counted), steps wrote only on change
    assert decoder.table_writes == len(decoder.admits) + sum(e["table_write"] for e in log)


def test_release_frees_pages_and_the_reservation_and_waiting_requests_use_them():
    page_size = 64
    # budgets: r0 ceil((200+30+1)/64)=4, r1 ceil((300+10+1)/64)=5; the pool holds 6 pages
    decoder = FakeDecoder(4, page_size=page_size, num_pages=7)
    assert lane_page_count(200, 30, page_size) == 4 and lane_page_count(300, 10, page_size) == 5
    batch = Qwen38ContinuousBatch(decoder, [Qwen38Request(0, prompt(200), 30, 0), Qwen38Request(1, prompt(300), 10, 0)])
    seen = []

    def check(tokens):
        seen.append((decoder.free_pages(), batch.outstanding_reservation()))
        assert decoder.free_pages() - batch.outstanding_reservation() >= 0

    decoder.on_step = check
    report = batch.run()
    by_id = {r["request_id"]: r for r in report["requests"]}
    # r1 does not fit next to r0 (4 + 5 > 6): it waits until r0 is released
    assert by_id[1]["admit_step"] == by_id[0]["finish_step"]
    assert decoder.free_pages() == 6 and batch.outstanding_reservation() == 0
    assert report["page_reuses"] > 0 and report["max_pages_in_use"] <= 6


def test_fifo_head_blocks_later_smaller_requests():
    page_size = 64
    decoder = FakeDecoder(4, page_size=page_size, num_pages=1 + 6)
    # r0 (budget 4) runs; r1 (budget 5) does not fit and blocks r2 (budget 1), which would fit
    specs = [(0, 200, 20, 0), (1, 300, 5, 0), (2, 10, 3, 0)]
    batch, _ = run(decoder, specs)
    assert [i for i, _ in batch.admission_order] == [0, 1, 2]
    assert batch.records[2]["admit_step"] >= batch.records[1]["admit_step"]
    assert batch.records[1]["admit_step"] == batch.records[0]["finish_step"]


def test_request_larger_than_the_pool_is_refused(expect_error):
    decoder = FakeDecoder(2, page_size=64, num_pages=4)
    with expect_error(ValueError, "needs"):
        Qwen38ContinuousBatch(decoder, [Qwen38Request(0, prompt(300), 10, 0)])


def test_idle_lanes_are_parked_before_every_step():
    decoder = FakeDecoder(4)
    run(decoder, [(0, 5, 4, 0), (1, 5, 2, 0), (2, 5, 3, 2)])
    assert len(decoder.parks) == len(decoder.steps) == 4
    # lane 1 is released after its second token; r2 (arrival 2) takes lane 1; lane 0 is released after step 2
    assert decoder.parks == [[2, 3], [1, 2, 3], [2, 3], [0, 2, 3]]


def test_no_idle_lane_no_park_call():
    decoder = FakeDecoder(2)
    run(decoder, [(0, 5, 3, 0), (1, 5, 3, 0)])
    assert decoder.parks == []


def test_freed_pages_are_reused_by_later_requests():
    page_size = 32
    decoder = FakeDecoder(2, page_size=page_size, num_pages=1 + 4)
    specs = [(0, 40, 20, 0), (1, 40, 20, 0), (2, 40, 20, 0), (3, 40, 20, 0)]
    batch, report = run(decoder, specs)
    assert report["page_reuses"] == 4  # a 4-page pool serves 4 requests of 2 pages each
    assert report["max_pages_in_use"] <= 4
    assert all(len(r["tokens"]) == 20 for r in report["requests"])


def test_metrics_report():
    decoder = FakeDecoder(2, page_size=64, num_pages=20)
    clock = iter(range(1000))
    batch = Qwen38ContinuousBatch(
        decoder, [Qwen38Request(0, prompt(100), 5, 0), Qwen38Request(1, prompt(100), 3, 0)], clock=lambda: next(clock)
    )
    report = batch.run()
    assert report["generated_tokens"] == 8 and report["steps"] == 4
    assert report["peak_active_lanes"] == 2
    assert report["effective_tok_s"] == pytest.approx(8 / report["wall_s"])
    r0 = report["requests"][0]
    assert r0["prefill_s"] == 0.5 and r0["import_s"] == 0.25 and r0["pages_used"] >= 1


# -- the real decoder's host logic (no device) ------------------------------------------------------------


def host_decoder(lanes=4, page_size=64, num_pages=20):
    decoder = Qwen38PlainLaneDecoder(lanes, paging=Qwen38LanePaging(page_size, num_pages, 128))
    writes = []
    decoder.model = SimpleNamespace(write_lane_page_table=lambda ls, rows: writes.append([list(r) for r in rows]))
    decoder.ls = SimpleNamespace()
    return decoder, writes


def test_decoder_grow_release_and_sync_write_the_table_once_per_change():
    decoder, writes = host_decoder()
    decoder.page_rows[0] = decoder.allocator.allocate(1)
    decoder.positions[0], decoder.contexts[0] = 10, (1, 2)
    assert decoder.sync_page_table() is False and writes == []
    decoder.grow_lane(0)
    decoder.page_rows[1] = decoder.allocator.allocate(2)
    decoder.grow_lane(1)
    assert writes == []  # nothing written until the step
    assert decoder.sync_page_table() is True and len(writes) == 1
    assert [len(r) for r in writes[0][:3]] == [2, 3, 0]
    assert decoder.sync_page_table() is False
    free = len(decoder.allocator.free)
    decoder.release_lane(0, write_table=False)
    assert writes[1:] == [] and len(decoder.allocator.free) == free + 2
    assert decoder.positions[0] is None and decoder.contexts[0] is None and decoder.page_rows[0] == []
    assert decoder.sync_page_table() is True and writes[-1][0] == []
    decoder.release_lane(1)  # default: writes the table at once
    assert writes[-1][1] == [] and decoder.sync_page_table() is False


def test_decoder_grow_beyond_the_table_is_refused(expect_error):
    decoder, _ = host_decoder(page_size=64, num_pages=300)
    decoder.page_rows[0] = decoder.allocator.allocate(128)
    with expect_error(RuntimeError, "table"):
        decoder.grow_lane(0)


def test_unpaged_decoder_release_only_clears_host_state():
    decoder = Qwen38PlainLaneDecoder(4)
    decoder.positions[2], decoder.contexts[2] = 30, (3, 4)
    decoder.release_lane(2)
    assert decoder.positions[2] is None and decoder.contexts[2] is None


def position_row(lanes, positions, shifts=None, residue_free=True):
    values = list(positions) + [0] * (MAX_LANES - len(positions))
    row = Qwen38TTNNDevicePositionRow(
        row=None,
        block_start_mask_row=None,
        lanes=lanes,
        positions=values,
        mesh_device=None,
        mesh_contract=None,
        shift_row=None,
        shifts=list(shifts) if shifts else [0] * MAX_LANES,
        residue_free=residue_free,
    )
    row.mirror_writes = 0

    def count():
        row.mirror_writes += 1

    row._write_mirror = count
    return row


def test_position_row_park_zeroes_positions_and_shifts_of_the_given_lanes_with_one_write():
    shifts = [0] * MAX_LANES
    shifts[1] = 4
    row = position_row(4, [10, 11, 12, 13], shifts)
    row.park([1, 3])
    assert row.positions[:4] == [10, 0, 12, 0] and row.shifts[1] == 0 and row.mirror_writes == 1
    row.park([])
    assert row.mirror_writes == 1  # nothing to park: no write


def test_position_row_park_validates(expect_error):
    row = position_row(4, [1, 2, 3, 4])
    with expect_error(ValueError, "lane"):
        row.park([4])
    with expect_error(ValueError, "lane"):
        row.park([-1])
    assert row.mirror_writes == 0 and row.positions[:4] == [1, 2, 3, 4]
    with expect_error(RuntimeError, "residue"):
        position_row(4, [1, 2, 3, 4], residue_free=False).park([0])


def test_decoder_park_idle_writes_only_when_an_idle_lane_is_off_zero():
    decoder, _ = host_decoder()
    row = position_row(4, [0, 0, 0, 0])
    decoder.ls = SimpleNamespace(position=row)
    assert decoder.park_idle([2, 3]) is False and row.mirror_writes == 0
    row.positions[2] = 5  # a replay advanced the idle lane
    row.positions[0] = 40  # an active lane: not touched
    assert decoder.park_idle([2, 3]) is True and row.mirror_writes == 1
    assert row.positions[:4] == [40, 0, 0, 0]
    assert decoder.park_idle([]) is False


def test_decoder_step_resets_idle_lane_ple_contexts():
    decoder, _ = host_decoder()
    decoder.positions[0], decoder.contexts[0] = 10, (7, 8)
    decoder.contexts[2] = (9, 9)  # stale context of an idle lane
    seen = {}

    def write_ple(ls, tokens, contexts):
        seen["contexts"] = list(contexts)
        return [(1, 1)] * len(tokens)

    decoder.model = SimpleNamespace(
        write_lane_ple_rows=write_ple, write_lane_tokens=lambda ls, tokens: None, mesh_device=None
    )
    decoder.ls = SimpleNamespace(position=SimpleNamespace(advance_replayed=lambda: None), token_row=None)
    decoder.mesh = None
    decoder.trace_ids = [0]
    import models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode as module

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(module.ttnn, "_ttnn_execute_trace", lambda *a, **k: None)
        mp.setattr(module.ttnn, "to_torch", lambda t: module.torch.arange(4))
        mp.setattr(module.ttnn, "get_device_tensors", lambda t: [t])
        decoder.step([5, WARM_TOKEN_ID, WARM_TOKEN_ID, WARM_TOKEN_ID])
    assert seen["contexts"] == [(7, 8), None, None, None]
    assert decoder.contexts[2] is None and decoder.contexts[0] == (1, 1)
