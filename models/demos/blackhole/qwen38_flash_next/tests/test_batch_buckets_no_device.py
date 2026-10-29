# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""W3 no-device tests: the decode-width buckets of ``Qwen38PlainLaneDecoder`` (validation, ``bucket_for_step``, the
step's trace choice and token slicing, the reads of the active bucket) over a fake model and fake trace calls."""

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_batch_decode as batch
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import Qwen38LaneBucket, Qwen38PlainLaneDecoder
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging

PAGING = Qwen38LanePaging(64, 20, 128)
BUCKETS = (1, 2, 4, 8, 16, 32)


class Rig:
    """A decoder with fake buckets: trace id 100 + R per bucket, fake model and fake trace calls (recorded)."""

    def __init__(self, monkeypatch, lanes=32, buckets=BUCKETS, paging=PAGING):
        self.decoder = decoder = Qwen38PlainLaneDecoder(lanes, paging=paging, buckets=buckets)
        self.calls = []
        self.copies = []
        self.token_row = torch.arange(1000, 1000 + lanes, dtype=torch.float32)
        decoder.mesh = "mesh"
        decoder.ls = SimpleNamespace(
            name="parent", token_row=self.token_row, position=SimpleNamespace(advance_replayed=self.advance)
        )
        for r, bucket in decoder.bucket_sets.items():
            bucket.ls = decoder.ls if r == lanes else SimpleNamespace(name=f"view{r}")
            bucket.trace_id = 100 + r
            bucket.candidate_trace_id = 200 + r
            bucket.normalizer_trace_id = 300 + r
            bucket.candidate_buffer = torch.arange(r * 256, dtype=torch.float32).reshape(1, 1, r, 256)
            bucket.normalizer_buffer = torch.ones(1, 1, r, 4)
            bucket.normalizer_inv_temperature = f"inv{r}"
            sizes = tuple(s for s in (1, 2) if s <= r)  # the real sizes are the powers of two up to R
            bucket.row_read_buckets = sizes
            bucket.row_read_indices = {s: f"idx{r}_{s}" for s in sizes}
            bucket.row_read_blocks = {s: torch.zeros(1, s, 4) for s in sizes}
            bucket.row_read_traces = {s: 300 + 100 * s + r for s in sizes}
        decoder.active = decoder.bucket_sets[lanes]
        decoder.model = SimpleNamespace(write_lane_ple_rows=self.write_ple, write_lane_tokens=self.write_tokens)
        monkeypatch.setattr(batch.ttnn, "_ttnn_execute_trace", self.execute, raising=False)
        monkeypatch.setattr(batch.ttnn, "get_device_tensors", lambda t: [t])
        monkeypatch.setattr(batch.ttnn, "to_torch", lambda t, **k: t)
        monkeypatch.setattr(batch.ttnn, "from_torch", lambda t, **k: t)
        monkeypatch.setattr(batch.ttnn, "copy_host_to_device_tensor", lambda host, dev: self.copies.append((host, dev)))
        monkeypatch.setattr(batch.ttnn, "ConcatMeshToTensor", lambda mesh, dim: None)
        monkeypatch.setattr(batch, "replicate_tensor_2d_mesh_mapper", lambda mesh: None)

    def advance(self):
        self.calls.append(("advance",))

    def write_ple(self, ls, tokens, contexts):
        self.calls.append(("ple", ls.name, list(tokens), list(contexts)))
        return [("ctx", t) for t in tokens]

    def write_tokens(self, ls, tokens):
        self.calls.append(("tokens", ls.name, list(tokens)))

    def execute(self, mesh, trace_id, cq_id=0, blocking=True):
        self.calls.append(("trace", trace_id))

    def traces(self):
        return [c[1] for c in self.calls if c[0] == "trace"]

    def live(self, *lanes):
        for lane in lanes:
            self.decoder.positions[lane] = 10 + lane


# -- construction -------------------------------------------------------------------------------------------


def test_buckets_default_is_the_one_full_bucket():
    decoder = Qwen38PlainLaneDecoder(8)
    assert decoder.buckets == (8,) and list(decoder.bucket_sets) == [8]
    assert decoder.active is decoder.bucket_sets[8] and decoder.active.lanes == 8
    assert decoder.force_bucket is None and decoder.bucket_steps == {8: 0}
    assert not decoder.has_candidates and not decoder.has_normalizer and not decoder.has_row_read
    assert decoder.row_read_buckets == () and decoder.row_read_traces == {}


def test_buckets_are_sorted_and_distinct():
    decoder = Qwen38PlainLaneDecoder(32, paging=PAGING, buckets=[32, 8, 8, 1, 16])
    assert decoder.buckets == (1, 8, 16, 32)
    assert list(decoder.bucket_sets) == [1, 8, 16, 32] and decoder.bucket_steps == {1: 0, 8: 0, 16: 0, 32: 0}
    assert all(isinstance(b, Qwen38LaneBucket) and b.lanes == r for r, b in decoder.bucket_sets.items())


def test_row_read_buckets_follow_each_bucket_width():
    decoder = Qwen38PlainLaneDecoder(32, paging=PAGING, buckets=(4, 32), candidates=True, row_read=True)
    assert decoder.bucket_sets[4].row_read_buckets == (1, 2, 4)
    assert decoder.bucket_sets[32].row_read_buckets == (1, 2, 4, 8, 16, 32)
    assert decoder.row_read_buckets == (1, 2, 4, 8, 16, 32)  # the active (full) bucket


@pytest.mark.parametrize(
    "lanes, buckets, paging, message",
    [
        (32, (8,), PAGING, "largest"),
        (32, (0, 32), PAGING, "buckets must be"),
        (32, (33, 32), PAGING, "buckets must be"),
        (32, (-1, 32), PAGING, "buckets must be"),
        (32, (), PAGING, "buckets must be"),
        (16, (8, 16), PAGING, "32 paged lanes"),
        (32, (8, 32), None, "32 paged lanes"),
    ],
)
def test_bucket_validation_rejects(lanes, buckets, paging, message, expect_error):
    with expect_error(ValueError, message):
        Qwen38PlainLaneDecoder(lanes, paging=paging, buckets=buckets)


def test_one_bucket_needs_neither_32_lanes_nor_paging():
    assert Qwen38PlainLaneDecoder(8, buckets=(8,)).buckets == (8,)
    assert Qwen38PlainLaneDecoder(16, buckets=[16]).buckets == (16,)


# -- bucket_for_step ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "live, expected",
    [((), 1), ((0,), 1), ((0, 5), 8), ((8,), 16), ((31,), 32), ((1,), 2), ((2,), 4), ((3,), 4), ((15,), 16)],
)
def test_bucket_for_step(monkeypatch, live, expected):
    rig = Rig(monkeypatch)
    rig.live(*live)
    assert rig.decoder.bucket_for_step() == expected


def test_bucket_for_step_without_the_one_bucket_is_the_full_width(monkeypatch):
    rig = Rig(monkeypatch, lanes=8, buckets=None, paging=None)
    assert rig.decoder.bucket_for_step() == 8
    rig.live(3)
    assert rig.decoder.bucket_for_step() == 8


def test_force_bucket(monkeypatch, expect_error):
    rig = Rig(monkeypatch)
    rig.live(0, 5)
    rig.decoder.force_bucket = 32
    assert rig.decoder.bucket_for_step() == 32
    rig.decoder.force_bucket = 8
    assert rig.decoder.bucket_for_step() == 8
    rig.decoder.force_bucket = 4  # lane 5 is live
    with expect_error(ValueError, "does not hold live lane 5"):
        rig.decoder.bucket_for_step()
    rig.decoder.force_bucket = 3  # not a bucket
    with expect_error(ValueError, "not one of the buckets"):
        rig.decoder.bucket_for_step()


# -- step ---------------------------------------------------------------------------------------------------


def test_step_replays_the_smallest_bucket_and_slices_the_tokens(monkeypatch):
    rig = Rig(monkeypatch)
    decoder = rig.decoder
    rig.live(0, 5)
    decoder.contexts[0] = "c0"
    decoder.contexts[5] = "c5"
    tokens = list(range(32))
    out = decoder.step(tokens)
    assert rig.traces() == [108]
    assert rig.calls[0] == ("ple", "view8", list(range(8)), ["c0", None, None, None, None, "c5", None, None])
    assert rig.calls[1] == ("tokens", "view8", list(range(8)))
    assert rig.calls[-1] == ("advance",)
    assert out == [1000 + i for i in range(8)] + [-1] * 24 and len(out) == 32
    assert decoder.active is decoder.bucket_sets[8] and decoder.bucket_steps[8] == 1
    assert decoder.contexts[0] == ("ctx", 0) and decoder.contexts[5] == ("ctx", 5)
    assert all(decoder.contexts[u] is None for u in (1, 2, 3, 4, 6, 7, 8, 31))  # idle lanes: None again
    assert len(decoder.contexts) == 32


def test_step_switches_buckets_with_the_live_lanes(monkeypatch):
    rig = Rig(monkeypatch)
    decoder = rig.decoder
    rig.live(0)
    decoder.step([0] * 32)
    rig.live(9)
    decoder.step([0] * 32)
    decoder.force_bucket = 32
    out = decoder.step([0] * 32)
    assert rig.traces() == [101, 116, 132]
    assert out == [1000 + i for i in range(32)]
    assert decoder.bucket_steps == {1: 1, 2: 0, 4: 0, 8: 0, 16: 1, 32: 1}
    assert decoder.active.lanes == 32
    ples = [c for c in rig.calls if c[0] == "ple"]
    assert [c[1] for c in ples] == ["view1", "view16", "parent"] and [len(c[2]) for c in ples] == [1, 16, 32]


def test_step_with_buckets_off_is_todays_step(monkeypatch):
    rig = Rig(monkeypatch, lanes=8, buckets=None, paging=None)
    decoder = rig.decoder
    rig.live(2)
    tokens = list(range(8))
    out = decoder.step(tokens)
    assert rig.traces() == [108] and decoder.bucket_steps == {8: 1}
    assert rig.calls[0][:3] == ("ple", "parent", tokens) and rig.calls[1] == ("tokens", "parent", tokens)
    assert out == [1000 + i for i in range(8)]  # no -1 entries
    assert len(decoder.captures) == 0 and decoder.active is decoder.bucket_sets[8]


# -- reads of the active bucket -----------------------------------------------------------------------------


def test_read_candidates_follow_the_active_bucket(monkeypatch):
    rig = Rig(monkeypatch)
    decoder = rig.decoder
    assert decoder.read_candidates().shape == (32, 256) and rig.traces() == [232]
    rig.live(0, 5)
    decoder.step([0] * 32)
    rows = decoder.read_candidates()
    assert rows.shape == (8, 256) and rig.traces()[-1] == 208
    assert decoder.candidate_trace_id == 208 and decoder.has_candidates and decoder.has_normalizer
    assert decoder.normalizer_inv_temperature == "inv8" and decoder.row_read_traces == {1: 408, 2: 508}


def test_read_candidates_normalized_length_rules(monkeypatch, expect_error):
    rig = Rig(monkeypatch)
    decoder = rig.decoder
    rig.live(0, 5)
    decoder.step([0] * 32)
    inv = [float(i) for i in range(32)]
    rows, lse = decoder.read_candidates_normalized(inv)  # lanes entries: the first 8 are used
    assert rows.shape == (8, 256) and lse.shape == (8, 4) and rig.traces()[-1] == 308
    host, dev = rig.copies[-1]
    assert dev == "inv8" and host.reshape(-1).tolist() == inv[:8]
    rows, lse = decoder.read_candidates_normalized(inv[:8])  # bucket width entries
    assert rows.shape == (8, 256) and rig.copies[-1][0].reshape(-1).tolist() == inv[:8]
    for bad in (7, 9, 16, 0):
        with expect_error(ValueError, "one inverse temperature per lane"):
            decoder.read_candidates_normalized([1.0] * bad)


def test_read_candidates_normalized_on_the_full_bucket_needs_all_lanes(monkeypatch, expect_error):
    rig = Rig(monkeypatch)
    rows, lse = rig.decoder.read_candidates_normalized([1.0] * 32)
    assert rows.shape == (32, 256) and lse.shape == (32, 4)
    with expect_error(ValueError, "one inverse temperature per lane"):
        rig.decoder.read_candidates_normalized([1.0] * 8)


def test_read_logits_rows_refuses_a_lane_outside_the_active_bucket(monkeypatch, expect_error):
    rig = Rig(monkeypatch)
    decoder = rig.decoder
    rig.live(0, 5)
    decoder.step([0] * 32)
    with expect_error(ValueError, "0..7"):
        decoder.read_logits_rows([0, 8])
    with expect_error(ValueError, "0..7"):
        decoder.read_logits_rows([31])
    rows = decoder.read_logits_rows([5, 0])
    assert rig.traces()[-1] == 508 and rows.shape[0] == 2
    assert [h.reshape(-1).tolist() for h, _ in rig.copies[-1:]] == [[5, 0]]
    assert rig.copies[-1][1] == "idx8_2"
    decoder.force_bucket = 32
    decoder.step([0] * 32)
    decoder.read_logits_rows([0, 8])  # a full-bucket step reads any lane
    assert rig.traces()[-1] == 532


def test_warm_bucket_reads_replays_every_bucket_and_restores_the_full_one(monkeypatch):
    rig = Rig(monkeypatch)
    decoder = rig.decoder
    rig.live(0)
    decoder.step([0] * 32)  # bucket 1 active
    rig.calls.clear()
    decoder.warm_bucket_reads()
    expected = []
    for r in BUCKETS:
        expected += [200 + r, 300 + r] + [300 + 100 * s + r for s in (1, 2) if s <= r]
    assert rig.traces() == expected
    assert decoder.active is decoder.bucket_sets[32]


# -- W5: compaction ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "live, buckets, expected",
    [
        ((), BUCKETS, []),
        ((0,), BUCKETS, []),
        ((0, 9), BUCKETS, [(9, 1)]),
        ((0, 1, 2, 17), BUCKETS, [(17, 3)]),
        ((5,), BUCKETS, [(5, 0)]),
        ((0, 1), BUCKETS, []),
        ((20, 25, 31), BUCKETS, [(31, 0), (25, 1), (20, 2)]),  # highest first, into the lowest free lanes in order
        ((3, 4, 5), BUCKETS, [(5, 0), (4, 1)]),  # three live: bucket 4, lanes 4 and 5 are above it
        ((0, 9), None, []),  # one bucket: nothing to gain
    ],
)
def test_compaction_moves(monkeypatch, live, buckets, expected):
    rig = Rig(monkeypatch, buckets=buckets)
    rig.live(*live)
    assert rig.decoder.compaction_moves() == expected


def test_compaction_moves_treat_a_lane_with_a_page_row_as_taken(monkeypatch):
    rig = Rig(monkeypatch)
    rig.live(0, 9)
    rig.decoder.page_rows[1] = [7]  # not live, but still holds a page row
    assert rig.decoder.compaction_moves() == []  # b = 2, lane 9 above it, but lane 1 is not free: no pair
    rig.decoder.page_rows[1] = []
    assert rig.decoder.compaction_moves() == [(9, 1)]


class MoveRig(Rig):
    """``Rig`` plus the move's device surface: a fake pager, position row (mirror, admit, park) and clock."""

    def __init__(self, monkeypatch):
        super().__init__(monkeypatch)
        decoder = self.decoder
        lanes = decoder.lanes
        position = decoder.ls.position
        position.positions, position.shifts = [0] * lanes, [0] * lanes
        position.park = lambda parked: self.calls.append(("park", list(parked)))
        decoder.model.admit_lane_position = lambda ls, lane, pos, rope_shift=0: self.calls.append(
            ("admit_position", ls.name, lane, pos, rope_shift)
        )
        decoder.pager_l = SimpleNamespace(
            _run_pack=lambda lane: self.calls.append(("pack", lane)),
            _run_unpack=lambda lane: self.calls.append(("unpack", lane)),
            _fence=lambda: self.calls.append(("fence",)),
        )
        monkeypatch.setattr(batch.ttnn, "synchronize_device", lambda mesh: self.calls.append(("sync",)), raising=False)
        monkeypatch.setattr(batch, "corruptible_allocation_scope", lambda mesh: _scope(self.calls))


class _scope:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        self.calls.append(("scope+",))

    def __exit__(self, *exc):
        self.calls.append(("scope-",))
        return False


def test_move_lane_moves_the_host_state_and_the_device_lane_in_order(monkeypatch):
    rig = MoveRig(monkeypatch)
    decoder = rig.decoder
    rig.live(0, 9)
    decoder.page_rows[9] = [4, 5]
    decoder.contexts[9] = "ctx9"
    decoder.ls.position.positions[9], decoder.ls.position.shifts[9] = 77, 3
    rig.calls.clear()
    decoder.move_lane(9, 1)
    names = [c[0] for c in rig.calls]
    assert names == ["scope+", "pack", "fence", "unpack", "fence", "scope-", "admit_position", "park", "sync"]
    assert rig.calls[1] == ("pack", 9) and rig.calls[3] == ("unpack", 1)  # pack(src) before unpack(dst)
    assert rig.calls[6] == ("admit_position", "parent", 1, 77, 3) and rig.calls[7] == ("park", [9])
    assert decoder.positions[1] == 19 and decoder.positions[9] is None and decoder.positions[0] == 10
    assert decoder.page_rows[1] == [4, 5] and decoder.page_rows[9] == [] and decoder.page_table_dirty
    assert decoder.contexts[1] == "ctx9" and decoder.contexts[9] is None
    assert decoder.lane_moves == 1 and decoder.compaction_moves() == []
    assert decoder.last_move["src"] == 9 and decoder.last_move["dst"] == 1 and decoder.last_move["ms"] >= 0.0


@pytest.mark.parametrize(
    "live, src, dst, message",
    [
        ((0, 9), 9, 0, "not free"),  # live destination
        ((0,), 9, 1, "holds no request"),  # free source
        ((0, 9), 9, 9, "two lanes"),
        ((0, 9), 9, 40, "in \\[0, 32\\)"),
    ],
)
def test_move_lane_refusals(monkeypatch, expect_error, live, src, dst, message):
    rig = MoveRig(monkeypatch)
    rig.live(*live)
    rig.calls.clear()
    with expect_error(ValueError, message):
        rig.decoder.move_lane(src, dst)
    assert rig.calls == [] and rig.decoder.lane_moves == 0


def test_move_lane_refuses_a_destination_that_holds_a_page_row(monkeypatch, expect_error):
    rig = MoveRig(monkeypatch)
    rig.live(9)
    rig.decoder.page_rows[1] = [3]
    with expect_error(ValueError, "not free"):
        rig.decoder.move_lane(9, 1)


def test_move_lane_needs_a_paged_decoder(monkeypatch, expect_error):
    rig = Rig(monkeypatch, lanes=8, buckets=None, paging=None)
    rig.live(3)
    with expect_error(RuntimeError, "paged"):
        rig.decoder.move_lane(3, 0)
