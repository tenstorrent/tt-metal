# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for the multi-lane prefill step message and the batched producer schedule (no device)."""

import struct
from collections import Counter

import pytest
import torch

from models.demos.common.prefill import step_message
from models.demos.common.prefill.runners import prefill_producer as producer
from models.demos.common.prefill.step_message import Lane

CHUNK = 2048
SP = 8


def test_metadata_sizes(expect_error):
    assert step_message.metadata_size_bytes(1) == 12
    assert step_message.metadata_size_bytes(4) == 52
    for lanes in (1, 2, 3, 4, 8):
        assert step_message.max_lanes_for_metadata(step_message.metadata_size_bytes(lanes)) == lanes
    for size in (0, 8, 16, 24, 30):
        with expect_error(ValueError, "metadata"):
            step_message.max_lanes_for_metadata(size)


def test_legacy_message_is_unchanged():
    lane = Lane(3, 8192, 10000)
    words = step_message.pack_words([lane], 1)
    assert struct.pack(f"<{len(words)}I", *words) == producer._pack_metadata(*lane)
    assert step_message.unpack_words(words, 1) == [lane]


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_round_trip(count):
    lanes = [Lane(slot, 2048 * slot, 2048 * slot + 100 + slot) for slot in range(count)]
    words = step_message.pack_words(lanes, 4)
    assert len(words) == 13
    assert words[:4] == [*lanes[0], count]
    assert words[4 + 3 * (count - 1) :] == [0] * (9 - 3 * (count - 1))
    assert step_message.unpack_words(words, 4) == lanes


def test_unpack_rejects_bad_counts(expect_error):
    for count in (0, 5):
        with expect_error(ValueError, "lanes|metadata words"):
            step_message.unpack_words([0, 0, 1, count] + [0] * 9, 4)
    with expect_error(ValueError, "lanes|metadata words"):
        step_message.unpack_words([0] * 12, 4)


def test_shutdown_sentinel():
    for lanes in (1, 4):
        words = step_message.shutdown_words(lanes)
        assert len(words) == step_message.metadata_words(lanes) and step_message.is_shutdown(words)
    assert not step_message.is_shutdown([0, 0, 2048, 1])


def test_stacked_rows_layout(expect_error):
    lanes = [torch.arange(CHUNK) + 10_000 * lane for lane in range(3)]
    rows = step_message.stacked_rows(lanes, CHUNK, SP, 4)
    slab = CHUNK // SP
    assert rows.shape == (SP, 1, 4 * slab)
    for rank in range(SP):
        # Each rank's row is every lane's rank-th slab in lane order, then zeros.
        expected = torch.cat([lane[rank * slab : (rank + 1) * slab] for lane in lanes] + [torch.zeros(slab)])
        assert torch.equal(rows[rank, 0], expected.to(torch.int64))
    with expect_error(ValueError, "padded chunk"):
        step_message.stacked_rows([torch.arange(CHUNK - 1)], CHUNK, SP, 4)


def _validate(lanes):
    step_message.validate_lanes(lanes, chunk_size=CHUNK, max_lanes=4, max_seq_len=262144, num_slots=4)


def test_validate_accepts_legal_steps():
    _validate([Lane(0, 0, 2048)])
    _validate([Lane(2, 4096, 6144), Lane(2, 6144, 8192), Lane(2, 8192, 9000), Lane(0, 0, 1)])
    _validate([Lane(1, 0, 2048), Lane(1, 2048, 4096), Lane(3, 2048, 4096), Lane(3, 4096, 4097)])


@pytest.mark.parametrize(
    "lanes",
    [
        [],
        [Lane(0, 0, 2048)] * 5,
        [Lane(4, 0, 2048)],
        [Lane(0, 1024, 2048)],
        [Lane(0, 0, 2049)],
        [Lane(0, 2048, 2048)],
        [Lane(0, 0, 1000), Lane(0, 2048, 4096)],  # partial lane before the slot's last
        [Lane(0, 0, 2048), Lane(0, 4096, 6144)],  # gap
        [Lane(0, 2048, 4096), Lane(0, 0, 2048)],  # out of order
        [Lane(0, 0, 2048), Lane(1, 0, 2048), Lane(0, 2048, 4096)],  # not adjacent
    ],
)
def test_validate_rejects_illegal_steps(lanes, expect_error):
    with expect_error(ValueError, "lane|step"):
        _validate(lanes)


def test_plan_step():
    left = {"a": 1, "b": 5, "c": 2}.get
    assert step_message.plan_step(["a"], 4, left) == [("a", 1)]
    assert step_message.plan_step(["b"], 4, left) == [("b", 4)]
    assert step_message.plan_step(["b", "a", "c"], 4, left) == [("b", 2), ("a", 1), ("c", 1)]
    assert step_message.plan_step(["a", "c", "b"], 4, left) == [("a", 1), ("c", 2), ("b", 1)]
    assert step_message.plan_step(list("abcde"), 4, lambda s: 3) == [(s, 1) for s in "abcd"]


def _config(num_users, lengths, max_requests=None, **overrides):
    cfg = producer.ProducerConfig(
        num_users=num_users,
        chunks_min=1,
        chunks_max=1,
        max_requests=max_requests or num_users,
        duration_s=float("inf"),
        p_gap=0.0,
        p_burst=0.0,
        gap_ms=(0.0, 0.0),
        mid_chunk_end_prob=0.0,
        seed=1234,
        verify=False,
        pcc_threshold=0.9,
        slot_lengths=dict(enumerate(lengths)),
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def _run(cfg, lanes=4):
    steps = []

    def push(step):
        steps.append(step)
        return 0.0

    stats = producer.run_batched_schedule(cfg, lanes, push_fn=push)
    return stats, steps


@pytest.fixture
def chunk_2048(monkeypatch):
    monkeypatch.setattr(producer, "CHUNK_SIZE", CHUNK)
    monkeypatch.setattr(producer, "MAX_SEQ_LEN", 262144)


def test_batched_schedule_mixed_lengths(chunk_2048):
    lengths = [16384, 6144, 2048, 7000]
    stats, steps = _run(_config(4, lengths))
    sent = Counter()
    ends = {}
    for step in steps:
        lanes = [Lane(slot, start, end) for slot, _, start, end, _ in step]
        _validate(lanes)
        active = {slot for slot, length in enumerate(lengths) if ends.get(slot, 0) < length}
        # Every slot with chunks left gets at least one lane in every step.
        assert {lane.slot for lane in lanes} == active
        for slot, chunk_idx, start, end, isl in step:
            assert start == chunk_idx * CHUNK and isl == lengths[slot]
            assert start == ends.get(slot, 0)
            ends[slot] = end
            sent[slot] += 1
    assert ends == dict(enumerate(lengths))
    assert stats.total_pushes == sum(sent.values()) == sum(-(-n // CHUNK) for n in lengths)
    assert len(stats.push_ms) == len(steps) and stats.completed == 4
    assert {slot: fill.real_len for slot, fill in stats.resident.items()} == dict(enumerate(lengths))
    assert all(len(step) == 4 for step in steps[:-1])


def test_batched_schedule_refills_slots(chunk_2048):
    stats, steps = _run(_config(2, [4096, 2048], max_requests=5))
    assert stats.completed == 5
    # Step 1 finishes both requests (slot 0 takes the spare lanes); both slots restart from 0 in step 2, the slot
    # order rotated by one.
    assert [(slot, start) for slot, _, start, _, _ in steps[0]] == [(0, 0), (0, 2048), (1, 0)]
    assert [(slot, start) for slot, _, start, _, _ in steps[1]] == [(1, 0), (0, 0), (0, 2048)]


def test_batched_schedule_respects_max_requests(chunk_2048):
    # Four one-chunk prompts would all finish in the first 4-lane step; only one request may run.
    stats, steps = _run(_config(4, [2048] * 4, max_requests=1))
    assert stats.completed == 1 and stats.total_pushes == 1
    assert [[slot for slot, *_ in step] for step in steps] == [[0]]


def test_batched_schedule_lone_request_takes_every_lane(chunk_2048):
    _, steps = _run(_config(1, [8192 + 100]))
    assert [[(start, end) for _, _, start, end, _ in step] for step in steps] == [
        [(0, 2048), (2048, 4096), (4096, 6144), (6144, 8192)],
        [(8192, 8292)],
    ]


def test_batched_schedule_rejects_multi_turn(chunk_2048, expect_error):
    with expect_error(ValueError, "multi-turn"):
        _run(_config(1, [2048], multi_turn_prob=0.5))


def test_slot_lengths_from_env(monkeypatch, expect_error):
    monkeypatch.delenv("PREFILL_PRODUCER_SLOT_LENGTHS", raising=False)
    assert producer._slot_lengths_from_env(4) is None
    monkeypatch.setenv("PREFILL_PRODUCER_SLOT_LENGTHS", "32768,2048")
    assert producer._slot_lengths_from_env(4) == {0: 32768, 1: 2048}  # slots 2 and 3 keep random lengths
    for bad in ("1,2,3,4,5", "2048,abc", "2048,0", "2048,"):
        monkeypatch.setenv("PREFILL_PRODUCER_SLOT_LENGTHS", bad)
        with expect_error(ValueError, "PREFILL_PRODUCER_SLOT_LENGTHS"):
            producer._slot_lengths_from_env(4)


def test_step_payload(chunk_2048, monkeypatch):
    monkeypatch.setattr(producer, "GLOBAL_MESH_SHAPE", (SP, 4))
    tokens = [list(range(CHUNK)), [7] * CHUNK]
    rows, metadata = producer.step_payload(tokens, [(1, 0, 2048), (2, 4096, 5000)], 4)
    assert rows.dtype.name == "uint32" and rows.nbytes == 4 * CHUNK * 4  # 32 KB, as one 8k chunk
    assert struct.unpack("<13I", metadata) == (1, 0, 2048, 2, 2, 4096, 5000, 0, 0, 0, 0, 0, 0)
    assert rows[0, 0, : CHUNK // SP].tolist() == list(range(CHUNK // SP))
    assert rows[0, 0, CHUNK // SP : 2 * CHUNK // SP].tolist() == [7] * (CHUNK // SP)


class _FakeService:
    def __init__(self, max_lanes):
        self.metadata_bytes = step_message.metadata_size_bytes(max_lanes)
        self.payload_bytes = 4 * CHUNK * max_lanes
        self.sent = []

    def metadata_size_bytes(self):
        return self.metadata_bytes

    def forward_to_tensor_bytes(self, payload, metadata):
        self.sent.append((payload, metadata))

    def barrier(self):
        pass


@pytest.mark.parametrize("max_lanes", [1, 4])
def test_service_format_drives_schedule_and_sentinel(chunk_2048, monkeypatch, max_lanes):
    monkeypatch.setattr(producer, "GLOBAL_MESH_SHAPE", (SP, 4))
    service = _FakeService(max_lanes)
    pool = list(range(4 * CHUNK))
    serial = []
    stats = producer.run_service_schedule(
        _config(2, [4096, 2048]),
        service,
        service.payload_bytes,
        {"trace": pool},
        {0: "trace", 1: "trace"},
        lambda *chunk: serial.append(chunk) or 0.0,
        "[test]",
    )
    assert stats.total_pushes == 3
    if max_lanes == 1:
        assert len(serial) == 3 and not service.sent
    else:
        # One step: slot 0's two chunks and slot 1's one, as one 52-byte message.
        assert not serial and len(service.sent) == 1
        assert struct.unpack("<13I", service.sent[0][1])[:7] == (0, 0, 2048, 3, 0, 2048, 4096)
    producer.push_shutdown(service, service.payload_bytes, "[test]")
    payload, metadata = service.sent[-1]
    assert payload.nbytes == service.payload_bytes and len(metadata) == service.metadata_bytes
    assert step_message.is_shutdown(struct.unpack(f"<{len(metadata) // 4}i", metadata))
