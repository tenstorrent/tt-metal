# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import struct

import pytest

from models.demos.gemma4_d_p.tt.runners.packing import (
    PackedPrefillSender,
    PrefillSegment,
    decode_metadata,
    encode_metadata,
    metadata_size_bytes,
    pack_positions,
    pack_tokens,
    validate_segments,
)


def unstripe(rows):
    return [x for group in range(len(rows[0]) // 32) for row in rows for x in row[group * 32 : (group + 1) * 32]]


@pytest.mark.parametrize("length", range(256, 8193, 256))
def test_every_striped_length(length):
    segment = PrefillSegment(0, 256, 256 + length)
    rows = pack_tokens([list(range(length))], [segment])
    assert rows[0][:32] == list(range(32))
    if length >= 512:
        assert rows[0][32:64] == list(range(256, 288))
    assert unstripe(rows) == list(range(length)) + [0] * (8192 - length)
    assert unstripe(pack_positions([segment])) == list(range(256, 256 + length)) + [0] * (8192 - length)


def test_packing_partial_requests_and_metadata():
    segments = (PrefillSegment(4, 512, 769), PrefillSegment(1, 0, 1600), PrefillSegment(3, 256, 768))
    tokens = [list(range(s.actual_start + 1, s.actual_end + 1)) for s in segments]
    rows = pack_tokens(tokens, segments)
    offset = 0
    for s, values in zip(segments, tokens):
        packed = unstripe(rows)
        assert packed[offset : offset + len(values)] == values
        assert packed[offset + len(values) : offset + s.padded_length()] == [0] * (s.padded_length() - len(values))
        offset += s.padded_length()
    encoded = encode_metadata(segments)
    assert len(encoded) == metadata_size_bytes()
    assert decode_metadata(struct.unpack(f"<{len(encoded) // 4}I", encoded)) == segments
    assert decode_metadata([0xFFFFFFFF] * 19) is None


@pytest.mark.parametrize("first_rank", range(8))
@pytest.mark.parametrize("length", [256, 257, 768])
def test_rotated_starts_keep_tokens_and_positions_on_their_cache_rank(first_rank, length):
    start = 2048 + first_rank * 32
    second_start = 4096 + (7 - first_rank) * 32
    segments = (PrefillSegment(0, start, start + length), PrefillSegment(1, second_start, second_start + 512))
    validate_segments(segments, chunk_size=2048, cp=8, num_users=2, max_seq_len=8192)
    tokens = [list(range(s.actual_start + 1, s.actual_end + 1)) for s in segments]
    rows = pack_tokens(tokens, segments, chunk_size=2048)
    positions = pack_positions(segments, chunk_size=2048)
    for rank in range(8):
        expected_positions, expected_tokens = [], []
        for segment in segments:
            owned = [p for p in range(segment.actual_start, segment.actual_end) if p // 32 % 8 == rank]
            padding = [0] * (segment.padded_length() // 8 - len(owned))
            expected_positions.extend(owned + padding)
            expected_tokens.extend([p + 1 for p in owned] + padding)
        padding = [0] * (2048 // 8 - len(expected_positions))
        assert positions[rank] == expected_positions + padding
        assert rows[rank] == expected_tokens + padding


@pytest.mark.parametrize(
    "segments",
    [
        (),
        (PrefillSegment(0, 0, 512), PrefillSegment(0, 0, 256)),
        (PrefillSegment(6, 0, 256),),
        (PrefillSegment(0, 1, 512),),
        (PrefillSegment(0, 256, 256),),
        (PrefillSegment(0, 0, 8193),),
        (PrefillSegment(0, 262144, 262400),),
    ],
)
def test_invalid_batches(segments, expect_error):
    with expect_error(ValueError, "request|Request|Padded"):
        validate_segments(segments, chunk_size=8192, cp=8, num_users=6, max_seq_len=262144)


def test_sender_flushes_budget_duplicate_slot_and_final_partial():
    sent = []
    sender = PackedPrefillSender(
        lambda rows, md: sent.append((unstripe(rows), decode_metadata(struct.unpack("<19I", md))))
    )
    for slot, length in ((0, 1600), (1, 512), (2, 768), (3, 4096), (4, 2048), (4, 256)):
        sender.append([slot + 10] * length, PrefillSegment(slot, 0, length))
    sender.flush()
    assert [[s.slot_id for s in segments] for _, segments in sent] == [[0, 1, 2, 3], [4], [4]]
    assert [sum(s.padded_length() for s in segments) for _, segments in sent] == [7168, 2048, 256]
    assert sent[0][0][1792:2304] == [11] * 512


def test_cache_table_matches_stripe_ownership():
    from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import iter_cache_chunk_locations

    locations = list(
        iter_cache_chunk_locations(
            seq_len=8192,
            chunk_size=256,
            cp=8,
            num_users=2,
            heads_per_device=4,
            local_head=2,
            num_banks=8,
            chunk_size_bytes=8704,
        )
    )
    assert len(locations) == 2 * 8192 // 32
    for rank, slot, position, bank, offset in locations:
        assert position // 32 % 8 == rank
        local_block = position // 256
        shard = (slot * 4 + 2) * (8192 // 256) + local_block
        assert (bank, offset) == (shard % 8, shard // 8 * 8704)


def test_producer_schedule_packs_requests_for_runner(monkeypatch):
    from models.demos.common.prefill.runners import prefill_producer as producer
    from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4PrefillAdapter

    adapter = Gemma4PrefillAdapter()
    lengths = [256, 512, 768, 1024, 1536, 2048]
    monkeypatch.setattr(producer, "CHUNK_SIZE", 8192)
    monkeypatch.setattr(producer, "MAX_SEQ_LEN", 262144)
    cfg = producer.ProducerConfig(
        num_users=6,
        chunks_min=1,
        chunks_max=1,
        max_requests=6,
        slot_lengths=dict(enumerate(lengths)),
        interleave="round_robin",
        duration_s=float("inf"),
        p_gap=0,
        p_burst=0,
        gap_ms=(0, 0),
        mid_chunk_end_prob=0,
        seed=1234,
        verify=False,
        pcc_threshold=0.98,
    )
    received = []

    def send(rows, metadata):
        words = struct.unpack(f"<{len(metadata) // 4}I", metadata)
        received.append((unstripe(rows), adapter.decode_prefill_metadata(words)["segments"]))

    sender = adapter.create_prefill_sender(send, chunk_size=8192, cp=8, num_users=6, max_seq_len=262144)
    stats = producer.run_schedule(
        cfg,
        push_fn=lambda slot, _, start, end, __: sender.append_request([slot + 1] * (end - start), slot, start, end)
        or 0,
        flush_fn=sender.flush,
    )
    assert stats.completed == stats.total_pushes == 6
    assert sender.num_batches == 1 and sender.num_tokens == sum(lengths)
    values, segments = received[0]
    assert [s.actual_end for s in segments] == lengths
    offset = 0
    for segment in segments:
        assert values[offset : offset + segment.actual_end] == [segment.slot_id + 1] * segment.actual_end
        offset += segment.padded_length()


def test_striped_geometry_allows_configurable_batch_budget():
    from models.demos.gemma4_d_p.tt.model import prefill_chunk_geometry_error

    assert prefill_chunk_geometry_error(6144, 8, 262144, tp_degree=4, stripe_size=32) is None
    assert prefill_chunk_geometry_error(768, 8, 262144, tp_degree=4, stripe_size=32) is not None
