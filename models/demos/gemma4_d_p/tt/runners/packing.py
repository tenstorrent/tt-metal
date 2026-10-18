# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed prefill wire format and 32-token CP ownership."""

import struct
from dataclasses import dataclass

STRIPE_SIZE = 32
SHUTDOWN = 0xFFFFFFFF


@dataclass(frozen=True)
class PrefillSegment:
    slot_id: int
    actual_start: int
    actual_end: int

    def padded_length(self, cp=8):
        quantum = cp * STRIPE_SIZE
        return ((self.actual_end - self.actual_start + quantum - 1) // quantum) * quantum


def validate_segments(segments, *, chunk_size, cp, num_users, max_seq_len, slot_ends=None):
    if not segments:
        raise ValueError("A packed prefill must contain at least one request")
    slots = set()
    for segment in segments:
        slot, start, end = segment.slot_id, segment.actual_start, segment.actual_end
        if not 0 <= slot < num_users or slot in slots:
            raise ValueError("Each request must use a distinct valid KV slot")
        slots.add(slot)
        if start < 0 or start % (cp * STRIPE_SIZE):
            raise ValueError(f"Request starts must align to {cp * STRIPE_SIZE} tokens")
        if not start < end <= max_seq_len:
            raise ValueError("Request must contain real tokens within the context limit")
        if slot_ends is not None and start and start != slot_ends[slot]:
            raise ValueError(f"Slot {slot} expects position {slot_ends[slot]}, got {start}")
    if sum(s.padded_length(cp) for s in segments) > chunk_size:
        raise ValueError("Padded requests exceed the packed chunk size")


def stripe_rows(values, cp=8):
    if len(values) % (cp * STRIPE_SIZE):
        raise ValueError("Packed values must contain whole CP stripe groups")
    return [
        [
            values[i]
            for group in range(0, len(values), cp * STRIPE_SIZE)
            for i in range(group + rank * STRIPE_SIZE, group + (rank + 1) * STRIPE_SIZE)
        ]
        for rank in range(cp)
    ]


def pack_tokens(token_lists, segments, *, chunk_size=8192, cp=8, pad_token_id=0):
    if len(token_lists) != len(segments):
        raise ValueError("Each request needs one token list")
    packed = []
    for tokens, segment in zip(token_lists, segments):
        length = segment.actual_end - segment.actual_start
        if len(tokens) != length or length <= 0:
            raise ValueError("Token count must match the request interval")
        packed.extend(tokens)
        packed.extend([pad_token_id] * (segment.padded_length(cp) - length))
    if len(packed) > chunk_size:
        raise ValueError("Padded requests exceed the packed chunk size")
    packed.extend([pad_token_id] * (chunk_size - len(packed)))
    return stripe_rows(packed, cp)


def pack_positions(segments, *, chunk_size=8192, cp=8):
    return pack_tokens(
        [list(range(s.actual_start, s.actual_end)) for s in segments],
        segments,
        chunk_size=chunk_size,
        cp=cp,
    )


def metadata_size_bytes(chunk_size=8192, cp=8, num_users=6):
    return 4 * (1 + 3 * min(num_users, chunk_size // (cp * STRIPE_SIZE)))


def encode_metadata(segments, *, chunk_size=8192, cp=8, num_users=6):
    capacity = metadata_size_bytes(chunk_size, cp, num_users) // 4
    words = [len(segments)]
    for s in segments:
        words.extend((s.slot_id, s.actual_start, s.actual_end))
    if not segments or len(words) > capacity:
        raise ValueError("Request count exceeds packed metadata capacity")
    return struct.pack(f"<{capacity}I", *(words + [0] * (capacity - len(words))))


def decode_metadata(words):
    count = int(words[0])
    if count == SHUTDOWN:
        return None
    if count < 1 or 1 + 3 * count > len(words):
        raise ValueError("Invalid packed prefill request count")
    return tuple(PrefillSegment(*(int(v) for v in words[i : i + 3])) for i in range(1, 1 + 3 * count, 3))


class PackedPrefillSender:
    """Coalesce ready requests; flush before reusing a slot or exceeding the row budget."""

    def __init__(self, send, *, chunk_size=8192, cp=8, num_users=6, max_seq_len=262144):
        self.send = send
        self.chunk_size, self.cp, self.num_users, self.max_seq_len = chunk_size, cp, num_users, max_seq_len
        self.segments, self.tokens = [], []
        self.num_batches = self.num_tokens = 0

    def append_request(self, tokens, slot_id, actual_start, actual_end):
        self.append(tokens, PrefillSegment(slot_id, actual_start, actual_end))

    def append(self, tokens, segment):
        validate_segments(
            (segment,), chunk_size=self.chunk_size, cp=self.cp, num_users=self.num_users, max_seq_len=self.max_seq_len
        )
        if len(tokens) != segment.actual_end - segment.actual_start:
            raise ValueError("Token count must match the request interval")
        if (
            any(s.slot_id == segment.slot_id for s in self.segments)
            or sum(s.padded_length(self.cp) for s in self.segments) + segment.padded_length(self.cp) > self.chunk_size
        ):
            self.flush()
        self.segments.append(segment)
        self.tokens.append(list(tokens))
        if (
            sum(s.padded_length(self.cp) for s in self.segments) == self.chunk_size
            or len(self.segments) == self.num_users
        ):
            self.flush()

    def flush(self):
        if self.segments:
            rows = pack_tokens(self.tokens, self.segments, chunk_size=self.chunk_size, cp=self.cp)
            metadata = encode_metadata(self.segments, chunk_size=self.chunk_size, cp=self.cp, num_users=self.num_users)
            self.send(rows, metadata)
            self.num_batches += 1
            self.num_tokens += sum(s.actual_end - s.actual_start for s in self.segments)
            self.segments, self.tokens = [], []
