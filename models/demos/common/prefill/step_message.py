# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The multi-lane prefill step message: up to L chunks (lanes) of up to L requests in one H2D transfer.

Opt-in: PREFILL_BATCH_MAX_LANES > 1 on the runner, and an adapter whose runtime implements prefill_step. L = 1 is
the one-chunk format: a 12-byte (slot, start, end) message and a [SP, 1, chunk / SP] payload.

Metadata, 4 * (3L + 1) bytes of uint32 words (52 bytes at L = 4):
    slot0, start0, end0, num_lanes, slot1, start1, end1, ..., slot(L-1), start(L-1), end(L-1)
Unused lanes are zero. Lane 0 keeps the one-chunk word positions, so the shutdown sentinel (every word -1) and
lane-0 logging read the same in both formats.

Payload: [SP, 1, L * chunk / SP] uint32, the same size as a one-chunk message of L * chunk tokens (32 KB for
4 x 2k, as for one 8k chunk). Row r (SP rank r) holds every lane's r-th slab of chunk / SP tokens, in lane order,
then zeros for unused lanes: the stacked rows a batched step embeds. Starts are chunk-aligned, so no rotation
(chunk_layout) applies.

Rules (validate_lanes): 1 to L lanes; each lane carries 1 to chunk real tokens from a chunk-aligned start; a slot's
lanes are adjacent and consecutive (each starts where the previous ends) and only its last may be partial.

Layer acks: one per KV layer per lane, lanes in message order, so a step of n lanes acks like n one-chunk messages
sent in that order, and no ack precedes its KV. The target is lane 0 layer by layer and lanes 1..n-1 after the
step's last layer. Until #58136 lands, a runtime may instead send every lane's acks from the host once the step
finishes (Gemma4 does). Either way acks arrive before the next step starts.

The runtime hook is ``prefill_step`` (see PrefillModelAdapter.supports_batched_steps).
"""

from typing import NamedTuple, Sequence

import torch

LEGACY_METADATA_WORDS = 3
SHUTDOWN_WORD = -1


class Lane(NamedTuple):
    slot: int
    start: int
    end: int


def metadata_words(max_lanes: int) -> int:
    if max_lanes < 1:
        raise ValueError(f"max_lanes must be >= 1, got {max_lanes}")
    return LEGACY_METADATA_WORDS if max_lanes == 1 else 3 * max_lanes + 1


def metadata_size_bytes(max_lanes: int) -> int:
    return 4 * metadata_words(max_lanes)


def max_lanes_for_metadata(size_bytes: int) -> int:
    """The L a service's metadata size encodes (12 bytes is the one-chunk message)."""
    words, rem = divmod(size_bytes, 4)
    if rem == 0 and words == LEGACY_METADATA_WORDS:
        return 1
    if rem or words < 7 or (words - 1) % 3:
        raise ValueError(f"{size_bytes}-byte metadata is neither the 12-byte chunk nor a step message")
    return (words - 1) // 3


def pack_words(lanes: Sequence[Lane], max_lanes: int) -> list[int]:
    if not 1 <= len(lanes) <= max_lanes:
        raise ValueError(f"a step carries 1 to {max_lanes} lanes, got {len(lanes)}")
    if max_lanes == 1:
        return list(lanes[0])
    words = [*lanes[0], len(lanes)]
    for lane in lanes[1:]:
        words += list(lane)
    return words + [0] * (metadata_words(max_lanes) - len(words))


def unpack_words(words: Sequence[int], max_lanes: int) -> list[Lane]:
    if len(words) != metadata_words(max_lanes):
        raise ValueError(f"expected {metadata_words(max_lanes)} metadata words, got {len(words)}")
    words = [int(w) for w in words]
    if max_lanes == 1:
        return [Lane(*words)]
    count = words[3]
    if not 1 <= count <= max_lanes:
        raise ValueError(f"step message names {count} lanes, outside [1, {max_lanes}]")
    rest = words[4:]
    return [Lane(*words[:3])] + [Lane(*rest[3 * i : 3 * i + 3]) for i in range(count - 1)]


def is_shutdown(words: Sequence[int]) -> bool:
    return all(int(w) == SHUTDOWN_WORD for w in words[:LEGACY_METADATA_WORDS])


def shutdown_words(max_lanes: int) -> list[int]:
    return [SHUTDOWN_WORD] * metadata_words(max_lanes)


def stacked_rows(lane_tokens: Sequence[Sequence[int]], chunk_size: int, sp: int, max_lanes: int) -> torch.Tensor:
    """[sp, 1, max_lanes * chunk_size / sp] int64: row r is each lane's r-th chunk_size / sp slab, in lane order."""
    if not 1 <= len(lane_tokens) <= max_lanes:
        raise ValueError(f"a step carries 1 to {max_lanes} lanes, got {len(lane_tokens)}")
    slab = chunk_size // sp
    rows = torch.zeros(sp, max_lanes, slab, dtype=torch.int64)
    for lane, tokens in enumerate(lane_tokens):
        if len(tokens) != chunk_size:
            raise ValueError(f"lane {lane} holds {len(tokens)} tokens, expected a padded chunk of {chunk_size}")
        rows[:, lane] = torch.as_tensor(tokens, dtype=torch.int64).view(sp, slab)
    return rows.reshape(sp, 1, max_lanes * slab)


def plan_step(active: Sequence, max_lanes: int, chunks_left) -> list:
    """[(slot, lanes)] for one batched step: one lane per active slot in the given order (at most max_lanes), then
    the spare lanes to those slots in the same order, each up to its chunks left (chunks_left(slot))."""
    plan = [[slot, 1] for slot in active[:max_lanes]]
    free = max_lanes - len(plan)
    for entry in plan:
        extra = min(free, chunks_left(entry[0]) - 1)
        entry[1] += extra
        free -= extra
    return [tuple(entry) for entry in plan]


def validate_lanes(lanes: Sequence[Lane], *, chunk_size: int, max_lanes: int, max_seq_len: int, num_slots: int):
    """Raise ValueError unless the lanes form a legal step (rules in the module docstring)."""
    if not 1 <= len(lanes) <= max_lanes:
        raise ValueError(f"a step carries 1 to {max_lanes} lanes, got {len(lanes)}")
    seen = set()
    for i, (slot, start, end) in enumerate(lanes):
        if not 0 <= slot < num_slots:
            raise ValueError(f"lane {i}: KV slot {slot} is outside [0, {num_slots})")
        if start < 0 or start % chunk_size:
            raise ValueError(f"lane {i}: start {start} is not a nonnegative multiple of {chunk_size}")
        if not start < end <= min(start + chunk_size, max_seq_len):
            raise ValueError(f"lane {i}: [{start}, {end}) must hold 1 to {chunk_size} tokens within {max_seq_len}")
        previous = lanes[i - 1] if i else None
        if previous is not None and previous.slot == slot:
            if previous.end != previous.start + chunk_size or start != previous.end:
                raise ValueError(f"lane {i}: slot {slot}'s lanes must be consecutive full chunks, partial last")
        elif slot in seen:
            raise ValueError(f"lane {i}: slot {slot}'s lanes must be adjacent")
        seen.add(slot)
