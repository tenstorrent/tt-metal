# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host planning for independent requests sharing prefill matrix operations."""

PAGE_SIZE = 128
MAX_PACKED_ROWS = 2048
MAX_BATCHED_SPAN = 1024


def padded_span(length):
    return (length + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE


def config_family(length):
    """Keep original norm ownership and matmul accumulation configuration."""
    rounded = padded_span(length)
    return rounded if rounded in (128, 256, 1024) else 512


def plan_prefill_batches(lengths, maximum, *, long_spans=False, packed_rows=MAX_PACKED_ROWS):
    if not 1 <= maximum <= 32:
        raise ValueError(f"Prefill request batch must be 1..32, got {maximum}")
    if not 0 < packed_rows <= 8192 or packed_rows % PAGE_SIZE:
        raise ValueError("Packed prefill capacity must be page aligned and at most 8192 rows")
    if any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("Prefill spans must be positive integer token counts")
    pending = sorted(range(len(lengths)), key=lambda row: lengths[row])
    groups = []
    while pending:
        group = [pending.pop(0)]
        while pending and len(group) < maximum:
            row = pending[0]
            padded = padded_span(lengths[row])
            long_group = long_spans and padded > MAX_BATCHED_SPAN and padded == padded_span(lengths[group[0]])
            if (
                config_family(lengths[row]) != config_family(lengths[group[0]])
                # Bound extra matrix rows relative to the native per-request
                # page rounding. Equal short partial spans must not be penalized
                # for padding that the serial path already requires.
                or 4 * padded * (len(group) + 1) > 5 * (sum(padded_span(lengths[index]) for index in group) + padded)
                or padded * (len(group) + 1) > (8192 if long_group else packed_rows)
                # Subdividing a native sliding SDPA chunk changes masked K
                # iterations and their finite-precision softmax corrections.
                or min(padded, MAX_BATCHED_SPAN) * (len(group) + 1) > packed_rows
                or (padded > MAX_BATCHED_SPAN and not long_group)
            ):
                break
            group.append(pending.pop(0))
        groups.append(group)
    return groups


class DeferredPrefillHistory(list):
    """Restore a cache-hit window only when its layer is about to consume it."""

    def __init__(self, layers, restore):
        super().__init__([None] * layers)
        self._pending = set(range(layers))
        self._restore = restore

    def __getitem__(self, index):
        if isinstance(index, int) and index in self._pending:
            value = self._restore(index)
            self[index] = value
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        self._pending.discard(index)
        if not self._pending:
            self._restore = None


from dataclasses import dataclass


@dataclass(frozen=True)
class RaggedPrefill:
    """Host-only request coordinates; packed offsets are never absolute starts."""

    lengths: tuple
    starts: tuple
    slots: tuple
    rows: tuple
    spans: tuple
    offsets: tuple
    family: int
    total: int

    @classmethod
    def build(cls, lengths, starts, slots, rows, *, max_context=262144):
        lengths, starts, slots, rows = map(tuple, (lengths, starts, slots, rows))
        count = len(lengths)
        if not 2 <= count <= 32 or any(len(values) != count for values in (starts, slots, rows)):
            raise ValueError("Ragged prefill needs matching metadata for2..32 requests")
        if any(type(value) is not int for values in (lengths, starts, slots, rows) for value in values):
            raise ValueError("Ragged coordinates must be integers")
        if (len(set(slots)) != count or len(set(rows)) != count or
                any(not 0 <= slot < 32 for slot in slots) or any(row < 0 for row in rows)):
            raise ValueError("Expected distinct valid slots and original result rows")
        if any(length <= 0 or start < 0 or start % 128 or start + length > max_context
               for length, start in zip(lengths, starts)):
            raise ValueError("Invalid logical span or page-aligned native start")
        spans = tuple(padded_span(length) for length in lengths)
        families = {config_family(length) for length in lengths}
        total = sum(spans)
        if len(families) != 1 or max(spans) > 1024 or total > 2048:
            raise ValueError("Ragged sharing requires one native family and at most2048 own padded rows")
        offsets = []
        position = 0
        for span in spans:
            offsets.append(position)
            position += span
        return cls(lengths, starts, slots, rows, spans, tuple(offsets), families.pop(), total)

    def reduction_buckets(self):
        """Request-local256 blocks plus optional128 tails, in stable request order."""
        blocks, tails = [], []
        for row, (offset, span) in enumerate(zip(self.offsets, self.spans)):
            for local in range(0, span // 256 * 256, 256):
                blocks.append((row, local, offset + local, 256))
            if span % 256:
                local = span // 256 * 256
                tails.append((row, local, offset + local, 128))
        return tuple(blocks), tuple(tails)


def plan_ragged_prefill_batches(lengths, maximum, *, long_spans=False, packed_rows=MAX_PACKED_ROWS):
    # Reuse original validation. The original rectangle function stays unchanged.
    plan_prefill_batches(lengths, maximum, long_spans=long_spans, packed_rows=packed_rows)
    pending = sorted(range(len(lengths)), key=lambda row: lengths[row])
    groups = []
    while pending:
        if padded_span(lengths[pending[0]]) > 1024:
            # Preserve original long-span grouping and chunk policy exactly.
            tail = plan_prefill_batches([lengths[row] for row in pending], maximum,
                                       long_spans=long_spans, packed_rows=packed_rows)
            groups.extend([[pending[index] for index in group] for group in tail])
            break
        # Equal-span special families and existing wide uniform groups keep
        # their original rectangular dispatch, capacity and streaming policy.
        original = plan_prefill_batches([lengths[row] for row in pending], maximum,
                                        long_spans=long_spans, packed_rows=packed_rows)[0]
        original_rows = [pending[index] for index in original]
        original_spans = [padded_span(lengths[row]) for row in original_rows]
        if (config_family(lengths[pending[0]]) != 512 or
                (len(set(original_spans)) == 1 and sum(original_spans) > 2048)):
            groups.append(original_rows)
            pending = pending[len(original_rows):]
            continue
        group = [pending.pop(0)]
        total = padded_span(lengths[group[0]])
        while pending and len(group) < maximum:
            row = pending[0]
            span = padded_span(lengths[row])
            if (span > 1024 or config_family(lengths[row]) != config_family(lengths[group[0]]) or
                    total + span > min(2048, packed_rows)):
                break
            group.append(pending.pop(0))
            total += span
        groups.append(group)
    return groups
