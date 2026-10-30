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
