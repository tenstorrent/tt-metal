# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Position-derived CP chunk layout shared by producers and model runtimes.

Keep this implementation independent of model-specific utilities: adding a
prefill adapter must not require changing another model's code or imports.
"""


def rotated_chip_positions(kv_actual_isl: int, sp: int, chunk_local: int) -> list[list[int]]:
    """Per-chip global position for each chip-local row after the server's KV-pad-aware rotation,
    mirroring the writer kernel EXACTLY. positions[c][r] is the global token position carried by
    chip c's r-th rotated row.

    Slab-aware: each chip writes chunk_local rows starting at update_idxt, and the cache cell at
    local row lr on chip c holds global position (lr // chunk_local)*chunk_size_global +
    c*chunk_local + (lr % chunk_local). The union over all (c, r) tiles
    [kv_actual_isl, kv_actual_isl + chunk_size_global) exactly. Positions below the
    exclusive real-token end are valid; remaining rows form a padded suffix on each chip.
    """
    chunk_size_global = sp * chunk_local
    boundary_slab = kv_actual_isl // chunk_size_global
    boundary_chip = (kv_actual_isl // chunk_local) % sp
    boundary_offset = kv_actual_isl % chunk_local

    positions = [[0] * chunk_local for _ in range(sp)]
    for c in range(sp):
        if c < boundary_chip:
            update_idxt = (boundary_slab + 1) * chunk_local
        elif c == boundary_chip:
            update_idxt = boundary_slab * chunk_local + boundary_offset
        else:
            update_idxt = boundary_slab * chunk_local
        for r in range(chunk_local):
            lr = update_idxt + r
            positions[c][r] = (lr // chunk_local) * chunk_size_global + c * chunk_local + (lr % chunk_local)
    return positions


def validate_chunk_range(start, end, chunk_size, sp, max_seq_len=None):
    if chunk_size <= 0 or sp <= 0 or chunk_size % (sp * 32):
        raise ValueError("chunk size must contain whole 32-token tiles on every SP rank")
    if start < 0 or start % 32:
        raise ValueError("actual_start must be nonnegative and 32-token aligned; replay the preceding partial tile")
    if not start < end <= start + chunk_size:
        raise ValueError(f"invalid chunk range [{start}, {end}) for chunk size {chunk_size}")
    if max_seq_len is not None and end > max_seq_len:
        raise ValueError("real tokens exceed the configured cache")


def chunk_positions(start, chunk_size, sp):
    """Absolute positions in chip-major input order, including padded positions."""
    validate_chunk_range(start, start + chunk_size, chunk_size, sp)
    return [p for row in rotated_chip_positions(start, sp, chunk_size // sp) for p in row]


def pack_chunk_tokens(tokens, start, end, chunk_size, sp, pad_token=0):
    """Pack sequential tokens starting at `start` into a full CP-major payload."""
    validate_chunk_range(start, end, chunk_size, sp)
    if len(tokens) < end - start:
        raise ValueError("token source is shorter than the real chunk range")
    return [int(tokens[p - start]) if p < end else pad_token for p in chunk_positions(start, chunk_size, sp)]


def chunk_row_for_position(position, start, chunk_size, sp):
    """Flattened chip-major output row for an absolute token position."""
    if not start <= position < start + chunk_size:
        raise ValueError("position is outside the chunk")
    return chunk_positions(start, chunk_size, sp).index(position)
