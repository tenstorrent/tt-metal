# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Independent logical-value and tensor-placement oracle for Llama KV table tests."""

from __future__ import annotations

import torch

NUM_SLOTS = 2
NUM_LAYERS = 32
NUM_HEADS = 8
MAX_SEQ_LEN = 2048
GLOBAL_CHUNK = 1024
LOCAL_CHUNK = 256
HEAD_DIM = 128
PAGE_TOKENS = 32


def _check_range(name, value, upper):
    if type(value) is not int or not 0 <= value < upper:
        raise ValueError(f"{name} must be an int in [0, {upper}), got {value!r}")


def logical_tag(slot, layer, head, position):
    """Pack every logical non-K/V field into a unique 20-bit integer."""
    _check_range("slot", slot, NUM_SLOTS)
    _check_range("layer", layer, NUM_LAYERS)
    _check_range("head", head, NUM_HEADS)
    _check_range("position", position, MAX_SEQ_LEN)
    return (((slot * NUM_LAYERS + layer) * NUM_HEADS + head) * MAX_SEQ_LEN) + position


def device_major_positions(start):
    """Return one 1K logical chunk grouped by the SP row that receives it."""
    if type(start) is not int or start not in range(0, MAX_SEQ_LEN, GLOBAL_CHUNK):
        raise ValueError(f"start must be a 1K chunk boundary below {MAX_SEQ_LEN}, got {start!r}")
    positions = range(start, start + GLOBAL_CHUNK)
    grouped = [
        position for owner in range(4) for position in positions if (position % GLOBAL_CHUNK) // LOCAL_CHUNK == owner
    ]
    assert len(grouped) == GLOBAL_CHUNK
    return grouped


def tagged_values(kind, slot, layer, positions):
    """Encode head and logical position into exact BF16/BFP8 values.

    K is positive and V negative.  Every 32-value BFP8 block contains only
    powers of two with exponents one apart, so BFP8 packing preserves all bits.
    """
    if kind not in ("k", "v"):
        raise ValueError(f"kind must be 'k' or 'v', got {kind!r}")
    _check_range("slot", slot, NUM_SLOTS)
    _check_range("layer", layer, NUM_LAYERS)
    positions = list(positions)
    if not positions:
        raise ValueError("positions cannot be empty")
    for position in positions:
        _check_range("position", position, MAX_SEQ_LEN)

    position_tensor = torch.tensor(positions, dtype=torch.int64)
    heads = torch.arange(NUM_HEADS, dtype=torch.int64)
    dimensions = torch.arange(HEAD_DIM, dtype=torch.int64)
    tags = (((slot * NUM_LAYERS + layer) * NUM_HEADS + heads[:, None]) * MAX_SEQ_LEN) + position_tensor[None, :]
    bits = (tags[:, :, None] >> (dimensions[None, None, :] % 20)) & 1
    values = (32 + bits * 32).float()
    return values if kind == "k" else -values


def tagged_page(kind, slot, layer, head, position):
    """Return the logical 32-token page for one K/V head."""
    _check_range("head", head, NUM_HEADS)
    if type(position) is not int or position % PAGE_TOKENS or not 0 <= position <= MAX_SEQ_LEN - PAGE_TOKENS:
        raise ValueError(f"position must be a 32-token page start below {MAX_SEQ_LEN}, got {position!r}")
    values = tagged_values(kind, slot, layer, range(position, position + PAGE_TOKENS))
    return values[head : head + 1].unsqueeze(0)


def independent_tensor_location(position):
    """Map a logical page to the live tensor without consulting the address table."""
    if type(position) is not int or position % PAGE_TOKENS or not 0 <= position < MAX_SEQ_LEN:
        raise ValueError(f"position must be a 32-token page start below {MAX_SEQ_LEN}, got {position!r}")
    offset_in_chunk = position % GLOBAL_CHUNK
    sp_row = offset_in_chunk // LOCAL_CHUNK
    local_position = (position // GLOBAL_CHUNK) * LOCAL_CHUNK + (offset_in_chunk % LOCAL_CHUNK)
    return sp_row, local_position


def assert_independent_page(actual, expected, description):
    """Require exact shape, dtype-independent value equality for one page."""
    if tuple(actual.shape) != (1, 1, PAGE_TOKENS, HEAD_DIM):
        raise AssertionError(
            f"independent tensor oracle mismatch for {description}: actual shape {tuple(actual.shape)}"
        )
    if tuple(expected.shape) != (1, 1, PAGE_TOKENS, HEAD_DIM):
        raise AssertionError(
            f"independent tensor oracle mismatch for {description}: expected shape {tuple(expected.shape)}"
        )
    if not torch.equal(actual.float(), expected.float()):
        mismatch_count = torch.count_nonzero(actual.float() != expected.float()).item()
        raise AssertionError(f"independent tensor oracle mismatch for {description}: {mismatch_count} values differ")
