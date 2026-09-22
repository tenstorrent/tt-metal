# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Independent logical-value and placement helpers for Llama KV-table tests."""

from __future__ import annotations

import torch

NUM_SLOTS = 2
NUM_LAYERS = 32
NUM_KV_HEADS = 8
MAX_SEQ_LEN = 2048
GLOBAL_CHUNK = 1024
LOCAL_CHUNK = 256
PAGE_TOKENS = 32
HEAD_DIM = 128


def _check_index(name: str, value: int, upper: int) -> None:
    if type(value) is not int or not 0 <= value < upper:
        raise ValueError(f"{name} must be an int in [0, {upper}), got {value!r}")


def logical_tag(slot: int, layer: int, head: int, position: int) -> int:
    """Pack all non-K/V coordinates into one collision-free integer."""
    _check_index("slot", slot, NUM_SLOTS)
    _check_index("layer", layer, NUM_LAYERS)
    _check_index("head", head, NUM_KV_HEADS)
    _check_index("position", position, MAX_SEQ_LEN)
    return (((slot * NUM_LAYERS + layer) * NUM_KV_HEADS + head) * MAX_SEQ_LEN) + position


def device_major_positions(start: int) -> list[int]:
    """Order one logical 1K chunk as four contiguous 256-token SP shards."""
    if type(start) is not int or start not in range(0, MAX_SEQ_LEN, GLOBAL_CHUNK):
        raise ValueError(f"start must be a 1K boundary below {MAX_SEQ_LEN}, got {start!r}")
    positions = range(start, start + GLOBAL_CHUNK)
    return [
        position for owner in range(4) for position in positions if (position % GLOBAL_CHUNK) // LOCAL_CHUNK == owner
    ]


def tagged_values(kind: str, slot: int, layer: int, positions) -> torch.Tensor:
    """Encode every logical coordinate in values preserved exactly by BF16 and BFP8."""
    if kind not in ("k", "v"):
        raise ValueError(f"kind must be 'k' or 'v', got {kind!r}")
    _check_index("slot", slot, NUM_SLOTS)
    _check_index("layer", layer, NUM_LAYERS)
    positions = list(positions)
    if not positions:
        raise ValueError("positions cannot be empty")
    for position in positions:
        _check_index("position", position, MAX_SEQ_LEN)

    position_tensor = torch.tensor(positions, dtype=torch.int64)
    heads = torch.arange(NUM_KV_HEADS, dtype=torch.int64)
    dimensions = torch.arange(HEAD_DIM, dtype=torch.int64)
    tags = (((slot * NUM_LAYERS + layer) * NUM_KV_HEADS + heads[:, None]) * MAX_SEQ_LEN) + position_tensor[None, :]
    bits = (tags[:, :, None] >> (dimensions[None, None, :] % 20)) & 1
    values = (32 + bits * 32).float()
    return values if kind == "k" else -values


def tagged_page(kind: str, slot: int, layer: int, head: int, position: int) -> torch.Tensor:
    """Build one logical 32-token page without consulting the production table."""
    _check_index("head", head, NUM_KV_HEADS)
    if type(position) is not int or position % PAGE_TOKENS or not 0 <= position <= MAX_SEQ_LEN - PAGE_TOKENS:
        raise ValueError(f"position must be a 32-token page start below {MAX_SEQ_LEN}, got {position!r}")
    return tagged_values(kind, slot, layer, range(position, position + PAGE_TOKENS))[head : head + 1].unsqueeze(0)


def independent_tensor_location(position: int) -> tuple[int, int]:
    """Map a logical page to its live cache shard without using table arithmetic."""
    if type(position) is not int or position % PAGE_TOKENS or not 0 <= position < MAX_SEQ_LEN:
        raise ValueError(f"position must be a 32-token page start below {MAX_SEQ_LEN}, got {position!r}")
    offset = position % GLOBAL_CHUNK
    sp_row = offset // LOCAL_CHUNK
    local_position = (position // GLOBAL_CHUNK) * LOCAL_CHUNK + (offset % LOCAL_CHUNK)
    return sp_row, local_position


def assert_page_equal(actual: torch.Tensor, expected: torch.Tensor, description: str) -> None:
    """Require exact values and the physical page shape used by the address table."""
    page_shape = (1, 1, PAGE_TOKENS, HEAD_DIM)
    if tuple(actual.shape) != page_shape or tuple(expected.shape) != page_shape:
        raise AssertionError(
            f"KV page mismatch for {description}: actual={tuple(actual.shape)}, expected={tuple(expected.shape)}"
        )
    if not torch.equal(actual.float(), expected.float()):
        mismatches = torch.count_nonzero(actual.float() != expected.float()).item()
        raise AssertionError(f"KV page mismatch for {description}: {mismatches} values differ")
