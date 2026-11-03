# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""L1 addresses for sizing a program's circular buffers below L1 buffers kept resident (DFlash's), from main's APIs.

A program's circular buffers start at the L1 allocator base and must end below the lowest allocated L1 buffer.
The memory view's block table holds the L1 allocator's blocks with addresses relative to that base (the allocator
adds the base when it hands an address out), so the lowest allocated block + base is the lowest occupied address.
"""
from __future__ import annotations

import ttnn


def l1_allocator_base(device) -> int:
    """L1 address where the L1 allocator, and a program's circular buffers, start on the compute cores."""
    return int(ttnn.get_allocator_base_address(device, ttnn.BufferType.L1))


def lowest_occupied_l1_address(device) -> int | None:
    """Lowest L1 address held by an allocated buffer on the compute cores, or None when no L1 buffer is allocated."""
    blocks = ttnn.get_memory_view(device, ttnn.BufferType.L1).block_table
    allocated = [int(b["address"]) for b in blocks if b["allocated"] == "yes"]
    return min(allocated) + l1_allocator_base(device) if allocated else None
