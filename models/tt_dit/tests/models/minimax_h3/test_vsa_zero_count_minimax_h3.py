# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only contract test: a zero-count block is a no-op in the fine stage.

Capacity-mode geometry lists empty (``block_counts == 0``) slots in every selection row -- the
exempt prefix names all exempt slots and unfilled candidate slots stay listed. The vsa_sdpa reader
skips them (the streaming leader never fetches a zero-count block; the distributed reader drops it
from the listing), so listing one must not change a row's output. This locks that invariant on the
torch reference the device A/B is checked against.
"""

from __future__ import annotations

import pytest
import torch

from models.tt_dit.tests.models.minimax_h3.vsa_oracle import VSA_TILE_TOKENS, fine_attention

VSA_INDEX_SENTINEL = 0xFFFFFFFF


def _tiles(n_blocks: int, dim: int = 16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One query tile against ``n_blocks`` key/value blocks (one index row per query tile)."""
    torch.manual_seed(0)
    kv = n_blocks * VSA_TILE_TOKENS
    q = torch.randn(1, 1, VSA_TILE_TOKENS, dim)
    k = torch.randn(1, 1, kv, dim)
    v = torch.randn(1, 1, kv, dim)
    return q, k, v


def _row(entries: list[int], width: int) -> torch.Tensor:
    padded = entries + [VSA_INDEX_SENTINEL] * (width - len(entries))
    return torch.tensor(padded, dtype=torch.int64).reshape(1, 1, 1, width).to(torch.uint32)


@pytest.mark.parametrize("zero_pos", ["exempt_prefix", "candidate"])
def test_zero_count_block_is_noop(zero_pos):
    """Listing block 2 (count 0) leaves the single q-tile's output bit-identical to omitting it."""
    n_blocks = 5
    q, k, v = _tiles(n_blocks)
    counts = torch.full((n_blocks,), VSA_TILE_TOKENS, dtype=torch.int64)
    counts[2] = 0  # empty slot

    width = n_blocks + 2
    listed = [2, 0, 3] if zero_pos == "exempt_prefix" else [0, 3, 2]
    omitted = [0, 3]

    with_zero = fine_attention(q, k, v, _row(listed, width), counts)
    without_zero = fine_attention(q, k, v, _row(omitted, width), counts)
    assert torch.equal(with_zero, without_zero)


def test_ragged_zero_count_mix():
    """A ragged (count < 64) real block and a zero-count block together: only the ragged one counts."""
    n_blocks = 4
    q, k, v = _tiles(n_blocks)
    counts = torch.tensor([VSA_TILE_TOKENS, 40, 0, VSA_TILE_TOKENS], dtype=torch.int64)
    width = n_blocks + 2
    with_zero = fine_attention(q, k, v, _row([0, 1, 2], width), counts)
    without_zero = fine_attention(q, k, v, _row([0, 1], width), counts)
    assert torch.equal(with_zero, without_zero)
