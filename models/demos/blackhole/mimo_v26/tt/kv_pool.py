# SPDX-License-Identifier: Apache-2.0
"""KV pool layout shared by prefill (tt/prefill.py) and decode (tt/decode.py, tt/generator.py). Host-only: no device allocation here.

Contract (one pool per mesh row; every row uses the SAME layout and holds different users; slot = user index inside the row, 0..slots-1):
  * Per decoder layer MiMoAttention owns k_cache and v_cache: bf16 TILE [n_blocks, 1, 64, 192] replicated on every device of the mesh (each device writes its own KV head:
    SWA mesh column c holds kv head c, GA column c holds kv head c//2). V is stored zero padded to 192 with V*0.707 already folded in.
  * Global (GA) layer: n_blocks = slots * nlog, nlog = max_pos / 64. Slot s, logical block i (positions 64i..64i+63) -> physical block s * nlog + i.
  * SWA layer (ring): n_blocks = slots * ring_blocks. Slot s, logical block i -> physical block s * ring_blocks + (i % ring_blocks). ring_blocks * 64 must be >= 128 (window) +
    the longest prefill chunk (chunk/64 + 2 blocks minimum; default chunk/64 + 8). Decode uses cache_position_modulo = ring_blocks * 64 with the [slots, ring_blocks] table;
    chunked prefill uses the logical row (ring table indexed i % ring_blocks, length nlog).
  * Page tables are int32 row-major, padded on the last dim to a multiple of 8 entries (32 B stick) when uploaded.
  * Positions: decode position p of a user is the index of the token being fed (prompt length for the first generated token). Per row AND per slot (rows differ).
  * Dead slots (no user / finished): feed position 0 and any token; their cache blocks are never read by live slots (no sharing between slots).
"""
from dataclasses import dataclass

import torch

BLOCK = 64


@dataclass(frozen=True)
class KVPoolLayout:
    max_pos: int
    ring_blocks: int
    slots: int = 32

    def __post_init__(self):
        assert self.max_pos % 256 == 0 and self.ring_blocks * BLOCK >= 128 + 64

    @property
    def nlog(self):
        return self.max_pos // BLOCK

    def n_blocks(self, swa: bool):
        return self.slots * (self.ring_blocks if swa else self.nlog)

    def page_table(self, swa: bool) -> torch.Tensor:
        """[slots, ring_blocks] (SWA ring table) or [slots, nlog] (GA) int32 physical block ids."""
        nb = self.ring_blocks if swa else self.nlog
        return (
            torch.arange(self.slots, dtype=torch.int32)[:, None] * nb + torch.arange(nb, dtype=torch.int32)[None, :]
        ).to(torch.int32)

    def logical_row(self, slot: int, swa: bool) -> torch.Tensor:
        """[nlog] physical block of each logical block of `slot` (ring aliasing for SWA)."""
        pt = self.page_table(swa)[slot]
        if not swa:
            return pt
        return torch.stack([pt[i % self.ring_blocks] for i in range(self.nlog)]).to(torch.int32)
