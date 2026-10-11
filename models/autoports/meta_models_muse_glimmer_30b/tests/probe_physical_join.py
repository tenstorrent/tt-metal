# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental physical-tile joining for logically unaligned prefill tails."""

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import FusedDecoder


class PhysicalJoinDecoder(FusedDecoder):
    def prefill_forward(self, x, *, kv_cache, page_table, start_pos=0, slots=None):
        """Chunked, append-only prefill; logical S may be any positive length.

        Initial unaligned prefix positions use single-row cache updates until the
        next page boundary. Padded final rows occupy only unused request-owned
        cache slots, are causally invisible, and are replaced on continuation.
        """
        batch, _, length, hidden = x.shape
        if not (
            0 <= start_pos < self.config.max_position_embeddings
            and 0 < length <= self.config.max_position_embeddings - start_pos
        ):
            raise ValueError("Prefill exceeds the advertised context or is empty")
        slots = tuple(range(batch)) if slots is None else tuple(slots)
        if len(slots) != batch or len(set(slots)) != batch:
            raise ValueError("Each batch row needs a distinct cache slot")
        results = []
        for b, slot in enumerate(slots):
            pieces, offset = ([], 0)
            while offset < length:
                absolute = start_pos + offset
                if absolute % 32:
                    token = ttnn.slice(x, [b, 0, offset, 0], [b + 1, 1, offset + 1, hidden])
                    row = ttnn.slice(page_table, [slot, 0], [slot + 1, page_table.shape[1]])
                    pos = ttnn.slice(self.positions, [absolute], [absolute + 1])
                    cos = (
                        ttnn.slice(self.cos, [0, 0, absolute, 0], [1, 1, absolute + 1, 128]) if self.use_rope else None
                    )
                    sin = (
                        ttnn.slice(self.sin, [0, 0, absolute, 0], [1, 1, absolute + 1, 128]) if self.use_rope else None
                    )
                    pieces.append(
                        self.decode_forward(
                            token, kv_cache=kv_cache, page_table=row, current_pos=pos, rope_cos=cos, rope_sin=sin
                        )
                    )
                    offset += 1
                    continue
                valid = min(self.CHUNK_SIZE, length - offset)
                physical = (valid + 31) // 32 * 32
                chunk = ttnn.slice(x, [b, 0, offset, 0], [b + 1, 1, offset + valid, hidden])
                if physical != valid:
                    chunk = ttnn.pad(chunk, [(0, 0), (0, 0), (0, physical - valid), (0, 0)], 0)
                out = self._prefill_chunk(
                    chunk, start_pos=absolute, kv_cache=kv_cache, page_table=page_table, slot=slot
                )
                pieces.append(ttnn.slice(out, [0, 0, 0, 0], [1, 1, valid, hidden]))
                offset += valid
            if len(pieces) > 1 and start_pos % self.PAGE_SIZE == 0:
                # Only the last piece has padding. Joining physical tile views
                # does not insert padding between logical tokens.
                physical_pieces = [ttnn.reshape(p, p.padded_shape, p.padded_shape) for p in pieces]
                joined = ttnn.concat(physical_pieces, dim=2)
                results.append(
                    ttnn.reshape(
                        joined, ttnn.Shape([1, 1, length, hidden]), ttnn.Shape([1, 1, (length + 31) // 32 * 32, hidden])
                    )
                )
            else:
                results.append(ttnn.concat(pieces, dim=2) if len(pieces) > 1 else pieces[0])
        return ttnn.concat(results, dim=0) if batch > 1 else results[0]


class LogicalJoinDecoder(FusedDecoder):
    def prefill_forward(self, x, *, kv_cache, page_table, start_pos=0, slots=None):
        """Chunked, append-only prefill; logical S may be any positive length.

        Initial unaligned prefix positions use single-row cache updates until the
        next page boundary. Padded final rows occupy only unused request-owned
        cache slots, are causally invisible, and are replaced on continuation.
        """
        batch, _, length, hidden = x.shape
        if not (
            0 <= start_pos < self.config.max_position_embeddings
            and 0 < length <= self.config.max_position_embeddings - start_pos
        ):
            raise ValueError("Prefill exceeds the advertised context or is empty")
        slots = tuple(range(batch)) if slots is None else tuple(slots)
        if len(slots) != batch or len(set(slots)) != batch:
            raise ValueError("Each batch row needs a distinct cache slot")
        results = []
        for b, slot in enumerate(slots):
            pieces, offset = ([], 0)
            while offset < length:
                absolute = start_pos + offset
                if absolute % 32:
                    token = ttnn.slice(x, [b, 0, offset, 0], [b + 1, 1, offset + 1, hidden])
                    row = ttnn.slice(page_table, [slot, 0], [slot + 1, page_table.shape[1]])
                    pos = ttnn.slice(self.positions, [absolute], [absolute + 1])
                    cos = (
                        ttnn.slice(self.cos, [0, 0, absolute, 0], [1, 1, absolute + 1, 128]) if self.use_rope else None
                    )
                    sin = (
                        ttnn.slice(self.sin, [0, 0, absolute, 0], [1, 1, absolute + 1, 128]) if self.use_rope else None
                    )
                    pieces.append(
                        self.decode_forward(
                            token, kv_cache=kv_cache, page_table=row, current_pos=pos, rope_cos=cos, rope_sin=sin
                        )
                    )
                    offset += 1
                    continue
                valid = min(self.CHUNK_SIZE, length - offset)
                physical = (valid + 31) // 32 * 32
                chunk = ttnn.slice(x, [b, 0, offset, 0], [b + 1, 1, offset + valid, hidden])
                if physical != valid:
                    chunk = ttnn.pad(chunk, [(0, 0), (0, 0), (0, physical - valid), (0, 0)], 0)
                out = self._prefill_chunk(
                    chunk, start_pos=absolute, kv_cache=kv_cache, page_table=page_table, slot=slot
                )
                pieces.append(ttnn.slice(out, [0, 0, 0, 0], [1, 1, valid, hidden]))
                offset += valid
            results.append(ttnn.concat(pieces, dim=2) if len(pieces) > 1 else pieces[0])
        return ttnn.concat(results, dim=0) if batch > 1 else results[0]
