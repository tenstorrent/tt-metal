# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixed 4x1K batches in CP-major, request-major, local-row order.

Unlike request-major ragged packing, every CP rank already owns its 128 rows
from every request. Attention needs only local slices and concatenation.
The durable cache geometry is 1K per request, never the combined 4K size.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkedRequest:
    request_id: int
    slot_id: int
    actual_start: int
    token_ids: tuple[int, ...]

    @property
    def actual_end(self):
        return self.actual_start + len(self.token_ids)


@dataclass(frozen=True)
class ChunkedBatchPlan:
    batch_size: int = 4
    chunk_size: int = 1024
    cp: int = 8
    tp: int = 4

    def __post_init__(self):
        if (self.batch_size, self.chunk_size, self.cp, self.tp) != (4, 1024, 8, 4):
            raise ValueError("Fixed chunked batching currently supports only 4x1024 on CP8/TP4")

    @property
    def local_rows(self):
        return self.chunk_size // self.cp

    @property
    def packed_size(self):
        return self.batch_size * self.chunk_size

    def validate(self, requests, *, num_slots, max_seq_len, vocab_size):
        if len(requests) != self.batch_size:
            raise ValueError(f"A fixed batch requires exactly {self.batch_size} requests")
        if len({r.slot_id for r in requests}) != self.batch_size:
            raise ValueError("A slot may appear only once in a batch")
        if len({r.request_id for r in requests}) != self.batch_size:
            raise ValueError("A request may appear only once in a batch")
        for req in requests:
            if not isinstance(req.slot_id, int) or not 0 <= req.slot_id < num_slots:
                raise ValueError("Request slot is outside the allocated cache")
            if not isinstance(req.actual_start, int) or req.actual_start < 0 or req.actual_start % self.chunk_size:
                raise ValueError("Request starts must be nonnegative multiples of 1024")
            if not 0 < len(req.token_ids) <= self.chunk_size or req.actual_end > max_seq_len:
                raise ValueError("Each request must contain 1..1024 tokens within the context capacity")
            if any(not isinstance(token, int) or not 0 <= token < vocab_size for token in req.token_ids):
                raise ValueError("Token IDs must be integers within the model vocabulary")

    def pack(self, requests, *, positions=False):
        rows = []
        for req in requests:
            useful = list(range(req.actual_start, req.actual_end)) if positions else list(req.token_ids)
            rows.append(useful + [0] * (self.chunk_size - len(useful)))
        return [
            value
            for rank in range(self.cp)
            for request in rows
            for value in request[rank * self.local_rows : (rank + 1) * self.local_rows]
        ]

    def unpack(self, packed):
        """Inverse host mapping: [..., CP*batch*local_rows, width] -> [batch, chunk, width]."""
        return (
            packed.reshape(self.cp, self.batch_size, self.local_rows, packed.shape[-1])
            .permute(1, 0, 2, 3)
            .reshape(self.batch_size, self.chunk_size, packed.shape[-1])
        )


class ChunkedAttentionLayout:
    def __init__(self, plan, metadata):
        self.plan = plan
        self.metadata = tuple(metadata)
        if len(self.metadata) != plan.batch_size:
            raise ValueError("One stable metadata buffer is required per batch lane")

    def split(self, tensor, lane):
        import ttnn

        start = lane * self.plan.local_rows
        return ttnn.slice(
            tensor,
            (0, 0, start, 0),
            (1, tensor.shape[1], start + self.plan.local_rows, tensor.shape[-1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @staticmethod
    def concatenate(outputs):
        import ttnn

        combined = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for output in outputs:
            output.deallocate(True)
        return combined
