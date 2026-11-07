# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode weights as one prefetched stream (rebuild stage 1).

ttnn.dram_prefetcher runs on 8 sender cores (one per DRAM bank) and pushes weight tensors into a GlobalCircularBuffer
ring on 64 receiver cores (8 per sender) while other ops run. It moves tiles, not values, so the ternary records go
through it as opaque bytes:

- One chunk tensor = [CHUNK_PAGES * 64 rows, 64 * 32 columns] bf16, TILE, width-sharded over the 8 banks. Tile column g
  is receiver g's part (g = 8 * sender + j, j the receiver's place in its sender's row-wise core order), and its tiles
  top to bottom are that receiver's next 2-KB halves of 4-KB record pages.
- The prefetcher reads each bank's shard in 64 blocks of (rows / 64) tile rows and gives receiver j its column of each
  block, so receiver g gets exactly its column, in order, BLOCK_BYTES per ring page.
- Every chunk has the same shape, so the ring's page size never changes and a 4-KB record page never wraps the ring.
- A whole decode step is a list of chunks (the prefetcher's per-"layer" address list); an op takes its pages from the
  receiver's ring in stream order (kernels/tgcb_reader.cpp), so ops may start and end anywhere inside a chunk.
"""

import numpy as np
import torch

import ttnn

PAGE = 4096
BLOCK_PAGES = 2  # 4-KB pages per prefetcher block (ring page): 8 KB
NBLK = 64  # prefetcher blocks per chunk (8 readers x 8 receivers)
CHUNK_PAGES = NBLK * BLOCK_PAGES  # 4-KB pages per receiver per chunk (512 KB)
RECEIVERS_PER_SENDER = 8
NUM_SENDERS = 8
NUM_RECEIVERS = NUM_SENDERS * RECEIVERS_PER_SENDER


def page_tiles(pages: np.ndarray) -> np.ndarray:
    """[n, 1024] uint32 4-KB pages -> [2n * 32, 32] int16 rows whose TILE layout (four 16x16 faces, each row-major)
    holds exactly the page bytes, two bf16 tiles per page."""
    h = np.ascontiguousarray(pages).view(np.int16).reshape(-1, 2, 2, 16, 16)  # [tile, face row, face col, r, c]
    return np.ascontiguousarray(h.transpose(0, 1, 3, 2, 4)).reshape(-1, 32)


def tiles_to_pages(rows: np.ndarray) -> np.ndarray:
    """Inverse of page_tiles."""
    t = np.ascontiguousarray(rows).reshape(-1, 2, 16, 2, 16).transpose(0, 1, 3, 2, 4)
    return np.ascontiguousarray(t).view(np.uint32).reshape(-1, 1024)


class Prefetch:
    """Sender/receiver cores, sub-devices and the ring. Ops that read the stream must run on `receivers` only."""

    def __init__(self, device, ring_bytes=CHUNK_PAGES * PAGE):
        from models.tt_transformers.tt.prefetcher import ARCH_CONFIG, generate_sender_receiver_mapping

        cfg = ARCH_CONFIG["blackhole"]
        raw = generate_sender_receiver_mapping(num_receivers_per_sender=RECEIVERS_PER_SENDER)
        senders = [(cfg["sender_cols"]["left"], y) for y in cfg["bank_ordered_y_coords"]["left"]]
        senders += [(cfg["sender_cols"]["right"], y) for y in cfg["bank_ordered_y_coords"]["right"]]
        assert len(senders) == NUM_SENDERS, senders
        self.mapping, self.receivers = [], []  # receivers[g] = CoreCoord of receiver g
        for sx, sy in senders:
            crs = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in raw[(sx, sy)]]
            )
            self.mapping.append((ttnn.CoreCoord(sx, sy), crs))
            self.receivers += ttnn.corerange_to_cores(crs, row_wise=True)
        self.sender_crs = ttnn.CoreRangeSet([ttnn.CoreRange(s, s) for s, _ in self.mapping])
        self.receiver_crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.receivers])
        self.device = device
        self.manager = device.create_sub_device_manager(
            [ttnn.SubDevice([self.sender_crs]), ttnn.SubDevice([self.receiver_crs])], 0
        )
        device.load_sub_device_manager(self.manager)
        device.set_sub_device_stall_group([ttnn.SubDeviceId(1)])
        self.ring_bytes = ring_bytes
        self.gcb = ttnn.create_global_circular_buffer(device, self.mapping, ring_bytes)

    def launch(self, chunks, layers=1):
        """Stream `chunks` (WeightStream.build) in order, `layers` times; returns at once (the senders run on their
        own). The chunks are the tensors of one prefetcher layer: a sender waits at the end of each layer until its
        receivers have acked every block (remote_cb_sender_barrier), so a layer per chunk would stall at every chunk.
        The reader keeps 3 + 3 per tensor runtime args, so a layer holds at most ~100 chunks."""
        addrs = torch.tensor([[c.buffer_address() for c in chunks] * layers] * NUM_SENDERS, dtype=torch.int32)
        self._addrs = ttnn.as_tensor(
            addrs,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.sender_crs, [1, layers * len(chunks)], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        ttnn.dram_prefetcher(
            list(chunks) + [self._addrs], num_layers=layers, enable_performance_mode=True, global_cb=self.gcb
        )

    def drain(self, blocks):
        """Consume `blocks` ring pages on every receiver (the stream's tail padding)."""
        if blocks:
            ttnn.experimental.test_dram_prefetcher_consumer(
                self.device, num_iters=blocks, page_size_bytes=BLOCK_PAGES * PAGE, global_cb=self.gcb
            )

    def close(self):
        self.device.reset_sub_device_stall_group()
        self.device.clear_loaded_sub_device_manager()
        self.device.remove_sub_device_manager(self.manager)


class WeightStream:
    """Per-receiver page streams, built op by op, then cut into chunk tensors."""

    def __init__(self):
        self.parts = [[] for _ in range(NUM_RECEIVERS)]
        self.length = [0] * NUM_RECEIVERS  # 4-KB pages so far, always whole blocks

    def add(self, per_receiver):
        """per_receiver[g]: [n_g, 1024] uint32 pages receiver g's op reads, in order. Returns (pages, blocks) per
        receiver for the consumer's runtime args."""
        out = []
        for g, p in enumerate(per_receiver):
            n = 0 if p is None else p.shape[0]
            nb = -(-n // BLOCK_PAGES)
            if n:
                self.parts[g].append(p)
            if nb * BLOCK_PAGES > n:
                self.parts[g].append(np.zeros((nb * BLOCK_PAGES - n, 1024), np.uint32))
            self.length[g] += nb * BLOCK_PAGES
            out.append((n, nb))
        return out

    def build(self, device):
        """-> (chunk tensors, tail blocks per receiver). Every receiver's stream is padded to the same whole number of
        chunks; the tail padding differs per receiver, so the stream must be drained per receiver (equal tails only for
        now: Prefetch.drain)."""
        total = -(-max(self.length) // CHUNK_PAGES) * CHUNK_PAGES
        tails = [(total - n) // BLOCK_PAGES for n in self.length]
        cols = []
        for g in range(NUM_RECEIVERS):
            s = np.concatenate(self.parts[g] + [np.zeros((total - self.length[g], 1024), np.uint32)])
            cols.append(page_tiles(s))  # [2 * total * 32, 32]
        rows_per_chunk = 2 * CHUNK_PAGES * 32
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NUM_SENDERS - 1, 0))}),
                [rows_per_chunk, RECEIVERS_PER_SENDER * 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        chunks = []
        for c in range(total // CHUNK_PAGES):
            m = np.concatenate([col[c * rows_per_chunk : (c + 1) * rows_per_chunk] for col in cols], axis=1)
            t = torch.from_numpy(np.ascontiguousarray(m)).view(torch.bfloat16).reshape(1, 1, rows_per_chunk, -1)
            chunks.append(
                ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
            )
        return chunks, tails
