# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded asynchronous-copy mechanism tests; these are not TT DMA validation."""

import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.demos.qwen38_27b_qb2.tt.prefix_checkpoint import MAX_CHUNK_BYTES, Checkpoint, Identity, Layout
from models.demos.qwen38_27b_qb2.tt.prefix_transfer import PackedCacheTransfer


class Host:
    def __init__(self, rank, size):
        self.rank, self.raw = rank, torch.zeros(size, dtype=torch.uint8)

    def host_buffer(self):
        return self

    def get_shard(self, coordinate):
        return self.raw if coordinate == (0, self.rank) else None


class RankTensor:
    def __init__(self, runtime, rank, shape, dtype, layout, row_bytes, *, raw=None):
        self.runtime, self.rank, self.shape = runtime, rank, tuple(shape)
        self.dtype, self.layout, self.row_bytes = dtype, layout, row_bytes
        self.raw = torch.randint(0, 256, (shape[0] * row_bytes,), dtype=torch.uint8) if raw is None else raw

    def buffer_address(self):
        return self.raw.data_ptr()

    def cpu(self, *, blocking):
        host = Host(self.rank, self.raw.numel())
        weak = weakref.ref(host)

        def complete():
            target = weak()
            if target is None:
                raise RuntimeError("Asynchronous download lost its host allocation")
            target.raw.copy_(self.raw)

        self.runtime.pending.append(complete)
        if blocking:
            self.runtime.synchronize_device(None)
        return host


class Runtime:
    bfloat8_b, float32, bfloat16 = "bfp8", "fp32", "bf16"
    TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = "tile", "row", "dram"

    def __init__(self):
        self.pending = []
        self.syncs = 0
        self.fail_sync = False

    @staticmethod
    def MeshCoordinate(row, column):
        return (row, column)

    @staticmethod
    def get_device_tensors(tensor):
        return tensor.shards

    @staticmethod
    def narrow(tensor, dimension, start, length):
        if dimension != 0:
            raise ValueError("Test runtime implements dimension-zero rank views")
        return RankTensor(
            tensor.runtime,
            tensor.rank,
            (length, *tensor.shape[1:]),
            tensor.dtype,
            tensor.layout,
            tensor.row_bytes,
            raw=tensor.raw[start * tensor.row_bytes : (start + length) * tensor.row_bytes],
        )

    def copy_host_to_device_tensor(self, host, view):
        weak = weakref.ref(host)

        def complete():
            source = weak()
            if source is None:
                raise RuntimeError("Asynchronous upload lost its host allocation")
            view.raw.copy_(source.raw)

        self.pending.append(complete)

    def synchronize_device(self, mesh):
        self.syncs += 1
        if self.fail_sync:
            raise OSError("injected incomplete device fence")
        pending, self.pending = self.pending, []
        for complete in pending:
            complete()

    def tensor(self, shape, dtype, layout, row_bytes):
        return SimpleNamespace(
            shape=shape,
            dtype=dtype,
            layout=layout,
            memory_config=lambda: self.DRAM_MEMORY_CONFIG,
            shards=[RankTensor(self, r, shape, dtype, layout, row_bytes) for r in range(4)],
        )


class PrefixTransferBatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.runtime = Runtime()
        patcher = patch.dict("sys.modules", {"ttnn": self.runtime})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mesh = SimpleNamespace(shape=(1, 4))
        r = self.runtime
        self.tensors = {
            "key": r.tensor((128, 1, 32, 256), r.bfloat8_b, r.TILE_LAYOUT, 8704),
            "value": r.tensor((128, 1, 32, 256), r.bfloat8_b, r.TILE_LAYOUT, 8704),
            "recurrent": r.tensor((4, 12, 128, 128), r.float32, r.TILE_LAYOUT, 786432),
            "conv": r.tensor((4, 3, 2560), r.bfloat16, r.ROW_MAJOR_LAYOUT, 15360),
        }
        self.cache = SimpleNamespace(
            batch_size=4,
            num_pages=128,
            layers=[
                SimpleNamespace(key=self.tensors["key"], value=self.tensors["value"]),
                SimpleNamespace(recurrent=self.tensors["recurrent"], conv=self.tensors["conv"]),
            ],
        )
        self.checkpoint = Checkpoint.for_tokens(
            Identity("weights", "implementation", "policy", "test"),
            Layout(("full_attention", "linear_attention"), 4, 32, 8704, 786432, 15360, "test-abi"),
            list(range(1025)),
            1024,
        )
        self.source_pages, self.destination_pages = tuple(range(0, 64, 2)), tuple(range(65, 128, 2))

    def transfer(self, *, batched, slot=1, pages=None):
        return PackedCacheTransfer(
            self.mesh,
            self.cache,
            self.checkpoint,
            slot=slot,
            pages=self.source_pages if pages is None else pages,
            batched=batched,
        )

    def test_batched_capture_reads_exact_packed_bytes_with_one_fence_per_bounded_group(self):
        serial, batched = self.transfer(batched=False), self.transfer(batched=True)
        key = next(self.checkpoint.segments())
        before = self.runtime.syncs
        expected = serial.read(key, 0, key.size)
        serial_syncs = self.runtime.syncs - before
        before = self.runtime.syncs
        actual = batched.read(key, 0, key.size)
        self.assertEqual(actual, expected)
        self.assertEqual(serial_syncs, 32)
        self.assertEqual(self.runtime.syncs - before, 1)
        self.assertLessEqual(batched.max_host_staging_bytes, MAX_CHUNK_BYTES)
        self.assertEqual(batched.pending, [])

    def test_arbitrary_chunks_restore_every_rank_and_preserve_all_neighbours(self):
        source = self.transfer(batched=True)
        target = self.transfer(batched=True, slot=3, pages=self.destination_pages)
        before = {name: [shard.raw.clone() for shard in value.shards] for name, value in self.tensors.items()}
        for segment in self.checkpoint.segments():
            payload = source.read(segment, 0, segment.size)
            for offset in range(0, segment.size, 77777):
                target.write(segment, offset, payload[offset : offset + 77777])
        source.close()
        target.close()
        for name, tensor in self.tensors.items():
            for rank, shard in enumerate(tensor.shards):
                expected = before[name][rank].clone()
                unit = 8704 if name in ("key", "value") else getattr(self.checkpoint.layout, name + "_bytes")
                pairs = zip(self.source_pages, self.destination_pages) if name in ("key", "value") else [(1, 3)]
                for start, destination in pairs:
                    expected[destination * unit : (destination + 1) * unit] = before[name][rank][
                        start * unit : (start + 1) * unit
                    ]
                self.assertTrue(torch.equal(expected, shard.raw), (name, rank))
        self.assertLessEqual(source.max_host_staging_bytes, MAX_CHUNK_BYTES)
        self.assertLessEqual(target.max_host_staging_bytes, MAX_CHUNK_BYTES)
        self.assertEqual(self.runtime.pending, [])

    def test_failed_fence_keeps_host_buffers_alive_and_refuses_more_submissions(self):
        transfer = self.transfer(batched=True)
        self.runtime.fail_sync = True
        key = next(self.checkpoint.segments())
        with self.assertRaises(OSError):
            transfer.read(key, 0, key.size)
        self.assertEqual(len(transfer.pending), 32)
        with self.assertRaises(RuntimeError):
            transfer.read(key, 0, key.size)
        self.runtime.fail_sync = False
        transfer.close(discard=True)
        self.assertEqual(transfer.pending, [])


if __name__ == "__main__":
    unittest.main()
