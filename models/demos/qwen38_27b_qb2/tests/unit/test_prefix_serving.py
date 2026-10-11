# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise request ownership and real-file restore under serving-style leases."""

import hashlib
import tempfile
import threading
import unittest
from pathlib import Path

from models.demos.qwen38_27b_qb2.tt.prefix_checkpoint import Checkpoint, Identity, Layout
from models.demos.qwen38_27b_qb2.tt.prefix_serving import (
    CancelledPrefixRequest,
    PrefixDeviceFailure,
    PrefixServingCache,
)
from models.demos.qwen38_27b_qb2.tt.prefix_storage import AtomicDirectoryStore


class MemoryDriver:
    """Opaque per-page/slot state; no TT or numerical correctness claim."""

    def __init__(self):
        self.data, self.events = {}, []
        self.on_write = None
        self.fail_write = self.fail_fence = False

    def prepare(self):
        self.events.append("publish_resident_state")
        self.fence()

    def fence(self):
        self.events.append("fence_all_ranks")
        if self.fail_fence:
            raise OSError("injected device fence failure")

    def reset(self, slot):
        self.events.append(("reset", slot))
        for key, value in self.data.items():
            if key[2] in ("recurrent", "conv") and key[3] == slot:
                value[:] = bytes(len(value))

    def transfer(self, checkpoint, slot, pages):
        driver = self

        class Transfer:
            def pieces(self, segment):
                if segment.kind in ("key", "value"):
                    return [
                        driver.data.setdefault(
                            (segment.rank, segment.layer, segment.kind, page),
                            bytearray(checkpoint.layout.kv_page_bytes),
                        )
                        for page in pages
                    ]
                return [
                    driver.data.setdefault((segment.rank, segment.layer, segment.kind, slot), bytearray(segment.size))
                ]

            def read(self, segment, offset, size):
                return b"".join(self.pieces(segment))[offset : offset + size]

            def write(self, segment, offset, data):
                parts = self.pieces(segment)
                joined = bytearray(b"".join(parts))
                joined[offset : offset + len(data)] = data
                cursor = 0
                for part in parts:
                    part[:] = joined[cursor : cursor + len(part)]
                    cursor += len(part)
                if driver.on_write is not None:
                    callback, driver.on_write = driver.on_write, None
                    callback()
                if driver.fail_write:
                    raise OSError("injected device write failure")

            def fence(self):
                driver.fence()

            def close(self, *, discard=False):
                driver.events.append(("close", discard))
                driver.fence()

        return Transfer()


class PrefixServingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.layout = Layout(("full_attention", "linear_attention"), 4, 4, 16, 64, 12, "test-abi")
        self.identity = Identity("weights-sha", "implementation-sha", "policy-sha", "deployment")
        self.store = AtomicDirectoryStore(Path(self.temporary.name) / "cache", max_bytes=100_000, create=True)
        self.driver = MemoryDriver()
        self.cache = PrefixServingCache(self.store, self.identity, self.layout, self.driver, slots=4, num_pages=100)
        self.tokens = tuple(range(17))

    def admit(self, request_id, slot, pages, *, tokens=None, namespace="tenant-A"):
        return self.cache.admit(
            request_id, slot=slot, tokens=self.tokens if tokens is None else tokens, pages=pages, namespace=namespace
        )

    def populate(self):
        handle = self.admit("source", 1, (8, 2, 9, 7, 6))
        self.cache.advance(handle, 8)
        checkpoint = Checkpoint.for_tokens(
            Identity("weights-sha", "implementation-sha", "policy-sha", "tenant-A"), self.layout, self.tokens, 8
        )
        transfer = self.driver.transfer(checkpoint, 1, (8, 2))
        expected = {}
        for segment in checkpoint.segments():
            expected[segment] = hashlib.shake_256(repr(segment).encode()).digest(segment.size)
            transfer.write(segment, 0, expected[segment])
        self.assertTrue(self.cache.capture(handle))
        return handle, checkpoint, expected

    def test_all_rank_restore_uses_private_pages_and_exact_consumed_frontier(self):
        with self.cache.execution():
            source, checkpoint, expected = self.populate()
            destination = self.admit("destination", 3, (33, 10, 24, 49, 31))
            self.assertEqual(self.cache.restore_longest(destination, [4, 8, 16]), 8)
            self.assertEqual(self.cache.frontier(destination), 8)
            transfer = self.driver.transfer(checkpoint, 3, (33, 10))
            self.assertEqual({s: transfer.read(s, 0, s.size) for s in checkpoint.segments()}, expected)
            self.assertEqual(self.cache.frontier(source), 8)
            self.assertEqual(self.cache.stats["hits"], 1)
            with self.assertRaises(ValueError):
                self.cache.restore_longest(destination, [8])

    def test_different_tenants_and_divergent_prefixes_are_misses(self):
        with self.cache.execution():
            self.populate()
            other = self.admit("other", 2, (20, 21, 22), namespace="tenant-B")
            changed = self.admit("changed", 3, (30, 31, 32), tokens=(99,) + self.tokens[1:])
            self.assertEqual(self.cache.restore_longest(other, [8]), 0)
            self.assertEqual(self.cache.restore_longest(changed, [8]), 0)

    def test_request_generation_prevents_stale_cancel_and_release_after_slot_reuse(self):
        with self.cache.execution():
            first = self.admit("same-id", 1, (8, 2))
            self.cache.cancel(first)
            self.cache.release(first)
            second = self.admit("same-id", 1, (8, 2))
            self.assertNotEqual(first, second)
            for callback in (self.cache.cancel, self.cache.release, self.cache.frontier):
                with self.assertRaises(CancelledPrefixRequest):
                    callback(first)
            self.assertEqual(self.cache.frontier(second), 0)

    def test_aliasing_or_unleased_cache_operations_are_rejected(self):
        with self.assertRaises(RuntimeError):
            self.admit("outside", 0, (0,))
        with self.cache.execution():
            handle = self.admit("one", 0, (0, 1))
            for slot, pages in ((0, (5,)), (1, (1, 2)), (1, (2, 2)), (4, (5,))):
                with self.assertRaises(ValueError):
                    self.admit("other", slot, pages)
            self.cache.cancel(handle)
            with self.assertRaises(ValueError):
                self.admit("still-held", 0, (5,))

    def test_update_preserves_consumed_history_pages_and_full_slot_permutation(self):
        with self.cache.execution():
            handle = self.admit("one", 1, (8, 2))
            self.cache.advance(handle, 8)
            self.cache.update(handle, tokens=self.tokens + (17,), pages=(8, 2, 9))
            for tokens, pages in (((99,) + self.tokens[1:], (8, 2, 9)), (self.tokens, (8, 3, 9))):
                with self.assertRaises(ValueError):
                    self.cache.update(handle, tokens=tokens, pages=pages)
            self.cache.remap_slots({0: 1, 1: 3, 2: 0, 3: 2})
            self.admit("reused-old-slot", 1, (20,))
            with self.assertRaises(ValueError):
                self.admit("aliases-moved-slot", 3, (30,))
            with self.assertRaises(ValueError):
                self.cache.remap_slots({1: 3})

    def test_corruption_resets_and_fences_before_cold_fallback(self):
        with self.cache.execution():
            _, checkpoint, _ = self.populate()
            path = self.store._path(checkpoint.key)
            content = bytearray(path.read_bytes())
            content[-12] ^= 1
            path.write_bytes(content)
            handle = self.admit("destination", 3, (30, 31, 32))
            self.assertEqual(self.cache.restore_longest(handle, [8]), 0)
            self.assertEqual(self.cache.frontier(handle), 0)
            self.assertEqual(self.driver.events[-2:], [("reset", 3), "fence_all_ranks"])
            for key, data in self.driver.data.items():
                if key[2] in ("conv", "recurrent") and key[3] == 3:
                    self.assertEqual(bytes(data), bytes(len(data)))

    def test_cancellation_during_transfer_aborts_before_publication_and_keeps_slot_held(self):
        with self.cache.execution():
            self.populate()
            handle = self.admit("destination", 3, (30, 31, 32))

            def cancel_from_another_thread():
                thread = threading.Thread(target=self.cache.cancel, args=(handle,))
                thread.start()
                thread.join(1)
                self.assertFalse(thread.is_alive(), "Cancellation must not wait on the device execution lease")

            self.driver.on_write = cancel_from_another_thread
            with self.assertRaises(CancelledPrefixRequest):
                self.cache.restore_longest(handle, [8])
            self.assertEqual(self.driver.events[-2:], [("reset", 3), "fence_all_ranks"])
            with self.assertRaises(ValueError):
                self.admit("slot-reuse-too-early", 3, (40,))
            self.cache.release(handle)
            self.admit("new-generation", 3, (40,))

    def test_device_failure_is_fatal_and_not_a_storage_miss(self):
        with self.cache.execution():
            self.populate()
            handle = self.admit("destination", 3, (30, 31, 32))
            self.driver.fail_write = True
            with self.assertRaises(PrefixDeviceFailure):
                self.cache.restore_longest(handle, [8])
            self.assertEqual(self.cache.stats["restore_storage_misses"], 0)
        with self.assertRaises(PrefixDeviceFailure), self.cache.execution():
            pass

    def test_exact_prompt_hit_leaves_a_token_for_logits_and_allocation_bounds_candidates(self):
        with self.cache.execution():
            self.populate()
            exact = self.admit("exact", 2, (20, 21), tokens=self.tokens[:8])
            short = self.admit("short-allocation", 3, (30,))
            self.assertEqual(self.cache.restore_longest(exact, [8]), 0)
            self.assertEqual(self.cache.restore_longest(short, [8]), 0)

    def test_multimodal_placeholder_tokens_cannot_hit_a_text_snapshot(self):
        with self.cache.execution(), self.assertRaises(ValueError):
            self.cache.admit("vision", slot=0, tokens=self.tokens, pages=(0,), namespace="tenant-A", multimodal=True)


if __name__ == "__main__":
    unittest.main()
