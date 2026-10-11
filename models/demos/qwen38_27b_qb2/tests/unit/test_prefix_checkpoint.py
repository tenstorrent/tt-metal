# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU protocol tests with opaque bytes and real local files; no TT accuracy claim.

The fake device deliberately shuffles page IDs and uses independent per-rank,
per-slot storage. It models fencing/cancellation so controller regressions are
detectable. It cannot prove a future TT adapter implements those obligations.
Run locally without Metal conftest: python -m unittest discover -s <this-dir>
-p test_prefix_checkpoint.py -v
"""

import hashlib
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from models.demos.qwen38_27b_qb2.tt.prefix_checkpoint import (
    COMPLETE,
    MAGIC,
    Checkpoint,
    Identity,
    Layout,
    capture,
    find_prefix,
    restore,
)
from models.demos.qwen38_27b_qb2.tt.prefix_storage import AtomicDirectoryStore


class FakeSource:
    def __init__(self, checkpoint, payload):
        self.checkpoint, self.payload = checkpoint, payload
        self.events = []
        self.fail_read = self.fail_exit = self.short_read = False
        self.max_read = 0

    @contextmanager
    def freeze(self, checkpoint):
        if checkpoint != self.checkpoint:
            raise ValueError("Source is not at the requested consumed frontier")
        self.events.append("fenced_and_pinned")
        try:
            yield self
            if self.fail_exit:
                raise OSError("Source lease fence failed")
        finally:
            self.events.append("unpin")

    def read(self, segment, offset, size):
        if self.fail_read:
            raise OSError("Injected export failure")
        self.max_read = max(self.max_read, size)
        data = self.payload[segment][offset : offset + size]
        return data[:-1] if self.short_read else data


class FakeTarget:
    """A scheduler-owned isolated request slot; publication requires commit."""

    def __init__(self, *, fail_write=False, cancelled=False):
        self.fail_write, self.cancelled = fail_write, cancelled
        self.events = []
        self.visible = None
        self.max_write = 0
        self.inactive_slot = bytearray(b"never change another request")
        self.recurrent_address = object()

    @contextmanager
    def begin(self, checkpoint):
        self.events.append("begin_private_slot")
        self.checkpoint = checkpoint
        # These page IDs intentionally have no relationship to the source.
        self.page_ids = [100 + 7 * i for i in reversed(range(checkpoint.consumed // checkpoint.layout.page_tokens))]
        self.pending = {segment: bytearray(segment.size) for segment in checkpoint.segments()}
        try:
            yield self
        except BaseException:
            self.pending = None
            self.events.append("abort_and_release")
            raise
        finally:
            self.events.append("end_lease")

    def write(self, segment, offset, data):
        self.max_write = max(self.max_write, len(data))
        self.pending[segment][offset : offset + len(data)] = data
        if self.fail_write:
            raise OSError("Injected transfer failure after a partial write")

    def commit(self, consumed):
        self.events.append("fence_all_rank_writes")
        if self.cancelled:
            raise ValueError("Request generation cancelled before commit")
        if consumed != self.checkpoint.consumed:
            raise ValueError("Wrong consumed frontier")
        self.pages = {}
        for segment, data in self.pending.items():
            if segment.kind in ("key", "value"):
                width = self.checkpoint.layout.kv_page_bytes
                for logical, page in enumerate(self.page_ids):
                    self.pages[segment, page] = bytearray(data[logical * width : (logical + 1) * width])
        self.visible = self.pending
        self.pending = None
        self.events.append("publish")


class PrefixCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = Identity("weights-sha", "implementation-sha", "all-effective-policy-sha", "tenant-A")
        self.layout = Layout(("linear_attention", "full_attention") * 2, 4, 4, 16, 64, 12, "tiny-test-abi")
        self.tokens = list(range(17))
        self.checkpoint = Checkpoint.for_tokens(self.identity, self.layout, self.tokens, 8)
        self.store = AtomicDirectoryStore(self.root / "cache", max_bytes=100_000, create=True)
        self.source = self.make_source(self.checkpoint)

    def make_source(self, checkpoint):
        payload = {
            segment: hashlib.shake_256(f"{segment.rank}:{segment.layer}:{segment.kind}".encode()).digest(segment.size)
            for segment in checkpoint.segments()
        }
        return FakeSource(checkpoint, payload)

    def capture(self, checkpoint=None):
        checkpoint = checkpoint or self.checkpoint
        source = self.source if checkpoint == self.checkpoint else self.make_source(checkpoint)
        capture(self.store, checkpoint, source, chunk_bytes=7)

    def corrupt(self, mutate):
        path = self.store._path(self.checkpoint.key)
        path.write_bytes(mutate(path.read_bytes()))

    def test_all_ranks_opaque_bytes_round_trip_to_new_page_ids(self):
        self.capture()
        target = FakeTarget()
        address = target.recurrent_address
        restore(self.store, self.checkpoint, target, chunk_bytes=5)
        self.assertEqual(target.visible, self.source.payload)
        self.assertIs(target.recurrent_address, address)
        self.assertEqual(target.inactive_slot, b"never change another request")
        self.assertEqual(target.events, ["begin_private_slot", "fence_all_rank_writes", "publish", "end_lease"])
        self.assertEqual(self.source.events, ["fenced_and_pinned", "unpin"])
        self.assertEqual(self.store.used_bytes, self.checkpoint.encoded_bytes)
        self.assertLessEqual(self.source.max_read, 7)
        self.assertLessEqual(target.max_write, 5)
        for segment, data in self.source.payload.items():
            if segment.kind in ("key", "value"):
                self.assertEqual(b"".join(target.pages[segment, page] for page in target.page_ids), data)

    def test_two_branches_do_not_share_mutable_state_or_kv_tail(self):
        self.capture()
        left, right = FakeTarget(), FakeTarget()
        restore(self.store, self.checkpoint, left)
        restore(self.store, self.checkpoint, right)
        for segment in self.checkpoint.segments():
            left.visible[segment][0] ^= 255
            self.assertEqual(right.visible[segment], self.source.payload[segment])
        for page in left.pages:
            left.pages[page][-1] ^= 255
            self.assertNotEqual(left.pages[page], right.pages[page])
        fresh = FakeTarget()
        restore(self.store, self.checkpoint, fresh)
        self.assertEqual(fresh.visible, self.source.payload)

    def test_prefix_hit_checks_consumed_tokens_not_suffix(self):
        self.capture()
        other_suffix = self.tokens[:8] + [100, 200, 300]
        self.assertEqual(find_prefix(self.store, self.identity, self.layout, other_suffix, [4, 8]), self.checkpoint)
        changed_prefix = other_suffix.copy()
        changed_prefix[3] += 1
        self.assertIsNone(find_prefix(self.store, self.identity, self.layout, changed_prefix, [4, 8]))

    def test_exact_prefix_hit_leaves_last_token_for_real_logits(self):
        shorter = Checkpoint.for_tokens(self.identity, self.layout, self.tokens, 4)
        self.capture(shorter)
        self.capture()
        hit = find_prefix(self.store, self.identity, self.layout, self.tokens[:8], [4, 8])
        self.assertEqual(hit, shorter)
        self.assertIsNone(find_prefix(self.store, self.identity, self.layout, self.tokens[:4], [4, 8]))

    def test_larger_kv_match_cannot_invent_a_recurrent_frontier(self):
        self.capture()
        self.assertEqual(
            find_prefix(self.store, self.identity, self.layout, self.tokens, [4, 8, 12, 16]), self.checkpoint
        )

    def test_namespaces_model_policy_layout_and_token_ids_isolate_keys(self):
        self.capture()
        variants = [replace(self.identity, **{field: "changed"}) for field in self.identity.__dataclass_fields__]
        for identity in variants:
            with self.subTest(identity=identity):
                self.assertIsNone(find_prefix(self.store, identity, self.layout, self.tokens, [8]))
        for layout in (replace(self.layout, abi="next-abi"), replace(self.layout, tp_size=8)):
            self.assertIsNone(find_prefix(self.store, self.identity, layout, self.tokens, [8]))

    def test_bad_header_rejected_before_destination_acquisition(self):
        self.capture()
        self.corrupt(lambda raw: b"X" + raw[1:])
        target = FakeTarget()
        with self.assertRaisesRegex(ValueError, "header"):
            restore(self.store, self.checkpoint, target)
        self.assertEqual(target.events, [])

    def test_missing_rank_corrupt_bytes_and_incomplete_trailer_abort(self):
        self.capture()
        original = self.store._path(self.checkpoint.key).read_bytes()
        payload_start = len(MAGIC) + 4 + len(self.checkpoint.header)
        damages = [
            original[:payload_start] + bytes([original[payload_start] ^ 1]) + original[payload_start + 1 :],
            original[: -len(COMPLETE) - 33] + b"X" + original[-len(COMPLETE) - 32 :],
            original[: payload_start + 30],
            original[: -len(COMPLETE)],
            original[:-1],
            original + b"unexpected trailing bytes",
        ]
        for data in damages:
            with self.subTest(length=len(data)):
                self.store._path(self.checkpoint.key).write_bytes(data)
                target = FakeTarget()
                with self.assertRaises(ValueError):
                    restore(self.store, self.checkpoint, target, chunk_bytes=11)
                self.assertIsNone(target.visible)
                self.assertIsNone(target.pending)
                self.assertIn("abort_and_release", target.events)
                self.assertNotIn("publish", target.events)

    def test_partial_transfer_and_cancelled_request_never_publish(self):
        self.capture()
        for target, error in ((FakeTarget(fail_write=True), OSError), (FakeTarget(cancelled=True), ValueError)):
            with self.subTest(error=error):
                with self.assertRaises(error):
                    restore(self.store, self.checkpoint, target)
                self.assertIsNone(target.visible)
                self.assertIsNone(target.pending)
                self.assertEqual(target.inactive_slot, b"never change another request")
                self.assertIn("abort_and_release", target.events)

    def test_source_errors_never_publish_or_leave_partial_bytes(self):
        for field in ("fail_read", "fail_exit", "short_read"):
            with self.subTest(field=field):
                source = self.make_source(self.checkpoint)
                setattr(source, field, True)
                with self.assertRaises((OSError, ValueError)):
                    capture(self.store, self.checkpoint, source)
                self.assertFalse(self.store.contains(self.checkpoint.key))
                self.assertEqual(self.store.used_bytes, 0)

    def test_unconsumed_sampled_token_cannot_be_snapshotted(self):
        next_page = Checkpoint.for_tokens(self.identity, self.layout, self.tokens, 12)
        with self.assertRaisesRegex(ValueError, "consumed frontier"):
            capture(self.store, next_page, self.source)
        self.assertEqual(self.store.used_bytes, 0)

    def test_reopen_and_explicit_eviction(self):
        self.capture()
        reopened = AtomicDirectoryStore(self.store.root, max_bytes=self.store.max_bytes)
        target = FakeTarget()
        restore(reopened, self.checkpoint, target)
        self.assertEqual(target.visible, self.source.payload)
        reopened.discard(self.checkpoint.key)
        self.assertFalse(self.store.contains(self.checkpoint.key))
        with self.assertRaises(FileNotFoundError):
            restore(self.store, self.checkpoint, FakeTarget())

    def test_new_process_can_read_published_bytes(self):
        self.capture()
        script = """
import hashlib, sys
from models.demos.qwen38_27b_qb2.tt.prefix_storage import AtomicDirectoryStore
store = AtomicDirectoryStore(sys.argv[1], max_bytes=int(sys.argv[2]))
with store.read(sys.argv[3]) as stream:
    print(hashlib.sha256(stream.read()).hexdigest())
"""
        result = subprocess.check_output(
            [sys.executable, "-c", script, str(self.store.root), str(self.store.max_bytes), self.checkpoint.key],
            text=True,
            timeout=10,
        ).strip()
        expected = hashlib.sha256(self.store._path(self.checkpoint.key).read_bytes()).hexdigest()
        self.assertEqual(result, expected)

    def test_two_store_instances_cannot_overbook_the_same_quota(self):
        second = Checkpoint.for_tokens(self.identity, self.layout, self.tokens, 4)
        quota = self.checkpoint.encoded_bytes
        first_store = AtomicDirectoryStore(self.root / "shared", max_bytes=quota, create=True)
        second_store = AtomicDirectoryStore(first_store.root, max_bytes=quota)
        barrier = threading.Barrier(2)
        outcomes = []

        def publish(store, checkpoint):
            try:
                barrier.wait(timeout=2)
                capture(store, checkpoint, self.make_source(checkpoint), chunk_bytes=7)
                outcomes.append("published")
            except OSError as error:
                outcomes.append(str(error))

        threads = [
            threading.Thread(target=publish, args=(store, cp), daemon=True)
            for store, cp in ((first_store, self.checkpoint), (second_store, second))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(outcomes.count("published"), 1)
        self.assertEqual(outcomes.count("Checkpoint exceeds available storage budget"), 1)
        self.assertLessEqual(first_store.used_bytes, quota)

    def test_immutable_blob_and_explicit_quota(self):
        self.capture()
        with self.assertRaises(FileExistsError):
            self.capture()
        small = AtomicDirectoryStore(self.root / "small", max_bytes=self.checkpoint.encoded_bytes - 1, create=True)
        with self.assertRaisesRegex(OSError, "storage budget"):
            capture(small, self.checkpoint, self.source)
        self.assertEqual(small.used_bytes, 0)
        with self.assertRaisesRegex(ValueError, "quota differs"):
            AtomicDirectoryStore(self.store.root, max_bytes=1)

    def test_interrupted_process_partial_file_is_not_a_hit_but_counts_in_quota(self):
        abandoned = self.store.root / "checkpoint-interrupted.partial"
        abandoned.write_bytes(b"x" * self.store.max_bytes)
        self.assertFalse(self.store.contains(self.checkpoint.key))
        with self.assertRaisesRegex(OSError, "storage budget"):
            self.capture()
        self.assertTrue(abandoned.exists(), "Never silently delete an operator's failed-run evidence")

    def test_blob_length_cannot_escape_reserved_quota(self):
        for size, payload in ((3, b"too long"), (20, b"too short")):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    with self.store.write(self.checkpoint.key, size) as output:
                        output.write(payload)
                self.assertFalse(self.store.contains(self.checkpoint.key))
                self.assertEqual(self.store.used_bytes, 0)

    def test_eviction_waits_for_restore_lease_to_keep_quota_accounting_honest(self):
        self.capture()
        started, finished = threading.Event(), threading.Event()
        errors = []

        def evict():
            try:
                started.set()
                self.store.discard(self.checkpoint.key)
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        with self.store.read(self.checkpoint.key) as stream:
            thread = threading.Thread(target=evict, daemon=True)
            thread.start()
            self.assertTrue(started.wait(1))
            self.assertFalse(finished.wait(0.05))
            self.assertEqual(len(stream.read()), self.checkpoint.encoded_bytes)
        thread.join(2)
        self.assertTrue(finished.is_set())
        self.assertEqual(errors, [])
        self.assertEqual(self.store.used_bytes, 0)

    def test_invalid_inputs_do_not_touch_storage(self):
        for consumed in (0, -1, 3, True, 20):
            with self.subTest(consumed=consumed), self.assertRaises(ValueError):
                Checkpoint.for_tokens(self.identity, self.layout, self.tokens, consumed)
        for token in (-1, 2**32, True, 1.5):
            with self.subTest(token=token), self.assertRaises(ValueError):
                Checkpoint.for_tokens(self.identity, self.layout, [token] * 8, 8)
        for chunk in (0, -1, True, 2**20 + 1):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                capture(self.store, self.checkpoint, self.source, chunk_bytes=chunk)
        for key in ("../escape", "A" * 64, "a" * 63):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.store.contains(key)
        self.assertEqual(self.store.used_bytes, 0)

    def test_qwen_layout_accounts_for_every_layer_and_rank(self):
        with self.assertRaisesRegex(ValueError, "all 64 Qwen layers"):
            Layout.qwen_tp4_bfp8(["linear_attention"] * 3 + ["full_attention"])

    def test_lookup_hashes_history_once_and_preserves_checkpoint_keys(self):
        tokens = list(range(257))
        expected = Checkpoint.for_tokens(self.identity, self.layout, tokens, 128)
        visited, checked = [], []

        class History:
            def __len__(self):
                return len(tokens)

            def __getitem__(self, index):
                visited.append(index)
                return tokens[index]

        class Store:
            def contains(self, key):
                checked.append(key)
                return key == expected.key

        found = find_prefix(Store(), self.identity, self.layout, History(), range(32, 257, 32))
        self.assertEqual(found, expected)
        self.assertEqual(visited, list(range(256)))
        self.assertEqual(
            checked,
            [Checkpoint.for_tokens(self.identity, self.layout, tokens, n).key for n in (256, 224, 192, 160, 128)],
        )

    def test_full_qwen_size_matches_published_capacity_scope(self):
        layout = Layout.qwen_tp4_bfp8((["linear_attention"] * 3 + ["full_attention"]) * 16)
        checkpoint = Checkpoint.for_tokens(self.identity, layout, [42] * 32768, 32768)
        self.assertEqual(len(list(checkpoint.segments())), 512)
        kv = sum(s.size for s in checkpoint.segments() if s.kind in ("key", "value"))
        state = checkpoint.payload_bytes - kv
        self.assertEqual(kv, 1140850688)
        self.assertEqual(state, 153944064)
        self.assertLess(checkpoint.encoded_bytes - checkpoint.payload_bytes, 65536)


if __name__ == "__main__":
    unittest.main()
