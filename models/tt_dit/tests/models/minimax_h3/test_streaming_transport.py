# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Transport protocol fault injection; not generated-media qualification."""

import hashlib
import importlib
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[3] / "pipelines/minimax_h3"
PACKAGE = "_h3_streaming_transport_test"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
t = importlib.import_module(PACKAGE + ".streaming_transport")
e = importlib.import_module(PACKAGE + ".streaming_encode")


def encoded(index=0):
    start = index * 22
    audio = start * 32000 // 24
    data = bytes([index + 1]) * 100
    return e.EncodedChunk(
        index, data, hashlib.sha256(data).hexdigest(), start, audio, 22, (start + 22) * 32000 // 24 - audio, 0.1
    )


def chunk(index=0):
    item = encoded(index)
    return SimpleNamespace(
        index=index,
        video_pts=item.video_pts,
        audio_pts=item.audio_pts,
        media=SimpleNamespace(frames=item.frames, samples=item.samples),
    )


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.accepted = []
        self.encoded = []
        self.clock = 10.0
        self.client = t.ReadyClient("session", self.accept, max_bytes=100)

    def accept(self, item):
        self.accepted.append(item.index)
        return item.video_pts  # Deliberately leave one chunk unplayed.

    def encode(self, item, *, cancelled):
        if cancelled():
            raise RuntimeError("cancelled encoder")
        self.encoded.append(item.index)
        return encoded(item.index)

    def sink(self, exchange=None, **kw):
        opts = dict(max_bytes=100, max_attempts=3, timeout_s=5, clock=lambda: self.clock)
        opts.update(kw)
        return t.TransportSink(
            "session", self.encode, exchange or (lambda delivery, deadline: self.client(delivery)), **opts
        )

    def test_lost_ack_resends_same_bytes_without_reencoding_or_reappend(self):
        sent = []

        def exchange(delivery, deadline):
            self.assertEqual(deadline, 15.0)
            sent.append(delivery)
            reply = self.client(delivery)
            if len(sent) == 1:
                raise TimeoutError("ack lost after client commit")
            return reply

        sink = self.sink(exchange)
        reply = sink(chunk())
        self.assertIs(sent[0], sent[1])
        self.assertEqual(self.accepted, [0])
        self.assertEqual(self.encoded, [0])
        self.assertEqual(
            (reply.index, reply.playable_frames, reply.playable_samples, reply.played_frames), (0, 22, 29333, 0)
        )
        self.assertEqual(sink.pending_bytes, 0)
        with self.assertRaises(FrozenInstanceError):
            sent[0].session_id = "changed"

    def test_duplicate_ack_cannot_advance_next_chunk(self):
        replies = []

        def exchange(delivery, deadline):
            if replies:
                return replies[0]
            replies.append(self.client(delivery))
            return replies[0]

        sink = self.sink(exchange)
        sink(chunk())
        with self.assertRaises(ValueError):
            sink(chunk(1))
        self.assertEqual(sink.index, 1)
        self.assertTrue(sink.closed)
        self.assertEqual(self.accepted, [0])

    def test_late_ack_and_clock_regression_are_not_success(self):
        for now in (15.0, 9.0, float("nan")):
            with self.subTest(now=now):
                self.setUp()

                def exchange(delivery, deadline):
                    reply = self.client(delivery)
                    self.clock = now
                    return reply

                sink = self.sink(exchange)
                with self.assertRaises((ValueError, TimeoutError)):
                    sink(chunk())
                self.assertEqual(sink.index, 0)
                self.assertTrue(sink.closed)
                self.assertEqual(sink.pending_bytes, 0)

    def test_retry_exhaustion_is_bounded_and_terminal(self):
        calls = []

        def exchange(delivery, deadline):
            calls.append(delivery.key)
            raise ConnectionError("connection lost")

        sink = self.sink(exchange)
        with self.assertRaises(TimeoutError):
            sink(chunk())
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(set(calls)), 1)
        with self.assertRaises(RuntimeError):
            sink(chunk())

    def test_deadline_overflow_fails_before_encode_or_delivery(self):
        sink = self.sink(clock=lambda: 1e308, timeout_s=1e308)
        with self.assertRaises(ValueError):
            sink(chunk())
        self.assertTrue(sink.closed)
        self.assertEqual(self.encoded, [])
        self.assertEqual(self.accepted, [])

    def test_abort_during_exchange_or_encoding_and_interrupt_preserve_failure(self):
        for failure in (RuntimeError("code 17"), KeyboardInterrupt()):
            self.setUp()

            def exchange(delivery, deadline):
                self.client(delivery)
                raise failure

            sink = self.sink(exchange)
            with self.assertRaises(type(failure)) as ctx:
                sink(chunk())
            self.assertIs(ctx.exception, failure)
            self.assertTrue(sink.closed)
            self.assertEqual(sink.pending_bytes, 0)
        self.setUp()
        sink = self.sink()

        def aborted(delivery, deadline):
            reply = self.client(delivery)
            sink.abort()
            return reply

        sink.exchange = aborted
        with self.assertRaises(RuntimeError):
            sink(chunk())
        self.assertEqual(sink.index, 0)
        sink = self.sink(cancelled=lambda: True)
        with self.assertRaises(RuntimeError):
            sink(chunk())
        self.assertEqual(sink.pending_bytes, 0)

    def test_partial_client_failure_never_reappends(self):
        def partial(item):
            self.accepted.append(item.index)
            raise ConnectionError("decoder failed after partial append")

        self.client = t.ReadyClient("session", partial, max_bytes=100)
        sink = self.sink()
        with self.assertRaises(RuntimeError):
            sink(chunk())
        self.assertEqual(self.accepted, [0])
        self.assertTrue(self.client.closed)
        self.assertTrue(sink.closed)

    def test_wrong_session_digest_metadata_and_mutable_payload_rejected(self):
        good = encoded()
        bad = [
            replace(good, data=b"z" * 100),
            replace(good, data=bytearray(good.data)),
            replace(good, frames=True),
            replace(good, audio_pts=1),
            replace(good, samples=1),
            replace(good, index=-1),
        ]
        for item in bad:
            self.setUp()
            with self.assertRaises(ValueError):
                self.client(t.Delivery("session", item))
            self.assertEqual(self.accepted, [])
        self.setUp()
        with self.assertRaises(ValueError):
            self.client(t.Delivery("other", good))

    def test_byte_cap_and_sequence_gap_rejected_before_client_append(self):
        for bound in (1, 50, 99):
            self.setUp()
            sink = self.sink(max_bytes=bound)
            with self.assertRaises(ValueError):
                sink(chunk())
            self.assertEqual(self.accepted, [])
            self.assertEqual(sink.pending_bytes, 0)
        self.setUp()
        with self.assertRaises(ValueError):
            self.client(t.Delivery("session", encoded(1)))
        self.assertEqual(self.accepted, [])

    def test_playhead_is_explicit_monotonic_and_not_past_ready_media(self):
        sink = self.sink()
        self.assertEqual(sink(chunk()).played_frames, 0)
        self.assertEqual(sink(chunk(1)).played_frames, 22)
        self.client.accept = lambda item: 21
        with self.assertRaises(ValueError):
            sink(chunk(2))
        for bad in (True, -1, 23, float("nan"), 0.0):
            self.setUp()
            self.client.accept = lambda item: bad
            with self.assertRaises(ValueError):
                self.sink()(chunk())

    def test_unready_or_mismatched_receipt_is_never_acknowledged(self):
        for mutate in (
            lambda r: replace(r, client_ready=False),
            lambda r: replace(r, client_ready=1),
            lambda r: replace(r, key=("forged",)),
            lambda r: None,
        ):
            self.setUp()
            sink = self.sink(lambda d, deadline: mutate(self.client(d)))
            with self.assertRaises(ValueError):
                sink(chunk())
            self.assertEqual(sink.index, 0)

    def test_dedup_cache_keeps_only_last_receipt_and_no_payload(self):
        for index in range(3):
            delivery = t.Delivery("session", encoded(index))
            reply = self.client(delivery)
            self.assertIs(self.client(delivery), reply)
        self.assertEqual(self.accepted, [0, 1, 2])
        self.assertFalse(any(isinstance(v, (bytes, e.EncodedChunk, t.Delivery)) for v in vars(self.client).values()))
        with self.assertRaises(ValueError):
            self.client(t.Delivery("session", encoded(0)))

    def test_invalid_limits_and_reentrance(self):
        for key in ("max_bytes", "max_attempts", "timeout_s"):
            for value in (True, 0, -1, float("nan")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.sink(**{key: value})
        sink = self.sink()
        sink.exchange = lambda delivery, deadline: sink(chunk())
        with self.assertRaises(RuntimeError):
            sink(chunk())
        self.assertTrue(sink.closed)


if __name__ == "__main__":
    unittest.main()
