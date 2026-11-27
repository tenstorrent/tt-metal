# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free executable contract tests; run this file with Python/unittest."""

import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path

MODULE = Path(__file__).resolve().parents[3] / "pipelines/minimax_h3/streaming.py"
spec = importlib.util.spec_from_file_location("h3_streaming_control", MODULE)
s = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s
spec.loader.exec_module(s)


def media(video=None, audio=None, overlap=0):
    video = tuple(range(124)) if video is None else tuple(video)
    audio = range((len(video) * 32000 + 23) // 24) if audio is None else audio

    def sliced(a, b, c, d):
        return media(video[a:b], audio[c:d])

    def repeated(previous, frames, samples):
        return video[:frames] == previous.video[-frames:] and audio[:samples] == previous.audio[-samples:]

    return s.Media(video, audio, len(video), len(audio), sliced, repeated, overlap)


class Clock:
    now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.cfg = s.StreamConfig(s.Identity("h3-r2v-high", "a" * 40, "build-1", "b" * 64, "c" * 64))
        self.rows, self.requests, self.chunks = [], [], []

    def session(self, render=None, sink=None, receipt=None):
        def default_render(request):
            self.requests.append((request.index, request.previous.frames if request.previous else 0))
            self.clock.advance(2)
            return media()

        def default_sink(chunk):
            self.chunks.append((chunk.video_pts, chunk.audio_pts, chunk.media.frames, chunk.media.samples))
            self.clock.advance(1)
            played = min(chunk.video_pts, max(0, int((self.clock.now - 3) * 24)))
            return s.Acknowledgement(chunk.index, chunk.media.frames, chunk.media.samples, played)

        return s.Session(
            self.cfg,
            ("fixed-reference",),
            render or default_render,
            sink or default_sink,
            receipt or self.rows.append,
            clock=self.clock,
        )

    def test_100_chunk_rational_pts_and_bounded_context(self):
        stream = self.session()
        for i in range(100):
            try:
                row = stream.step("prompt", i)
            except s.Backpressure as wait:
                self.clock.advance(wait.retry_after_s)
                stream.update_playout(min(stream.frames, max(0, int((self.clock.now - 3) * 24 + 1e-7))))
                row = stream.step("prompt", i)
            self.assertEqual(row["retained_context_frames"], 24)
            self.assertFalse(row["quality_validated"])
            self.assertEqual(row["measurement_kind"], "control_plane_incomplete")
            self.assertEqual(row["session_id"], self.cfg.session_id)
            self.assertAlmostEqual(row["media_end_s"] - row["media_start_s"], 124 / 24)
            self.assertEqual(row["audio_pts"] + row["new_samples"], (i + 1) * 124 * 32000 // 24)
            self.assertLessEqual(row["buffer_s"], self.cfg.max_buffer_s)
            self.assertLessEqual(row["client_unplayed_frames"] / 24, self.cfg.max_buffer_s)
        self.assertEqual(stream.frames, 12400)
        self.assertEqual(stream.previous.frames, 24)
        self.assertEqual(stream.previous.samples, 32000)
        self.assertEqual(self.requests[0], (0, 0))
        self.assertEqual(self.requests[-1], (99, 24))
        self.assertAlmostEqual(row["service_rtf"], 3 * 24 / 124)
        self.assertEqual(row["underruns"], 0)

    def test_backpressure_does_not_render_or_poison_session(self):
        stream = self.session()
        for i in range(2):
            stream.step("prompt", i)
        with self.assertRaises(s.Backpressure) as caught:
            stream.step("prompt", 2)
        self.assertEqual(len(self.requests), 2)
        self.assertFalse(stream.closed)
        self.clock.advance(caught.exception.retry_after_s)
        with self.assertRaises(s.Backpressure):
            stream.step("prompt", 2)  # time alone is not a client consumption receipt
        stream.update_playout(min(stream.frames, int((self.clock.now - 3) * 24 + 1e-7)))
        stream.step("prompt", 2)

    def test_playhead_cannot_regress_or_invent_playable_frames(self):
        stream = self.session()
        stream.step("prompt", 0)
        stream.update_playout(10)
        for frames in (9, 125):
            with self.assertRaises(ValueError):
                stream.update_playout(frames)

    def test_sink_latency_and_pacing_are_included(self):
        stream = self.session()
        first = stream.step("prompt", 0)
        self.clock.advance(10)
        row = stream.step("prompt", 1)
        self.assertEqual(first["startup_s"], 3)
        self.assertEqual(row["client_interval_s"], 13)
        self.assertEqual(row["underruns"], 1)
        self.assertAlmostEqual(row["elapsed_rtf"], 16 * 24 / 248)
        self.assertAlmostEqual(row["service_rtf"], 6 * 24 / 248)

    def test_no_render_until_ack_and_reentry_is_refused(self):
        stream = None

        def sink(chunk):
            self.assertEqual(stream.index, chunk.index)
            with self.assertRaises(RuntimeError):
                stream.step("reentry", 4)
            return s.Acknowledgement(chunk.index, chunk.media.frames, chunk.media.samples)

        stream = self.session(sink=sink)
        stream.step("prompt", 0)
        stream.step("prompt", 1)
        self.assertEqual(len(self.requests), 2)

    def test_wrong_ack_poison_closes_chain(self):
        stream = self.session(sink=lambda chunk: s.Acknowledgement(99, 124, 0))
        with self.assertRaisesRegex(ValueError, "acknowledge"):
            stream.step("prompt", 0)
        self.assertIsNone(stream.previous)
        with self.assertRaises(RuntimeError):
            stream.step("prompt", 0)

    def test_sink_failure_releases_context(self):
        stream = self.session()
        stream.step("prompt", 0)

        def failure(chunk):
            raise OSError("transport failed after partial write")

        stream.sink = failure
        with self.assertRaises(OSError):
            stream.step("prompt", 1)
        self.assertTrue(stream.closed)
        self.assertIsNone(stream.previous)

    def test_unverified_overlap_is_rejected(self):
        stream = self.session()
        stream.step("prompt", 0)
        stream.render = lambda request: media(overlap=24)
        with self.assertRaisesRegex(ValueError, "unverified overlap"):
            stream.step("prompt", 1)

    def test_exact_video_audio_overlap_is_removed(self):
        stream = self.session()
        stream.step("prompt", 0)
        previous = stream.previous

        def repeated(request):
            return media(
                previous.video + tuple(range(100)),
                range(previous.audio.start, previous.audio.start + 165334),
                overlap=24,
            )

        stream.render = repeated
        row = stream.step("prompt", 1)
        self.assertEqual(row["new_frames"], 100)
        self.assertEqual(row["total_playable_s"], 224 / 24)
        self.assertEqual(row["trimmed_frames"], 24)

    def test_first_chunk_cannot_claim_overlap(self):
        stream = self.session(render=lambda request: media(overlap=1))
        with self.assertRaisesRegex(ValueError, "unverified overlap"):
            stream.step("prompt", 0)

    def test_short_audio_and_wrong_frames_are_rejected(self):
        for raw in (media(audio=range(10)), media(video=range(123))):
            with self.subTest(frames=raw.frames, samples=raw.samples):
                stream = self.session(render=lambda request: raw)
                with self.assertRaises(ValueError):
                    stream.step("prompt", 0)
                self.assertFalse(self.chunks)

    def test_clock_regression_and_nonfinite_are_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.clock = Clock()

                def render(request):
                    self.clock.now = value
                    return media()

                stream = self.session(render=render)
                with self.assertRaisesRegex(ValueError, "clock"):
                    stream.step("prompt", 0)

    def test_variant_identity_cannot_silently_change(self):
        with self.assertRaisesRegex(ValueError, "adapter"):
            replace(self.cfg.identity, variant="h3-r2v-turbo4")
        with self.assertRaisesRegex(ValueError, "base route"):
            replace(self.cfg.identity, adapter_sha256="d" * 64)
        for variant in ("h3-r2v-turbo4", "h3-r2v-hyperflow8"):
            identity = replace(self.cfg.identity, variant=variant, adapter_sha256="d" * 64)
            self.assertEqual(identity.variant, variant)

    def test_configuration_rejects_nonfinite_buffer_and_invalid_lengths(self):
        for kwargs in (
            {"raw_frames": 123},
            {"context_frames": 124},
            {"initial_buffer_s": float("nan")},
            {"fps": 25},
            {"sample_rate": 44100},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.cfg, **kwargs)

    def test_receipt_failure_cannot_continue_unrecorded_chain(self):
        def failure(row):
            raise OSError("receipt unavailable")

        stream = self.session(receipt=failure)
        with self.assertRaises(OSError):
            stream.step("prompt", 0)
        self.assertTrue(stream.closed)

    def test_reference_slot_and_close(self):
        with self.assertRaises(ValueError):
            s.Session(self.cfg, tuple(range(12)), None, None, None)
        stream = self.session()
        stream.step("prompt", 0)
        stream.close()
        self.assertIsNone(stream.previous)


if __name__ == "__main__":
    unittest.main()
