# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real CPU codec regression tests; synthetic inputs never qualify streaming."""

import importlib
import io
import sys
import unittest
import weakref
from contextlib import redirect_stderr
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from unittest import mock

import av
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3] / "pipelines/minimax_h3"
PACKAGE = "_h3_streaming_encode_test"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
e = importlib.import_module(PACKAGE + ".streaming_encode")
s = importlib.import_module(PACKAGE + ".streaming")
b = importlib.import_module(PACKAGE + ".streaming_pipeline")


class EncodeTests(unittest.TestCase):
    def setUp(self):
        self.identity = s.Identity("h3-r2v-high", "a" * 40, "cpu-test", "b" * 64, "c" * 64)
        self.limits = dict(max_frames=22, max_width=32, max_height=16, max_samples=29334, max_bytes=500000)

    def chunk(self, index=0, start=0, frames=22):
        audio_start = start * 32000 // 24
        samples = (start + frames) * 32000 // 24 - audio_start
        # RGB values exactly on the declared 8-bit lattice; PCM includes sign.
        video = (torch.arange(frames * 3 * 16 * 32) % 256).reshape(frames, 3, 16, 32).float() / 255
        audio = torch.linspace(-0.8, 0.8, samples * 2).reshape(2, samples)
        return s.Chunk(
            index,
            b.tensor_media(video, audio),
            start,
            audio_start,
            Fraction(1, 24),
            Fraction(1, 32000),
            0,
            self.identity,
        )

    def encoder(self, **kwargs):
        limits = self.limits | kwargs
        return e.NutEncoder(pixel_format="rgb8", **limits)

    def decode(self, encoded, kind):
        with av.open(io.BytesIO(encoded.data), mode="r") as container:
            return list(container.decode(**{kind: 0}))

    def test_transport_real_decode_lost_ack_and_session_playhead(self):
        t = importlib.import_module(PACKAGE + ".streaming_transport")
        decoded, rows, attempts = [], [], []

        def accept(encoded):
            video, audio = self.decode(encoded, "video"), self.decode(encoded, "audio")
            self.assertEqual(len(video), encoded.frames)
            self.assertEqual(sum(frame.samples for frame in audio), encoded.samples)
            self.assertEqual(video[0].pts * video[0].time_base, Fraction(encoded.video_pts, 24))
            decoded.append(encoded.index)
            return encoded.video_pts

        cfg = s.StreamConfig(self.identity, raw_frames=22, context_frames=2, max_buffer_s=1)
        client = t.ReadyClient(cfg.session_id, accept, max_bytes=500000)

        def exchange(delivery, deadline):
            attempts.append(delivery)
            receipt = client(delivery)
            if len(attempts) == 1:
                raise ConnectionError("ack lost after real decode")
            return receipt

        sink = t.TransportSink(cfg.session_id, self.encoder(), exchange, max_bytes=500000)
        session = s.Session(cfg, ["synthetic"], lambda r: self.chunk(r.index, r.index * 22).media, sink, rows.append)
        session.step("synthetic test", 0)
        with self.assertRaises(s.Backpressure):
            session.step("synthetic test", 1)
        session.update_playout(22)
        session.step("synthetic test", 1)
        self.assertEqual(decoded, [0, 1])
        self.assertIs(attempts[0], attempts[1])
        self.assertEqual(session.frames, 44)
        self.assertEqual(session.played_frames, 22)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["client_ready_s"] >= row["render_done_s"] for row in rows))
        self.assertTrue(all(row["measurement_kind"] == "control_plane_incomplete" for row in rows))
        self.assertEqual(sink.pending_bytes, 0)

    def test_real_mux_roundtrip_preserves_frame_sample_counts_and_global_pts(self):
        for index, start in enumerate((0, 22, 44)):
            chunk = self.chunk(index=index, start=start)
            encoded = self.encoder()(chunk)
            video, audio = self.decode(encoded, "video"), self.decode(encoded, "audio")
            self.assertEqual(len(video), 22)
            self.assertEqual(sum(frame.samples for frame in audio), chunk.media.samples)
            self.assertEqual(video[0].pts * video[0].time_base, Fraction(start, 24))
            self.assertEqual(audio[0].pts * audio[0].time_base, Fraction(chunk.audio_pts, 32000))
            for i, frame in enumerate(video):
                self.assertEqual(frame.pts * frame.time_base, Fraction(start + i, 24))
            rgb = np.stack([frame.to_ndarray(format="rgb24") for frame in video])
            np.testing.assert_array_equal(rgb, (chunk.media.video.movedim(1, -1).numpy() * 255).round().astype("uint8"))
            pcm = np.concatenate([frame.to_ndarray().reshape(-1, 2).T for frame in audio], axis=1)
            np.testing.assert_array_equal(pcm, chunk.media.audio.numpy())
            self.assertLessEqual(len(encoded.data), self.limits["max_bytes"])
            self.assertEqual(encoded.index, index)

    def test_size_cap_fails_without_returning_partial_container(self):
        for cap in (64, 4000, 60000):
            with self.subTest(cap=cap), self.assertRaises(e.EncodingLimit):
                self.encoder(max_bytes=cap)(self.chunk())

    def test_rejects_invalid_limits_formats_shapes_and_pts(self):
        for key in self.limits:
            for invalid in (0, -1, True, 1.5):
                with self.subTest(key=key, value=invalid), self.assertRaises(ValueError):
                    self.encoder(**{key: invalid})
        with self.assertRaises(ValueError):
            e.NutEncoder(pixel_format="automatic", **self.limits)
        for bound in (dict(max_frames=21), dict(max_width=31), dict(max_height=15), dict(max_samples=29332)):
            with self.subTest(bound=bound), self.assertRaises(e.EncodingLimit):
                self.encoder(**bound)(self.chunk())
        from dataclasses import replace

        for changes in (
            dict(video_pts=-1),
            dict(audio_pts=1),
            dict(video_pts=True),
            dict(video_time_base=Fraction(1, 30)),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.encoder()(replace(self.chunk(), **changes))

    def test_rejects_nonfinite_and_out_of_range_pixels_without_clipping(self):
        for value in (float("nan"), float("inf"), -0.01, 1.01):
            chunk = self.chunk()
            chunk.media.video[0, 0, 0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.encoder()(chunk)
        chunk = self.chunk()
        chunk.media.audio[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.encoder()(chunk)

    def test_cancellation_aborts_and_next_independent_encode_works(self):
        encoder = self.encoder()
        for at in (1, 3, 10, 20):
            calls = 0

            def cancelled():
                nonlocal calls
                calls += 1
                return calls >= at

            with self.subTest(at=at), self.assertRaises(e.EncodingAborted):
                encoder(self.chunk(), cancelled=cancelled)
        self.assertTrue(encoder(self.chunk()).data)

    def test_bounded_buffer_seek_overwrite_and_interrupt(self):
        buffer = e._BoundedBytes(8)
        buffer.write(b"12345678")
        buffer.seek(0)
        buffer.write(b"ab")
        self.assertEqual(buffer.getvalue(), b"ab345678")
        for offset, whence in ((9, 0), (9, 1), (1, 2)):
            with self.subTest(offset=offset, whence=whence), self.assertRaises(e.EncodingLimit):
                buffer.seek(offset, whence)
        with self.assertRaises(e.EncodingLimit):
            buffer.write(b"x" * 9)

        for at in (1, 10, 20):
            calls = 0

            def interrupted():
                nonlocal calls
                calls += 1
                if calls >= at:
                    raise KeyboardInterrupt
                return False

            stderr = io.StringIO()
            with self.subTest(at=at), redirect_stderr(stderr):
                with self.assertRaises(KeyboardInterrupt):
                    self.encoder()(self.chunk(), cancelled=interrupted)
                self.assertNotIn("Exception ignored", stderr.getvalue())

    def test_deadline_and_codec_failure_never_return_success(self):
        for value in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.encoder(timeout_s=value)
        with mock.patch.object(e.time, "monotonic", side_effect=(0, 1)):
            with self.assertRaises(e.EncodingAborted):
                self.encoder(timeout_s=0.5)(self.chunk())
        with mock.patch.object(av, "open", side_effect=RuntimeError("codec failed")):
            with self.assertRaisesRegex(RuntimeError, "codec failed"):
                self.encoder()(self.chunk())
        mux = mock.Mock()
        mux.close.side_effect = RuntimeError("close failed")
        with mock.patch.object(av, "open", return_value=mux):
            with self.assertRaisesRegex(RuntimeError, "close failed"):
                with e._mux(io.BytesIO()):
                    pass

    def test_encoder_does_not_retain_input_tensors(self):
        encoder = self.encoder()
        chunk = self.chunk()
        video_ref, audio_ref = weakref.ref(chunk.media.video), weakref.ref(chunk.media.audio)
        encoded = encoder(chunk)
        del chunk
        self.assertIsNone(video_ref())
        self.assertIsNone(audio_ref())
        self.assertTrue(encoded.data)


if __name__ == "__main__":
    unittest.main()
