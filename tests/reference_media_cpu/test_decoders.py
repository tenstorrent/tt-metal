# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU integration tests with real PyAV; no tt-metal runtime or device fixtures."""

import importlib
import struct
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import torch

# Import the real pure-CPU modules without pipelines/__init__.py (which imports ttnn).
PACKAGE = "_reference_media_cpu"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[2] / "models/tt_dit/pipelines/minimax_h3")]
sys.modules[PACKAGE] = package
decoders = importlib.import_module(f"{PACKAGE}.packing_ref2va")


def write_media(path, *, container_format="mov", video_codec="mpeg4", audio_codec="pcm_s16le"):
    with av.open(str(path), "w", format=container_format) as container:
        video = container.add_stream(video_codec, rate=2)
        video.width, video.height, video.pix_fmt = 32, 32, "yuv420p"
        video.codec_context.thread_count = 1
        audio = container.add_stream(audio_codec, rate=8000)
        audio.layout = "mono"
        for index in range(2):
            pixels = np.full((32, 32, 3), 40 + 80 * index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        samples = np.full((1, 8000), 1234, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate, frame.pts = 8000, 0
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)


def atom(kind, payload):
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def external_mov(source, target, *, absolute=False):
    """Replace both tracks' drefs with genuine QuickTime aliases and erase embedded samples.

    The canary remains an ordinary MOV: its sample offsets match the external
    movie because mdat precedes moov. Opt-in decoding must recover its media.
    Layout follows FFmpeg n8.0.1 libavformat/mov.c:mov_read_dref.
    """
    filename = target.name.encode()
    pathname = str(target).encode()
    alias = b"\0" * 10 + b"\x03vol" + b"\0" * 24 + b"\0" * 12
    alias += bytes([len(filename)]) + filename.ljust(63, b"\0") + b"\0" * 16
    alias += struct.pack(">HH", 0 if absolute else 1, 0 if absolute else 1) + b"\0" * 16
    alias += struct.pack(">HH", 2, len(pathname)) + pathname + b"\0" * (len(pathname) % 2)
    alias += struct.pack(">hH", -1, 0)
    replacement = atom(b"dref", b"\0" * 4 + struct.pack(">I", 1) + atom(b"alis", b"\0" * 4 + alias))
    count = 0

    def rewrite(data):
        nonlocal count
        result = b""
        offset = 0
        while offset < len(data):
            size, kind = struct.unpack_from(">I4s", data, offset)
            assert 8 <= size <= len(data) - offset
            payload = data[offset + 8 : offset + size]
            if kind == b"dref":
                result += replacement
                count += 1
            else:
                if kind in (b"moov", b"trak", b"mdia", b"minf", b"dinf"):
                    payload = rewrite(payload)
                elif kind == b"mdat":
                    payload = bytes(len(payload))
                result += atom(kind, payload)
            offset += size
        return result

    original = source.read_bytes()
    assert original.index(b"mdat") < original.index(b"moov")
    result = rewrite(original)
    assert count == 2
    return result


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reference-media-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.canary = self.root / "canary.mov"
        write_media(self.canary)

    def assert_media(self, path):
        frames, rate, soundtrack = decoders.decode_reference_video(path)
        self.assertEqual(frames.shape, (2, 32, 32, 3))
        self.assertEqual(frames.dtype, np.uint8)
        self.assertEqual(rate, 2.0)
        np.testing.assert_allclose(frames[0], 39, atol=2)
        np.testing.assert_allclose(frames[1], 119, atol=2)
        audio, sample_rate = decoders.decode_reference_audio(path)
        self.assertEqual(sample_rate, 8000)
        self.assertEqual(audio.shape, (1, 8000))
        torch.testing.assert_close(audio, torch.full((1, 8000), 1234 / 32768), rtol=0, atol=0)
        self.assertEqual(soundtrack[1], sample_rate)
        torch.testing.assert_close(soundtrack[0], audio, rtol=0, atol=0)

    def test_embedded_mov_and_matroska(self):
        self.assert_media(self.canary)
        matroska = self.root / "embedded.mkv"
        write_media(matroska, container_format="matroska")
        self.assert_media(matroska)

    def test_embedded_mp4_with_aac_soundtrack(self):
        mp4 = self.root / "embedded.mp4"
        write_media(mp4, container_format="mp4", video_codec="libx264", audio_codec="aac")
        frames, rate, soundtrack = decoders.decode_reference_video(mp4)
        self.assertEqual(frames.shape, (2, 32, 32, 3))
        self.assertEqual(rate, 2.0)
        np.testing.assert_allclose(frames[1], 119, atol=2)
        audio, sample_rate = decoders.decode_reference_audio(mp4)
        self.assertEqual(sample_rate, 8000)
        self.assertGreaterEqual(audio.shape[1], 8000)
        # Preserve the ordinary decoder's exact samples, including AAC padding.
        with av.open(str(mp4)) as container:
            expected, expected_rate = decoders._decode_soundtrack(av, container, container.streams.audio[0])
        self.assertEqual(sample_rate, expected_rate)
        torch.testing.assert_close(audio, expected, rtol=0, atol=0)
        # Match the existing two-pass video path separately: AAC seek priming
        # can differ from decoding audio in a newly opened container.
        with av.open(str(mp4)) as container:
            list(container.decode(video=0))
            container.seek(0)
            expected_soundtrack, expected_rate = decoders._decode_soundtrack(av, container, container.streams.audio[0])
        self.assertEqual(soundtrack[1], expected_rate)
        torch.testing.assert_close(soundtrack[0], expected_soundtrack, rtol=0, atol=0)

    def test_embedded_audio_only(self):
        wav = self.root / "embedded.wav"
        with wave.open(str(wav), "wb") as output:
            output.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            output.writeframes(np.full(800, 4321, dtype="<i2").tobytes())
        audio, rate = decoders.decode_reference_audio(wav)
        self.assertEqual(rate, 8000)
        torch.testing.assert_close(audio, torch.full((1, 800), 4321 / 32768), rtol=0, atol=0)

    def test_concat_local_reference_is_rejected(self):
        # Safe relative filename: concat's default safe=1 does NOT prevent this read.
        playlist = self.root / "upload.mov"
        playlist.write_text("ffconcat version 1.0\nfile canary.mov\n")
        # Positive controls: the fixture works, even with a file-only whitelist.
        for options in ({}, {"protocol_whitelist": "file"}):
            with av.open(str(playlist), options=options) as container:
                frames = list(container.decode(video=0))
                self.assertEqual(len(frames), 2)
                np.testing.assert_allclose(frames[1].to_ndarray(format="rgb24"), 119, atol=2)
            with av.open(str(playlist), options=options) as container:
                audio, rate = decoders._decode_soundtrack(av, container, container.streams.audio[0])
                self.assertEqual(rate, 8000)
                torch.testing.assert_close(audio, torch.full((1, 8000), 1234 / 32768), rtol=0, atol=0)
        for decode in (decoders.decode_reference_video, decoders.decode_reference_audio):
            with self.subTest(decoder=decode.__name__):
                with self.assertRaises((ValueError, av.FFmpegError)):
                    decode(playlist)

    def test_root_protocols_are_not_interpreted(self):
        for decode in (decoders.decode_reference_video, decoders.decode_reference_audio):
            for path in (f"file:{self.canary}", f"concat:{self.canary}|{self.canary}"):
                with self.subTest(decoder=decode.__name__, path=path):
                    with self.assertRaises(OSError):
                        decode(path)

    def test_hls_local_reference_is_rejected(self):
        segment = self.root / "canary.ts"
        write_media(segment, container_format="mpegts", video_codec="libx264", audio_codec="aac")
        playlist = self.root / "upload.m3u8"
        playlist.write_text(
            "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n"
            "#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:1.0,\ncanary.ts\n#EXT-X-ENDLIST\n"
        )
        with av.open(str(playlist)) as container:
            self.assertEqual(len(list(container.decode(video=0))), 2)
        with av.open(str(playlist)) as container:
            audio, rate = decoders._decode_soundtrack(av, container, container.streams.audio[0])
            self.assertEqual(rate, 8000)
            self.assertGreater(audio.shape[1], 0)
        for decode in (decoders.decode_reference_video, decoders.decode_reference_audio):
            with self.subTest(decoder=decode.__name__):
                with self.assertRaisesRegex(ValueError, "self-contained"):
                    decode(playlist)

    def test_mov_alias_defaults_and_positive_controls(self):
        real_open = av.open
        for absolute in (False, True):
            external = self.root / "alias.mov"
            external.write_bytes(external_mov(self.canary, self.canary, absolute=absolute))
            # Prove this is a working external reference, not just malformed input.
            options = {"enable_drefs": "1", "use_absolute_path": str(int(absolute))}
            with real_open(str(external), options=options) as container:
                frames = list(container.decode(video=0))
                self.assertEqual(len(frames), 2)
                np.testing.assert_allclose(frames[1].to_ndarray(format="rgb24"), 119, atol=2)
            with real_open(str(external), options=options) as container:
                audio, rate = decoders._decode_soundtrack(av, container, container.streams.audio[0])
                self.assertEqual(rate, 8000)
                torch.testing.assert_close(audio, torch.full((1, 8000), 1234 / 32768), rtol=0, atol=0)

            for decode in (decoders.decode_reference_video, decoders.decode_reference_audio):
                opened = []

                def record_open(url, flags, options):
                    if url == str(external):
                        return open(external, "rb")
                    opened.append(url)
                    raise AssertionError("MOV defaults must not attempt an external open")

                def observe(*args, **kwargs):
                    kwargs["io_open"] = record_open
                    return real_open(*args, **kwargs)

                with self.subTest(absolute=absolute, decoder=decode.__name__):
                    with patch.object(av, "open", observe):
                        with self.assertRaises((ValueError, RuntimeError, av.FFmpegError)):
                            decode(external)
                    self.assertEqual(opened, [])


if __name__ == "__main__":
    unittest.main()
