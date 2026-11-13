# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU tensor contracts; no mesh, model weights, or generated-video claims."""

import gc
import importlib
import sys
import unittest
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3] / "pipelines/minimax_h3"
PACKAGE = "_h3_streaming_tensor_test"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
b = importlib.import_module(PACKAGE + ".streaming_pipeline")
s = importlib.import_module(PACKAGE + ".streaming")
p = importlib.import_module(PACKAGE + ".packing_ref2va")
r = importlib.import_module(PACKAGE + ".references")


class TensorMediaTests(unittest.TestCase):
    def setUp(self):
        self.video = torch.arange(10 * 3 * 4 * 8, dtype=torch.float32).reshape(10, 3, 4, 8)
        self.audio = torch.arange(2 * 100, dtype=torch.float32).reshape(2, 100)

    def test_tail_copies_storage_and_detaches_autograd(self):
        self.video.requires_grad_()
        self.audio.requires_grad_()
        raw = b.tensor_media(self.video * 2, self.audio * 2)
        tail = raw.slice_copy(8, 10, 80, 100)
        self.assertFalse(tail.video.requires_grad)
        self.assertFalse(tail.audio.requires_grad)
        self.assertIsNone(tail.video.grad_fn)
        self.assertIsNone(tail.audio.grad_fn)
        self.assertNotEqual(tail.video.untyped_storage().data_ptr(), raw.video.untyped_storage().data_ptr())
        self.assertNotEqual(tail.audio.untyped_storage().data_ptr(), raw.audio.untyped_storage().data_ptr())
        torch.testing.assert_close(tail.video, raw.video[8:10])
        torch.testing.assert_close(tail.audio, raw.audio[:, 80:100])
        raw.video.detach().zero_()
        raw.audio.detach().zero_()
        self.assertNotEqual(float(tail.video.sum()), 0)
        self.assertNotEqual(float(tail.audio.sum()), 0)

    def test_invalid_slices_fail_instead_of_silently_clamping(self):
        media = b.tensor_media(self.video, self.audio)
        cases = [
            (-1, 10, 0, 100),
            (0, 11, 0, 100),
            (9, 8, 0, 100),
            (0, 10, -1, 100),
            (0, 10, 0, 101),
            (0, 10, 50, 49),
            (True, 10, 0, 100),
            (0.0, 10, 0, 100),
        ]
        for bounds in cases:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                media.slice_copy(*bounds)

    def test_invalid_tensor_formats_rejected(self):
        cases = [
            (self.video.to(torch.uint8), self.audio),
            (self.video, self.audio.to(torch.int16)),
            (self.video[:, :2], self.audio),
            (self.video, self.audio[0]),
            (self.video.to("meta"), self.audio),
        ]
        for video, audio in cases:
            with self.subTest(shape=video.shape, dtype=video.dtype), self.assertRaises(ValueError):
                b.tensor_media(video, audio)

    def test_overlap_requires_matching_soundtrack_and_pixels(self):
        raw = b.tensor_media(self.video, self.audio)
        tail = raw.slice_copy(8, 10, 80, 100)
        prefix = b.tensor_media(tail.video.clone(), tail.audio.clone())
        self.assertTrue(prefix.prefix_is_previous_tail(tail, 2, 20))
        prefix.audio[0, 0] += 1
        self.assertFalse(prefix.prefix_is_previous_tail(tail, 2, 20))
        prefix.audio.copy_(tail.audio)
        prefix.video[0, 0, 0, 0] += 1
        self.assertFalse(prefix.prefix_is_previous_tail(tail, 2, 20))


class CurrentSourceBridgeTests(unittest.TestCase):
    def setUp(self):
        self.identity = s.Identity("h3-r2v-high", "a" * 40, "test-build", "b" * 64, "c" * 64)
        self.config = s.StreamConfig(self.identity, raw_frames=22, context_frames=4, measurement_mode="capacity")
        self.calls = []
        self.outputs = []
        owner = self

        class Pipeline:
            task, vae_output_type, video_shift, audio_shift = "ref2va", "float", 12, 3

            def __call__(self, prompt, **kwargs):
                owner.calls.append((prompt, kwargs))
                video = torch.rand(1, 3, 22, 4, 8)
                audio = torch.rand(1, 2, 29334)
                owner.outputs.extend((weakref.ref(video), weakref.ref(audio)))
                return SimpleNamespace(
                    video=video, audio=audio, fps=24, sampling_rate=32000, num_frames=22, video_format="rgb_float"
                )

        self.renderer = b.PipelineRenderer(Pipeline(), width=8, height=4, identity=self.identity)

    def test_real_reference_factory_roundtrip(self):
        video = torch.rand(4, 3, 4, 8)
        audio = torch.rand(2, 5333)
        previous = b.tensor_media(video, audio).slice_copy(0, 4, 0, 5333)
        request = s.Request(
            1, "continue the reference", 7, (p.MiniMaxH3Reference(image=torch.zeros(3, 4, 8)),), previous, self.config
        )
        got = self.renderer(request)
        ref = self.calls[-1][1]["references"][-1]
        self.assertIsInstance(ref, p.MiniMaxH3Reference)
        self.assertEqual(p.reference_media_to_uint8(ref.video).shape, (4, 4, 8, 3))
        self.assertEqual(ref.sample_rate, 32000)
        torch.testing.assert_close(ref.audio, audio)
        self.assertEqual(tuple(got.video.shape), (22, 3, 4, 8))
        self.assertEqual(got.repeated_prefix_frames, 0)

    def test_100_tensor_chunks_release_full_outputs_and_keep_rational_pts(self):
        endpoints = []

        def sink(chunk):
            self.assertEqual(chunk.audio_pts, chunk.video_pts * 32000 // 24)
            endpoints.append(chunk.video_pts + chunk.media.frames)
            return s.Acknowledgement(chunk.index, chunk.media.frames, chunk.media.samples, endpoints[-1])

        session = s.Session(
            self.config, (p.MiniMaxH3Reference(image=torch.zeros(3, 4, 8)),), self.renderer, sink, lambda row: None
        )
        for i in range(100):
            session.step("prompt " + str(i % 2), i)
            # The fake pipeline's call recorder must not itself keep reference tensors.
            self.calls.clear()
            gc.collect()
            self.assertTrue(all(ref() is None for ref in self.outputs))
            self.outputs.clear()
            self.assertEqual(session.previous.frames, 4)
            self.assertEqual(session.previous.samples, 5333)
            self.assertEqual(session.previous.video.untyped_storage().nbytes(), 4 * 3 * 4 * 8 * 4)
        self.assertEqual(endpoints[-1], 2200)
        self.assertEqual(session.samples, 2200 * 32000 // 24)
        session.close()
        self.assertIsNone(session.previous)

    def test_tail_reaches_current_source_reference_preparation(self):
        # Default matching follows the target area; diffusers mode retains the
        # reference canvas. Constant per-frame colors survive spatial rescaling
        # and let this check retain exact temporal/channel and audio assertions.
        video = torch.linspace(0, 1, 66).reshape(22, 3, 1, 1).expand(22, 3, 32, 32)
        audio = torch.rand(2, 29333)
        tail = b.tensor_media(video, audio).slice_copy(0, 22, 0, 29333)
        ref = p.MiniMaxH3Reference(video=tail.video, audio=tail.audio, fps=24, sample_rate=32000)
        expected_colors = p.reference_media_to_uint8(video)[:, :1, :1, :]
        for mode in ("match", "diffusers"):
            for target_height, target_width in ((32, 32), (64, 96)):
                with self.subTest(mode=mode, target=(target_height, target_width)):
                    prepared, frames = r.prepare_references(
                        [ref], 124, 32000, reference_resize_mode=mode,
                        target_height=target_height, target_width=target_width
                    )
                    side = target_height if mode == "match" else 768
                    self.assertEqual(frames, 124)
                    self.assertEqual(prepared[0].frames.shape, (22, side, side, 3))
                    self.assertTrue((prepared[0].frames == expected_colors).all())
                    torch.testing.assert_close(prepared[0].waveform, audio)

    def test_interrupted_delivery_closes_chain_and_releases_context(self):
        def sink(chunk):
            raise KeyboardInterrupt("delivery interrupted after possible partial write")

        session = s.Session(
            self.config, (p.MiniMaxH3Reference(image=torch.zeros(3, 4, 8)),), self.renderer, sink, lambda row: None
        )
        with self.assertRaises(KeyboardInterrupt):
            session.step("interrupt", 1)
        self.calls.clear()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in self.outputs))
        self.assertTrue(session.closed)
        self.assertIsNone(session.previous)
        with self.assertRaises(RuntimeError):
            session.step("retry would duplicate a partial chunk", 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
