# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Lightweight bridge contracts; tensor slicing/device numerics need host validation."""

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[3] / "pipelines/minimax_h3"
PACKAGE = "_h3_streaming_bridge_test"
sys.modules[PACKAGE] = ModuleType(PACKAGE)
for name in ("streaming", "streaming_pipeline"):
    spec = importlib.util.spec_from_file_location(PACKAGE + "." + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
s = sys.modules[PACKAGE + ".streaming"]
b = sys.modules[PACKAGE + ".streaming_pipeline"]


class Shape:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, index):
        return self

    def permute(self, *axes):
        return self


class Pipeline:
    task, vae_output_type = "ref2va", "float"

    def __init__(self):
        self.calls = []
        self.output = SimpleNamespace(
            fps=24,
            sampling_rate=32000,
            video_format="rgb_float",
            num_frames=124,
            video=Shape((1, 3, 124, 8, 8)),
            audio=Shape((1, 2, 165334)),
        )

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.output


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.pipe = Pipeline()
        self.identity = s.Identity("h3-r2v-high", "a" * 40, "build", "b" * 64, "c" * 64)
        self.original_media = b.tensor_media
        b.tensor_media = lambda video, audio: (video, audio)
        self.addCleanup(setattr, b, "tensor_media", self.original_media)

    def renderer(self, identity=None):
        return b.PipelineRenderer(
            self.pipe,
            width=1344,
            height=768,
            identity=identity or self.identity,
            reference_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        )

    def request(self, identity=None, previous=None, refs=None):
        return s.Request(
            1,
            "original prompt",
            7,
            refs or (SimpleNamespace(image="fixed"),),
            previous,
            s.StreamConfig(identity or self.identity),
        )

    def test_base_and_rolling_reference_order(self):
        renderer = self.renderer()
        previous = SimpleNamespace(video="tail-video", audio="tail-audio")
        renderer(self.request(previous=previous))
        prompt, call = self.pipe.calls[0]
        self.assertEqual(prompt, "original prompt")
        self.assertEqual(call["num_inference_steps"], 50)
        self.assertEqual(call["references"][0].image, "fixed")
        self.assertEqual(call["references"][1].video, "tail-video")
        self.assertEqual(call["references"][1].audio, "tail-audio")
        self.assertEqual(call["seed"], 7)
        self.assertNotIn("latents", call)

    def test_optional_variants_require_real_adapter_bytes_and_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test-adapter"
            path.write_bytes(b"test-only; not real model weights")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for variant, shift, steps in (("h3-r2v-turbo4", 6, 5), ("h3-r2v-hyperflow8", 12, 9)):
                with self.subTest(variant=variant):
                    self.pipe.lora_path, self.pipe.lora_strength = path, 1.0
                    self.pipe.video_shift, self.pipe.audio_shift = shift, 3
                    self.pipe.hyperflow = object() if steps == 9 else None
                    identity = s.Identity(variant, "a" * 40, "build", "b" * 64, "c" * 64, digest)
                    self.renderer(identity)(self.request(identity))
                    self.assertEqual(self.pipe.calls[-1][1]["num_inference_steps"], steps)
                    wrong = s.Identity(variant, "a" * 40, "build", "b" * 64, "c" * 64, "f" * 64)
                    with self.assertRaisesRegex(ValueError, "digest"):
                        self.renderer(wrong)

    def test_base_cannot_silently_use_adapter(self):
        self.pipe.lora_path = "some-adapter"
        with self.assertRaisesRegex(ValueError, "base route"):
            self.renderer()

    def test_reserves_rolling_video_reference_slot(self):
        renderer = self.renderer()
        request = self.request(refs=tuple(SimpleNamespace(video=i) for i in range(3)))
        with self.assertRaisesRegex(ValueError, "rolling context slot"):
            renderer(request)
        self.assertFalse(self.pipe.calls)

    def test_yuv_is_rejected_without_unmeasured_conversion(self):
        self.pipe.vae_output_type = "yuv420"
        with self.assertRaisesRegex(ValueError, "float output"):
            self.renderer()

    def test_mismatched_output_rates_or_metadata_are_rejected(self):
        renderer = self.renderer()
        for name, value in (("fps", 25), ("sampling_rate", 44100), ("num_frames", 123)):
            with self.subTest(name=name):
                before = getattr(self.pipe.output, name)
                setattr(self.pipe.output, name, value)
                with self.assertRaises(ValueError):
                    renderer(self.request())
                setattr(self.pipe.output, name, before)

    def test_request_identity_is_bound_to_renderer(self):
        different = s.Identity("h3-r2v-high", "d" * 40, "build", "b" * 64, "c" * 64)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.renderer()(self.request(different))
        self.assertFalse(self.pipe.calls)


if __name__ == "__main__":
    unittest.main()
