# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU buffer-ownership contracts; native copy/trace correctness needs hardware.

Run directly without device conftest. Actual production handoff/allocation and
StateTensor methods run against an adapter with borrowed, mutable trace outputs.
"""

import ast
import copy
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    def __init__(self, shape, values):
        self.shape = shape
        self.values = values


class DeviceOperations:
    bfloat16 = "BF16"
    TILE_LAYOUT = "TILE"

    def __init__(self):
        self.allocations = []
        self.copies = []

    def zeros(self, shape, **kwargs):
        result = Tensor(shape, [0.0])
        self.allocations.append(result)
        return result

    @staticmethod
    def unsqueeze(value, dim):
        assert dim == 0
        return Tensor((1, *value.shape), value.values)

    def copy(self, source, destination):
        assert source.shape == destination.shape
        destination.values[:] = copy.deepcopy(source.values)
        self.copies.append((source, destination))

    @staticmethod
    def to_torch(*args, **kwargs):
        raise AssertionError("handoff must not read embeddings to host")


def load(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return namespace


class PromptHandoffTest(unittest.TestCase):
    def setUp(self):
        self.ops = DeviceOperations()
        self.traces = SimpleNamespace(_traces_live={})
        ns = {"ttnn": self.ops, "torch": SimpleNamespace(is_tensor=lambda _: False), "Tracer": self.traces}
        state_type = load(ROOT / "utils/tracing.py", {"StateTensor"}, ns)["StateTensor"]
        allocate = load(ROOT / "pipelines/ltx/pipeline_ltx_distilled.py", {"_allocate_device_prompt_buffers"}, ns)[
            "_allocate_device_prompt_buffers"
        ]
        handoff = load(ROOT / "encoders/gemma3/encoder_pair.py", {"encode_to_device_buffers"}, ns)[
            "encode_to_device_buffers"
        ]
        self.video = Tensor((1, 1024, 4096), [0.0])
        self.audio = Tensor((1, 1024, 2048), [0.0])
        self.encoded = []

        def encode(prompt):
            self.encoded.append(prompt)
            value = {"a": 1.25, "b": -2.5}[prompt]
            self.video.values[:] = [value, -0.0]
            self.audio.values[:] = [-value, 0.0]
            return self.video, self.audio

        self.pair = SimpleNamespace(
            dynamic_load=False,
            _sequence_length=1024,
            _video_dim=4096,
            _audio_dim=2048,
            sequence_length=1024,
            video_dim=4096,
            audio_dim=2048,
            _encode_prompt_device=encode,
        )
        self.pair.encode_to_device_buffers = MethodType(handoff, self.pair)
        self.pipe = SimpleNamespace(
            gemma_encoder_pair=self.pair,
            dynamic_load=False,
            cross_attention_dim=4096,
            mesh_device=SimpleNamespace(id=lambda: 13),
            _prompt_v=state_type(),
            _prompt_a=state_type(),
        )
        self.pipe.allocate = MethodType(allocate, self.pipe)

    def test_changed_prompts_restore_without_aliasing_or_reallocating(self):
        self.pipe.allocate()
        video, audio = self.pipe._prompt_v.value, self.pipe._prompt_a.value
        self.assertEqual(len(self.ops.allocations), 2)
        for prompt in ("a", "b", "a"):
            self.pipe.allocate()
            self.pair.encode_to_device_buffers(prompt, video, audio)
            expected = 1.25 if prompt == "a" else -2.5
            self.assertEqual(video.values, [expected, -0.0])
            self.assertEqual(audio.values, [-expected, 0.0])
            # Simulate a later replay overwriting borrowed encoder output storage.
            self.video.values[:] = [999.0]
            self.audio.values[:] = [999.0]
            self.assertEqual(video.values, [expected, -0.0])
            self.assertEqual(audio.values, [-expected, 0.0])
            self.assertIs(self.pipe._prompt_v.value, video)
            self.assertIs(self.pipe._prompt_a.value, audio)
        self.assertEqual(self.encoded, ["a", "b", "a"])
        self.assertEqual(len(self.ops.allocations), 2)
        self.assertEqual(len(self.ops.copies), 6)

    def test_unsupported_routes_fail_before_encoder_or_allocation(self):
        self.pipe.dynamic_load = True
        with self.assertRaises(AssertionError):
            self.pipe.allocate()
        self.pipe.dynamic_load = False
        self.pipe.cross_attention_dim = 8192
        with self.assertRaises(AssertionError):
            self.pipe.allocate()
        self.assertEqual(self.ops.allocations, [])
        self.pipe.cross_attention_dim = 4096
        self.pipe.allocate()
        video, audio = self.pipe._prompt_v.value, self.pipe._prompt_a.value
        self.pair.dynamic_load = True
        with self.assertRaises(AssertionError):
            self.pair.encode_to_device_buffers("a", video, audio)
        self.pair.dynamic_load = False
        with self.assertRaises(AssertionError):
            self.pair.encode_to_device_buffers("a", Tensor((1, 1024, 4096), []), audio)
        self.assertEqual(self.encoded, [])

    def test_reallocation_is_rejected_while_any_trace_is_live(self):
        self.traces._traces_live[13] = 1
        with self.assertRaisesRegex(AssertionError, "before capture"):
            self.pipe.allocate()
        self.assertEqual(self.ops.allocations, [])
        self.traces._traces_live[13] = 0
        self.pipe.allocate()
        self.traces._traces_live[13] = 3
        self.pipe.allocate()  # Existing addresses are safe to reuse.
        self.assertEqual(len(self.ops.allocations), 2)
        self.pipe._prompt_v._data = None  # release_traces discards the sinks.
        self.pipe._prompt_a._data = None
        with self.assertRaisesRegex(AssertionError, "before capture"):
            self.pipe.allocate()
        self.assertEqual(len(self.ops.allocations), 2)


if __name__ == "__main__":
    unittest.main()
