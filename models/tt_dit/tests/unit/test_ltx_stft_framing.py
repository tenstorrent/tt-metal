# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU framing and RM staging contracts; run directly without device conftest.

The production methods run against torch copy operations plus the native RM
reshape's aligned-page scratch formula. This catches the production-size L1
failure and indexing regressions; device execution/replay is separately covered
by test_audio_device_chain.py and cannot be established by this adapter.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


class Tensor:
    def __init__(self, value):
        self.value = value
        self.shape = tuple(value.shape)
        self.dtype = value.dtype
        self.layout = "RM"


def _reshape_scratch(source_width, destination_width):
    # Aligned/divisible FP32 pages in reshape_rm_program_factory.cpp: two
    # kernels, each with two source slots and one destination slot.
    source_bytes = source_width * 4
    destination_bytes = destination_width * 4
    assert source_bytes % 16 == destination_bytes % 16 == 0
    assert source_bytes % destination_bytes == 0
    source_slot = ((source_bytes - 1) & ~63) + 128
    destination_slot = ((destination_bytes - 1) & ~63) + 80
    return 4 * source_slot + 2 * destination_slot


class DeviceCopies:
    Tensor = Tensor
    float32 = torch.float32
    uint32 = torch.int32
    ROW_MAJOR_LAYOUT = "RM"

    def __init__(self):
        self.reshape_scratch = []
        self.gathers = 0
        self.uploads = 0

    def from_torch(self, value, **kwargs):
        self.uploads += 1
        return Tensor(value.clone())

    def reshape(self, tensor, shape):
        if tensor.shape[-1] > shape[-1] and shape[-1] == 512:
            scratch = _reshape_scratch(tensor.shape[-1], shape[-1])
            self.reshape_scratch.append(scratch)
            # Leave room for the allocator/dispatch prefix; the original ~9.8MB
            # request exceeds even the full BH per-core 1.5MiB capacity.
            if scratch > 1_572_864 - 128 * 1024:
                raise RuntimeError("RM reshape source row exceeds per-core L1")
        return Tensor(tensor.value.reshape(shape))

    def pad(self, tensor, pairs, value):
        return Tensor(torch.nn.functional.pad(tensor.value, tuple(v for p in reversed(pairs) for v in p), value=value))

    def gather(self, tensor, dimension, indices, **kwargs):
        self.gathers += 1
        return Tensor(torch.gather(tensor.value, dimension, indices.value.long()))

    def slice(self, tensor, starts, ends):
        assert starts[-1] * 4 % 128 == 0
        return Tensor(tensor.value[tuple(slice(start, end) for start, end in zip(starts, ends))].contiguous())

    def concat(self, tensors, dim):
        assert dim == 1 and all(tensor.shape[-1] == 512 for tensor in tensors)
        return Tensor(torch.cat([tensor.value for tensor in tensors], dim=dim))


def _production_methods(adapter):
    path = Path(__file__).resolve().parents[2] / "models" / "audio_vae" / "bwe_ltx.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_STFTFn")
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in {"prepare_device_windows", "_frame_device"}
    ]
    namespace = {"ttnn": adapter, "torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _stft():
    return SimpleNamespace(
        left_pad=432,
        win_length=512,
        hop_length=80,
        mesh_device=object(),
        _window_indices={},
        _gather_grid="64 workers",
    )


class STFTFramingTest(unittest.TestCase):
    def test_original_production_reshape_exceeds_l1(self):
        for length in (96160, 96640):
            frames = length // 80
            self.assertGreater(_reshape_scratch(frames * 512, 512), 9_800_000)
            adapter = DeviceCopies()
            with self.assertRaisesRegex(RuntimeError, "per-core L1"):
                adapter.reshape(Tensor(torch.empty(1, 1, 2, frames * 512)), (2, frames, 512))

    def test_copy_exact_at_causal_strip_batch_and_production_boundaries(self):
        adapter = DeviceCopies()
        methods = _production_methods(adapter)
        stft = _stft()
        lengths = (80, 159, 160, 511, 512, 1025, 5119, 5120, 5199, 5200, 10240, 10320, 96160, 96640)
        for batch in (1, 2, 4):
            for length in lengths:
                with self.subTest(batch=batch, length=length):
                    # Unique batch offsets expose accidental frame/batch interleaving.
                    values = torch.arange(batch * length, dtype=torch.float32).reshape(batch, length, 1)
                    values = (values % 8191 - 4096) / 8192
                    values[:, 0] = -0.0
                    methods["prepare_device_windows"](stft, batch, length)
                    uploads = adapter.uploads
                    gathers = adapter.gathers
                    observed = methods["_frame_device"](stft, Tensor(values)).value
                    expected = torch.nn.functional.pad(values.squeeze(-1), (432, 0)).unfold(-1, 512, 80).contiguous()
                    self.assertTrue(torch.equal(observed.view(torch.int32), expected.view(torch.int32)))
                    self.assertEqual(adapter.uploads, uploads, "framing must not upload waveform values")
                    self.assertEqual(adapter.gathers, gathers + 1, "gather the waveform only once")
        self.assertLessEqual(max(adapter.reshape_scratch), 530 * 1024)

    def test_changed_input_reuses_indices_without_stale_values(self):
        adapter = DeviceCopies()
        methods = _production_methods(adapter)
        stft = _stft()
        for length in (5200, 96160):
            methods["prepare_device_windows"](stft, 2, length)
            indices = stft._window_indices[(2, length)]
            uploads = adapter.uploads
            first = torch.arange(2 * length, dtype=torch.float32).reshape(2, length, 1) / (2 * length)
            second = -first.flip(1)
            outputs = []
            for values in (first, second, first):
                methods["prepare_device_windows"](stft, 2, length)
                observed = methods["_frame_device"](stft, Tensor(values)).value
                expected = torch.nn.functional.pad(values.squeeze(-1), (432, 0)).unfold(-1, 512, 80).contiguous()
                self.assertTrue(torch.equal(observed.view(torch.int32), expected.view(torch.int32)))
                outputs.append(observed)
            self.assertTrue(torch.equal(outputs[0], outputs[2]))
            self.assertFalse(torch.equal(outputs[0], outputs[1]))
            self.assertEqual(adapter.uploads, uploads)
            self.assertIs(stft._window_indices[(2, length)], indices)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
