# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject unsafe compact layouts before any device program can be built."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from models.demos.qwen38_27b_qb2.tt.gdn_epilogue.op import epilogue


class Validated(Exception):
    pass


class CompactEpilogueContractTest(unittest.TestCase):
    def setUp(self):
        self.ops = SimpleNamespace(
            float32="fp32",
            bfloat16="bf16",
            TILE_LAYOUT="tile",
            ROW_MAJOR_LAYOUT="row",
            DRAM_MEMORY_CONFIG=SimpleNamespace(buffer_type="dram"),
            L1_MEMORY_CONFIG=SimpleNamespace(buffer_type="l1"),
        )

        def stop():
            raise Validated()

        self.mesh = SimpleNamespace(arch=lambda: "BLACKHOLE", compute_with_storage_grid_size=stop)

    def tensors(self, batch=16, compact_gate=True, compact_output=True, width=4160):
        gate = (1, batch, width) if compact_gate else (batch, 1, 1536)
        out = (1, batch, 1536) if compact_output else (batch, 1, 1536)
        shapes = [(batch * 12, 128), gate, (128,), out]
        padded = [
            shapes[0],
            (1 if compact_gate else batch, 32, gate[-1]),
            (32, 128),
            (1 if compact_output else batch, 32, 1536),
        ]
        return [
            SimpleNamespace(
                shape=shape,
                padded_shape=pad,
                dtype="fp32" if i == 0 else "bf16",
                layout="row" if i == 0 else "tile",
                device=lambda: self.mesh,
                memory_config=lambda: self.ops.DRAM_MEMORY_CONFIG,
                buffer_address=lambda i=i: 4096 * (i + 1),
            )
            for i, (shape, pad) in enumerate(zip(shapes, padded))
        ]

    def call(self, tensors, **kwargs):
        with patch.dict(sys.modules, {"ttnn": self.ops}):
            return epilogue(*tensors, **kwargs)

    def test_valid_layouts_and_boundary_batches_reach_program_boundary(self):
        for batch in (1, 16, 17, 31, 32):
            for gate in (False, True):
                for output in (False, True):
                    with self.subTest(batch=batch, gate=gate, output=output), self.assertRaises(Validated):
                        self.call(
                            self.tensors(batch, gate, output),
                            compact_gate=gate,
                            compact_output=output,
                            gate_offset=2560 if gate else 0,
                        )

    def test_rejects_out_of_range_or_unaligned_gate_offset(self):
        for offset in (-32, 1, True, 2688):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.call(self.tensors(), compact_gate=True, compact_output=True, gate_offset=offset)
        with self.assertRaises(ValueError):
            self.call(self.tensors(compact_gate=False), compact_output=True, gate_offset=32)

    def test_rejects_partial_or_oversized_physical_tiles(self):
        for index in (1, 3):
            values = self.tensors()
            values[index].padded_shape = (1, 64, values[index].shape[-1])
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "physical"):
                self.call(values, compact_gate=True, compact_output=True, gate_offset=2560)
        with self.assertRaises(ValueError):
            self.call(self.tensors(batch=33), compact_gate=True, compact_output=True, gate_offset=2560)

    def test_rejects_aliasing_and_non_boolean_layout_flags(self):
        values = self.tensors()
        values[-1].buffer_address = values[1].buffer_address
        with self.assertRaisesRegex(ValueError, "alias"):
            self.call(values, compact_gate=True, compact_output=True, gate_offset=2560)
        for flag in ("compact_gate", "compact_output", "multiply_z"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "Boolean"):
                self.call(self.tensors(), **{flag: 1})
