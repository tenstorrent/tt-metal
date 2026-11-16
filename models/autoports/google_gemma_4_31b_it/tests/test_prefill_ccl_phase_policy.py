# SPDX-License-Identifier: Apache-2.0
"""Host-only regression tests for precision-aware ring prefill phase handling.

The tests execute the actual MultichipDecoder methods extracted from production
source, with inert tensor/CCL doubles. They import neither TTNN nor the model.
"""

import ast
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

SOURCE = Path(__file__).parents[1] / "tt/multichip_decoder.py"


class Tensor:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class HostTTNN:
    class Topology:
        Ring = "ring"

    TILE_LAYOUT = "tile"
    DRAM_MEMORY_CONFIG = "dram"
    bfloat16 = "bf16"
    bfloat8_b = "bfp8"

    def __init__(self):
        self.zero_calls = []
        self.concat_calls = []
        self.slice_calls = []
        self.reduce_calls = []
        self.experimental = SimpleNamespace(reduce_scatter_minimal_async=self.reduce_scatter)

    def zeros(self, shape, *, dtype, layout, device, memory_config):
        self.zero_calls.append({"shape": tuple(shape), "dtype": dtype, "layout": layout})
        return Tensor(shape, dtype)

    def concat(self, tensors, *, dim):
        self.concat_calls.append((tuple(t.shape for t in tensors), dim))
        shape = list(tensors[0].shape)
        shape[dim] = sum(t.shape[dim] for t in tensors)
        assert all(t.dtype == tensors[0].dtype for t in tensors)
        return Tensor(shape, tensors[0].dtype)

    def slice(self, tensor, start, end):
        self.slice_calls.append((tuple(start), tuple(end), tensor.shape))
        shape = tuple(e - s for s, e in zip(start, end))
        return Tensor(shape, tensor.dtype)

    def reduce_scatter(self, tensor, **kwargs):
        self.reduce_calls.append((tensor, kwargs))
        return Tensor(tensor.shape, tensor.dtype)


def _production_methods():
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MultichipDecoder")
    wanted = {"_prefill_ccl_zero_rows", "_reduce_output", "prepare_prefill_ccl"}
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in wanted}
    assert methods.keys() == wanted, methods.keys()
    return methods


def _decoder(*, ttnn, bfp8, topology="ring", decoding=False, phase=32):
    methods = _production_methods()
    env = {"ttnn": ttnn}
    exec(compile(ast.Module(body=list(methods.values()), type_ignores=[]), str(SOURCE), "exec"), env)
    obj = SimpleNamespace(
        prefill_ccl_phase_padding={},
        prefill_ccl_padding={},
        prefill_ccl_phase=phase,
        prefill_ccl_bfp8=bfp8,
        residual_dtype=ttnn.bfloat16,
        topology=topology,
        decoding=decoding,
        mesh_device="mesh",
        hidden=5376,
        persistent_ccl={},
        ccl=SimpleNamespace(
            get_and_cycle_rs_semaphore_handles=lambda: "rs",
            get_and_cycle_barrier_semaphore_handle=lambda: "barrier",
        ),
    )
    for name, method in env.items():
        if name in methods:
            setattr(obj, name, MethodType(method, obj))
    return obj


class PrefillCclPhasePolicyTest(unittest.TestCase):
    def test_bf16_ring_prefill_gets_phase_padding_and_trimmed_result(self):
        ttnn = HostTTNN()
        decoder = _decoder(ttnn=ttnn, bfp8=False)
        result = decoder._reduce_output(Tensor((1, 1, 32, 5376), "bf16"), "self_attn.o_proj")
        self.assertEqual(len(ttnn.zero_calls), 1, "BF16 must not be gated out of ring phase correction")
        self.assertEqual(ttnn.zero_calls[0]["dtype"], "bf16")
        self.assertEqual(ttnn.reduce_calls[0][0].shape, (1, 1, 64, 5376))
        self.assertEqual(len(ttnn.slice_calls), 1)
        self.assertEqual(result.shape, (1, 1, 32, 5376))
        self.assertEqual(result.dtype, "bf16")

    def test_bfp8_ring_prefill_behavior_is_preserved(self):
        ttnn = HostTTNN()
        decoder = _decoder(ttnn=ttnn, bfp8=True)
        result = decoder._reduce_output(Tensor((1, 1, 32, 5376), "bfp8"), "mlp.down_proj")
        self.assertEqual(ttnn.zero_calls[0]["dtype"], "bfp8")
        self.assertEqual(ttnn.reduce_calls[0][0].shape, (1, 1, 64, 5376))
        self.assertEqual(result.shape, (1, 1, 32, 5376))

    def test_decode_and_nonring_prefill_remain_unpadded(self):
        for options, dtype, workers in (
            ({"bfp8": True, "decoding": True}, "bfp8", 2),
            ({"bfp8": False, "topology": "linear"}, "bf16", None),
        ):
            with self.subTest(options=options):
                ttnn = HostTTNN()
                decoder = _decoder(ttnn=ttnn, **options)
                result = decoder._reduce_output(Tensor((1, 1, 32, 5376), dtype), "self_attn.o_proj")
                self.assertEqual(ttnn.zero_calls, [])
                self.assertEqual(ttnn.slice_calls, [])
                self.assertEqual(ttnn.reduce_calls[0][0].shape, (1, 1, 32, 5376))
                self.assertEqual(ttnn.reduce_calls[0][1]["num_workers_per_link"], workers)
                self.assertEqual(result.shape, (1, 1, 32, 5376))

    def test_warmup_uses_policy_dtype_for_both_ring_modes(self):
        for bfp8, dtype in ((False, "bf16"), (True, "bfp8")):
            with self.subTest(bfp8=bfp8):
                ttnn = HostTTNN()
                decoder = _decoder(ttnn=ttnn, bfp8=bfp8, phase=7)
                decoder.prepare_prefill_ccl(batch=2, max_rows=32)
                self.assertEqual(len(ttnn.reduce_calls), 8)
                self.assertEqual({tensor.dtype for tensor, _ in ttnn.reduce_calls}, {dtype})
                self.assertTrue(all(tensor.shape[0:2] == (2, 1) for tensor, _ in ttnn.reduce_calls))
                self.assertEqual(decoder.prefill_ccl_phase, 7)

    def test_zero_row_cache_is_reused_and_dtype_separated(self):
        ttnn = HostTTNN()
        decoder = _decoder(ttnn=ttnn, bfp8=False)
        a = decoder._prefill_ccl_zero_rows(1, 32, "bf16")
        b = decoder._prefill_ccl_zero_rows(1, 32, "bf16")
        c = decoder._prefill_ccl_zero_rows(1, 32, "bfp8")
        self.assertIs(a, b)
        self.assertIsNot(a, c)
        self.assertEqual(a.shape, (1, 1, 32, 5376))
        self.assertEqual([call["dtype"] for call in ttnn.zero_calls], ["bf16", "bfp8"])


if __name__ == "__main__":
    unittest.main()
