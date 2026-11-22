# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Decode router top-k: drop-in for the single-group ttnn.experimental.deepseek_prefill.moe_grouped_topk call in
tt/moe.py (sigmoid, + bias, top-k, gather of the unbiased scores, normalize + eps, * route scale), one token.

One core, as before, but the top-k runs as scalar code on the writer instead of the tile sort network (which
processes 32 token rows). The sigmoid / bias pass and the normalize / scale tail are the op's own compute blocks
with its CB ids, formats and compute config, and the selection reproduces its stable rank-tag order (17-bit
mantissa keys, equal keys by lower index), so weights and ids are bit-equal to the op.
"""

import struct

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

OP_DIR = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels"


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


class FusedRouterTopk:
    def __init__(self, device, bias, k, route_scale, epsilon):
        """bias: [1,1,1,E] fp32 TILE device tensor (as passed to moe_grouped_topk)."""
        assert bias.dtype == ttnn.float32 and bias.layout == ttnn.TILE_LAYOUT
        self.device = device
        self.bias = bias
        self.width_tiles = bias.shape[-1] // 32
        self.k = k
        self.packed = dict(packed_eps=_f32_bits(epsilon), packed_scale=_f32_bits(route_scale))
        fp32 = [0, 1, 5]  # raw fp32 inputs and the biased keys unpack to DEST, as in the op's factory
        self.compute_cfg = F.compute_config(fp32)
        fp, wt = ttnn.float32, self.width_tiles
        self.cbs = [
            F.cb(0, fp, wt), F.cb(1, fp, wt), F.cb(2, ttnn.bfloat16, 2), F.cb(3, ttnn.uint16, 2, 2048),
            F.cb(4, fp, wt), F.cb(5, fp, wt), F.cb(15, fp, 2), F.cb(17, fp, 1), F.cb(18, fp, 1), F.cb(19, fp, 1),
            F.cb(20, fp, 2), F.cb(21, fp, 2), F.cb(22, fp, 2),
        ]  # fmt: skip

    def __call__(self, logits):
        """logits [1,1,1,E] fp32 TILE -> (weights [1,1,1,k] bf16 TILE L1, ids [1,1,1,k] uint16 TILE L1)."""
        assert logits.dtype == ttnn.float32 and logits.layout == ttnn.TILE_LAYOUT, (logits.dtype, logits.layout)
        assert tuple(logits.shape) == (1, 1, 1, 32 * self.width_tiles), logits.shape  # one token
        shape = ttnn.Shape([1, 1, 1, self.k])
        l1 = ttnn.L1_MEMORY_CONFIG
        w = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, l1)
        ids = ttnn.allocate_tensor_on_device(shape, ttnn.uint16, ttnn.TILE_LAYOUT, self.device, l1)
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        reader = ttnn.KernelDescriptor(
            kernel_source=f"{OP_DIR}/dataflow/reader_moe_grouped_topk.cpp",
            core_ranges=F.CORES,
            compile_time_args=acc(logits) + acc(self.bias),
            named_compile_time_args=[
                ("cb_in_scores", 0),
                ("cb_in_bias", 1),
                ("width_tiles", self.width_tiles),
                ("scores_page_size", 4096),
                ("bias_page_size", 4096),
            ],  # fmt: skip
            runtime_args=[(F.CORE, [logits.buffer_address(), self.bias.buffer_address(), 0, 1])],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{F.KDIR}/router_topk_writer.cpp",
            core_ranges=F.CORES,
            compile_time_args=acc(w) + acc(ids),
            named_compile_time_args=[("width_tiles", self.width_tiles), ("k", self.k), *self.packed.items()],
            runtime_args=[(F.CORE, [w.buffer_address(), ids.buffer_address()])],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = F.kernel("router_topk_compute.cpp", self.compute_cfg, dict(width_tiles=self.width_tiles))
        prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=self.cbs, semaphores=[])
        ttnn.generic_op([logits, self.bias, w, ids], prog)
        return w, ids
