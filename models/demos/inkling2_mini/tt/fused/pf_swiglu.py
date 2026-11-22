# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill softcapped SwiGLU over the grouped expert gate/up output in one program: gu [.., rows, 2 H] bf16 TILE
(gate columns [0, H), up columns [H, 2 H)) -> [.., rows, H] bf16 = 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25).

Replaces the two column slices and moe.softcapped_swiglu's two ttnn.mul calls (with their fused activations) with
the same SFPU steps and bf16 rounding points (binary_ng's SFPU MUL path on bf16), reading g and u in
place."""

import struct

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.moe import GATE_SOFTCAP, UP_SOFTCAP


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


class PfSwiglu:
    def __init__(self, device):
        self.device = device

    def __call__(self, gu):
        shape = list(gu.shape)
        h2 = shape[-1]
        rows = 1
        for d in shape[:-1]:
            rows *= d
        assert gu.dtype == ttnn.bfloat16 and gu.layout == ttnn.TILE_LAYOUT and rows % 32 == 0 and h2 % 64 == 0, gu
        ht, rt = h2 // 64, rows // 32
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape[:-1] + [h2 // 2]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                             self.device, ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        grid = self.device.compute_with_storage_grid_size()
        n_cores = min(rt, grid.x * grid.y)
        cores, crs = F.core_list(self.device, n_cores)
        r_args, w_args, c_args = {}, {}, {}
        for i, core in enumerate(cores):
            r0, r1 = rt * i // n_cores, rt * (i + 1) // n_cores
            r_args[core] = [gu.buffer_address(), r0, r1]
            w_args[core] = [out.buffer_address(), r0, r1]
            c_args[core] = [r1 - r0]
        named = dict(ht=ht, g_scale=_bits(1.0 / GATE_SOFTCAP), u_scale=_bits(1.0 / UP_SOFTCAP),
                     post=_bits(GATE_SOFTCAP * UP_SOFTCAP))  # fmt: skip
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=False)  # fmt: skip
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in range(6):  # as binary_ng's SFPU path: inputs and activation intermediates unpacked to DST as fp32
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
        kernels = [
            F.kernel("pf_swiglu_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, crs),
            F.kernel("pf_swiglu_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, crs),
            F.kernel("pf_swiglu_compute.cpp", cfg, named, c_args, crs),
        ]
        bf = ttnn.bfloat16
        cbs = [F.cb(i, bf, 2, cores=crs) for i in (0, 1, 2, 3, 4, 5)] + [F.cb(16, bf, 2, cores=crs)]
        F.run([gu, out], kernels, cbs)
        return out
