# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a `router_tail`: the decode router tail as ONE generic_op program.

Replaces (OptimizedRouter decode, router_impl="topk", no topk padding): sharded_to_interleaved of the logits, slice to
the 320 padded columns, sigmoid, + bias, topk (1 core, ~54 us), the id typecast, the 288-column slice and the bf16
typecast of the scores: 7-8 ops, ~71 us at B = 1.

Per token row t (B <= 32): scores = sigmoid(logits[:320]), choice = scores + bias (bias -1e9 on the padded experts
288..319), ids = the 8 largest choices in descending order (ties to the lower expert id), scores[:288] in bf16.
Program: one core per token row (logical (t % 8, t // 8)); the row's 10 logit column tiles are packed into one tile
(packed row r = experts 32r .. 32r+31) so the SFPU runs one tile; the top-8 runs on the writer RISC with integer keys
of the fp32 bits (no FPU on the RISC). sigmoid / add / bf16 rounding are the same SFPU functions as ttnn.sigmoid /
the fp32 BinaryNg add / ttnn.typecast, so choice and scores are bit-identical to the op chain and the ids identical up
to the tie rule (ttnn.topk's order among exactly equal choices is not specified). `reference_step` is the host oracle.
"""

from __future__ import annotations

import os

import torch

import ttnn

KERNEL_DIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "router_tail")
FP32, BF16 = ttnn.float32, ttnn.bfloat16
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
TILE = 32
COLS = 320
N_EXPERTS = 288
TOP_K = 8
CB = dict(LOG=0, BIAS=1, CHOICE=2, SCORES=3, IDX=4)


def packed_bias(bias320: torch.Tensor) -> torch.Tensor:
    """bias [320] -> [32, 32] tile, row r = bias[32r : 32r + 32] (rows 10..31 zero)."""
    out = torch.zeros(TILE, TILE)
    out[: COLS // TILE] = bias320.float().reshape(COLS // TILE, TILE)
    return out


def reference_step(logits: torch.Tensor, bias320: torch.Tensor):
    """logits [B, >= 320] fp32 -> (ids [B, 8] int64 descending, ties to the lower id; scores [B, 288] bf16)."""
    scores = torch.sigmoid(logits[:, :COLS].float())
    choice = scores + bias320.float()
    # stable descending order: sort by (-choice, id)
    order = torch.stack([torch.tensor(sorted(range(N_EXPERTS), key=lambda e: (-float(row[e]), e))[:TOP_K])
                         for row in choice[:, :N_EXPERTS]])  # fmt: skip
    return order, scores[:, :N_EXPERTS].to(torch.bfloat16)


class RouterTail:
    """One MoE layer's fused router tail (bound to its selection bias); call per decode step on the router logits."""

    def __init__(self, mesh_device, bias320: torch.Tensor):
        self.device = mesh_device
        self.bias = ttnn.from_torch(packed_bias(bias320), dtype=FP32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                    memory_config=DRAM)  # fmt: skip
        self.host_bias = bias320.float()
        self.calls = 0

    def __call__(self, logits, *, rows: int, out_memory_config=L1):
        """logits [1,1,B,N >= 320] fp32 (the router matmul's output, interleaved or sharded) -> (ids [1,1,B,8] uint16,
        scores [1,1,B,288] bf16) in fused_experts' format."""
        if not 1 <= rows <= TILE or logits.dtype != FP32 or logits.shape[-1] < COLS:
            raise ValueError(f"router_tail: rows {rows}, logits {logits.dtype} {tuple(logits.shape)}")
        b = rows
        idx = ttnn.empty((1, 1, b, TOP_K), ttnn.uint16, ttnn.TILE_LAYOUT, self.device, out_memory_config)
        scores = ttnn.empty((1, 1, b, N_EXPERTS), BF16, ttnn.TILE_LAYOUT, self.device, out_memory_config)
        cores = [(t % 8, t // 8) for t in range(b)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])

        def rt(fn):
            r = ttnn.RuntimeArgs()
            for t, (x, y) in enumerate(cores):
                r[x][y] = [int(v) for v in fn(t)]
            return r

        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNEL_DIR, "reader.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=acc(logits) + acc(self.bias),
            runtime_args=rt(lambda t: [logits.buffer_address(), self.bias.buffer_address(), t]),
            config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
        )  # fmt: skip
        writer = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNEL_DIR, "writer.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs, compile_time_args=acc(idx) + acc(scores),
            runtime_args=rt(lambda t: [idx.buffer_address(), scores.buffer_address(), t]),
            config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
        )  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[CB["LOG"]] = modes[CB["BIAS"]] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        compute = ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNEL_DIR, "compute.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            # the row argument (router_tail_common.hpp: first row | (rows - 1) << 16; here one row per core): the compute
            # loops over the core's rows — every caller must pass it (without it the kernel reads an unset arg and hangs)
            core_ranges=crs, compile_time_args=[], runtime_args=rt(lambda t: [t]), config=cc,
        )  # fmt: skip
        cb = lambda i, dt, page: ttnn.CBDescriptor(
            total_size=page, core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )  # fmt: skip
        cbs = [cb(CB["LOG"], FP32, 4096), cb(CB["BIAS"], FP32, 4096), cb(CB["CHOICE"], FP32, 4096),
               cb(CB["SCORES"], BF16, 2048), cb(CB["IDX"], BF16, 64)]  # fmt: skip
        ttnn.generic_op([logits, self.bias, idx, scores], ttnn.ProgramDescriptor(kernels=[reader, writer, compute],
                                                                              semaphores=[], cbs=cbs))  # fmt: skip
        self.calls += 1
        return idx, scores
