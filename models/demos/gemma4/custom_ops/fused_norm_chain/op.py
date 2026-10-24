# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``FusedNormChainDecode``: two chained decode-row RMSNorms of a dense Gemma 4 layer as one multi-core program:

    h = (residual + norm(x) * g1) * scalar          (post_attention_layernorm + residual [* layer scalar])
    y = norm(h) * g2                                 (pre_feedforward_layernorm of the new residual)

with ``norm(t) = t * rsqrt(mean(t^2) + eps)``. ``x`` / ``residual`` are ``[1, 1, R, W]`` (bf16, TILE, interleaved,
``R <= 32`` rows on one row tile), ``g1`` / ``g2`` the ``[1, 1, 32, W]`` row-replicated gammas (``FusedNormDecode.
full_gamma``). Returns ``(h, y)`` in ``x``'s memory config; ``h`` is rounded to bf16 before the second norm reads
it, exactly as the two separate programs round it, so the pair is bit-identical to ``FusedNormDecode`` (residual
form) followed by ``FusedNormDecode`` (single form).

``N`` cores (the ``FusedNormDecode.num_cores`` choice: 21 x 8 column tiles at W=5376, 20 x 6 at 3840) own
contiguous column tiles; each norm is one partial-row-sum exchange (one 128-byte NoC write per core and
destination, one semaphore counting both rounds). This replaces ``post_attention_layernorm + residual`` and
``pre_feedforward_layernorm`` (2 fused programs, or 7 of the shipped width-sharded chain) with one program.
"""

import struct
from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid, core_range_set, plan_norm_cores, row_major_cores

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class FusedNormChainDecode:
    CB_X, CB_G1, CB_G2, CB_RES, CB_SCALER = 0, 1, 2, 3, 4
    CB_SQ, CB_PART, CB_RECV, CB_RECV2, CB_INV, CB_H, CB_STAGE, CB_STAGE2, CB_SEND = 5, 6, 7, 8, 9, 10, 11, 12, 13
    CB_OUT0, CB_OUT1 = 16, 17
    SEM = 0

    @staticmethod
    def golden(x, residual, g1, g2, eps, scalar=1.0):
        """torch reference with the chain's bf16 rounding of h."""

        def norm(t):
            tf = t.float()
            return tf * torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + eps)

        h = (residual.float() + norm(x) * g1.float()) * scalar
        h_bf = h.to(torch.bfloat16)
        y = norm(h_bf) * g2.float()
        return h, y

    @classmethod
    def op(cls, x, residual, g1, g2, eps: float, scalar: float = 1.0, y_memory_config=None):
        """``y_memory_config``: ``None`` (x's memory config) or an L1 WIDTH_SHARDED one-row-tile layout of whole
        column tiles (the DRAM-sharded gate_up matmul's activation layout) for ``y``."""
        assert (
            x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16 and not x.is_sharded()
        ), "x: bf16 TILE interleaved"
        shape = [int(v) for v in x.shape]
        assert (
            len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and shape[2] <= TILE
        ), f"x must be [1,1,<=32,W], got {shape}"
        W = shape[3]
        assert W % TILE == 0 and x.padded_shape[-2] == TILE, f"width {W} must be a multiple of 32 on one row tile"
        assert [int(v) for v in residual.shape] == shape and residual.layout == ttnn.TILE_LAYOUT, residual.shape
        for g in (g1, g2):
            assert g.layout == ttnn.TILE_LAYOUT and tuple(int(v) for v in g.shape) == (1, 1, TILE, W), g.shape
        cols_t = W // TILE
        device = x.device()
        grid = compute_grid(device)
        N = plan_norm_cores(grid, cols_t, 1)
        per = cols_t // N
        RT = (N + TILE - 1) // TILE
        ym = y_memory_config
        if ym is not None and ym.is_sharded():
            assert (
                ym.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
                and int(ym.shard_spec.shape[0]) == TILE
                and int(ym.shard_spec.shape[1]) % TILE == 0
                and W % int(ym.shard_spec.shape[1]) == 0
            ), f"sharded y must hold whole column tiles of one row tile: {ym}"
        outs = [
            ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, device, x.memory_config()),
            ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, device, x.memory_config() if ym is None else ym),
        ]
        cores = row_major_cores(grid, N)
        core_set = core_range_set(cores)
        noc = [device.worker_core_from_logical_core(c) for c in cores]
        tile_bytes = TILE * TILE * 2
        f32_tile = TILE * TILE * 4

        def cb(idx, pages, page=tile_bytes, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=pages * page,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
            )

        cbs = [
            cb(cls.CB_X, per),
            cb(cls.CB_G1, per),
            cb(cls.CB_G2, per),
            cb(cls.CB_RES, per),
            cb(cls.CB_SCALER, 1),
            cb(cls.CB_SQ, per, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_PART, 1, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_RECV, RT, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_RECV2, RT, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_INV, 1, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_H, per),
            cb(cls.CB_STAGE, 1, page=N * 128, fmt=ttnn.float32),
            cb(cls.CB_STAGE2, 1, page=N * 128, fmt=ttnn.float32),
            cb(cls.CB_SEND, 1, page=128, fmt=ttnn.float32),
            cb(cls.CB_OUT0, per),
            cb(cls.CB_OUT1, per),
        ]
        reader_ct = [cls.CB_X, cls.CB_G1, cls.CB_G2, cls.CB_RES, cls.CB_SCALER, cls.CB_RECV, cls.CB_RECV2, cls.CB_STAGE,
                     cls.CB_STAGE2, per, N, RT, cls.SEM]  # fmt: skip
        for t in (x, g1, g2, residual):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [cls.CB_PART, cls.CB_SEND, cls.CB_STAGE, cls.CB_STAGE2, cls.CB_OUT0, cls.CB_OUT1, per, N, cls.SEM]
        for t in outs:
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cls.CB_X, cls.CB_G1, cls.CB_G2, cls.CB_RES, cls.CB_SCALER, cls.CB_SQ, cls.CB_PART, cls.CB_RECV,
                      cls.CB_RECV2, cls.CB_INV, cls.CB_H, cls.CB_OUT0, cls.CB_OUT1, per, N, RT, _f32_bits(eps),
                      _f32_bits(1.0 / W), _f32_bits(scalar)]  # fmt: skip
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        coords = []
        for c in noc:
            coords += [c.x, c.y]
        for i, core in enumerate(cores):
            reader_rt[core.x][core.y] = [
                x.buffer_address(), g1.buffer_address(), g2.buffer_address(), residual.buffer_address(), i * per, i * per
            ]  # fmt: skip
            writer_rt[core.x][core.y] = [outs[0].buffer_address(), outs[1].buffer_address(), i * per, i] + coords
            compute_rt[core.x][core.y] = [0]
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_fused_norm_chain.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_fused_norm_chain.cpp"),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_fused_norm_chain.cpp"),
                core_ranges=core_set,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
        ]
        sems = [ttnn.SemaphoreDescriptor(cls.SEM, ttnn.CoreType.WORKER, core_set, 0)]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        ttnn.generic_op([x, residual, g1, g2] + outs, program)
        return outs[0], outs[1]
