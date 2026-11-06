# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``FusedNormDecode``: the decode-row RMSNorm family of a Gemma 4 layer as one multi-core program.

``x`` is ``[1, 1, R, W]`` (bf16, TILE, DRAM or L1 interleaved, ``W`` a multiple of 32; ``R <= 32`` decode rows on
one row tile, or tile-padded prefill rows: every row tile gets its own ``N``-core exchange group, so a 128-row
call runs the same kernels on ``4 N`` cores). ``out_o = (norm(x) * gamma_o [+ norm(x2) * gamma2] [+ residual]) * scalar_o`` with ``norm(x) = x *
rsqrt(mean(x^2) + eps)`` (the layer applies its scalar after the residual add). Forms:

* one to three outputs from one input (``gammas = [g0, g1, g2]``, each a ``[1, 1, 32, W]`` bf16 TILE tensor with
  the gamma row replicated on every row -- ``full_gamma()`` -- or ``None``): the three norms a layer takes of the
  same residual (pre_feedforward, pre_feedforward_2, the router's) share one row-sum exchange and one program;
* ``residual``: ``out = (residual + norm(x) * gamma) * scalar`` (post_attention / post_feedforward norms followed by
  the residual add and the layer scalar);
* ``x2`` / ``gamma2``: ``out = norm(x) * gamma + norm(x2) * gamma2`` (post_feedforward_1 of the shared MLP output
  plus post_feedforward_2 of the expert output).

``N`` cores (a divisor of the ``W/32`` column tiles, ``GEMMA4_FUSED_NORM_CORES`` caps it, default 22) each own
``per`` contiguous column tiles: they square their tiles (FPU, fp32 dest), row-reduce them (FPU reduce with a unit
scaler) into one fp32 partial tile, send its column (32 fp32) to every core (one 128-byte NoC write per input and
destination), scatter the ``N`` received columns into one fp32 tile and row-reduce it again, so the full row sums
and ``rsqrt(sum / W + eps)`` come out of the FPU / SFPU on every core; the outputs are then ``x * inv`` (FPU column
broadcast) ``* gamma`` (FPU, dest reused) ``[+ residual]`` ``[* scalar]`` with one bf16 rounding. Replaces the
decode ``to_memory_config -> rms_norm (width-sharded) -> sharded_to_interleaved`` triple of every norm and the
``add`` / ``mul`` programs around the norms: 12 programs per layer -> 5.
"""

import struct
from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import NORM_MAX_CORES, compute_grid, plan_norm_split

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class FusedNormDecode:
    CB_X0, CB_X1, CB_G0, CB_G1, CB_G2, CB_RES, CB_SCALER = 0, 1, 2, 3, 4, 5, 6
    CB_SQ, CB_PART, CB_RECV, CB_INV, CB_STAGE, CB_SEND = 7, 8, 9, 10, 11, 12
    CB_OUT0, CB_OUT1, CB_OUT2 = 16, 17, 18
    SEM = 0
    MAX_CORES = NORM_MAX_CORES  # two fp32 receive tiles of 32 columns

    @staticmethod
    def golden(x, gammas, eps, x2=None, gamma2=None, residual=None, scalars=None):
        """torch reference. ``gammas``: list of ``[W]`` tensors or None; returns the list of outputs."""

        def norm(t):
            tf = t.float()
            return tf * torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + eps)

        n = norm(x)
        scalars = scalars or [1.0] * len(gammas)
        outs = []
        for g, s in zip(gammas, scalars):
            o = n * g.float() if g is not None else n.clone()
            if x2 is not None:
                n2 = norm(x2)
                o = o + (n2 * gamma2.float() if gamma2 is not None else n2)
            if residual is not None:
                o = o + residual.float()
            outs.append(o * s)
        return outs

    @staticmethod
    def full_gamma(gamma_row: ttnn.Tensor) -> ttnn.Tensor:
        """The ``[1, 1, 32, W]`` bf16 TILE gamma the op takes, from the ``[1, 1, 1, W]`` TILE row (an on-device
        repeat, so a warm weight cache never persists it as a placeholder)."""
        assert tuple(int(v) for v in gamma_row.shape)[:3] == (1, 1, 1), gamma_row.shape
        return ttnn.repeat(gamma_row, ttnn.Shape([1, 1, TILE, 1]))

    @staticmethod
    def num_cores(cols_t: int, nin: int = 1, grid_cores_: int = None) -> int:
        """Largest divisor of the column tiles up to the cap (``GEMMA4_FUSED_NORM_CORES``, default 22, at most 64)
        whose per-core tile count fits one fp32 dest window (8 tiles, 4 per input with two inputs); with
        ``grid_cores_`` the count is also limited to the grid (one row tile)."""
        return FusedNormDecode.plan(cols_t, nin, 1, grid_cores_ or FusedNormDecode.MAX_CORES)[0]

    @staticmethod
    def plan(cols_t: int, nin: int, rt_in: int, grid_cores_: int):
        """The work split for ``rt_in`` row tiles on a grid of ``grid_cores_`` cores: ``(N, per, groups, passes)``
        (``custom_ops/grid.plan_norm_split``, the one planner of the fused norms). Raises ValueError when no split
        fits (the callers fall back to the composite path)."""
        return plan_norm_split(grid_cores_, cols_t, nin, rt_in)

    @staticmethod
    def plan_ok(cols_t: int, nin: int, rt_in: int, grid_cores_: int) -> bool:
        try:
            FusedNormDecode.plan(cols_t, nin, rt_in, grid_cores_)
            return True
        except ValueError:
            return False

    @classmethod
    def op(
        cls,
        x: ttnn.Tensor,
        gammas,
        eps: float,
        x2=None,
        gamma2=None,
        residual=None,
        scalars=None,
        out_memory_configs=None,
    ):
        """``out_memory_configs``: per output, ``None`` (the input's memory config) or an L1 WIDTH_SHARDED config
        of whole column tiles on one row tile (a DRAM-sharded decode matmul's activation layout,
        ``DramShardedLinear.act_memory_config``): the writer's page accessor lands every column tile on its shard
        core, so the consuming matmul skips its interleaved -> sharded copy."""
        assert (
            x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16 and not x.is_sharded()
        ), "x: bf16 TILE interleaved"
        shape = [int(v) for v in x.shape]
        assert len(shape) == 4 and shape[0] == 1 and shape[1] == 1, f"x must be [1,1,R,W], got {shape}"
        W = shape[3]
        # one row tile (decode rows) or, on prefill rows, RT_IN row tiles: one N-core exchange group per row tile
        RT_IN = (shape[2] + TILE - 1) // TILE
        assert x.padded_shape[-2] == RT_IN * TILE, f"rows {shape[2]} must be tile-padded"
        assert W % TILE == 0, f"width {W} must be a multiple of 32"
        gammas = list(gammas)
        nout = len(gammas)
        assert 1 <= nout <= 3, "one to three outputs"
        nin = 2 if x2 is not None else 1
        assert nin == 1 or nout == 1, "the two-input form has one output"
        assert residual is None or nout == 1, "the residual form has one output"
        scalars = list(scalars) if scalars is not None else [1.0] * nout
        assert len(scalars) == nout
        for t, name in ((x2, "x2"), (residual, "residual")):
            if t is not None:
                assert [int(v) for v in t.shape] == shape and t.layout == ttnn.TILE_LAYOUT, f"{name}: {t.shape}"
        gamma_list = [gammas[0], gamma2] if nin == 2 else gammas
        for g in gamma_list:
            if g is not None:
                assert g.layout == ttnn.TILE_LAYOUT and tuple(int(v) for v in g.shape) == (1, 1, TILE, W), g.shape
        gmask = sum(1 << i for i, g in enumerate(gamma_list) if g is not None)
        cols_t = W // TILE
        device = x.device()
        gx, gy = compute_grid(device)  # (x, y), capped by GEMMA4_TEST_GRID
        N, per, groups, passes = cls.plan(cols_t, nin, RT_IN, gx * gy)
        RT = (N + TILE - 1) // TILE
        out_mems = list(out_memory_configs) if out_memory_configs is not None else [None] * nout
        assert len(out_mems) == nout
        for om in out_mems:
            if om is not None and om.is_sharded():
                assert RT_IN == 1, "a sharded output needs one row tile"
                assert (
                    om.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
                    and int(om.shard_spec.shape[0]) == TILE
                    and int(om.shard_spec.shape[1]) % TILE == 0
                    and W % int(om.shard_spec.shape[1]) == 0
                ), f"sharded output must hold whole column tiles of one row tile: {om}"
        outs = [
            ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, device, x.memory_config() if om is None else om)
            for om in out_mems
        ]
        # one N-core exchange group per row tile of a pass; row tiles beyond the grid's groups run in further passes
        # of the same program (the row-tile offset is a runtime arg)
        cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(N * groups)]  # group g = cores[g*N:(g+1)*N]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        noc = [device.worker_core_from_logical_core(c) for c in cores]
        tile_bytes = TILE * TILE * 2
        f32_tile = TILE * TILE * 4

        def cb(idx, pages, page=tile_bytes, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=pages * page,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
            )

        block = nin * 128
        cbs = [
            cb(cls.CB_X0, per),
            cb(cls.CB_X1, per if nin == 2 else 1),
            cb(cls.CB_G0, per if (gmask & 1) else 1),
            cb(cls.CB_G1, per if (gmask & 2) else 1),
            cb(cls.CB_G2, per if (gmask & 4) else 1),
            cb(cls.CB_RES, per if residual is not None else 1),
            cb(cls.CB_SCALER, 1),
            cb(cls.CB_SQ, per, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_PART, nin, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_RECV, nin * RT, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_INV, nin, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_STAGE, 1, page=N * block, fmt=ttnn.float32),
            cb(cls.CB_SEND, 1, page=block, fmt=ttnn.float32),
            cb(cls.CB_OUT0, per),
            cb(cls.CB_OUT1, per if nout > 1 else 1),
            cb(cls.CB_OUT2, per if nout > 2 else 1),
        ]
        x1_t = x2 if x2 is not None else x
        g_t = [g if g is not None else x for g in (gamma_list + [None, None, None])[:3]]
        res_t = residual if residual is not None else x
        out_t = (outs + [outs[0], outs[0]])[:3]
        common = [per, N, RT, nin, nout, int(residual is not None), gmask]
        reader_ct = [cls.CB_X0, cls.CB_X1, cls.CB_G0, cls.CB_G1, cls.CB_G2, cls.CB_RES, cls.CB_SCALER, cls.CB_RECV,
                     cls.CB_STAGE] + common + [cls.SEM]  # fmt: skip
        for t in (x, x1_t, g_t[0], g_t[1], g_t[2], res_t):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [cls.CB_PART, cls.CB_SEND, cls.CB_STAGE, cls.CB_OUT0, cls.CB_OUT1, cls.CB_OUT2, per, N, nin, nout,
                     cls.SEM]  # fmt: skip
        for t in out_t:
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cls.CB_X0, cls.CB_X1, cls.CB_G0, cls.CB_G1, cls.CB_G2, cls.CB_RES, cls.CB_SCALER, cls.CB_SQ,
                      cls.CB_PART, cls.CB_RECV, cls.CB_INV, cls.CB_OUT0, cls.CB_OUT1, cls.CB_OUT2] + common  # fmt: skip
        # eps, 1/W and the scalars are RUNTIME args of the compute kernel: the learned layer scalar differs per
        # layer, and as a compile-time arg it made every layer its own JIT build (30 x 3 TRISC compiles on the
        # cold decode path, again on prefill); one binary per form and split now serves every layer.
        compute_rt_vals = [_f32_bits(eps), _f32_bits(1.0 / W)] + [_f32_bits(s) for s in (scalars + [1.0, 1.0])[:3]]
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        sems = [ttnn.SemaphoreDescriptor(cls.SEM, ttnn.CoreType.WORKER, core_set, 0)]
        io = [x] + ([x2] if x2 is not None else []) + [g for g in gamma_list if g is not None]
        io += ([residual] if residual is not None else []) + outs
        for p in range(passes):
            reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            first, last = p * groups, min(RT_IN, (p + 1) * groups)
            for g_idx, t in enumerate(range(first, last)):
                coords = []
                for c in noc[g_idx * N : (g_idx + 1) * N]:  # the exchange stays within the row tile's group
                    coords += [c.x, c.y]
                for i in range(N):
                    core = cores[g_idx * N + i]
                    c0 = t * cols_t + i * per  # tile index in the [RT_IN, cols_t] tile grid of x / residual / outputs
                    reader_rt[core.x][core.y] = [
                        x.buffer_address(), x1_t.buffer_address(), g_t[0].buffer_address(), g_t[1].buffer_address(),
                        g_t[2].buffer_address(), res_t.buffer_address(), c0, i * per,
                    ]  # fmt: skip
                    writer_rt[core.x][core.y] = [
                        out_t[0].buffer_address(), out_t[1].buffer_address(), out_t[2].buffer_address(), c0, i
                    ] + coords  # fmt: skip
                    compute_rt[core.x][core.y] = list(compute_rt_vals)
            active = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores[: (last - first) * N]])
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "reader_fused_norm_decode.cpp"),
                    core_ranges=active,
                    compile_time_args=reader_ct,
                    runtime_args=reader_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "writer_fused_norm_decode.cpp"),
                    core_ranges=active,
                    compile_time_args=writer_ct,
                    runtime_args=writer_rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "compute_fused_norm_decode.cpp"),
                    core_ranges=active,
                    compile_time_args=compute_ct,
                    runtime_args=compute_rt,
                    config=compute_cfg,
                ),
            ]
            program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
            ttnn.generic_op(io, program)
        return outs
