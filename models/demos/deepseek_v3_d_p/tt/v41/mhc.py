# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 hyper-connection residual (graph nodes B1, B2, B17, B18, B23, F1).

V4.1 lags the collapse by one sublayer (``inference/model.py`` ``Block.forward``): attention collapses the
streams with the ``pre`` the previous block's FFN produced (a one-hot on copy 0 before the first block), the
FFN collapses with the ``pre`` split before attention, and the block hands the ``pre`` split before its FFN
to the next block. After the last block the model collapses with that last ``pre``; there is no head mix.

Streams are ``[1, 1, S/sp, hc_mult * hidden/tp]`` fp32 (per chip: its hidden slice of every stream);
``pre_mix`` is ``[1, 1, S/sp, hc_mult]`` fp32. The sublayers run in bf16: the collapse accumulates in fp32 and
packs its output to bf16 (``Block.hc_pre``'s ``y.to(x.dtype)``), and hc_post takes the bf16 sublayer output as
an exact fp32 term, so no typecast op runs around a sublayer.
"""

import struct

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

SUBLAYER_DTYPE = ttnn.bfloat16


def mhc_config(config) -> MHCConfig:
    return MHCConfig(
        dim=config.EMB_SIZE,
        n=config.HC_MULT,
        sinkhorn_iters=config.HC_SINKHORN_ITERS,
        eps=config.HC_EPS,
        norm_eps=config.RMS_NORM_EPS,
    )


def initial_pre_mix(mesh_device, config, tokens: int) -> ttnn.Tensor:
    """One-hot on copy 0 for the first block's attention, token-sharded over SP."""
    pre = torch.zeros(1, 1, tokens, config.HC_MULT)
    pre[..., 0] = 1.0
    return ttnn.from_torch(
        pre,
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), dims=(2, None)),
    )


class _V41Site(TtMHCWrap):
    """``TtMHCWrap`` with its split projection, collapse and hc_post as one fused op each (one pass over the streams).

    collapse and hc_post are bit-identical to the composite path (with its typecasts to and from the bf16 sublayer);
    the projection accumulates the same fp32 matmul and an fp32 sum of squares, and TP-sums both in one all-reduce."""

    def __init__(self, device, cfg, fn, base, scale, **kwargs):
        # fn_T gets zero columns up to the tile width: the fused projection puts mean(x^2) + eps in column mix_hc
        self.mix_hc = fn.shape[0]
        assert self.mix_hc < 32, self.mix_hc
        fn = torch.cat([fn.float(), torch.zeros(32 - self.mix_hc, fn.shape[1])])
        super().__init__(device, cfg, fn, base, scale, **kwargs)

    def project(self, x):
        tp_sum = self._tp_sum if self.tp_factor > 1 else None
        return fused_rms_project(x, self.fn_T, self.mix_hc, self.norm_eps, self.tp_factor, tp_sum)

    def collapse(self, x, pre, dtype=SUBLAYER_DTYPE):
        return fused_collapse(x, pre, self.n, dtype)

    def hc_post(self, x, residual, post, comb):
        return fused_hc_post(x, residual, post, comb, self.n)


class TtV41HyperConnections(LightweightModule):
    """The two hyper-connection sites of one block, applied around caller-provided sublayers."""

    def __init__(self, mesh_device, config, hc_attn, hc_ffn):
        """``hc_attn`` / ``hc_ffn``: the checkpoint's ``(fn [24, 4*dim], base [24], scale [3])`` per site."""
        cfg = mhc_config(config)
        topology = per_axis_topology()[1]  # the TP axis (tp_axis=1) of the opened fabric
        self.attn_site = _V41Site(mesh_device, cfg, *hc_attn, tp_axis=1, topology=topology)
        self.ffn_site = _V41Site(mesh_device, cfg, *hc_ffn, tp_axis=1, topology=topology)

    def forward(self, x, pre_mix, attention, ffn):
        """(streams, incoming pre_mix) -> (streams, pre_mix for the next block).

        ``attention`` and ``ffn`` map a collapsed ``[1, 1, S/sp, hidden/tp]`` bf16 stream to the same shape;
        the norms are theirs."""
        attn_pre, attn_post, attn_comb = self.attn_site.split(x)
        h = attention(self.attn_site.collapse(x, pre_mix))
        x = self.attn_site.hc_post(h, x, attn_post, attn_comb)

        ffn_pre, ffn_post, ffn_comb = self.ffn_site.split(x)
        h = ffn(self.ffn_site.collapse(x, attn_pre))
        x = self.ffn_site.hc_post(h, x, ffn_post, ffn_comb)
        return x, ffn_pre

    def final_collapse(self, x, pre_mix):
        """After the last block: collapse the streams with its FFN ``pre`` -> ``[1, 1, S/sp, hidden/tp]`` fp32."""
        return self.ffn_site.collapse(x, pre_mix, ttnn.float32)


_KERNEL_DIR = "models/demos/deepseek_v3_d_p/tt/v41/kernels"
_CB_IN, _CB_COEF, _CB_CSRC, _CB_H, _CB_OUT = 0, 1, 2, 3, 16
_FP32_TILE_BYTES = 32 * 32 * 4
_BF16_TILE_BYTES = 32 * 32 * 2
_MAX_BLOCK_UNITS = 6  # units per block (a divisor of the stream width in tiles); + 2 must fit the 8 fp32 DST tiles


def _stream_mix(streams, n: int, coef_srcs, table, outputs: int, x=None, dtype=ttnn.float32) -> ttnn.Tensor:
    """One fused op: ``out_j = sum_k coef[j][k] * in_k`` per token, fp32, ``in = [x?] + streams 0..n-1``.

    ``streams`` [1, 1, T, n*C] fp32, ``x`` optional [1, 1, T, C] fp32 or bf16 (exact in fp32), ``coef_srcs`` one or
    two [1, 1, T, <=32] fp32 tensors and ``table[j][k] = (src, col)``: term k of output j is weighted by column
    ``col`` of ``coef_srcs[src]``. Returns [1, 1, T, outputs*C] ``dtype`` (fp32, or bf16: the fp32 result rounded
    to nearest-even, as ``ttnn.typecast``; output j at columns [j*C, (j+1)*C)). The terms of each output are accumulated in fp32 in order
    k = 0, 1, ... as one multiply and then addcmuls, like ``tt_mhc._mix``."""
    tensors = [streams] + ([x] if x is not None else []) + list(coef_srcs)
    for t in tensors:
        allowed = (ttnn.float32, ttnn.bfloat16) if t is x else (ttnn.float32,)
        assert t.dtype in allowed and t.layout == ttnn.TILE_LAYOUT, (t.dtype, t.layout)
        assert not t.memory_config().is_sharded(), "the mHC mix takes interleaved tensors"
    assert dtype in (ttnn.float32, ttnn.bfloat16), dtype
    tokens, width = streams.shape[-2], streams.shape[-1]
    assert tokens % 32 == 0 and width % (32 * n) == 0, (tokens, width, n)
    ct = width // n // 32
    k_terms = n + (x is not None)
    assert x is None or tuple(x.shape)[-2:] == (tokens, width // n), (tuple(x.shape), tokens, width // n)
    assert len(coef_srcs) in (1, 2) and all(c.shape[-2] == tokens and c.shape[-1] <= 32 for c in coef_srcs)
    assert len(table) == outputs and all(len(row) == k_terms for row in table)
    codes = [src << 8 | col for row in table for src, col in row]

    device = streams.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, tokens, outputs * (width // n)]),
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        streams.memory_config(),
    )
    # whole blocks of one tile row per reader barrier: every CB reservation stays contiguous
    block = max(b for b in range(1, _MAX_BLOCK_UNITS + 1) if ct % b == 0)
    blocks = tokens // 32 * ct // block
    grid = device.compute_with_storage_grid_size()
    num_cores = min(grid.x * grid.y, blocks)
    base, extra = divmod(blocks, num_cores)
    last = ttnn.CoreCoord((num_cores - 1) % grid.x, (num_cores - 1) // grid.x)
    ranges = []
    if last.y > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, last.y - 1)))
    ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, last.y), last))
    cores = ttnn.CoreRangeSet(ranges)

    c0 = coef_srcs[0]
    c1 = coef_srcs[1] if len(coef_srcs) > 1 else c0
    xt = x if x is not None else streams
    addrs = [streams.buffer_address(), xt.buffer_address(), c0.buffer_address(), c1.buffer_address()]
    reader_args, writer_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for i in range(num_cores):
        cx, cy = i % grid.x, i // grid.x
        count = (base + (i < extra)) * block
        reader_args[cx][cy] = addrs + [start, count]
        writer_args[cx][cy] = [out.buffer_address(), start, count]
        compute_args[cx][cy] = [start, count]
        start += count

    def cb(index: int, tiles: int, fmt=ttnn.float32) -> ttnn.CBDescriptor:
        size = _FP32_TILE_BYTES if fmt == ttnn.float32 else _BF16_TILE_BYTES
        page = ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=size)
        return ttnn.CBDescriptor(total_size=tiles * size, core_ranges=cores, format_descriptors=[page])

    accessor = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    # fp32 operands unpack straight to DST; a bf16 x goes through SrcA (exact: bf16 fits its 19-bit datums)
    modes[_CB_IN] = modes[_CB_COEF] = ttnn.UnpackToDestMode.UnpackToDestFp32
    x_fp32 = x is None or x.dtype == ttnn.float32
    if x_fp32:
        modes[_CB_H] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = modes
    cb_x = _CB_H if x is not None else _CB_IN  # without x the kernels' x handle aliases the streams' CB, unused
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/mhc_mix_reader.cpp",
            core_ranges=cores,
            compile_time_args=[
                _CB_IN,
                _CB_COEF,
                _CB_CSRC,
                ct,
                n,
                int(x is not None),
                outputs,
                len(coef_srcs),
                block,
                cb_x,
            ]
            + codes
            + accessor(streams)
            + accessor(xt)
            + accessor(c0)
            + accessor(c1),
            runtime_args=reader_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/mhc_mix_writer.cpp",
            core_ranges=cores,
            compile_time_args=[_CB_OUT, ct, outputs, block] + accessor(out),
            runtime_args=writer_args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/mhc_mix_compute.cpp",
            core_ranges=cores,
            compile_time_args=[_CB_IN, _CB_COEF, _CB_OUT, ct, n, outputs, block, int(x is not None), cb_x, int(x_fp32)]
            + [int(dtype == ttnn.bfloat16)],
            runtime_args=compute_args,
            config=compute_config,
        ),
    ]
    cbs = [
        cb(_CB_IN, 2 * block * n),
        cb(_CB_COEF, 2 * outputs * k_terms),
        cb(_CB_CSRC, len(coef_srcs)),
        cb(_CB_OUT, 2 * block * outputs, dtype),
    ]
    if x is not None:
        cbs.append(cb(_CB_H, 2 * block, x.dtype))
    return ttnn.generic_op(tensors + [out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))


def fused_collapse(x, pre, n: int, dtype=ttnn.float32) -> ttnn.Tensor:
    """``sum_i pre_i * x_i``: [1, 1, T, n*C] fp32 streams, [1, 1, T, n] fp32 ``pre`` -> [1, 1, T, C] ``dtype``
    (accumulated in fp32; bf16 rounds the fp32 sum once, as a typecast of the fp32 collapse)."""
    return _stream_mix(x, n, [pre], [[(0, i) for i in range(n)]], outputs=1, dtype=dtype)


def fused_hc_post(h, residual, post, comb, n: int) -> ttnn.Tensor:
    """``new_j = post_j * h + sum_i comb[i, j] * residual_i`` -> [1, 1, T, n*C] fp32 (``TtMHCWrap.hc_post``); ``h``
    fp32 or bf16 (the bf16 sublayer output, taken exactly)."""
    table = [[(0, j)] + [(1, i * n + j) for i in range(n)] for j in range(n)]
    return _stream_mix(residual, n, [post, comb], table, outputs=n, x=h)


_PROJ_BK = 8  # stream/weight tiles per read barrier (the largest divisor of the width in tiles up to this)
_PROJ_X_BLOCKS = 6  # x blocks buffered per core (measured: x reads, not compute, bound the op)
_CB_X, _CB_XSQ, _CB_W, _CB_CONST, _CB_MIX, _CB_HI, _CB_LO = 0, 1, 2, 3, 4, 5, 6


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def fused_project(x, fn_T, mix_col: int, width: int, eps_part: float) -> ttnn.Tensor:
    """One pass over the streams: [1, 1, T, K] fp32 ``x``, [1, 1, K, 32] fp32 ``fn_T`` (zero from ``mix_col`` on)
    -> [1, 1, T, 32] fp32 with ``x @ fn_T`` in columns < ``mix_col`` and ``sum(x^2) / width + eps_part`` in
    column ``mix_col``: this chip's partial mixes and share of ``mean(x^2) + eps``, summed across TP by the caller."""
    for t in (x, fn_T):
        assert t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT, (t.dtype, t.layout)
        assert not t.memory_config().is_sharded(), "the mHC projection takes interleaved tensors"
    tokens, k = x.shape[-2], x.shape[-1]
    assert tokens % 32 == 0 and k % 32 == 0, (tokens, k)
    assert tuple(fn_T.padded_shape)[-2:] == (k, 32) and mix_col < 32, (tuple(fn_T.padded_shape), k, mix_col)
    kt, rows = k // 32, tokens // 32
    bk = max(b for b in range(1, _PROJ_BK + 1) if kt % b == 0)

    device = x.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, tokens, 32]), ttnn.float32, ttnn.TILE_LAYOUT, device, x.memory_config()
    )
    grid = device.compute_with_storage_grid_size()
    num_cores = min(grid.x * grid.y, rows)
    base, extra = divmod(rows, num_cores)
    last = ttnn.CoreCoord((num_cores - 1) % grid.x, (num_cores - 1) // grid.x)
    ranges = []
    if last.y > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, last.y - 1)))
    ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, last.y), last))
    cores = ttnn.CoreRangeSet(ranges)

    reader_args, writer_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for i in range(num_cores):
        cx, cy = i % grid.x, i // grid.x
        count = base + (i < extra)
        reader_args[cx][cy] = [x.buffer_address(), start, count]
        writer_args[cx][cy] = [fn_T.buffer_address(), out.buffer_address(), start, count]
        compute_args[cx][cy] = [count]
        start += count

    def cb(index: int, tiles: int) -> ttnn.CBDescriptor:
        page = ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=_FP32_TILE_BYTES)
        return ttnn.CBDescriptor(total_size=tiles * _FP32_TILE_BYTES, core_ranges=cores, format_descriptors=[page])

    accessor = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    # exact fp32 copies into DST: the squares' inputs and the mixes reload
    modes[_CB_XSQ] = modes[_CB_MIX] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = modes
    # eps_part lands on all 32 entries of the row's partial sums of squares, which the E matmul sums
    compute_ct = [
        _CB_X,
        _CB_XSQ,
        _CB_W,
        _CB_CONST,
        _CB_MIX,
        _CB_HI,
        _CB_LO,
        _CB_OUT,
        kt,
        bk,
        _f32_bits(1.0 / width),
        _f32_bits(eps_part / 32),
    ]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/mhc_proj_reader.cpp",
            core_ranges=cores,
            compile_time_args=[_CB_X, _CB_XSQ, _CB_CONST, kt, bk, mix_col] + accessor(x),
            runtime_args=reader_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/mhc_proj_writer.cpp",
            core_ranges=cores,
            compile_time_args=[_CB_W, _CB_OUT, kt, bk] + accessor(fn_T) + accessor(out),
            runtime_args=writer_args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNEL_DIR}/mhc_proj_compute.cpp",
            core_ranges=cores,
            compile_time_args=compute_ct,
            runtime_args=compute_args,
            config=compute_config,
        ),
    ]
    x_page = lambda index: ttnn.CBFormatDescriptor(
        buffer_index=index, data_format=ttnn.float32, page_size=_FP32_TILE_BYTES
    )
    cbs = [
        # cb_x and cb_xsq: one buffer, two views (the matmul operand and the fp32 unpack-to-DST input)
        ttnn.CBDescriptor(
            total_size=_PROJ_X_BLOCKS * bk * _FP32_TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[x_page(_CB_X), x_page(_CB_XSQ)],
        ),
        cb(_CB_W, 2 * bk),
        cb(_CB_CONST, 1),
        cb(_CB_MIX, 1),
        cb(_CB_HI, 1),
        cb(_CB_LO, 1),
        cb(_CB_OUT, 2),
    ]
    return ttnn.generic_op([x, fn_T, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))


def fused_rms_project(x, fn_T, mix_col: int, norm_eps: float, tp_factor: int = 1, tp_sum=None) -> ttnn.Tensor:
    """``tt_mhc._project`` in one pass over the streams and one TP reduction: [1, 1, T, K] -> [1, 1, T, 32] with
    ``RMSNorm(x) @ fn`` in columns < ``mix_col`` (``tp_sum`` sums the partial mixes and mean of squares together,
    replicated on every TP chip). Columns >= ``mix_col`` are scratch the Sinkhorn op does not read."""
    # every core reads all of the (small) weight: from L1, not 80x from DRAM
    w = ttnn.to_memory_config(fn_T, ttnn.L1_MEMORY_CONFIG)
    r = fused_project(x, w, mix_col, x.shape[-1] * tp_factor, norm_eps / tp_factor)
    ttnn.deallocate(w)
    if tp_sum is not None:
        r = tp_sum(r)
    rs = ttnn.rsqrt(r)  # column mix_col: rsqrt(mean(x^2) + eps); the other columns are scratch
    # every column times column mix_col of rs, per token: no column slice (untilize + slice + tilize)
    return _stream_mix(r, 1, [rs], [[(0, mix_col)]], outputs=1)
