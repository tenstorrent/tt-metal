# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill (M = 32..2048 rows) for the A3 exact-ternary sites (BONSAI_NATIVE=pq2_0).

The A3 op is M=1. For prefill, each projection is
    y = H(D x) @ dequant(codes, scales)
  - dequant(): an on-device kernel reads A3's own weight tensors (bfp2 codes, bank-shuffled K-sticks;
    fp32 scales in a3.scale_rows layout) and writes codes * scale as bfp8_b into a per-call
    interleaved DRAM scratch [K, N]. No extra resident weight memory, no extra cache.
  - rotate(): x' = H(D x) for M rows as one batched matmul on x^T: x'^T block b = (D_b H)^T x_b^T
    (splitting x^T's K rows into 1024-row blocks is a free view).
  - then the stock ttnn.linear the dense path uses (LoFi, fp32 acc).
bfp8 of codes * fp16 scale is the dense checkpoint re-quantized the way the dense demo does it.
"""

from __future__ import annotations

import torch

import ttnn

from . import op as a3
from .pmm import MAX_ROWS, pmm

import os

_FP32_DEST = os.environ.get("BONSAI_DQ_FP32", "1") == "1"
_DQ_FID = ttnn.MathFidelity.HiFi4
KDIR = "models/demos/blackhole/bonsai2/native/a3/kernels/"

_HIFI4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                          packer_l1_acc=False)
_LOFI = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True,
                                         packer_l1_acc=False)


def dequant(w, s, k: int, n: int, gu: bool = False, dtype=ttnn.bfloat8_b, fidelity=None):
    """w, s: a3.weights_for_device output for codes [K, N_pad] (gu: (gate, up) column pairs).
    n: real output columns (gu: per half). Returns [1, 1, K, n] DRAM tensor(s) (gu: (gate, up))."""
    device = w.device()
    banks = device.dram_grid_size().x
    kt = k // 32
    assert kt % 32 == 0 and n % 32 == 0
    ng = kt // 32
    s_tiles = (ng + 3) // 4
    cpb = w.memory_config().shard_spec.shape[1] // 32  # weight columns per bank
    wtb = w.get_tile().get_tile_size(w.dtype)
    ncols = (2 if gu else 1) * n // 32  # real weight columns (padding columns sit past these)
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, k, n]), dtype, ttnn.TILE_LAYOUT, device,
                                           ttnn.DRAM_MEMORY_CONFIG) for _ in range(2 if gu else 1)]
    otb = ttnn.Tile([32, 32]).get_tile_size(dtype)

    grid = device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    units = ncols * ng
    per = -(-units // len(cores))
    cores = cores[: -(-units // per)]
    crs = a3.core_set(cores)
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    o2 = outs[-1].buffer_address()
    for i, c in enumerate(cores):
        u0, nu = i * per, min(per, units - i * per)
        rd[c.x][c.y] = [w.buffer_address(), s.buffer_address(), u0, nu]
        cp[c.x][c.y] = [u0, nu]
        wr[c.x][c.y] = [outs[0].buffer_address(), o2, u0, nu]

    def cb(idx, fmt, page, ntiles):
        return ttnn.CBDescriptor(total_size=ntiles * page, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)])

    kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
        kernel_source=KDIR + src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
        compile_time_args=ct, runtime_args=rt, config=cfg)
    prog = ttnn.ProgramDescriptor(
        kernels=[
            kd("dequant_reader.cpp", [ng, s_tiles, cpb, wtb, ncols], rd, ttnn.ReaderConfigDescriptor()),
            kd("dequant_writer.cpp", [ng, n // 32, int(gu), otb, int(os.environ.get("BONSAI_DQ_SKIPW", "0")), ncols], wr, ttnn.WriterConfigDescriptor()),
            kd("dequant_compute.cpp", [ng, 4 if _FP32_DEST else 8, ncols], cp, ttnn.ComputeConfigDescriptor(
                math_fidelity=fidelity or _DQ_FID, fp32_dest_acc_en=_FP32_DEST, math_approx_mode=False)),
        ],
        semaphores=[],
        cbs=[cb(0, w.dtype, wtb, 64), cb(1, ttnn.float32, 4096, 2), cb(16, dtype, otb, 64)])
    ttnn.generic_op([w, s] + outs, prog)
    return tuple(outs) if gu else outs[0]


class Rotator:
    """x' = H(D x) for [1, 1, S, K] rows. One [1, K/1024, 1024, 1024] tensor per width: block b = (D_b H)^T."""

    def __init__(self, device, had, widths=(5120, 6144, 17408)):
        self.mt = {}
        for k in widths:
            mb = a3.hadamard_blocks(had, k).reshape(k // 1024, 1024, 1024)  # block b = D_b H (+-1/32)
            self.mt[k] = ttnn.from_torch(mb.transpose(1, 2).contiguous().unsqueeze(0), dtype=ttnn.bfloat8_b,
                                         layout=ttnn.TILE_LAYOUT, device=device,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def __call__(self, x, mem=ttnn.DRAM_MEMORY_CONFIG):
        s, k = x.shape[-2], x.shape[-1]
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        xt = ttnn.transpose(ttnn.reshape(x, [1, 1, s, k]), -2, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xt4 = ttnn.reshape(xt, [1, k // 1024, 1024, s])
        # S=128: small M blocks beat auto, bit-identical (sweep_rot.py, chip 3, ms): K=17408 0.140 vs 0.522
        # (per_core_M 8), K=5120 0.071 vs 0.170 and K=6144 0.086 vs 0.201 (per_core_M 4). At 2048 auto beats
        # every reuse config tried.
        pc = None
        if s == 128:
            g = x.device().compute_with_storage_grid_size()
            pcm = 4 if (k // 1024) * 8 <= g.x * g.y else 8
            pc = ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(g.x, g.y), in0_block_w=4,
                                                        out_subblock_h=pcm // 4, out_subblock_w=4 if pcm == 4 else 2,
                                                        per_core_M=pcm, per_core_N=4)
        rt = ttnn.matmul(self.mt[k], xt4, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                         dtype=ttnn.bfloat16, program_config=pc)  # dtype: default would be in0's bfp8
        ttnn.deallocate(xt)
        y = ttnn.transpose(ttnn.reshape(rt, [1, 1, k, s]), -2, -1, memory_config=mem)
        ttnn.deallocate(rt)
        return y


_rot = {}


class _Dispatch:
    """rot.KRotator (one kernel) for big S*K, the stock Rotator for small ones. rot.py self-check, chip 3, ms:
    17408: 0.87 vs 3.26 (S=2048), 0.63 vs 1.12 (S=640); 5120/6144 at 2048: 0.75/0.66 vs 0.99/1.17; stock wins
    at S=128 and for 5120/6144 at 640. Same precision (PCC 0.9999986 vs float64 for both)."""

    def __init__(self, device, had):
        from .rot import KRotator

        self.stock, self.kern = Rotator(device, had), KRotator(device, had)

    def __call__(self, x, mem=ttnn.DRAM_MEMORY_CONFIG):
        big = x.shape[-2] * x.shape[-1] >= 10_000_000 and os.environ.get("BONSAI_PF_KROT", "1") == "1"
        return (self.kern if big else self.stock)(x, mem)


def rotator(shared):
    if id(shared) not in _rot:
        _rot[id(shared)] = _Dispatch(shared.device, shared.had)
    return _rot[id(shared)]


def wide_pc(s: int, k: int, n: int):
    """2D mcast config for the wide prefill matmuls at S > 512 on the 8x10 prefill grid: whole per-core N
    in one out block, out_block_h=1, in0_block_w=4. [2048,5120]@[5120,17408] bfp8: 3.80 ms vs 12.12 ms with
    ttnn's auto config (sweep_gate.py, chip 3). None (auto) for short S, where auto wins (0.30 vs 0.54 ms at 128)."""
    gx, gy = 8, 10
    pcn = -(-n // 32 // gx)
    if s <= 512 or (k // 32) % 4 or pcn > 68:
        return None
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=4, out_subblock_h=1,
        out_subblock_w=max(d for d in (1, 2, 4) if pcn % d == 0), out_block_h=1, out_block_w=pcn,
        per_core_M=-(-s // 32 // gy), per_core_N=pcn, transpose_mcast=False, fused_activation=None, fuse_batch=False)


def _use_pmm(xr) -> bool:
    """Short prefill rows (S <= 128): pmm.py, A3 at M>1, dequantizes in L1 with no scratch. S=128, chip 3, ms vs
    dequant+matmul: gate|up 0.93 vs 1.95, down 0.85 vs 1.16, gdn_in 0.57 vs 1.08, gdn_out 0.55 vs 0.76,
    PCC 0.99999 vs float64 (0.99986 on the scratch path). BONSAI_PF_PMM=0 turns it off."""
    return xr.shape[-2] <= MAX_ROWS and os.environ.get("BONSAI_PF_PMM", "1") == "1"


def _pf_dtype():
    """BONSAI_PF_W=bfp4: every prefill scratch except MLP gate/up (BONSAI_PF_GU) in bfp4 instead of bfp8."""
    return ttnn.bfloat4_b if os.environ.get("BONSAI_PF_W") == "bfp4" else ttnn.bfloat8_b


def prefill_linear(tl, x, mem=ttnn.DRAM_MEMORY_CONFIG, program_config=None):
    """Prefill for any a3.linear.TernaryLinear site: x [.., S, K] -> [1, 1, S, N] bf16."""
    xr = rotator(tl.sh)(x)
    if _use_pmm(xr):
        y = pmm(xr, tl.w, tl.s, tl.k, tl.n)
        ttnn.deallocate(xr)
        if mem != ttnn.DRAM_MEMORY_CONFIG:
            y2 = ttnn.to_memory_config(y, mem)
            ttnn.deallocate(y)
            y = y2
        return ttnn.reshape(y, list(x.shape)[:-1] + [tl.n])
    wd = dequant(tl.w, tl.s, tl.k, tl.n, dtype=_pf_dtype())
    pc = program_config or wide_pc(xr.shape[-2], tl.k, tl.n)
    y = ttnn.linear(xr, wd, compute_kernel_config=_LOFI, memory_config=mem, program_config=pc)
    ttnn.deallocate(wd)
    ttnn.deallocate(xr)
    return ttnn.reshape(y, list(x.shape)[:-1] + [tl.n])


def prefill_mlp(mlp, x, mem=ttnn.DRAM_MEMORY_CONFIG, down_pc=None):
    """Prefill for a3.mlp.TernaryMLP, same op sequence as the dense Qwen36MLP prefill."""
    sh = mlp.sh
    rot = rotator(sh)
    k, h = x.shape[-1], mlp.w_d.shape[-2]  # w_d is [K=hidden, N=dim]
    xr = rot(x)
    # BONSAI_PF_GU=bfp4: gate/up scratch in bfp4 (the dense path's gate/up dtype): 0.9 vs 1.3 ms dequant per layer.
    if _use_pmm(xr):
        g, u = pmm(xr, mlp.w_gu, mlp.s_gu, k, h, gu=True)
        ttnn.deallocate(xr)
    else:
        gu_dt = ttnn.bfloat4_b if os.environ.get("BONSAI_PF_GU") == "bfp4" else ttnn.bfloat8_b
        wg, wu = dequant(mlp.w_gu, mlp.s_gu, k, h, gu=True, dtype=gu_dt)
        pc = wide_pc(xr.shape[-2], k, h)
        g = ttnn.linear(xr, wg, compute_kernel_config=_LOFI, memory_config=mem, program_config=pc)
        u = ttnn.linear(xr, wu, compute_kernel_config=_LOFI, memory_config=mem, program_config=pc)
        for t in (wg, wu, xr):
            ttnn.deallocate(t)
    hid = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
    ttnn.deallocate(g)
    ttnn.deallocate(u)
    hr = rot(hid)
    ttnn.deallocate(hid)
    if _use_pmm(hr):
        y = pmm(hr, mlp.w_d, mlp.s_d, h, k)
    else:
        wd = dequant(mlp.w_d, mlp.s_d, h, k, dtype=_pf_dtype())
        y = ttnn.linear(hr, wd, compute_kernel_config=_LOFI, memory_config=mem, program_config=down_pc)
        ttnn.deallocate(wd)
    ttnn.deallocate(hr)
    return ttnn.reshape(y, list(x.shape))
