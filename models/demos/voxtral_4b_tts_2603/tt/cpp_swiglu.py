# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on the acoustic SwiGLU `silu(x @ w1) * (x @ w3)`, through ttnn.generic_op.

The stock path is two 1-D in0-multicast matmuls (each streams its own bf8_b weight as single-tile
reads spread over every DRAM bank, and multicasts the whole activation to every core), then a third
op for silu * product. At the acoustic height (3-6 tile rows) each matmul is weight-stream bound.

This is one program. The gate and up weights are fused at build into ONE bf8_b tensor with their
tile columns interleaved g(0) u(0) g(1) u(1) ..., width-sharded over the DRAM banks. Each core owns
PN output columns of both projections; those 2*PN weight tiles are contiguous in one bank's shard
row, so the core streams its weight as one read per K row from one bank (on one RISC / NoC), reads
the L1-resident activation one K block at a time (on the other), accumulates both products into an
interleaved gate/up DEST subblock, and after the last K block applies silu and the product in DEST,
writing only the gated tile. Neither the gate, the up nor a multiply output round-trips memory.

Measured 2026-09-26 at 96 x 3072 x 9216 (x2), 96 cores: stock pair + multiply ~208 us a call, this
187 us; device 302.9 -> 300.2 ms, trace acoustic 27.9 -> 27.3 ms, PCC 0.999778 -> 0.999564. The
layout and blocking both mattered: interleaved single-tile weight reads ran ~290 GB/s (210 us); a
bank-sharded weight with 12 ADJACENT cores per bank was slower (259 us), spread c -> bank c % banks
fixed it (205 us); 3 wide K blocks cost a 70 us pipeline fill/tail (258 us); 12 blocks of 8, 4-deep,
RT x 2 subblocks gave 187 us. Compute still trails the weight stream by ~26 us.

Off unless VOXTRAL_CPP_SWIGLU=1: its gate/up weight is bf8_b and its output bf16, below the acoustic stage's accuracy bar. generic_op hashes the runtime-arg COUNT, not the values, so
`custom_program_hash` carries the buffer addresses (voxtral_mini's cpp_matmul plumbing).
"""

from __future__ import annotations

import os
import pathlib

import torch

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_swiglu_kernels"
_READER_W = str(_DIR / "reader_w.cpp")
_READER_X = str(_DIR / "reader_x_writer.cpp")
_COMPUTE = str(_DIR / "compute.cpp")

_TILE = 32
_CB_BUDGET = 900 * 1024
_DEST_TILES = 8
# Small K blocks, deep weight buffering: compute starts after one block lands and trails the stream
# by one block, and at RT x 2 subblocks a spill/reload round is cheap next to a block's math.
_KB_DEPTH = ((8, 4), (8, 3), (8, 2), (6, 3), (4, 3), (4, 2), (2, 2), (1, 2))
_BF16 = 2048
# The fused gate/up weight's format: bf4_b halves the bytes the weight-stream-bound kernel reads
# (576 B a tile: 512 B of mantissas + 64 B of shared exponents), as it did for the down projection.
_W_DTYPE, _BF8 = ttnn.bfloat8_b, 1088


def _log(msg):
    """Opt-in dispatch trace (VOXTRAL_CPP_SWIGLU_LOG=path): proves the kernel ran under a gate."""
    path = os.environ.get("VOXTRAL_CPP_SWIGLU_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(msg + "\n")


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_SWIGLU", "0") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_swiglu: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


class Fused:
    """The gate/up weights as one bank-sharded tensor, plus the core split it was laid out for."""

    def __init__(self, tensor, k, n, pn, ncores, banks):
        self.tensor, self.k, self.n, self.pn, self.ncores, self.banks = tensor, k, n, pn, ncores, banks
        self.per_bank = ncores // banks
        self.shard_tiles = 2 * n // _TILE // banks


def fuse(w1, w3, device):
    """`w1`, `w3`: torch `[K, N]` (already transposed / gamma-folded). None when the shape has no split."""
    if not enabled():
        return None
    k, n = int(w1.shape[0]), int(w1.shape[1])
    if k % _TILE or n % _TILE:
        return None
    grid = device.compute_with_storage_grid_size()
    ncores_max = int(grid.x) * int(grid.y)
    banks = int(device.dram_grid_size().x)
    nt = n // _TILE
    pn = next((p for p in range(1, nt + 1) if nt % p == 0 and nt // p <= ncores_max and (nt // p) % banks == 0), None)
    _log(f"fuse {k}x{n} banks={banks} grid={ncores_max} pn={pn}")
    if pn is None:
        return None
    ncores = nt // pn
    inter = torch.stack([w1.float().reshape(k, nt, _TILE), w3.float().reshape(k, nt, _TILE)], dim=2)
    # Core c streams from bank c % banks, so neighbouring cores load different banks: shard s holds
    # the column blocks of cores s, s + banks, s + 2 * banks, ... in that order.
    order = [q * banks + s for s in range(banks) for q in range(ncores // banks)]
    inter = inter.reshape(k, ncores, 2 * pn * _TILE)[:, order].reshape(k, 2 * n)
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            [k, 2 * n // banks],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    mapper = ttnn.ReplicateTensorToMesh(device) if device.__class__.__name__ == "MeshDevice" else None
    t = ttnn.from_torch(
        inter.contiguous(),
        dtype=_W_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
        mesh_mapper=mapper,
    )
    return Fused(t, k, n, pn, ncores, banks)


class _Plan:
    def __init__(self, device, mt, fused):
        grid = device.compute_with_storage_grid_size()
        self.gx = int(grid.x)
        self.mt, self.kt, self.nt = mt, fused.k // _TILE, fused.n // _TILE
        self.pn, self.ncores = fused.pn, fused.ncores
        # DEST in full-sync fp32 holds 8 tiles: an RT x 2 subblock (RT rows of one gate/up pair).
        w = 2 * self.pn
        self.rt = max(r for r in range(1, _DEST_TILES // 2 + 1) if mt % r == 0)
        fixed = mt * w * 4096 + 2 * self.rt * _BF16
        self.kb, self.depth = next(
            (
                (c, d)
                for c, d in _KB_DEPTH
                if self.kt % c == 0 and fixed + 2 * mt * c * _BF16 + d * c * w * _BF8 <= _CB_BUDGET
            ),
            (None, None),
        )
        if self.kb is None:
            raise RuntimeError(f"cpp_swiglu: no K block for {mt}x{self.kt}x{self.nt}")
        self.nb = self.kt // self.kb
        full, rem = divmod(self.ncores, self.gx)
        ranges = []
        if full:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.gx - 1, full - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
        self.cores = ttnn.CoreRangeSet(ranges)

    def _cb(self, index, fmt, tile_bytes, tiles):
        return ttnn.CBDescriptor(
            total_size=tiles * tile_bytes,
            core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=tile_bytes)],
        )

    def descriptor(self, x, fused, y):
        mt, kt, nt, kb, nb, pn, rt = self.mt, self.kt, self.nt, self.kb, self.nb, self.pn, self.rt
        w = fused.tensor
        xa, wa, ya = x.buffer_address(), w.buffer_address(), y.buffer_address()
        rw, rx, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c in range(self.ncores):
            cy, cx = divmod(c, self.gx)
            q, bank = divmod(c, fused.banks)
            rw[cx][cy] = [wa, bank, q * 2 * pn * _BF8]
            rx[cx][cy] = [xa, ya, c * pn]
            cp[cx][cy] = []
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_READER_W,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[kb, nb, fused.shard_tiles * _BF8, 2 * pn * _BF8, 2 * pn],
                runtime_args=rw,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_READER_X,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[mt, kb, nb, pn, rt, kt, nt] + _accessor_args(x) + _accessor_args(y),
                runtime_args=rx,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_COMPUTE,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[mt, kb, nb, pn, rt],
                runtime_args=cp,
                config=ttnn.ComputeConfigDescriptor(),
            ),
        ]
        cfg = kernels[2].config
        cfg.math_fidelity = ttnn.MathFidelity.HiFi2
        cfg.fp32_dest_acc_en = True
        cfg.math_approx_mode = False
        cfg.dst_full_sync_en = True
        cbs = [
            self._cb(0, ttnn.bfloat16, _BF16, 2 * mt * kb),
            self._cb(1, _W_DTYPE, _BF8, self.depth * kb * 2 * pn),
            self._cb(16, ttnn.bfloat16, _BF16, 2 * rt),
            self._cb(24, ttnn.float32, 4096, mt * 2 * pn),
        ]
        desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        desc.custom_program_hash = hash(("voxtral_cpp_swiglu", mt, kt, nt, kb, pn, rt, xa, wa, ya)) & 0xFFFFFFFFFFFFFFFF
        return desc


_PLANS: dict = {}


def _rows(x):
    rows = 1
    for d in tuple(x.padded_shape)[:-1]:
        rows *= int(d)
    return rows


def serves(x, fused) -> bool:
    if fused is None or not enabled():
        return False
    try:
        if x.layout != ttnn.TILE_LAYOUT or x.is_sharded() or x.dtype != ttnn.bfloat16:
            return False
        rows, k = _rows(x), int(tuple(x.padded_shape)[-1])
        return k == fused.k and rows % _TILE == 0 and 1 <= rows // _TILE <= 8
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def apply(x, fused):
    """`silu(x @ w1) * (x @ w3)` as bf16 in L1; x is moved to L1 first if it is not there."""
    device = x.device()
    shape = [int(d) for d in x.shape]
    rows, k, n = _rows(x), shape[-1], fused.n
    mt = rows // _TILE
    key = (mt, id(fused.tensor))
    plan = _PLANS.get(key)
    if plan is None:
        plan = _PLANS[key] = _Plan(device, mt, fused)
        _log(
            f"plan {mt}x{plan.kt}x{plan.nt} cores={plan.ncores} pn={plan.pn} kb={plan.kb} depth={plan.depth} "
            f"rt={plan.rt} banks={fused.banks}"
        )
    flat = ttnn.reshape(x, [1, 1, rows, k])
    if flat.memory_config().buffer_type != ttnn.BufferType.L1:
        flat = ttnn.to_memory_config(flat, ttnn.L1_MEMORY_CONFIG)
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    ttnn.generic_op([flat, fused.tensor, y], plan.descriptor(flat, fused, y))
    return ttnn.reshape(y, shape[:-1] + [n])
