# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model dimensions and the streamer layout helpers shared by the resident decode, the prefill and the benches."""
import os
import struct

import torch

import ttnn
from qwen36_resident import PKG_DIR

TILE = 32
# S0-9 stage 2: "bfp2" stands for the Bfp2_b tile format ttnn has no dtype for (raw uint32 pages, host codec below)
BFP2 = "bfp2"
BFP2_FORMAT = 15  # tt::DataFormat::Bfp2_b (CBFormatDescriptor.data_format_as_uint8)
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, BFP2: 320}


def bfp2_round(x):
    """x [..., n] (n a multiple of 16) on the Bfp2_b grid: per 16 consecutive elements of the last dim (a tile face row)
    e = floor(log2 max |x|), each element sign(x) 2^e if |x| >= 2^(e - 1) else 0 (ties up; acc_agree's bfp_round(x, 1)
    but for the ties) — what the decode writer and the prefill store"""
    sh = x.shape
    g = x.reshape(-1, 16).float()
    amax = g.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(amax.clamp(min=1e-38)))
    v = torch.pow(2.0, e)
    q = torch.where(g.abs() >= v / 2, v, torch.zeros_like(v)) * torch.sign(g)
    q = torch.where(amax > 0, q, torch.zeros_like(q))
    return q.reshape(sh)


def bfp2_encode(x):
    """x [R * 32, C] (C a multiple of 32) -> uint8 [R, (C / 32) * 320]: tile row r's tiles, each 320 B — 64 exponent
    bytes (face F row i at 16 F + i), then 4 B a face row (element c at bits 2 (c % 4) of byte c / 4, the pair's high
    bit the sign, low bit the magnitude), value m 2^(e - 127) (tests/bfp2_probe.py: the unpacker decodes it so)"""
    R, C = x.shape[0] // 32, x.shape[1] // 32
    t = x.float().reshape(R, 32, C, 32).permute(0, 2, 1, 3)  # [R, C, 32, 32]
    f = t.reshape(R, C, 2, 16, 2, 16).permute(0, 1, 2, 4, 3, 5).reshape(R, C, 4, 16, 16)  # faces [F, row, col]
    amax = f.abs().amax(-1)  # [R, C, 4, 16]
    e = torch.floor(torch.log2(amax.clamp(min=1e-38)))
    ebyte = torch.where(amax > 0, (e + 127).clamp(0, 255), torch.zeros_like(e)).to(torch.uint8)
    v = torch.pow(2.0, e)[..., None]
    m = ((f.abs() >= v / 2) & (amax[..., None] > 0)).to(torch.uint8)
    sgn = ((f < 0) & (m > 0)).to(torch.uint8)
    two = (sgn << 1) | m  # [R, C, 4, 16, 16]
    two = two.reshape(R, C, 4, 16, 4, 4).to(torch.int32)
    mant = (two[..., 0] | (two[..., 1] << 2) | (two[..., 2] << 4) | (two[..., 3] << 6)).to(torch.uint8)
    out = torch.cat([ebyte.reshape(R, C, 64), mant.reshape(R, C, 256)], -1)  # [R, C, 320]
    return out.reshape(R, C * 320)


def bfp2_decode(b, C):
    """uint8 [R, (C / 32) * 320] (bfp2_encode's) -> float [R * 32, C]"""
    R, nt = b.shape[0], C // 32
    t = b.reshape(R, nt, 320)
    e = t[..., :64].to(torch.float32).reshape(R, nt, 4, 16)
    mant = t[..., 64:].to(torch.int32).reshape(R, nt, 4, 16, 4)
    two = torch.stack([(mant >> (2 * j)) & 3 for j in range(4)], -1).reshape(R, nt, 4, 16, 16)
    mag = (two & 1).float() * torch.pow(2.0, e - 127)[..., None]
    f = torch.where((two >> 1) & 1 > 0, -mag, mag)  # [R, nt, F, row, col]
    t = f.reshape(R, nt, 2, 2, 16, 16).permute(0, 1, 2, 4, 3, 5).reshape(R, nt, 32, 32)
    return t.permute(0, 2, 1, 3).reshape(R * 32, nt * 32)


HIDDEN = 5120
INTER = 17408  # full MLP width; each chip holds INTER / num_chips
EPS = 1e-6
PER_BANK = int(os.environ.get("RESIDENT_PER_BANK", "4"))  # streamer cores per DRAM bank
L1_WEIGHT_BUDGET = int(os.environ.get("RESIDENT_RING_BYTES", "1000000"))  # weight ring bytes per streamer


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


MAX_PAGE = int(os.environ.get("RESIDENT_MAX_PAGE", "8192"))  # DRAM read packet size cap


def block_geometry(Kt, dtype, max_block=20_000, max_page=None):
    max_page = max_page or MAX_PAGE
    tb = TILE_BYTES[dtype]
    sb = max(d for d in range(2, Kt + 1, 2) if Kt % d == 0 and d * tb <= max_block)
    block = sb * tb
    page = (max_page // tb) * tb
    while block % page:
        page -= tb
    return sb, block // page, page, block


def bank_core_cols(total_tiles, banks, offset=0, first_extra=0):
    """core_cols for column tiles [0, total_tiles) split bank-major, then contiguously over PER_BANK cores."""
    per_bank = total_tiles // banks
    return [
        [
            list(range(offset + b * per_bank + f, offset + b * per_bank + f + c))
            for c, f in split(per_bank, PER_BANK, first_extra)
        ]
        for b in range(banks)
    ]


def gate_up_core_cols(it_total, banks, first_extra=0):
    """core_cols of cat(G, U): each core streams its gate tiles, then its up tiles."""
    g = bank_core_cols(it_total, banks, first_extra=first_extra)
    u = bank_core_cols(it_total, banks, offset=it_total, first_extra=first_extra)
    return [[gj + uj for gj, uj in zip(gb, ub)] for gb, ub in zip(g, u)]


def split(n, parts, first_extra=0):
    """(count, first) of `parts` contiguous pieces of n; the n % parts larger pieces start at part first_extra
    (cyclically), so entries with a remainder can put it on different cores."""
    q, r = divmod(n, parts)
    out, first = [], 0
    for j in range(parts):
        c = q + ((j - first_extra) % parts < r)
        out.append((c, first))
        first += c
    return out


def streamer_cores(device):
    """PER_BANK cores per DRAM bank. The banks' adjacent cores sit in two columns; each column's banks get
    one compact block of two columns (the bank column and the next) by PER_BANK / 2 rows, the banks in row
    order, so the hub's slot multicast covers all streamers with one rectangle per block."""
    primary = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    grid = device.compute_with_storage_grid_size()
    assert PER_BANK % 2 == 0
    rows = PER_BANK // 2
    groups = [None] * len(primary)
    used = set()
    for x in sorted({c.x for c in primary}):
        banks = sorted((c.y, i) for i, c in enumerate(primary) if c.x == x)
        height = rows * len(banks)
        top = min(max(round(sum(y for y, _ in banks) / len(banks) - height / 2), 0), grid.y - height)
        assert x + 1 < grid.x and top >= 0
        for k, (_, i) in enumerate(banks):
            g = [ttnn.CoreCoord(x + dx, top + k * rows + dy) for dy in range(rows) for dx in range(2)]
            used |= {(c.x, c.y) for c in g}
            groups[i] = g
    assert len(used) == PER_BANK * len(primary)
    return groups, used


def rect_cover(mesh, cores):
    """rectangles covering exactly `cores` (logical), as hub multicast args: count, then per rectangle
    noc x0, y0, x1, y1 (physical) and its number of cores"""
    left = {(c.x, c.y) for c in cores}
    out = []
    for y, x in sorted((c.y, c.x) for c in cores):
        if (x, y) not in left:
            continue
        x1 = x
        while (x1 + 1, y) in left:
            x1 += 1
        y1 = y
        while all((xx, y1 + 1) in left for xx in range(x, x1 + 1)):
            y1 += 1
        for yy in range(y, y1 + 1):
            for xx in range(x, x1 + 1):
                left.discard((xx, yy))
        p0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        p1 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x1, y1))
        out += [p0.x, p0.y, p1.x, p1.y, (x1 - x + 1) * (y1 - y + 1)]
    return [len(out) // 5] + out


def quantize(w, dtype):
    return ttnn.to_torch(ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


ZERO_DRAM_TENSORS = (341 - 2) // 2  # per zero_dram program: 2 + 2 args per tensor per core, max_runtime_args 341


def zero_dram(mesh, tensors, cores_per_bank=4, chunk=32768):
    """zero-fill DRAM tensors on device (kernels/dram_zero.cpp), on every chip, without a host image: per bank
    the allocator's part of each buffer, ceil(pages / banks) aligned pages from the buffer address (interleaved
    or sharded over all banks). All-zero bytes are zeros in every format used (bf16, fp32, bf8 / bf4 tiles,
    integers).

    Every core's runtime args are padded to 2 + 2 * len(tensors) and len(tensors) is a compile-time arg: generic_op's
    program hash covers the number of cores with runtime args but not each core's arg count, and a cache hit only
    overwrites the cached program's args in place, so a call with more ranges per core than the program that seeded
    the cache wrote past them (a 28-tensor PPPrefill.reset after a 1-tensor device_zeros: TT_FATAL "Index 4 is larger
    than runtime args size 4", then a segfault or a hung dispatch)."""
    tensors = list(tensors)
    for i in range(0, len(tensors), ZERO_DRAM_TENSORS):
        _zero_dram(mesh, tensors[i : i + ZERO_DRAM_TENSORS], cores_per_bank, chunk)
    ttnn.synchronize_device(mesh)


def _zero_dram(mesh, tensors, cores_per_bank, chunk):
    banks = mesh.dram_grid_size().x
    grid = mesh.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(banks * cores_per_bank)]
    ranges = [[] for _ in cores]
    for t in tensors:
        per_bank = -(-t.buffer_num_pages() // banks) * t.buffer_aligned_page_size()
        step = -(-per_bank // cores_per_bank // 64) * 64
        for b in range(banks):
            for k in range(cores_per_bank):
                lo, hi = k * step, min(per_bank, (k + 1) * step)
                if lo < hi:
                    ranges[b * cores_per_bank + k] += [t.buffer_address() + lo, hi - lo]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    rt = ttnn.RuntimeArgs()
    width = 2 * len(tensors)  # (a tensor gives a core one range at most)
    for i, c in enumerate(cores):
        rt[c.x][c.y] = [i // cores_per_bank, len(ranges[i]) // 2] + ranges[i] + [0] * (width - len(ranges[i]))
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=f"{PKG_DIR}/kernels/dram_zero.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_set,
                compile_time_args=[chunk, len(tensors)],  # (the args' length, in the program hash)
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        ],
        semaphores=[],
        cbs=[
            ttnn.CBDescriptor(
                total_size=chunk,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=chunk)],
            )
        ],
    )
    n = mesh.get_num_devices()
    pd = ttnn.MeshProgramDescriptor()
    pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, n - 1))] = program
    ttnn.generic_op(tensors * 2 if len(tensors) == 1 else tensors, pd)  # (generic_op takes an input and an output)


def device_zeros(mesh, shape, dtype, memory_config, layout=ttnn.TILE_LAYOUT):
    """a zero-filled device tensor of `shape` per chip (allocated on device, zeroed by zero_dram)"""
    t = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, mesh, memory_config)
    zero_dram(mesh, [t])
    return t


class Deferred:
    """one chip's matrix of a layer, read from the checkpoint when placed (load(lazy=True)): fn() returns every
    chip's, and the last fn's result is kept, so the chips of one layer read it once"""

    _last = (None, None)

    def __init__(self, key, fn, chip):
        self.key, self.fn, self.chip = key, fn, chip

    def get(self):
        if Deferred._last[0] != self.key:
            Deferred._last = (None, None)  # drop the previous layer's first
            Deferred._last = (self.key, self.fn())
        return Deferred._last[1][self.chip]

    @staticmethod
    def release():
        Deferred._last = (None, None)


def materialize(w):
    """a weight that may be Deferred, as a tensor"""
    return w.get() if isinstance(w, Deferred) else w
