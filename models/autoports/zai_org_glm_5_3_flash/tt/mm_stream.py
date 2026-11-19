# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a megakernel M2 building block: decode matmul y[M <= 32, N] = x[M, K] @ W[K, N] with the weight in the
per-core stream layout "mk-stream-v1" and the work spread over many worker cores (not just the 16 DS bank readers),
so that in the layer program every core streams its own weight slice and the cross-phase prefetch capacity is the L1
of the whole grid.

Layout mk-stream-v1 (MK_STREAM_LAYOUT_VERSION; the contract with Stage 12, doc/ttnn/07a-fused-streaming-layer/
megakernel_layout.md): core c (of C) owns output tiles [c * NT, (c + 1) * NT) and is served by DRAM bank b(c) = c mod 8;
its stream is the K x NT tiles of its columns in K-row-major order (tile (k, j) at stream index k * NT + j), stored as
CONSECUTIVE pages of bank b(c) of one DRAM-interleaved tile tensor (interleaved page p lives in bank p mod 8 at in-bank
index p // 8, so the stream's in-bank indices m0(c) .. m0(c) + K * NT - 1 map to pages b + 8 m). The cores of a bank are
packed one after another in the bank; the tensor is padded to the largest bank. Tile formats as today (bf8 / bf16).

Program: BRISC streams the core's weight in K blocks (one contiguous byte range per block) into a 3-deep ring
(kernels/mm_stream/wreader.cpp); NCRISC pulls the activation K blocks and writes the NT output tiles; TRISC = the
ds_mm compute (packer-L1-accumulated fp32 partials). kernels ds_mm/mover.cpp + ds_mm/compute.cpp are shared.
"""

from __future__ import annotations

import os

import torch

import ttnn

MK_STREAM_LAYOUT_VERSION = "mk-stream-v1"
KERNEL_DIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels")
TILE = 32
BF16, BF8, FP32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
TILE_BYTES = {BF16: 2048, BF8: 1088, FP32: 4096}
BANKS = 8
MAX_BURST = 16384


def stream_layout(k: int, n: int, cores: int) -> dict:
    """Placement of a [K, N] weight over `cores` cores: NT tiles per core, bank b(c) = c mod 8, in-bank start m0(c),
    pages per bank. N must be cores * NT tiles (callers choose cores so that NT divides)."""
    kt, nt_total = k // TILE, n // TILE
    if nt_total % cores:
        raise ValueError(f"{nt_total} column tiles do not split over {cores} cores")
    nt = nt_total // cores
    used = [0] * BANKS
    m0 = []
    for c in range(cores):
        b = c % BANKS
        m0.append(used[b])
        used[b] += kt * nt
    return dict(version=MK_STREAM_LAYOUT_VERSION, k=k, n=n, kt=kt, nt=nt, cores=cores, m0=m0,
                pages_per_bank=max(used))  # fmt: skip


def stream_tiles(w: torch.Tensor, lay: dict) -> torch.Tensor:
    """W [K, N] (host) -> the [pages * 32, 32] tile array of the interleaved stream tensor (zero tiles as padding)."""
    kt, nt, cores = lay["kt"], lay["nt"], lay["cores"]
    pages = lay["pages_per_bank"] * BANKS
    out = torch.zeros(pages, TILE, TILE, dtype=w.dtype)
    wt = w.reshape(kt, TILE, cores * nt, TILE).permute(0, 2, 1, 3)  # [kt, n tiles, 32, 32]
    for c in range(cores):
        b, m0 = c % BANKS, lay["m0"][c]
        tiles = wt[:, c * nt : (c + 1) * nt].reshape(kt * nt, TILE, TILE)  # (k, j) -> k * nt + j
        idx = b + BANKS * (m0 + torch.arange(kt * nt))
        out[idx] = tiles
    return out.reshape(pages * TILE, TILE)


def _tiles_per_txn(n: int, tb: int) -> int:
    return max(d for d in range(1, n + 1) if n % d == 0 and d * tb <= MAX_BURST)


class StreamMatmul:
    """y = x @ W with W [K, N] (host torch) loaded in the mk-stream-v1 layout on `cores` worker cores."""

    def __init__(self, device, w: torch.Tensor, *, cores: int, kb: int, w_dtype=BF8, out_dtype=BF16,
                 fidelity=ttnn.MathFidelity.HiFi2):  # fmt: skip
        self.device = device
        k, n = int(w.shape[0]), int(w.shape[1])
        self.lay = stream_layout(k, n, cores)
        self.k, self.n, self.kt, self.nt, self.ncores = k, n, self.lay["kt"], self.lay["nt"], cores
        if self.kt % kb:
            raise ValueError(f"K block {kb} does not divide {self.kt}")
        self.kb, self.nb = kb, self.kt // kb
        self.w_dtype, self.out_dtype, self.fidelity = w_dtype, out_dtype, fidelity
        self.tb = TILE_BYTES[w_dtype]
        self.tpt = _tiles_per_txn(kb * self.nt, self.tb)
        self.subw = max(s for s in (4, 2, 1) if self.nt % s == 0)
        self.weight = ttnn.from_torch(stream_tiles(w.float(), self.lay).reshape(1, 1, -1, TILE), dtype=w_dtype,
                                      layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)  # fmt: skip
        grid = device.compute_with_storage_grid_size()
        allc = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
        if cores > len(allc):
            raise ValueError(f"{cores} cores > grid {grid}")
        self.core_list = allc[:cores]  # core c = row-major over the grid
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.core_list])
        self.calls = 0

    def _rt(self, fn):
        rt = ttnn.RuntimeArgs()
        for i, c in enumerate(self.core_list):
            rt[c.x][c.y] = [int(v) for v in fn(i)]
        return rt

    def output(self, x, out_memory_config=L1):
        m = int(x.shape[-2])
        if x.layout != ttnn.TILE_LAYOUT or int(x.shape[-1]) != self.k or m > TILE:
            raise ValueError(f"mm_stream: x {tuple(x.shape)}")
        return ttnn.empty((1, 1, m, self.n), self.out_dtype, ttnn.TILE_LAYOUT, self.device, out_memory_config)

    def parts(self, x, y, *, cb_base: int = 0, wait=None):
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        defines = [("DSMM_CB_BASE", str(cb_base))] if cb_base else []
        mdef = list(defines) + ([("DSMM_WAIT_SEM_ID", str(wait[0])), ("DSMM_WAIT_COUNT", str(wait[1]))] if wait else [])
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        w_addr = self.weight.buffer_address()
        kd = lambda name, ct, rt, defs, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNEL_DIR, name), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores, compile_time_args=ct, runtime_args=rt, defines=defs, config=cfg,
        )  # fmt: skip
        wreader = kd("mm_stream/wreader.cpp", [self.nb, self.kb, self.nt, self.tb, self.tpt],
                     self._rt(lambda i: [w_addr, i % BANKS, self.lay["m0"][i], (i // BANKS) & 3]), defines,
                     dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default))  # fmt: skip
        mover = kd("ds_mm/mover.cpp", [self.nb, self.kb, self.nt] + acc(x) + acc(y),
                   self._rt(lambda i: [x.buffer_address(), y.buffer_address(), i * self.nt]), mdef,
                   dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default))  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, math_approx_mode=False, fp32_dest_acc_en=True)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[cb_base + 2] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        compute = kd("ds_mm/compute.cpp", [self.nb, self.kb, self.nt, self.subw], ttnn.RuntimeArgs(), defines, cc)
        cb = lambda i, dt, page, n: ttnn.CBDescriptor(
            total_size=page * n, core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
        )  # fmt: skip
        b = cb_base
        cbs = [cb(b + 0, x.dtype, TILE_BYTES[x.dtype], 2 * self.kb), cb(b + 1, self.w_dtype, self.tb, 3 * self.kb * self.nt),
               cb(b + 2, FP32, 4096, self.nt), cb(b + 3, self.out_dtype, TILE_BYTES[self.out_dtype], self.nt)]  # fmt: skip
        return [wreader, mover, compute], cbs

    def __call__(self, x, *, out_memory_config=L1):
        y = self.output(x, out_memory_config)
        kernels, cbs = self.parts(x, y)
        ttnn.generic_op([x, self.weight, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        self.calls += 1
        return y
