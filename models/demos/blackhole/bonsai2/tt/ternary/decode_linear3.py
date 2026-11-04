# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ternary decode linear v3: y[n] = sum_kb s[n, kb] * sum_{k in kb} x'[k] t[k, n] on one activation row.

Weights in Bonsai's own values: t as 2-bit Bfp2_b tile data with a forced exponent (pack.py), one record per output
tile column (its K tiles in order, 256 B each; a multiple of 4 KB for every Bonsai K), stored as 4-KB pages so a
core's consecutive pages rotate over the DRAM banks. Scales: s as IEEE fp16, one [1, 32] tile (64 B) per (output
tile column, kb), the 32 columns in order. Compute: kernels/tdec3_compute.cpp.
"""

from dataclasses import dataclass

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.pack import TILE, pack_ternary_tiles

KBLOCK = 128
PAGE_BYTES = 4096
KERNELS = "models/demos/blackhole/bonsai2/tt/ternary/kernels/"


def pack_weights(t_kn: np.ndarray) -> np.ndarray:
    """t_kn [K, N] int8 in {-1, 0, 1} -> [N/32 * K/512, 1024] uint32: record nt = tiles (0..Kt-1, nt)."""
    K, N = t_kn.shape
    Kt, Nt = K // TILE, N // TILE
    assert Kt % 16 == 0, K
    tiles = t_kn.reshape(Kt, TILE, Nt, TILE).transpose(2, 0, 1, 3)  # [Nt, Kt, 32, 32]
    return pack_ternary_tiles(tiles).reshape(-1, PAGE_BYTES // 4)


def pack_scales(s_nb: np.ndarray, fp32: bool) -> np.ndarray:
    """s_nb [N, KB] float16 -> [N/32, KB * 32] uint16 (or float32): record nt = for each kb the 32 columns' scales."""
    N, KB = s_nb.shape
    r = np.ascontiguousarray(s_nb.reshape(N // TILE, TILE, KB).transpose(0, 2, 1)).reshape(N // TILE, KB * TILE)
    return r.astype(np.float32) if fp32 else r.view(np.uint16)


def pack_scales_group16(s_nb: np.ndarray) -> np.ndarray:
    """s_nb [N, KB] float16 -> [N/32, G * 1024] uint16, G = ceil(KB/32) fp16 32x32 tiles per output tile column (face
    order): kb 32g + j of group g at tile row j; rows past KB zero."""
    N, KB = s_nb.shape
    Nt, G = N // TILE, -(-KB // 32)
    t = np.zeros((Nt, G, TILE, TILE), dtype=np.float16)
    for kb in range(KB):
        g, j = divmod(kb, 32)
        t[:, g, j, :] = s_nb[:, kb].reshape(Nt, TILE)
    faces = t.reshape(Nt, G, 2, 16, 2, 16).transpose(0, 1, 2, 4, 3, 5)  # [Nt, G, fr, fc, 16, 16]
    return np.ascontiguousarray(faces).reshape(Nt, G * TILE * TILE).view(np.uint16)


def pack_records_inline(t_kn: np.ndarray, s_nb: np.ndarray) -> tuple[np.ndarray, int]:
    """Weights with each group's scale tile inline: record nt = for each group g of 32 kbs, its fp16 scale tile (2 KB,
    pack_scales_group16 layout) then its kbs' ternary tiles (256 B each); zero padding to a multiple of 4 KB.
    Returns ([N/32 * pages_per_record, 1024] uint32, pages_per_record)."""
    K, N = t_kn.shape
    Kt, Nt, KB = K // TILE, N // TILE, K // KBLOCK
    G = -(-KB // 32)
    tiles = t_kn.reshape(Kt, TILE, Nt, TILE).transpose(2, 0, 1, 3)
    w = pack_ternary_tiles(tiles)  # [Nt, Kt, 64]
    sc = pack_scales_group16(s_nb).view(np.uint32).reshape(Nt, G, 512)
    parts = []
    for g in range(G):
        parts.append(sc[:, g])
        k0, k1 = g * 128, min(Kt, (g + 1) * 128)
        parts.append(w[:, k0:k1].reshape(Nt, -1))
    rec = np.concatenate(parts, axis=1)
    pad = -rec.shape[1] % (PAGE_BYTES // 4)
    rec = np.concatenate([rec, np.zeros((Nt, pad), dtype=np.uint32)], axis=1)
    return rec.reshape(-1, PAGE_BYTES // 4), rec.shape[1] * 4 // PAGE_BYTES


def reference(x_rot: torch.Tensor, t_kn: np.ndarray, s_nb: np.ndarray) -> torch.Tensor:
    K, N = t_kn.shape
    xb = x_rot.float().reshape(K // KBLOCK, KBLOCK)
    tb = torch.from_numpy(t_kn).float().reshape(K // KBLOCK, KBLOCK, N)
    p = torch.einsum("bk,bkn->bn", xb, tb)
    return (p * torch.from_numpy(s_nb.astype(np.float32)).T).sum(0)


@dataclass
class TernaryDecodeLinear3:
    K: int
    N: int
    w: ttnn.Tensor
    s: ttnn.Tensor
    cores: int

    scale_fp32: bool = False
    group16: bool = False
    inline_scales: bool = False
    ppr: int = 0

    @classmethod
    def from_packed(cls, device, rec: np.ndarray, ppr: int, K: int, N: int, cores=None):
        """Upload records made by pack_records_inline (host packing and device upload can happen apart)."""
        w = ttnn.from_torch(
            torch.from_numpy(rec.view(np.int32)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        grid = device.compute_with_storage_grid_size()
        return cls(K, N, w, None, cores or grid.x * grid.y, False, True, True, ppr)

    @classmethod
    def from_host(cls, device, t_kn, s_nb, cores=None, scale_fp32=False, group16=False, inline_scales=False):
        if inline_scales:
            rec, ppr = pack_records_inline(t_kn, s_nb)
            w = ttnn.from_torch(
                torch.from_numpy(rec.view(np.int32)),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            grid = device.compute_with_storage_grid_size()
            return cls(t_kn.shape[0], t_kn.shape[1], w, None, cores or grid.x * grid.y, False, True, True, ppr)
        w = ttnn.from_torch(
            torch.from_numpy(pack_weights(t_kn).view(np.int32)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sp = pack_scales_group16(s_nb) if group16 else pack_scales(s_nb, scale_fp32)
        s = ttnn.from_torch(
            torch.from_numpy(sp if scale_fp32 else sp.view(np.int16)),
            dtype=ttnn.float32 if scale_fp32 else ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        grid = device.compute_with_storage_grid_size()
        return cls(t_kn.shape[0], t_kn.shape[1], w, s, cores or grid.x * grid.y, scale_fp32, group16)

    def __call__(
        self,
        x_rot,
        inflight=8,
        sfpu_iter=10,
        dbg=False,
        sfpu_reinit=True,
        extra_defines=(),
        dm_defines=(),
        lead=(1, 1, TILE),
        split=0,
    ):
        """split > 0: return two tensors, output columns [0, split) and [split, N) (split a multiple of 32).
        x_rot: [C, K] bf16 ROW_MAJOR, C copies of the rotated activation (one per DRAM bank: page c is in bank
        c mod 8) -> [32, N] bf16 TILE, row 0 = y."""
        device = x_rot.device()
        K, N = self.K, self.N
        Kt, Nt, KB = K // TILE, N // TILE, K // KBLOCK
        ppr = self.ppr if self.inline_scales else Kt * 256 // PAGE_BYTES
        # lead: the output's leading logical dims; any of them tile-pad to one 32-row tile row, so the pages match.
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(lead) + [(split or N) * (3 if dbg else 1)]),
            ttnn.float32 if dbg else ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
        )
        out2 = (
            ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(lead) + [N - split]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device
            )
            if split
            else None
        )
        grid = device.compute_with_storage_grid_size()
        cores = min(self.cores, Nt)
        base, extra = divmod(Nt, cores)
        coords, spans, start = [], [], 0
        for c in range(cores):
            n = base + (1 if c < extra else 0)
            coords.append(ttnn.CoreCoord(c % grid.x, c // grid.x))
            spans.append((start, n))
            start += n
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc) for cc in coords])
        max_n = max(n for _, n in spans)
        tiny = ttnn.TileDescriptor(1, 32, False)

        def cb(index, fmt, page, pages, tile=None):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page, tile=tile)
                ],
            )

        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for cc, (s, n) in zip(coords, spans):
            s_addr = 0 if self.inline_scales else self.s.buffer_address()
            rd[cc.x][cc.y] = [
                x_rot.buffer_address(),
                s_addr,
                self.w.buffer_address(),
                s,
                n,
                (cc.x + cc.y * grid.x) % x_rot.shape[0],
            ]
            wr[cc.x][cc.y] = [self.w.buffer_address(), out.buffer_address(), s, n, 0] + (
                [out2.buffer_address()] if split else []
            )
            cp[cc.x][cc.y] = [n]
        rd_ct = [Kt, KB, ppr, inflight]
        for t in (x_rot, self.w if self.inline_scales else self.s, self.w):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = [ppr, inflight] + ttnn.TensorAccessorArgs(self.w).get_compile_time_args()
        wr_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
        if split:
            assert split % TILE == 0, split
            wr_ct += ttnn.TensorAccessorArgs(out2).get_compile_time_args()
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "tdec3_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    defines=list(dm_defines)
                    + ([("INLINE_SCALES", "1")] if self.inline_scales else [])
                    + [("S_TILE_BYTES", "128" if self.scale_fp32 else "64")]
                    + ([("S_RECORD_BYTES", str(-(-KB // 32) * 2048))] if self.group16 else []),
                    runtime_args=rd,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "tdec3_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=wr_ct,
                    defines=list(dm_defines)
                    + ([("INLINE_SCALES", "1")] if self.inline_scales else [])
                    + [("OUT_PER_NT", "3" if dbg else "1"), ("SPLIT_NT", str(split // TILE))]
                    + [("S_PER_NT", str(-(-KB // 32) if self.group16 else KB))],
                    runtime_args=wr,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "tdec3_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[Kt, KB],
                    defines=[("SFPU_ITER", str(sfpu_iter)), ("SCALE_FP32", "1" if self.scale_fp32 else "0")]
                    + [("SFPU_REINIT", "1" if sfpu_reinit else "0")]
                    + ([("SCALE_GROUP16", "1")] if self.group16 else [])
                    + ([("INLINE_SCALES", "1"), ("RECORD_PAGES", str(ppr * 16))] if self.inline_scales else [])
                    + list(extra_defines)
                    + ([("DBG_FIRST_KB", "1")] if dbg else []),
                    runtime_args=cp,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, dst_full_sync_en=True
                    ),
                ),
            ],
            semaphores=[],
            cbs=[
                cb(0, ttnn.bfloat16, 64, Kt, tiny),
                cb(1, ttnn.bfloat8_b, 256, 2 * ppr * 16),
                cb(2, ttnn.float32, 2048, max_n * -(-KB // 32))
                if self.group16
                else cb(2, ttnn.float32, 128 if self.scale_fp32 else 64, max_n * KB, tiny),
                cb(6, ttnn.bfloat8_b, 256, 2 * ppr * 16),
                cb(16, ttnn.float32 if dbg else ttnn.bfloat16, 4096 if dbg else 2048, max_n * (3 if dbg else 1)),
            ],
        )
        res = ttnn.generic_op(
            [x_rot] + ([] if self.inline_scales else [self.s]) + [self.w, out] + ([out2] if split else []), program
        )
        return (out, out2) if split else res  # generic_op returns the last tensor of its list
