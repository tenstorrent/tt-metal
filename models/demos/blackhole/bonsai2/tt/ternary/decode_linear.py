# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ternary decode linear: y[n] = sum_kb s[n, kb] * sum_{k in kb} x'[k] t[k, n], t in {-1, 0, 1}, s fp16 per
128-wide K block (Bonsai PQ2_0), on one tile row of activation.

Weights stay in Bonsai's own values: t as 2-bit Bfp2_b data with a forced exponent (pack.py), s as IEEE fp16.
Record nt is output tile column nt: for each row block m of 32 K blocks (128 K tiles), its ternary tiles (256 B
each, K order), then one fp16 scale tile whose row r, column c holds s[32nt + c, 32m + r]; zero padding to a multiple
of 4 KB. The tensor's pages are 4 KB (DRAM-interleaved, so a core's consecutive pages rotate over the banks, the
pattern that reads at 82% of the 512 GB/s peak from one RISC: bonsai/mb/dram_bw.py on this P150).
"""

from dataclasses import dataclass

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.pack import TILE, pack_ternary_tiles

KBLOCK = 128  # Bonsai PQ2_0 block: one fp16 scale per 128 weights along K
KT_PER_BLOCK = KBLOCK // TILE  # 4 K tiles per scale block
BLOCKS_PER_M = TILE  # one scale block per row of a 32-row tile
KT_PER_M = KT_PER_BLOCK * BLOCKS_PER_M  # 128 K tiles per row block
W_TILE_BYTES = 256
S_TILE_BYTES = 2048
PAGE_BYTES = 4096
KERNELS = "models/demos/blackhole/bonsai2/tt/ternary/kernels/"


def _fp16_tile_words(tile16: np.ndarray) -> np.ndarray:
    """[32, 32] float16 -> 512 uint32 of the tile in face order (four 16x16 faces, rows inside a face)."""
    f = tile16.reshape(2, 16, 2, 16).transpose(0, 2, 1, 3).reshape(-1)
    return f.view(np.uint16).view(np.uint32)


def pack_decode_weights(t_kn: np.ndarray, s_nb: np.ndarray) -> np.ndarray:
    """t_kn [K, N] int8 in {-1, 0, 1}, s_nb [N, K/128] float16 -> [N/32, record_words] uint32 pages."""
    K, N = t_kn.shape
    assert K % KBLOCK == 0 and N % TILE == 0, (K, N)
    Kt, Nt, KB = K // TILE, N // TILE, K // KBLOCK
    Mt = -(-KB // BLOCKS_PER_M)
    assert s_nb.shape == (N, KB) and s_nb.dtype == np.float16, s_nb.shape
    tiles = t_kn.reshape(Kt, TILE, Nt, TILE).transpose(2, 0, 1, 3)  # [Nt, Kt, 32, 32]
    words = pack_ternary_tiles(tiles)  # [Nt, Kt, 64]
    s_pad = np.zeros((N, Mt * BLOCKS_PER_M), dtype=np.float16)
    s_pad[:, :KB] = s_nb
    records = []
    for nt in range(Nt):
        parts = []
        for m in range(Mt):
            k0, k1 = m * KT_PER_M, min(Kt, (m + 1) * KT_PER_M)
            parts.append(words[nt, k0:k1].reshape(-1))
            s_tile = s_pad[nt * TILE : (nt + 1) * TILE, m * TILE : (m + 1) * TILE].T  # [row = kb, col = n]
            parts.append(_fp16_tile_words(np.ascontiguousarray(s_tile)))
        rec = np.concatenate(parts)
        pad = -len(rec) % (PAGE_BYTES // 4)
        records.append(np.concatenate([rec, np.zeros(pad, dtype=np.uint32)]))
    return np.stack(records).astype(np.uint32).reshape(-1, PAGE_BYTES // 4)


def block_diag_activation(x_rot: torch.Tensor) -> torch.Tensor:
    """x' [K] -> [32, K]: column k carries x'[k] in row (k // 128) % 32, zeros elsewhere."""
    K = x_rot.shape[-1]
    rows = (torch.arange(K) // KBLOCK) % TILE
    a = torch.zeros(TILE, K, dtype=x_rot.dtype)
    a[rows, torch.arange(K)] = x_rot.reshape(-1)
    return a


def reference(x_rot: torch.Tensor, t_kn: np.ndarray, s_nb: np.ndarray) -> torch.Tensor:
    """fp32 torch reference of the decode linear."""
    K, N = t_kn.shape
    xb = x_rot.float().reshape(K // KBLOCK, KBLOCK)
    tb = torch.from_numpy(t_kn).float().reshape(K // KBLOCK, KBLOCK, N)
    p = torch.einsum("bk,bkn->bn", xb, tb)  # [KB, N]
    return (p * torch.from_numpy(s_nb.astype(np.float32)).T).sum(0)


@dataclass
class TernaryDecodeLinear:
    K: int
    N: int
    w: ttnn.Tensor  # [N/32, record_words] uint32, DRAM interleaved, one page per output tile column
    cores: int
    pages_per_record: int  # 4 KB pages

    @classmethod
    def from_host(cls, device, t_kn, s_nb, cores=None):
        rec = pack_decode_weights(t_kn, s_nb)
        ppr = rec.shape[0] // (t_kn.shape[1] // TILE)
        w = ttnn.from_torch(
            torch.from_numpy(rec.view(np.int32)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        grid = device.compute_with_storage_grid_size()
        return cls(t_kn.shape[0], t_kn.shape[1], w, cores or grid.x * grid.y, ppr)

    def __call__(self, x_rot: ttnn.Tensor, inflight=8, debug=0) -> ttnn.Tensor:
        """x_rot: [1, K] bf16 ROW_MAJOR rotated activation -> [32, N] bf16 TILE, row 0 = y."""
        device = x_rot.device()
        K, N = self.K, self.N
        Kt, Nt, KB = K // TILE, N // TILE, K // KBLOCK
        Mt = -(-KB // BLOCKS_PER_M)
        record_pages = self.pages_per_record * PAGE_BYTES // W_TILE_BYTES  # 256-B CB pages, padding included
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([TILE, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        grid = device.compute_with_storage_grid_size()
        cores = min(self.cores, Nt)
        base, extra = divmod(Nt, cores)
        coords, spans = [], []
        start = 0
        for c in range(cores):
            n = base + (1 if c < extra else 0)
            coords.append(ttnn.CoreCoord(c % grid.x, c // grid.x))
            spans.append((start, n))
            start += n
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc) for cc in coords])

        def cb(index, fmt, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for cc, (s, n) in zip(coords, spans):
            rd[cc.x][cc.y] = [self.w.buffer_address(), s, n]
            wr[cc.x][cc.y] = [x_rot.buffer_address(), out.buffer_address(), s, n, self.w.buffer_address()]
            cp[cc.x][cc.y] = [n]
        rd_ct = [self.pages_per_record, inflight] + ttnn.TensorAccessorArgs(self.w).get_compile_time_args()
        wr_ct = [Kt, self.pages_per_record, inflight] + ttnn.TensorAccessorArgs(x_rot).get_compile_time_args()
        wr_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
        wr_ct += ttnn.TensorAccessorArgs(self.w).get_compile_time_args()
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "tdec_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    runtime_args=rd,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "tdec_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=wr_ct,
                    runtime_args=wr,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "tdec_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[Kt, Mt, debug, record_pages],
                    runtime_args=cp,
                    config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
                ),
            ],
            semaphores=[],
            cbs=[
                cb(0, ttnn.bfloat16, 2048, Kt),
                cb(1, ttnn.bfloat8_b, W_TILE_BYTES, 2 * record_pages),
                cb(6, ttnn.bfloat8_b, W_TILE_BYTES, 2 * record_pages),
                cb(3, ttnn.bfloat16, 2048, 1),
                cb(5, ttnn.bfloat16, Kt * 64, 1),
                cb(24, ttnn.float32, 4096, 1),
                cb(16, ttnn.bfloat16, 2048, max(n for _, n in spans)),  # BRISC writes them after its stream
            ],
        )
        return ttnn.generic_op([x_rot, self.w, out], program)
