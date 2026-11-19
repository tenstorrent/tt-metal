# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``LocalRoutedUntilize``: the GeGLU routed-expert output (bf8 TILE ``[M, H]``) back to bf16 ROW_MAJOR rows for the
row-indexed ``post_combine_reduce``, for the real rows of this chip's expert regions only.

The local routed prefill (``tt/experts/local_routed_prefill.py``) sizes its expert buffer for the drop-free worst case
(``M = T * K + 32 * E_loc`` rows, every token's K slots on one chip) while a chip's regions hold about ``T * K / tp``
of them; a whole-tensor typecast + untilize would convert all ``M`` rows. Here the unit is one tile-row of a local
expert region holding real rows (``ceil(count / 32)`` per expert, read from the device-side counts / region offsets),
dealt round-robin over the grid: the reader streams the unit's ``H / 32`` tiles, compute pack-untilizes them (the
unpacker reads bf8, the packer writes bf16 rows) and the writer writes the unit's real rows to the same rows of the
destination. The destination is normally the GeGLU op's own input buffer (``[M, H]`` bf16 ROW_MAJOR, consumed by
then): no allocation. Rows outside the regions and the padding rows of a region are not written.

Contract (per chip):
| tensor | shape / dtype / layout | role |
|---|---|---|
| ``src`` | ``[M, H]`` bf8 TILE DRAM | the GeGLU op's output |
| ``counts``, ``offsets`` | ``[1, E]`` uint32 ROW_MAJOR DRAM | per GLOBAL expert id (``LocalRoutedGather``'s outputs) |
| ``idx`` | ``[E_loc]`` uint32 ROW_MAJOR DRAM | local slot -> global expert id |
| ``dst`` (in/out) | ``[M, H]`` bf16 ROW_MAJOR DRAM | real rows of the regions overwritten with the expert outputs |
"""

from pathlib import Path

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


def block_ct_dim(h_tiles: int) -> int:
    """Largest divisor of ``h_tiles`` that is <= 8 (the pack_untilize block width in half-sync 16-bit dest)."""
    b = 8
    while h_tiles % b:
        b -= 1
    return b


class LocalRoutedUntilize:
    CB_IN, CB_OUT, CB_N, CB_META_R, CB_META_W = 0, 16, 1, 2, 3

    @staticmethod
    def golden(src_rows, counts, offsets, idx):
        """The rows the op writes: ``{row: src_rows[row]}`` for every real row of the local regions."""
        out = {}
        for g in [int(v) for v in idx]:
            for r in range(int(offsets[g]), int(offsets[g]) + int(counts[g])):
                out[r] = src_rows[r]
        return out

    def op(self, src: ttnn.Tensor, counts: ttnn.Tensor, offsets: ttnn.Tensor, idx: ttnn.Tensor, dst: ttnn.Tensor):
        assert src.layout == ttnn.TILE_LAYOUT and src.dtype == ttnn.bfloat8_b, "src: bf8 TILE"
        assert dst.layout == ttnn.ROW_MAJOR_LAYOUT and dst.dtype == ttnn.bfloat16, "dst: bf16 ROW_MAJOR"
        M, H = int(src.padded_shape[-2]), int(src.padded_shape[-1])
        assert [int(v) for v in dst.shape] == [M, H], f"dst {dst.shape} != [{M}, {H}]"
        E, E_loc = int(counts.shape[-1]), int(idx.shape[-1])
        for t in (counts, offsets, idx):
            assert t.layout == ttnn.ROW_MAJOR_LAYOUT and t.dtype == ttnn.uint32, t
        device = src.device()
        H_T = H // TILE
        bct = block_ct_dim(H_T)
        gx, gy = compute_grid(device)
        n = gx * gy  # units (tile-rows of the regions) are data-dependent: every core takes every n-th one
        cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(n)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        vec_page = counts.buffer_aligned_page_size()
        assert offsets.buffer_aligned_page_size() == vec_page
        src_page = src.buffer_aligned_page_size()
        dst_page = dst.buffer_aligned_page_size()
        row_bytes = 2 * H
        meta_bytes = 2 * (-(-vec_page // 64) * 64) + (-(-(E_loc * 4) // 64) * 64)

        def cb(index, total, page, fmt):
            return ttnn.CBDescriptor(
                total_size=total,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        cbs = [
            cb(self.CB_IN, 2 * bct * src_page, src_page, ttnn.bfloat8_b),
            cb(self.CB_OUT, 2 * TILE * row_bytes, row_bytes, ttnn.bfloat16),
            cb(self.CB_N, 64, 64, ttnn.uint32),
            cb(self.CB_META_R, meta_bytes, meta_bytes, ttnn.uint32),
            cb(self.CB_META_W, meta_bytes, meta_bytes, ttnn.uint32),
        ]
        reader_ct = [self.CB_IN, self.CB_N, self.CB_META_R, E, E_loc, H_T, bct, n, vec_page, src_page]
        writer_ct = [self.CB_OUT, self.CB_META_W, E, E_loc, n, vec_page, row_bytes, dst_page]
        for t in (src, counts, offsets, idx):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        for t in (dst, counts, offsets, idx):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        r_addrs = [src.buffer_address(), counts.buffer_address(), offsets.buffer_address(), idx.buffer_address()]
        w_addrs = [dst.buffer_address()] + r_addrs[1:]
        for i, c in enumerate(cores):
            reader_rt[c.x][c.y] = r_addrs + [i]
            writer_rt[c.x][c.y] = w_addrs + [i]
            compute_rt[c.x][c.y] = [i]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_local_untilize.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_local_untilize.cpp"),
                core_ranges=core_set,
                compile_time_args=[self.CB_IN, self.CB_OUT, self.CB_N, H_T, bct],
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_local_untilize.cpp"),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op([src, counts, offsets, idx, dst], program)
        return dst
