# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``PagedSparseSdpa``: tt-metal's ``sparse_sdpa`` (the sparse DSA prefill attention) with the K rows gathered straight
out of the paged bf16 TILE latent cache instead of a whole-prefix ROW_MAJOR copy, as a model-local ``ttnn.generic_op``
(kernels in ``kernels/``; contract in README.md).

The program mirrors the stock ``SparseSDPAProgramFactory`` (bf16, no scaled / fp8 kv, no block-cyclic kv, no attention
sink): the same circular buffers (ids = the stock ``SparseCB`` order), the stock compute kernel with its K tilize skipped
(the reader pushes the chunk already tiled), the stock writer without the K-half gather, and a reader that gathers
4-token pools out of the cache's tiles. One program per (H, S, TOPK, k_dim, v_dim, cache shape, scale); the user's slot
(``slot_block0``) and the tensor addresses are runtime arguments, so every DSA layer and every user share it.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
SENTINEL = 0xFFFFFFFF

# the stock factory's CB ids (SparseCB enum order); the ids are passed to the kernels as compile-time args
(
    CB_Q_RM,
    CB_Q_IN,
    CB_K_RM,
    CB_K_IN,
    CB_NEGINF,
    CB_MASK_PART,
    CB_SCALE,
    CB_QK_IM,
    CB_MAX_A,
    CB_MAX_B,
    CB_SUM_A,
    CB_SUM_B,
    CB_OUT_A,
    CB_OUT_B,
    CB_CORR,
    CB_OUT_IM,
    CB_OUT_RM,
    CB_IDX,
    CB_CTRL,
    CB_COL_IDENTITY,
    CB_RECIP_SCRATCH,
    CB_KREQ,
    CB_KACK,
    CB_K_ROPE_RM,
    CB_K_SCALE_BCAST,
    CB_K_LATENT_TILE,
    CB_K_ROPE_TILE,
    CB_ATTENTION_SINK,
    CB_SINK_SCRATCH,
) = range(29)

TILE_BYTES = 2048  # bf16 32x32
CTRL_PAGE_BYTES = 16  # sparse_sdpa::control_message::PAGE_BYTES (2 words, 16-byte aligned)


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


class PagedSparseSdpa:
    """``op(q, cache, indices, slot_block0=, blocks_per_seq=, v_dim=, scale=)`` -> ``out`` [1,H,S,v_dim] bf16 ROW_MAJOR.

    ``q`` [1,H,S,K_DIM] bf16 ROW_MAJOR (H a multiple of 32), ``cache`` [max_blocks,1,32,K_DIM] bf16 TILE, ``indices``
    [1,1,S,TOPK] uint32 ROW_MAJOR (4-token pools at 4-aligned row positions, the tail pool's ids, sentinels).
    ``slot_block0``: the user's first physical block in ``cache`` (identity page table); ``blocks_per_seq``: the
    user's block count (ids outside it read as zero rows). ``k_chunk_size`` and ``pool`` as the DSA path uses them."""

    def __init__(self, device, *, k_chunk_size: int = 128, pool: int = 4, math_fidelity=ttnn.MathFidelity.HiFi2):
        self.device = device
        self.k_chunk = int(k_chunk_size)
        self.pool = int(pool)
        self.fidelity = math_fidelity  # the stock op's default (init_device_compute_kernel_config: HiFi2, approx on)
        grid = device.compute_with_storage_grid_size()
        self.grid_x, self.grid_y = int(grid.x), int(grid.y)
        if self.k_chunk % TILE:
            raise ValueError(f"k_chunk_size {self.k_chunk} must be a multiple of {TILE}")
        if 16 % self.pool:
            raise ValueError(f"pool {self.pool} must divide a 16-row tile face")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def check_indices(indices_host) -> None:
        """The index-row contract the reader relies on (host tensor [.., S, TOPK] of int64 / uint32): below the
        first sentinel every 4-aligned position starts a run of consecutive ids whose first id is 4-aligned (the
        indexer's pool expansion), and the sentinels are a contiguous tail."""
        import torch

        rows = indices_host.reshape(-1, indices_host.shape[-1]).to(torch.int64)
        for row in rows:
            sent = (row == SENTINEL).nonzero().flatten()
            nv = int(sent[0]) if len(sent) else row.numel()
            if len(sent) and not bool((row[nv:] == SENTINEL).all()):
                raise ValueError("sentinels must be a contiguous tail")
            valid = row[:nv]
            for g0 in range(0, nv, 4):
                run = valid[g0 : g0 + 4]
                if int(run[0]) % 4:
                    raise ValueError(f"row position {g0}: id {int(run[0])} is not pool-aligned")
                if not bool((run == run[0] + torch.arange(len(run))).all()):
                    raise ValueError(f"row position {g0}: ids {run.tolist()} are not one pool's consecutive tokens")

    # ------------------------------------------------------------------ program
    def __call__(
        self,
        q: ttnn.Tensor,
        cache: ttnn.Tensor,
        indices: ttnn.Tensor,
        *,
        slot_block0: int,
        blocks_per_seq: int,
        v_dim: int,
        scale: float,
    ) -> ttnn.Tensor:
        assert q.layout == ttnn.ROW_MAJOR_LAYOUT and q.dtype == ttnn.bfloat16, (q.layout, q.dtype)
        assert cache.layout == ttnn.TILE_LAYOUT and cache.dtype == ttnn.bfloat16, (cache.layout, cache.dtype)
        assert indices.layout == ttnn.ROW_MAJOR_LAYOUT and indices.dtype == ttnn.uint32, (indices.layout, indices.dtype)
        qs, cs, ids = list(q.shape), list(cache.shape), list(indices.shape)
        assert len(qs) == 4 and qs[0] == 1 and qs[1] % TILE == 0, qs
        _, H, S, k_dim = qs
        assert len(cs) == 4 and cs[1] == 1 and cs[2] == TILE and cs[3] == k_dim, (cs, k_dim)
        assert ids[:3] == [1, 1, S], (ids, S)
        topk = ids[3]
        assert k_dim % TILE == 0 and v_dim % TILE == 0 and v_dim <= k_dim, (k_dim, v_dim)
        assert topk % self.k_chunk == 0 or topk >= self.k_chunk, topk
        assert 0 <= slot_block0 and slot_block0 + blocks_per_seq <= cs[0], (slot_block0, blocks_per_seq, cs[0])
        assert cache.buffer_aligned_page_size() == TILE_BYTES, cache.buffer_aligned_page_size()

        DHt, vDHt, Skt, Sqt = k_dim // TILE, v_dim // TILE, self.k_chunk // TILE, H // TILE
        q_row_bytes = k_dim * 2
        idx_row_bytes = topk * 4
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, H, S, v_dim]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        # the stock split: S tokens over the whole grid, the first S % cores cores take one more
        cores = [ttnn.CoreCoord(x, y) for y in range(self.grid_y) for x in range(self.grid_x)]
        n_cores = len(cores)
        core_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.grid_x - 1, self.grid_y - 1))]
        )
        base, extra = S // n_cores, S % n_cores

        def cb(index, page, pages, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        cbs = [
            cb(CB_Q_RM, q_row_bytes, H),  # H row-sticks of Q
            cb(CB_Q_IN, TILE_BYTES, Sqt * DHt),  # tiled Q
            cb(CB_K_RM, 64, 1),  # unused under PAGED (the compute still names it)
            cb(CB_K_IN, TILE_BYTES, Skt * DHt),  # the tiled K chunk (the reader's gather lands here)
            cb(CB_NEGINF, TILE_BYTES, 1),
            cb(CB_MASK_PART, TILE_BYTES, 1),
            cb(CB_SCALE, TILE_BYTES, 1),
            cb(CB_QK_IM, TILE_BYTES, Sqt * Skt),
            cb(CB_MAX_A, TILE_BYTES, Sqt),
            cb(CB_MAX_B, TILE_BYTES, Sqt),
            cb(CB_SUM_A, TILE_BYTES, Sqt),
            cb(CB_SUM_B, TILE_BYTES, Sqt),
            cb(CB_OUT_A, TILE_BYTES, Sqt * vDHt),
            cb(CB_OUT_B, TILE_BYTES, Sqt * vDHt),
            cb(CB_CORR, TILE_BYTES, Sqt),
            cb(CB_OUT_IM, TILE_BYTES, Sqt * vDHt),
            cb(CB_OUT_RM, TILE_BYTES, Sqt * vDHt),
            cb(CB_IDX, idx_row_bytes, 1),
            cb(CB_CTRL, CTRL_PAGE_BYTES, 2),
            cb(CB_COL_IDENTITY, TILE_BYTES, 1),
            cb(CB_RECIP_SCRATCH, TILE_BYTES, 1),
            cb(CB_KREQ, 32, 2),  # unused under PAGED
            cb(CB_KACK, 16, 2),  # unused under PAGED
        ]

        reader_ct = [
            H,
            S,
            topk,
            self.k_chunk,
            k_dim,
            CB_Q_RM,
            CB_K_IN,
            CB_MASK_PART,
            CB_IDX,
            CB_CTRL,
            DHt,
            Skt,
            int(blocks_per_seq),
            self.pool,
        ]
        for t in (q, cache, indices):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [H, S, vDHt, CB_OUT_RM, CB_SCALE, CB_COL_IDENTITY, CB_NEGINF, 2]
        writer_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
        # the DST-bound primitives hold qsb query tile-rows: the largest divisor of Sqt <= 8 (no fp32 accumulation)
        qsb = next(d for d in range(min(Sqt, 8), 0, -1) if Sqt % d == 0)
        compute_ct = [
            H, DHt, vDHt, Skt, _f32_bits(float(scale)),
            CB_Q_RM, CB_Q_IN, CB_K_RM, CB_K_IN, CB_NEGINF, CB_MASK_PART, CB_SCALE, CB_QK_IM,
            CB_MAX_A, CB_MAX_B, CB_SUM_A, CB_SUM_B, CB_OUT_A, CB_OUT_B, CB_CORR, CB_OUT_IM, CB_OUT_RM,
            CB_CTRL, CB_COL_IDENTITY, CB_RECIP_SCRATCH,
            0, CB_K_ROPE_RM, CB_K_SCALE_BCAST, CB_K_LATENT_TILE, CB_K_ROPE_TILE,  # scaled_kv off (ids unused)
            1, qsb, 0,  # math_approx_mode (the stock default), query sub-block, packed_row_bytes
            0, CB_ATTENTION_SINK,  # no attention sink
            1,  # PAGED (index sparse_sdpa::compute_ct_arg::END): the K chunk arrives tiled in cb_k_in
        ]  # fmt: skip

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, core in enumerate(cores):
            tok_start = i * base + min(i, extra)
            tok_count = base + (1 if i < extra else 0)
            reader_rt[core.x][core.y] = [
                tok_start, tok_count, q.buffer_address(), cache.buffer_address(), indices.buffer_address(), int(slot_block0)
            ]  # fmt: skip
            writer_rt[core.x][core.y] = [tok_start, tok_count, out.buffer_address()]
            compute_rt[core.x][core.y] = [tok_start, tok_count]

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_paged_sparse_sdpa.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_paged_sparse_sdpa.cpp"),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_paged_sparse_sdpa.cpp"),
                core_ranges=core_set,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=self.fidelity, fp32_dest_acc_en=False, math_approx_mode=True, dst_full_sync_en=False
                ),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op([q, cache, indices, out], program)
        return out
