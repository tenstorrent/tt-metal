# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paged PREFILL KV (workstream PF1 / PF2 of PAGED_PREFILL_APC_DESIGN.md).

Mode (env ``DSV41_PF_PAGED``, default ``read``; ``0`` = the previous release path with dense kv tables, untouched; ``set_mode`` changes it at run time, the next chunk-trace capture follows it):

  * ``write`` (PF1, dual write): the owner layers scatter the chunk's latents into the paged pool BEFORE the indexer / attention (the hand-off sink no longer writes them after). The dense
    kv tables stay and are the ones the attention reads: behaviour is unchanged, the pool rows must equal the table rows.
  * ``read``  (PF2): the dense kv tables are NOT allocated. ``sparse_sdpa`` reads the paged pool [1,1,R,512]: the selected FIFO slots are translated to pool rows by a small dataflow kernel
    (``paged_kernels/prefill_id_translate.cpp``), the window rows (halo + chunk) go to a scratch region of the pool (per prefill slot, after the rings). Default pool: bf16 (P1: the pool dtype is the
    prefill's KV precision; APC needs it); ``DSV41_PF_PAGED_FP8=1`` keeps the fp8 pool of the build (P2a: window rows and latents rounded to fp8; APC is then off).

The key FIFOs (indexer) and the halos are unchanged (PF3 / PF5).
"""

import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_single as PS
from models.demos.blackhole.deepseek_v41_flash.tt.h2d import h2d
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_single import slot_s0

DEFAULT_MODE = "read"  # shipped default (release 2026-10): paged prefill, reads the pool; DSV41_PF_PAGED=0 selects the previous dense-table release path
MODE = os.environ.get("DSV41_PF_PAGED", DEFAULT_MODE)
assert MODE in ("0", "write", "read"), MODE
BOUND = {}  # "sink": the model's PagedStateSink (slot -> decode user, pool)
WINDOW = 128
SKIP = 0xFFFFFFFF


def set_mode(m):
    global MODE
    assert m in ("0", "write", "read")
    MODE = m


def write_on():
    return MODE in ("write", "read")


def read_on():
    return MODE == "read"


def scratch_rows(Up, c_max):
    """pool rows the paged prefill needs for the window scratch of every prefill slot (halo 128 + chunk)."""
    return Up * (WINDOW + c_max)


def bind(sink):
    BOUND["sink"] = sink


class PagedCtx:
    """Per DynCtx persistent tensors of the paged read (refreshed by ``update`` before every replay)."""

    def __init__(self, md, ctx, U):
        sink = BOUND["sink"]
        self.md, self.ctx, self.U, self.sink, self.pool = md, ctx, U, sink, sink.pool
        self.rows, self.cols = tuple(md.shape)
        self.C = ctx.C
        self.W = WINDOW + ctx.C
        assert self.pool.scratch_rows >= U * self.W, (
            f"pool built with {self.pool.scratch_rows} scratch rows, the paged prefill of {U} slots x chunk {ctx.C} needs {U * self.W} "
            "(build with DSV41_PF_PAGED=write|read and DSV41_PF_CMAX >= chunk)"
        )
        shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        rep = ttnn.ReplicateTensorToMesh(md)
        self._shard, self._rep = shard, rep
        self.maxp = self.pool.max_pages
        self.pt = ttnn.from_torch(
            torch.full((self.rows * U, self.maxp), -1, dtype=torch.int32),
            device=md,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        self.per_row = bool(
            getattr(ctx, "per_row", False)
        )  # per-row chunk starts: off / win hold one copy per mesh row (U == 1 slot per row)
        self._m = shard if self.per_row else rep
        nr = self.rows if self.per_row else 1
        self.off = {
            r: ttnn.from_torch(
                torch.zeros(nr, 32, dtype=torch.int32),
                device=md,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._m,
            )
            for r in ctx.ratios
        }
        self.win = ttnn.from_torch(
            torch.zeros(U * nr, 1, ctx.C, WINDOW, dtype=torch.int32),
            device=md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._m,
        )
        # bounded replay (DSV41_PF_REPLAY): the window ids of the layers >= REPLAY_BACK ("first chunk" = first replay chunk)
        self.win_b = (
            ttnn.from_torch(
                torch.zeros(U * nr, 1, ctx.C, WINDOW, dtype=torch.int32),
                device=md,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._m,
            )
            if int(os.environ.get("DSV41_PF_REPLAY", "0")) > 0
            else None
        )

    def scratch_base(self, u):
        return self.pool.scratch_origin + u * self.W

    def update(self, s0):
        ctx, C, U = self.ctx, self.C, self.U
        sink = self.sink
        tab = self.pool.table_host()
        pt = torch.full((self.rows * U, self.maxp), -1, dtype=torch.int32)
        for r in range(self.rows):
            for u in range(U):
                b = sink._user_of(r, u)
                if b is not None:
                    pt[r * U + u] = tab[b]
        h2d(ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._shard), self.pt)
        s0s = slot_s0(s0, self.rows if self.per_row else 1)
        for r, dev in self.off.items():
            o = torch.zeros(len(s0s), 32, dtype=torch.int32)
            for k, x in enumerate(s0s):
                o[k, 0] = (
                    x // r if PS.enabled() else ctx.L[r] - (x + C) // r
                )  # single trace: first entry of the chunk (scratch slots); FIFO: slot - off = entry
            h2d(ttnn.from_torch(o, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._m), dev)
        i = torch.arange(C).view(-1, 1)
        c = torch.arange(WINDOW).view(1, -1)
        ws = {}
        for x in set(s0s) | {0, -1}:  # (-1: the not-first window)
            if x == 0:  # kv of position p at scratch row base + 128 + p; valid positions max(0, i - 127) .. i
                p = (i - (WINDOW - 1)).clamp(min=0) + c
                rel = torch.where(p <= i, WINDOW + p, torch.full_like(p, -1))
            else:  # position p = x + i - 127 + c at scratch row base + p - (x - 128) = base + 1 + i + c
                rel = (1 + i + c).expand(C, WINDOW)
            w = torch.stack(
                [torch.where(rel >= 0, self.scratch_base(u) + rel, torch.full_like(rel, -1)) for u in range(U)]
            ).reshape(U, 1, C, WINDOW)
            ws[x] = torch.where(w < 0, torch.full_like(w, -1), w).to(torch.int64).to(torch.int32)  # -1 -> 0xFFFFFFFF
        w = torch.cat([ws[x] for x in s0s], dim=0)
        h2d(ttnn.from_torch(w, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._m), self.win)
        if self.win_b is not None:
            wb = torch.cat([ws[0] if f else ws[-1] for f in ctx.first_b(s0s)], dim=0)
            h2d(ttnn.from_torch(wb, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._m), self.win_b)

    def translate(self, ids, ratio, src_layer):
        """ids [U,1,C,512] uint32 RM (FIFO slots; single trace: slots of the scratch / history layout, tt/prefill_single.py) -> pool rows (same shape)."""
        if PS.enabled():
            return PS.translate_ids_single(
                ids, self.pt, self.off[ratio], ratio=ratio, src_off=P.SRC_OFF[src_layer], C=self.C, ccur=self.C // ratio
            )
        return translate_ids(ids, self.pt, self.off[ratio], ratio=ratio, src_off=P.SRC_OFF[src_layer], C=self.C)

    def write_window(self, full_rm):
        """full_rm [U,1,128+C,512] row-major (pool dtype): rows of slot u -> scratch of slot u."""
        U, W = self.U, self.W
        x = ttnn.reshape(full_rm, [1, 1, U * W, 512])
        o = self.pool.scratch_origin
        ttnn.experimental.slice_write(x, self.pool.pool, [0, 0, o, 0], [1, 1, o + U * W, 512], [1, 1, 1, 1])


def translate_ids(ids, pt, off, *, ratio, src_off, C, n_cores=64):
    md = ids.device()
    U = int(ids.shape[0])
    NID = int(ids.shape[-1])
    rows = U * C
    n = min(n_cores, rows)
    while rows % n or C % (rows // n) and (rows // n) % C:
        n -= 1
    rpc = rows // n
    assert C % rpc == 0 or rpc % C == 0
    assert rpc <= C, "a core handles rows of one user only"
    maxp = int(pt.shape[-1])
    ids2 = ttnn.reshape(ids, [rows, NID])
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, NID]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
    )
    cores = [ttnn.CoreCoord(i % 8, i // 8) for i in range(n)]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    total = ((maxp * 4 + 63) // 64) * 64 + 64 + 2 * NID * 4
    total = (total + 63) // 64 * 64
    cb = ttnn.CBDescriptor(
        total_size=total,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=total)],
    )
    ct = [rpc, C, ratio, src_off, maxp, NID, P.PAGE_TOKENS, P.PAGE_ROWS, 0]
    ct += P._acc(ids2) + P._acc(pt) + P._acc(off) + P._acc(out)
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        rt[c.x][c.y] = [i]
    k = ttnn.KernelDescriptor(
        kernel_source=f"{P.KDIR}/prefill_id_translate.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[
            ids2.buffer_address(),
            pt.buffer_address(),
            off.buffer_address(),
            out.buffer_address(),
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
    prog.custom_program_hash = (0x9A8 << 40) | (hash(tuple(ct)) & ((1 << 40) - 1))
    res = ttnn.generic_op([ids2, pt, off, out], prog)
    return ttnn.reshape(res, [U, 1, C, NID])
