# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device side of the paged KV cache of DeepSeek-V4.1-Flash decode: the row-major pool, the page table tensor and the one custom op
(``paged_kv_step``, a generic_op with ``paged_kernels/paged_kv_step.cpp``) that

  * writes the new window row into the layer's ring region of the pool        (in place, position read on the device),
  * writes the new compressed latent into the shared page pool                (in place, page-table translated),
  * builds the ``sparse_sdpa`` index rows (ring rows + indexer-selected compressed rows, sentinel tail).

This is the "row-write primitive" of design doc section 4.4 / open item 7.1: no op of the repo writes ONE row at a data-dependent index into
a ROW_MAJOR cache (``paged_update_cache`` is TILE-only), so it is a small data-movement kernel. Pool layout (per mesh row, replicated over
the 8 columns; user u of the row = local user):

    rows [0, NP * 320)          shared pages, 320 rows per 128-token page: [src2: 64 | src8: 64 | src14: 64 | src20: 128]
    rows [NP * 320, ...)        rings: ring row of (ring slot l, user u, slot s) = NP*320 + (l * T + u) * RING + s
"""

import os

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paged_kernels")
HEAD_DIM = 512
PAGE_TOKENS = 128
SOURCES = (2, 8, 14, 20)
SRC_RATIO = {2: 2, 8: 2, 14: 2, 20: 1}
SRC_OFF = {2: 0, 8: 64, 14: 128, 20: 192}
PAGE_ROWS = 320
WINDOW = 128
RING_ROWS = 128  # 160 with spec-decode slack (any value >= WINDOW; the op takes it as a parameter)


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


class PagedKVPool:
    """Device tensors of the paged KV cache of ONE mesh: pool, page table, ring regions; plus the host allocators (one per mesh row)."""

    def __init__(
        self, mesh_device, users_per_row, num_pages, n_ring_layers, max_ctx, ring_rows=RING_ROWS, dtype=ttnn.bfloat16
    ):
        from models.demos.blackhole.deepseek_v41_flash.tt.kv_paged import PageAllocator

        assert dtype in (
            ttnn.bfloat16,
            ttnn.fp8_e4m3,
        ), "pool dtype: bf16 or fp8_e4m3 (rows are converted by the write kernels)"
        self.dtype = dtype
        self.row_bytes = 1024 if dtype == ttnn.bfloat16 else 512
        self.md, self.T, self.num_pages, self.ring_rows = mesh_device, users_per_row, num_pages, ring_rows
        self.rows, self.cols = tuple(mesh_device.shape)
        self.B = self.rows * users_per_row
        self.max_pages = -(-max_ctx // PAGE_TOKENS)
        self.ring_origin = num_pages * PAGE_ROWS
        self.n_ring_layers = n_ring_layers
        self.total_rows = self.ring_origin + n_ring_layers * users_per_row * ring_rows
        if dtype == ttnn.bfloat16:
            self.pool = ttnn.zeros(
                ttnn.Shape([1, 1, self.total_rows, HEAD_DIM]),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:  # uninitialised rows are never read (indices only reference written rows)
            self.pool = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, self.total_rows, HEAD_DIM]),
                dtype,
                ttnn.ROW_MAJOR_LAYOUT,
                mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        self.allocs = [PageAllocator(num_pages, PAGE_TOKENS, self.max_pages) for _ in range(self.rows)]
        self.page_table = None
        self.sync_page_table()

    # ---- host side -------------------------------------------------------------------------------------------------------------
    def user_key(self, b):
        """global user b (row-major over the mesh rows) -> (mesh row, key in that row's allocator)."""
        return b // self.T, b

    def admit(self, b, tokens, reserve_tokens=0):
        r, k = self.user_key(b)
        return self.allocs[r].admit(k, tokens, reserve_tokens)

    def grow(self, b, tokens, reserve_tokens=0):
        r, k = self.user_key(b)
        return self.allocs[r].grow(k, tokens, reserve_tokens)

    def ensure(self, positions, lookahead=128):
        """Make sure every user owns the pages for positions up to pos + lookahead (the device loop advances the position by one per replay without the
        host: call this between bursts of <= ``lookahead`` replays). Uploads the page table into the persistent device tensor only when it changed.
        """
        changed = False
        for b in range(self.B):
            r, k = self.user_key(b)
            before = len(self.allocs[r].pages[k])
            self.allocs[r].grow(k, int(positions[b]) + 1, lookahead)
            changed |= len(self.allocs[r].pages[k]) != before
        if changed:
            self.sync_page_table()
        return changed

    def free_pages(self):
        return [a.free_pages() for a in self.allocs]

    def release(self, b):
        r, k = self.user_key(b)
        self.allocs[r].release(k)

    def table_host(self):
        """torch int32 [B, max_pages] (-1 padded)."""
        return torch.cat(
            [a.page_table(list(range(r * self.T, (r + 1) * self.T)), self.max_pages) for r, a in enumerate(self.allocs)]
        )

    def sync_page_table(self):
        """Upload the host page tables into the PERSISTENT device tensor (created on the first call, afterwards copied in place: trace-safe)."""
        host = ttnn.from_torch(
            self.table_host(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(self.rows, self.cols)),
        )
        if self.page_table is None:
            self.page_table = ttnn.to_device(host, self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn.copy_host_to_device_tensor(host, self.page_table)

    def ring_base(self, slot):
        return self.ring_origin + slot * self.T * self.ring_rows

    # ---- host <-> pool (tests, seeding; slow, not for the decode loop) -----------------------------------------------------------
    def phys_rows(self, b, src, entries):
        """physical pool rows of entries (torch long) of kv source layer ``src`` for global user b."""
        from models.demos.blackhole.deepseek_v41_flash.tt.kv_paged import PageLayout

        pt = self.table_host()[b].long()
        return PageLayout().phys_rows(pt, SOURCES.index(src), entries)

    def stage_begin(self):
        """Host staging copy of the pool [rows, R, 512] bf16 (zeros); fill with ``stage_*`` then ``stage_commit``."""
        self._stage = torch.zeros(self.rows, self.total_rows, HEAD_DIM, dtype=torch.bfloat16)

    def stage_ring(self, slot, window):
        """window [B, RING_or_128, 512]: ring content (ring slot s = position % RING) of every global user for ring region ``slot``."""
        n = window.shape[1]
        for b in range(self.B):
            r, u = b // self.T, b % self.T
            base = self.ring_base(slot) + u * self.ring_rows
            self._stage[r, base : base + n] = window[b].to(torch.bfloat16)

    def stage_comp(self, src, comp, lengths=None):
        """comp [B, N, 512]: compressed latents of kv source layer ``src`` (entries 0..N-1) -> the users' pages."""
        N = comp.shape[1]
        ent = torch.arange(N)
        for b in range(self.B):
            n = N if lengths is None else int(lengths[b])
            rows = self.phys_rows(b, src, ent[:n])
            self._stage[b // self.T, rows] = comp[b, :n].to(torch.bfloat16)

    def stage_commit(self, chunk_rows=16384):
        """Upload the staging pool into the device pool IN PLACE and in chunks (``paged_scatter_rows``: no second pool-sized device buffer, no host fp32
        copy for an fp8 pool: the kernel converts bf16 -> e4m3). All-zero chunks are skipped."""
        mapper = ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(self.rows, self.cols))
        rep = ttnn.ReplicateTensorToMesh(self.md)
        ids = {}
        for a in range(0, self.total_rows, chunk_rows):
            n = min(chunk_rows, self.total_rows - a)
            chunk = self._stage[:, a : a + n]
            if not bool(chunk.any()):
                continue
            if n not in ids:
                ids[n] = ttnn.from_torch(
                    torch.arange(n, dtype=torch.int32).reshape(1, n),
                    device=self.md,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=rep,
                )
            src = ttnn.from_torch(
                chunk.reshape(self.rows, 1, n, HEAD_DIM).contiguous(),
                device=self.md,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            paged_scatter_rows(self.pool, src, ids[n], base_offset=a)
            ttnn.synchronize_device(self.md)
            ttnn.deallocate(src)
        for t in ids.values():
            ttnn.deallocate(t)
        self._stage = None


def paged_kv_step(
    pool,
    kv,
    lat,
    pos,
    page_table,
    ids,
    *,
    ring_base,
    layer_key,
    ratio,
    src_off,
    topk_out,
    ring_rows=RING_ROWS,
    kv_mode=0,
    nq=1,
    window=WINDOW,
    ratio_page_tokens=PAGE_TOKENS,
    write_kv=True,
):
    """One op: ring write (+ latent write when ``lat`` is given) + sparse_sdpa index rows. Returns ``indices`` uint32 [1,1,rows,topk_out].

    pool: the pool tensor (written in place); kv: bf16 TILE ([1,T,32,512] for kv_mode 0, else [1,1,rows,512]); lat: bf16 TILE [1,1,rows,512] or None;
    pos: int32 ROW_MAJOR [rows]; page_table: int32 ROW_MAJOR [users, MAXP]; ids: uint32 [rows,1,1,512] indexer output or None (ratio > 0 only).
    """
    md = pool.device()
    rows = int(pos.shape[-1])
    maxp = int(page_table.shape[-1])
    if (
        rows > 64
    ):  # spec verify with > 64 rows (DSV41_SPEC_ROWS): one core per row, 64 rows per call (users of the half shift ring_base / page table)
        assert kv_mode == 0 and rows % 64 == 0 and 64 % nq == 0
        outs = []
        for r0 in range(0, rows, 64):
            u0 = r0 // nq
            outs.append(
                paged_kv_step(
                    pool,
                    ttnn.slice(kv, [0, r0, 0, 0], [1, r0 + 64, kv.shape[2], kv.shape[3]]),
                    None if lat is None else ttnn.slice(lat, [0, 0, r0, 0], [1, 1, r0 + 64, lat.shape[3]]),
                    ttnn.slice(pos, [r0], [r0 + 64]),
                    ttnn.slice(page_table, [u0, 0], [u0 + 64 // nq, maxp]),
                    None if ids is None else ttnn.slice(ids, [r0, 0, 0, 0], [r0 + 64, 1, 1, ids.shape[3]]),
                    ring_base=ring_base + u0 * ring_rows,
                    layer_key=layer_key,
                    ratio=ratio,
                    src_off=src_off,
                    topk_out=topk_out,
                    ring_rows=ring_rows,
                    kv_mode=kv_mode,
                    nq=nq,
                    window=window,
                    ratio_page_tokens=ratio_page_tokens,
                    write_kv=write_kv,
                )
            )
        return ttnn.concat(outs, dim=2)
    assert kv.dtype == ttnn.bfloat16 and kv.layout == ttnn.TILE_LAYOUT
    assert pool.dtype in (ttnn.bfloat16, ttnn.fp8_e4m3)
    assert pos.dtype == ttnn.int32 and page_table.dtype == ttnn.int32
    has_lat = lat is not None
    if has_lat:
        assert lat.dtype == ttnn.bfloat16 and lat.layout == ttnn.TILE_LAYOUT and rows <= 128
    ids_t = ids if ids is not None else pos
    lat_t = lat if has_lat else kv
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, topk_out]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
    )
    cores = [ttnn.CoreCoord(i % 8, i // 8) for i in range(rows)]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    pos_b = (rows * 4 + 63) // 64 * 64
    total = pos_b + 16384 + 16384 + (maxp * 4 + 63) // 64 * 64 + 2048 + 1024 + 1024 + topk_out * 4
    total = (total + 63) // 64 * 64
    cb = ttnn.CBDescriptor(
        total_size=total,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=total)],
    )
    ct = [
        rows,
        nq,
        ring_rows,
        window,
        ratio,
        src_off,
        ratio_page_tokens,
        PAGE_ROWS,
        topk_out,
        maxp,
        int(has_lat),
        kv_mode,
        int(write_kv),
        0,
        int(pool.dtype == ttnn.fp8_e4m3),
    ]
    ct += _acc(kv) + _acc(lat_t) + _acc(pos) + _acc(page_table) + _acc(ids_t) + _acc(pool) + _acc(out)
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        rt[c.x][c.y] = [i]
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/paged_kv_step.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[
            kv.buffer_address(),
            lat_t.buffer_address(),
            pos.buffer_address(),
            page_table.buffer_address(),
            ids_t.buffer_address(),
            pool.buffer_address(),
            out.buffer_address(),
            ring_base,
        ],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
    prog.custom_program_hash = (0x9A6 << 40) | (hash((tuple(ct), layer_key)) & ((1 << 40) - 1))
    return ttnn.generic_op([kv, lat_t, pos, page_table, ids_t, pool, out], prog)


def paged_scatter_rows(pool, src, row_ids, n_cores=16, base_offset=0):
    """In place: pool[row_ids[i] + base_offset] = src[i] (``base_offset``: runtime constant, e.g. ``pool.ring_base(l)`` so one ids tensor serves all layers' rings). src: bf16 ROW_MAJOR [N, 512] (per device), row_ids: uint32 ROW_MAJOR [1, N] (0xFFFFFFFF = skip). This is the
    write path of prefill chunks and multi-row (spec decode) appends: the caller translates logical rows to physical pool rows with the page table
    (``PagedKVPool.phys_rows`` on the host, or ``paged_kv_step`` on the device for decode)."""
    md = pool.device()
    N = int(src.shape[-2])
    assert src.dtype == ttnn.bfloat16 and src.layout == ttnn.ROW_MAJOR_LAYOUT and row_ids.dtype == ttnn.uint32
    n_cores = min(n_cores, N)
    cores = [ttnn.CoreCoord(i % 8, i // 8) for i in range(n_cores)]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    total = ((N * 4 + 63) // 64) * 64 + 8 * 1024  # ids + 8 rows in flight
    cb = ttnn.CBDescriptor(
        total_size=total,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.int32, page_size=total)],
    )
    ct = [N, n_cores, 0, int(pool.dtype == ttnn.fp8_e4m3)] + _acc(src) + _acc(row_ids) + _acc(pool)
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        rt[c.x][c.y] = [i]
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/row_scatter.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=[src.buffer_address(), row_ids.buffer_address(), pool.buffer_address(), base_offset],
        config=ttnn.ReaderConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
    prog.custom_program_hash = (0x9A7 << 40) | (hash(tuple(ct)) & ((1 << 40) - 1))
    return ttnn.generic_op([src, row_ids, pool], prog)
