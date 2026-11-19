# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention of the DFlash verify block with one K / V cache read for all M rows (fused_decode/kernels/ba_*.cpp),
row attention's design (dflash/row_attention.py: its own kernels, fused into the QKV program so the history is read
while QKV runs) with the rows sharing the reads:

  - the M rows' queries of kv head g are Tt = M / 4 packed tiles (tile t, rows 8 i .. 8 i + 7 = block row 4 t + i's
    query heads 8 g .. 8 g + 7: every tile row is real; the verify QKV writes Q in this layout with spec_tt);
  - one core per (kv head g, tile group t, history slice s), each running one packed Q tile (tile t): the block's
    history (positions 0 .. pos[M - 1]) is cut into A <= S contiguous slices of at least min_chunks 32-position chunks;
    each slice core runs its Q tile over the slice's chunks (row group i masked causally at pos[4 t + i]), in groups of
    G chunks with a running softmax (any length streams through L1);
  - the Tt = 2 tile-group cores of a (kv head, slice) share the K / V reads: the tile-group-0 core (the leader) reads
    each group from DRAM, its partner copies it from the leader's L1 (the groups holding the block's rows, written by
    QKV, the partner reads from DRAM itself);
  - the slices of each tile group merge their softmax partials (m, l, O) up a binary tree (log2 A rounds, NOC writes
    of the partial into the parent's L1); slice 0 (with the sink logits) writes out = O / l as its tile of the packed
    output, which o_proj reads with spec_tt.
"""

import struct

import torch

import ttnn

from ..fused_decode.boundary import KERNEL_DIR

TILE = ttnn.TILE_SIZE
TREE_SEM = 9
LAND_SEM, COPY_SEM = 10, 11  # the shared K / V reads: groups landed in the leader / copied by its partner
MIN_CHUNKS = 4  # 32-position chunks per slice at least
G = 8  # chunks per K / V group (one running-softmax step)
CAP = 3  # groups in flight per core (4 clashes with the verify pass's L1 buffers; at least 2: the compute kernel's
# pipelined loop computes group g + 1's scores while it still holds group g's V)
assert CAP >= 2


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class BlockAttention:
    def __init__(
        self,
        mesh_device,
        rows,
        n_heads,
        n_kv,
        head_dim,
        block_size,
        cores,
        slices,
        uniform=False,
        window=0,
        q_rows=False,
        min_chunks=None,
    ):
        """cores: cores_needed(n_kv, rows, slices) logical cores: (kv head g, tile group t, slice s) = cores[(g Tt + t)
        slices + s], each tile group (packed Q tile t, one per core) with its own slices and merge tree.
        uniform: every row attends [0, pos[M - 1]] (the drafter's block: no causal mask inside it), no sinks; window
        (uniform only): the rows attend the last `window` positions [pos[M - 1] + 1 - window, pos[M - 1]].
        q_rows: q comes in the rows layout [1, M, heads, 64] (the reader gathers the packed tiles: no repacking op).
        """
        assert head_dim == 64 and n_heads // n_kv == 8 and rows == 8
        assert uniform or not window
        self.mesh, self.M, self.nq, self.nkv, self.hd = mesh_device, rows, n_heads, n_kv, head_dim
        self.block, self.S, self.Tt = block_size, slices, rows // 4
        self.uniform, self.window, self.q_rows = uniform, window, q_rows
        # one DRAM read of each K / V group per (kv head, slice): the tile-group-1 core copies it from the tile-group-0
        # core's L1
        assert len(cores) == self.cores_needed(n_kv, rows, slices)
        self.core_list = list(cores)
        self.grid = mesh_device.compute_with_storage_grid_size()
        assert len(self.core_list) < 0xFF
        self.cores = _rects(self.core_list)
        self.min_chunks = min_chunks or MIN_CHUNKS
        self.rounds = max(1, (slices - 1).bit_length())
        self.mask_lib = ttnn.from_torch(
            self._mask_lib(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # the packed Q (a producer writes it in this layout: the verify QKV with spec_tt) and the packed output (o_proj
        # reads it with spec_tt), both in L1 (the consumer's reads skip DRAM latency; o_proj reads rows straight from
        # an L1 source, a DRAM one is staged first)
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        self.q, self.out = (
            ttnn.from_torch(
                torch.zeros(1, n_kv, self.Tt * TILE, head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            for _ in range(2)
        )

    @staticmethod
    def _mask_lib():
        """The mask library: segment (a, b, f) (columns [a, b) kept, a, b = 0 .. 32, half f) = 8 rows x 16 BF16 columns
        of a 32x32 tile face, 0 at the kept columns and -1e30 elsewhere (64 words)."""
        w = torch.zeros(33, 33, 2, 8, 8, dtype=torch.int64)
        cols = torch.arange(32).reshape(2, 8, 2)  # [f, c, lo / hi half of the word]
        for a in range(33):
            for k in range(33):
                cut = ((cols < a) | (cols >= k)).long() * 0xF14A
                w[a, k] = (cut[..., 0] | (cut[..., 1] << 16))[:, None, :].expand(2, 8, 8)
        w = w.reshape(1, -1)
        return torch.where(w >= 1 << 31, w - (1 << 32), w).to(torch.int32)

    @staticmethod
    def cores_needed(n_kv, rows, slices):
        return n_kv * (rows // 4) * slices

    def noc_cores(self):
        return [self.mesh.worker_core_from_logical_core(c) for c in self.core_list]

    def notify_order(self):
        """NOC coordinates of the cores in slice order (the producer's signals reach slice 0 first: the slices that
        have work at short context)."""
        noc = self.noc_cores()
        return [noc[i] for i in sorted(range(len(noc)), key=lambda i: (i % self.S, i))]

    def parts(self, q, k_cache, v_cache, pos, page_table, sinks, scale, wait, patch=None, out_notify=None):
        """(kernels, cbs, io); q: the packed Q [1, nkv, Tt * 32, 64] (tile (g, t, d) at page (g Tt + t) 2 + d); pos:
        INT32 [>= M] (row r at pos[r], non-decreasing, consecutive rows within 32 positions); page_table: INT32 [*,
        blocks] (row 0 used); sinks: the heads' [heads, 32] BF16 sink tile; wait = (semaphore id, count): the QKV
        writers' signal that q and the block's K / V rows are written."""
        cap, S = CAP, self.S
        sinks = sinks if sinks is not None else self.mask_lib  # uniform: no sinks (a placeholder address)
        kv_dtype = k_cache.dtype
        kv_tile = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[kv_dtype]
        pt_page = page_table.shape[-1] * 4
        pos_page = pos.shape[-1] * 4
        f32, bf16 = ttnn.float32, ttnn.bfloat16
        partial = 4  # [m, l, O_0, O_1] of the core's Q tile
        cb_q, cb_k, cb_v, cb_aux, cb_scr, cb_qs, cb_s, cb_p = range(8)
        cb_m, cb_l, cb_o = 8, 10, 12  # the running state
        cb_ab, cb_out, cb_send, cb_recv, cb_scr2, cb_lacc = range(14, 20)

        def cb(i, dtype, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=self.cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=page)],
            )

        cbs = [
            cb(cb_q, bf16, 2048, 2),
            cb(cb_k, kv_dtype, kv_tile, 2 * G * cap),
            cb(cb_v, kv_dtype, kv_tile, 2 * G * cap),
            cb(cb_aux, bf16, 2048, 6),
            cb(cb_scr, bf16, 256 + max(pt_page, 64), 1),
            cb(cb_qs, bf16, 2048, 2),
            cb(cb_s, f32, 4096, 2 * (G + 1)),  # 2 groups: the next group's scores while P of this one runs
            cb(cb_p, bf16, 2048, G + 1),
            cb(cb_ab, f32, 4096, 2),
            cb(cb_out, bf16, 2048, 2),
            cb(cb_send, f32, 4096, partial),
            cb(cb_recv, f32, 4096, partial * self.rounds),
            cb(cb_scr2, bf16, max(64, pos_page), 1),
            cb(cb_lacc, f32, 4096, 1),
            # the running state, double-buffered (the update pushes before it pops)
            cb(cb_m, f32, 4096, 2),
            cb(cb_l, f32, 4096, 2),
            cb(cb_o, f32, 4096, 4),
        ]
        reader_ct = [
            cb_q,
            cb_k,
            cb_v,
            cb_aux,
            cb_scr,
            cb_recv,
            self.M,
            self.nkv,
            self.block,
            kv_tile,
            pt_page,
            pos_page,
        ]
        reader_ct += [S, self.min_chunks, G, cap, partial]
        for t in (q, k_cache, v_cache, pos, page_table, sinks, self.mask_lib):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [cb_out, cb_scr2, cb_send, cb_recv, self.M, S, self.min_chunks, partial]
        writer_ct += ttnn.TensorAccessorArgs(self.out).get_compile_time_args()
        compute_ct = [cb_q, cb_k, cb_v, cb_aux, cb_qs, cb_s, cb_p, cb_m, cb_l, cb_o, cb_ab, cb_out, cb_send, cb_recv]
        compute_ct += [cb_lacc, _f32_bits(scale), self.M, G]
        # every argument is common (no per-core runtime args: dispatch writes those core by core): each core finds its
        # index in the core list from its logical coordinates (a byte table over the grid) and its tree parent's NOC
        # coordinates from a table over the core list
        noc = self.noc_cores()
        gx, gy = self.grid.x, self.grid.y
        idx = [0xFF] * (gx * gy)
        for i, c in enumerate(self.core_list):
            idx[c.y * gx + c.x] = i
        idx += [0xFF] * (-len(idx) % 4)
        idx_words = [sum(idx[w + k] << (8 * k) for k in range(4)) for w in range(0, len(idx), 4)]
        nocs = [(p.x << 8) | p.y for p in noc] + [0] * (len(noc) % 2)
        noc_words = [nocs[w] | (nocs[w + 1] << 16) for w in range(0, len(nocs), 2)]
        reader_common = (
            [
                q.buffer_address(),
                k_cache.buffer_address(),
                v_cache.buffer_address(),
                pos.buffer_address(),
                page_table.buffer_address(),
                sinks.buffer_address(),
                self.mask_lib.buffer_address(),
            ]
            + idx_words
            + noc_words
        )
        if patch is not None:
            # patch = (k, v, noc coords (x, y) per row): the head-sharded K / V the block's rows come from (uniform,
            # BF16 cache; the wait semaphore signals them written, dflash/kv_write.py DraftKVWrite.parts notify)
            assert self.uniform and kv_dtype == ttnn.bfloat16 and len(patch[2]) >= self.M
            patch_arg = len(reader_common)
            pn = [(x << 8) | y for x, y in patch[2][: self.M]] + [0] * (self.M % 2)
            reader_common += [patch[0].buffer_address(), patch[1].buffer_address()]
            reader_common += [pn[w] | (pn[w + 1] << 16) for w in range(0, len(pn), 2)]
        writer_common = [self.out.buffer_address()] + idx_words + noc_words
        defines = [("BA_WAIT_SEM", str(wait[0])), ("BA_WAIT_N", str(wait[1])), ("BA_TREE_SEM", str(TREE_SEM))]
        defines += [("BA_TQ", str(self.Tt)), ("BA_GX", str(gx))]
        defines += [("BA_NIDX", str(len(idx_words))), ("BA_CB_POS", str(cb_scr2))]
        defines += [("BA_LAND_SEM", str(LAND_SEM)), ("BA_COPY_SEM", str(COPY_SEM))]
        if self.uniform:
            defines.append(("BA_UNIFORM", "1"))
        if self.q_rows:
            defines.append(("BA_Q_ROWS", "1"))
        if patch is not None:
            defines.append(("BA_PATCH", str(patch_arg)))
        if out_notify is not None:  # (noc cores, semaphore id): the roots signal a consumer once out is written
            nw = [(p.x << 8) | p.y for p in out_notify[0]] + [0] * (len(out_notify[0]) % 2)
            defines += [
                ("BA_OUT_NOTIFY_N", str(len(out_notify[0]))),
                ("BA_OUT_NOTIFY_SEM", str(out_notify[1])),
                ("BA_OUT_NOTIFY_ARG", str(len(writer_common))),
            ]
            writer_common += [nw[w] | (nw[w + 1] << 16) for w in range(0, len(nw), 2)]
        if self.window:  # one window for all rows
            defines.append(("BA_WINDOW", str(self.window)))
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "ba_reader.cpp"),
                core_ranges=self.cores,
                compile_time_args=reader_ct,
                runtime_args=[],
                common_runtime_args=reader_common,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "ba_writer.cpp"),
                core_ranges=self.cores,
                compile_time_args=writer_ct,
                runtime_args=[],
                common_runtime_args=writer_common,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "ba_compute.cpp"),
                core_ranges=self.cores,
                compile_time_args=compute_ct,
                runtime_args=[],
                defines=[(k, v) for k, v in defines if k == "BA_UNIFORM"],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                ),
            ),
        ]
        io = [q, k_cache, v_cache, pos, page_table, sinks, self.mask_lib, self.out]
        return kernels, cbs, [t for i, t in enumerate(io) if all(t is not u for u in io[:i])]

    def semaphores(self):
        """The op's own program semaphores (besides the wait semaphore)."""
        return [
            ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, self.cores, 0) for i in (TREE_SEM, LAND_SEM, COPY_SEM)
        ]


def _rects(cores):
    """The cores as few rectangles as possible (row runs, stacked when consecutive rows share the run): dispatch writes
    kernel binaries and arguments per core range."""
    rows = {}
    for c in cores:
        rows.setdefault(c.y, []).append(c.x)
    runs = []  # (x0, x1, y)
    for y in sorted(rows):
        xs = sorted(rows[y])
        x0 = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is None or x != prev + 1:
                runs.append((x0, prev, y))
                x0 = x
            prev = x if x is not None else prev
    rects = []  # [x0, x1, y0, y1]
    for x0, x1, y in runs:
        r = next((r for r in rects if r[0] == x0 and r[1] == x1 and r[3] == y - 1), None)
        if r is not None:
            r[3] = y
        else:
            rects.append([x0, x1, y, y])
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, c), ttnn.CoreCoord(b, d)) for a, b, c, d in rects])


def free_cores(mesh_device, taken, n):
    """n logical worker cores not in taken ((x, y) set), row-major from the grid's top-left."""
    grid = mesh_device.compute_with_storage_grid_size()
    out = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
    assert len(out) >= n, f"{len(out)} free cores, {n} needed"
    return out[:n]
