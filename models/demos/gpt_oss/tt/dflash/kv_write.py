# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""K / V rows of one sequence -> its paged BF16 draft cache in one single-core op (fused_decode/kernels/
draft_kv_write.cpp): replaces one paged_update_cache per row (the op's per-row tile read-modify-write races when rows of
a call share a tile), and takes its positions from a device tensor, so it can be traced."""

import ttnn

from ..fused_decode.boundary import KERNEL_DIR

TILE = ttnn.TILE_SIZE
CORE = ttnn.CoreCoord(0, 0)
HEAD_GRID_X = 8  # the head-sharded K / V tensors' grid width (user r on core (r % 8, r // 8) from core_offset)


class DraftKVWrite:
    def __init__(self, mesh_device, rows, kv_heads, head_dim, block_size, core_offset=(0, 0)):
        """rows: rows written per call (row r of the K / V head tensors: user r, sharded one user per core on an
        8-wide row-major grid from core_offset)."""
        self.mesh_device = mesh_device
        self.rows, self.kv_heads, self.head_tiles, self.block = rows, kv_heads, head_dim // TILE, block_size
        self.coords = []
        for r in range(rows):
            x0, y0 = core_offset
            p = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x0 + r % HEAD_GRID_X, y0 + r // HEAD_GRID_X))
            self.coords += [p.x, p.y]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])
        # scratch: a 512-byte header (256 bytes per K / V kernel: positions, page-table rows), then per layer the
        # landed cache tiles (K / V, the 2 touched 32-row tiles, kv heads, head tiles)
        self.scratch = 512 + 4 * kv_heads * self.head_tiles * 2048

    def write_layers(self, k, v, caches, pos, page_table, cores=1):
        """One program for several layers' caches (the drafter's context pass): layer l's K / V heads are head rows
        l * kv_heads .. of k / v; caches = [(k_cache, v_cache)] per layer. cores > 1: the layers split over that many
        cores (CORE, CORE + (1, 0), ..), each writing its own in parallel."""
        n = len(caches)
        cores = max(1, min(cores, n))
        kernels, cbs = [], []
        for g in range(cores):
            l0, l1 = g * n // cores, (g + 1) * n // cores
            c = ttnn.CoreCoord(CORE.x + g, CORE.y)
            kk, kc = self.parts(
                k,
                v,
                caches[l0][0],
                caches[l0][1],
                pos,
                page_table,
                head_offset=l0 * self.kv_heads,
                core=c,
                more_caches=caches[l0 + 1 : l1],
            )
            kernels += kk
            cbs += kc
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        io = [k, v, pos, page_table] + [t for kc_vc in caches for t in kc_vc]
        ttnn.generic_op(io, program)

    def parts(
        self,
        k,
        v,
        k_cache,
        v_cache,
        pos,
        page_table,
        head_offset=0,
        core=CORE,
        cb_id=0,
        wait=None,
        more_caches=(),
        notify=None,
    ):
        """(kernels, cbs) of the write on `core`; wait = (first semaphore id, count): the K / V kernel first waits for
        `count` increments of semaphore id / id + 1 (fused after the producer of k / v, dflash/qk_prep.py); notify =
        (noc cores, semaphore id): each kernel increments it on every core once its source shards are written (the
        consumer reads the block's rows from the shards, dflash/block_attention.py patch)."""
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        ct = [cb_id, self.rows, self.kv_heads, self.head_tiles, self.block, page_table.shape[-1] * 4, head_offset]
        for t in (k_cache, v_cache, pos, page_table):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        rt[core.x][core.y] = [
            k.buffer_address(),
            v.buffer_address(),
            k_cache.buffer_address(),
            v_cache.buffer_address(),
            pos.buffer_address(),
            page_table.buffer_address(),
        ] + self.coords
        rt[core.x][core.y] = list(rt[core.x][core.y]) + [t.buffer_address() for kc_vc in more_caches for t in kc_vc]
        extra = [] if wait is None else [("KV_WAIT_SEM", str(wait[0])), ("KV_WAIT_N", str(wait[1]))]
        if more_caches:
            extra.append(("KV_LAYERS", str(1 + len(more_caches))))
        if notify is not None:  # (noc cores, semaphore id): both K and V kernels signal each core once written
            n_args = len(rt[core.x][core.y])
            rt[core.x][core.y] = list(rt[core.x][core.y]) + [v_ for p in notify[0] for v_ in (p.x, p.y)]
            extra += [
                ("KV_NOTIFY_N", str(len(notify[0]))),
                ("KV_NOTIFY_SEM", str(notify[1])),
                ("KV_NOTIFY_ARG", str(n_args)),
            ]
        # K on BRISC / NOC0 and V on NCRISC / NOC1 of the same core, in parallel
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "draft_kv_write.cpp"),
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=(ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1)[sel],
                    noc=(ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)[sel],
                ),
                defines=[("KV_SEL", str(sel))] + extra,
            )
            for sel in (0, 1)
        ]
        scratch = self.scratch + len(more_caches) * 4 * self.kv_heads * self.head_tiles * 2048
        cb = ttnn.CBDescriptor(
            total_size=scratch,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=ttnn.bfloat16, page_size=scratch)
            ],
        )
        return kernels, [cb]
