# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Engram tables on device: rows sharded over all chips (global row r -> chip r // rows_per_chip, chips in mesh row-major
order), raw fp8 e4m3 rows (uint8 [rpc,256], 256 B pages) + e8m0 scales (uint8 [ceil(rpc/8),64]: 8 B/row, 8 rows per 64 B
page, so every NOC read is 64 B aligned) per chip; a gather+dequant generic_op (data-movement kernels) and the
cross-chip combine (reduce_scatter over mesh rows + all_reduce over mesh columns)."""

import time

import numpy as np
import torch

import ttnn

KDIR = "models/demos/blackhole/deepseek_v41_flash/tt/engram_hash_kernels"
u8, bf16, i32 = ttnn.uint8, ttnn.bfloat16, ttnn.int32
K = 24  # ids per (user, layer)


def rows_per_chip(num_rows, n_chips=32):
    """Rows per chip, rounded up to a multiple of 8 so every chip's scale slice is a whole number of 64 B pages and every
    chip's slice of the checkpoint is contiguous + page aligned (the host shards can then be zero-copy memmap views)."""
    return -(-num_rows // (n_chips * 8)) * 8


def alloc_table(mesh, rpc):
    """Per-chip device tensors (uninitialised): rows [rpc,256] u8, scales [ceil(rpc/8),64] u8 (DRAM interleaved)."""
    mk = lambda shape: ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), u8, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    return mk([rpc, 256]), mk([-(-rpc // 8), 64])


def shard_host(table, chip, rpc):
    """Host tensors (rows [rpc,256] u8, scales [rpc/8,64] u8) for chip's rows of a _MappedTable. Zero-copy views of the
    memory-mapped checkpoint unless the chip's range is ragged (last chip): then a zero-padded copy."""
    n = table.w.shape[0]
    a, b = min(chip * rpc, n), min((chip + 1) * rpc, n)
    if b - a == rpc:
        w, s = table.w[a:b].view(torch.uint8), table.s[a:b].view(torch.uint8).reshape(-1, 64)
    else:
        w = torch.zeros((rpc, 256), dtype=torch.uint8)
        w[: b - a] = table.w[a:b].view(torch.uint8)
        s = torch.zeros(rpc * 8, dtype=torch.uint8)
        s[: (b - a) * 8] = table.s[a:b].view(torch.uint8).reshape(-1)
        s = s.reshape(-1, 64)
    h = lambda t: ttnn.from_torch(t, dtype=u8, layout=ttnn.ROW_MAJOR_LAYOUT)
    return h(w), h(s)


def load_table(mesh, dev_rows, dev_scales, table, rpc, chips=None):
    """Stream the shards of `table` (a _MappedTable) into the per-chip device tensors: one host-shard list over the mesh, so
    ONE write fills every chip (copy_host_to_device_tensor on a single get_device_tensors() view writes ALL chips).
    chips = iterable of chips to really load (others get a copy of chip chips[0]'s shard: reduced-table test); None = all.
    Returns (host_build_s, device_write_s)."""
    n = mesh.get_num_devices()
    chips = list(range(n)) if chips is None else list(chips)
    t0 = time.perf_counter()
    built = {c: shard_host(table, c, rpc) for c in chips}
    hw = [built.get(c, built[chips[0]])[0] for c in range(n)]
    hs = [built.get(c, built[chips[0]])[1] for c in range(n)]
    t1 = time.perf_counter()
    for host, dev in ((hw, dev_rows), (hs, dev_scales)):
        ttnn.copy_host_to_device_tensor(ttnn.from_host_shards(host, mesh.shape), dev)
    ttnn.synchronize_device(mesh)
    return t1 - t0, time.perf_counter() - t1


class DSV41EngramGather:
    """tables: [(rows, scales, num_rows), (rows, scales, num_rows)] for the 2 Engram layers (device tensors from alloc_table)."""

    def __init__(self, mesh, tables, users=16, n_cores=96):
        self.mesh, self.U, self.tables = mesh, users, tables
        n = mesh.get_num_devices()
        self.NROWS = users * 2 * K
        assert self.NROWS % (2 * n_cores) == 0 and users % 4 == 0
        self.n_cores = n_cores
        self.per = self.NROWS // n_cores // 2  # rows per kernel instance (2 instances per core: both RISC-V DM cores)
        info = np.zeros((n, 16), dtype=np.int32)
        for c in range(n):
            for l, (_, _, nr) in enumerate(tables):
                rpc = rows_per_chip(nr, n)
                info[c, 2 * l] = c * rpc
                info[c, 2 * l + 1] = max(0, min(nr, (c + 1) * rpc) - c * rpc)
        self.info = ttnn.from_torch(
            torch.from_numpy(info),
            device=mesh,
            dtype=i32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        self.out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.NROWS, 256]), bf16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )

    def upload_ids(self, ids):
        """ids int32 [U,64] (all users; cols layer*24+k = global rows) replicated to every chip."""
        return ttnn.from_torch(
            ids.to(torch.int32),
            device=self.mesh,
            dtype=i32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def __call__(self, ids):
        grid = self.mesh.compute_with_storage_grid_size()
        cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)][: self.n_cores]
        core_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores]
        )
        per = self.per
        cbsz = (self.U * 256 + 64 + per * (256 + 64 + 512) + 63) // 64 * 64
        cbs = [
            ttnn.CBDescriptor(
                total_size=cbsz,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=u8, page_size=cbsz)],
            )
            for i in range(2)
        ]
        (t0, s0, _), (t1, s1, _) = self.tables
        common = [x.buffer_address() for x in (ids, self.info, t0, s0, t1, s1, self.out)]
        acc = _acc(ids) + _acc(self.info) + _acc(t0) + _acc(s0) + _acc(t1) + _acc(s1) + _acc(self.out)
        kernels = []
        for inst, cfg in enumerate((ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor())):
            rt = ttnn.RuntimeArgs()
            for k, (cx, cy) in enumerate(cores):
                rt[cx][cy] = [(2 * k + inst) * per, per]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/engram_gather.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_set,
                    compile_time_args=[self.U, 2, K, per, inst] + acc,
                    runtime_args=rt,
                    common_runtime_args=common,
                    config=cfg,
                )
            )
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        prog.custom_program_hash = (0x3E7 << 40) | (hash((self.U, self.n_cores, tuple(acc))) & ((1 << 40) - 1))
        ttnn.generic_op([ids, self.info, t0, s0, t1, s1, self.out], prog)
        return self.out


def combine(partial, mesh, num_links=2, topology=ttnn.Topology.Linear):
    """partial [1,1,NROWS,256] bf16 per chip (rows ordered (user group g, layer, user-in-group, k)) -> reduce_scatter over mesh
    rows (axis 0, dim 2) then all_reduce over mesh columns (axis 1): chip (r, *) ends with the summed rows of user group r.
    """
    rs = ttnn.reduce_scatter(partial, dim=2, cluster_axis=0, num_links=num_links, topology=topology)
    return ttnn.all_reduce(rs, cluster_axis=1, num_links=num_links, topology=topology)
