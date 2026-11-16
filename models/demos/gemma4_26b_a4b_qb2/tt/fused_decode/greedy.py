# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 greedy decode terminal: the argmax of row 0 of the vocab-sharded logits, written straight into the decode
token input (kernels/greedy_*.cpp). Replaces the common sampler's top-k pipeline for greedy batch-1 requests (the
final-logit softcap is monotonic, so the argmax is taken on the raw LM-head output; the softcap's BF16 ties are
resolved to the lowest id through a per-key class table, as the common sampler's greedy tie-break).

  1. per device, one program: each worker core scans row 0 of a range of logits tiles (BF16 bits mapped to an
     order-preserving key) and posts (key, id) to a merge core, which writes the device's best (key, global id);
  2. an all-gather of that one tile over the mesh row;
  3. one core per device picks the best of the devices (ties after the softcap: the lowest id) and writes it as
     element 0 of the [1, 1, 1, 32] UINT32 token tensor (identical on every device).
"""

from pathlib import Path

import torch

import ttnn

KERNELS = Path(__file__).resolve().parent / "kernels"
TILE = 32


class GreedyB1:
    def __init__(self, mesh, vocab_per_device, softcap=None):
        """softcap: the model's final-logit softcap as a device function (BF16 TILE -> BF16); ties after it pick the
        lowest id, as the common sampler's greedy path (`classes`). None: plain argmax."""
        assert vocab_per_device % TILE == 0
        self.mesh = mesh
        self.vd = vocab_per_device
        self.n_tiles = vocab_per_device // TILE
        grid = mesh.compute_with_storage_grid_size()
        self.merge = ttnn.CoreCoord(grid.x - 1, grid.y - 1)
        self.workers = [
            ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) != (self.merge.x, self.merge.y)
        ]
        per = -(-self.n_tiles // len(self.workers))
        self.ranges = [(j, min(j + per, self.n_tiles)) for j in range(0, self.n_tiles, per)]
        self.workers = self.workers[: len(self.ranges)]
        self.per = per
        self.merge_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.merge, self.merge)])
        self.worker_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.workers])
        self.all_set = self.worker_set.merge(self.merge_set)
        mnoc = mesh.worker_core_from_logical_core(self.merge)
        self.merge_noc = (mnoc.x, mnoc.y)
        self.local = ttnn.from_torch(
            torch.zeros(1, 1, TILE, TILE, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.merge_set, (TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.classes = self._classes(mesh, softcap)
        self.pick_core = ttnn.CoreCoord(0, 0)
        self.pick_set = ttnn.CoreRangeSet([ttnn.CoreRange(self.pick_core, self.pick_core)])

    @staticmethod
    def _classes(mesh, softcap):
        """UINT32 [65536] DRAM table: order-preserving BF16 key k -> the first key of its run of equal softcapped values
        (computed with the device's own softcap op on every BF16 bit pattern)."""
        keys = torch.arange(65536, dtype=torch.int64)
        bits = torch.where(keys >= 0x8000, keys & 0x7FFF, (~keys) & 0xFFFF)  # key -> BF16 bits
        values = bits.to(torch.int16).view(torch.bfloat16).float()
        if softcap is None:
            capped = values
        else:
            x = ttnn.from_torch(
                values.reshape(1, 1, 32, 2048).bfloat16(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            capped = ttnn.to_torch(ttnn.get_device_tensors(softcap(x))[0]).float().reshape(-1)
        start = torch.empty(65536, dtype=torch.int64)
        first = 0
        for k in range(65536):  # equal neighbours (NaN never equal) share the class of the first one
            if k and not (capped[k] == capped[k - 1]):
                first = k
            start[k] = first
        return ttnn.from_torch(
            start.int().reshape(1, 1, 1, 65536),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def _cb(self, index, size, cores):
        fmt = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=size)]
        return ttnn.CBDescriptor(total_size=size, core_ranges=cores, format_descriptors=fmt)

    def __call__(self, logits, tokens):
        """logits: [1, 1, 1|32, vd] BF16 TILE (row 0 = this device's vocab shard, DRAM or L1); tokens: [1, 1, 1, 32]
        UINT32 row-major (DRAM). Writes the greedy token into tokens[..., 0] on every device."""
        assert logits.dtype == ttnn.bfloat16 and logits.layout == ttnn.TILE_LAYOUT
        assert logits.padded_shape[-1] == self.vd
        cb_scr, cb_slots, sem = 0, 1, 0
        cbs = [
            self._cb(cb_scr, 128 * self.per + 128, self.all_set),
            self._cb(cb_slots, 16 * len(self.workers), self.all_set),
        ]
        sems = [ttnn.SemaphoreDescriptor(sem, ttnn.CoreType.WORKER, self.all_set, 0)]
        worker_ct = [cb_scr, sem, cb_slots] + ttnn.TensorAccessorArgs(logits).get_compile_time_args()
        worker_ct += ttnn.TensorAccessorArgs(self.classes).get_compile_time_args()
        worker_rt = ttnn.RuntimeArgs()
        for i, (c, (j0, j1)) in enumerate(zip(self.workers, self.ranges)):
            worker_rt[c.x][c.y] = [
                logits.buffer_address(),
                j0,
                j1,
                self.merge_noc[0],
                self.merge_noc[1],
                i,
                self.classes.buffer_address(),
            ]
        mesh_program = ttnn.MeshProgramDescriptor()
        for d in range(self.mesh.shape[1]):
            coord = ttnn.MeshCoordinate(0, d)
            merge_rt = ttnn.RuntimeArgs()
            merge_rt[self.merge.x][self.merge.y] = [self.local.buffer_address(), d * self.vd]
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNELS / "greedy_local.cpp"),
                    core_ranges=self.worker_set,
                    compile_time_args=worker_ct,
                    runtime_args=worker_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNELS / "greedy_merge.cpp"),
                    core_ranges=self.merge_set,
                    compile_time_args=[len(self.workers), sem, cb_slots],
                    runtime_args=merge_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                    ),
                ),
            ]
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(
                kernels=kernels, semaphores=sems, cbs=cbs
            )
        ttnn.generic_op([logits, self.classes, self.local], mesh_program)
        return self.finish(tokens)

    def finish(self, tokens):
        """All-gather of the devices' (key, class, id) tiles and the pick into tokens[..., 0] (also used by
        fused_decode/lmhead.py, whose LM-head program writes the per-device tile itself)."""
        cb_scr = 0
        gathered = ttnn.all_gather(
            self.local,
            dim=3,
            cluster_axis=1,
            num_links=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        pick_rt = ttnn.RuntimeArgs()
        pick_rt[self.pick_core.x][self.pick_core.y] = [gathered.buffer_address(), tokens.buffer_address()]
        pick_ct = [cb_scr, self.mesh.shape[1]]
        pick_ct += ttnn.TensorAccessorArgs(gathered).get_compile_time_args()
        pick_ct += ttnn.TensorAccessorArgs(tokens).get_compile_time_args()
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNELS / "greedy_pick.cpp"),
                    core_ranges=self.pick_set,
                    compile_time_args=pick_ct,
                    runtime_args=pick_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                    ),
                )
            ],
            semaphores=[],
            cbs=[self._cb(cb_scr, 256, self.pick_set)],
        )
        ttnn.generic_op([gathered, tokens], program)
        return tokens
