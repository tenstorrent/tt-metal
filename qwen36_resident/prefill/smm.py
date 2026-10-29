# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small-M streaming matmul for short prefill chunks (kernels/smm_*.cpp): out [M, N] = x [M, K] @ W [K, N].

At a few hundred rows a projection is bound by reading W. Each core of a Q x R grid owns nt output tile
columns and streams its slab of W from a bank-contiguous DRAM layout on both NOCs; x goes K block by K
block to every core, read by one core per grid row and multicast along it."""
import math

import torch

import ttnn
from qwen36_resident import PKG_DIR

TILE = 32
KDIR = f"{PKG_DIR}/prefill/kernels/"
TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}


class SmallMatmul:
    def __init__(self, mesh, M, K, N, wdtype, kb=8, ring=3):
        self.mesh, self.n = mesh, mesh.get_num_devices()
        grid = mesh.compute_with_storage_grid_size()
        self.Q, self.R = grid.x, grid.y
        self.P = self.Q * self.R
        self.Mt, self.Kt, self.Nt = M // TILE, K // TILE, N // TILE
        self.nt = math.ceil(self.Nt / self.P)
        assert self.Kt % kb == 0
        self.kb, self.ring, self.wdtype = kb, ring, wdtype
        self.blocks = self.Kt // kb
        self.banks = mesh.dram_grid_size().x
        self.per_bank = math.ceil(self.P / self.banks)
        self.bank_tiles = self.blocks * self.per_bank * kb * self.nt

    def weights(self, w):
        """[n, K, N] (one matrix per chip) -> the bank-major DRAM layout: block b of core j's column slab
        (kb x nt tiles, K-major) in bank j % banks at ((b * per_bank + j // banks) * kb * nt) tiles"""
        n, nt, kb, B, S = self.n, self.nt, self.kb, self.banks, self.per_bank
        Pp = S * B  # cores rounded up to whole banks; core j = slot * banks + bank
        wp = torch.zeros(n, self.Kt * TILE, Pp * nt * TILE, dtype=torch.bfloat16)
        wp[:, :, : w.shape[2]] = w
        t = wp.view(n, self.blocks, kb, TILE, S, B, nt, TILE).permute(0, 5, 1, 4, 2, 6, 3, 7)  # n B b S kb nt y x
        host = t.reshape(n, B, self.bank_tiles, TILE, TILE).permute(0, 3, 1, 2, 4).reshape(n * TILE, -1)
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(B - 1, 0))})
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, [TILE, self.bank_tiles * TILE], ttnn.ShardOrientation.ROW_MAJOR),
        )
        return ttnn.from_torch(
            host,
            dtype=self.wdtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
            memory_config=mc,
        )

    def __call__(self, x, w, out, silu=False):
        mesh, Q, R, Mt, nt, kb = self.mesh, self.Q, self.R, self.Mt, self.nt, self.kb
        cores = [ttnn.CoreCoord(q, r) for r in range(R) for q in range(Q)]
        phys = {(c.x, c.y): mesh.worker_core_from_logical_core(c) for c in cores}
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(Q - 1, R - 1))])
        bf = TILE_BYTES[ttnn.bfloat16]
        wt = TILE_BYTES[self.wdtype]
        sw = max(d for d in range(1, 9) if nt % d == 0)
        sh = max(d for d in range(1, 9) if Mt % d == 0 and d * sw <= 8)

        def cb(idx, dt, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
            )

        cbs = [
            cb(0, ttnn.bfloat16, bf, 2 * Mt * kb),
            cb(1, self.wdtype, wt, self.ring * kb * nt),
            cb(2, ttnn.bfloat16, bf, Mt * kb),
            cb(24, ttnn.bfloat16, bf, Mt * nt),
            cb(16, ttnn.bfloat16, bf, Mt * nt),
        ]
        r0, r1 = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        a, b = phys[(0, 0)], phys[(Q - 1, R - 1)]
        peers = [v for c in cores for v in (phys[(c.x, c.y)].x, phys[(c.x, c.y)].y)]
        for j, c in enumerate(cores):
            cols = max(0, min(nt, self.Nt - j * nt))
            r0[c.x][c.y] = [j, a.x, a.y, b.x, b.y, out.buffer_address(), j * nt, cols] + peers
            r1[c.x][c.y] = [j, w.buffer_address(), cols, x.buffer_address()]
        compute = ttnn.ComputeConfigDescriptor()
        compute.math_fidelity = ttnn.MathFidelity.LoFi
        compute.fp32_dest_acc_en = False
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=KDIR + src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=grid,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        kernels = [
            kd(
                "smm_in0.cpp",
                [Mt, kb, self.blocks, self.P, self.Nt, nt],
                r0,
                dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
            ),
            kd(
                "smm_in1.cpp",
                [kb, nt, self.blocks, self.ring, self.banks, self.per_bank, Mt, self.Kt],
                r1,
                dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
            ),
            kd("smm_compute.cpp", [Mt, kb, nt, self.blocks, sh, sw, int(silu)], ttnn.RuntimeArgs(), compute),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=int(i == 4)) for i in range(6)]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        pd = ttnn.MeshProgramDescriptor()
        pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, self.n - 1))] = program
        ttnn.generic_op([x, w, out], pd)
        return out
