# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device merge of the multi-row LM head's per-core argmax lists (LinearStream out_mode 6) across cores and devices
(fused_decode/kernels/argmax_merge.cpp, one program per device): each device merges its cores' lists, exchanges its
per-row bests with the other devices through the fabric (kernels/boundary_sender.cpp) and merges those, then writes
either the verify block [last token, 7 drafts] (kind 0: drafts go straight from the drafter to the verify pass) or
[block, 8 target argmaxes] (kind 1: the host's single read per round; with `accept`, also the round's acceptance and
the next round's inputs)."""

import torch

import ttnn

from ..fused_decode.boundary import KERNEL_DIR

CORE = ttnn.CoreCoord(0, 0)


class ArgmaxGather:
    def __init__(self, mesh_device, rows, n_lists, vocab_per_device, kind):
        self.mesh_device = mesh_device
        self.rows, self.n_lists, self.vd, self.kind = rows, n_lists, vocab_per_device, kind
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])
        self.coords = [ttnn.MeshCoordinate(0, c) for c in range(mesh_device.shape[1])]
        self.ring = mesh_device.shape[1]
        self.slot = 64  # [key, id] x rows (<= 8)

        def on_core(words):
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.cores, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
            )
            return ttnn.from_torch(
                torch.zeros(1, 1, 1, words, dtype=torch.int32),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=mem,
                mesh_mapper=rep,
            )

        self.payload_buf = on_core(self.slot // 4)
        self.slots = on_core(self.ring * self.slot // 4)
        self.recv_sem = ttnn.create_global_semaphore(mesh_device, self.cores, 0)
        phys = mesh_device.worker_core_from_logical_core(CORE)
        self.noc_xy = (phys.x, phys.y)

    def _kernel(self, ct, rt_args, scratch, defines=None):
        rt = ttnn.RuntimeArgs()
        rt[CORE.x][CORE.y] = rt_args
        k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "argmax_merge.cpp"),
            core_ranges=self.cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
            defines=list(defines or []),
        )
        cb = ttnn.CBDescriptor(
            total_size=scratch,
            core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)],
        )
        return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])

    SEND_SEM = 1  # merge -> sender (same core)

    def __call__(self, lists, tokens_src, out, accept=None):
        """lists: the LM head's [1, n_lists * rows * 2] UINT32 lists; tokens_src: kind 0 the drafter's token input
        (element 0 = the block's first token), kind 1 the verify block tokens; out: kind 0 the verify token input
        [1, rows] UINT32, kind 1 a [1, 2 * rows] UINT32 result buffer; accept = (mask id, [verify pos, t_toks, t_pos,
        t_rot, t_bound, t_cpos, t_crot]): device rounds (kind 1)."""
        rows, mesh = self.rows, self.mesh_device
        lists_page = self.n_lists * rows * 8
        src_page = tokens_src.shape[-1] * 4
        out_page = out.shape[-1] * 4
        ct = [rows, self.n_lists, self.kind, 0, lists_page, out_page, src_page]
        for t in (lists, out, tokens_src):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        io = [lists, tokens_src, out, self.payload_buf, self.slots]
        defines = [("ARGMAX_RING", str(self.ring)), ("ARGMAX_SEND_SEM", str(self.SEND_SEM))]
        acc_addrs = []
        if accept is not None:
            mask_id, ts = accept
            for t in ts:
                ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
            acc_addrs = [t.buffer_address() for t in ts]
            io += list(ts)
            defines += [("ACCEPT", "1"), ("ACCEPT_MASK_ID", str(mask_id))]
        scratch = max(lists_page, 64) + src_page + 2048
        fwd_hops = self.ring // 2
        bwd_hops = self.ring - 1 - fwd_hops
        sem_addr = ttnn.get_global_semaphore_address(self.recv_sem)
        mesh_program = ttnn.MeshProgramDescriptor()
        for d, coord in enumerate(self.coords):
            program = self._kernel(
                ct,
                [lists.buffer_address(), out.buffer_address(), tokens_src.buffer_address(), d * self.vd]
                + [self.payload_buf.buffer_address(), self.slots.buffer_address(), sem_addr]
                + acc_addrs,
                scratch,
                defines,
            )
            program.semaphores = [ttnn.SemaphoreDescriptor(self.SEND_SEM, ttnn.CoreType.WORKER, self.cores, 0)]
            srt = ttnn.RuntimeArgs()
            srt[CORE.x][CORE.y] = [
                self.payload_buf.buffer_address(),
                self.slots.buffer_address() + d * self.slot,
                self.noc_xy[0],
                self.noc_xy[1],
                sem_addr,
            ]
            sender = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "boundary_sender.cpp"),
                core_ranges=self.cores,
                compile_time_args=[self.slot, self.slot, fwd_hops, bwd_hops, 1, self.SEND_SEM, 0, 1, 1, 0, 0],
                runtime_args=srt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                ),
            )
            program.kernels = list(program.kernels) + [sender]
            src = mesh.get_fabric_node_id(coord)
            fabric_args = []
            for step, hops in ((1, fwd_hops), (-1, bwd_hops)):
                fabric_args.append(1 if hops > 0 else 0)
                if hops > 0:
                    dst = ttnn.MeshCoordinate(coord[0], (coord[1] + step) % self.ring)
                    fabric_args += ttnn.setup_fabric_connection(
                        src_fabric_node_id=src,
                        dst_fabric_node_id=mesh.get_fabric_node_id(dst),
                        link_idx=0,
                        program_descriptor=program,
                        worker_core=CORE,
                    )
            program.kernels[len(program.kernels) - 1].runtime_args[CORE.x][CORE.y].extend(fabric_args)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        ttnn.generic_op(io, mesh_program)
