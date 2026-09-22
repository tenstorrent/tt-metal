# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact QB2 reduction descriptors using the current native direct kernels.

Two workers own one fabric link each and sixteen output tiles. A native
invocation-start barrier permits one staging buffer without parity halves. Arithmetic, source order, fabric protocol
and monotonic arrival counters are the native direct reduce-scatter's.
"""

from pathlib import Path

import ttnn

from .mlp import _grid


class CompactReduceScatter:
    def __init__(self, mesh, output_memory_config):
        if tuple(mesh.shape) != (1, 4) or ttnn.get_fabric_config() != ttnn.FabricConfig.FABRIC_1D_RING:
            raise ValueError("Compact reduction requires the four-chip QB2 1D ring")
        self.mesh = mesh
        self.cores = [ttnn.CoreCoord(x, 4) for x in range(2)]
        self.grid = _grid(self.cores)
        # Receive staging is a program-local CB. The native all-peer start
        # barrier prevents an early sender overwriting a peer's preceding op.
        # This leaves the large native prefill/head L1 workspace available.
        self.output = ttnn.empty(
            (1, 1, 1, 1024),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=output_memory_config,
        )
        # Native contract: arrivals[4], reader_gen, writer_gen, compute_gen,
        # init_sync. Keep these alive through all cached programs and traces.
        self.semaphores = [ttnn.create_global_semaphore(mesh, self.grid, 0, ttnn.BufferType.L1_SMALL) for _ in range(8)]
        self.addresses = [ttnn.get_global_semaphore_address(s) for s in self.semaphores]
        ttnn.synchronize_device(mesh)

    def append(self, program, input_tensor, rank, *, wait_for_mlp=False, residual=None):
        if tuple(input_tensor.shape) != (1, 1, 1, 4096) or input_tensor.dtype != ttnn.bfloat16:
            raise ValueError("Expected BF16 batch-one four-way reduction input")
        # All descriptors in this body use disjoint worker cores. The local MLP
        # producer releases semaphore 5 after every down shard's NoC write.
        native = "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_direct/device/kernels/"
        source_reader = (
            str(Path(__file__).with_name("kernels") / "reduce_reader.cpp")
            if wait_for_mlp
            else native + "reduce_scatter_minimal_direct_reader.cpp"
        )
        accessor = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()
        reader_ct = [2048, 8, 4, 32, 32, 128, 4, 0, 1, 1, 0] + accessor(input_tensor) + accessor(self.output)
        residual_offset = len(reader_ct)
        if residual is not None:
            reader_ct += accessor(residual)
        writer_ct = [2048, 8, 4, 32, 4, 4, 0, 16, 1, 0, 1] + accessor(self.output) + accessor(self.output)
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        kernel_base = len(program.kernels)
        additions = [
            ttnn.KernelDescriptor(
                kernel_source=source_reader,
                defines=(
                    [("FUSE_RESIDUAL", "1"), ("RESIDUAL_CT_OFFSET", str(residual_offset))]
                    if residual is not None
                    else []
                ),
                core_ranges=self.grid,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=native + "reduce_scatter_minimal_direct_writer.cpp",
                core_ranges=self.grid,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
                defines=[("LOCAL_STAGING_CB", "1")],
            ),
            ttnn.KernelDescriptor(
                kernel_source=(
                    str(Path(__file__).with_name("kernels") / "reduce_compute.cpp")
                    if residual is not None
                    else native + "reduce_scatter_minimal_direct_compute.cpp"
                ),
                core_ranges=self.grid,
                compile_time_args=[8, 4, 1, 16, 0],
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False),
            ),
        ]
        program.kernels = [*program.kernels, *additions]
        cbs = list(program.cbs)
        for index in ((0, 16, 2, 17) if residual is not None else (0, 16)):
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=16 * 2048,
                    core_ranges=self.grid,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=2048)
                    ],
                )
            )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=64 * 2048,
                core_ranges=self.grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=2048)],
            )
        )
        program.cbs = cbs

        # Native block order: own contribution, then other ranks ascending.
        # Native send order: farthest first, ties choose forward.
        destinations = []
        for dst in range(4):
            if dst == rank:
                continue
            fwd, bwd = (dst - rank) % 4, (rank - dst) % 4
            destinations.append((dst, 0 if fwd <= bwd else 1, min(fwd, bwd)))
        destinations.sort(key=lambda d: -d[2])
        nodes = [self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, d)) for d in range(4)]
        for worker, core in enumerate(self.cores):
            physical = self.mesh.worker_core_from_logical_core(core)
            reader_rt[core.x][core.y] = [
                input_tensor.buffer_address(),
                0,
                rank,
                worker * 2,
                2,
                worker * 16,
                16,
                self.addresses[4],
                *[self.addresses[s] for s in range(4) if s != rank],
                *[d[0] for d in destinations],
            ]
            if residual is not None:
                reader_rt[core.x][core.y] = [*reader_rt[core.x][core.y], residual.buffer_address()]
            compute_rt[core.x][core.y] = [2, 16, self.addresses[6]]
            args = [
                0,
                self.output.buffer_address(),
                rank,
                worker * 2,
                2,
                worker * 16,
                16,
                self.addresses[5],
                self.addresses[rank],
                physical.x,
                physical.y,
                2,
                self.addresses[7],
                2,
                1,
                *[d[1] for d in destinations],
                *[d[2] for d in destinations],
                *[rank + 1 if rank < d[0] else rank for d in destinations],
                *[nodes[d[0]].chip_id for d in destinations],
                *[int(nodes[d[0]].mesh_id) for d in destinations],
            ]
            args.extend(
                ttnn.setup_routing_plane_connection(
                    nodes[rank],
                    [nodes[(rank + 1) % 4], nodes[(rank - 1) % 4]],
                    [worker % 2, worker % 2],
                    program,
                    kernel_base + 1,
                    core,
                )
            )
            writer_rt[core.x][core.y] = args
        # Descriptor bindings copy RuntimeArgs; apply after population. Preserve
        # fabric setup's added defines and semaphore descriptors.
        kernels = list(program.kernels)
        for kernel, rt in zip(kernels[kernel_base:], (reader_rt, writer_rt, compute_rt)):
            kernel.runtime_args = rt
        program.kernels = kernels
        return program

    def __call__(self, input_tensor):
        programs = ttnn.MeshProgramDescriptor()
        for rank in range(4):
            coord = ttnn.MeshCoordinate(0, rank)
            programs[ttnn.MeshCoordinateRange(coord, coord)] = self.append(ttnn.ProgramDescriptor(), input_tensor, rank)
        return ttnn.generic_op([input_tensor, self.output], programs)
