# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""select_final_carry as a ProgramDescriptor that another program can carry on its spare cores.

Mirrors ttnn/cpp/ttnn/operations/experimental/kda/select_final_carry/device/select_final_carry_program_factory.cpp
with the same kernels, so the final carry's copy or sequence-parallel multicast can run inside the fused output
projection and reduce-scatter instead of as its own program: nothing in the layer reads the carry, so its
transfer overlaps the projection.
"""

from __future__ import annotations

import ttnn

_KERNEL_DIR = "ttnn/cpp/ttnn/operations/experimental/kda/select_final_carry/device/kernels/dataflow/"
_TILES_CB = 0
_MODE_CB = 1
# The reader issues up to eight packets under one barrier and the tiles buffer holds two such batches.
_BATCH_PACKETS = 8


# Fabric configs that wrap a mesh axis (axis 0 = rows), as ttnn::ccl::get_usable_topology sees them.
_WRAPPED_AXES = {
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY: (0, 1),
    ttnn.FabricConfig.FABRIC_1D_RING: (0, 1),
    ttnn.FabricConfig.FABRIC_2D_TORUS_X: (1,),
    ttnn.FabricConfig.FABRIC_2D_TORUS_Y: (0,),
}


def fabric_wraps_axis(axis: int) -> bool:
    """Whether the current fabric config wraps mesh axis `axis` into a ring."""
    return axis in _WRAPPED_AXES.get(ttnn.get_fabric_config(), ())


def _ring_mcast_distance(ring_size: int, ring_index: int) -> tuple[int, int]:
    """ttnn::ccl::get_forward_backward_line_mcast_distance for a ring with static alternation."""
    forward = (ring_size - 1 + 1) // 2
    backward = ring_size - 1 - forward
    if ring_index % 2 == 0:
        forward, backward = backward, forward
    return forward, backward


class FinalCarryProgram:
    """Persistent state (global semaphores) of the carried final-carry program for one mesh."""

    def __init__(self, mesh_device: ttnn.MeshDevice, sequence_parallel_axis: int, num_links: int):
        self.mesh_device = mesh_device
        self.axis = sequence_parallel_axis
        self.num_links = num_links
        grid = mesh_device.compute_with_storage_grid_size()
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        # Both counters persist across replays; L1_SMALL, when the device has one, keeps a fabric mux from clobbering
        # them (ttnn::ccl::prefer_l1_small_buffer_type).
        try:
            self.barrier = ttnn.create_global_semaphore(mesh_device, all_cores, 0, ttnn.BufferType.L1_SMALL)
            self.arrival = ttnn.create_global_semaphore(mesh_device, all_cores, 0, ttnn.BufferType.L1_SMALL)
        except RuntimeError:
            self.barrier = ttnn.create_global_semaphore(mesh_device, all_cores, 0, ttnn.BufferType.L1)
            self.arrival = ttnn.create_global_semaphore(mesh_device, all_cores, 0, ttnn.BufferType.L1)
        ttnn.synchronize_device(mesh_device)

    def allocate_output(self, prefix_final: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.allocate_tensor_on_device(
            prefix_final.shape, ttnn.float32, ttnn.TILE_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )

    def program(
        self,
        coord: tuple[int, int],
        cores: list[ttnn.CoreCoord],
        *,
        rank_final: ttnn.Tensor,
        prefix_final: ttnn.Tensor,
        output: ttnn.Tensor,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None,
        local_rows: int,
        prefix_on_last_rank: bool = False,
    ) -> ttnn.ProgramDescriptor:
        mesh = self.mesh_device
        sp_size = tuple(mesh.shape)[self.axis]
        sp_rank = coord[self.axis]
        shape = tuple(prefix_final.padded_shape)
        state_tiles = 1
        for dim in shape:
            state_tiles *= dim
        state_tiles //= 32 * 32
        head_tiles = shape[-2] * shape[-1] // (32 * 32)
        tail_groups = rank_final.shape[1] if len(rank_final.shape) == 4 else 1
        page_size = 32 * 32 * 4
        packet_pages = min(2, ttnn.get_tt_fabric_max_payload_size_bytes() // page_size)
        assert packet_pages >= 1 and sp_size > 1, "the carried final carry needs a sequence-parallel axis"

        fabric_cores = cores[: self.num_links]
        copy_cores = cores
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

        def split(tiles, parts, index):
            per = (tiles + parts - 1) // parts
            return min(tiles, index * per), min(tiles, (index + 1) * per)

        # A ring when the fabric wraps the SP axis (and it has more than two ranks), else a line, as the C++ op.
        ring = sp_size > 2 and fabric_wraps_axis(self.axis)
        neighbors = []
        for step in (1, -1):
            if ring or 0 <= sp_rank + step < sp_size:
                peer = list(coord)
                peer[self.axis] = (sp_rank + step) % sp_size
                neighbors.append(tuple(peer))
            else:
                neighbors.append(None)
        forward, backward = neighbors
        if ring:
            targets_forward, targets_backward = _ring_mcast_distance(sp_size, sp_rank)
        else:
            targets_forward, targets_backward = sp_size - 1 - sp_rank, sp_rank
        node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))

        end_tensor = actual_end if actual_end is not None else actual_start
        reader_ct = [
            _TILES_CB,
            _MODE_CB,
            packet_pages,
            sp_rank,
            sp_size,
            local_rows,
            int(actual_end is not None),
            tail_groups,
            head_tiles,
        ]
        for tensor in (rank_final, prefix_final, actual_start, end_tensor):
            reader_ct += list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        writer_ct = [
            _TILES_CB,
            _MODE_CB,
            packet_pages,
            page_size,
            targets_forward + targets_backward,
            int(forward is not None),
            targets_forward if forward is not None else 0,
            int(backward is not None),
            targets_backward if backward is not None else 0,
        ] + list(ttnn.TensorAccessorArgs(output).get_compile_time_args())

        program = ttnn.ProgramDescriptor()
        program.cbs = [
            ttnn.CBDescriptor(
                total_size=2 * _BATCH_PACKETS * packet_pages * page_size,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=_TILES_CB, data_format=ttnn.float32, page_size=page_size)
                ],
            ),
            ttnn.CBDescriptor(
                total_size=64,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=_MODE_CB, data_format=ttnn.uint32, page_size=64)
                ],
            ),
        ]
        program.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_KERNEL_DIR + "reader_select_final_carry.cpp",
                core_ranges=core_set,
                compile_time_args=reader_ct,
                # Without a split tail, the last chronological rank multicasts the prefix's final state.
                defines=[("PREFIX_ON_LAST_RANK", "1")] if prefix_on_last_rank else [],
                runtime_args=ttnn.RuntimeArgs(),
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_KERNEL_DIR + "writer_select_final_carry.cpp",
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=ttnn.RuntimeArgs(),
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]

        drain = mesh.worker_core_from_logical_core(fabric_cores[0])
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        addresses = [t.buffer_address() for t in (rank_final, prefix_final, actual_start, end_tensor)]
        barrier = int(ttnn.get_global_semaphore_address(self.barrier))
        arrival = int(ttnn.get_global_semaphore_address(self.arrival))
        for index, core in enumerate(cores):
            copy = split(state_tiles, len(copy_cores), index)
            link = index if index < len(fabric_cores) else None
            send = split(state_tiles, self.num_links, link) if link is not None else (0, 0)
            reader_rt[core.x][core.y] = addresses + [copy[0], copy[1], send[0], send[1]]
            args = [output.buffer_address(), copy[0], copy[1], send[0], send[1], int(link is not None)]
            if link is not None:
                worker = mesh.worker_core_from_logical_core(core)
                # Connections are packed in destination order: forward, then backward.
                peers = [node(peer) for peer in (forward, backward) if peer is not None]
                args += [arrival, drain.x, drain.y, int(link == 0), self.num_links, barrier, worker.x, worker.y]
                args += [len(peers)]
                args += list(ttnn.setup_routing_plane_connection(node(coord), peers, [link], program, 1, core))
            writer_rt[core.x][core.y] = args
        kernels = program.kernels
        kernels[0].runtime_args = reader_rt
        kernels[1].runtime_args = writer_rt
        program.kernels = kernels
        return program
