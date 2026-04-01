# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Reduce-to-All B1 — Ring + Cross-Column

Requires FABRIC_2D_TORUS_X (or TORUS_XY) for 1-hop ring wrap-around.

Phase 1 — Column all-reduce (2-round ring, A/B split, all 1-hop):
    Type A workers: R1 → FWD (row+1), R2 → BWD (row-1).
    Type B workers: R1 → BWD (row-1), R2 → FWD (row+1).
    FC BRISC (conn → FWD neighbor): forwards R1(A) + R2(B).
    FC NCRISC (conn → BWD neighbor): forwards R1(B) + R2(A).
    Every packet travels exactly 1 hop (ring/torus topology).

Phase 2 — Cross-column exchange (FC-forwarded):
    Workers write R3 to FC's R3 buffer area and signal r3_fwd_sem.
    FC BRISC closes FWD conn, opens cross-column conn, forwards R3.

After both phases every device holds sum(all 8 inputs).

Mesh layout (coord = [row, col]):
    [0,0] a0  |  b0 [0,1]
    [1,0] a1  |  b1 [1,1]
    [2,0] a2  |  b2 [2,1]
    [3,0] a3  |  b3 [3,1]
"""

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class ReduceToAllB1:
    """Multi-device all-reduce (sum) across a 4x2 mesh."""

    @staticmethod
    def golden(input_tensors: list) -> torch.Tensor:
        """PyTorch reference: sum of all input tensors."""
        return torch.sum(torch.stack(input_tensors), dim=0)

    @staticmethod
    def op(
        input_tensor_mesh: ttnn.Tensor,
        intermediate_tensor: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
        semaphores: list,
        num_iterations: int = 1,
    ) -> ttnn.Tensor:
        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        if mesh_rows != 4 or mesh_cols != 2:
            raise ValueError(f"Mesh shape must be 4x2, got {mesh_rows}x{mesh_cols}")

        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        output_tensors_per_device = ttnn.get_device_tensors(output_tensor)
        intermediate_per_device = ttnn.get_device_tensors(intermediate_tensor)

        sem_round1_addr = ttnn.get_global_semaphore_address(semaphores[0])
        sem_round2_addr = ttnn.get_global_semaphore_address(semaphores[1])
        sem_round3_addr = ttnn.get_global_semaphore_address(semaphores[2])

        # Tensor geometry
        input_sample = input_tensors_per_device[0]
        tile = input_sample.tile
        tile_height, tile_width = tile.tile_shape
        dtype = input_sample.dtype
        element_size = 2  # bfloat16

        page_size_bytes = tile_height * tile_width * element_size
        shard_spec = input_sample.memory_config().shard_spec
        shard_shape = shard_spec.shape
        shard_width = shard_shape[1]
        num_pages = shard_width // tile_width
        payload_size_bytes = num_pages * page_size_bytes

        compute_tile_height = 32
        compute_tile_width = 32
        compute_tile_size_bytes = compute_tile_height * compute_tile_width * element_size
        shard_elements = shard_shape[0] * shard_shape[1]
        num_compute_tiles = (shard_elements + compute_tile_height * compute_tile_width - 1) // (
            compute_tile_height * compute_tile_width
        )

        packet_header_size_bytes = 96
        slot_size_bytes = packet_header_size_bytes + payload_size_bytes

        # CB indices
        local_cb = 0
        received_cb = 1
        output_cb = 2
        packet_cb = 3
        reload_cb = 4
        scratch_cb = 5

        shard_grid = input_sample.memory_config().shard_spec.grid
        shard_cores = ttnn.corerange_to_cores(shard_grid, row_wise=True)

        sample_columns = {}
        for c in shard_cores:
            sample_columns.setdefault(c.x, []).append(c)
        sorted_column_keys = sorted(sample_columns.keys())
        num_columns = len(sorted_column_keys)
        num_workers_per_link = len(shard_cores) // num_columns

        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/reduce_to_all_b1/kernels/reduce_to_all_kernel.cpp"

        # Forwarder bitmask semaphore IDs (program-local, matching sdpa_reduce_to_all)
        fwd_r1_sem_id = 0
        fwd_r2_sem_id = 1
        bwd_r1_sem_id = 2
        bwd_r2_sem_id = 3
        r3_fwd_sem_id = 4

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col

                input_tensor_device = input_tensors_per_device[device_idx]
                output_tensor_device = output_tensors_per_device[device_idx]
                intermediate_device = intermediate_per_device[device_idx]
                device = input_tensor_device.device()

                # Worker / fabric core topology
                input_shard_grid = input_tensor_device.memory_config().shard_spec.grid
                input_cores_list = ttnn.corerange_to_cores(input_shard_grid, row_wise=True)

                column_to_cores = {}
                for core in input_cores_list:
                    column_to_cores.setdefault(core.x, []).append(core)
                sorted_columns = sorted(column_to_cores.keys())
                for x in sorted_columns:
                    column_to_cores[x].sort(key=lambda c: c.y)

                all_y_coords = set(core.y for core in input_cores_list)
                is_horizontal_layout = len(all_y_coords) <= 2

                fabric_cores = []
                column_to_fabric_core = {}
                for x in sorted_columns:
                    bottom_core = max(column_to_cores[x], key=lambda c: c.y)
                    if is_horizontal_layout:
                        fc = ttnn.CoreCoord(bottom_core.x, bottom_core.y + 1)
                    else:
                        fc = ttnn.CoreCoord(bottom_core.x + 1, bottom_core.y)
                    fabric_cores.append(fc)
                    column_to_fabric_core[x] = fc

                fabric_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in fabric_cores])
                all_cores_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in input_cores_list + fabric_cores])

                # Ring neighbors (1-hop each, ring/torus topology)
                fwd_row = (row + 1) % mesh_rows
                bwd_row = (row - 1 + mesh_rows) % mesh_rows
                r3_col = col ^ 1

                fwd_coord = ttnn.MeshCoordinate(fwd_row, col)
                bwd_coord = ttnn.MeshCoordinate(bwd_row, col)
                r3_coord = ttnn.MeshCoordinate(row, r3_col)

                fabric_node_id = mesh_device.get_fabric_node_id(coord)
                fwd_fabric_node_id = mesh_device.get_fabric_node_id(fwd_coord)
                bwd_fabric_node_id = mesh_device.get_fabric_node_id(bwd_coord)
                r3_fabric_node_id = mesh_device.get_fabric_node_id(r3_coord)

                intermediate_base = intermediate_device.buffer_address()
                r1_recv_l1 = intermediate_base
                r2_recv_l1 = intermediate_base + payload_size_bytes
                r3_recv_l1 = intermediate_base + 2 * payload_size_bytes

                # A/B split: half workers per link in each type
                slots_per_direction = num_workers_per_link // 2

                # FC buffer layout:
                #   BRISC area [FWD]: [R1: slots_per_dir slots][R2: slots_per_dir slots]
                #   NCRISC area [BWD]: [R1: slots_per_dir slots][R2: slots_per_dir slots]
                #   R3 area: [num_workers_per_link slots] (forwarded by FC BRISC after Phase 1)
                r2_buf_offset = slots_per_direction * slot_size_bytes
                brisc_buf_size = 2 * slots_per_direction * slot_size_bytes
                ncrisc_buf_offset = brisc_buf_size
                r3_buf_offset = 4 * slots_per_direction * slot_size_bytes
                packet_cb_slots = 4 * slots_per_direction + num_workers_per_link

                # Gather per-link info
                links = []
                for link_idx, x in enumerate(sorted_columns):
                    cores_for_link = column_to_cores[x]
                    fc = column_to_fabric_core[x]
                    fc_phys = device.worker_core_from_logical_core(fc)
                    links.append(
                        {
                            "link_idx": link_idx,
                            "cores": cores_for_link,
                            "fc": fc,
                            "fc_phys": fc_phys,
                        }
                    )

                # === Compile-time args ===
                reader_ct_args = [
                    ("num_tiles", num_compute_tiles),
                    ("local_cb", local_cb),
                    ("received_cb", received_cb),
                    ("slots_per_direction", slots_per_direction),
                    ("slot_size_bytes", slot_size_bytes),
                    ("packet_cb", packet_cb),
                    ("payload_size_bytes", payload_size_bytes),
                    ("r2_buffer_offset", r2_buf_offset),
                    ("ncrisc_buffer_offset", ncrisc_buf_offset),
                    ("num_loop_iters", num_iterations),
                ]

                writer_ct_args = [
                    ("num_tiles", num_compute_tiles),
                    ("payload_size_bytes", payload_size_bytes),
                    ("local_cb", local_cb),
                    ("scratch_cb", scratch_cb),
                    ("packet_cb", packet_cb),
                    ("slot_size_bytes", slot_size_bytes),
                    ("fwd_dst_chip_id", fwd_fabric_node_id.chip_id),
                    ("fwd_dst_mesh_id", int(fwd_fabric_node_id.mesh_id)),
                    ("bwd_dst_chip_id", bwd_fabric_node_id.chip_id),
                    ("bwd_dst_mesh_id", int(bwd_fabric_node_id.mesh_id)),
                    ("r3_dst_chip_id", r3_fabric_node_id.chip_id),
                    ("r3_dst_mesh_id", int(r3_fabric_node_id.mesh_id)),
                    ("reload_cb", reload_cb),
                    ("compute_tile_size", compute_tile_size_bytes),
                    ("slots_per_direction", slots_per_direction),
                    ("r2_buffer_offset", r2_buf_offset),
                    ("ncrisc_buffer_offset", ncrisc_buf_offset),
                    ("r3_buffer_offset", r3_buf_offset),
                    ("num_loop_iters", num_iterations),
                ]

                compute_ct_args = [
                    ("num_tiles", num_compute_tiles),
                    ("local_cb", local_cb),
                    ("received_cb", received_cb),
                    ("scratch_cb", scratch_cb),
                    ("reload_cb", reload_cb),
                    ("num_loop_iters", num_iterations),
                ]

                # === Common Runtime Args ===
                reader_common_rt_args = [sem_round1_addr, sem_round2_addr, sem_round3_addr]

                # === Per-Core Runtime Args ===
                brisc_per_core_args = []
                ncrisc_per_core_args = []

                for link_info in links:
                    fc_phys = link_info["fc_phys"]
                    fc = link_info["fc"]
                    link_idx = link_info["link_idx"]

                    # Direction configs — mirrors sdpa_reduce_to_all DirectionConfig
                    # FWD: buffer starts at packet_cb base (offset 0)
                    # BWD: buffer starts at ncrisc_buf_offset
                    class DirCfg:
                        def __init__(self, r1_sem, r2_sem, buf_base_offset):
                            self.r1_sem = r1_sem
                            self.r2_sem = r2_sem
                            self.buf_base_offset = buf_base_offset
                            self.r1_count = 0
                            self.r2_count = 0

                    fwd_cfg = DirCfg(fwd_r1_sem_id, fwd_r2_sem_id, 0)
                    bwd_cfg = DirCfg(bwd_r1_sem_id, bwd_r2_sem_id, ncrisc_buf_offset)

                    for worker_idx, core in enumerate(link_info["cores"]):
                        is_type_a = ((row + worker_idx) % 2) == 0

                        # Type A: R1→FWD, R2→BWD;  Type B: R1→BWD, R2→FWD
                        r1_cfg = fwd_cfg if is_type_a else bwd_cfg
                        r2_cfg = bwd_cfg if is_type_a else fwd_cfg

                        r1_slot_idx = r1_cfg.r1_count
                        r1_cfg.r1_count += 1
                        r2_slot_idx = r2_cfg.r2_count
                        r2_cfg.r2_count += 1

                        r1_slot_offset = r1_cfg.buf_base_offset + r1_slot_idx * slot_size_bytes
                        r1_slot_bit = 1 << r1_slot_idx
                        r1_sem = r1_cfg.r1_sem

                        r2_slot_offset = r2_cfg.buf_base_offset + r2_buf_offset + r2_slot_idx * slot_size_bytes
                        r2_slot_bit = 1 << r2_slot_idx
                        r2_sem = r2_cfg.r2_sem

                        # R3: worker writes to FC's R3 area (after BRISC+NCRISC areas)
                        r3_slot_off = r3_buf_offset + worker_idx * slot_size_bytes
                        r3_slot_b = 1 << worker_idx

                        worker_args = [
                            fc_phys.x,  # 0
                            fc_phys.y,  # 1
                            1 if is_type_a else 0,  # 2
                            r1_slot_offset,  # 3
                            r1_slot_bit,  # 4
                            r1_sem,  # 5
                            r2_slot_offset,  # 6
                            r2_slot_bit,  # 7
                            r2_sem,  # 8
                            r1_recv_l1,  # 9
                            sem_round1_addr,  # 10
                            r2_recv_l1,  # 11
                            sem_round2_addr,  # 12
                            r3_recv_l1,  # 13
                            sem_round3_addr,  # 14
                            output_tensor_device.buffer_address(),  # 15
                            r3_slot_off,  # 16
                            r3_slot_b,  # 17
                            r3_fwd_sem_id,  # 18 (sem ID, resolved via get_semaphore)
                        ]
                        brisc_per_core_args.append((core, worker_args))

                    # FC BRISC: FWD forwarding + R3 cross-column forwarding
                    # [0] fwd_r1_sem_id, [1] fwd_r2_sem_id, [2] r3_fwd_sem_id,
                    # then FWD conn args, then cross-column conn args (appended later)
                    brisc_per_core_args.append((fc, [fwd_r1_sem_id, fwd_r2_sem_id, r3_fwd_sem_id]))

                    # FC NCRISC: BWD forwarding (2 sem IDs + conn appended later)
                    ncrisc_per_core_args.append((fc, [bwd_r1_sem_id, bwd_r2_sem_id]))

                # === CB Descriptors ===
                compute_tile_desc = ttnn.TileDescriptor(compute_tile_height, compute_tile_width)

                cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(local_cb, input_tensor_device)
                cb0_desc.core_ranges = all_cores_set
                cb0_desc.total_size = payload_size_bytes
                cb0_desc.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=local_cb,
                        data_format=dtype,
                        page_size=payload_size_bytes,
                        tile=compute_tile_desc,
                    )
                ]

                cb1_desc = ttnn.cb_descriptor_from_sharded_tensor(received_cb, intermediate_device)
                cb1_desc.core_ranges = all_cores_set
                cb1_desc.total_size = 3 * payload_size_bytes
                cb1_desc.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=received_cb,
                        data_format=dtype,
                        page_size=payload_size_bytes,
                        tile=compute_tile_desc,
                    )
                ]

                cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor_device)
                cb2_desc.core_ranges = all_cores_set
                cb2_desc.total_size = payload_size_bytes
                cb2_desc.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=output_cb,
                        data_format=dtype,
                        page_size=payload_size_bytes,
                        tile=compute_tile_desc,
                    )
                ]

                cb3_desc = ttnn.CBDescriptor(
                    total_size=packet_cb_slots * slot_size_bytes,
                    core_ranges=all_cores_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=packet_cb,
                            data_format=dtype,
                            page_size=slot_size_bytes,
                        )
                    ],
                )

                cb4_desc = ttnn.CBDescriptor(
                    total_size=num_compute_tiles * compute_tile_size_bytes,
                    core_ranges=all_cores_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=reload_cb,
                            data_format=dtype,
                            page_size=compute_tile_size_bytes,
                            tile=compute_tile_desc,
                        )
                    ],
                )

                scratch_num_pages = 4
                cb5_desc = ttnn.CBDescriptor(
                    total_size=scratch_num_pages * num_compute_tiles * compute_tile_size_bytes,
                    core_ranges=all_cores_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=scratch_cb,
                            data_format=dtype,
                            page_size=compute_tile_size_bytes,
                            tile=compute_tile_desc,
                        )
                    ],
                )

                cb_list = [cb0_desc, cb1_desc, cb2_desc, cb3_desc, cb4_desc, cb5_desc]

                # Forwarder bitmask semaphores (program-local, on FC cores only)
                forwarder_semaphores = [
                    ttnn.SemaphoreDescriptor(id=fwd_r1_sem_id, core_ranges=fabric_core_set, initial_value=0),
                    ttnn.SemaphoreDescriptor(id=fwd_r2_sem_id, core_ranges=fabric_core_set, initial_value=0),
                    ttnn.SemaphoreDescriptor(id=bwd_r1_sem_id, core_ranges=fabric_core_set, initial_value=0),
                    ttnn.SemaphoreDescriptor(id=bwd_r2_sem_id, core_ranges=fabric_core_set, initial_value=0),
                    ttnn.SemaphoreDescriptor(id=r3_fwd_sem_id, core_ranges=fabric_core_set, initial_value=0),
                ]

                # === Unified kernel descriptor ===
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source=kernel_path,
                    core_ranges=all_cores_set,
                    ncrisc_named_compile_time_args=reader_ct_args,
                    brisc_named_compile_time_args=writer_ct_args,
                    trisc_named_compile_time_args=compute_ct_args,
                    ncrisc_common_runtime_args=reader_common_rt_args,
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=False,
                        math_approx_mode=False,
                    ),
                    unified_compile_time_core_descriptors=[
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_fabric_core",
                            core_range=fabric_core_set,
                            value=1,
                            other_value=0,
                        ),
                    ],
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        brisc_args=brisc_per_core_args,
                        ncrisc_args=ncrisc_per_core_args,
                    ),
                    noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                )

                kernel_result = unified_kernel.get_kernel_descriptors()

                fabric_group = kernel_result.get_group_by_arg("is_fabric_core", 1)
                worker_group = kernel_result.get_group_by_arg("is_fabric_core", 0)

                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    semaphores=forwarder_semaphores,
                    cbs=cb_list,
                )

                # Append FC fabric connections:
                #   BRISC: FWD conn (Phase 1) + cross-column conn (Phase 2)
                #   NCRISC: BWD conn (Phase 1)
                brisc_idx = fabric_group.brisc_kernel_index
                ncrisc_idx = fabric_group.ncrisc_kernel_index
                for link_info in links:
                    fc = link_info["fc"]
                    link_idx = link_info["link_idx"]

                    fwd_conn_args = ttnn.setup_fabric_connection(
                        fabric_node_id,
                        fwd_fabric_node_id,
                        link_idx,
                        program,
                        fc,
                    )
                    program.kernels[brisc_idx].runtime_args[fc.x][fc.y].extend(fwd_conn_args)

                    r3_conn_args = ttnn.setup_fabric_connection(
                        fabric_node_id,
                        r3_fabric_node_id,
                        link_idx,
                        program,
                        fc,
                    )
                    program.kernels[brisc_idx].runtime_args[fc.x][fc.y].extend(r3_conn_args)

                    bwd_conn_args = ttnn.setup_fabric_connection(
                        fabric_node_id,
                        bwd_fabric_node_id,
                        link_idx,
                        program,
                        fc,
                    )
                    program.kernels[ncrisc_idx].runtime_args[fc.x][fc.y].extend(bwd_conn_args)

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        input_list = [input_tensor_mesh, output_tensor, intermediate_tensor]
        ttnn.generic_op(input_list, mesh_program_descriptor)

        return output_tensor
