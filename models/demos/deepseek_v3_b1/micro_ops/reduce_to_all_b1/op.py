# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Reduce-to-All B1 Operation using ttnn.generic_op

This module implements a multi-device all-reduce operation for a 4x2 mesh
where all 8 devices end up with the sum of every device's input, using a
3-round hypercube bidirectional exchange.

Mesh layout (coord = [row, col]):
    [0,0] a0  |  b0 [0,1]
    [1,0] a1  |  b1 [1,1]
    [2,0] a2  |  b2 [2,1]
    [3,0] a3  |  b3 [3,1]

Hypercube all-reduce (device_index = row * 2 + col):
    Round 1 (adjacent rows):  (row, col) <-> (row^1, col)
    Round 2 (2-apart rows):   (row, col) <-> (row^2, col)
    Round 3 (cross-column):   (row, col) <-> (row, col^1)

After 3 rounds every device holds sum(all 8 inputs).

R1/R2 are forwarded by each fabric core's BRISC via the same-column EDM
connection. R3 is forwarded by FC1's NCRISC via a separate cross-column
EDM connection (link_idx=1), avoiding circular deadlock.
"""

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


def get_hypercube_partner(row: int, col: int, round_num: int):
    """Return (partner_row, partner_col) for the given hypercube round."""
    if round_num == 1:
        return (row ^ 1, col)
    elif round_num == 2:
        return (row ^ 2, col)
    elif round_num == 3:
        return (row, col ^ 1)
    raise ValueError(f"Invalid round number: {round_num}")


class ReduceToAllB1:
    """
    Multi-device all-reduce (sum) across a 4x2 mesh.

    Uses 3-round hypercube bidirectional exchange so that every device
    accumulates the full reduction result.
    """

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

        print(f"[ReduceToAllB1] mesh_shape={mesh_rows}x{mesh_cols}, num_iterations={num_iterations}")

        if mesh_rows != 4 or mesh_cols != 2:
            raise ValueError(f"Mesh shape must be 4x2, got {mesh_rows}x{mesh_cols}")

        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        output_tensors_per_device = ttnn.get_device_tensors(output_tensor)
        intermediate_per_device = ttnn.get_device_tensors(intermediate_tensor)

        sem_round1_addr = ttnn.get_global_semaphore_address(semaphores[0])
        sem_round2_addr = ttnn.get_global_semaphore_address(semaphores[1])
        sem_round3_addr = ttnn.get_global_semaphore_address(semaphores[2])
        print(f"[ReduceToAllB1] sem addrs: R1={sem_round1_addr:#x} R2={sem_round2_addr:#x} R3={sem_round3_addr:#x}")

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

        print(f"[ReduceToAllB1] shard_shape={shard_shape} num_pages={num_pages} num_compute_tiles={num_compute_tiles}")
        print(
            f"[ReduceToAllB1] page_size_bytes={page_size_bytes} payload_size_bytes={payload_size_bytes} slot_size_bytes={slot_size_bytes}"
        )

        # CB indices
        local_cb = 0
        received_cb = 1
        output_cb = 2
        packet_cb = 3
        scratch_cb = 5

        # Worker-fabric semaphores for BRISC forwarding (R1/R2, reused across rounds)
        shard_grid = input_sample.memory_config().shard_spec.grid
        shard_cores = ttnn.corerange_to_cores(shard_grid, row_wise=True)

        sample_columns = {}
        for c in shard_cores:
            sample_columns.setdefault(c.x, []).append(c)
        num_workers_per_column = len(sample_columns[sorted(sample_columns.keys())[0]])
        device_grid_size = mesh_device.compute_with_storage_grid_size()
        worker_fabric_sem_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )
        worker_fabric_global_sems = [
            ttnn.create_global_semaphore(mesh_device, worker_fabric_sem_cores, 0) for _ in range(num_workers_per_column)
        ]
        worker_fabric_sem_addrs = [ttnn.get_global_semaphore_address(s) for s in worker_fabric_global_sems]

        # R3 gate semaphore (for column-1 defer)
        r3_gate_sem = ttnn.create_global_semaphore(mesh_device, worker_fabric_sem_cores, 0)
        r3_gate_addr = ttnn.get_global_semaphore_address(r3_gate_sem)

        # R3 NCRISC semaphores: 8 per device (all workers signal FC1 NCRISC)
        num_total_workers = len(shard_cores)
        r3_ncrisc_global_sems = [
            ttnn.create_global_semaphore(mesh_device, worker_fabric_sem_cores, 0) for _ in range(num_total_workers)
        ]
        r3_ncrisc_sem_addrs = [ttnn.get_global_semaphore_address(s) for s in r3_ncrisc_global_sems]

        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/reduce_to_all_b1/kernels/reduce_to_all_kernel.cpp"

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

                num_columns = len(sorted_columns)
                num_workers_per_column = len(column_to_cores[sorted_columns[0]])

                all_y_coords = set(core.y for core in input_cores_list)
                is_horizontal_layout = len(all_y_coords) <= 2

                fabric_cores = []
                column_to_fabric_core = {}
                for x in sorted_columns:
                    bottom_core = max(column_to_cores[x], key=lambda c: c.y)
                    if is_horizontal_layout:
                        fabric_core = ttnn.CoreCoord(bottom_core.x, bottom_core.y + 1)
                    else:
                        fabric_core = ttnn.CoreCoord(bottom_core.x + 1, bottom_core.y)
                    fabric_cores.append(fabric_core)
                    column_to_fabric_core[x] = fabric_core

                core_to_slot_idx = {}
                for x in sorted_columns:
                    for slot_idx, core in enumerate(column_to_cores[x]):
                        core_to_slot_idx[(core.x, core.y)] = slot_idx

                # FC1 is the second fabric core (link_idx=1, cross-column)
                fc1 = fabric_cores[1]
                fc1_phys = device.worker_core_from_logical_core(fc1)

                fabric_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in fabric_cores])
                fc1_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(fc1, fc1)])
                all_cores_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in input_cores_list + fabric_cores])

                # Hypercube partners for the 3 rounds
                intermediate_base = intermediate_device.buffer_address()
                partners = {}
                for rnd in (1, 2, 3):
                    pr, pc = get_hypercube_partner(row, col, rnd)
                    dest_coord = ttnn.MeshCoordinate(pr, pc)
                    dest_fabric_node_id = mesh_device.get_fabric_node_id(dest_coord)
                    page_offset = (rnd - 1) * payload_size_bytes
                    dst_l1_addr = intermediate_base + page_offset
                    dst_sem_addr = [sem_round1_addr, sem_round2_addr, sem_round3_addr][rnd - 1]
                    partners[rnd] = {
                        "coord": dest_coord,
                        "fabric_node_id": dest_fabric_node_id,
                        "dst_l1_addr": dst_l1_addr,
                        "dst_sem_addr": dst_sem_addr,
                    }
                    print(
                        f"[ReduceToAllB1] dev({row},{col}) R{rnd}: partner=({pr},{pc}) chip_id={dest_fabric_node_id.chip_id} mesh_id={int(dest_fabric_node_id.mesh_id)} dst_l1={dst_l1_addr:#x} dst_sem={dst_sem_addr:#x}"
                    )

                fabric_node_id = mesh_device.get_fabric_node_id(coord)
                print(
                    f"[ReduceToAllB1] dev({row},{col}) intermediate_base={intermediate_base:#x} fabric_node_id=chip{fabric_node_id.chip_id},mesh{int(fabric_node_id.mesh_id)}"
                )

                # === Compile-time args ===
                reader_ct_args = [
                    ("num_tiles", num_compute_tiles),
                    ("local_cb", local_cb),
                    ("received_cb", received_cb),
                    ("defer_r3_send", col),
                    ("num_workers", num_workers_per_column),
                    ("slot_size_bytes", slot_size_bytes),
                    ("packet_cb", packet_cb),
                    ("payload_size_bytes", payload_size_bytes),
                    ("num_r3_workers", num_total_workers),
                    ("num_loop_iters", num_iterations),
                ]

                writer_ct_args = [
                    ("num_tiles", num_compute_tiles),
                    ("payload_size_bytes", payload_size_bytes),
                    ("local_cb", local_cb),
                    ("scratch_cb", scratch_cb),
                    ("packet_cb", packet_cb),
                    ("num_workers", num_workers_per_column),
                    ("slot_size_bytes", slot_size_bytes),
                    ("r1_dst_chip_id", partners[1]["fabric_node_id"].chip_id),
                    ("r1_dst_mesh_id", int(partners[1]["fabric_node_id"].mesh_id)),
                    ("r2_dst_chip_id", partners[2]["fabric_node_id"].chip_id),
                    ("r2_dst_mesh_id", int(partners[2]["fabric_node_id"].mesh_id)),
                    ("r3_dst_chip_id", partners[3]["fabric_node_id"].chip_id),
                    ("r3_dst_mesh_id", int(partners[3]["fabric_node_id"].mesh_id)),
                    ("defer_r3_send", col),
                    ("num_loop_iters", num_iterations),
                ]

                compute_ct_args = [
                    ("num_tiles", num_compute_tiles),
                    ("local_cb", local_cb),
                    ("received_cb", received_cb),
                    ("scratch_cb", scratch_cb),
                    ("num_loop_iters", num_iterations),
                ]

                # === Common Runtime Args ===
                reader_common_rt_args = [sem_round1_addr, sem_round2_addr, sem_round3_addr, r3_gate_addr]

                print(f"[ReduceToAllB1] dev({row},{col}) worker_cores={[(c.x,c.y) for c in input_cores_list]}")
                print(f"[ReduceToAllB1] dev({row},{col}) fabric_cores={[(c.x,c.y) for c in fabric_cores]}")
                print(f"[ReduceToAllB1] dev({row},{col}) fc1_phys=({fc1_phys.x},{fc1_phys.y})")

                # === Per-Core Runtime Args ===
                # BRISC: worker cores get per-core args; fabric cores get BRISC sem addrs
                brisc_per_core_args = []
                for global_idx, core in enumerate(input_cores_list):
                    fabric_core = column_to_fabric_core[core.x]
                    fabric_core_phys = device.worker_core_from_logical_core(fabric_core)
                    slot_idx = core_to_slot_idx[(core.x, core.y)]

                    worker_args = [
                        fabric_core_phys.x,
                        fabric_core_phys.y,
                        slot_idx,
                        worker_fabric_sem_addrs[slot_idx],
                        partners[1]["dst_l1_addr"],
                        partners[1]["dst_sem_addr"],
                        partners[2]["dst_l1_addr"],
                        partners[2]["dst_sem_addr"],
                        partners[3]["dst_l1_addr"],
                        partners[3]["dst_sem_addr"],
                        output_tensor_device.buffer_address(),
                        r3_gate_addr,
                        fc1_phys.x,
                        fc1_phys.y,
                        global_idx,
                        r3_ncrisc_sem_addrs[global_idx],
                    ]
                    brisc_per_core_args.append((core, worker_args))
                    print(
                        f"[ReduceToAllB1] dev({row},{col}) worker core({core.x},{core.y}) slot={slot_idx} r3_slot={global_idx} fab_phys=({fabric_core_phys.x},{fabric_core_phys.y}) r3_fab=({fc1_phys.x},{fc1_phys.y})"
                    )

                # Fabric core BRISC: worker semaphore addresses for R1/R2
                for fc in fabric_cores:
                    brisc_per_core_args.append((fc, list(worker_fabric_sem_addrs)))

                # NCRISC: FC1 gets R3 NCRISC per-core args (sems); FC0 gets nothing
                ncrisc_per_core_args = [(fc1, list(r3_ncrisc_sem_addrs))]

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

                # packet_cb: 2 rounds × num_workers (BRISC R1/R2) + num_total_workers (NCRISC R3)
                packet_cb_slots = 2 * num_workers_per_column + num_total_workers
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
                print(
                    f"[ReduceToAllB1] dev({row},{col}) packet_cb slots={packet_cb_slots} total={packet_cb_slots * slot_size_bytes}"
                )

                scratch_num_pages = 4
                cb_size_bytes = scratch_num_pages * num_compute_tiles * compute_tile_size_bytes
                cb5_desc = ttnn.CBDescriptor(
                    total_size=cb_size_bytes,
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

                cb_list = [cb0_desc, cb1_desc, cb2_desc, cb3_desc, cb5_desc]

                # === Unified kernel descriptor ===
                unified_ct_core_descriptors = [
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_fabric_core",
                        core_range=fabric_core_set,
                        value=1,
                        other_value=0,
                    ),
                    UnifiedCompileTimeCoreDescriptor(
                        named_compile_time_arg="is_r3_forwarder",
                        core_range=fc1_core_set,
                        value=1,
                        other_value=0,
                    ),
                ]

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
                    unified_compile_time_core_descriptors=unified_ct_core_descriptors,
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        brisc_args=brisc_per_core_args,
                        ncrisc_args=ncrisc_per_core_args,
                    ),
                )

                kernel_result = unified_kernel.get_kernel_descriptors()

                # With two UnifiedCompileTimeCoreDescriptors we get 3 groups:
                #   workers:  is_fabric_core=0, is_r3_forwarder=0
                #   FC0:      is_fabric_core=1, is_r3_forwarder=0
                #   FC1:      is_fabric_core=1, is_r3_forwarder=1
                fc0_group = None
                fc1_group = None
                for g in kernel_result.groups:
                    v = g.compile_time_arg_values
                    if v.get("is_fabric_core") == 1 and v.get("is_r3_forwarder") == 0:
                        fc0_group = g
                    elif v.get("is_fabric_core") == 1 and v.get("is_r3_forwarder") == 1:
                        fc1_group = g

                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    semaphores=[],
                    cbs=cb_list,
                )

                fc0 = fabric_cores[0]
                # FC0 BRISC: connection to R1 partner (same-column, link_idx=0)
                print(f"[ReduceToAllB1] dev({row},{col}) FC0({fc0.x},{fc0.y}) BRISC link_idx=0")
                conn_args_fc0 = ttnn.setup_fabric_connection(
                    fabric_node_id,
                    partners[1]["fabric_node_id"],
                    0,
                    program,
                    fc0,
                )
                program.kernels[fc0_group.brisc_kernel_index].runtime_args[fc0.x][fc0.y].extend(conn_args_fc0)
                print(
                    f"[ReduceToAllB1] dev({row},{col}) FC0({fc0.x},{fc0.y}) BRISC conn_args count={len(conn_args_fc0)}"
                )

                # FC1 BRISC: connection to R1 partner (same-column, link_idx=1)
                print(f"[ReduceToAllB1] dev({row},{col}) FC1({fc1.x},{fc1.y}) BRISC link_idx=1")
                conn_args_fc1 = ttnn.setup_fabric_connection(
                    fabric_node_id,
                    partners[1]["fabric_node_id"],
                    1,
                    program,
                    fc1,
                )
                program.kernels[fc1_group.brisc_kernel_index].runtime_args[fc1.x][fc1.y].extend(conn_args_fc1)
                print(
                    f"[ReduceToAllB1] dev({row},{col}) FC1({fc1.x},{fc1.y}) BRISC conn_args count={len(conn_args_fc1)}"
                )

                # FC1 NCRISC: connection to R3 partner (cross-column, link_idx=1)
                r3_conn_args = ttnn.setup_fabric_connection(
                    fabric_node_id,
                    partners[3]["fabric_node_id"],
                    1,
                    program,
                    fc1,
                )
                program.kernels[fc1_group.ncrisc_kernel_index].runtime_args[fc1.x][fc1.y].extend(r3_conn_args)
                print(
                    f"[ReduceToAllB1] dev({row},{col}) FC1({fc1.x},{fc1.y}) NCRISC R3 conn_args count={len(r3_conn_args)}"
                )

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        input_list = [
            input_tensor_mesh,
            output_tensor,
            intermediate_tensor,
        ]
        print(f"[ReduceToAllB1] calling generic_op...")
        ttnn.generic_op(input_list, mesh_program_descriptor)
        print(f"[ReduceToAllB1] generic_op returned")

        return output_tensor
