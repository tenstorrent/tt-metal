# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SDPA reduce-to-all operation using ttnn.generic_op.

Torus/Ring topology only. Single-run correctness (no trace replay).
"""

import torch

import ttnn
from models.demos.deepseek_v3_b1.utils import float_to_uint32


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _get_neighbor_coord(mesh_shape, row, col, offset, cluster_axis=0):
    if cluster_axis == 0:
        neighbor_row = (row + offset) % mesh_shape[0]
        return neighbor_row, col
    neighbor_col = (col + offset) % mesh_shape[1]
    return row, neighbor_col


def _get_element_size_bytes(dtype):
    if dtype == ttnn.bfloat16:
        return 2
    if dtype == ttnn.float32:
        return 4
    raise ValueError(f"Unsupported dtype for sdpa_reduce_to_all: {dtype}")


class SdpaReduceToAll:
    @staticmethod
    def golden(
        l_data_per_device,
        s_data_per_device,
        m_data_per_device,
        num_cores=8,
        scale_value=1.0,
        position_mask=None,
        final_reduction=True,
    ):
        """
        PyTorch reference implementation for SDPA reduce-to-all.

        Args:
            l_data_per_device: list of L tensors [batch, l_width * num_cores]
            s_data_per_device: list of S tensors [batch, num_cores]
            m_data_per_device: list of M tensors [batch, num_cores]
            position_mask: optional tensor [num_devices] with 1.0 for valid devices, 0.0 for masked devices
        """

        def compute_reduction(l1, s1, m1, l2, s2, m2, scale, valid1, valid2):
            """Conditional reduction based on validity flags.

            Args:
                normalize: If True, normalize the output L by dividing by S
            """
            if not valid1 and not valid2:
                l_out = l1
                s_out = s1
                m_out = m1
            elif valid1 and not valid2:
                l_out = l1
                s_out = s1
                m_out = m1
            elif not valid1 and valid2:
                l_out = l2
                s_out = s2
                m_out = m2
            # Both valid: perform normal reduction
            else:
                m_new = torch.maximum(m1, m2)
                exp_m1 = torch.exp((m1 - m_new) * scale)
                exp_m2 = torch.exp((m2 - m_new) * scale)
                s_out = s1 * exp_m1 + s2 * exp_m2
                l_out = l1 * exp_m1 + l2 * exp_m2
                m_out = m_new
            return l_out, s_out, m_out

        num_devices = len(l_data_per_device)

        # Default mask: all devices valid
        if position_mask is None:
            position_mask = torch.ones(num_devices, dtype=torch.float32)

        def split_by_cores(tensor_list, num_cores):
            result = []
            for device_tensor in tensor_list:
                cores = torch.chunk(device_tensor, num_cores, dim=1)
                result.append(cores)
            return result

        l_per_device_per_core = split_by_cores(l_data_per_device, num_cores)

        l_final_cores = []
        s_final_cores = []
        m_final_cores = []

        for core_idx in range(num_cores):
            l_dev = [l_per_device_per_core[d][core_idx] for d in range(num_devices)]
            s_dev = [s_data_per_device[d][:, core_idx : core_idx + 1] for d in range(num_devices)]
            m_dev = [m_data_per_device[d][:, core_idx : core_idx + 1] for d in range(num_devices)]

            l_r1_01, s_r1_01, m_r1_01 = compute_reduction(
                l_dev[0],
                s_dev[0],
                m_dev[0],
                l_dev[1],
                s_dev[1],
                m_dev[1],
                scale_value,
                position_mask[0].item() > 0.5,
                position_mask[1].item() > 0.5,
            )
            l_r1_23, s_r1_23, m_r1_23 = compute_reduction(
                l_dev[2],
                s_dev[2],
                m_dev[2],
                l_dev[3],
                s_dev[3],
                m_dev[3],
                scale_value,
                position_mask[2].item() > 0.5,
                position_mask[3].item() > 0.5,
            )

            r1_01_valid = (position_mask[0].item() > 0.5) or (position_mask[1].item() > 0.5)
            r1_23_valid = (position_mask[2].item() > 0.5) or (position_mask[3].item() > 0.5)

            # Always normalize in R2 (simplest approach)
            l_final, s_final, m_final = compute_reduction(
                l_r1_01,
                s_r1_01,
                m_r1_01,
                l_r1_23,
                s_r1_23,
                m_r1_23,
                scale_value,
                r1_01_valid,
                r1_23_valid,
            )
            if final_reduction:
                l_final = l_final / s_final.expand(-1, l_final.shape[1])
            l_final_cores.append(l_final)
            s_final_cores.append(s_final)
            m_final_cores.append(m_final)

        return torch.cat(l_final_cores, dim=1), torch.cat(s_final_cores, dim=1), torch.cat(m_final_cores, dim=1)

    @staticmethod
    def op(
        input_tensor_l_mesh,
        input_tensor_ms_mesh,
        output_tensor_l_mesh,
        r1_recv_tensor_mesh,
        r2_recv_tensor_mesh,
        forwarder_scratch_mesh,
        semaphores,
        scale_fp32=1.0,
        cluster_axis=0,
        input_forwarder_cores=None,
        scatter_dest_tensor_mesh=None,
        scatter_dest_grid=None,
        position_tensor_mesh=None,
        final_reduction=True,
    ):
        mesh_device = input_tensor_l_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]
        num_devices = mesh_rows * mesh_cols

        if input_forwarder_cores is None or len(input_forwarder_cores) != 2:
            raise ValueError("input_forwarder_cores must be provided with exactly 2 cores")

        forwarder_cores = input_forwarder_cores
        forwarder_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in forwarder_cores])

        # Position tensor setup (optional conditional reduction)
        # Position tensor will be passed to kernel via CB (no host-side extraction)
        position_enabled = position_tensor_mesh is not None

        input_l_per_device = ttnn.get_device_tensors(input_tensor_l_mesh)
        input_ms_per_device = ttnn.get_device_tensors(input_tensor_ms_mesh)
        output_l_per_device = ttnn.get_device_tensors(output_tensor_l_mesh)
        r1_recv_per_device = ttnn.get_device_tensors(r1_recv_tensor_mesh)
        r2_recv_per_device = ttnn.get_device_tensors(r2_recv_tensor_mesh)
        fwd_scratch_per_device = ttnn.get_device_tensors(forwarder_scratch_mesh)
        position_per_device = None

        if position_enabled:
            position_per_device = ttnn.get_device_tensors(position_tensor_mesh)

        # Scatter destination setup (optional)
        scatter_enabled = scatter_dest_tensor_mesh is not None and scatter_dest_grid is not None
        scatter_dest_per_device = None
        scatter_dest_cores_list = None
        if scatter_enabled:
            scatter_dest_per_device = ttnn.get_device_tensors(scatter_dest_tensor_mesh)
            scatter_dest_cores_list = ttnn.corerange_to_cores(scatter_dest_grid, row_wise=True)

        r1_recv_sem_addr = ttnn.get_global_semaphore_address(semaphores[0])
        r2_recv_sem_addr = ttnn.get_global_semaphore_address(semaphores[1])

        packet_header_size_bytes = ttnn.get_tt_fabric_packet_header_size_bytes()
        max_fabric_payload_size = ttnn.get_tt_fabric_max_payload_size_bytes()

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # Forwarder semaphore IDs (same across cores)
        fwd_r1_sem_id = 0
        fwd_r2_sem_id = 1
        bwd_r1_sem_id = 2
        bwd_r2_sem_id = 3

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col

                input_l_device = input_l_per_device[device_idx]
                input_ms_device = input_ms_per_device[device_idx]
                output_l_device = output_l_per_device[device_idx]
                r1_recv_device = r1_recv_per_device[device_idx]
                r2_recv_device = r2_recv_per_device[device_idx]
                fwd_scratch_device = fwd_scratch_per_device[device_idx]

                position_device = None
                if position_enabled:
                    position_device = position_per_device[device_idx]

                device = input_l_device.device()

                shard_spec = input_l_device.memory_config().shard_spec
                shard_grid = shard_spec.grid
                shard_cores = ttnn.corerange_to_cores(shard_grid, row_wise=True)
                num_shard_cores = len(shard_cores)

                tile = input_l_device.tile
                tile_height, tile_width = tile.tile_shape
                element_size_bytes = _get_element_size_bytes(input_l_device.dtype)
                input_page_size_bytes = element_size_bytes * tile_height * tile_width
                l1_alignment = 16
                aligned_page_size = _round_up(input_page_size_bytes, l1_alignment)

                input_l_num_pages = (shard_spec.shape[0] // tile_height) * (shard_spec.shape[1] // tile_width)

                PNH = 8
                DH = input_l_num_pages * tile_width
                DHt = DH // tile_width
                PNHt = PNH // tile_height
                Sq_chunk_t = PNHt
                out_tiles = Sq_chunk_t * DHt

                max_tiles_per_chunk = 8
                min_num_l_chunks = (out_tiles + max_tiles_per_chunk - 1) // max_tiles_per_chunk
                num_l_chunks = max(min_num_l_chunks, 4)
                if out_tiles % num_l_chunks != 0:
                    raise ValueError("out_tiles must be divisible by num_l_chunks")

                tiles_per_l_chunk = out_tiles // num_l_chunks
                l_chunk_size_bytes = tiles_per_l_chunk * input_page_size_bytes
                ms_tile_size_bytes = aligned_page_size

                if l_chunk_size_bytes > max_fabric_payload_size:
                    raise ValueError("L chunk payload exceeds fabric max payload size")

                # Slots are sized for the largest payload (L chunk); MS uses slot 0.
                header_cb_size = _round_up(packet_header_size_bytes, l1_alignment)
                slot_size = _round_up(packet_header_size_bytes + l_chunk_size_bytes, l1_alignment)

                num_links = 2
                num_workers_per_link = num_shard_cores // num_links
                workers_per_type = num_workers_per_link // 2
                slots_per_worker = 1 + num_l_chunks
                # Bit-packed forwarder semaphores support up to 32 slots per round.
                slots_per_round = workers_per_type * slots_per_worker

                if slots_per_round > 32:
                    raise ValueError("slots_per_round exceeds 32-bit semaphore capacity")

                # Per-core forwarder buffer layout: BRISC [R1][R2], NCRISC after BRISC.
                # forwarder_buffer_base is per-core L1; scratch size must be per-core.
                r2_buffer_offset = slots_per_round * slot_size
                brisc_buffer_size = 2 * slots_per_round * slot_size
                ncrisc_buffer_offset = brisc_buffer_size
                forwarder_buffer_base = fwd_scratch_device.buffer_address()

                scale_val = float_to_uint32(scale_fp32)

                # CB indices
                cb_local_l = 0
                cb_local_ms = 1
                cb_r1_neighbor_l = 2
                cb_r1_neighbor_ms = 3
                cb_r1_result_l = 4
                cb_r1_result_ms = 5
                cb_r2_neighbor_l = 6
                cb_r2_neighbor_ms = 7
                cb_l_out = 8
                cb_ms_out = 9
                cb_packet_slot = 10
                cb_position = 11

                # Kernel compile-time args
                reader_ct_args = [
                    cb_local_l,
                    cb_local_ms,
                    cb_r1_neighbor_l,
                    cb_r1_neighbor_ms,
                    cb_r2_neighbor_l,
                    cb_r2_neighbor_ms,
                    ms_tile_size_bytes,
                    l_chunk_size_bytes,
                    num_l_chunks,
                    tiles_per_l_chunk,
                    cb_position,  # Position CB index
                    1 if position_enabled else 0,  # Enable/disable position
                ]

                # Scatter compile-time parameters
                if scatter_enabled:
                    dest_tile = scatter_dest_per_device[device_idx].tile
                    dest_h, dest_w = dest_tile.tile_shape
                    assert dest_w == tile_width, "Source and dest tile widths must match"
                    assert tile_height % dest_h == 0, "Source tile height must be divisible by dest tile height"
                    scatter_ct_num_rows = tile_height // dest_h
                    scatter_ct_num_tiles = shard_spec.shape[1] // tile_width
                    scatter_ct_src_tile_size = input_page_size_bytes
                    scatter_ct_dst_tile_size = dest_h * dest_w * element_size_bytes
                    scatter_ct_face_size = tile_height * (tile_width // 2) * element_size_bytes
                    scatter_ct_row_face_size = dest_h * (tile_width // 2) * element_size_bytes
                else:
                    scatter_ct_num_rows = 0
                    scatter_ct_num_tiles = 0
                    scatter_ct_src_tile_size = 0
                    scatter_ct_dst_tile_size = 0
                    scatter_ct_face_size = 0
                    scatter_ct_row_face_size = 0

                writer_ct_args = [
                    cb_local_l,
                    cb_local_ms,
                    cb_r1_result_l,
                    cb_r1_result_ms,
                    cb_packet_slot,
                    l1_alignment,
                    input_page_size_bytes,
                    slot_size,
                    ms_tile_size_bytes,
                    l_chunk_size_bytes,
                    num_l_chunks,
                    tiles_per_l_chunk,
                    # Scatter phase args (indices 12-18)
                    cb_l_out,
                    scatter_ct_num_tiles,
                    scatter_ct_src_tile_size,
                    scatter_ct_dst_tile_size,
                    scatter_ct_face_size,
                    scatter_ct_row_face_size,
                    scatter_ct_num_rows,
                ]

                compute_ct_args = [
                    cb_local_l,
                    cb_local_ms,
                    cb_r1_neighbor_l,
                    cb_r1_neighbor_ms,
                    cb_r1_result_l,
                    cb_r1_result_ms,
                    cb_r2_neighbor_l,
                    cb_r2_neighbor_ms,
                    cb_l_out,
                    cb_ms_out,
                    scale_val,
                    tiles_per_l_chunk,
                    num_l_chunks,
                    cb_position,  # Position CB index
                    1 if position_enabled else 0,  # Enable/disable conditional reduction
                    final_reduction,
                ]

                forwarder_ct_args = [slots_per_round, slot_size, r2_buffer_offset]

                reader_kernel = ttnn.KernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/micro_ops/sdpa_reduce_to_all/kernels/reader.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=shard_grid,
                    compile_time_args=reader_ct_args,
                    config=ttnn.ReaderConfigDescriptor(),
                )

                writer_kernel = ttnn.KernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/micro_ops/sdpa_reduce_to_all/kernels/writer.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=shard_grid,
                    compile_time_args=writer_ct_args,
                    config=ttnn.WriterConfigDescriptor(),
                )

                compute_kernel = ttnn.KernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/micro_ops/sdpa_reduce_to_all/kernels/compute.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=shard_grid,
                    compile_time_args=compute_ct_args,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=False,
                        dst_full_sync_en=False,
                        math_approx_mode=False,
                    ),
                )

                forwarder_brisc_kernel = ttnn.KernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/micro_ops/sdpa_reduce_to_all/kernels/forwarder.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=forwarder_core_range_set,
                    compile_time_args=forwarder_ct_args,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0,
                        noc=ttnn.NOC.RISCV_0_default,
                        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                    ),
                )

                forwarder_ncrisc_kernel = ttnn.KernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/micro_ops/sdpa_reduce_to_all/kernels/forwarder.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=forwarder_core_range_set,
                    compile_time_args=forwarder_ct_args,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1,
                        noc=ttnn.NOC.RISCV_0_default,
                        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                    ),
                )

                # CB descriptors
                cb_local_l_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_local_l, input_l_device)
                cb_local_ms_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_local_ms, input_ms_device)
                cb_l_out_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_l_out, output_l_device)

                cb_r1_neighbor_l_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_r1_neighbor_l, r1_recv_device)
                cb_r1_neighbor_l_desc.total_size = out_tiles * aligned_page_size

                cb_r2_neighbor_l_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_r2_neighbor_l, r2_recv_device)
                cb_r2_neighbor_l_desc.total_size = out_tiles * aligned_page_size

                tile_desc = ttnn.TileDescriptor(tile_height, tile_width)
                input_dtype = input_l_device.dtype

                cb_r1_neighbor_ms_desc = ttnn.CBDescriptor(
                    total_size=aligned_page_size,
                    core_ranges=shard_grid,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=cb_r1_neighbor_ms,
                            data_format=input_dtype,
                            page_size=aligned_page_size,
                            tile=tile_desc,
                        )
                    ],
                )

                cb_r1_result_l_desc = ttnn.CBDescriptor(
                    total_size=out_tiles * aligned_page_size,
                    core_ranges=shard_grid,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=cb_r1_result_l,
                            data_format=input_dtype,
                            page_size=aligned_page_size,
                            tile=tile_desc,
                        )
                    ],
                )

                cb_r1_result_ms_desc = ttnn.CBDescriptor(
                    total_size=aligned_page_size,
                    core_ranges=shard_grid,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=cb_r1_result_ms,
                            data_format=input_dtype,
                            page_size=aligned_page_size,
                            tile=tile_desc,
                        )
                    ],
                )

                cb_r2_neighbor_ms_desc = ttnn.CBDescriptor(
                    total_size=aligned_page_size,
                    core_ranges=shard_grid,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=cb_r2_neighbor_ms,
                            data_format=input_dtype,
                            page_size=aligned_page_size,
                            tile=tile_desc,
                        )
                    ],
                )

                cb_ms_out_desc = ttnn.CBDescriptor(
                    total_size=aligned_page_size,
                    core_ranges=shard_grid,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=cb_ms_out,
                            data_format=input_dtype,
                            page_size=aligned_page_size,
                            tile=tile_desc,
                        )
                    ],
                )

                cb_packet_slot_desc = ttnn.CBDescriptor(
                    total_size=2 * header_cb_size,
                    core_ranges=shard_grid,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=cb_packet_slot,
                            data_format=ttnn.uint32,
                            page_size=header_cb_size,
                            tile=tile_desc,
                        )
                    ],
                )

                # Position CB (aliased from position tensor if enabled, otherwise dummy)
                if position_enabled:
                    cb_position_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_position, position_device)
                else:
                    # Dummy CB when position is disabled
                    cb_position_desc = ttnn.CBDescriptor(
                        total_size=aligned_page_size,
                        core_ranges=shard_grid,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=cb_position,
                                data_format=input_dtype,
                                page_size=aligned_page_size,
                                tile=tile_desc,
                            )
                        ],
                    )

                # Semaphores
                forwarder_semaphores = [
                    ttnn.SemaphoreDescriptor(id=fwd_r1_sem_id, core_ranges=forwarder_core_range_set, initial_value=0),
                    ttnn.SemaphoreDescriptor(id=fwd_r2_sem_id, core_ranges=forwarder_core_range_set, initial_value=0),
                    ttnn.SemaphoreDescriptor(id=bwd_r1_sem_id, core_ranges=forwarder_core_range_set, initial_value=0),
                    ttnn.SemaphoreDescriptor(id=bwd_r2_sem_id, core_ranges=forwarder_core_range_set, initial_value=0),
                ]

                program = ttnn.ProgramDescriptor(
                    kernels=[
                        reader_kernel,
                        writer_kernel,
                        compute_kernel,
                        forwarder_brisc_kernel,
                        forwarder_ncrisc_kernel,
                    ],
                    semaphores=forwarder_semaphores,
                    cbs=[
                        cb_local_l_desc,
                        cb_local_ms_desc,
                        cb_r1_neighbor_l_desc,
                        cb_r1_neighbor_ms_desc,
                        cb_r1_result_l_desc,
                        cb_r1_result_ms_desc,
                        cb_r2_neighbor_l_desc,
                        cb_r2_neighbor_ms_desc,
                        cb_l_out_desc,
                        cb_ms_out_desc,
                        cb_packet_slot_desc,
                        cb_position_desc,
                    ],
                )

                # Runtime args
                reader_rt_args = ttnn.RuntimeArgs()
                writer_rt_args = ttnn.RuntimeArgs()
                compute_rt_args = ttnn.RuntimeArgs()
                forwarder_brisc_rt_args = ttnn.RuntimeArgs()
                forwarder_ncrisc_rt_args = ttnn.RuntimeArgs()

                r1_recv_buffer_addr = r1_recv_device.buffer_address()
                r2_recv_buffer_addr = r2_recv_device.buffer_address()

                # Neighbor coords (torus/ring only)
                fwd_row, fwd_col = _get_neighbor_coord(mesh_shape, row, col, +1, cluster_axis)
                bwd_row, bwd_col = _get_neighbor_coord(mesh_shape, row, col, -1, cluster_axis)

                fwd_coord = ttnn.MeshCoordinate(fwd_row, fwd_col)
                bwd_coord = ttnn.MeshCoordinate(bwd_row, bwd_col)

                # Calculate forward and backward device indices (for position mask lookup)
                fwd_device_idx = fwd_row * mesh_cols + fwd_col
                bwd_device_idx = bwd_row * mesh_cols + bwd_col

                fabric_node_id = mesh_device.get_fabric_node_id(coord)
                fwd_fabric_node_id = mesh_device.get_fabric_node_id(fwd_coord)
                bwd_fabric_node_id = mesh_device.get_fabric_node_id(bwd_coord)

                # Split shard cores per link
                cores_per_link = num_shard_cores // num_links
                cores_link_1 = shard_cores[:cores_per_link]
                cores_link_2 = shard_cores[cores_per_link:]

                class DirectionConfig:
                    def __init__(self, r1_sem, r2_sem, dst_node_id, buffer_base):
                        self.r1_sem = r1_sem
                        self.r2_sem = r2_sem
                        self.dst_node_id = dst_node_id
                        self.buffer_base = buffer_base
                        self.r1_worker_count = 0
                        self.r2_worker_count = 0

                # Reader args - need to be set per-core for position indices
                # Set base args first (common to all cores)
                for core in shard_cores:
                    reader_rt_args[core.x][core.y] = [
                        r1_recv_sem_addr,
                        r2_recv_sem_addr,
                        r1_recv_buffer_addr,
                        r2_recv_buffer_addr,
                    ]

                for link_idx in range(num_links):
                    cores_for_link = cores_link_1 if link_idx == 0 else cores_link_2
                    fwd_core = forwarder_cores[link_idx]
                    fwd_core_noc = device.worker_core_from_logical_core(fwd_core)

                    fwd_cfg = DirectionConfig(fwd_r1_sem_id, fwd_r2_sem_id, fwd_fabric_node_id, forwarder_buffer_base)
                    bwd_cfg = DirectionConfig(
                        bwd_r1_sem_id,
                        bwd_r2_sem_id,
                        bwd_fabric_node_id,
                        forwarder_buffer_base + ncrisc_buffer_offset,
                    )

                    for worker_idx, core in enumerate(cores_for_link):
                        is_type_a = ((row + worker_idx) % 2) == 0
                        r1_cfg = fwd_cfg if is_type_a else bwd_cfg
                        r2_cfg = bwd_cfg if is_type_a else fwd_cfg

                        r1_slot_idx = r1_cfg.r1_worker_count * slots_per_worker
                        r1_cfg.r1_worker_count += 1
                        r2_slot_idx = r2_cfg.r2_worker_count * slots_per_worker
                        r2_cfg.r2_worker_count += 1

                        r1_slot_addr = r1_cfg.buffer_base + (r1_slot_idx * slot_size)
                        r2_slot_addr = r2_cfg.buffer_base + r2_buffer_offset + (r2_slot_idx * slot_size)

                        core_noc = device.worker_core_from_logical_core(core)

                        writer_rt_args[core.x][core.y] = [
                            int(r1_cfg.dst_node_id.mesh_id),
                            r1_cfg.dst_node_id.chip_id,
                            r1_recv_buffer_addr,
                            r1_recv_sem_addr,
                            int(r2_cfg.dst_node_id.mesh_id),
                            r2_cfg.dst_node_id.chip_id,
                            r2_recv_buffer_addr,
                            r2_recv_sem_addr,
                            core_noc.x,
                            core_noc.y,
                            fwd_core_noc.x,
                            fwd_core_noc.y,
                            r1_slot_addr,
                            r1_cfg.r1_sem,
                            r1_slot_idx,
                            r2_slot_addr,
                            r2_cfg.r2_sem,
                            r2_slot_idx,
                        ]

                        # Compute runtime args: device indices for position lookup
                        if position_enabled:
                            # Determine this core's neighbor device IDs based on direction
                            # Type A: R1=forward, R2=backward
                            # Type B: R1=backward, R2=forward
                            if is_type_a:
                                r1_neighbor_device_idx = fwd_device_idx
                                r2_neighbor_device_idx = bwd_device_idx
                            else:
                                r1_neighbor_device_idx = bwd_device_idx
                                r2_neighbor_device_idx = fwd_device_idx

                            # Compute R2 neighbor's R1 neighbor based on R2 neighbor's type
                            # R2 neighbor's worker_idx at the same core position
                            r2_neighbor_worker_idx = worker_idx  # Same relative position in the link
                            r2_neighbor_row = r2_neighbor_device_idx // mesh_cols
                            r2_neighbor_is_type_a = ((r2_neighbor_row + r2_neighbor_worker_idx) % 2) == 0

                            # R2 neighbor's R1 direction
                            if r2_neighbor_is_type_a:
                                # R2 neighbor is Type A → its R1 is forward
                                r2_neighbor_r1_neighbor_idx = (r2_neighbor_device_idx + 1) % num_devices
                            else:
                                # R2 neighbor is Type B → its R1 is backward
                                r2_neighbor_r1_neighbor_idx = (r2_neighbor_device_idx - 1 + num_devices) % num_devices

                            # Pass device indices - kernel will read position values from CB
                            compute_rt_args[core.x][core.y] = [
                                device_idx,
                                r1_neighbor_device_idx,
                                r2_neighbor_device_idx,
                                r2_neighbor_r1_neighbor_idx,
                            ]

                            reader_rt_args[core.x][core.y].extend(
                                [
                                    device_idx,
                                    r1_neighbor_device_idx,
                                    r2_neighbor_device_idx,
                                    r2_neighbor_r1_neighbor_idx,
                                ]
                            )
                        # If position is disabled, no runtime args needed (compute kernel uses defaults)

                        # Append scatter runtime args (only when scatter is enabled)
                        if scatter_enabled:
                            global_worker_idx = link_idx * cores_per_link + worker_idx
                            scatter_dest_l1_addr = scatter_dest_per_device[device_idx].buffer_address()
                            scatter_rt = [scatter_dest_l1_addr]
                            for row_j in range(scatter_ct_num_rows):
                                dest_core = scatter_dest_cores_list[row_j * num_shard_cores + global_worker_idx]
                                dest_core_noc = device.worker_core_from_logical_core(dest_core)
                                scatter_rt.extend([dest_core_noc.x, dest_core_noc.y])
                            writer_rt_args[core.x][core.y].extend(scatter_rt)

                    forwarder_brisc_rt_args[fwd_core.x][fwd_core.y] = [
                        forwarder_buffer_base,
                        0,
                        fwd_r1_sem_id,
                        fwd_r2_sem_id,
                    ]
                    brisc_fabric_args = ttnn.setup_fabric_connection(
                        src_fabric_node_id=fabric_node_id,
                        dst_fabric_node_id=fwd_fabric_node_id,
                        link_idx=link_idx,
                        program_descriptor=program,
                        worker_core=fwd_core,
                    )
                    forwarder_brisc_rt_args[fwd_core.x][fwd_core.y].extend(brisc_fabric_args)

                    forwarder_ncrisc_rt_args[fwd_core.x][fwd_core.y] = [
                        forwarder_buffer_base,
                        ncrisc_buffer_offset,
                        bwd_r1_sem_id,
                        bwd_r2_sem_id,
                    ]
                    ncrisc_fabric_args = ttnn.setup_fabric_connection(
                        src_fabric_node_id=fabric_node_id,
                        dst_fabric_node_id=bwd_fabric_node_id,
                        link_idx=link_idx,
                        program_descriptor=program,
                        worker_core=fwd_core,
                    )
                    forwarder_ncrisc_rt_args[fwd_core.x][fwd_core.y].extend(ncrisc_fabric_args)

                program.kernels[0].runtime_args = reader_rt_args
                program.kernels[1].runtime_args = writer_rt_args
                program.kernels[2].runtime_args = compute_rt_args
                program.kernels[3].runtime_args = forwarder_brisc_rt_args
                program.kernels[4].runtime_args = forwarder_ncrisc_rt_args

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        io_tensors = [
            input_tensor_l_mesh,
            input_tensor_ms_mesh,
            output_tensor_l_mesh,
            r1_recv_tensor_mesh,
            r2_recv_tensor_mesh,
            forwarder_scratch_mesh,
        ]
        if scatter_enabled:
            io_tensors.append(scatter_dest_tensor_mesh)
        if position_enabled:
            io_tensors.append(position_tensor_mesh)
        ttnn.generic_op(io_tensors, mesh_program_descriptor)

        return output_tensor_l_mesh
