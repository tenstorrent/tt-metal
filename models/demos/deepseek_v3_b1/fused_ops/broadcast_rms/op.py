# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.ccl_broadcast.op import DeepseekMinimalBroadcast
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor
from models.demos.deepseek_v3_b1.utils import float_to_uint32


class BroadcastRMSNorm:
    """
    Fused Broadcast + RMSNorm operation using ttnn.generic_op.
    NCRISC: ccl_broadcast reader kernel + RMSNorm reader (or just RMSNorm if skip_ccl)
    BRISC: ccl_broadcast writer (or no-op if skip_ccl)
    TRISC: RMSNorm compute

    When skip_ccl=True, the operation runs on a single device without CCL broadcast,
    effectively performing only the RMSNorm computation.
    """

    @staticmethod
    def golden(input_tensor, gamma_tensor, epsilon=1e-6):
        """
        PyTorch reference implementation of RMS norm for validation.
        Args:
            input_tensor: Input tensor (torch.Tensor)
            gamma_tensor: Gamma/weight tensor (torch.Tensor)
            epsilon: Small value to avoid division by zero
        Returns:
            Output tensor with RMS norm applied
        """
        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        normalized = input_tensor * torch.rsqrt(variance + epsilon)
        return normalized * gamma_tensor

    @staticmethod
    def op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        gamma_tensor,
        sender_coord,
        output_tensor,
        semaphores=None,
        cluster_axis=0,
        secondary_cluster_axis=None,
        num_links=1,
        epsilon=1e-6,
        fp32_dest_acc_en=False,
        rsqrt_fast_approx=False,
        skip_ccl=False,
        socket=None,
        *,
        fabric_config=None,
        broadcast_topology_override=None,
    ):
        """
        Execute fused Broadcast+RMSNorm operation.

        Args:
            skip_ccl: If True, skip CCL broadcast and run RMSNorm only (single-device mode).
                      In this mode, semaphores and sender_coord are ignored.
        """

        # Get mesh/device info
        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        # Get per-device tensors
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)

        # Get tile info from input tensor (use a sample device tensor)
        input_tensor_sample = input_tensors_per_device[0]
        tile = input_tensor_sample.tile
        tile_height, tile_width = tile.tile_shape

        input_shape = input_tensor_sample.shape
        dtype = input_tensor_sample.dtype
        element_size = dtype_size(dtype)
        tile_id_start = 0

        # bcast cb info
        payload_size_bytes = input_shape[0] * input_shape[1] * element_size
        packet_size_bytes = payload_size_bytes
        page_size_bytes = 32 * 32 * element_size  # interpret it as 32x32 tile to use the same cb as rmsnorm
        assert (
            payload_size_bytes % page_size_bytes == 0
        ), f"payload_size_bytes {payload_size_bytes} must be a multiple of page_size_bytes {page_size_bytes}"
        input_num_pages = payload_size_bytes // page_size_bytes
        num_pages_per_packet = packet_size_bytes // page_size_bytes

        socket_page_size = packet_size_bytes
        assert socket_page_size % 16 == 0, f"socket_page_size {socket_page_size} must be 16-byte aligned"
        assert socket_page_size == input_num_pages * page_size_bytes, (
            f"single-shot requires socket_page_size {socket_page_size} == full payload "
            f"{input_num_pages} * {page_size_bytes}"
        )

        # CB indices:
        #   0..2 are owned by RMSNorm in this fused op.
        #   broadcast-private CB id is explicit via bcast_cb_id.
        input_cb = 0
        gamma_cb = 1
        output_cb = 2
        bcast_cb_id = 3

        bcast_config = DeepseekMinimalBroadcast.configure(
            mesh_device=mesh_device,
            input_tensor_mesh=input_tensor_mesh,
            output_tensor=intermediate_tensor_mesh,
            sender_coord=sender_coord,
            semaphores=semaphores,
            socket=socket,
            skip_ccl=skip_ccl,
            bcast_cb_id=bcast_cb_id,
            num_links=num_links,
            fabric_config=fabric_config,
            broadcast_topology_override=broadcast_topology_override,
        )

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # for rms norm: interpret tile sizes
        data_format = dtype
        FULL_32x32_TILE = ttnn.Tile((32, 32))
        HALF_16x32_TILE = ttnn.Tile((16, 32))
        is_16x32_tile = (input_shape[1] // FULL_32x32_TILE.tile_shape[1]) % FULL_32x32_TILE.tile_shape[0] != 0
        interpreted_tile = HALF_16x32_TILE if is_16x32_tile else FULL_32x32_TILE
        tile_height, tile_width = interpreted_tile.tile_shape

        # Calculate single tile size in bytes (bfloat16 = 2 bytes per element)
        tile_size = interpreted_tile.get_tile_size(data_format)

        # Calculate num_tiles from tensor shape
        num_tiles = (input_shape[0] * input_shape[1]) // (tile_height * tile_width)

        # Number of elements
        numel = input_tensor_mesh.logical_volume()

        # Calculate runtime args
        epsilon_packed = float_to_uint32(epsilon)

        # Compute 1/sqrt(num_elements) for RMS reduction
        inv_sqrt_numel = 1.0 / math.sqrt(float(numel))
        scalar_packed = float_to_uint32(inv_sqrt_numel)

        # Define circular buffer page size
        cb_page_size = tile_size

        # Socket mode is an op-level setting
        use_socket = socket is not None

        # For each device in the mesh, create appropriate program
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)

                # Get the device's input and output tensors
                device_idx = row * mesh_cols + col
                input_tensor_device = input_tensors_per_device[device_idx]

                # Broadcast and RMSNorm currently share the same worker core.
                # Keep the roles explicit so we can separate them cleanly in future fused ops.
                bcast_worker_core = bcast_config.get_worker_core(coord)
                bcast_worker_core_set = bcast_config.get_worker_core_set(coord)

                # rmsnorm_worker_core same as bcast_worker_core
                # rmsnorm_worker_core_set same as bcast_worker_core_set

                fused_worker_core_set = (
                    bcast_worker_core_set  # Currently the same core(s) run both broadcast and RMSNorm
                )

                # CB roles:
                #   input_cb (0): RMSNorm input — always read by TRISC; has explicit page_size=2048 override.
                #   pkt_cb   (1): Broadcast staging — used by NCRISC/BRISC in CCL mode only.
                #
                # Who delivers data into input_cb depends on mode:
                #   CCL         (skip_ccl=False):           NCRISC signals it (intermediate_cb) after broadcast
                #   local       (skip_ccl=True, no socket): NCRISC signals it directly
                #   socket recv (skip_ccl=True, socket):    BRISC writes received payload here
                #
                ncrisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("use_socket", 1 if use_socket else 0),
                    ("rmsnorm_input_cb", input_cb),
                    ("rmsnorm_num_tiles", num_tiles),
                    # Only read in active CCL path in the kernel; value is ignored when skip_ccl=True.
                    ("intermediate_cb", input_cb),
                    ("gamma_cb", gamma_cb),
                ]
                ncrisc_named_compile_time_args.extend(bcast_config.get_ncrisc_named_ct_args(coord))

                brisc_named_compile_time_args = [("skip_ccl", 1 if skip_ccl else 0)]
                if bcast_config.has_bypass_socket_reader:
                    # Skip-CCL socket path reads directly into input_cb, which uses RMSNorm pages (num_tiles).
                    brisc_named_compile_time_args.extend(
                        bcast_config.get_socket_reader_ct_args(coord, input_cb, target_num_pages=num_tiles)
                    )
                    brisc_common_runtime_args = bcast_config.get_socket_reader_rt_args(coord)
                else:
                    brisc_named_compile_time_args.extend(bcast_config.get_brisc_named_ct_args(coord))
                    # Socket runtime args for BRISC (zeros when not using socket)
                    brisc_common_runtime_args = bcast_config.get_brisc_common_rt_args(coord)

                # Named compile-time args for TRISC (rmsnorm compute)
                trisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("rmsnorm_input_cb", input_cb),
                    ("rmsnorm_gamma_cb", gamma_cb),
                    ("rmsnorm_output_cb", output_cb),
                    ("rmsnorm_fp32_acc", 1 if fp32_dest_acc_en else 0),
                    ("rmsnorm_num_tiles", num_tiles),
                    ("rmsnorm_rsqrt_fast_approx", 1 if rsqrt_fast_approx else 0),
                ]

                # Common runtime args for writer (broadcast args shared across cores)
                writer_common_rt_args = bcast_config.get_ncrisc_common_rt_args(coord)

                # Create tile descriptor for proper tile dimensions
                tile_descriptor = ttnn.TileDescriptor(interpreted_tile)

                # Create circular buffer descriptors
                # CB 0: In multi-device mode, backed by intermediate_tensor_mesh (broadcast destination)
                #       In single-device mode, backed by input_tensor_mesh (direct input)
                cb0_backing_tensor = input_tensor_mesh if skip_ccl else intermediate_tensor_mesh
                in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, cb0_backing_tensor)
                in_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                in_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                # CB 1: Gamma (created from sharded gamma tensor)
                gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gamma_cb, gamma_tensor)
                gamma_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                gamma_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                # CB 2: Output (created from sharded tensor)
                out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor)
                out_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                out_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                bcast_cb_descriptor = bcast_config.get_cb_descriptor(coord)

                # Broadcast contributes the current define set. If this fused op
                # adds extra defines later, merge/de-dupe at the op layer.
                kernel_defines = bcast_config.get_kernel_defines(coord)

                # Unified kernel descriptor for fused op
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/broadcast_rms/kernels/broadcast_rms_kernel.cpp",
                    core_ranges=fused_worker_core_set,
                    ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
                    brisc_named_compile_time_args=brisc_named_compile_time_args,
                    trisc_named_compile_time_args=trisc_named_compile_time_args,
                    ncrisc_common_runtime_args=writer_common_rt_args,
                    brisc_common_runtime_args=brisc_common_runtime_args,
                    noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                    trisc_common_runtime_args=[epsilon_packed, scalar_packed],
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=False,
                        fp32_dest_acc_en=fp32_dest_acc_en,
                        dst_full_sync_en=fp32_dest_acc_en,
                    ),
                    defines=kernel_defines,
                    # Per-core runtime args: empty for BRISC (fabric args appended later)
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(bcast_worker_core, [])],  # Fabric args appended after program creation
                    ),
                )

                # Program descriptor
                # Keep descriptor order aligned with buffer indices: RMSNorm CBs first (0..2),
                # followed by any broadcast-private descriptors (starting at bcast_cb_id).
                program = ttnn.ProgramDescriptor(
                    kernels=unified_kernel.get_kernel_descriptors().kernels,
                    cbs=[in_cb_descriptor, gamma_cb_descriptor, out_cb_descriptor]
                    + ([] if bcast_cb_descriptor is None else [bcast_cb_descriptor]),
                    semaphores=[],
                )
                writer_rt_args_ref = program.kernels[0].runtime_args[bcast_worker_core.x][bcast_worker_core.y]
                writer_rt_args_ref.extend(bcast_config.get_ncrisc_per_core_rt_args(coord, program, bcast_worker_core))

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute generic_op
        result = ttnn.generic_op(
            [input_tensor_mesh, intermediate_tensor_mesh, gamma_tensor, output_tensor], mesh_program_descriptor
        )

        return result
