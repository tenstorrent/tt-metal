# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Forward Operation using ttnn.generic_op

Per-device data transfer into an output tensor. Can operate in two modes:

1. Socket mode (ENABLE_SOCKET_READER defined): BRISC reads from a D2D socket
   into the CB, then NCRISC writes CB data to the output tensor.

2. Tensor mode (no socket): data is pre-loaded in an input tensor that backs
   the CB. BRISC just marks the CB as ready, NCRISC writes to the output tensor.
   This mode is used for standalone perf benchmarking (comparable to broadcast).

Optionally, entry-column devices forward the data to a partner device in
another column via fabric (cross-column mode).
"""

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor

FORWARD_KERNEL_PATH = "models/demos/deepseek_v3_b1/micro_ops/forward/kernels/forward_kernel.cpp"


class DeepseekForward:
    """
    Per-device forward using ttnn.generic_op.

    Supports two input modes:
    - Socket mode: sockets parameter provides per-device receiver sockets.
    - Tensor mode: input_tensor_mesh parameter provides pre-loaded data (like broadcast).

    In both modes, NCRISC writes CB data into the output tensor. With cross-column
    enabled, entry-column devices also fabric-send to a partner in the other column.
    """

    @staticmethod
    def golden(input_tensor):
        """All devices should end up with a copy of the input tensor."""
        return input_tensor

    @staticmethod
    def op(
        output_tensor,
        sockets=None,
        *,
        input_tensor_mesh=None,
        forward_core,
        cross_column_semaphore=None,
        num_iterations=1,
        enable_cross_column=False,
        entry_column=0,
    ):
        """
        Execute forward operation.

        Exactly one of `sockets` or `input_tensor_mesh` must be provided:
        - sockets: List of per-device D2DSocket receivers. ENABLE_SOCKET_READER is defined.
        - input_tensor_mesh: Pre-loaded input tensor mesh. No socket reader — data is
          already in the tensor that backs the CB (like broadcast's non-socket path).

        When enable_cross_column=True, entry_column specifies which column has the
        source data and forwards to the partner column via fabric.
        """
        use_socket = sockets is not None
        use_tensor = input_tensor_mesh is not None
        if use_socket == use_tensor:
            raise ValueError("Exactly one of `sockets` or `input_tensor_mesh` must be provided")

        mesh_device = output_tensor.device()
        mesh_shape = mesh_device.shape
        mesh_rows = int(mesh_shape[0])
        mesh_cols = int(mesh_shape[1])

        output_per_device = ttnn.get_device_tensors(output_tensor)
        if use_tensor:
            input_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        else:
            input_per_device = None

        element_size = dtype_size(output_per_device[0].dtype)

        sample = output_per_device[0]
        shard_spec = sample.memory_config().shard_spec
        shard_h, shard_w = shard_spec.shape
        tile_h, tile_w = sample.tile.tile_shape
        page_size_bytes = tile_h * tile_w * element_size
        num_pages = (shard_h // tile_h) * (shard_w // tile_w)
        payload_size_bytes = num_pages * page_size_bytes
        socket_page_size = payload_size_bytes

        cross_col_sem_addr = 0
        if cross_column_semaphore is not None:
            cross_col_sem_addr = int(ttnn.get_global_semaphore_address(cross_column_semaphore))

        fabric_max_payload = 0
        num_fabric_packets = 0
        cross_col_payload = 0
        if enable_cross_column:
            fabric_max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
            cross_col_payload = payload_size_bytes
            num_fabric_packets = (cross_col_payload + fabric_max_payload - 1) // fabric_max_payload

        worker_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(forward_core, forward_core)])

        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        common_ct_args = [("forward_num_iterations", num_iterations)]

        all_coords = {}
        for r in range(mesh_rows):
            for c in range(mesh_cols):
                all_coords[(r, c)] = ttnn.MeshCoordinate(r, c)

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = all_coords[(row, col)]
                chip_id = row * mesh_cols + col

                if use_socket:
                    socket = sockets[chip_id] if chip_id < len(sockets) else None
                    is_entry = socket is not None
                else:
                    socket = None
                    if enable_cross_column:
                        is_entry = col == entry_column
                    else:
                        is_entry = True

                output_dev = output_per_device[chip_id]
                tensor_address = int(output_dev.buffer_address())
                data_core_physical = output_dev.device().worker_core_from_logical_core(forward_core)
                my_noc_x = int(data_core_physical.x)
                my_noc_y = int(data_core_physical.y)

                socket_config_addr = int(socket.get_config_buffer_address()) if socket is not None else 0

                is_entry_val = 1 if is_entry else 0
                fwd_fabric_max = fabric_max_payload if enable_cross_column else 0
                fwd_num_packets = num_fabric_packets if enable_cross_column else 0
                fwd_cross_payload = cross_col_payload if enable_cross_column else 0

                partner_tensor_addr = 0
                partner_noc_x = 0
                partner_noc_y = 0
                partner_chip_id = 0
                partner_mesh_id = 0
                if enable_cross_column and is_entry:
                    p_col = 1 - entry_column
                    p_chip = row * mesh_cols + p_col
                    partner_dev = output_per_device[p_chip]
                    partner_tensor_addr = int(partner_dev.buffer_address())
                    partner_core_phys = partner_dev.device().worker_core_from_logical_core(forward_core)
                    partner_noc_x = int(partner_core_phys.x)
                    partner_noc_y = int(partner_core_phys.y)
                    partner_fabric_node = mesh_device.get_fabric_node_id(all_coords[(row, p_col)])
                    partner_chip_id = int(partner_fabric_node.chip_id)
                    partner_mesh_id = int(partner_fabric_node.mesh_id)

                brisc_ct_args = [
                    ("forward_cb_id", 0),
                    ("forward_num_pages", num_pages),
                    ("forward_is_entry_column", is_entry_val),
                ] + common_ct_args

                ncrisc_ct_args = [
                    ("forward_cb_id", 0),
                    ("forward_num_pages", num_pages),
                    ("forward_page_size", page_size_bytes),
                    ("forward_is_entry_column", is_entry_val),
                    ("forward_fabric_max_payload", fwd_fabric_max),
                    ("forward_num_fabric_packets", fwd_num_packets),
                    ("forward_cross_column_payload", fwd_cross_payload),
                ] + common_ct_args

                brisc_common_rt = [
                    socket_config_addr,
                    socket_page_size,
                    1,  # socket_num_pages
                ]

                ncrisc_common_rt = [
                    tensor_address,
                    my_noc_x,
                    my_noc_y,
                    cross_col_sem_addr,
                    partner_tensor_addr,
                    partner_noc_x,
                    partner_noc_y,
                    partner_chip_id,
                    partner_mesh_id,
                ]

                defines = []
                if use_socket:
                    defines.append(("ENABLE_SOCKET_READER", "1"))

                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source=FORWARD_KERNEL_PATH,
                    core_ranges=worker_core_set,
                    ncrisc_named_compile_time_args=ncrisc_ct_args,
                    brisc_named_compile_time_args=brisc_ct_args,
                    trisc_named_compile_time_args=common_ct_args,
                    ncrisc_common_runtime_args=ncrisc_common_rt,
                    brisc_common_runtime_args=brisc_common_rt,
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(forward_core, [])],
                    ),
                    noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                    defines=defines,
                )

                kernel_result = unified_kernel.get_kernel_descriptors()

                if use_tensor:
                    cb_desc = ttnn.cb_descriptor_from_sharded_tensor(0, input_per_device[chip_id])
                else:
                    cb_desc = ttnn.cb_descriptor_from_sharded_tensor(0, output_dev)

                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels[:2],
                    semaphores=[],
                    cbs=[cb_desc],
                )

                if enable_cross_column and is_entry:
                    my_fabric_node = mesh_device.get_fabric_node_id(coord)
                    partner_fabric_node = mesh_device.get_fabric_node_id(all_coords[(row, 1 - entry_column)])
                    ncrisc_kernel_idx = 0
                    fwd_conn = ttnn.setup_fabric_connection(
                        my_fabric_node, partner_fabric_node, 0, program, forward_core
                    )
                    program.kernels[ncrisc_kernel_idx].runtime_args[forward_core.x][forward_core.y].extend(fwd_conn)

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        op_inputs = [output_tensor]
        if use_tensor:
            op_inputs = [input_tensor_mesh, output_tensor]

        return ttnn.generic_op(op_inputs, mesh_program_descriptor)
