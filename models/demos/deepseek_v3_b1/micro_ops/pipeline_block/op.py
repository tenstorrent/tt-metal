# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import MeshWrapper, SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size


class PipelineBlock:
    def __init__(
        self,
        mesh_device,
        pipeline_core_coord,
        h2d_socket_fifo_size,
        d2h_socket_fifo_size,
        intermediate_socket_fifo_size,
        entry_node_downstream=None,
        exit_node_upstream=None,
        embedding_tensor=None,
    ):
        self.my_mesh_id = mesh_device.get_system_mesh_id()
        self.is_pipeline_start = self.my_mesh_id == 0

        pipeline_config = ttnn._ttnn.operations.experimental.generate_blitz_decode_pipeline(mesh_device)

        num_procs = int(ttnn.distributed_context_get_size())
        assert len(pipeline_config) == num_procs + 1

        self.embedding_size_bytes = embedding_tensor.shape[3] * dtype_size(embedding_tensor.dtype)

        if self.is_pipeline_start:
            h2d_device_coord = pipeline_config[self.my_mesh_id].entry_node_coord
            d2h_device_coord = pipeline_config[num_procs].exit_node_coord
            token_size_bytes = 64  # Hardcode for now - don't expect this to change

            assert (
                h2d_socket_fifo_size >= token_size_bytes
            ), "H2D Socket FIFO Size must be greater than or equal to token size"
            assert (
                d2h_socket_fifo_size >= token_size_bytes
            ), "D2H Socket FIFO Size must be greater than or equal to token size"

            self.h2d_socket = ttnn.H2DSocket(
                mesh_device,
                ttnn.MeshCoreCoord(h2d_device_coord, pipeline_core_coord),
                ttnn.BufferType.L1,
                h2d_socket_fifo_size,
                ttnn.H2DMode.HOST_PUSH,  # Host Push is faster than Device Pull for small page sizes
            )
            self.d2h_socket = ttnn.D2HSocket(
                mesh_device, ttnn.MeshCoreCoord(d2h_device_coord, pipeline_core_coord), d2h_socket_fifo_size
            )

            self.host_io = HostInterface(
                self.h2d_socket,
                self.d2h_socket,
                token_size_bytes,
                self.embedding_size_bytes,
                core_to_core_socket_buffer_size=intermediate_socket_fifo_size,
                h2d_downstream_core=ttnn.MeshCoreCoord(
                    pipeline_config[self.my_mesh_id].exit_node_coord, pipeline_core_coord
                ),
                d2h_upstream_core=ttnn.MeshCoreCoord(pipeline_config[num_procs].entry_node_coord, pipeline_core_coord),
                embedding_tensor=embedding_tensor,
            )

            self.exit_socket_interface = SocketInterface(
                self.embedding_size_bytes,
                intermediate_socket_fifo_size,
                self.embedding_size_bytes,
                ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id].exit_node_coord, pipeline_core_coord),
                ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id + 1].entry_node_coord, pipeline_core_coord),
                upstream_socket=self.host_io.get_downstream_socket(),
                sender_mesh=MeshWrapper(mesh_device),
                receiver_mesh=MeshWrapper(mesh_id=self.my_mesh_id + 1),
            )

            self.entry_socket_interface = SocketInterface(
                self.embedding_size_bytes,
                intermediate_socket_fifo_size,
                self.embedding_size_bytes,
                ttnn.MeshCoreCoord(pipeline_config[num_procs - 1].exit_node_coord, pipeline_core_coord),
                ttnn.MeshCoreCoord(pipeline_config[num_procs].entry_node_coord, pipeline_core_coord),
                downstream_socket=self.host_io.get_upstream_socket(),
                sender_mesh=MeshWrapper(mesh_id=num_procs - 1),
                receiver_mesh=MeshWrapper(mesh_device),
            )

        else:
            self.entry_socket_interface = SocketInterface(
                self.embedding_size_bytes,
                intermediate_socket_fifo_size,
                self.embedding_size_bytes,
                ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id - 1].exit_node_coord, pipeline_core_coord),
                ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id].entry_node_coord, pipeline_core_coord),
                downstream_core_coord=entry_node_downstream
                if entry_node_downstream
                else ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id].exit_node_coord, pipeline_core_coord),
                sender_mesh=MeshWrapper(mesh_id=self.my_mesh_id - 1),
                receiver_mesh=MeshWrapper(mesh_device),
            )
            self.exit_socket_interface = SocketInterface(
                self.embedding_size_bytes,
                intermediate_socket_fifo_size,
                self.embedding_size_bytes,
                ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id].exit_node_coord, pipeline_core_coord),
                ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id + 1].entry_node_coord, pipeline_core_coord),
                upstream_core_coord=exit_node_upstream,
                upstream_socket=self.entry_socket_interface.get_downstream_socket() if not exit_node_upstream else None,
                sender_mesh=MeshWrapper(mesh_device),
                receiver_mesh=MeshWrapper(mesh_id=self.my_mesh_id + 1 if self.my_mesh_id < num_procs - 1 else 0),
            )

    def run(self):
        if self.is_pipeline_start:
            self.host_io.run()
            self.exit_socket_interface.run()
            self.entry_socket_interface.run()
        else:
            self.exit_socket_interface.run()
            self.entry_socket_interface.run()

    def terminate(self):
        # Multi-Process barrier here that all outstanding requests issued to pipeline block
        # are completed by all stages before termination signal is sent
        ttnn.distributed_context_barrier()
        if self.is_pipeline_start:
            self.host_io.terminate(False)
            self.entry_socket_interface.terminate(False)
            self.exit_socket_interface.terminate(True)
        else:
            self.entry_socket_interface.terminate(False)
            self.exit_socket_interface.terminate(True)

    def is_first_pipeline_stage(self):
        return self.is_pipeline_start

    def write_token(self, token_tensor):
        assert self.is_first_pipeline_stage(), "Token can only be written to the first pipeline stage"
        self.h2d_socket.write_tensor(token_tensor)

    def read_output(self, output_tensor):
        assert self.is_first_pipeline_stage(), "Output can only be read from the first pipeline stage"
        self.d2h_socket.read_tensor(output_tensor)
