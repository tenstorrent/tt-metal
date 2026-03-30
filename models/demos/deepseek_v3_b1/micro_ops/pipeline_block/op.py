# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Host Pipeline Block.

Provides a PipelineBlock class that encapsulates a single stage of a multi-host
pipeline. Each PipelineBlock manages the setup and lifecycle of D2D socket
interfaces for forwarding data between pipeline stages across hosts.

Pipeline topology is determined by generate_blitz_decode_pipeline(), which
maps physical ASIC locations to logical mesh coordinates for each stage.

There are four distinct stage configurations:

  1. First stage (mesh_id == 0):
     H2D socket receives tokens from host, looks up embedding in DRAM,
     forwards embedding rows downstream via exit D2D socket.
     If loopback is enabled, also receives results from the last stage
     via entry D2D socket and sends them back to host via D2H socket.

  2. Middle stage (0 < mesh_id < num_procs - 1):
     Entry D2D socket receives data from previous stage, exit D2D socket
     forwards it to the next stage. Pure passthrough.

  3. Last stage with loopback (mesh_id == num_procs - 1, loopback enabled):
     Entry D2D receives from previous stage, exit D2D sends back to
     stage 0's loopback entry.

  4. Last stage without loopback (mesh_id == num_procs - 1, loopback disabled):
     Entry D2D receives from previous stage, D2H socket sends results
     directly to the host on this process. The entry D2D's downstream
     socket is wired to the D2H kernel's upstream socket so data flows
     through a single shared socket pair.
"""

import ttnn
from models.demos.deepseek_v3_b1.metadata.metadata import DeepseekMetadata
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import MeshWrapper, SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size


class PipelineBlock:
    def __init__(
        self,
        mesh_device,
        pipeline_core_coord,
        upstream_d2d_socket_fifo_size,
        downstream_d2d_socket_fifo_size,
        upstream_d2d_socket_page_size,
        downstream_d2d_socket_page_size,
        h2d_socket_fifo_size=None,
        d2h_socket_fifo_size=None,
        d2h_socket_page_size=None,
        entry_node_downstream=None,
        exit_node_upstream=None,
        exit_upstream_page_size=None,
        embedding_tensor=None,
        initialize_loopback=True,
        forward_metadata=False,
    ):
        assert (
            upstream_d2d_socket_fifo_size >= upstream_d2d_socket_page_size
        ), "Upstream D2D Socket FIFO Size must be greater than or equal to upstream D2D Socket Page Size"
        assert (
            downstream_d2d_socket_fifo_size >= downstream_d2d_socket_page_size
        ), "Downstream D2D Socket FIFO Size must be greater than or equal to downstream D2D Socket Page Size"

        self.my_mesh_id = mesh_device.get_system_mesh_id()
        self.num_procs = int(ttnn.distributed_context_get_size())
        self.initialize_loopback = initialize_loopback

        pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
        if initialize_loopback:
            assert len(pipeline_config) == self.num_procs + 1

        self.is_pipeline_start = self.my_mesh_id == 0
        self.is_last_stage = self.my_mesh_id == self.num_procs - 1
        self.has_exit = not self.is_last_stage or initialize_loopback
        self.has_d2h = (self.is_last_stage and not initialize_loopback) or (
            self.is_pipeline_start and initialize_loopback
        )

        self.host_io = None
        self.h2d_socket = None
        self.d2h_socket = None
        self.entry_socket_interface = None
        self.exit_socket_interface = None

        token_size_bytes = 64

        if self.is_pipeline_start:
            self._init_first_stage(
                mesh_device,
                pipeline_config,
                pipeline_core_coord,
                token_size_bytes,
                upstream_d2d_socket_fifo_size,
                downstream_d2d_socket_fifo_size,
                upstream_d2d_socket_page_size,
                downstream_d2d_socket_page_size,
                h2d_socket_fifo_size,
                d2h_socket_fifo_size,
                d2h_socket_page_size,
                embedding_tensor,
            )
        elif self.is_last_stage and not initialize_loopback:
            self._init_last_stage_with_d2h(
                mesh_device,
                pipeline_config,
                pipeline_core_coord,
                token_size_bytes,
                upstream_d2d_socket_fifo_size,
                upstream_d2d_socket_page_size,
                downstream_d2d_socket_page_size,
                d2h_socket_fifo_size,
                d2h_socket_page_size,
                exit_node_upstream,
            )
        else:
            self._init_forwarding_stage(
                mesh_device,
                pipeline_config,
                pipeline_core_coord,
                upstream_d2d_socket_fifo_size,
                downstream_d2d_socket_fifo_size,
                upstream_d2d_socket_page_size,
                downstream_d2d_socket_page_size,
                entry_node_downstream,
                exit_node_upstream,
                exit_upstream_page_size,
                DeepseekMetadata.aligned_size_bytes() if forward_metadata else 0,
            )

    def _init_first_stage(
        self,
        mesh_device,
        pipeline_config,
        pipeline_core_coord,
        token_size_bytes,
        upstream_d2d_socket_fifo_size,
        downstream_d2d_socket_fifo_size,
        upstream_d2d_socket_page_size,
        downstream_d2d_socket_page_size,
        h2d_socket_fifo_size,
        d2h_socket_fifo_size,
        d2h_socket_page_size,
        embedding_tensor,
    ):
        assert h2d_socket_fifo_size is not None, "H2D Socket FIFO Size must be provided to first pipeline stage"
        assert embedding_tensor is not None, "Embedding Tensor must be provided to first pipeline stage"

        h2d_device_coord = pipeline_config[self.my_mesh_id].entry_node_coord

        embedding_size_bytes = embedding_tensor.shape[-1] * dtype_size(embedding_tensor.dtype)

        if self.initialize_loopback:
            assert d2h_socket_fifo_size is not None, "D2H Socket FIFO Size must be provided to first pipeline stage"
            assert d2h_socket_page_size is not None, "D2H Socket Page Size must be provided to first pipeline stage"
            assert d2h_socket_fifo_size >= d2h_socket_page_size

        assert h2d_socket_fifo_size >= token_size_bytes
        assert downstream_d2d_socket_page_size == embedding_size_bytes
        assert upstream_d2d_socket_page_size == d2h_socket_page_size

        self.h2d_socket = ttnn.H2DSocket(
            mesh_device,
            ttnn.MeshCoreCoord(h2d_device_coord, pipeline_core_coord),
            ttnn.BufferType.L1,
            h2d_socket_fifo_size,
            ttnn.H2DMode.HOST_PUSH,
        )

        if self.initialize_loopback:
            d2h_device_coord = pipeline_config[self.num_procs].exit_node_coord
            self.d2h_socket = ttnn.D2HSocket(
                mesh_device, ttnn.MeshCoreCoord(d2h_device_coord, pipeline_core_coord), d2h_socket_fifo_size
            )

        self.host_io = HostInterface(
            self.h2d_socket,
            self.d2h_socket,
            token_size_bytes,
            d2h_socket_page_size,
            core_to_core_socket_buffer_size=downstream_d2d_socket_fifo_size,
            h2d_downstream_core=ttnn.MeshCoreCoord(
                pipeline_config[self.my_mesh_id].exit_node_coord, pipeline_core_coord
            ),
            d2h_upstream_core=ttnn.MeshCoreCoord(pipeline_config[self.num_procs].entry_node_coord, pipeline_core_coord),
            embedding_tensor=embedding_tensor,
        )

        self.exit_socket_interface = SocketInterface(
            downstream_d2d_socket_page_size,
            downstream_d2d_socket_fifo_size,
            downstream_d2d_socket_page_size,
            ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id].exit_node_coord, pipeline_core_coord),
            ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id + 1].entry_node_coord, pipeline_core_coord),
            upstream_socket=self.host_io.get_downstream_socket(),
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=self.my_mesh_id + 1),
        )

        if self.initialize_loopback:
            self.entry_socket_interface = SocketInterface(
                upstream_d2d_socket_page_size,
                upstream_d2d_socket_fifo_size,
                upstream_d2d_socket_page_size,
                ttnn.MeshCoreCoord(pipeline_config[self.num_procs - 1].exit_node_coord, pipeline_core_coord),
                ttnn.MeshCoreCoord(pipeline_config[self.num_procs].entry_node_coord, pipeline_core_coord),
                downstream_socket=self.host_io.get_upstream_socket(),
                sender_mesh=MeshWrapper(mesh_id=self.num_procs - 1),
                receiver_mesh=MeshWrapper(mesh_device),
            )

    def _init_last_stage_with_d2h(
        self,
        mesh_device,
        pipeline_config,
        pipeline_core_coord,
        token_size_bytes,
        upstream_d2d_socket_fifo_size,
        upstream_d2d_socket_page_size,
        downstream_d2d_socket_page_size,
        d2h_socket_fifo_size,
        d2h_socket_page_size,
        exit_node_upstream,
    ):
        assert d2h_socket_fifo_size is not None, "D2H Socket FIFO Size must be provided to last pipeline stage"
        assert d2h_socket_page_size is not None, "D2H Socket Page Size must be provided to last pipeline stage"
        assert d2h_socket_fifo_size >= d2h_socket_page_size

        d2h_device_coord = pipeline_config[self.num_procs - 1].exit_node_coord
        d2h_upstream_core = (
            exit_node_upstream
            if exit_node_upstream
            else ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id].entry_node_coord, pipeline_core_coord)
        )

        self.d2h_socket = ttnn.D2HSocket(
            mesh_device, ttnn.MeshCoreCoord(d2h_device_coord, pipeline_core_coord), d2h_socket_fifo_size
        )

        # HostInterface creates an upstream_socket_pair (entry_core → exit_core) for its D2H
        # kernel. We then wire the entry_socket_interface's downstream to the same pair so that
        # the D2D exchange kernel pushes into the socket the D2H kernel reads from.
        self.host_io = HostInterface(
            None,
            self.d2h_socket,
            token_size_bytes,
            d2h_socket_page_size,
            core_to_core_socket_buffer_size=downstream_d2d_socket_page_size,
            d2h_upstream_core=d2h_upstream_core,
        )

        self.entry_socket_interface = SocketInterface(
            upstream_d2d_socket_page_size,
            upstream_d2d_socket_fifo_size,
            upstream_d2d_socket_page_size,
            ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id - 1].exit_node_coord, pipeline_core_coord),
            ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id].entry_node_coord, pipeline_core_coord),
            downstream_socket=self.host_io.get_upstream_socket(),
            sender_mesh=MeshWrapper(mesh_id=self.my_mesh_id - 1),
            receiver_mesh=MeshWrapper(mesh_device),
        )

    def _init_forwarding_stage(
        self,
        mesh_device,
        pipeline_config,
        pipeline_core_coord,
        upstream_d2d_socket_fifo_size,
        downstream_d2d_socket_fifo_size,
        upstream_d2d_socket_page_size,
        downstream_d2d_socket_page_size,
        entry_node_downstream,
        exit_node_upstream,
        exit_upstream_page_size=None,
        forward_metadata_size_bytes=0,
    ):
        self.entry_socket_interface = SocketInterface(
            upstream_d2d_socket_page_size,
            upstream_d2d_socket_fifo_size,
            upstream_d2d_socket_page_size,
            ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id - 1].exit_node_coord, pipeline_core_coord),
            ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id].entry_node_coord, pipeline_core_coord),
            downstream_core_coord=entry_node_downstream
            if entry_node_downstream
            else ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id].exit_node_coord, pipeline_core_coord),
            sender_mesh=MeshWrapper(mesh_id=self.my_mesh_id - 1),
            receiver_mesh=MeshWrapper(mesh_device),
        )

        next_mesh_id = self.my_mesh_id + 1 if not self.is_last_stage else 0
        use_multi_upstream = isinstance(exit_node_upstream, list)
        if use_multi_upstream:
            assert exit_upstream_page_size is not None, "exit_upstream_page_size required for multi-upstream mode"
            self.exit_socket_interface = SocketInterface(
                downstream_d2d_socket_page_size,
                downstream_d2d_socket_fifo_size,
                downstream_d2d_socket_page_size,
                ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id].exit_node_coord, pipeline_core_coord),
                ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id + 1].entry_node_coord, pipeline_core_coord),
                sender_mesh=MeshWrapper(mesh_device),
                receiver_mesh=MeshWrapper(mesh_id=next_mesh_id),
                upstream_core_coords=exit_node_upstream,
                upstream_page_size=exit_upstream_page_size,
                forward_metadata_size_bytes=forward_metadata_size_bytes,
            )
        else:
            self.exit_socket_interface = SocketInterface(
                downstream_d2d_socket_page_size,
                downstream_d2d_socket_fifo_size,
                downstream_d2d_socket_page_size,
                ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id].exit_node_coord, pipeline_core_coord),
                ttnn.MeshCoreCoord(pipeline_config[self.my_mesh_id + 1].entry_node_coord, pipeline_core_coord),
                upstream_core_coord=exit_node_upstream,
                upstream_socket=self.entry_socket_interface.get_downstream_socket() if not exit_node_upstream else None,
                sender_mesh=MeshWrapper(mesh_device),
                receiver_mesh=MeshWrapper(mesh_id=next_mesh_id),
            )

    def run(self):
        if self.is_pipeline_start:
            self.host_io.run()
            self.exit_socket_interface.run()
            if self.initialize_loopback:
                self.entry_socket_interface.run()
        else:
            self.entry_socket_interface.run()
            if self.has_exit:
                self.exit_socket_interface.run()
            if self.host_io is not None:
                self.host_io.run()

    def terminate(self):
        # Multi-Process barrier here that all outstanding requests issued to pipeline block
        # are completed by all stages before termination signal is sent
        ttnn.distributed_context_barrier()
        if self.is_pipeline_start:
            self.host_io.terminate(False)
            if self.initialize_loopback:
                self.entry_socket_interface.terminate(False)
            self.exit_socket_interface.terminate(True)
        else:
            self.entry_socket_interface.terminate(False)
            if self.has_exit:
                self.exit_socket_interface.terminate(not self.has_d2h)
            if self.host_io is not None:
                self.host_io.terminate(True)

    def is_first_pipeline_stage(self):
        return self.is_pipeline_start

    def write_token(self, token_tensor):
        assert self.is_first_pipeline_stage(), "Token can only be written to the first pipeline stage"
        self.h2d_socket.write_tensor(token_tensor)

    def read_output(self, output_tensor):
        assert (
            self.d2h_socket is not None
        ), "read_output requires a D2H socket: valid on stage 0 with loopback, or last stage without loopback"
        self.d2h_socket.read_tensor(output_tensor)

    def get_upstream_socket(self):
        if hasattr(self, "exit_socket_interface"):
            return self.exit_socket_interface.get_upstream_socket()
        elif hasattr(self, "host_io"):
            return self.host_io.get_upstream_socket()

    def get_downstream_socket(self):
        return self.entry_socket_interface.get_downstream_socket()

    def get_upstream_sockets(self):
        """Return list of sender sockets for multi-upstream exit SocketInterface."""
        return self.exit_socket_interface.get_upstream_sockets()
