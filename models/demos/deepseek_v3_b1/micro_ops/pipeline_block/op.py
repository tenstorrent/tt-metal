# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Host Pipeline Block.

Provides a PipelineBlock class that encapsulates a single stage of a multi-host
pipeline. Each PipelineBlock manages the setup and lifecycle of D2D socket
interfaces for forwarding data between pipeline stages across hosts.

Pipeline topology is determined by generate_blitz_decode_pipeline(), which
maps physical ASIC locations to logical mesh coordinates for each stage.

There are five distinct stage configurations:

  1. First stage (mesh_id == 0, num_procs > 1):
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

  5. Combined H2D + D2H stage (num_procs == 1, no loopback):
     Single-process pipeline that owns both host sockets on the same stage.
     H2D delivers each token + metadata to ``entry_node_downstream`` (compute
     input core); compute produces N output shards on ``exit_node_upstream``
     cores; the multi-upstream D2H sender (BRISC kernel) reads one page from
     each upstream and assembles them into a single D2H page back to host.
     No D2D socket interfaces are constructed — all I/O is through ``host_io``.
     Intended for plumbing a ``DecoderStage``-shaped compute directly to host.

Per-device parallel mode (pipeline_device_coords):
  When a list of MeshCoordinate device coordinates is provided, each channel
  maps to a different device in the mesh. Within each device, pipeline_core_coord
  (entry) and pipeline_exit_core_coord (exit) handle incoming and outgoing data,
  matching the caller-supplied pattern used by the single-channel forwarding stage.
  All programs are merged per device and dispatched in a single generic_op to avoid
  deadlock. Optionally supports the d2d_exchange_multiple_upstreams kernel per device
  via exit_upstream_cores and exit_upstream_page_size parameters.

entry_socket_interface / exit_socket_interface can be any of:
  - None: not applicable for this stage
  - SocketInterface: single-channel mode
  - list[SocketInterface]: per-device parallel mode (one per device)
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

import ttnn
from models.demos.deepseek_v3_b1.demo.pipeline_routing import (
    EdgeTransport,
    LocalStageEdge,
    LocalStageSocketPlan,
    StageEndpointRef,
    local_input_edge,
    local_output_edge,
)
from models.demos.deepseek_v3_b1.metadata.metadata import DeepseekMetadata
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import (
    MeshWrapper,
    SocketInterface,
    _create_socket_resource,
    _group_by_device,
)
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size


@dataclass
class StageMetadata:
    """Per-stage routing info: which MPI rank owns the stage and its mesh ID."""

    rank: int
    mesh_id: int


@dataclass
class PipelineConfigEntry:
    """Entry/exit device coordinates for one pipeline stage within its local submesh coord space."""

    entry_node_coord: ttnn.MeshCoordinate
    exit_node_coord: ttnn.MeshCoordinate


@dataclass
class HostIoPlacement:
    """Per-core placement for the four socket kernels at pipeline stage 0.

    When the H2D chip and the forward D2D exit chip are the same device,
    fwd_d2d_core must differ from h2d_core so two persistent BRISC kernels
    are not dispatched to the same Tensix core.  The same constraint applies
    to lb_d2d_core / d2h_core when the loopback D2D entry chip and D2H chip
    are the same device.
    """

    h2d_core: ttnn.CoreCoord
    d2h_core: ttnn.CoreCoord
    fwd_d2d_core: ttnn.CoreCoord
    lb_d2d_core: ttnn.CoreCoord

    @staticmethod
    def default(core: ttnn.CoreCoord) -> "HostIoPlacement":
        return HostIoPlacement(h2d_core=core, d2h_core=core, fwd_d2d_core=core, lb_d2d_core=core)


class LoopbackConfig:
    """Controls the loopback topology and host I/O socket placement for pipeline stage 0.

    Use the named constructors:
      ``LoopbackConfig.fabric_loopback()``  — physical ethernet from last stage to first
      ``LoopbackConfig.host_loopback()``    — MPI-based return from last stage to first
      ``LoopbackConfig.no_loopback()``      — no return path; pipeline terminates at last stage

    ``host_io_placement`` is optional for all modes; when ``None`` the default placement
    (all four sockets on ``pipeline_core_coord``) is applied at stage 0.
    """

    def __init__(self, *, _mode: str, host_io_placement: HostIoPlacement):
        self._mode = _mode
        self.host_io_placement = host_io_placement

    @staticmethod
    def fabric_loopback(host_io_placement: HostIoPlacement) -> "LoopbackConfig":
        """Physical fabric loopback: last stage sends back to first via ethernet."""
        return LoopbackConfig(_mode="fabric", host_io_placement=host_io_placement)

    @staticmethod
    def host_loopback(host_io_placement: HostIoPlacement) -> "LoopbackConfig":
        """Host MPI loopback: last stage returns data to first via host."""
        return LoopbackConfig(_mode="host", host_io_placement=host_io_placement)

    @staticmethod
    def no_loopback(host_io_placement: HostIoPlacement) -> "LoopbackConfig":
        """No loopback: pipeline terminates at last stage."""
        return LoopbackConfig(_mode="none", host_io_placement=host_io_placement)

    @property
    def initialize_loopback(self) -> bool:
        return self._mode == "fabric"


@runtime_checkable
class PipelineBlockKind(Protocol):
    """Structural interface for any pipeline-block-like object that ``Pipeline`` can drive.

    Both :class:`PipelineBlock` and the combined-stage block in
    ``demo/stage.py`` implement this. Methods listed here are the ones consumed by
    :class:`Pipeline` and by stage ``setup`` / ``launch_compute`` / ``terminate``;
    parallel-mode-only methods (``get_upstream_sockets`` etc.) are intentionally
    omitted because they're stage-specific and the caller knows the concrete type.
    """

    def run(self) -> None:
        ...

    def terminate(self) -> None:
        ...

    def is_first_pipeline_stage(self) -> bool:
        ...

    def write_token(self, token_tensor: ttnn.Tensor) -> None:
        ...

    def read_output(self, output_tensor: ttnn.Tensor) -> None:
        ...

    def push_dummy_token(self) -> None:
        ...

    def drain_dummy_output(self) -> None:
        ...

    def get_upstream_socket(self):
        ...

    def get_downstream_socket(self):
        ...

    def export_host_socket_descriptors(self, io_socket_descriptor_prefix: str) -> None:
        ...


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
        h2d_socket_page_size=None,
        d2h_socket_fifo_size=None,
        d2h_socket_page_size=None,
        entry_node_downstream=None,
        exit_node_upstream=None,
        exit_upstream_page_size=None,
        embedding_tensor=None,
        loopback=None,
        pipeline_device_coords=None,
        pipeline_exit_core_coord=None,
        entry_downstream_core=None,
        exit_upstream_cores=None,
        my_stage_idx=None,
        stages_metadata=None,
        pipeline_config=None,
        stage_plan: LocalStageSocketPlan | None = None,
        forward_metadata=False,
    ):
        if loopback is None:
            # Middle stages don't need a loopback config; fall back to a no-loopback placeholder.
            loopback = LoopbackConfig.no_loopback(HostIoPlacement.default(pipeline_core_coord))
        assert (
            upstream_d2d_socket_fifo_size >= upstream_d2d_socket_page_size
        ), "Upstream D2D Socket FIFO Size must be greater than or equal to upstream D2D Socket Page Size"
        assert (
            downstream_d2d_socket_fifo_size >= downstream_d2d_socket_page_size
        ), "Downstream D2D Socket FIFO Size must be greater than or equal to downstream D2D Socket Page Size"

        if my_stage_idx is None:
            my_stage_idx = mesh_device.get_system_mesh_id()

        self.my_stage_idx = my_stage_idx
        self.num_procs = int(ttnn.distributed_context_get_size())
        self.initialize_loopback = loopback.initialize_loopback
        self._loopback_mode = loopback._mode  # "fabric" | "host" | "none"
        self.mesh_device = mesh_device
        self._stage_plan = stage_plan
        self.parallel_devices = pipeline_device_coords is not None and len(pipeline_device_coords) > 0
        # Compute stages route input/output through dedicated compute cores; the plan-driven
        # builder does not yet model those taps, so they keep their dedicated init paths.
        self._has_compute_taps = entry_node_downstream is not None or exit_node_upstream is not None
        if stages_metadata is None:
            self._stages = {i: StageMetadata(rank=i, mesh_id=i) for i in range(self.num_procs)}
            pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(
                loopback.initialize_loopback
            )
        else:
            self._stages = stages_metadata
            assert pipeline_config is not None, "pipeline_config must be provided when stages_metadata is set"

        self.num_stages = len(pipeline_config) - 1 if self.initialize_loopback else len(pipeline_config)
        if self.initialize_loopback:
            assert len(pipeline_config) == self.num_stages + 1

        self.is_pipeline_start = self.my_stage_idx == 0
        self.is_last_stage = self.my_stage_idx == self.num_stages - 1
        self.has_exit = not self.is_last_stage or self.initialize_loopback
        self.has_d2h = (self.is_last_stage and not self.initialize_loopback) or (
            self.is_pipeline_start and self.initialize_loopback
        )

        self.host_io = None
        self.h2d_socket = None
        self.d2h_socket = None
        self.entry_socket_interface = None
        self.exit_socket_interface = None
        self._plan_forwarders = []
        self._h2d_page_size_bytes = None
        self._d2h_page_size_bytes = None

        # Default H2D page = DeepseekMetadata struct (256 B). Callers like the combined
        # H2D+D2H stage in inject-hidden-states mode override this to fit a full
        # (activation || metadata) payload directly on the H2D wire.
        token_size_bytes = (
            h2d_socket_page_size if h2d_socket_page_size is not None else DeepseekMetadata.aligned_size_bytes()
        )
        if self.is_pipeline_start and self.is_last_stage and not self.initialize_loopback:
            # Single-process pipeline (one stage spanning the mesh) with both H2D and D2H on
            # the same stage. Plugs a decoder-shaped compute (entry_node_downstream /
            # exit_node_upstream-list) directly into host I/O, with the D2H sender pulling
            # from N upstream cores. None of the upstream/downstream D2D parameters apply
            # here — only the H2D / D2H sizes and the H2D-to-compute local socket size do.
            self._init_combined_h2d_d2h_stage(
                mesh_device,
                pipeline_config,
                token_size_bytes,
                h2d_socket_fifo_size,
                d2h_socket_fifo_size,
                d2h_socket_page_size,
                h2d_to_compute_socket_buffer_size=downstream_d2d_socket_fifo_size,
                embedding_tensor=embedding_tensor,
                forward_metadata=forward_metadata,
                entry_node_downstream=entry_node_downstream,
                exit_node_upstream=exit_node_upstream,
                exit_upstream_page_size=exit_upstream_page_size,
                host_io_placement=loopback.host_io_placement,
            )
        elif self._uses_plan():
            self._init_from_plan(
                mesh_device,
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
                forward_metadata,
                entry_node_downstream,
                exit_node_upstream,
                loopback.host_io_placement,
            )
        elif self.is_pipeline_start:
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
                forward_metadata,
                host_io_placement=loopback.host_io_placement,
            )
        elif self.is_last_stage and not self.initialize_loopback:
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
                loopback.host_io_placement,
                entry_node_downstream,
            )
        else:
            if self.parallel_devices:
                assert (
                    pipeline_exit_core_coord is not None
                ), "pipeline_exit_core_coord is required for per-device parallel mode"
                self._init_parallel_device_forwarding_stage(
                    mesh_device,
                    pipeline_device_coords,
                    upstream_d2d_socket_fifo_size,
                    downstream_d2d_socket_fifo_size,
                    upstream_d2d_socket_page_size,
                    downstream_d2d_socket_page_size,
                    core_entry=pipeline_core_coord,
                    core_exit=pipeline_exit_core_coord,
                    entry_downstream_core=entry_downstream_core,
                    exit_upstream_cores=exit_upstream_cores,
                    exit_upstream_page_size=exit_upstream_page_size,
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

    def _uses_plan(self) -> bool:
        """Route a stage through the unified plan-driven socket builder (:meth:`_init_from_plan`).

        Covers every transport / host-I/O stage that has a resolved plan: split stages
        (any role), non-split pure-forwarding stages, and non-split stages that own host
        I/O — stage-0 embedding ingress and host-/fabric-loopback egress (the plan builder
        does rank-local host-socket placement via ``host_io_placement``, so H2D/D2H kernels
        no longer collide with their feeder forwarders).

        Stages that still use their dedicated init paths:
          - compute-tap stages (decoder / LM head): the plan does not yet model compute
            input/output cores (``entry_node_downstream`` / ``exit_node_upstream``);
          - per-device parallel mode;
          - the no-plan fallback, when ``_stage_plan is None``.
        """
        if self._stage_plan is None or self.parallel_devices or self._has_compute_taps:
            return False
        return True

    @staticmethod
    def _entry_owner(stage) -> int:
        if hasattr(stage, "entry_owner_rank"):
            return int(stage.entry_owner_rank)
        return int(stage.rank)

    @staticmethod
    def _exit_owner(stage) -> int:
        if hasattr(stage, "exit_owner_rank"):
            return int(stage.exit_owner_rank)
        return int(stage.rank)

    @staticmethod
    def _mesh_id(stage) -> int:
        return int(stage.mesh_id)

    @staticmethod
    def _core_for_endpoint(endpoint: StageEndpointRef, pipeline_core_coord: ttnn.CoreCoord) -> ttnn.MeshCoreCoord:
        return ttnn.MeshCoreCoord(endpoint.placement.mesh_coord, pipeline_core_coord)

    @staticmethod
    def _mesh_wrapper_for_endpoint(mesh_device, endpoint: StageEndpointRef, my_rank: int) -> MeshWrapper:
        if endpoint.owner_rank == my_rank:
            return MeshWrapper(mesh_device)
        return MeshWrapper(rank=endpoint.owner_rank, mesh_id=endpoint.mesh_id)

    def _create_socket_resource_for_edge(
        self,
        edge: LocalStageEdge,
        pipeline_core_coord: ttnn.CoreCoord,
        socket_fifo_size: int,
        local_endpoint_type,
    ):
        # Rank-scope every socket. A mesh-id-scoped handshake requires *all* host ranks of a
        # mesh to create the socket, but on a split (multi-host) mesh only the endpoint owner
        # does, so the handshake times out. Keying on the specific sender/receiver ranks works
        # uniformly for both CROSS_MESH (stage->stage) and SAME_MESH_CROSS_RANK (split intra)
        # edges. Both ends of an edge must agree, so the legacy SocketInterface paths
        # (_init_first_stage) rank-scope too.
        return _create_socket_resource(
            self.mesh_device,
            self._core_for_endpoint(edge.src, pipeline_core_coord),
            self._core_for_endpoint(edge.dst, pipeline_core_coord),
            socket_fifo_size,
            self._mesh_wrapper_for_endpoint(self.mesh_device, edge.src, self._stage_plan.my_rank),
            self._mesh_wrapper_for_endpoint(self.mesh_device, edge.dst, self._stage_plan.my_rank),
            use_rank_scoped_mesh_socket=True,
            local_endpoint_type=local_endpoint_type,
        )

    def _create_local_socket_pair_for_edge(
        self, edge: LocalStageEdge, pipeline_core_coord: ttnn.CoreCoord, socket_fifo_size: int
    ):
        socket_connection = ttnn.SocketConnection(
            self._core_for_endpoint(edge.src, pipeline_core_coord),
            self._core_for_endpoint(edge.dst, pipeline_core_coord),
        )
        socket_memory_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, socket_fifo_size)
        socket_config = ttnn.SocketConfig([socket_connection], socket_memory_config)
        return ttnn.create_socket_pair(self.mesh_device, self.mesh_device, socket_config)

    def _build_forwarder_from_existing_sockets(
        self,
        runtime_endpoint: StageEndpointRef,
        upstream_socket,
        downstream_socket,
        page_size: int,
        pipeline_core_coord: ttnn.CoreCoord,
    ):
        return SocketInterface.from_existing_sockets(
            page_size,
            page_size,
            mesh_device=self.mesh_device,
            runtime_core_coord=self._core_for_endpoint(runtime_endpoint, pipeline_core_coord),
            upstream_socket=upstream_socket,
            downstream_socket=downstream_socket,
        )

    def _init_from_plan(
        self,
        mesh_device,
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
        forward_metadata,
        entry_node_downstream,
        exit_node_upstream,
        host_io_placement,
    ):
        """Unified plan-driven socket builder for split and non-split transport stages.

        Builds the recv/send/intra forwarders and host I/O sockets a rank owns from its
        resolved :class:`LocalStageSocketPlan`. Works for both same-rank (LOCAL intra,
        shared socket pair) and cross-rank (SAME_MESH_CROSS_RANK intra, rank-scoped
        forwarder) stages. Compute taps are not yet modeled here (see :meth:`_uses_plan`).
        """
        if entry_node_downstream is not None or exit_node_upstream is not None:
            raise RuntimeError("Plan-driven PipelineBlock routing currently supports passthrough/host I/O stages only")
        assert self._stage_plan is not None
        stage_plan = self._stage_plan

        if stage_plan.host_io.owns_h2d:
            assert h2d_socket_fifo_size is not None, "H2D Socket FIFO Size must be provided to first pipeline stage"
            assert embedding_tensor is not None, "Embedding Tensor must be provided to first pipeline stage"
            # Place the H2D kernel on host_io_placement.h2d_core, not pipeline_core_coord:
            # when this rank also owns the forward D2D forwarder (same chip), two persistent
            # kernels would otherwise land on the same Tensix core and fail program merge.
            self.h2d_socket = ttnn.H2DSocket(
                mesh_device,
                self._core_for_endpoint(stage_plan.host_io.h2d_target, host_io_placement.h2d_core),
                ttnn.BufferType.L1,
                h2d_socket_fifo_size,
                ttnn.H2DMode.HOST_PUSH,
            )
            self._h2d_page_size_bytes = token_size_bytes

        if stage_plan.host_io.owns_d2h:
            assert d2h_socket_fifo_size is not None, "D2H Socket FIFO Size must be provided to host egress stage"
            # Same rationale as H2D: keep the D2H kernel off the upstream forwarder's core
            # (host loopback puts the D2H on the last stage's entry chip, shared with the
            # incoming forwarder) so the per-device program merge has distinct core ranges.
            self.d2h_socket = ttnn.D2HSocket(
                mesh_device,
                self._core_for_endpoint(stage_plan.host_io.d2h_source, host_io_placement.d2h_core),
                d2h_socket_fifo_size,
            )
            self._d2h_page_size_bytes = d2h_socket_page_size

        input_edge = local_input_edge(stage_plan)
        output_edge = local_output_edge(stage_plan)
        local_intra_socket_pair = None
        if (
            stage_plan.intra_stage_edge is not None
            and stage_plan.intra_stage_edge.transport == EdgeTransport.LOCAL
            and not stage_plan.host_io.owns_h2d
            and not stage_plan.host_io.owns_d2h
        ):
            local_intra_socket_pair = self._create_local_socket_pair_for_edge(
                stage_plan.intra_stage_edge, pipeline_core_coord, downstream_d2d_socket_fifo_size
            )

        h2d_downstream_core = None
        if stage_plan.host_io.owns_h2d:
            assert output_edge is not None, "Host ingress requires a local output edge"
            h2d_downstream_core = self._core_for_endpoint(output_edge.src, pipeline_core_coord)

        d2h_upstream_core = None
        if stage_plan.host_io.owns_d2h:
            assert input_edge is not None, "Host egress requires a local input edge"
            d2h_upstream_core = self._core_for_endpoint(input_edge.dst, pipeline_core_coord)

        if self.h2d_socket is not None or self.d2h_socket is not None:
            embedding_size_bytes = (
                embedding_tensor.shape[-1] * dtype_size(embedding_tensor.dtype) if embedding_tensor is not None else 0
            )
            metadata_size_bytes = downstream_d2d_socket_page_size - embedding_size_bytes if forward_metadata else 0
            self.host_io = HostInterface(
                self.h2d_socket,
                self.d2h_socket,
                token_size_bytes,
                d2h_socket_page_size,
                core_to_core_socket_buffer_size=downstream_d2d_socket_fifo_size,
                h2d_downstream_core=h2d_downstream_core,
                d2h_upstream_core=d2h_upstream_core,
                embedding_tensor=embedding_tensor,
                metadata_size_bytes=metadata_size_bytes,
            )

        forwarders = []

        def build_input_forwarder():
            # Receiver edge: incoming activation (or split cross-rank intra) -> D2H / local intra pair.
            if input_edge is not None and (stage_plan.host_io.owns_d2h or local_intra_socket_pair is not None):
                upstream_socket = self._create_socket_resource_for_edge(
                    input_edge,
                    pipeline_core_coord,
                    upstream_d2d_socket_fifo_size,
                    ttnn.SocketEndpoint.RECEIVER,
                )
                downstream_socket = (
                    self.host_io.get_upstream_socket() if stage_plan.host_io.owns_d2h else local_intra_socket_pair[0]
                )
                forwarders.append(
                    self._build_forwarder_from_existing_sockets(
                        input_edge.dst,
                        upstream_socket,
                        downstream_socket,
                        upstream_d2d_socket_page_size,
                        pipeline_core_coord,
                    )
                )

        def build_output_forwarder():
            # Sender edge: H2D / local intra / split-recv -> outgoing activation.
            if output_edge is not None:
                if stage_plan.host_io.owns_h2d:
                    upstream_socket = self.host_io.get_downstream_socket()
                elif local_intra_socket_pair is not None:
                    upstream_socket = local_intra_socket_pair[1]
                else:
                    assert input_edge is not None, "Expected an input edge for split output forwarding"
                    upstream_socket = self._create_socket_resource_for_edge(
                        input_edge,
                        pipeline_core_coord,
                        upstream_d2d_socket_fifo_size,
                        ttnn.SocketEndpoint.RECEIVER,
                    )
                downstream_socket = self._create_socket_resource_for_edge(
                    output_edge,
                    pipeline_core_coord,
                    downstream_d2d_socket_fifo_size,
                    ttnn.SocketEndpoint.SENDER,
                )
                forwarders.append(
                    self._build_forwarder_from_existing_sockets(
                        output_edge.src,
                        upstream_socket,
                        downstream_socket,
                        downstream_d2d_socket_page_size,
                        pipeline_core_coord,
                    )
                )

        # MeshSocket construction blocks on the cross-host handshake. Fabric loopback closes the
        # pipeline into a ring (last stage -> stage-0 loopback entry), so if every rank created its
        # receiver socket before its sender the whole ring deadlocks: each rank blocks on its
        # incoming handshake and never reaches its outgoing one. The pipeline-start stage is the
        # only rank that owns both a forward sender (H2D -> stage 1) and a loopback receiver
        # (last stage -> D2H); building its sender first -- mirroring legacy _init_first_stage --
        # breaks the cycle so the handshakes cascade around the ring. Every other stage keeps
        # receiver-first ordering (relied on by the linear/host-loopback and split cases).
        if stage_plan.host_io.owns_h2d and stage_plan.host_io.owns_d2h:
            build_output_forwarder()
            build_input_forwarder()
        else:
            build_input_forwarder()
            build_output_forwarder()

        self._plan_forwarders = forwarders
        self.entry_socket_interface = forwarders[0] if len(forwarders) == 1 else forwarders
        self.exit_socket_interface = None

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
        forward_metadata,
        host_io_placement=None,
    ):
        assert h2d_socket_fifo_size is not None, "H2D Socket FIFO Size must be provided to first pipeline stage"
        assert embedding_tensor is not None, "Embedding Tensor must be provided to first pipeline stage"
        assert host_io_placement is not None, "host_io_placement must be provided to first pipeline stage"

        h2d_device_coord = pipeline_config[self.my_stage_idx].entry_node_coord
        embedding_size_bytes = embedding_tensor.shape[-1] * dtype_size(embedding_tensor.dtype)

        if forward_metadata:
            assert downstream_d2d_socket_page_size == embedding_size_bytes + DeepseekMetadata.aligned_size_bytes()
        else:
            assert downstream_d2d_socket_page_size == embedding_size_bytes

        if self.initialize_loopback:
            assert d2h_socket_fifo_size is not None, "D2H Socket FIFO Size must be provided to first pipeline stage"
            assert d2h_socket_page_size is not None, "D2H Socket Page Size must be provided to first pipeline stage"
            assert d2h_socket_fifo_size >= d2h_socket_page_size

        assert h2d_socket_fifo_size >= token_size_bytes
        assert upstream_d2d_socket_page_size == d2h_socket_page_size

        self.h2d_socket = ttnn.H2DSocket(
            mesh_device,
            ttnn.MeshCoreCoord(h2d_device_coord, host_io_placement.h2d_core),
            ttnn.BufferType.L1,
            h2d_socket_fifo_size,
            ttnn.H2DMode.HOST_PUSH,
        )
        self._h2d_page_size_bytes = token_size_bytes

        if self.initialize_loopback:
            d2h_device_coord = pipeline_config[self.num_stages].exit_node_coord
            self.d2h_socket = ttnn.D2HSocket(
                mesh_device, ttnn.MeshCoreCoord(d2h_device_coord, host_io_placement.d2h_core), d2h_socket_fifo_size
            )
            self._d2h_page_size_bytes = d2h_socket_page_size

        self.host_io = HostInterface(
            self.h2d_socket,
            self.d2h_socket,
            token_size_bytes,
            d2h_socket_page_size,
            core_to_core_socket_buffer_size=downstream_d2d_socket_fifo_size,
            h2d_downstream_core=ttnn.MeshCoreCoord(
                pipeline_config[self.my_stage_idx].exit_node_coord, host_io_placement.fwd_d2d_core
            ),
            d2h_upstream_core=(
                ttnn.MeshCoreCoord(pipeline_config[self.num_stages].entry_node_coord, host_io_placement.lb_d2d_core)
                if self.initialize_loopback
                else None
            ),
            embedding_tensor=embedding_tensor,
            metadata_size_bytes=downstream_d2d_socket_page_size - embedding_size_bytes,
        )

        next_stage = self.my_stage_idx + 1
        ns = self._stages[next_stage]
        next_entry_owner = self._entry_owner(ns)
        self.exit_socket_interface = SocketInterface(
            downstream_d2d_socket_page_size,
            downstream_d2d_socket_fifo_size,
            downstream_d2d_socket_page_size,
            ttnn.MeshCoreCoord(pipeline_config[self.my_stage_idx].exit_node_coord, host_io_placement.fwd_d2d_core),
            ttnn.MeshCoreCoord(pipeline_config[next_stage].entry_node_coord, pipeline_core_coord),
            upstream_socket=self.host_io.get_downstream_socket(),
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(rank=next_entry_owner, mesh_id=self._mesh_id(ns)),
            # Rank-scope so this legacy stage-0 sender agrees with the (plan-driven)
            # receiver on the next stage. Mesh-id scoping is ambiguous once that stage's
            # mesh spans multiple host ranks. See _create_socket_resource_for_edge.
            use_rank_scoped_mesh_socket=True,
        )

        last_stage = self.num_stages - 1
        ls = self._stages[last_stage]
        last_exit_owner = self._exit_owner(ls)
        if self.initialize_loopback:
            self.entry_socket_interface = SocketInterface(
                upstream_d2d_socket_page_size,
                upstream_d2d_socket_fifo_size,
                upstream_d2d_socket_page_size,
                ttnn.MeshCoreCoord(pipeline_config[last_stage].exit_node_coord, pipeline_core_coord),
                ttnn.MeshCoreCoord(pipeline_config[self.num_stages].entry_node_coord, host_io_placement.lb_d2d_core),
                downstream_socket=self.host_io.get_upstream_socket(),
                sender_mesh=MeshWrapper(rank=last_exit_owner, mesh_id=self._mesh_id(ls)),
                receiver_mesh=MeshWrapper(mesh_device),
                # Rank-scope to agree with the plan-driven last-stage sender (the loopback
                # entry edge); mesh-id scoping is ambiguous for multi-host last stages.
                use_rank_scoped_mesh_socket=True,
            )

    def _init_combined_h2d_d2h_stage(
        self,
        mesh_device,
        pipeline_config,
        token_size_bytes,
        h2d_socket_fifo_size,
        d2h_socket_fifo_size,
        d2h_socket_page_size,
        h2d_to_compute_socket_buffer_size,
        embedding_tensor,
        forward_metadata,
        entry_node_downstream,
        exit_node_upstream,
        exit_upstream_page_size,
        host_io_placement,
    ):
        """Combined H2D + multi-upstream D2H stage for a single-process pipeline.

        The stage spans the mesh: H2D lives on the stage's entry node and D2H lives on its
        exit node. H2D delivers each token + metadata to ``entry_node_downstream`` (the
        compute input core, on the entry node); the compute produces N output shards on
        ``exit_node_upstream`` cores (on the exit node); the multi-upstream D2H sender
        (BRISC kernel) reads one page from each upstream and assembles them into a single
        D2H page sent back to host. No D2D socket interfaces are constructed — all I/O
        flows through ``self.host_io``.

        ``h2d_to_compute_socket_buffer_size`` sizes the local FIFO of the H2D→compute
        socket pair on the entry node.
        """
        assert h2d_socket_fifo_size is not None, "h2d_socket_fifo_size must be provided"
        assert d2h_socket_fifo_size is not None, "d2h_socket_fifo_size must be provided"
        assert d2h_socket_page_size is not None, "d2h_socket_page_size must be provided"
        assert d2h_socket_fifo_size >= d2h_socket_page_size
        assert h2d_socket_fifo_size >= token_size_bytes
        # embedding_tensor is optional: when None, HostInterface skips the fused embedding
        # kernel and uses h2d_receiver.cpp to forward the host-supplied page verbatim
        # (used by inject-hidden-states mode where the host pushes a full
        # `activation || metadata` page instead of just a token id).
        assert (
            entry_node_downstream is not None
        ), "entry_node_downstream is required for the combined H2D+D2H stage (the compute input core)"
        assert (
            isinstance(exit_node_upstream, list) and len(exit_node_upstream) >= 1
        ), "exit_node_upstream must be a non-empty list of MeshCoreCoord (compute output cores feeding D2H)"
        assert exit_upstream_page_size is not None, "exit_upstream_page_size is required for the combined H2D+D2H stage"

        metadata_size_bytes = DeepseekMetadata.aligned_size_bytes() if forward_metadata else 0
        assert d2h_socket_page_size == len(exit_node_upstream) * exit_upstream_page_size + metadata_size_bytes, (
            f"d2h_socket_page_size ({d2h_socket_page_size}) must equal len(exit_node_upstream) "
            f"({len(exit_node_upstream)}) * exit_upstream_page_size ({exit_upstream_page_size}) "
            f"+ metadata_size_bytes ({metadata_size_bytes})"
        )

        # H2D on the entry node, D2H on the exit node. HostInterface's same_device check
        # falls through to two separate per-device programs in this layout.
        h2d_device_coord = pipeline_config[self.my_stage_idx].entry_node_coord
        d2h_device_coord = pipeline_config[self.my_stage_idx].exit_node_coord

        self.h2d_socket = ttnn.H2DSocket(
            mesh_device,
            ttnn.MeshCoreCoord(h2d_device_coord, host_io_placement.h2d_core),
            ttnn.BufferType.L1,
            h2d_socket_fifo_size,
            ttnn.H2DMode.HOST_PUSH,
        )
        self._h2d_page_size_bytes = token_size_bytes

        self.d2h_socket = ttnn.D2HSocket(
            mesh_device,
            ttnn.MeshCoreCoord(d2h_device_coord, host_io_placement.d2h_core),
            d2h_socket_fifo_size,
        )
        self._d2h_page_size_bytes = d2h_socket_page_size

        self.host_io = HostInterface(
            self.h2d_socket,
            self.d2h_socket,
            token_size_bytes,
            d2h_socket_page_size,
            core_to_core_socket_buffer_size=h2d_to_compute_socket_buffer_size,
            h2d_downstream_core=entry_node_downstream,
            embedding_tensor=embedding_tensor,
            metadata_size_bytes=metadata_size_bytes,
            d2h_upstream_cores=exit_node_upstream,
            d2h_upstream_page_size=exit_upstream_page_size,
            d2h_socket_fifo_size=d2h_socket_fifo_size,
            d2h_forward_metadata_size_bytes=metadata_size_bytes,
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
        host_io_placement,
        entry_node_downstream=None,
    ):
        assert d2h_socket_fifo_size is not None, "D2H Socket FIFO Size must be provided to last pipeline stage"
        assert d2h_socket_page_size is not None, "D2H Socket Page Size must be provided to last pipeline stage"
        assert d2h_socket_fifo_size >= d2h_socket_page_size
        # For no_loopback the C++ placeholder sets exit_node_coord = entry_node_coord, so
        # d2h_device_coord ends up on the same chip as the entry recv kernel.  Use
        # host_io_placement.d2h_core (rather than pipeline_core_coord) so the D2H kernel
        # lands on a different core and avoids a same-core dispatch deadlock.
        d2h_device_coord = pipeline_config[self.num_stages - 1].entry_node_coord
        d2h_core = host_io_placement.d2h_core
        d2h_upstream_core = (
            exit_node_upstream
            if exit_node_upstream
            else ttnn.MeshCoreCoord(pipeline_config[self.my_stage_idx].entry_node_coord, pipeline_core_coord)
        )

        self.d2h_socket = ttnn.D2HSocket(
            mesh_device, ttnn.MeshCoreCoord(d2h_device_coord, d2h_core), d2h_socket_fifo_size
        )
        self._d2h_page_size_bytes = d2h_socket_page_size

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

        prev_stage = self.my_stage_idx - 1
        ps = self._stages[prev_stage]
        prev_exit_owner = self._exit_owner(ps)
        if entry_node_downstream is not None:
            # Compute stage (e.g. LMHead): entry socket delivers to the compute input core;
            # D2H reads independently from the compute output core (exit_node_upstream).
            self.entry_socket_interface = SocketInterface(
                upstream_d2d_socket_page_size,
                upstream_d2d_socket_fifo_size,
                upstream_d2d_socket_page_size,
                ttnn.MeshCoreCoord(pipeline_config[prev_stage].exit_node_coord, pipeline_core_coord),
                ttnn.MeshCoreCoord(pipeline_config[self.my_stage_idx].entry_node_coord, pipeline_core_coord),
                downstream_core_coord=entry_node_downstream,
                sender_mesh=MeshWrapper(rank=prev_exit_owner, mesh_id=self._mesh_id(ps)),
                receiver_mesh=MeshWrapper(mesh_device),
            )
        else:
            # Passthrough stage: entry socket relays directly into the D2H socket.
            self.entry_socket_interface = SocketInterface(
                upstream_d2d_socket_page_size,
                upstream_d2d_socket_fifo_size,
                upstream_d2d_socket_page_size,
                ttnn.MeshCoreCoord(pipeline_config[prev_stage].exit_node_coord, pipeline_core_coord),
                ttnn.MeshCoreCoord(pipeline_config[self.my_stage_idx].entry_node_coord, pipeline_core_coord),
                downstream_socket=self.host_io.get_upstream_socket(),
                sender_mesh=MeshWrapper(rank=prev_exit_owner, mesh_id=self._mesh_id(ps)),
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
        prev_stage = self.my_stage_idx - 1
        ps = self._stages[prev_stage]
        prev_exit_owner = self._exit_owner(ps)
        self.entry_socket_interface = SocketInterface(
            upstream_d2d_socket_page_size,
            upstream_d2d_socket_fifo_size,
            upstream_d2d_socket_page_size,
            ttnn.MeshCoreCoord(pipeline_config[prev_stage].exit_node_coord, pipeline_core_coord),
            ttnn.MeshCoreCoord(pipeline_config[self.my_stage_idx].entry_node_coord, pipeline_core_coord),
            downstream_core_coord=(
                entry_node_downstream
                if entry_node_downstream
                else ttnn.MeshCoreCoord(pipeline_config[self.my_stage_idx].exit_node_coord, pipeline_core_coord)
            ),
            sender_mesh=MeshWrapper(rank=prev_exit_owner, mesh_id=self._mesh_id(ps)),
            receiver_mesh=MeshWrapper(mesh_device),
        )

        next_stage = self.my_stage_idx + 1 if not self.is_last_stage else 0
        ns = self._stages[next_stage]
        next_entry_owner = self._entry_owner(ns)
        # pipeline_config index: always my_stage_idx+1 (sequential), even for
        # the last stage where it points to the loopback config entry at
        # pipeline_config[num_stages] rather than wrapping to 0.
        next_cfg_idx = self.my_stage_idx + 1
        use_multi_upstream = isinstance(exit_node_upstream, list)
        if use_multi_upstream:
            assert exit_upstream_page_size is not None, "exit_upstream_page_size required for multi-upstream mode"
            self.exit_socket_interface = SocketInterface(
                downstream_d2d_socket_page_size,
                downstream_d2d_socket_fifo_size,
                downstream_d2d_socket_page_size,
                ttnn.MeshCoreCoord(pipeline_config[self.my_stage_idx].exit_node_coord, pipeline_core_coord),
                ttnn.MeshCoreCoord(pipeline_config[next_cfg_idx].entry_node_coord, pipeline_core_coord),
                sender_mesh=MeshWrapper(mesh_device),
                receiver_mesh=MeshWrapper(rank=next_entry_owner, mesh_id=self._mesh_id(ns)),
                upstream_core_coords=exit_node_upstream,
                upstream_page_size=exit_upstream_page_size,
                forward_metadata_size_bytes=forward_metadata_size_bytes,
            )
        else:
            self.exit_socket_interface = SocketInterface(
                downstream_d2d_socket_page_size,
                downstream_d2d_socket_fifo_size,
                downstream_d2d_socket_page_size,
                ttnn.MeshCoreCoord(pipeline_config[self.my_stage_idx].exit_node_coord, pipeline_core_coord),
                ttnn.MeshCoreCoord(pipeline_config[next_cfg_idx].entry_node_coord, pipeline_core_coord),
                upstream_core_coord=exit_node_upstream,
                upstream_socket=self.entry_socket_interface.get_downstream_socket() if not exit_node_upstream else None,
                sender_mesh=MeshWrapper(mesh_device),
                receiver_mesh=MeshWrapper(rank=next_entry_owner, mesh_id=self._mesh_id(ns)),
            )

    def _init_parallel_device_forwarding_stage(
        self,
        mesh_device,
        pipeline_device_coords,
        upstream_d2d_socket_fifo_size,
        downstream_d2d_socket_fifo_size,
        upstream_d2d_socket_page_size,
        downstream_d2d_socket_page_size,
        core_entry,
        core_exit,
        entry_downstream_core=None,
        exit_upstream_cores=None,
        exit_upstream_page_size=None,
    ):
        """Per-device parallel forwarding stage.

        Each channel maps to a different device in the mesh. Within each device,
        core_entry (pipeline_core_coord) and core_exit (pipeline_exit_core_coord)
        handle incoming and outgoing data respectively. Both are caller-supplied,
        mirroring the pattern used by _init_forwarding_stage where device coords
        come from the pipeline config and the core coord is caller-supplied.

        Optional entry_downstream_core specifies where entry forwards data for compute.

        exit_upstream_cores controls the exit kernel variant, matching the
        isinstance-based pattern in _init_forwarding_stage:
          - None: single-upstream passthrough (d2d_exchange.cpp)
          - [] (empty list): multi-upstream passthrough — uses
            d2d_exchange_multiple_upstreams.cpp with the entry's downstream
            socket as a single-element upstream list
          - [CoreCoord, ...] (non-empty list): multi-upstream with separate
            socket pairs to the specified cores (for compute integration)

        Sets entry_socket_interface / exit_socket_interface to list[SocketInterface].
        """
        use_multi_upstream = isinstance(exit_upstream_cores, list)
        prev_stage = self.my_stage_idx - 1
        ps = self._stages[prev_stage]
        prev_exit_owner = self._exit_owner(ps)
        next_stage = self.my_stage_idx + 1 if not self.is_last_stage else 0
        ns = self._stages[next_stage]
        next_entry_owner = self._entry_owner(ns)

        self.entry_socket_interface = []
        self.exit_socket_interface = []

        # Two-pass creation: all entries first, then all exits.
        # MeshSocket creation is a blocking pairwise handshake between
        # sender and receiver processes. Process 0 creates all exit sockets
        # (via ParallelSocketInterface) before all entry sockets.  Forwarding
        # stages must match that order — entries first (matching the previous
        # stage's exits), then exits (matching the next stage's entries) — to
        # avoid cascading delays that accumulate into timeouts with many channels.

        effective_downstream_core = entry_downstream_core if entry_downstream_core else core_exit

        for dc in pipeline_device_coords:
            entry_si = SocketInterface(
                upstream_d2d_socket_page_size,
                upstream_d2d_socket_fifo_size,
                upstream_d2d_socket_page_size,
                ttnn.MeshCoreCoord(dc, core_exit),
                ttnn.MeshCoreCoord(dc, core_entry),
                downstream_core_coord=ttnn.MeshCoreCoord(dc, effective_downstream_core),
                sender_mesh=MeshWrapper(rank=prev_exit_owner, mesh_id=self._mesh_id(ps)),
                receiver_mesh=MeshWrapper(mesh_device),
            )
            self.entry_socket_interface.append(entry_si)

        for i, dc in enumerate(pipeline_device_coords):
            entry_si = self.entry_socket_interface[i]

            if use_multi_upstream and len(exit_upstream_cores) > 0:
                per_device_upstream_cores = [ttnn.MeshCoreCoord(dc, uc) for uc in exit_upstream_cores]
                exit_si = SocketInterface(
                    downstream_d2d_socket_page_size,
                    downstream_d2d_socket_fifo_size,
                    downstream_d2d_socket_page_size,
                    ttnn.MeshCoreCoord(dc, core_exit),
                    ttnn.MeshCoreCoord(dc, core_entry),
                    sender_mesh=MeshWrapper(mesh_device),
                    receiver_mesh=MeshWrapper(rank=next_entry_owner, mesh_id=self._mesh_id(ns)),
                    upstream_core_coords=per_device_upstream_cores,
                    upstream_page_size=exit_upstream_page_size,
                )
            elif use_multi_upstream:
                exit_si = SocketInterface(
                    downstream_d2d_socket_page_size,
                    downstream_d2d_socket_fifo_size,
                    downstream_d2d_socket_page_size,
                    ttnn.MeshCoreCoord(dc, core_exit),
                    ttnn.MeshCoreCoord(dc, core_entry),
                    sender_mesh=MeshWrapper(mesh_device),
                    receiver_mesh=MeshWrapper(rank=next_entry_owner, mesh_id=self._mesh_id(ns)),
                    upstream_sockets=[entry_si.get_downstream_socket()],
                    upstream_page_size=upstream_d2d_socket_page_size,
                )
            else:
                exit_si = SocketInterface(
                    downstream_d2d_socket_page_size,
                    downstream_d2d_socket_fifo_size,
                    downstream_d2d_socket_page_size,
                    ttnn.MeshCoreCoord(dc, core_exit),
                    ttnn.MeshCoreCoord(dc, core_entry),
                    upstream_socket=entry_si.get_downstream_socket(),
                    sender_mesh=MeshWrapper(mesh_device),
                    receiver_mesh=MeshWrapper(rank=next_entry_owner, mesh_id=self._mesh_id(ns)),
                )
            self.exit_socket_interface.append(exit_si)

    def _dispatch_parallel_device_programs(self):
        """Collect programs from all per-device socket interfaces and dispatch in a single generic_op."""
        all_entries = []
        for si in self.entry_socket_interface:
            all_entries.extend(si.build_programs())
        for si in self.exit_socket_interface:
            all_entries.extend(si.build_programs())

        dummy_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device
        )
        groups = _group_by_device(all_entries)
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        for device_coord, progs in groups:
            merged = ttnn.merge_program_descriptors(progs) if len(progs) > 1 else progs[0]
            mesh_program_descriptor[ttnn.MeshCoordinateRange(device_coord, device_coord)] = merged
        return ttnn.generic_op([dummy_tensor, dummy_tensor], mesh_program_descriptor)

    def _is_combined_h2d_d2h_stage(self):
        return (
            self.is_pipeline_start and self.is_last_stage and not self.initialize_loopback and not self.parallel_devices
        )

    def _run_from_plan(self):
        all_entries = []
        if self.host_io is not None:
            all_entries.extend(self.host_io._build_programs())
        for forwarder in self._plan_forwarders:
            all_entries.extend(forwarder.build_programs())

        if not all_entries:
            return

        dummy_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device
        )
        groups = _group_by_device(all_entries)
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()
        for device_coord, progs in groups:
            merged = ttnn.merge_program_descriptors(progs) if len(progs) > 1 else progs[0]
            mesh_program_descriptor[ttnn.MeshCoordinateRange(device_coord, device_coord)] = merged

        ttnn.generic_op([dummy_tensor, dummy_tensor], mesh_program_descriptor)

    def _terminate_from_plan(self):
        for forwarder in self._plan_forwarders:
            forwarder.terminate(False)
        if self.host_io is not None:
            # Plan-path host I/O only exists on split (multi-host) stages, where the idle peer
            # rank owns no host_io and so does NOT call it. synchronize_device is collective
            # across the mesh's host ranks, so terminating host_io with sync=True here issues an
            # extra collective sync the peer never matches -> the two split halves deadlock. Pass
            # sync=False; the outer Pipeline.terminate synchronize_device (called by every rank)
            # drains the device symmetrically and still waits for the D2H/forwarder kernels.
            self.host_io.terminate(False)

    def run(self):
        if self.parallel_devices:
            self._dispatch_parallel_device_programs()
        elif self._is_combined_h2d_d2h_stage():
            self.host_io.run()
        elif self._uses_plan():
            self._run_from_plan()
        elif self.is_pipeline_start:
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
        ttnn.distributed_context_barrier()
        if self.parallel_devices:
            for si in self.entry_socket_interface:
                si.terminate(False)
            for i, si in enumerate(self.exit_socket_interface):
                last = i == len(self.exit_socket_interface) - 1
                si.terminate(last)
        elif self._is_combined_h2d_d2h_stage():
            self.host_io.terminate(True)
        elif self._uses_plan():
            self._terminate_from_plan()
        elif self.is_pipeline_start:
            self.host_io.terminate(False)
            if self.initialize_loopback:
                self.entry_socket_interface.terminate(False)
            self.exit_socket_interface.terminate(True)
        else:
            self.entry_socket_interface.terminate(False)
            if self.has_exit:
                sync = not self.has_d2h
                self.exit_socket_interface.terminate(sync)
            if self.host_io is not None:
                self.host_io.terminate(True)

    def is_first_pipeline_stage(self):
        return self.is_pipeline_start

    def write_token(self, token_tensor):
        assert self.is_first_pipeline_stage(), "Token can only be written to the first pipeline stage"
        self.h2d_socket.write_tensor(token_tensor)

    def read_output(self, output_tensor):
        if self._loopback_mode == "host":
            return self._read_output_host_loopback(output_tensor)
        else:
            assert (
                self.d2h_socket is not None
            ), "read_output requires a D2H socket: valid on stage 0 with loopback, or last stage without loopback"
            self.d2h_socket.read_tensor(output_tensor)

    def _read_output_host_loopback(self, output_tensor):
        # Lazy import: only available after tt-metal is rebuilt with send_bytes/recv_bytes bindings.
        import torch
        from ttnn._ttnn.multi_device import recv_bytes, send_bytes

        if self.is_pipeline_start:
            # Rank 0 receives the result from the last rank via host MPI.
            # ttnn.to_torch returns a copy, so we can't write back through it.
            # Return the received torch tensor directly instead.
            backing = ttnn.to_torch(output_tensor)
            last_stage = self._stages[self.num_stages - 1]
            raw = recv_bytes(backing.numel() * backing.element_size(), self._exit_owner(last_stage))
            received = torch.frombuffer(bytearray(raw), dtype=backing.dtype).reshape(backing.shape)
            return received
        elif self.is_last_stage and self.d2h_socket is not None:
            # The last stage's D2H owner reads from the D2H socket then forwards to rank 0 via
            # host MPI. On a split last stage only the entry/egress owner holds the D2H (rank 0
            # receives from _exit_owner(last_stage), which falls back to the entry owner); the
            # peer half of the split has no D2H and must not read or send.
            self.d2h_socket.read_tensor(output_tensor)
            result = ttnn.to_torch(output_tensor).reshape(-1).contiguous()
            send_bytes(result.view(torch.uint8).numpy().tobytes(), 0)

    def push_dummy_token(self):
        """Push a single zeroed token through the H2D socket.

        Used during pipeline teardown to wake every stage's compute loop one last
        iteration so each persistent kernel returns to its top-of-loop termination
        check and breaks cleanly.
        """
        assert self.is_first_pipeline_stage(), "push_dummy_token() requires the first pipeline stage"
        assert self.h2d_socket is not None and self._h2d_page_size_bytes is not None
        page_words = self._h2d_page_size_bytes // 4
        dummy = ttnn.from_torch(
            torch.zeros(1, page_words, dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.h2d_socket.write_tensor(dummy)

    def drain_dummy_output(self):
        """Drain one page from the D2H socket — pairs with :meth:`push_dummy_token` during teardown."""
        if self.d2h_socket is None or self._d2h_page_size_bytes is None:
            return
        page_words = self._d2h_page_size_bytes // 4
        sink = ttnn.from_torch(
            torch.zeros(1, page_words, dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.d2h_socket.read_tensor(sink)

    def get_upstream_socket(self):
        """Return a single upstream socket (non-parallel mode only)."""
        assert not self.parallel_devices, "Use get_upstream_sockets() for parallel mode"
        if self.exit_socket_interface is not None:
            return self.exit_socket_interface.get_upstream_socket()
        elif hasattr(self, "host_io"):
            return self.host_io.get_upstream_socket()

    def get_downstream_socket(self):
        """Return a single downstream socket (non-parallel mode only)."""
        assert not self.parallel_devices, "Use get_downstream_sockets() for parallel mode"
        if self.entry_socket_interface is None:
            # Combined H2D+D2H stage: compute reads from the H2D's downstream socket directly.
            return self.host_io.get_downstream_socket()
        return self.entry_socket_interface.get_downstream_socket()

    def get_upstream_sockets(self):
        """Return list of upstream sockets (per-device parallel or multi-upstream modes)."""
        if self.parallel_devices:
            return [si.get_upstream_sockets() for si in self.exit_socket_interface]
        if self.exit_socket_interface is None:
            # Combined H2D+D2H stage: compute writes into the multi-upstream D2H feeders.
            return self.host_io.get_upstream_sockets()
        return self.exit_socket_interface.get_upstream_sockets()

    def get_downstream_sockets(self):
        """Return list of downstream sockets (parallel mode)."""
        assert self.parallel_devices, "get_downstream_sockets() requires per-device parallel mode"
        return [si.get_downstream_socket() for si in self.entry_socket_interface]

    def export_host_socket_descriptors(self, io_socket_descriptor_prefix: str) -> None:
        assert self.is_first_pipeline_stage(), "Host socket descriptors can only be exported from the first stage"
        assert self.h2d_socket is not None, "Expected H2D socket on the first pipeline stage"
        assert self.d2h_socket is not None, "Expected D2H socket on the first pipeline stage"

        self.h2d_socket.export_descriptor(f"{io_socket_descriptor_prefix}_h2d")
        self.d2h_socket.export_descriptor(f"{io_socket_descriptor_prefix}_d2h")
