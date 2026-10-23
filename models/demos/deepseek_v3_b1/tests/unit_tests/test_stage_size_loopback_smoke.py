# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Generic stage-size loopback smoke built from the resolved Blitz allocation.

This file intentionally avoids the DeepSeek demo pipeline/model stack. Instead it derives a per-rank
socket plan from `resolve_blitz_decode_pipeline_allocation()` and wires the minimal HostInterface /
SocketInterface chain needed for a simple payload loopback smoke.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.pipeline import create_passthrough_pipeline_configuration
from models.demos.deepseek_v3_b1.demo.pipeline_routing import (
    EdgeTransport,
    LocalRole,
    LocalStageSocketPlan,
    build_local_stage_socket_plan,
    build_local_stage_socket_plans,
    build_stage_routing,
)
from models.demos.deepseek_v3_b1.demo.stage import ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES
from models.demos.deepseek_v3_b1.demo.stage_family import (
    StageFamily,
    fabric_config_for_stage_family,
    query_global_stage_mesh_shape,
    stage_family_from_shape,
)
from models.demos.deepseek_v3_b1.demo.weight_provider import SyntheticWeightProvider
from models.demos.deepseek_v3_b1.metadata.metadata import DeepseekMetadata
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import (
    MeshWrapper,
    SocketInterface,
    _create_socket_resource,
    _group_by_device,
)
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size, ttnn_dtype_from_torch_dtype
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineConfigEntry
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions


def create_fabric_router_config(max_payload_size: int) -> Any:
    """Create a FabricRouterConfig with the requested max payload size."""

    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@dataclass(frozen=True)
class PhysicalTopologyConfig:
    """Static test inputs needed to open the mesh and configure fabric."""

    name: str
    stage_family: StageFamily
    mesh_device_param: tuple[int, int]
    fabric_config: ttnn.FabricConfig
    initialize_loopback: bool = True
    fabric_router_max_payload_size: int | None = None

    def make_device_params(self) -> dict[str, Any]:
        device_params: dict[str, Any] = {"fabric_config": self.fabric_config}
        if self.fabric_router_max_payload_size is not None:
            device_params["fabric_router_config"] = create_fabric_router_config(self.fabric_router_max_payload_size)
        return device_params


def get_generic_stage_size_loopback_topology_config() -> PhysicalTopologyConfig:
    """Return the loopback smoke topology config derived from the selected MGD."""

    mesh_shape = query_global_stage_mesh_shape()
    stage_family = stage_family_from_shape(mesh_shape)
    return PhysicalTopologyConfig(
        name=f"generic-{stage_family.value}-loopback",
        stage_family=stage_family,
        mesh_device_param=(int(mesh_shape[0]), int(mesh_shape[1])),
        fabric_config=fabric_config_for_stage_family(stage_family),
        initialize_loopback=True,
        fabric_router_max_payload_size=15232,
    )


# These tests are a multi-rank pipeline (one MPI rank per stage) launched via tt-run on galaxy
# stage-size hardware. During collection, the distributed context may not be initialized yet, so
# collection-time gating must use the launcher environment rather than context size.
if not ttnn.using_distributed_env():
    pytest.skip(
        "stage-size loopback smoke requires a tt-run distributed launch on galaxy stage-size HW; "
        "plain pytest / single-box execution detected",
        allow_module_level=True,
    )
try:
    GENERIC_STAGE_SIZE_LOOPBACK_CONFIG = get_generic_stage_size_loopback_topology_config()
except ValueError as exc:
    pytest.skip(
        f"stage-size loopback smoke requires a 4x2/4x4/8x4 stage mesh: {exc}",
        allow_module_level=True,
    )
PIPELINE_ENDPOINT_CORE_COORD = ttnn.CoreCoord(0, 0)


def _require_multi_rank_distributed_context() -> int:
    """Return the distributed rank count once the runtime context is available."""

    if not ttnn.distributed_context_is_initialized():
        pytest.skip(
            "stage-size loopback smoke requires an initialized distributed context; "
            "launch under tt-run so rank setup completes before test execution"
        )

    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs <= 1:
        pytest.skip(f"stage-size loopback smoke requires at least 2 distributed ranks, got {num_procs}")

    return num_procs


def _core_for_mesh_view_endpoint(endpoint, pipeline_core_coord):
    # All runtime socket and program-placement paths operate in the current
    # mesh_device view, using canonical stage-mesh coordinates.
    return ttnn.MeshCoreCoord(endpoint.placement.mesh_coord, pipeline_core_coord)


def _mesh_wrapper_for_endpoint(mesh_device, endpoint, my_rank):
    if endpoint.owner_rank == my_rank:
        return MeshWrapper(mesh_device)
    return MeshWrapper(rank=endpoint.owner_rank, mesh_id=endpoint.mesh_id)


def _local_input_edge(stage_plan):
    if stage_plan.incoming_edge is not None:
        return stage_plan.incoming_edge
    if (
        stage_plan.intra_stage_edge is not None
        and stage_plan.intra_stage_edge.transport != EdgeTransport.LOCAL
        and stage_plan.intra_stage_edge.local_role == LocalRole.RECEIVER
    ):
        return stage_plan.intra_stage_edge
    return None


def _local_output_edge(stage_plan):
    if stage_plan.outgoing_edge is not None:
        return stage_plan.outgoing_edge
    if (
        stage_plan.intra_stage_edge is not None
        and stage_plan.intra_stage_edge.transport != EdgeTransport.LOCAL
        and stage_plan.intra_stage_edge.local_role == LocalRole.SENDER
    ):
        return stage_plan.intra_stage_edge
    return None


def _build_host_io(mesh_device, stage_plan, pipeline_core_coord, tensor_size_bytes, fifo_size, h2d_mode):
    if not (stage_plan.host_io.owns_h2d or stage_plan.host_io.owns_d2h):
        return None

    h2d_socket = None
    if stage_plan.host_io.owns_h2d:
        h2d_socket = ttnn.H2DSocket(
            mesh_device,
            _core_for_mesh_view_endpoint(stage_plan.host_io.h2d_target, pipeline_core_coord),
            ttnn.BufferType.L1,
            fifo_size,
            h2d_mode,
        )

    d2h_socket = None
    if stage_plan.host_io.owns_d2h:
        d2h_socket = ttnn.D2HSocket(
            mesh_device,
            _core_for_mesh_view_endpoint(stage_plan.host_io.d2h_source, pipeline_core_coord),
            fifo_size,
        )

    h2d_downstream_core = None
    if stage_plan.host_io.owns_h2d:
        output_edge = _local_output_edge(stage_plan)
        assert output_edge is not None, "Host ingress requires a local stage output edge"
        h2d_downstream_core = _core_for_mesh_view_endpoint(output_edge.src, pipeline_core_coord)

    d2h_upstream_core = None
    if stage_plan.host_io.owns_d2h:
        input_edge = _local_input_edge(stage_plan)
        assert input_edge is not None, "Host egress requires a local stage input edge"
        d2h_upstream_core = _core_for_mesh_view_endpoint(input_edge.dst, pipeline_core_coord)

    return HostInterface(
        h2d_socket,
        d2h_socket,
        tensor_size_bytes,
        tensor_size_bytes,
        core_to_core_socket_buffer_size=fifo_size,
        h2d_downstream_core=h2d_downstream_core,
        d2h_upstream_core=d2h_upstream_core,
    )


def _create_socket_resource_for_edge(
    mesh_device,
    edge,
    pipeline_core_coord,
    fifo_size,
    my_rank,
    local_endpoint_type=None,
):
    assert edge is not None, "Socket resource creation requires an edge"
    if local_endpoint_type is None:
        assert edge.local_role in (
            LocalRole.SENDER,
            LocalRole.RECEIVER,
        ), f"Explicit local_endpoint_type required for local role {edge.local_role}"
        local_endpoint_type = (
            ttnn.SocketEndpoint.SENDER if edge.local_role == LocalRole.SENDER else ttnn.SocketEndpoint.RECEIVER
        )

    return _create_socket_resource(
        mesh_device,
        _core_for_mesh_view_endpoint(edge.src, pipeline_core_coord),
        _core_for_mesh_view_endpoint(edge.dst, pipeline_core_coord),
        fifo_size,
        _mesh_wrapper_for_endpoint(mesh_device, edge.src, my_rank),
        _mesh_wrapper_for_endpoint(mesh_device, edge.dst, my_rank),
        use_rank_scoped_mesh_socket=True,
        local_endpoint_type=local_endpoint_type,
    )


def _create_local_socket_pair_for_edge(mesh_device, edge, pipeline_core_coord, fifo_size):
    assert edge is not None and edge.transport == EdgeTransport.LOCAL, "Expected a local intra-stage edge"

    socket_connection = ttnn.SocketConnection(
        _core_for_mesh_view_endpoint(edge.src, pipeline_core_coord),
        _core_for_mesh_view_endpoint(edge.dst, pipeline_core_coord),
    )
    socket_memory_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, fifo_size)
    socket_config = ttnn.SocketConfig([socket_connection], socket_memory_config)
    return ttnn.create_socket_pair(mesh_device, mesh_device, socket_config)


def _build_forwarder(
    mesh_device, runtime_endpoint, upstream_socket, downstream_socket, tensor_size_bytes, pipeline_core_coord
):
    return SocketInterface.from_existing_sockets(
        tensor_size_bytes,
        tensor_size_bytes,
        mesh_device=mesh_device,
        runtime_core_coord=_core_for_mesh_view_endpoint(runtime_endpoint, pipeline_core_coord),
        upstream_socket=upstream_socket,
        downstream_socket=downstream_socket,
    )


def _build_local_stage_runtime(mesh_device, stage_plan: LocalStageSocketPlan, tensor_size_bytes, fifo_size, h2d_mode):
    pipeline_core_coord = PIPELINE_ENDPOINT_CORE_COORD
    host_io = _build_host_io(
        mesh_device,
        stage_plan,
        pipeline_core_coord,
        tensor_size_bytes,
        fifo_size,
        h2d_mode,
    )

    input_edge = _local_input_edge(stage_plan)
    output_edge = _local_output_edge(stage_plan)

    local_intra_socket_pair = None
    if (
        stage_plan.intra_stage_edge is not None
        and stage_plan.intra_stage_edge.transport == EdgeTransport.LOCAL
        and not stage_plan.host_io.owns_h2d
        and not stage_plan.host_io.owns_d2h
    ):
        local_intra_socket_pair = _create_local_socket_pair_for_edge(
            mesh_device, stage_plan.intra_stage_edge, pipeline_core_coord, fifo_size
        )

    forwarders = []

    def _append_input_forwarder():
        if input_edge is None or not (stage_plan.host_io.owns_d2h or local_intra_socket_pair is not None):
            return

        upstream_socket = _create_socket_resource_for_edge(
            mesh_device,
            input_edge,
            pipeline_core_coord,
            fifo_size,
            stage_plan.my_rank,
            local_endpoint_type=ttnn.SocketEndpoint.RECEIVER,
        )
        downstream_socket = host_io.get_upstream_socket() if stage_plan.host_io.owns_d2h else local_intra_socket_pair[0]
        forwarders.append(
            _build_forwarder(
                mesh_device,
                input_edge.dst,
                upstream_socket,
                downstream_socket,
                tensor_size_bytes,
                pipeline_core_coord,
            )
        )

    def _append_output_forwarder():
        if output_edge is None:
            return

        if stage_plan.host_io.owns_h2d:
            assert host_io is not None, "Expected host I/O for host ingress path"
            upstream_socket = host_io.get_downstream_socket()
        elif local_intra_socket_pair is not None:
            upstream_socket = local_intra_socket_pair[1]
        else:
            assert input_edge is not None, "Expected an input edge for output-side forwarder"
            assert _core_for_mesh_view_endpoint(input_edge.dst, pipeline_core_coord) == _core_for_mesh_view_endpoint(
                output_edge.src, pipeline_core_coord
            ), "Expected direct one-core path when no local handoff or host ingress exists"
            upstream_socket = _create_socket_resource_for_edge(
                mesh_device,
                input_edge,
                pipeline_core_coord,
                fifo_size,
                stage_plan.my_rank,
                local_endpoint_type=ttnn.SocketEndpoint.RECEIVER,
            )

        downstream_socket = _create_socket_resource_for_edge(
            mesh_device,
            output_edge,
            pipeline_core_coord,
            fifo_size,
            stage_plan.my_rank,
            local_endpoint_type=ttnn.SocketEndpoint.SENDER,
        )
        forwarders.append(
            _build_forwarder(
                mesh_device,
                output_edge.src,
                upstream_socket,
                downstream_socket,
                tensor_size_bytes,
                pipeline_core_coord,
            )
        )

    # The host-facing loopback stage participates in both the stage-0 forward
    # hop and the return hop. Seed the ring with the outgoing sender first so
    # rank 0 does not block waiting on the return edge before any peer can
    # create its matching sender.
    build_output_first = stage_plan.host_io.owns_h2d and input_edge is not None and output_edge is not None
    if build_output_first:
        _append_output_forwarder()

    _append_input_forwarder()

    if not build_output_first:
        _append_output_forwarder()

    return host_io, forwarders


def _dispatch_merged_programs(mesh_device, host_io, forwarders):
    """Dispatch all local persistent kernels in one generic_op.

    Sequential generic_op calls can deadlock once a rank owns multiple local
    service kernels, because the first persistent launch can occupy the device
    queue before its local peers are submitted.
    """

    all_entries = []
    if host_io is not None:
        all_entries.extend(host_io._build_programs())
    for forwarder in forwarders:
        all_entries.extend(forwarder.build_programs())

    assert all_entries, "Expected at least one local program to dispatch"

    dummy_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device
    )
    mesh_program_descriptor = ttnn.MeshProgramDescriptor()
    grouped_entries = _group_by_device(all_entries)
    for device_coord, progs in grouped_entries:
        merged = ttnn.merge_program_descriptors(progs) if len(progs) > 1 else progs[0]
        mesh_program_descriptor[ttnn.MeshCoordinateRange(device_coord, device_coord)] = merged
    return ttnn.generic_op([dummy_tensor, dummy_tensor], mesh_program_descriptor)


def _terminate_components(mesh_device, host_io, forwarders):
    for forwarder in forwarders:
        forwarder.terminate(False)
    if host_io is not None:
        host_io.terminate(False)
    ttnn.synchronize_device(mesh_device)


@pytest.mark.skip(
    reason="This test requires first and last stage to be physically connected, enable only when applicable"
)
@pytest.mark.parametrize(
    "tensor_size_bytes,fifo_size,num_iterations",
    [
        (64, 128, 64),
        (32768, 65536, 64),
    ],
)
@pytest.mark.parametrize("h2d_mode", [ttnn.H2DMode.HOST_PUSH, ttnn.H2DMode.DEVICE_PULL])
@pytest.mark.parametrize(
    ("topology_config", "mesh_device", "device_params"),
    [
        (
            GENERIC_STAGE_SIZE_LOOPBACK_CONFIG,
            GENERIC_STAGE_SIZE_LOOPBACK_CONFIG.mesh_device_param,
            GENERIC_STAGE_SIZE_LOOPBACK_CONFIG.make_device_params(),
        )
    ],
    ids=[GENERIC_STAGE_SIZE_LOOPBACK_CONFIG.name],
    indirect=["mesh_device", "device_params"],
)
def test_generic_stage_size_loopback_smoke(
    topology_config,
    mesh_device,
    device_params,
    tensor_size_bytes,
    fifo_size,
    num_iterations,
    h2d_mode,
):
    """Simple loopback smoke that derives its local wiring from the resolved stage socket plan."""

    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    _require_multi_rank_distributed_context()
    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    allocation = ttnn._ttnn.multi_device.experimental.resolve_blitz_decode_pipeline_allocation(
        topology_config.initialize_loopback
    )
    my_rank = int(ttnn.distributed_context_get_rank())

    local_stage_plans = []
    for stage_idx in range(len(allocation.stages)):
        stage_plan = build_local_stage_socket_plan(allocation, stage_idx, my_rank)
        if stage_plan is not None:
            local_stage_plans.append(stage_plan)
    assert (
        len(local_stage_plans) == 1
    ), f"Expected exactly one local stage plan for rank {my_rank}, got {len(local_stage_plans)}"

    stage_plan = local_stage_plans[0]
    host_io, forwarders = _build_local_stage_runtime(mesh_device, stage_plan, tensor_size_bytes, fifo_size, h2d_mode)
    _dispatch_merged_programs(mesh_device, host_io, forwarders)

    if stage_plan.host_io.owns_h2d:
        assert host_io is not None
        token_size_datums = tensor_size_bytes // 4
        h2d_socket = host_io.h2d_socket
        d2h_socket = host_io.d2h_socket
        assert h2d_socket is not None and d2h_socket is not None

        for i in range(num_iterations):
            torch_input = torch.arange(i * token_size_datums, (i + 1) * token_size_datums, dtype=torch.float32).reshape(
                1, token_size_datums
            )

            input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
            torch_output = torch.zeros(1, token_size_datums, dtype=torch.float32)
            output_tensor = ttnn.from_torch(torch_output, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)

            h2d_socket.write_tensor(input_tensor)
            d2h_socket.read_tensor(output_tensor)

            result_torch = ttnn.to_torch(output_tensor)
            assert torch.equal(
                torch_input, result_torch
            ), f"Generic stage-size loopback mismatch for {topology_config.name}.\nExpected: {torch_input}\nGot: {result_torch}"

    ttnn.distributed_context_barrier()
    _terminate_components(mesh_device, host_io, forwarders)


def _build_first_stage_input():
    """Build the stage-0 H2D metadata page and the expected embedding row (rank 0 only)."""

    token_id = 17
    token_size_datums = DeepseekMetadata.aligned_size_bytes() // dtype_size(torch.uint32)
    torch_input = torch.zeros(1, token_size_datums, dtype=torch.uint32)
    metadata_words = DeepseekMetadata(token_id=token_id).to_list()
    torch_input[0, : len(metadata_words)] = torch.tensor(metadata_words, dtype=torch.uint32)
    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn_dtype_from_torch_dtype(torch.uint32), layout=ttnn.ROW_MAJOR_LAYOUT
    )
    # Reproduce SyntheticWeightProvider.load_embedding's table (seed 42, randn(VOCAB, HIDDEN) bf16)
    # to get the expected embedding row for token_id. torch.randn fills a contiguous tensor
    # row-major from the seeded generator, so generating only the first token_id+1 rows yields a
    # row bit-identical to the full (VOCAB, HIDDEN) table's row (holds while HIDDEN is even, so
    # rows align with the normal kernel's value pairs). Avoids a ~1.8GB host allocation on rank 0.
    generator = torch.Generator().manual_seed(42)
    expected = torch.randn(
        token_id + 1,
        LogicalModelDimensions.HIDDEN_SIZE,
        generator=generator,
        dtype=torch.bfloat16,
    )[token_id]
    return input_tensor, expected


def _make_readback_tensor():
    """A bf16 host tensor sized to the ring payload (embedding row + trailing DeepseekMetadata struct)."""
    num_elems = ACTIVATION_W_TOKEN_META_PAGE_SIZE_BYTES // dtype_size(torch.bfloat16)
    return ttnn.from_torch(
        torch.zeros(1, num_elems, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )


@pytest.mark.parametrize(
    "loopback_mode",
    [
        "host",
        pytest.param(
            "fabric",
            marks=pytest.mark.skip(
                reason="fabric loopback requires first and last stage to be physically connected, enable only when applicable"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    ("topology_config", "mesh_device", "device_params"),
    [
        (
            GENERIC_STAGE_SIZE_LOOPBACK_CONFIG,
            GENERIC_STAGE_SIZE_LOOPBACK_CONFIG.mesh_device_param,
            GENERIC_STAGE_SIZE_LOOPBACK_CONFIG.make_device_params(),
        )
    ],
    ids=[f"{GENERIC_STAGE_SIZE_LOOPBACK_CONFIG.name}-pipeline"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_level_passthrough_transport_loopback(topology_config, mesh_device, device_params, loopback_mode):
    """Drive real Pipeline/PipelineBlock plumbing with allocation-derived routing.

    Runs the fused embedding kernel (the real model's config) and returns the activation via the
    last stage's D2H + MPI to rank 0 (the production host-loopback return path).
    """

    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    _require_multi_rank_distributed_context()
    use_fabric_loopback = loopback_mode == "fabric"
    host_loopback = loopback_mode == "host"

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    allocation = ttnn._ttnn.multi_device.experimental.resolve_blitz_decode_pipeline_allocation(use_fabric_loopback)
    num_stages = len(allocation.stages)
    my_rank = int(ttnn.distributed_context_get_rank())
    local_stage_plans = build_local_stage_socket_plans(allocation, my_rank)
    assert (
        len(local_stage_plans) == 1
    ), f"Expected exactly one local stage plan for rank {my_rank}, got {len(local_stage_plans)}"
    stage_plan = local_stage_plans[0]

    pipeline_config = [
        PipelineConfigEntry(
            stage.entry_endpoint.mesh_coord,
            (stage.exit_endpoint if stage.exit_endpoint is not None else stage.entry_endpoint).mesh_coord,
        )
        for stage in allocation.stages
    ]
    if allocation.initialize_loopback:
        # Fabric loopback only: the linear (host-loopback) allocation has no loopback entry.
        pipeline_config.append(
            PipelineConfigEntry(
                allocation.loopback_entry_endpoint.mesh_coord,
                allocation.host_egress_endpoint.mesh_coord,
            )
        )

    config = create_passthrough_pipeline_configuration(
        SyntheticWeightProvider(), num_stages, host_loopback=host_loopback
    )
    pipeline = config.build_pipeline(
        mesh_device,
        my_stage_idx=stage_plan.logical_stage_index,
        stages_metadata=build_stage_routing(allocation),
        pipeline_config=pipeline_config,
        stage_plan=stage_plan,
    )
    try:
        pipeline.setup_and_run()
        last_stage_idx = num_stages - 1

        # Gate on H2D ownership, not my_stage_idx == 0: a split stage 0 puts logical stage 0 on
        # two ranks, but only the H2D owner can write_token() / read back the looped embedding.
        if stage_plan.host_io.owns_h2d:
            input_tensor, expected = _build_first_stage_input()
            output_tensor = _make_readback_tensor()
            pipeline.write_token(input_tensor)
            returned = pipeline.read_output(output_tensor)
            # Host loopback returns the MPI-received tensor; fabric writes into output_tensor.
            result_torch = (returned if host_loopback else ttnn.to_torch(output_tensor)).reshape(-1)
            assert torch.equal(
                result_torch[: LogicalModelDimensions.HIDDEN_SIZE], expected
            ), f"{loopback_mode} loopback mismatch"
        elif host_loopback and pipeline.my_stage_idx == last_stage_idx:
            # Host loopback: the last stage reads the looped activation from its D2H and MPI-sends
            # it to rank 0.
            pipeline.read_output(_make_readback_tensor())

        pipeline.barrier()
    finally:
        pipeline.terminate()
