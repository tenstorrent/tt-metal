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

import pytest
import torch

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import (
    MeshWrapper,
    SocketInterface,
    _create_socket_resource,
    _group_by_device,
)
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.tests.unit_tests.stage_size_test_utils import (
    EdgeTransport,
    LocalRole,
    LocalStageSocketPlan,
    build_local_stage_socket_plan,
    get_generic_stage_size_loopback_topology_config_from_env,
)

GENERIC_STAGE_SIZE_LOOPBACK_CONFIG = get_generic_stage_size_loopback_topology_config_from_env()
PIPELINE_ENDPOINT_CORE_COORD = ttnn.CoreCoord(0, 0)


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
    if not (stage_plan.host_io.needs_h2d or stage_plan.host_io.needs_d2h):
        return None

    h2d_socket = None
    if stage_plan.host_io.needs_h2d:
        h2d_socket = ttnn.H2DSocket(
            mesh_device,
            _core_for_mesh_view_endpoint(stage_plan.host_io.h2d_target, pipeline_core_coord),
            ttnn.BufferType.L1,
            fifo_size,
            h2d_mode,
        )

    d2h_socket = None
    if stage_plan.host_io.needs_d2h:
        d2h_socket = ttnn.D2HSocket(
            mesh_device,
            _core_for_mesh_view_endpoint(stage_plan.host_io.d2h_source, pipeline_core_coord),
            fifo_size,
        )

    h2d_downstream_core = None
    if stage_plan.host_io.needs_h2d:
        output_edge = _local_output_edge(stage_plan)
        assert output_edge is not None, "Host ingress requires a local stage output edge"
        h2d_downstream_core = _core_for_mesh_view_endpoint(output_edge.src, pipeline_core_coord)

    d2h_upstream_core = None
    if stage_plan.host_io.needs_d2h:
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
        and not stage_plan.host_io.needs_h2d
        and not stage_plan.host_io.needs_d2h
    ):
        local_intra_socket_pair = _create_local_socket_pair_for_edge(
            mesh_device, stage_plan.intra_stage_edge, pipeline_core_coord, fifo_size
        )

    forwarders = []

    def _append_input_forwarder():
        if input_edge is None or not (stage_plan.host_io.needs_d2h or local_intra_socket_pair is not None):
            return

        upstream_socket = _create_socket_resource_for_edge(
            mesh_device,
            input_edge,
            pipeline_core_coord,
            fifo_size,
            stage_plan.my_rank,
            local_endpoint_type=ttnn.SocketEndpoint.RECEIVER,
        )
        downstream_socket = (
            host_io.get_upstream_socket() if stage_plan.host_io.needs_d2h else local_intra_socket_pair[0]
        )
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

        if stage_plan.host_io.needs_h2d:
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
    build_output_first = stage_plan.host_io.needs_h2d and input_edge is not None and output_edge is not None
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

    if stage_plan.host_io.needs_h2d:
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
