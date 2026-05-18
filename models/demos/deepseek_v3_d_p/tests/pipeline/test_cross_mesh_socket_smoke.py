# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Single-galaxy 2-mesh cross-mesh socket smoke test.

Goal
----
Verify that two submeshes carved out of one Blackhole galaxy can exchange
a tensor end-to-end via the production socket stack:

    host ──H2D──> chip on mesh_id=0 ──cross-mesh D2D (fabric)──> chip on mesh_id=1 ──D2H──> host

This is the minimal cross-mesh fabric exercise. It's a 2-rank cut-down of
``models/demos/deepseek_v3_b1/tests/unit_tests/test_multi_host_pipeline.py
::test_multi_host_loopback_pipeline`` (currently @pytest.mark.skip per
#43085) — one hop, no fabric loopback.

If this passes on a single galaxy, the same Python test should also work
on a 2-galaxy setup (step 2) by swapping the rank-binding YAML.
If this hangs, the cross-mesh fabric path is the same bug area as #43079 /
#43085 / our earlier HostIoDecoderStage hang.

How to run
----------
See ``runme.sh`` next to this file. In short::

    tt-run --bare \
        --rank-binding tests/tt_metal/distributed/config/mock_galaxy_single_host_subcontext_b_rank_bindings.yaml \
        --mpi-args "--oversubscribe" \
        python -m pytest models/demos/deepseek_v3_d_p/tests/pipeline/test_cross_mesh_socket_smoke.py -svv

The rank-binding YAML pins:
  - rank 0 → mesh_id=0, TT_VISIBLE_DEVICES=4,7,10,14,15,18,25,30 (galaxy slice 2)
  - rank 1 → mesh_id=1, TT_VISIBLE_DEVICES=0,2,3,6,8,21,28,31 (galaxy slice 3)
  mesh_graph_desc_path = bh_galaxy_dual_2x4_intermesh.textproto
"""


import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import MeshWrapper, SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import create_fabric_router_config

# Production-shape fabric + L1 settings; mirrors test_multi_host_pipeline.py.
# FABRIC_2D is the simpler 2D fabric (no torus). #43085 was specifically about
# FABRIC_2D_TORUS_Y; FABRIC_2D gives us a control-knob to see if the bug
# follows the fabric mode or is broader.
_FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_2D
_FABRIC_ROUTER_MAX_PAYLOAD_BYTES = 2048


@pytest.mark.parametrize(
    "tensor_size_bytes, fifo_size, num_iterations",
    [
        (64, 256, 4),
    ],
)
@pytest.mark.parametrize(
    "h2d_mode",
    [
        ttnn.H2DMode.HOST_PUSH,
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(2, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": _FABRIC_CONFIG,
            "fabric_router_config": create_fabric_router_config(_FABRIC_ROUTER_MAX_PAYLOAD_BYTES),
        }
    ],
    indirect=True,
)
def test_cross_mesh_socket_smoke(mesh_device, tensor_size_bytes, fifo_size, num_iterations, h2d_mode):
    """One H2D → one cross-mesh D2D hop → one D2H, across 2 ranks on 1 galaxy.

    Requires tt-run with the dual-2x4 intermesh rank-binding (2 procs total).
    Each rank owns a (2, 4) submesh on a distinct mesh_id.

    Rank 0 (sender, mesh_id=0):
        - Builds H2DSocket on a local entry-node core.
        - HostInterface forwards H2D bytes to a local "exit-node" core via a
          core-to-core L1 socket.
        - SocketInterface (sender side) sends from that exit-node core, across
          the intermesh fabric, to rank 1's entry-node core (cross-mesh D2D).

    Rank 1 (receiver, mesh_id=1):
        - SocketInterface (receiver side) lands incoming bytes on a local
          entry-node core.
        - HostInterface forwards from there to a local exit-node core, where
          D2HSocket pushes bytes back to host.

    Per iteration:
        rank 0 calls h2d_socket.write_tensor(input);
        rank 1 calls d2h_socket.read_tensor(output);
        rank 1 asserts output == input (rank 0's input is built deterministically
        from the iteration index so rank 1 can reconstruct it).
    """
    if not is_slow_dispatch():
        pytest.skip("Sockets require slow dispatch (set TT_METAL_SLOW_DISPATCH_MODE=1).")

    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 2:
        pytest.skip(f"This test runs with exactly 2 ranks; got num_procs={num_procs}")

    my_mesh_id = mesh_device.get_system_mesh_id()
    logger.info(
        f"[rank with mesh_id={my_mesh_id}] starting cross-mesh socket smoke, "
        f"tensor_size_bytes={tensor_size_bytes}, fifo_size={fifo_size}, iters={num_iterations}"
    )

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    # Entry / exit "node" coords inside this rank's (2, 4) submesh — same convention
    # as test_multi_host_pipeline.py (entry at (0,0), exit at (1,0)) so we hit two
    # distinct chips within the mesh, exercising intra-mesh routing in addition to
    # the cross-mesh hop.
    entry_node_coord = ttnn.MeshCoordinate((0, 0))
    exit_node_coord = ttnn.MeshCoordinate((1, 0))
    pipeline_core_coord = ttnn.CoreCoord(0, 0)

    tensor_size_datums = tensor_size_bytes // 4  # int32 datums

    if my_mesh_id == 0:
        # ---------- SENDER ----------
        logger.info("[mesh_id=0] building H2D socket + sender SocketInterface")

        h2d_socket = ttnn.H2DSocket(
            mesh_device,
            ttnn.MeshCoreCoord(entry_node_coord, pipeline_core_coord),
            ttnn.BufferType.L1,
            fifo_size,
            h2d_mode,
        )

        # h2d_socket only; d2h_socket=None is allowed. HostInterface still wants a
        # d2h_page_size value — pass the same size to keep validators happy.
        host_io = HostInterface(
            h2d_socket=h2d_socket,
            d2h_socket=None,
            h2d_page_size=tensor_size_bytes,
            d2h_page_size=tensor_size_bytes,
            core_to_core_socket_buffer_size=fifo_size,
            h2d_downstream_core=ttnn.MeshCoreCoord(exit_node_coord, pipeline_core_coord),
        )

        # Cross-mesh D2D: my exit-node core ──fabric──> peer mesh_id=1's entry-node core.
        exit_socket = SocketInterface(
            page_size=tensor_size_bytes,
            socket_fifo_size=fifo_size,
            data_size_per_transfer=tensor_size_bytes,
            send_core_coord=ttnn.MeshCoreCoord(exit_node_coord, pipeline_core_coord),
            recv_core_coord=ttnn.MeshCoreCoord(entry_node_coord, pipeline_core_coord),
            upstream_socket=host_io.get_downstream_socket(),
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=1),
        )

        host_io.run()
        exit_socket.run()

        logger.info("[mesh_id=0] waiting on barrier before send")
        ttnn.distributed_context_barrier()

        for i in range(num_iterations):
            torch_input = torch.arange(i * tensor_size_datums, (i + 1) * tensor_size_datums, dtype=torch.int32).reshape(
                1, tensor_size_datums
            )
            input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

            logger.info(
                f"[mesh_id=0] iter {i}: write_tensor (range {i * tensor_size_datums}..{(i + 1) * tensor_size_datums})"
            )
            h2d_socket.write_tensor(input_tensor)

        logger.info("[mesh_id=0] sent all iterations; barrier")
        ttnn.distributed_context_barrier()

        logger.info("[mesh_id=0] terminating")
        host_io.terminate(False)
        exit_socket.terminate(True)

    else:
        # ---------- RECEIVER (mesh_id == 1) ----------
        logger.info("[mesh_id=1] building entry SocketInterface + D2H socket")

        d2h_socket = ttnn.D2HSocket(
            mesh_device,
            ttnn.MeshCoreCoord(exit_node_coord, pipeline_core_coord),
            fifo_size,
        )

        host_io = HostInterface(
            h2d_socket=None,
            d2h_socket=d2h_socket,
            h2d_page_size=tensor_size_bytes,
            d2h_page_size=tensor_size_bytes,
            core_to_core_socket_buffer_size=fifo_size,
            d2h_upstream_core=ttnn.MeshCoreCoord(entry_node_coord, pipeline_core_coord),
        )

        # Cross-mesh D2D receiver: rank 0's exit-node core ──fabric──> my entry-node core.
        entry_socket = SocketInterface(
            page_size=tensor_size_bytes,
            socket_fifo_size=fifo_size,
            data_size_per_transfer=tensor_size_bytes,
            send_core_coord=ttnn.MeshCoreCoord(exit_node_coord, pipeline_core_coord),
            recv_core_coord=ttnn.MeshCoreCoord(entry_node_coord, pipeline_core_coord),
            downstream_socket=host_io.get_upstream_socket(),
            sender_mesh=MeshWrapper(mesh_id=0),
            receiver_mesh=MeshWrapper(mesh_device),
        )

        host_io.run()
        entry_socket.run()

        logger.info("[mesh_id=1] waiting on barrier before recv")
        ttnn.distributed_context_barrier()

        for i in range(num_iterations):
            torch_output = torch.zeros(1, tensor_size_datums, dtype=torch.int32)
            output_tensor = ttnn.from_torch(torch_output, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

            logger.info(f"[mesh_id=1] iter {i}: read_tensor")
            d2h_socket.read_tensor(output_tensor)

            result_torch = ttnn.to_torch(output_tensor).to(torch.int32)
            expected_torch = torch.arange(
                i * tensor_size_datums, (i + 1) * tensor_size_datums, dtype=torch.int32
            ).reshape(1, tensor_size_datums)
            match = torch.equal(expected_torch, result_torch)
            assert match, (
                f"[mesh_id=1] iter {i}: cross-mesh roundtrip mismatch.\n"
                f"Expected: {expected_torch}\n"
                f"Got:      {result_torch}"
            )
            logger.info(f"[mesh_id=1] iter {i}: roundtrip OK")

        logger.info("[mesh_id=1] received all iterations; barrier")
        ttnn.distributed_context_barrier()

        logger.info("[mesh_id=1] terminating")
        host_io.terminate(False)
        entry_socket.terminate(True)

    logger.info(f"[rank with mesh_id={my_mesh_id}] smoke OK")
