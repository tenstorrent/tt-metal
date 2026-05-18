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


import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import MeshWrapper, ParallelSocketInterface, SocketInterface
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
    [
        # Single-galaxy 2-mesh: rank 0 = tray 1, rank 1 = tray 2; each mesh is (2, 4).
        # Pairs with dual_tray_2x4_rank_bindings.yaml.
        (2, 4),
        # Two-galaxy 2-mesh: rank 0 = galaxy A, rank 1 = galaxy B; each mesh is (4, 8).
        # Pairs with dual_galaxy_rank_bindings.yaml + runme_2galaxy_cross_mesh_smoke.sh.
        (4, 8),
    ],
    ids=["1galaxy", "2galaxy"],
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


# ---------------------------------------------------------------------------
# Perf variant — measures end-to-end wall time + bandwidth for shipping a
# logical (rows × cols) bf16 tensor across the cross-mesh socket.
#
# The logical tensor is chunked into ``page_size_bytes``-sized H2D pages
# because the H2D/D2H sockets transfer one page per ``write_tensor`` /
# ``read_tensor`` call. fifo_size must be >= page_size; we set it 4×
# page_size to allow pipelining (sender can buffer a few pages ahead while
# the receiver is still draining).
#
# Two timing modes are measured for each (logical tensor shape, page_size):
#   - Bit-exact pass (one iteration): asserts the bytes round-trip identically.
#   - Hot loop (``num_logical_tensors`` iterations): no per-iter assertions,
#     just write/read in a tight loop, timed across MPI barriers. Reports
#     ms per logical-tensor transfer + GB/s.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    # (logical_rows, logical_cols, page_size_bytes, num_logical_tensors)
    # 640 × 1792 bf16 = 2,293,760 bytes per logical tensor. Sweep page sizes
    # to map the bandwidth curve (per-call overhead vs L1 footprint).
    # Constraint: logical_tensor_bytes % page_size_bytes == 0 (verified inline).
    # Each point allocates fifo = 4 × page_size in L1.
    "logical_rows, logical_cols, page_size_bytes, num_logical_tensors",
    [
        (640, 1792, 256, 20),
        (640, 1792, 1024, 20),
        (640, 1792, 2048, 20),
        (640, 1792, 4096, 20),
        (640, 1792, 8192, 20),
        (640, 1792, 16384, 20),
    ],
    ids=["pg256", "pg1024", "pg2048", "pg4096", "pg8192", "pg16384"],
)
@pytest.mark.parametrize("h2d_mode", [ttnn.H2DMode.HOST_PUSH])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (2, 4),
        (4, 8),
    ],
    ids=["1galaxy", "2galaxy"],
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
def test_cross_mesh_socket_perf(
    mesh_device, logical_rows, logical_cols, page_size_bytes, num_logical_tensors, h2d_mode
):
    """End-to-end perf of cross-mesh H2D→D2D→D2H for a (logical_rows, logical_cols) bf16 tensor.

    Reports ms per logical-tensor transfer + effective bandwidth. The logical
    tensor is chunked into ``page_size_bytes``-byte pages (one per write_tensor
    call). Wall time is measured across MPI barriers so both ranks include the
    same start/end points.
    """
    if not is_slow_dispatch():
        pytest.skip("Sockets require slow dispatch (set TT_METAL_SLOW_DISPATCH_MODE=1).")

    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 2:
        pytest.skip(f"This test runs with exactly 2 ranks; got num_procs={num_procs}")

    BF16_BYTES = 2
    logical_tensor_bytes = logical_rows * logical_cols * BF16_BYTES
    assert (
        logical_tensor_bytes % page_size_bytes == 0
    ), f"logical_tensor_bytes ({logical_tensor_bytes}) must be a multiple of page_size_bytes ({page_size_bytes})"
    pages_per_logical_tensor = logical_tensor_bytes // page_size_bytes

    # FIFO must be ≥ page_size; 4× pages of pipelining headroom.
    fifo_size = page_size_bytes * 4
    # Page size in uint32 datums (write_tensor takes a uint32 tensor).
    page_size_datums = page_size_bytes // 4

    my_mesh_id = mesh_device.get_system_mesh_id()
    logger.info(
        f"[mesh_id={my_mesh_id}] PERF setup: logical={logical_rows}x{logical_cols} bf16 "
        f"({logical_tensor_bytes} bytes), page={page_size_bytes} B "
        f"({pages_per_logical_tensor} pages/tensor), fifo={fifo_size} B, "
        f"hot-loop iters={num_logical_tensors}"
    )

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    entry_node_coord = ttnn.MeshCoordinate((0, 0))
    exit_node_coord = ttnn.MeshCoordinate((1, 0))
    pipeline_core_coord = ttnn.CoreCoord(0, 0)

    # Build sender or receiver side identically to the smoke test.
    if my_mesh_id == 0:
        h2d_socket = ttnn.H2DSocket(
            mesh_device,
            ttnn.MeshCoreCoord(entry_node_coord, pipeline_core_coord),
            ttnn.BufferType.L1,
            fifo_size,
            h2d_mode,
        )
        host_io = HostInterface(
            h2d_socket=h2d_socket,
            d2h_socket=None,
            h2d_page_size=page_size_bytes,
            d2h_page_size=page_size_bytes,
            core_to_core_socket_buffer_size=fifo_size,
            h2d_downstream_core=ttnn.MeshCoreCoord(exit_node_coord, pipeline_core_coord),
        )
        exit_socket = SocketInterface(
            page_size=page_size_bytes,
            socket_fifo_size=fifo_size,
            data_size_per_transfer=page_size_bytes,
            send_core_coord=ttnn.MeshCoreCoord(exit_node_coord, pipeline_core_coord),
            recv_core_coord=ttnn.MeshCoreCoord(entry_node_coord, pipeline_core_coord),
            upstream_socket=host_io.get_downstream_socket(),
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=1),
        )
        host_io.run()
        exit_socket.run()

        # Pre-build one full logical tensor of input pages. Bytes are arbitrary but
        # deterministic so the receiver can verify the first iteration bit-exactly.
        # Reused across the hot loop to keep the hot path free of per-iter alloc.
        input_pages = [
            ttnn.from_torch(
                torch.arange(p * page_size_datums, (p + 1) * page_size_datums, dtype=torch.int32).reshape(
                    1, page_size_datums
                ),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            for p in range(pages_per_logical_tensor)
        ]

        logger.info("[mesh_id=0] PERF: pre-built input pages; barrier before hot loop")
        ttnn.distributed_context_barrier()

        # --- hot loop (timed) ---
        t0 = time.perf_counter()
        for _t in range(num_logical_tensors):
            for p in range(pages_per_logical_tensor):
                h2d_socket.write_tensor(input_pages[p])
        ttnn.distributed_context_barrier()
        t1 = time.perf_counter()
        # ---

        elapsed_s = t1 - t0
        total_bytes = num_logical_tensors * logical_tensor_bytes
        ms_per_tensor = (elapsed_s / num_logical_tensors) * 1000.0
        gbps = total_bytes / elapsed_s / 1e9
        logger.info(
            f"[mesh_id=0] PERF RESULT: shape={logical_rows}x{logical_cols} bf16  "
            f"page={page_size_bytes}B  iters={num_logical_tensors}  "
            f"total={total_bytes/1e6:.2f} MB  elapsed={elapsed_s:.3f}s  "
            f"per-tensor={ms_per_tensor:.3f} ms  bw={gbps:.3f} GB/s"
        )

        host_io.terminate(False)
        exit_socket.terminate(True)

    else:
        d2h_socket = ttnn.D2HSocket(
            mesh_device,
            ttnn.MeshCoreCoord(exit_node_coord, pipeline_core_coord),
            fifo_size,
        )
        host_io = HostInterface(
            h2d_socket=None,
            d2h_socket=d2h_socket,
            h2d_page_size=page_size_bytes,
            d2h_page_size=page_size_bytes,
            core_to_core_socket_buffer_size=fifo_size,
            d2h_upstream_core=ttnn.MeshCoreCoord(entry_node_coord, pipeline_core_coord),
        )
        entry_socket = SocketInterface(
            page_size=page_size_bytes,
            socket_fifo_size=fifo_size,
            data_size_per_transfer=page_size_bytes,
            send_core_coord=ttnn.MeshCoreCoord(exit_node_coord, pipeline_core_coord),
            recv_core_coord=ttnn.MeshCoreCoord(entry_node_coord, pipeline_core_coord),
            downstream_socket=host_io.get_upstream_socket(),
            sender_mesh=MeshWrapper(mesh_id=0),
            receiver_mesh=MeshWrapper(mesh_device),
        )
        host_io.run()
        entry_socket.run()

        # One reusable output buffer; read_tensor overwrites it each call.
        output_tensor = ttnn.from_torch(
            torch.zeros(1, page_size_datums, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        logger.info("[mesh_id=1] PERF: ready; barrier before hot loop")
        ttnn.distributed_context_barrier()

        # --- hot loop (timed) ---
        t0 = time.perf_counter()
        verified_first = False
        for _t in range(num_logical_tensors):
            for p in range(pages_per_logical_tensor):
                d2h_socket.read_tensor(output_tensor)
                if not verified_first:
                    # Bit-exact verify only the very first page to confirm the path
                    # is healthy without slowing the hot loop on every iter.
                    got = ttnn.to_torch(output_tensor).to(torch.int32)
                    expected = torch.arange(
                        p * page_size_datums, (p + 1) * page_size_datums, dtype=torch.int32
                    ).reshape(1, page_size_datums)
                    assert torch.equal(expected, got), (
                        f"[mesh_id=1] first-page bit-exact check failed.\n"
                        f"Expected: {expected[:, :8]}...\n"
                        f"Got:      {got[:, :8]}..."
                    )
                    logger.info("[mesh_id=1] first page bit-exact OK; entering hot loop")
                    verified_first = True
        ttnn.distributed_context_barrier()
        t1 = time.perf_counter()
        # ---

        elapsed_s = t1 - t0
        total_bytes = num_logical_tensors * logical_tensor_bytes
        ms_per_tensor = (elapsed_s / num_logical_tensors) * 1000.0
        gbps = total_bytes / elapsed_s / 1e9
        logger.info(
            f"[mesh_id=1] PERF RESULT: shape={logical_rows}x{logical_cols} bf16  "
            f"page={page_size_bytes}B  iters={num_logical_tensors}  "
            f"total={total_bytes/1e6:.2f} MB  elapsed={elapsed_s:.3f}s  "
            f"per-tensor={ms_per_tensor:.3f} ms  bw={gbps:.3f} GB/s"
        )

        host_io.terminate(False)
        entry_socket.terminate(True)

    logger.info(f"[rank with mesh_id={my_mesh_id}] PERF done")


# ---------------------------------------------------------------------------
# Parallel variant — N chip-pair sockets in parallel via ParallelSocketInterface.
#
# Per-channel layout (per chip):
#   core (0,0) "entry"  — D2D recv core on receiver, sender's exit-node-recv on sender
#   core (0,1) "exit"   — D2D send core on sender, receiver's entry-node-send on receiver
#   core (0,2) "io"     — H2D socket on rank 0; D2H socket on rank 1
#
# Per chip on rank 0: H2D socket → HostInterface forwards to (0,1) exit core
#                   → ParallelSocketInterface channel N → fabric → rank 1's chip N entry core
# Per chip on rank 1: ParallelSocketInterface channel N delivers to (0,0) entry core
#                   → HostInterface forwards to (0,2) → D2H socket back to host
#
# One ParallelSocketInterface per rank owns N channels (one per chip in the
# mesh). All N sockets run in one kernel dispatch and can carry data
# simultaneously across the inter-mesh fabric. This is the same machinery
# the production decode pipeline uses for inter-stage data movement; we test
# it at the unit level here.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    # (num_channels, page_size_bytes, num_logical_tensors_per_channel)
    # Each channel transfers a 640×1792 bf16 (= 2,293,760 bytes) "logical tensor"
    # per iteration. Aggregate per iteration = num_channels × that.
    "num_channels, page_size_bytes, num_logical_tensors_per_channel",
    [
        (1, 4096, 20),
        (4, 4096, 20),
        (8, 4096, 20),
        (16, 4096, 20),
        (32, 4096, 20),
    ],
    ids=["nc1", "nc4", "nc8", "nc16", "nc32"],
)
@pytest.mark.parametrize("h2d_mode", [ttnn.H2DMode.HOST_PUSH])
@pytest.mark.parametrize(
    "mesh_device",
    [
        (2, 4),
        (4, 8),
    ],
    ids=["1galaxy", "2galaxy"],
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
def test_cross_mesh_socket_parallel_perf(
    mesh_device, num_channels, page_size_bytes, num_logical_tensors_per_channel, h2d_mode
):
    """N-channel parallel cross-mesh socket: one socket per chip-pair, all in flight at once.

    Each channel ships a 640×1792 bf16 logical tensor per iteration (chunked into
    page_size_bytes-sized H2D writes). Wall time is measured across MPI barriers;
    aggregate bandwidth = (num_channels × logical_bytes_per_chan × iters) / elapsed.

    Skipped when num_channels exceeds the available chip count for the chosen
    mesh shape: max 8 for (2,4) single-galaxy, max 32 for (4,8) two-galaxy.

    First page of channel 0 is bit-exact verified to confirm the path is healthy;
    rest of the hot loop runs without per-iter asserts to keep timing honest.
    """
    if not is_slow_dispatch():
        pytest.skip("Sockets require slow dispatch (set TT_METAL_SLOW_DISPATCH_MODE=1).")

    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 2:
        pytest.skip(f"This test runs with exactly 2 ranks; got num_procs={num_procs}")

    mesh_rows, mesh_cols = mesh_device.shape
    num_chips_in_mesh = int(mesh_rows) * int(mesh_cols)
    if num_channels > num_chips_in_mesh:
        pytest.skip(f"num_channels={num_channels} exceeds chips in this mesh ({num_chips_in_mesh})")

    BF16_BYTES = 2
    logical_rows = 640
    logical_cols = 1792
    logical_bytes_per_chan = logical_rows * logical_cols * BF16_BYTES  # 2,293,760
    assert (
        logical_bytes_per_chan % page_size_bytes == 0
    ), f"logical_bytes_per_chan ({logical_bytes_per_chan}) must be a multiple of page_size_bytes ({page_size_bytes})"
    pages_per_chan = logical_bytes_per_chan // page_size_bytes

    fifo_size = page_size_bytes * 4
    page_size_datums = page_size_bytes // 4

    my_mesh_id = mesh_device.get_system_mesh_id()
    peer_mesh_id = 1 if my_mesh_id == 0 else 0

    # Pick the first num_channels chips in row-major order.
    device_coords = [ttnn.MeshCoordinate((r, c)) for r in range(int(mesh_rows)) for c in range(int(mesh_cols))][
        :num_channels
    ]
    core_entry = ttnn.CoreCoord(0, 0)
    core_exit = ttnn.CoreCoord(0, 1)
    core_io = ttnn.CoreCoord(0, 2)

    logger.info(
        f"[mesh_id={my_mesh_id}] PARALLEL setup: nc={num_channels} on mesh {mesh_rows}x{mesh_cols}, "
        f"page={page_size_bytes} B ({pages_per_chan} pages/chan/iter), fifo={fifo_size} B, "
        f"iters={num_logical_tensors_per_channel}, "
        f"aggregate per logical_tensor = {num_channels * logical_bytes_per_chan / 1e6:.2f} MB"
    )

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    if my_mesh_id == 0:
        # ----------------- SENDER -----------------
        h2d_sockets = []
        host_ios = []
        for dc in device_coords:
            h2d = ttnn.H2DSocket(
                mesh_device,
                ttnn.MeshCoreCoord(dc, core_io),
                ttnn.BufferType.L1,
                fifo_size,
                h2d_mode,
            )
            hio = HostInterface(
                h2d_socket=h2d,
                d2h_socket=None,
                h2d_page_size=page_size_bytes,
                d2h_page_size=page_size_bytes,
                core_to_core_socket_buffer_size=fifo_size,
                h2d_downstream_core=ttnn.MeshCoreCoord(dc, core_exit),
            )
            h2d_sockets.append(h2d)
            host_ios.append(hio)

        # Same chip on each side: send from chip's exit core, receive on
        # corresponding chip's entry core in the peer mesh.
        send_cores = [ttnn.MeshCoreCoord(dc, core_exit) for dc in device_coords]
        recv_cores = [ttnn.MeshCoreCoord(dc, core_entry) for dc in device_coords]

        parallel_socket = ParallelSocketInterface(
            page_size=page_size_bytes,
            socket_fifo_size=fifo_size,
            send_core_coords=send_cores,
            recv_core_coords=recv_cores,
            upstream_sockets=[hio.get_downstream_socket() for hio in host_ios],
            sender_mesh=MeshWrapper(mesh_device),
            receiver_mesh=MeshWrapper(mesh_id=peer_mesh_id),
        )

        for hio in host_ios:
            hio.run()
        parallel_socket.run()

        # Pre-build pages so the hot loop is free of host-side construction.
        # All channels share the same input pages (channel 0's data is the
        # bit-exact reference; other channels carry the same bytes — receiver
        # will verify channel 0's first page).
        input_pages = [
            ttnn.from_torch(
                torch.arange(p * page_size_datums, (p + 1) * page_size_datums, dtype=torch.int32).reshape(
                    1, page_size_datums
                ),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            for p in range(pages_per_chan)
        ]

        logger.info(f"[mesh_id=0] PARALLEL: pages pre-built; barrier before hot loop")
        ttnn.distributed_context_barrier()

        # --- hot loop (timed) ---
        t0 = time.perf_counter()
        for _t in range(num_logical_tensors_per_channel):
            for p in range(pages_per_chan):
                # Push the same page into each channel's H2D socket. The
                # parallel kernel drains all channels' FIFOs simultaneously.
                for ch in range(num_channels):
                    h2d_sockets[ch].write_tensor(input_pages[p])
        ttnn.distributed_context_barrier()
        t1 = time.perf_counter()
        # ---

        elapsed_s = t1 - t0
        total_bytes = num_logical_tensors_per_channel * num_channels * logical_bytes_per_chan
        per_logical_ms = (elapsed_s / num_logical_tensors_per_channel) * 1000.0
        agg_gbps = total_bytes / elapsed_s / 1e9
        per_chan_gbps = (total_bytes / num_channels) / elapsed_s / 1e9
        logger.info(
            f"[mesh_id=0] PARALLEL RESULT: nc={num_channels}  page={page_size_bytes}B  "
            f"iters={num_logical_tensors_per_channel}  total={total_bytes/1e6:.2f} MB  "
            f"elapsed={elapsed_s:.3f}s  per-logical-tensor={per_logical_ms:.3f} ms  "
            f"aggregate_bw={agg_gbps:.3f} GB/s  per_channel_bw={per_chan_gbps:.3f} GB/s"
        )

        for hio in host_ios:
            hio.terminate(False)
        parallel_socket.terminate(True)

    else:
        # ----------------- RECEIVER -----------------
        d2h_sockets = []
        host_ios = []
        for dc in device_coords:
            d2h = ttnn.D2HSocket(
                mesh_device,
                ttnn.MeshCoreCoord(dc, core_io),
                fifo_size,
            )
            hio = HostInterface(
                h2d_socket=None,
                d2h_socket=d2h,
                h2d_page_size=page_size_bytes,
                d2h_page_size=page_size_bytes,
                core_to_core_socket_buffer_size=fifo_size,
                d2h_upstream_core=ttnn.MeshCoreCoord(dc, core_entry),
            )
            d2h_sockets.append(d2h)
            host_ios.append(hio)

        # Mirror the sender's wiring: same chip pairs send/recv between the meshes.
        send_cores = [ttnn.MeshCoreCoord(dc, core_exit) for dc in device_coords]
        recv_cores = [ttnn.MeshCoreCoord(dc, core_entry) for dc in device_coords]

        parallel_socket = ParallelSocketInterface(
            page_size=page_size_bytes,
            socket_fifo_size=fifo_size,
            send_core_coords=send_cores,
            recv_core_coords=recv_cores,
            downstream_sockets=[hio.get_upstream_socket() for hio in host_ios],
            sender_mesh=MeshWrapper(mesh_id=peer_mesh_id),
            receiver_mesh=MeshWrapper(mesh_device),
        )

        for hio in host_ios:
            hio.run()
        parallel_socket.run()

        # One reusable output tensor per channel (read_tensor overwrites in place).
        output_tensors = [
            ttnn.from_torch(
                torch.zeros(1, page_size_datums, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            for _ in range(num_channels)
        ]

        logger.info(f"[mesh_id=1] PARALLEL: ready; barrier before hot loop")
        ttnn.distributed_context_barrier()

        # --- hot loop (timed) ---
        verified_first = False
        t0 = time.perf_counter()
        for _t in range(num_logical_tensors_per_channel):
            for p in range(pages_per_chan):
                for ch in range(num_channels):
                    d2h_sockets[ch].read_tensor(output_tensors[ch])
                    if not verified_first:
                        got = ttnn.to_torch(output_tensors[ch]).to(torch.int32)
                        expected = torch.arange(
                            p * page_size_datums, (p + 1) * page_size_datums, dtype=torch.int32
                        ).reshape(1, page_size_datums)
                        assert torch.equal(expected, got), f"[mesh_id=1] ch={ch} first-page bit-exact check failed."
                        logger.info(f"[mesh_id=1] ch=0 first page bit-exact OK; entering hot loop")
                        verified_first = True
        ttnn.distributed_context_barrier()
        t1 = time.perf_counter()
        # ---

        elapsed_s = t1 - t0
        total_bytes = num_logical_tensors_per_channel * num_channels * logical_bytes_per_chan
        per_logical_ms = (elapsed_s / num_logical_tensors_per_channel) * 1000.0
        agg_gbps = total_bytes / elapsed_s / 1e9
        per_chan_gbps = (total_bytes / num_channels) / elapsed_s / 1e9
        logger.info(
            f"[mesh_id=1] PARALLEL RESULT: nc={num_channels}  page={page_size_bytes}B  "
            f"iters={num_logical_tensors_per_channel}  total={total_bytes/1e6:.2f} MB  "
            f"elapsed={elapsed_s:.3f}s  per-logical-tensor={per_logical_ms:.3f} ms  "
            f"aggregate_bw={agg_gbps:.3f} GB/s  per_channel_bw={per_chan_gbps:.3f} GB/s"
        )

        for hio in host_ios:
            hio.terminate(False)
        parallel_socket.terminate(True)

    logger.info(f"[rank with mesh_id={my_mesh_id}] PARALLEL done")
