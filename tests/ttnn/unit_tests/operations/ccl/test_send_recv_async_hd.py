# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the host <-> device socket ops.

Covers:

- ``ttnn.experimental.recv_async_h2d``: streams pages from an ``H2DSocket`` into a
  pre-allocated device output tensor.
- ``ttnn.experimental.send_async_d2h``: streams pages of a device input tensor out to
  a ``D2HSocket`` so the host can read them.
- An end-to-end pipeline that wires both ops together with a matmul in between: host
  pushes input pages via H2D, the device runs a matmul against a pre-allocated weight
  tensor, and the result is streamed back to the host via D2H. The host-side result
  is compared against a torch reference.

"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# recv_async_h2d
# ---------------------------------------------------------------------------


def _run_recv_async_h2d(
    mesh_device,
    page_size_bytes,
    num_pages,
    fifo_size_bytes,
    num_iterations,
    h2d_mode,
):
    """Drive ``recv_async_h2d`` for ``num_iterations`` of ``num_pages`` each.

    Each iteration:
      1. Allocates a zeroed device ``output_tensor``.
      2. Kicks off the device program; the kernel parks at ``socket_wait_for_pages``.
      3. Pushes ``num_pages`` pages of monotonically-increasing uint32 data via the
         H2D socket from host.
      4. Synchronizes the device and reads ``output_tensor`` back.
      5. Asserts byte-exact equality vs the host-side input.
    """
    page_size_datums = page_size_bytes // 4
    tensor_shape = (num_pages, page_size_datums)

    device_coord = ttnn.MeshCoordinate(0, 0)
    core_coord = ttnn.CoreCoord(0, 0)
    socket_core = ttnn.MeshCoreCoord(device_coord, core_coord)

    logger.info(
        f"H2D mode={h2d_mode}, page_size={page_size_bytes}B, num_pages={num_pages}, "
        f"fifo_size={fifo_size_bytes}B, iterations={num_iterations}"
    )

    h2d_socket = ttnn.H2DSocket(mesh_device, socket_core, ttnn.BufferType.L1, fifo_size_bytes, h2d_mode)
    # recv_async_h2d's validator cross-checks that the socket's page size matches the
    # output tensor's aligned page size. Configure it once up front; the kernel will
    # also call set_receiver_socket_page_size internally.
    h2d_socket.set_page_size(page_size_bytes)

    for iteration in range(num_iterations):
        torch_input = torch.arange(
            iteration * num_pages * page_size_datums,
            (iteration + 1) * num_pages * page_size_datums,
            dtype=torch.int32,
        ).reshape(tensor_shape)

        # Pre-allocate the device output tensor that the op will write into.
        output_tensor = ttnn.from_torch(
            torch.zeros(tensor_shape, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Dispatch the receiver kernel. It blocks on socket_wait_for_pages until the
        # host pushes the matching pages below.
        ttnn.experimental.recv_async_h2d(output_tensor, h2d_socket)

        h2d_socket.write_tensor(torch_input)

        # Ensure the kernel has popped/written every page before reading the tensor.
        ttnn.synchronize_device(mesh_device)

        result = ttnn.to_torch(output_tensor).to(torch.int32)
        assert torch.equal(torch_input, result), (
            f"recv_async_h2d output mismatch on iteration {iteration} "
            f"(h2d_mode={h2d_mode}, page_size={page_size_bytes}B, num_pages={num_pages}).\n"
            f"Expected first row: {torch_input[0, :8].tolist()}\n"
            f"Got first row:      {result[0, :8].tolist()}"
        )


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "h2d_mode",
    [
        ttnn.H2DMode.HOST_PUSH,
    ],
)
@pytest.mark.parametrize(
    "page_size_bytes, num_pages, fifo_size_bytes, num_iterations",
    [
        # Tiny pages, many iterations: stresses FIFO wrap-around and per-page notify.
        (64, 1, 128, 32),
        (64, 4, 256, 16),
        (64, 8, 512, 16),
        # Medium pages: FIFO holds multiple pages.
        (256, 4, 1024, 8),
        (512, 2, 1024, 8),
    ],
)
def test_recv_async_h2d_basic(
    mesh_device,
    h2d_mode,
    page_size_bytes,
    num_pages,
    fifo_size_bytes,
    num_iterations,
):
    _run_recv_async_h2d(
        mesh_device,
        page_size_bytes=page_size_bytes,
        num_pages=num_pages,
        fifo_size_bytes=fifo_size_bytes,
        num_iterations=num_iterations,
        h2d_mode=h2d_mode,
    )


# ---------------------------------------------------------------------------
# send_async_d2h
# ---------------------------------------------------------------------------


def _run_send_async_d2h(
    mesh_device,
    page_size_bytes,
    num_pages,
    fifo_size_bytes,
    num_iterations,
):
    """Drive ``send_async_d2h`` for ``num_iterations`` of ``num_pages`` each.

    Each iteration:
      1. Allocates a device ``input_tensor`` filled with monotonically-increasing
         uint32 data.
      2. Kicks off the device program; the kernel reads each tensor page into L1 and
         pushes it to the D2H socket FIFO in pinned host memory via PCIe.
      3. Reads ``num_pages`` pages off the D2H socket from host into a pre-allocated
         host tensor.
      4. Synchronizes the device.
      5. Asserts byte-exact equality vs the original device-side input.
    """
    page_size_datums = page_size_bytes // 4
    tensor_shape = (num_pages, page_size_datums)

    device_coord = ttnn.MeshCoordinate(0, 0)
    core_coord = ttnn.CoreCoord(0, 0)
    socket_core = ttnn.MeshCoreCoord(device_coord, core_coord)

    logger.info(
        f"page_size={page_size_bytes}B, num_pages={num_pages}, "
        f"fifo_size={fifo_size_bytes}B, iterations={num_iterations}"
    )

    d2h_socket = ttnn.D2HSocket(mesh_device, socket_core, fifo_size_bytes)
    # send_async_d2h's validator cross-checks that the socket's page size matches the
    # input tensor's aligned page size. Configure it once up front; the kernel will
    # also call set_sender_socket_page_size internally.
    d2h_socket.set_page_size(page_size_bytes)

    for iteration in range(num_iterations):
        torch_input = torch.arange(
            iteration * num_pages * page_size_datums,
            (iteration + 1) * num_pages * page_size_datums,
            dtype=torch.int32,
        ).reshape(tensor_shape)

        input_tensor = ttnn.from_torch(
            torch_input,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Dispatch the sender kernel. It blocks on socket_reserve_pages until host
        # reads (below) free space in the FIFO.
        ttnn.experimental.send_async_d2h(input_tensor, d2h_socket)

        result = torch.zeros(torch_input.shape, dtype=torch.uint32)
        d2h_socket.read_tensor(result)

        # Ensure the kernel has finished updating the socket state before the next
        # iteration re-uses the same socket.
        ttnn.synchronize_device(mesh_device)
        result = result.to(torch.int32)
        assert torch.equal(torch_input, result), (
            f"send_async_d2h output mismatch on iteration {iteration} "
            f"(page_size={page_size_bytes}B, num_pages={num_pages}).\n"
            f"Expected first row: {torch_input[0, :8].tolist()}\n"
            f"Got first row:      {result[0, :8].tolist()}"
        )


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "page_size_bytes, num_pages, fifo_size_bytes, num_iterations",
    [
        # Tiny pages, many iterations: stresses FIFO wrap-around and per-page notify.
        (64, 1, 128, 32),
        (64, 4, 256, 16),
        (64, 8, 512, 16),
        # Medium pages: FIFO holds multiple pages.
        (256, 4, 1024, 8),
        (512, 2, 1024, 8),
    ],
)
def test_send_async_d2h_basic(
    mesh_device,
    page_size_bytes,
    num_pages,
    fifo_size_bytes,
    num_iterations,
):
    _run_send_async_d2h(
        mesh_device,
        page_size_bytes=page_size_bytes,
        num_pages=num_pages,
        fifo_size_bytes=fifo_size_bytes,
        num_iterations=num_iterations,
    )


# ---------------------------------------------------------------------------
# Combined: recv_async_h2d -> matmul -> send_async_d2h
# ---------------------------------------------------------------------------


def _run_recv_matmul_send_async(
    mesh_device,
    M,
    K,
    N,
    h2d_mode,
    num_iterations,
):
    """End-to-end host <-> device matmul pipeline.

    Each iteration:
      1. Host pushes an ``(M, K)`` bfloat16 input tensor through an H2DSocket; the
         device kernel writes it into a pre-allocated row-major L1 tensor.
      2. The tensor is converted to TILE layout and matmul'd against a static
         weight tensor pre-allocated on device in DRAM.
      3. The TILE output is converted back to ROW_MAJOR (so the page layout matches
         the D2H socket) and streamed to the host via send_async_d2h.
      4. The host result is compared against ``torch_input @ torch_weight`` with PCC.

    The H2D and D2H sockets live on disjoint cores so their kernel programs do not
    contend for the same tensix.
    """
    # bfloat16 = 2 bytes per element. ROW_MAJOR tensors page one row at a time.
    bytes_per_element = 2
    input_page_size_bytes = K * bytes_per_element
    output_page_size_bytes = N * bytes_per_element

    # FIFO sized to hold a handful of pages so the per-page reserve / wait paths
    # exercise wrap-around as well.
    h2d_fifo_size_bytes = max(2048, input_page_size_bytes * 4)
    d2h_fifo_size_bytes = max(2048, output_page_size_bytes * 4)

    device_coord = ttnn.MeshCoordinate(0, 0)
    h2d_core = ttnn.MeshCoreCoord(device_coord, ttnn.CoreCoord(0, 0))
    d2h_core = ttnn.MeshCoreCoord(device_coord, ttnn.CoreCoord(1, 0))

    logger.info(
        f"Pipeline: H2D mode={h2d_mode}, M={M}, K={K}, N={N}, "
        f"input_page={input_page_size_bytes}B x{M}, output_page={output_page_size_bytes}B x{M}, "
        f"iterations={num_iterations}"
    )

    h2d_socket = ttnn.H2DSocket(mesh_device, h2d_core, ttnn.BufferType.L1, h2d_fifo_size_bytes, h2d_mode)
    h2d_socket.set_page_size(input_page_size_bytes)

    d2h_socket = ttnn.D2HSocket(mesh_device, d2h_core, d2h_fifo_size_bytes)
    d2h_socket.set_page_size(output_page_size_bytes)

    # Static weight, allocated on device once and reused across iterations.
    torch_weight = torch.randn(K, N, dtype=torch.float32)
    weight_tensor = ttnn.from_torch(
        torch_weight,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for iteration in range(num_iterations):
        torch_input = torch.randn(M, K, dtype=torch.float32)
        torch_input_bf16 = torch_input.to(torch.bfloat16)
        # 1) Pre-allocate the row-major L1 input tensor that recv_async_h2d will fill,
        # then dispatch the receiver. The kernel parks at socket_wait_for_pages.
        input_tensor_rm = ttnn.from_torch(
            torch.zeros(M, K, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.experimental.recv_async_h2d(input_tensor_rm, h2d_socket)
        h2d_socket.write_tensor(torch_input_bf16)

        # 2) Convert to tile layout for the matmul; the matmul output stays in tile.
        input_tensor_tile = ttnn.to_layout(input_tensor_rm, ttnn.TILE_LAYOUT)
        matmul_output_tile = ttnn.matmul(input_tensor_tile, weight_tensor)

        # 3) Convert back to row-major so the per-page layout matches the D2H socket,
        # and stream the result back to the host.
        matmul_output_rm = ttnn.to_layout(matmul_output_tile, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.send_async_d2h(matmul_output_rm, d2h_socket)

        result = torch.zeros(M, N, dtype=torch.bfloat16)
        d2h_socket.read_tensor(result)
        ttnn.synchronize_device(mesh_device)

        # 4) Compare to a torch reference. We use PCC because the device matmul is
        # bfloat16 and won't be bit-exact against an fp32 reference.
        torch_expected = torch_input.to(torch.float32) @ torch_weight
        result = result.to(torch.float32)
        assert_with_pcc(torch_expected, result, pcc=0.99)


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "h2d_mode",
    [
        ttnn.H2DMode.HOST_PUSH,
    ],
)
@pytest.mark.parametrize(
    "M, K, N, num_iterations",
    [
        # Single-tile matmul: cheapest end-to-end smoke.
        (32, 32, 32, 4),
        # Multi-tile matmul along all three axes.
        (64, 64, 64, 2),
    ],
)
def test_recv_matmul_send_async(
    mesh_device,
    h2d_mode,
    M,
    K,
    N,
    num_iterations,
):
    _run_recv_matmul_send_async(
        mesh_device,
        M=M,
        K=K,
        N=N,
        h2d_mode=h2d_mode,
        num_iterations=num_iterations,
    )
