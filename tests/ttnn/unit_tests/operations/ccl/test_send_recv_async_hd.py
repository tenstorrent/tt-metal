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


def _assert_pages_equal(expected, actual, context):
    assert torch.equal(expected, actual), (
        f"{context}: output mismatch.\n"
        f"Expected first row: {expected[0, :8].tolist()}\n"
        f"Got first row:      {actual[0, :8].tolist()}"
    )


def _assert_num_program_cache_entries(mesh_device, expected, context):
    actual = mesh_device.num_program_cache_entries()
    assert actual == expected, f"{context}: expected {expected} program cache entries, found {actual}"


# Socket-move cases for the program-cache tests: the first socket always sits on device (0, 0),
# core (0, 0); the second one moves either to another core or to another device.
_SOCKET_MOVE_PARAMS = pytest.mark.parametrize(
    "mesh_device, second_device_coord, second_core_coord",
    [
        pytest.param((1, 1), (0, 0), (1, 0), id="new_core"),
        pytest.param((1, 2), (0, 1), (0, 0), id="new_device"),
    ],
    indirect=["mesh_device"],
)


def _device_index(mesh_device, device_coord):
    return device_coord[0] * mesh_device.shape[1] + device_coord[1]


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


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("page_size_bytes, num_pages, fifo_size_bytes", [(64, 4, 256), (256, 4, 1024)])
def test_recv_async_h2d_program_cache(mesh_device, page_size_bytes, num_pages, fifo_size_bytes):
    """Second call hits the program cache and must write into the new output tensor.

    The first output tensor is kept alive so the second one gets a different address; a stale
    output binding would make the hit write into the first tensor instead.
    """
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()

    page_size_datums = page_size_bytes // 4
    tensor_shape = (num_pages, page_size_datums)
    socket_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
    h2d_socket = ttnn.H2DSocket(mesh_device, socket_core, ttnn.BufferType.L1, fifo_size_bytes, ttnn.H2DMode.HOST_PUSH)
    h2d_socket.set_page_size(page_size_bytes)

    def allocate_output():
        return ttnn.from_torch(
            torch.zeros(tensor_shape, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    num_datums = num_pages * page_size_datums
    torch_input_miss = torch.arange(0, num_datums, dtype=torch.int32).reshape(tensor_shape)
    torch_input_hit = torch.arange(num_datums, 2 * num_datums, dtype=torch.int32).reshape(tensor_shape)

    output_miss = allocate_output()
    ttnn.experimental.recv_async_h2d(output_miss, h2d_socket)
    h2d_socket.write_tensor(torch_input_miss)
    ttnn.synchronize_device(mesh_device)
    _assert_pages_equal(torch_input_miss, ttnn.to_torch(output_miss).to(torch.int32), "recv_async_h2d miss")
    num_cache_entries = mesh_device.num_program_cache_entries()

    output_hit = allocate_output()
    ttnn.experimental.recv_async_h2d(output_hit, h2d_socket)
    h2d_socket.write_tensor(torch_input_hit)
    ttnn.synchronize_device(mesh_device)

    _assert_num_program_cache_entries(
        mesh_device, num_cache_entries, "recv_async_h2d second call with the same socket must hit the cache"
    )
    _assert_pages_equal(torch_input_hit, ttnn.to_torch(output_hit).to(torch.int32), "recv_async_h2d hit")
    # A stale output binding on the hit would have overwritten the miss output.
    _assert_pages_equal(
        torch_input_miss, ttnn.to_torch(output_miss).to(torch.int32), "recv_async_h2d miss output after hit"
    )


@skip_for_wormhole_b0("This test is for blackhole")
@_SOCKET_MOVE_PARAMS
def test_recv_async_h2d_program_cache_new_socket_core(mesh_device, second_device_coord, second_core_coord):
    """A socket on another core or device must miss the program cache and run there.

    The second socket is created after the first one is freed, so it reuses the first socket's
    config buffer address; only its active core differs. A key without the active core would hit
    and dispatch the cached program on the first socket's core.
    """
    page_size_bytes, num_pages, fifo_size_bytes = 64, 4, 256
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()

    page_size_datums = page_size_bytes // 4
    tensor_shape = (num_pages, page_size_datums)
    first_device_coord, first_core_coord = (0, 0), (0, 0)

    def make_socket(device_coord, core_coord):
        socket_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(*device_coord), ttnn.CoreCoord(*core_coord))
        socket = ttnn.H2DSocket(mesh_device, socket_core, ttnn.BufferType.L1, fifo_size_bytes, ttnn.H2DMode.HOST_PUSH)
        socket.set_page_size(page_size_bytes)
        return socket

    def allocate_output():
        return ttnn.from_torch(
            torch.zeros(tensor_shape, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def read_device_copies(output_tensor):
        return [ttnn.to_torch(t).to(torch.int32) for t in ttnn.get_device_tensors(output_tensor)]

    num_datums = num_pages * page_size_datums
    torch_input_miss = torch.arange(0, num_datums, dtype=torch.int32).reshape(tensor_shape)
    torch_input_hit = torch.arange(num_datums, 2 * num_datums, dtype=torch.int32).reshape(tensor_shape)
    torch_zeros = torch.zeros(tensor_shape, dtype=torch.int32)

    # Both outputs are allocated before the first socket so the second socket can take over the
    # L1 the first one frees.
    output_miss = allocate_output()
    output_hit = allocate_output()

    first_socket = make_socket(first_device_coord, first_core_coord)
    first_config_address = first_socket.get_config_buffer_address()
    ttnn.experimental.recv_async_h2d(output_miss, first_socket)
    first_socket.write_tensor(torch_input_miss)
    ttnn.synchronize_device(mesh_device)
    first_index = _device_index(mesh_device, first_device_coord)
    _assert_pages_equal(torch_input_miss, read_device_copies(output_miss)[first_index], "recv_async_h2d miss")
    num_cache_entries = mesh_device.num_program_cache_entries()
    del first_socket

    second_socket = make_socket(second_device_coord, second_core_coord)
    assert second_socket.get_config_buffer_address() == first_config_address, (
        "test precondition: the second H2DSocket must reuse the first socket's config buffer address "
        f"({first_config_address:#x}) so that only the active core distinguishes the two calls, "
        f"got {second_socket.get_config_buffer_address():#x}"
    )

    ttnn.experimental.recv_async_h2d(output_hit, second_socket)
    # Checked before the host write: a stale hit runs on the first socket's core and never completes.
    _assert_num_program_cache_entries(
        mesh_device, num_cache_entries + 1, "recv_async_h2d with a socket on a new core must miss the cache"
    )
    second_socket.write_tensor(torch_input_hit)
    ttnn.synchronize_device(mesh_device)

    second_index = _device_index(mesh_device, second_device_coord)
    for index, device_copy in enumerate(read_device_copies(output_hit)):
        expected = torch_input_hit if index == second_index else torch_zeros
        _assert_pages_equal(expected, device_copy, f"recv_async_h2d second socket, device index {index}")
    _assert_pages_equal(
        torch_input_miss, read_device_copies(output_miss)[first_index], "recv_async_h2d miss output after second call"
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


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("page_size_bytes, num_pages, fifo_size_bytes", [(64, 4, 256), (256, 4, 1024)])
def test_send_async_d2h_program_cache(mesh_device, page_size_bytes, num_pages, fifo_size_bytes):
    """Second call hits the program cache and must stream the new input tensor.

    The first input tensor is kept alive so the second one gets a different address; a stale
    input binding would make the hit stream the first tensor's data instead.
    """
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()

    page_size_datums = page_size_bytes // 4
    tensor_shape = (num_pages, page_size_datums)
    socket_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
    d2h_socket = ttnn.D2HSocket(mesh_device, socket_core, fifo_size_bytes)
    d2h_socket.set_page_size(page_size_bytes)

    def send_and_read(torch_input):
        input_tensor = ttnn.from_torch(
            torch_input,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.experimental.send_async_d2h(input_tensor, d2h_socket)
        result = torch.zeros(tensor_shape, dtype=torch.uint32)
        d2h_socket.read_tensor(result)
        ttnn.synchronize_device(mesh_device)
        return input_tensor, result.to(torch.int32)

    num_datums = num_pages * page_size_datums
    torch_input_miss = torch.arange(0, num_datums, dtype=torch.int32).reshape(tensor_shape)
    torch_input_hit = torch.arange(num_datums, 2 * num_datums, dtype=torch.int32).reshape(tensor_shape)

    input_miss, result_miss = send_and_read(torch_input_miss)
    _assert_pages_equal(torch_input_miss, result_miss, "send_async_d2h miss")
    num_cache_entries = mesh_device.num_program_cache_entries()

    input_hit, result_hit = send_and_read(torch_input_hit)
    _assert_num_program_cache_entries(
        mesh_device, num_cache_entries, "send_async_d2h second call with the same socket must hit the cache"
    )
    _assert_pages_equal(torch_input_hit, result_hit, "send_async_d2h hit")
    # Both inputs stay alive until here so the hit could not reuse the miss input's address.
    assert (
        input_miss.is_allocated() and input_hit.is_allocated()
    ), "send_async_d2h: both input tensors must stay allocated so the hit cannot reuse the miss input's address"


@skip_for_wormhole_b0("This test is for blackhole")
@_SOCKET_MOVE_PARAMS
def test_send_async_d2h_program_cache_new_socket_core(mesh_device, second_device_coord, second_core_coord):
    """A socket on another core or device must miss the program cache and stream from there.

    The second socket is created after the first one is freed, so it reuses the first socket's
    config buffer address; only its active core differs. A key without the active core would hit
    and dispatch the cached program on the first socket's core. Every device holds different
    input data, so the host result also identifies the device that streamed it.
    """
    page_size_bytes, num_pages, fifo_size_bytes = 64, 4, 256
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()

    page_size_datums = page_size_bytes // 4
    tensor_shape = (num_pages, page_size_datums)
    num_devices = mesh_device.get_num_devices()
    first_device_coord, first_core_coord = (0, 0), (0, 0)

    def make_socket(device_coord, core_coord):
        socket_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(*device_coord), ttnn.CoreCoord(*core_coord))
        socket = ttnn.D2HSocket(mesh_device, socket_core, fifo_size_bytes)
        socket.set_page_size(page_size_bytes)
        return socket

    def make_input(start):
        # One (num_pages, page_size_datums) slice per device, each with distinct data.
        torch_input = torch.arange(
            start, start + num_devices * num_pages * page_size_datums, dtype=torch.int32
        ).reshape(num_devices * num_pages, page_size_datums)
        input_tensor = ttnn.from_torch(
            torch_input,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        return input_tensor, torch.chunk(torch_input, num_devices, dim=0)

    def read_from(socket):
        result = torch.zeros(tensor_shape, dtype=torch.uint32)
        socket.read_tensor(result)
        ttnn.synchronize_device(mesh_device)
        return result.to(torch.int32)

    # Both inputs are allocated before the first socket so the second socket can take over the
    # L1 the first one frees.
    input_miss, torch_slices_miss = make_input(0)
    input_hit, torch_slices_hit = make_input(num_devices * num_pages * page_size_datums)

    first_socket = make_socket(first_device_coord, first_core_coord)
    first_config_address = first_socket.get_config_buffer_address()
    ttnn.experimental.send_async_d2h(input_miss, first_socket)
    result_miss = read_from(first_socket)
    _assert_pages_equal(
        torch_slices_miss[_device_index(mesh_device, first_device_coord)], result_miss, "send_async_d2h miss"
    )
    num_cache_entries = mesh_device.num_program_cache_entries()
    del first_socket

    second_socket = make_socket(second_device_coord, second_core_coord)
    assert second_socket.get_config_buffer_address() == first_config_address, (
        "test precondition: the second D2HSocket must reuse the first socket's config buffer address "
        f"({first_config_address:#x}) so that only the active core distinguishes the two calls, "
        f"got {second_socket.get_config_buffer_address():#x}"
    )

    ttnn.experimental.send_async_d2h(input_hit, second_socket)
    # Checked before the host read: a stale hit runs on the first socket's core and never completes.
    _assert_num_program_cache_entries(
        mesh_device, num_cache_entries + 1, "send_async_d2h with a socket on a new core must miss the cache"
    )
    result_hit = read_from(second_socket)
    _assert_pages_equal(
        torch_slices_hit[_device_index(mesh_device, second_device_coord)], result_hit, "send_async_d2h second socket"
    )
    assert input_miss.is_allocated() and input_hit.is_allocated(), (
        "send_async_d2h: both input tensors must stay allocated so the second call cannot reuse the first "
        "input's address"
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
