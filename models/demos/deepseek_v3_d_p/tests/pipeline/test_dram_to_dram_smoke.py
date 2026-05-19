# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pure-D2D DRAM→fabric→DRAM cross-mesh smoke — host out of the data path.

Single chip-pair, single cross-mesh socket. Ships one logical tensor of size
``num_pages × _PAGE_SIZE_BYTES`` uint32s from a DRAM-resident source on rank
0's chip (0, 0) to a DRAM-resident destination on rank 1's chip (0, 0), then
verifies the destination matches the source bit-exactly.

Why this exists
---------------
The companion ``test_cross_mesh_socket_perf`` is bottlenecked by PCIe H2D
throughput (it calls ``h2d_socket.write_tensor`` in the timed loop). For the
prefill pipeline design the activations live in DRAM on the sender, never
touch host, and land in DRAM on the receiver. This test exercises that path
end-to-end with the smallest possible scope: 1 chip-pair, 1 transfer, no
hot loop, no perf timing — just "does the wiring work and is the tensor
correct."

Kernels
-------
* ``kernels/dram_to_socket_sender.cpp`` — Tensix worker on the sender chip's
  core (0, 0). Walks pages of the DRAM source via ``TensorAccessor``, stages
  each in a local L1 CB, and pushes via single-link fabric writes into the
  cross-mesh MeshSocket data buffer.
* ``kernels/socket_to_dram_receiver.cpp`` — Tensix worker on the receiver
  chip's core (0, 0). Pulls each page from the socket FIFO and
  ``noc_async_write``s it into the DRAM destination via ``TensorAccessor``.

Known cross-host (2-galaxy) limitation
--------------------------------------
On 2-galaxy the receiver's ``fabric_socket_notify_sender_stateful`` ack
writes (inline NOC writes encoded in fabric packet headers) don't reach the
sender's ``bytes_acked`` counter — the data path works fine, but the credit
return path goes silent. Symptom: sender hangs in ``socket_barrier`` waiting
for the last ack. The data transfer itself is correct (verified bit-exact
end-to-end). Workaround: the sender kernel takes a ``wait_for_acks``
compile-time arg; we set it to 0 for 2-galaxy. Consequence: ``num_pages``
must fit inside the socket FIFO (no page recycling), so 2-galaxy is capped
at ``fifo_size / page_size`` pages until the ack path is fixed.

Once this smoke passes, scaling to all 32 chip-pairs in parallel is just
"add 31 more SocketConnections + 31 more program dispatches."
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import create_fabric_router_config

_FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_2D
_FABRIC_ROUTER_MAX_PAYLOAD_BYTES = 2048

_SENDER_KERNEL = "models/demos/deepseek_v3_d_p/tests/pipeline/kernels/dram_to_socket_sender.cpp"
_RECEIVER_KERNEL = "models/demos/deepseek_v3_d_p/tests/pipeline/kernels/socket_to_dram_receiver.cpp"

# Tensor sized so the per-page L1 footprint fits comfortably alongside the
# socket FIFO + packet header CB + kernel binary on a single Tensix core
# (~1.5 MB total L1 on Blackhole). Shape is (NUM_PAGES, ELEMS_PER_PAGE) so a
# ROW_MAJOR DRAM tensor's intrinsic page (one row) == our kernel's page,
# letting TensorAccessor.get_noc_addr(p) walk real pages rather than past
# the end of a single-row allocation.
#
# Per-topology page counts:
#   - 1-galaxy: 256 pages (2 MB total). socket_barrier works intra-galaxy.
#   - 2-galaxy: 4 pages (32 KB). The ack credit-return path is broken cross-
#     host, so wait_for_acks is off and num_pages must fit in the FIFO.
_ELEMS_PER_PAGE = 2048
_PAGE_SIZE_BYTES = _ELEMS_PER_PAGE * 4  # 8 KB
_FIFO_PAGES = 4  # FIFO sized to 4 pages (= 32 KB)

# Per-topology config. The 1-galaxy variant exercises the full intra-galaxy
# path; the 2-galaxy variant is the cross-host smoke and is FIFO-capped.
_TOPOLOGY_CONFIG = {
    "1galaxy": {"num_pages": 256, "wait_for_acks": True},
    "2galaxy": {"num_pages": _FIFO_PAGES, "wait_for_acks": False},
}

_DATA_CB_INDEX = 0
_PACKET_HEADER_CB_INDEX = 1


def _packet_header_cb(core_range_set):
    """Two-slot CB for fabric packet headers (data header + socket-notify header)."""
    packet_header_size = ttnn.get_tt_fabric_packet_header_size_bytes()
    return ttnn.CBDescriptor(
        total_size=2 * packet_header_size,
        core_ranges=core_range_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_PACKET_HEADER_CB_INDEX,
                data_format=ttnn.uint32,
                page_size=packet_header_size,
            )
        ],
    )


def _data_cb(core_range_set, page_size_bytes):
    """Single-page L1 CB used to stage each DRAM page before pushing it via fabric."""
    return ttnn.CBDescriptor(
        total_size=page_size_bytes,
        core_ranges=core_range_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_DATA_CB_INDEX,
                data_format=ttnn.uint32,
                page_size=page_size_bytes,
            )
        ],
    )


def _build_sender_program(
    mesh_device, mesh_socket, send_core, recv_core, source_tensor, page_size, num_pages, wait_for_acks
):
    core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(send_core.core_coord, send_core.core_coord)])

    tensor_accessor_args = ttnn.TensorAccessorArgs(source_tensor).get_compile_time_args()

    compile_time_args = [
        _DATA_CB_INDEX,
        _PACKET_HEADER_CB_INDEX,
        page_size,
        num_pages,
        1 if wait_for_acks else 0,
        *tensor_accessor_args,
    ]

    sender_kernel = ttnn.KernelDescriptor(
        kernel_source=_SENDER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range_set,
        compile_time_args=compile_time_args,
        defines=[("FABRIC_MAX_PACKET_SIZE", str(ttnn.get_tt_fabric_max_payload_size_bytes()))],
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[sender_kernel],
        semaphores=[],
        cbs=[_data_cb(core_range_set, page_size), _packet_header_cb(core_range_set)],
    )

    cx, cy = send_core.core_coord.x, send_core.core_coord.y
    program.kernels[0].runtime_args[cx][cy] = [source_tensor.buffer_address(), mesh_socket.get_config_buffer_address()]

    sender_fabric_node_id = mesh_device.get_fabric_node_id(send_core.device_coord)
    recv_fabric_node_id = mesh_socket.get_fabric_node_id(ttnn.SocketEndpoint.RECEIVER, recv_core.device_coord)
    fabric_args = ttnn.setup_fabric_connection(
        sender_fabric_node_id, recv_fabric_node_id, 0, program, send_core.core_coord
    )
    program.kernels[0].runtime_args[cx][cy].extend(fabric_args)

    return program


def _build_receiver_program(mesh_device, mesh_socket, send_core, recv_core, dest_tensor, page_size, num_pages):
    core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(recv_core.core_coord, recv_core.core_coord)])

    tensor_accessor_args = ttnn.TensorAccessorArgs(dest_tensor).get_compile_time_args()

    compile_time_args = [_PACKET_HEADER_CB_INDEX, page_size, num_pages, *tensor_accessor_args]

    receiver_kernel = ttnn.KernelDescriptor(
        kernel_source=_RECEIVER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range_set,
        compile_time_args=compile_time_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[receiver_kernel],
        semaphores=[],
        cbs=[_packet_header_cb(core_range_set)],
    )

    cx, cy = recv_core.core_coord.x, recv_core.core_coord.y
    program.kernels[0].runtime_args[cx][cy] = [mesh_socket.get_config_buffer_address(), dest_tensor.buffer_address()]

    recv_fabric_node_id = mesh_device.get_fabric_node_id(recv_core.device_coord)
    sender_fabric_node_id = mesh_socket.get_fabric_node_id(ttnn.SocketEndpoint.SENDER, send_core.device_coord)
    fabric_args = ttnn.setup_fabric_connection(
        recv_fabric_node_id, sender_fabric_node_id, 0, program, recv_core.core_coord
    )
    program.kernels[0].runtime_args[cx][cy].extend(fabric_args)

    return program


def _dispatch(mesh_device, program, device_coord):
    dummy_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([0, 0, 0, 0]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device
    )
    mesh_pd = ttnn.MeshProgramDescriptor()
    mesh_pd[ttnn.MeshCoordinateRange(device_coord, device_coord)] = program
    return ttnn.generic_op([dummy_tensor, dummy_tensor], mesh_pd)


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
def test_dram_to_dram_smoke(mesh_device):
    if not is_slow_dispatch():
        pytest.skip("Sockets require slow dispatch (set TT_METAL_SLOW_DISPATCH_MODE=1).")

    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 2:
        pytest.skip(f"This test runs with exactly 2 ranks; got num_procs={num_procs}")

    # Topology-aware config: 2-galaxy is FIFO-capped (see module docstring).
    mesh_shape = mesh_device.shape
    topology_id = "1galaxy" if (int(mesh_shape[0]), int(mesh_shape[1])) == (2, 4) else "2galaxy"
    cfg = _TOPOLOGY_CONFIG[topology_id]
    num_pages = cfg["num_pages"]
    wait_for_acks = cfg["wait_for_acks"]
    num_elems_per_tensor = num_pages * _ELEMS_PER_PAGE
    total_bytes = num_elems_per_tensor * 4

    fifo_size_bytes = _PAGE_SIZE_BYTES * _FIFO_PAGES

    my_mesh_id = mesh_device.get_system_mesh_id()
    peer_mesh_id = 1 if my_mesh_id == 0 else 0

    # Single chip-pair at coord (0, 0), single core (0, 0). Cross-mesh.
    sender_device_coord = ttnn.MeshCoordinate((0, 0))
    receiver_device_coord = ttnn.MeshCoordinate((0, 0))
    core_coord = ttnn.CoreCoord(0, 0)
    send_core = ttnn.MeshCoreCoord(sender_device_coord, core_coord)
    recv_core = ttnn.MeshCoreCoord(receiver_device_coord, core_coord)

    logger.info(
        f"[mesh_id={my_mesh_id}] DRAM smoke ({topology_id}): total={total_bytes} B, "
        f"page={_PAGE_SIZE_BYTES} B, num_pages={num_pages}, fifo={fifo_size_bytes} B, "
        f"wait_for_acks={wait_for_acks}"
    )

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)

    # Build the cross-mesh socket (mesh 0 sender ↔ mesh 1 receiver, both at (0,0)).
    socket_connection = ttnn.SocketConnection(send_core, recv_core)
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, fifo_size_bytes)
    socket_config = ttnn.SocketConfig(
        connections=[socket_connection],
        memory_config=socket_mem_config,
        sender_mesh_id=0,
        receiver_mesh_id=1,
    )
    mesh_socket = ttnn.MeshSocket(mesh_device, socket_config)

    # Pre-fill source DRAM with a deterministic arange pattern; pre-zero the
    # destination. Tensor shape is (NUM_PAGES, ELEMS_PER_PAGE) so that the
    # ROW_MAJOR per-row page equals the kernel's page_size (8 KB). Use
    # ReplicateTensorToMesh explicitly so read-back semantics are deterministic.
    if my_mesh_id == 0:
        torch_local = (
            torch.arange(num_elems_per_tensor, dtype=torch.int64).to(torch.int32).reshape(num_pages, _ELEMS_PER_PAGE)
        )
    else:
        torch_local = torch.zeros(num_pages, _ELEMS_PER_PAGE, dtype=torch.int32)
    local_tensor = ttnn.from_torch(
        torch_local,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info(f"[mesh_id={my_mesh_id}] tensors allocated; buffer_address=0x{local_tensor.buffer_address():x}")

    # Build the right program for this rank.
    if my_mesh_id == 0:
        program = _build_sender_program(
            mesh_device, mesh_socket, send_core, recv_core, local_tensor, _PAGE_SIZE_BYTES, num_pages, wait_for_acks
        )
        my_device_coord = sender_device_coord
    else:
        program = _build_receiver_program(
            mesh_device, mesh_socket, send_core, recv_core, local_tensor, _PAGE_SIZE_BYTES, num_pages
        )
        my_device_coord = receiver_device_coord

    logger.info(f"[mesh_id={my_mesh_id}] program built; barrier before dispatch")
    ttnn.distributed_context_barrier()

    _dispatch(mesh_device, program, my_device_coord)
    ttnn.synchronize_device(mesh_device)

    logger.info(f"[mesh_id={my_mesh_id}] dispatch complete; barrier before readback")
    ttnn.distributed_context_barrier()

    # Bit-exact check on the receiver. Only chip (0, 0) of the receiver mesh
    # actually got data; the rest still have the pre-kernel zeros. Use a
    # ConcatMeshToTensor composer to deterministically lay out per-chip shards
    # side-by-side; chip (0, 0) is the first shard.
    if my_mesh_id == 1:
        # Concat along the trailing dim — chip-(0, 0)'s rows sit at columns
        # 0.._ELEMS_PER_PAGE on the concatenated readback; flatten and take
        # the first _NUM_ELEMS_PER_TENSOR elements.
        readback = ttnn.to_torch(
            ttnn.from_device(local_tensor),
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
        ).to(torch.int32)
        # readback shape: (num_pages, num_chips * _ELEMS_PER_PAGE) ROW_MAJOR.
        # Chip (0, 0) is the first cols slice; flatten that to compare against arange.
        first_chip = readback[:, :_ELEMS_PER_PAGE].reshape(-1)[:num_elems_per_tensor]

        expected = torch.arange(num_elems_per_tensor, dtype=torch.int64).to(torch.int32)
        match = torch.equal(expected, first_chip)
        if not match:
            mismatches = (expected != first_chip).nonzero(as_tuple=True)[0]
            n_mismatch = mismatches.numel()
            first = mismatches[0].item() if n_mismatch else -1
            sample_expected = expected[first : first + 8].tolist() if n_mismatch else []
            sample_got = first_chip[first : first + 8].tolist() if n_mismatch else []
            assert match, (
                f"[mesh_id=1] DRAM→DRAM bit-exact mismatch: {n_mismatch} elements differ.\n"
                f"  first mismatch at index {first}\n"
                f"  expected (8 elems from there): {sample_expected}\n"
                f"  got      (8 elems from there): {sample_got}"
            )
        logger.info(f"[mesh_id=1] DRAM→DRAM bit-exact: PASS ({num_elems_per_tensor} uint32 elems = {total_bytes} B)")

    logger.info(f"[mesh_id={my_mesh_id}] smoke OK")
