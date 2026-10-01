# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Minimal multi-process example of the D2DStreamService Python API, modeled on
# tests/ttnn/distributed/test_multi_mesh.py. It shows a consumer how to stand up a
# device-to-device streaming pair ACROSS PROCESSES from Python: process 0 builds the
# sender endpoint, process 1 builds the receiver endpoint, and constructing them
# performs the cross-process MeshSocket rendezvous over tt-fabric. Each side then
# reaches the backing tensor + the handshake addresses its worker kernels will use.
#
# SCOPE — host-side setup only. The per-iteration DATA PATH is intentionally NOT driven
# here: producing into the sender backing + incrementing data_ready_counter, the
# fabric-link lease (release/wait_for_fabric_links), and consuming the receiver backing
# + incrementing consumed_counter are all DEVICE worker-kernel work and can't be
# expressed in pure Python. For the full kernel-level integration (the relay/gate/lease
# loop end-to-end) see the C++ reference:
#   tests/ttnn/unit_tests/gtests/multiprocess/test_cross_process_d2d_stream_service.cpp
# This example covers: construct the endpoints (rendezvous), inspect the tensors /
# addresses your worker kernels bind to, and tear down cleanly.
#
# Run on a Galaxy (2 processes, one 4x4 mesh each):
#   python3 tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py
#   tt-run --rank-binding 4x4_multi_mesh_rank_binding.yaml \
#       --mpi-args "--allow-run-as-root --tag-output" \
#       python3 tests/ttnn/unit_tests/base_functionality/test_d2d_stream_service_multiprocess.py
#
# Mesh graph descriptor (same as test_multi_mesh.py):
#   tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_galaxy_split_4x4_multi_mesh.textproto

import ttnn

SENDER_RANK = 0
RECEIVER_RANK = 1


def run_d2d_multiprocess_example():
    # Route data over Ethernet between meshes; the distributed context auto-initializes
    # when the mesh device is opened (no explicit init needed).
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh_shape = ttnn.MeshShape(4, 4)
    device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    if not ttnn.distributed_context_is_initialized():
        raise ValueError("Distributed context not initialized")
    if int(ttnn.distributed_context_get_size()) != 2:
        raise ValueError("This example requires exactly 2 processes (sender + receiver)")
    rank = int(ttnn.distributed_context_get_rank())

    # One full tensor per transfer, replicated across the mesh: the service wires sender
    # coord (x, y) 1:1 to receiver coord (x, y). UINT32 / ROW_MAJOR / DRAM backing tensor.
    global_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, 32, 64]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    # The mapper is REQUIRED (unlike H2D there is no replicate-on-null default) and its
    # ownership transfers into the service; build a fresh one per endpoint. Here: replicate
    # the global tensor onto every device of the local mesh.
    placements = [ttnn.PlacementReplicate() for _ in range(device.shape.dims())]
    mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig(placements=placements))

    # The worker grid the service synchronizes with — i.e. where the consumer's
    # produce/consume op + handshake run. A single core here.
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))

    # Process 0 builds the SENDER, process 1 the RECEIVER. Each factory blocks in the
    # cross-process MeshSocket rendezvous until its peer endpoint is constructed, so both
    # processes must reach this point. The distributed context is taken from the current
    # world automatically (no context argument needed).
    if rank == SENDER_RANK:
        endpoint = ttnn.D2DStreamService.create_sender(
            sender_mesh=device,
            global_spec=global_spec,
            mapper=mapper,
            fifo_size_bytes=4096,
            sender_worker_cores=worker_cores,
            receiver_worker_cores=worker_cores,
            sender_rank=SENDER_RANK,
            receiver_rank=RECEIVER_RANK,
        )
        print(f"[rank {rank}] D2D sender ready; backing shape = {endpoint.get_backing_tensor().shape}")
        # Addresses a producing worker kernel binds to (per participating coord):
        #   - write the backing tensor, then atomic-inc data_ready_counter on the service
        #     core to tell the sender service to forward.
        #   - the standalone overwrite-gate op waits on consumed_sem before the next write.
        for coord in ttnn.MeshCoordinateRange(mesh_shape):
            assert endpoint.get_data_ready_counter_addr(coord) != 0
            _ = endpoint.get_service_core(coord)  # logical CoreCoord on this coord's device
        assert endpoint.get_consumed_sem_addr() != 0
    else:
        endpoint = ttnn.D2DStreamService.create_receiver(
            receiver_mesh=device,
            global_spec=global_spec,
            mapper=mapper,
            fifo_size_bytes=4096,
            sender_worker_cores=worker_cores,
            receiver_worker_cores=worker_cores,
            sender_rank=SENDER_RANK,
            receiver_rank=RECEIVER_RANK,
        )
        print(f"[rank {rank}] D2D receiver ready; backing shape = {endpoint.get_backing_tensor().shape}")
        # Addresses a consuming worker kernel binds to (per participating coord):
        #   - spin on data_ready_sem until a transfer lands, read the backing tensor, then
        #     atomic-inc consumed_counter on the service core to free the next receive.
        for coord in ttnn.MeshCoordinateRange(mesh_shape):
            assert endpoint.get_consumed_counter_addr(coord) != 0
            _ = endpoint.get_service_core(coord)
        assert endpoint.get_data_ready_sem_addr() != 0

    # Both endpoints are now resident (the rendezvous inside create_* already synchronized
    # the pair). A real workload would here enter its per-iter loop — driving the worker
    # handshake + the fabric-link lease from device kernels (see the C++ reference above).
    ttnn.distributed_context_barrier()
    print(f"[rank {rank}] D2D setup complete")

    # Tear down the service BEFORE closing the device: its destructor terminates the
    # persistent service kernel and releases the claimed service core, which needs the
    # device + command queue still alive.
    del endpoint
    ttnn.distributed_context_barrier()
    ttnn.close_device(device)


if __name__ == "__main__":
    run_d2d_multiprocess_example()
