# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Worker-side handshake for `ttnn.H2DStreamService` exposed as a one-shot ttnn op.

Pairs with `models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/h2d_socket_sync.cpp`:
each worker core waits on the service's `data_ready_sem`, copies its slice of
the backing tensor into a freshly allocated output tensor, then acks the
service core's `consumed_counter`. Replaces `service.barrier()` for the common
pipeline pattern

    service.forward_to_tensor_bytes(data)
    synced = h2d_socket_sync(service)
    downstream = tt_emb(synced)
"""

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import get_tensor_accessor_args

_KERNEL_PATH = "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/h2d_socket_sync.cpp"
_SCRATCH_CB_INDEX = 0


def _worker_cores_in_range(core_range: ttnn.CoreRange) -> list[ttnn.CoreCoord]:
    """Enumerate worker cores in `core_range` in row-major order. Match
    HostInterface's CoreRange enumeration so per-worker page slices stay stable."""
    cores: list[ttnn.CoreCoord] = []
    for y in range(core_range.start.y, core_range.end.y + 1):
        for x in range(core_range.start.x, core_range.end.x + 1):
            cores.append(ttnn.CoreCoord(x, y))
    return cores


def h2d_socket_sync(
    service,
    worker_cores: ttnn.CoreRange,
    *,
    metadata_size_bytes: int = 0,
) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
    """Wait for the next transfer to land in `service.get_backing_tensor()`, copy it
    into a fresh device tensor, and ack the service core's consumed counter.

    Works for the receiver side of EITHER an `ttnn.H2DStreamService` or a
    `ttnn.D2DStreamServiceReceiver` — both expose the same receiver getters
    (get_backing_tensor / get_data_ready_sem_addr / get_consumed_counter_addr(coord) /
    get_service_core(coord) / get_metadata_addr), so this op is service-agnostic. Pair the
    D2D receiver with d2d_socket_push() on the upstream stage's sender.

    Args:
        service: A persistent H2DStreamService or D2DStreamServiceReceiver constructed with
            `worker_cores` set to the same CoreRange passed here. Provides the data-ready
            semaphore, per-coord consumed counter, and per-coord service-core coordinates.
        worker_cores: Worker CoreRange — must match the `worker_cores` the service
            was constructed with. Each core runs one iteration of the
            wait → copy slice → ack protocol.
        metadata_size_bytes: When > 0, must match the value passed to the service
            constructor. The kernel additionally copies the inline metadata
            multicast by the service core (lives at `service.get_metadata_addr()`
            in worker L1) into a fresh DRAM tensor; this function then returns
            `(tokens_tensor, metadata_tensor)` instead of just the tokens tensor.
            Must be a multiple of 4 bytes (we expose the metadata buffer as a
            uint32 tensor).

    Returns:
        When `metadata_size_bytes == 0`: a single ttnn.Tensor with the same
        per-shard spec as `service.get_backing_tensor()`.
        When `metadata_size_bytes > 0`: a tuple `(tokens_tensor, metadata_tensor)`
        where the metadata tensor has shape `[1, 1, 1, metadata_size_bytes // 4]`
        uint32 ROW_MAJOR DRAM, replicated across the mesh (each coord's worker
        writes the same metadata it received from its service core multicast).
    """
    backing = service.get_backing_tensor()
    mesh_device = backing.device()
    mesh_shape = mesh_device.shape

    # --- allocate output (same per-shard spec as backing) ---
    output = ttnn.allocate_tensor_on_device(
        backing.shape,
        backing.dtype,
        backing.layout,
        mesh_device,
        backing.memory_config(),
    )

    # --- (optional) allocate metadata output ---
    metadata_output = None
    if metadata_size_bytes > 0:
        assert (
            metadata_size_bytes % 4 == 0
        ), f"metadata_size_bytes must be a multiple of 4 (uint32-aligned), got {metadata_size_bytes}"
        # One page per coord, sized to exactly metadata_size_bytes. The kernel
        # does a single noc_async_write of `metadata_size_bytes` to page 0.
        metadata_output = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, metadata_size_bytes // 4]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

    # --- distribute pages across the worker cores in row-major order ---
    workers = _worker_cores_in_range(worker_cores)
    num_workers = len(workers)
    assert num_workers > 0, "h2d_socket_sync: worker_cores must contain at least one core"

    page_size = backing.buffer_page_size()
    num_pages = backing.buffer_num_pages()
    # Contiguous chunks; the last worker absorbs the remainder so we never drop pages.
    base_pages_per_worker = num_pages // num_workers
    remainder = num_pages % num_workers
    page_ranges: list[tuple[int, int]] = []
    cursor = 0
    for i in range(num_workers):
        n = base_pages_per_worker + (1 if i < remainder else 0)
        page_ranges.append((cursor, cursor + n))
        cursor += n
    assert cursor == num_pages, f"page distribution mismatch: covered {cursor}/{num_pages}"

    # --- CT args (uniform across mesh) ---
    backing_acc_args = get_tensor_accessor_args(backing)
    output_acc_args = get_tensor_accessor_args(output)
    ct_args = [
        service.get_data_ready_sem_addr(),
        backing.buffer_address(),
        output.buffer_address(),
        page_size,
        num_pages,
        _SCRATCH_CB_INDEX,
        metadata_size_bytes,  # 0 disables the metadata path in the kernel
        service.get_metadata_addr() if metadata_size_bytes > 0 else 0,
        metadata_output.buffer_address() if metadata_output is not None else 0,
    ]
    ct_args.extend(backing_acc_args)
    ct_args.extend(output_acc_args)
    if metadata_output is not None:
        ct_args.extend(get_tensor_accessor_args(metadata_output))

    # --- scratch CB (single slot, sized to one page) ---
    cb_descriptor = ttnn.CBDescriptor(
        total_size=page_size,
        core_ranges=ttnn.CoreRangeSet([worker_cores]),
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_SCRATCH_CB_INDEX,
                data_format=backing.dtype,
                page_size=page_size,
            )
        ],
    )

    # --- per-coord Program ---
    mesh_program_descriptor = ttnn.MeshProgramDescriptor()

    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)

            # Logical service core for this device → physical NoC coord.
            service_logical = service.get_service_core(coord)
            service_phys = mesh_device.worker_core_from_logical_core(service_logical)
            consumed_addr = service.get_consumed_counter_addr(coord)

            kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source=_KERNEL_PATH,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=ttnn.CoreRangeSet([worker_cores]),
                compile_time_args=ct_args,
                config=ttnn.WriterConfigDescriptor(),
            )

            program = ttnn.ProgramDescriptor(
                kernels=[kernel_descriptor],
                semaphores=[],
                cbs=[cb_descriptor],
            )

            # Per-worker RT args: ack target + page slice.
            for worker, (start_page, end_page) in zip(workers, page_ranges):
                program.kernels[0].runtime_args[worker.x][worker.y] = [
                    consumed_addr,
                    service_phys.x,
                    service_phys.y,
                    start_page,
                    end_page,
                ]

            mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

    # generic_op returns the LAST io_tensor (= the persistent service backing),
    # whose buffer the receiver kernel overwrites on the next H2D push. Returning
    # it would hand the caller a tensor that gets clobbered, so iter=N could see
    # chunk N+1's tokens. Return the fresh `output` (and `metadata_output`)
    # explicitly instead.
    io_tensors = [output, metadata_output, backing] if metadata_output is not None else [output, backing]
    ttnn.generic_op(io_tensors, mesh_program_descriptor)
    if metadata_output is not None:
        return output, metadata_output
    return output
