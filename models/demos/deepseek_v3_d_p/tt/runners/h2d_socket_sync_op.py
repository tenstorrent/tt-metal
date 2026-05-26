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


def h2d_socket_sync(service: ttnn.H2DStreamService, worker_cores: ttnn.CoreRange) -> ttnn.Tensor:
    """Wait for the next H2D transfer to land in `service.get_backing_tensor()`,
    copy it into a fresh device tensor, and ack the service core's consumed counter.

    Args:
        service: A persistent H2DStreamService constructed with `worker_cores` set
            to the same CoreRange passed here. Provides the data-ready semaphore,
            per-coord consumed counter, and per-coord service-core coordinates.
        worker_cores: Worker CoreRange — must match the `worker_cores` the service
            was constructed with. Each core runs one iteration of the
            wait → copy slice → ack protocol.

    Returns:
        A new ttnn.Tensor with the same per-shard spec as `service.get_backing_tensor()`,
        populated with the latest transfer's data. Independent of the backing tensor —
        the next `forward_to_tensor` call will not overwrite it.
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
    ]
    ct_args.extend(backing_acc_args)
    ct_args.extend(output_acc_args)

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

    # io_tensors order: output first (so generic_op returns it), backing second
    # so its buffer is reachable from the kernel via the address baked into CT args.
    return ttnn.generic_op([output, backing], mesh_program_descriptor)
