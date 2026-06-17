# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Worker-side producer for `ttnn.D2DStreamServiceSender` exposed as a one-shot ttnn op.

The inverse of `h2d_socket_sync`: it copies an activation tensor into the sender's
backing tensor, writes inline metadata to the sender service core, then inc's the
service's data_ready_counter and RETURNS — it does not wait for the forward. The
sender forwards once it has num_workers acks AND the host grants the fabric lease
(`sender.release_fabric_links()`), which the caller invokes after this op finishes.
Pairs with the kernel `kernels/d2d_socket_push.cpp` and, on the downstream stage, with
`h2d_socket_sync` driven against the matching `D2DStreamServiceReceiver`. Per-chunk
pattern (host drives the lease cadence):

    sender.wait_for_fabric_links()                                   # reclaim (prev forward drained)
    d2d_socket_push(sender, activation, metadata=[slot, start, end, is_last])  # push, no wait
    sender.release_fabric_links()                                    # grant -> forward this chunk
    ... downstream stage ...
    act, meta = h2d_socket_sync(receiver, worker_cores, metadata_size_bytes=16)
"""

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import get_tensor_accessor_args

_KERNEL_PATH = "models/demos/deepseek_v3_d_p/tt/runners/kernels/d2d_socket_push.cpp"
_SCRATCH_CB_INDEX = 0


def _worker_cores_in_range(core_range: ttnn.CoreRange) -> list[ttnn.CoreCoord]:
    """Row-major enumeration; matches the service's CoreRange order so per-worker page
    slices line up with the backing tensor's page layout."""
    cores: list[ttnn.CoreCoord] = []
    for y in range(core_range.start.y, core_range.end.y + 1):
        for x in range(core_range.start.x, core_range.end.x + 1):
            cores.append(ttnn.CoreCoord(x, y))
    return cores


def d2d_socket_push(
    sender,
    activation: ttnn.Tensor,
    worker_cores: ttnn.CoreRange,
    *,
    metadata: list[int] | None = None,
) -> None:
    """Push `activation` to the downstream stage via `sender` (a D2DStreamServiceSender),
    shipping `metadata` inline. Blocks until the transfer drains over fabric.

    Args:
        sender: persistent D2DStreamServiceSender constructed with `worker_cores` set to
            the same CoreRange passed here. Provides the per-coord data-ready counter,
            consumed semaphore, service-core coords, and metadata L1 buffer.
        activation: the stage's output hidden state. MUST have the same per-shard spec as
            `sender.get_backing_tensor()` (same shape/dtype/layout/memory_config) — the op
            copies it page-for-page into the backing tensor.
        worker_cores: must match the `sender_worker_cores` the service was built with.
        metadata: uint32 words shipped inline (e.g. [slot_id, actual_start, actual_end,
            is_last]). len*4 must equal the service's configured metadata_size_bytes. The
            designated worker (page slice starting at 0) writes them to the service core.
            None / [] = no metadata (service must have metadata_size_bytes == 0).
    """
    backing = sender.get_backing_tensor()
    mesh_device = backing.device()
    mesh_shape = mesh_device.shape

    assert activation.shape == backing.shape and activation.dtype == backing.dtype, (
        f"d2d_socket_push: activation spec {activation.shape}/{activation.dtype} must match the sender "
        f"backing tensor {backing.shape}/{backing.dtype}"
    )

    metadata = metadata or []
    metadata_size_bytes = len(metadata) * 4

    # --- distribute pages across the worker cores in row-major order (same split the
    #     downstream receiver uses, so page p maps to the same logical element both sides) ---
    workers = _worker_cores_in_range(worker_cores)
    num_workers = len(workers)
    assert num_workers > 0, "d2d_socket_push: worker_cores must contain at least one core"

    page_size = backing.buffer_page_size()
    num_pages = backing.buffer_num_pages()
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
    ct_args = [
        backing.buffer_address(),
        activation.buffer_address(),
        page_size,
        num_pages,
        _SCRATCH_CB_INDEX,
        metadata_size_bytes,  # 0 disables the metadata path in the kernel
    ]
    ct_args.extend(get_tensor_accessor_args(backing))
    ct_args.extend(get_tensor_accessor_args(activation))

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

    mesh_program_descriptor = ttnn.MeshProgramDescriptor()
    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            service_logical = sender.get_service_core(coord)
            service_phys = mesh_device.worker_core_from_logical_core(service_logical)
            data_ready_addr = sender.get_data_ready_counter_addr(coord)
            metadata_l1_addr = sender.get_metadata_addr(coord) if metadata_size_bytes > 0 else 0

            kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source=_KERNEL_PATH,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=ttnn.CoreRangeSet([worker_cores]),
                compile_time_args=ct_args,
                config=ttnn.WriterConfigDescriptor(),
            )
            program = ttnn.ProgramDescriptor(kernels=[kernel_descriptor], semaphores=[], cbs=[cb_descriptor])

            for worker, (start_page, end_page) in zip(workers, page_ranges):
                is_writer = 1 if start_page == 0 else 0
                rt = [
                    data_ready_addr,
                    service_phys.x,
                    service_phys.y,
                    start_page,
                    end_page,
                    is_writer,
                    metadata_l1_addr,
                ]
                if is_writer and metadata_size_bytes > 0:
                    rt.extend(int(m) & 0xFFFFFFFF for m in metadata)  # only the writer ships the blob
                program.kernels[0].runtime_args[worker.x][worker.y] = rt

            mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

    # The op writes into the persistent sender backing tensor (and signals the service); it
    # returns nothing — the downstream stage reads the data after the service forwards it.
    ttnn.generic_op([activation, backing], mesh_program_descriptor)
