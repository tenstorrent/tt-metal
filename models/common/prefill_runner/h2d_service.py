# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""H2D socket-service plumbing for the prefill runner — model-agnostic.

`build_h2d_service` constructs the persistent `ttnn.H2DStreamService` whose per-shard backing tensor
matches the runner's token-input layout; `h2d_socket_sync` is the worker-side handshake op that
drains each push. Both are generic; the only model-specific input is the `mapper_config` (how a token
push shards across the mesh), which the runner passes through from the active adapter.

NOTE: `h2d_socket_sync` runs a kernel that currently lives under `models/demos/deepseek_v3_b1`
(a cross-demo dependency that predates this package). Follow-up: relocate the kernel +
get_tensor_accessor_args to a shared micro_ops location so this `common/` module has no demo import.
"""

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import get_tensor_accessor_args

# Sync-op worker core. Single core suffices: the kernel only copies the
# backing tensor's pages into a fresh output, no per-core parallelism needed.
H2D_SYNC_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))

# Inline metadata payload — packed by the producer per-iter, surfaced by the
# kernel via h2d_socket_sync's optional metadata output. Matches the prefill
# scheduler's PrefillMetadata wire struct (12 bytes, 3 × uint32):
#   [0] slot_id        — which per-slot KV-cache buffer to write
#   [1] actual_start   — inclusive absolute KV pos of the first real token
#   [2] actual_end     — exclusive absolute KV pos past the last real token
# Trailing positions in the chunk past `actual_end` are PAD_ID. See
# include/tt_llm_engine/scheduler/prefill/prefill_metadata.hpp for the
# source-of-truth definition the scheduler builds.
H2D_METADATA_SIZE_BYTES = 12

_KERNEL_PATH = "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/h2d_socket_sync.cpp"
_SCRATCH_CB_INDEX = 0


def make_global_spec(mesh_shape: tuple, chunk_size: int) -> ttnn.TensorSpec:
    """Per-push input spec used by `build_h2d_service` to set the service's
    global tensor shape (the producer matches it on the host side). One push carries one
    chunk_size-token chunk. Shape `(sp_factor, 1, chunk_size // sp_factor)` uint32 ROW_MAJOR DRAM."""
    sp_factor = mesh_shape[0]
    isl_per_chip = chunk_size // sp_factor
    return ttnn.TensorSpec(
        shape=ttnn.Shape([sp_factor, 1, isl_per_chip]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def build_h2d_service(
    mesh_device: ttnn.MeshDevice,
    *,
    mesh_shape: tuple,
    chunk_size: int,
    mapper_config: ttnn.MeshMapperConfig,
    worker_cores: ttnn.CoreRange,
    metadata_size_bytes: int,
) -> ttnn.H2DStreamService:
    """Construct an H2DStreamService whose per-shard backing tensor matches
    what `prepare_prefill_input_tensor` would have produced. Each push carries one chunk_size-token
    chunk (chunked prefill streams one chunk per push), not the full sequence.

    Per-shard target: `(1, 1, chunk_size // sp_factor)` uint32 ROW_MAJOR DRAM.
    Achieved by setting global_spec.shape = `(sp_factor, 1, chunk_size // sp_factor)` and
    mapping `[Shard(0), Replicate]` on a `(sp, tp)` mesh — first axis of the
    tensor is sharded across mesh rows (sp), nothing else is split.
    """
    from loguru import logger

    sp_factor, tp_factor = mesh_shape
    assert chunk_size % sp_factor == 0, f"chunk_size={chunk_size} must be divisible by sp_factor={sp_factor}"
    isl_per_chip = chunk_size // sp_factor
    per_chip_bytes = isl_per_chip * 4  # uint32

    global_spec = make_global_spec(mesh_shape, chunk_size)
    mapper = ttnn.create_mesh_mapper(mesh_device, mapper_config)
    # worker_cores set so the service-core kernel multicasts a data-ready inc
    # after each transfer; h2d_socket_sync() waits on that on-device, which
    # avoids the host-side barrier() round-trip per iteration.
    # metadata_size_bytes set so the producer can ship per-iter control bytes
    # (slot_id, actual_start, actual_end) inline with the token push.
    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=8 * per_chip_bytes,  # 8 in-flight pages of headroom
        scratch_cb_size_bytes=per_chip_bytes,  # one page; service requires >= page_size
        mapper=mapper,
        worker_cores=worker_cores,
        metadata_size_bytes=metadata_size_bytes,
    )
    logger.info(
        f"[h2d] H2DStreamService built: global_shape=({sp_factor},1,{isl_per_chip}) "
        f"uint32 ROW_MAJOR DRAM, per_chip_bytes={per_chip_bytes}, worker_cores={worker_cores}"
    )
    return service


def _worker_cores_in_range(core_range: ttnn.CoreRange) -> list[ttnn.CoreCoord]:
    """Enumerate worker cores in `core_range` in row-major order. Match
    HostInterface's CoreRange enumeration so per-worker page slices stay stable."""
    cores: list[ttnn.CoreCoord] = []
    for y in range(core_range.start.y, core_range.end.y + 1):
        for x in range(core_range.start.x, core_range.end.x + 1):
            cores.append(ttnn.CoreCoord(x, y))
    return cores


def h2d_socket_sync(
    service: ttnn.H2DStreamService,
    worker_cores: ttnn.CoreRange,
    *,
    metadata_size_bytes: int = 0,
) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
    """Wait for the next H2D transfer to land in `service.get_backing_tensor()`,
    copy it into a fresh device tensor, and ack the service core's consumed counter.

    Args:
        service: A persistent H2DStreamService constructed with `worker_cores` set
            to the same CoreRange passed here. Provides the data-ready semaphore,
            per-coord consumed counter, and per-coord service-core coordinates.
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
