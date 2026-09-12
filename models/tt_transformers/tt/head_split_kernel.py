# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A wider QKV head split for SHORT prefill chunks, via ``ttnn.generic_op``.

``nlp_create_qkv_heads``'s interleaved program factory takes one work unit per
``(batch, seq_tile)`` pair -- ``num_blocks = shape[0] * shape[1] * shape[2] / TILE`` -- and
each unit owns the WHOLE ``qkv_width`` row, scattering it into q/k/v. At a 128-token prompt
that is FOUR units, so the op runs on four cores of a 110-core grid and takes 14.9 us/layer
to move ~0.8 MB, about 13 GB/s per core against the ~25 a core can issue.

The split is a pure tile permutation, and it decomposes into runs that are contiguous on
BOTH sides. Number the input tiles ``(s, c) -> s * ct + c``. One head's slice of one
sequence tile-row is ``dt`` consecutive input tiles, and in the output it is ``dt``
consecutive tiles as well (``head * st * dt + s * dt``). So a work unit of
``(seq_tile, head)`` is a contiguous read and a contiguous write -- which the STOCK unary
reader and writer kernels already do, given a start page and a count. No kernel source is
authored here; what changes is the work split, from ``st`` units to ``st * (n_q + 2*n_kv)``.

The three destinations are handled by three writer kernels on three disjoint core sets
(q's, k's, v's), each carrying its own tensor accessor, rather than by a writer that
branches per tile.

Returns ``None`` whenever the shape, dtype or layout is outside what this covers, and the
caller keeps ``ttnn.experimental.nlp_create_qkv_heads``.
"""

from __future__ import annotations

import ttnn

TILE = 32
# Reader kernel hard-codes cb_id_in0 = 0, so the writer is pointed at the same CB and the
# pair runs producer/consumer with no compute kernel in between.
_CB_INDEX = 0
_READER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp"
_WRITER = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"
# Page size per tile, which is what both the CB and the tensor's page addressing use. A
# block-float tile is its mantissa block plus a separate exponent section (1024 + 64 for
# bf8_b); the copy moves the page whole, so the sections need no special handling, but the
# SIZE has to be exact or reader and writer disagree about page boundaries.
_TILE_BYTES = {
    ttnn.bfloat16: 2 * TILE * TILE,
    ttnn.bfloat8_b: TILE * TILE + TILE * 2,
}


def _core_list(mesh_device, count):
    """The first ``count`` cores of the compute grid, row-major, or ``None`` if too few."""
    grid = mesh_device.compute_with_storage_grid_size()
    if count > grid.x * grid.y:
        return None
    return [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def create_qkv_heads(xqkv, *, num_heads, num_kv_heads, mesh_device, memory_config):
    """``(q, k, v)`` from a ``[1, 1, S, (n_q + 2*n_kv) * head_dim]`` prefill QKV, or ``None``."""
    shape = list(xqkv.shape)
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 1:
        return None
    if xqkv.layout != ttnn.TILE_LAYOUT or xqkv.dtype not in _TILE_BYTES:
        return None
    if xqkv.is_sharded() or memory_config.is_sharded():
        return None
    seq_len, qkv_width = int(shape[2]), int(shape[3])
    sections = num_heads + 2 * num_kv_heads
    if seq_len % TILE or qkv_width % sections:
        return None
    head_dim = qkv_width // sections
    if head_dim % TILE:
        return None

    st, dt, ct = seq_len // TILE, head_dim // TILE, qkv_width // TILE
    # (output_index, head, seq_tile) per unit, in q-then-k-then-v order so each output's
    # units are contiguous and can be handed to one writer over one core range.
    units = [
        (out_idx, head, s)
        for out_idx, n_heads in enumerate((num_heads, num_kv_heads, num_kv_heads))
        for head in range(n_heads)
        for s in range(st)
    ]
    cores = _core_list(mesh_device, len(units))
    if cores is None:
        return None

    outs = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, n_heads, seq_len, head_dim]),
            xqkv.dtype,
            ttnn.TILE_LAYOUT,
            mesh_device,
            memory_config,
        )
        for n_heads in (num_heads, num_kv_heads, num_kv_heads)
    ]
    # Column offset, in tiles, at which each output's heads start inside the QKV row.
    col_base = (0, num_heads * dt, (num_heads + num_kv_heads) * dt)

    page_bytes = _TILE_BYTES[xqkv.dtype]
    cb_descriptor = ttnn.CBDescriptor(
        total_size=4 * page_bytes,
        core_ranges=_core_range_set(cores),
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_INDEX, data_format=xqkv.dtype, page_size=page_bytes)
        ],
    )

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = [ttnn.RuntimeArgs() for _ in outs]
    writer_cores = [[] for _ in outs]
    src_addr = xqkv.buffer_address()
    for core, (out_idx, head, s) in zip(cores, units):
        reader_rt[core.x][core.y] = [src_addr, dt, s * ct + col_base[out_idx] + head * dt]
        writer_rt[out_idx][core.x][core.y] = [outs[out_idx].buffer_address(), dt, head * st * dt + s * dt]
        writer_cores[out_idx].append(core)

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=_core_range_set(cores),
            compile_time_args=ttnn.TensorAccessorArgs(xqkv).get_compile_time_args(),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
    ]
    for out_idx, out in enumerate(outs):
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_WRITER,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=_core_range_set(writer_cores[out_idx]),
                compile_time_args=[_CB_INDEX] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
                runtime_args=writer_rt[out_idx],
                config=ttnn.WriterConfigDescriptor(),
            )
        )

    ttnn.generic_op(
        [xqkv, *outs],
        ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb_descriptor]),
    )
    return outs[0], outs[1], outs[2]


def concat_heads(attn_out, *, mesh_device, memory_config):
    """``[1, H, S, D] -> [1, 1, S, H*D]``, or ``None`` if the shape is outside this path.

    The mirror of :func:`create_qkv_heads`, and starved the same way: ``nlp_concat_heads``
    also splits over ``(batch, seq_tile)``, so it runs a 128-token prompt on four cores and
    takes 9.7 us/layer. The inverse permutation decomposes the same way -- for one head and
    one sequence tile-row, ``dt`` consecutive input tiles land on ``dt`` consecutive output
    tiles -- so the same stock reader/writer pair covers it with ``st * H`` units.
    """
    shape = list(attn_out.shape)
    if len(shape) != 4 or shape[0] != 1:
        return None
    if attn_out.layout != ttnn.TILE_LAYOUT or attn_out.dtype not in _TILE_BYTES:
        return None
    if attn_out.is_sharded() or memory_config.is_sharded():
        return None
    n_heads, seq_len, head_dim = int(shape[1]), int(shape[2]), int(shape[3])
    if n_heads < 2 or seq_len % TILE or head_dim % TILE:
        return None

    st, dt = seq_len // TILE, head_dim // TILE
    units = [(head, s) for head in range(n_heads) for s in range(st)]
    cores = _core_list(mesh_device, len(units))
    if cores is None:
        return None

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, seq_len, n_heads * head_dim]),
        attn_out.dtype,
        ttnn.TILE_LAYOUT,
        mesh_device,
        memory_config,
    )

    page_bytes = _TILE_BYTES[attn_out.dtype]
    core_set = _core_range_set(cores)
    cb_descriptor = ttnn.CBDescriptor(
        total_size=4 * page_bytes,
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_INDEX, data_format=attn_out.dtype, page_size=page_bytes)
        ],
    )

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    src_addr, dst_addr = attn_out.buffer_address(), out.buffer_address()
    for core, (head, s) in zip(cores, units):
        reader_rt[core.x][core.y] = [src_addr, dt, head * st * dt + s * dt]
        writer_rt[core.x][core.y] = [dst_addr, dt, s * n_heads * dt + head * dt]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=ttnn.TensorAccessorArgs(attn_out).get_compile_time_args(),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[_CB_INDEX] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    ttnn.generic_op([attn_out, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb_descriptor]))
    return out
