# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Static exact value-column recurrence wrapper, CPU-safe on import.

No runtime Python source extraction or execution. Source transformations operate
only on explicit kernel strings, fail closed on drift, and preserve arithmetic.
Parent supplies `variant="cumulative_exp_scalar"` for the validated family.
"""


def replace_once(source, before, after):
    assert source.count(before) == 1, (source.count(before), before)
    return source.replace(before, after, 1)


def column_partition(batch, available_cores, cores=None, grouped_qk=True, *, grain=1):
    """Contiguous ownership in units of 1 or 2 complete value columns."""
    assert 1 <= batch <= 32 and available_cores >= 1 and grain in (1, 2)
    assert cores is None or cores >= 1
    units = batch * 48 // grain
    ncores = min(units, available_cores, cores or available_cores)
    result, start = [], 0
    for core in range(ncores):
        count = (units // ncores + (core < units % ncores)) * grain
        end = start + count
        first_head, first_column = divmod(start, 4)
        last_head = (end - 1) // 4
        result.append((first_head, last_head - first_head + 1, first_column, (end - 1) % 4 + 1))
        start = end
    assert start == batch * 48
    return bool(grouped_qk and batch >= 3), result


def column_sources(reader, compute, writer):
    """Restrict state reads, math columns and all state/output writes together."""
    reader = replace_once(
        reader,
        "    const uint32_t start=get_arg_val<uint32_t>(6),count=get_arg_val<uint32_t>(7);",
        "    const uint32_t start=get_arg_val<uint32_t>(6),count=get_arg_val<uint32_t>(7);\n"
        "    const uint32_t first_n=get_arg_val<uint32_t>(8),last_n=get_arg_val<uint32_t>(9);",
    )
    boundary = (
        "    for(uint32_t h=start;h<start+count;h++) {\n"
        "        const uint32_t n_begin=h==start?first_n:0,n_end=h+1==start+count?last_n:4;"
    )
    reader = replace_once(reader, "    for(uint32_t h=start;h<start+count;h++) {", boundary)
    reader = replace_once(
        reader,
        """        for(uint32_t i=0;i<16;i++)
            noc.async_read(s,state,STATE_BYTES,{.page_id=h*16+i},{.offset_bytes=i*STATE_BYTES});""",
        """        // Fetch only owned columns: another owner may already persist
        // an unowned column of this same head. Its data is never consumed here.
        for(uint32_t m=0;m<4;m++) for(uint32_t n=n_begin;n<n_end;n++) {
            const uint32_t i=m*4+n;
            noc.async_read(s,state,STATE_BYTES,{.page_id=h*16+i},{.offset_bytes=i*STATE_BYTES});
        }""",
    )
    compute = replace_once(
        compute,
        "    const uint32_t start_head=get_arg_val<uint32_t>(1);",
        "    const uint32_t start_head=get_arg_val<uint32_t>(1);\n"
        "    const uint32_t first_n=get_arg_val<uint32_t>(2),last_n=get_arg_val<uint32_t>(3);",
    )
    compute = replace_once(
        compute,
        "        for(uint32_t n=0;n<4;n++) {",
        "        const uint32_t n_begin=head==0?first_n:0,n_end=head+1==count?last_n:4;\n"
        "        for(uint32_t n=n_begin;n<n_end;n++) {",
    )
    writer = replace_once(
        writer,
        "    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);",
        "    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);\n"
        "    const uint32_t first_n=get_arg_val<uint32_t>(4),last_n=get_arg_val<uint32_t>(5);",
    )
    writer = replace_once(writer, "    for(uint32_t h=start;h<start+count;h++) {", boundary)
    writer = replace_once(
        writer,
        """        for(uint32_t i=0;i<16;i++)
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(updated),s,STATE_BYTES,{.offset_bytes=i*STATE_BYTES},{.page_id=h*16+i});""",
        """        for(uint32_t m=0;m<4;m++) for(uint32_t n=n_begin;n<n_end;n++) {
            const uint32_t i=m*4+n;
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(updated),s,STATE_BYTES,{.offset_bytes=i*STATE_BYTES},{.page_id=h*16+i});
        }""",
    )
    loop = "for(uint32_t i=0;i<4;i++) for(uint32_t face=0;face<2;face++)"
    assert writer.count(loop) == 2
    # Restrict both useful output and last-head padding; different final-head
    # fragments must never race by writing each other's output tile columns.
    writer = writer.replace(loop, "for(uint32_t i=n_begin;i<n_end;i++) for(uint32_t face=0;face<2;face++)")
    return reader, compute, writer


def column_recurrent(
    q,
    k,
    v,
    g,
    beta,
    state,
    *,
    normalized=False,
    packed_qkv=False,
    output=None,
    cores=None,
    grouped_qk=True,
    variant="baseline",
    source_overrides=None,
    grain=1,
    logical_batch=None,
):
    """Run exact value-column ownership for raw packed local12 recurrence.

    State is mutated in place; output is FP32 [1,B*12,128]. `grain=1` owns
    columns and `grain=2` owns pairs. `cores` is an optional worker cap.
    Default source assembly uses `variant` and the resolved grouped-QK bit.
    Explicit overrides must already be the complete column-aware R/C/W tuple.
    """
    import ttnn

    from .optimized_decoder import (
        _fused_delta_row_compact_fp32_compute_config,
        _fused_delta_row_GDN_FUSE_ACCUMULATOR,
        _fused_delta_row_GDN_FUSE_DECAY_PREDICT,
        _fused_delta_row_GDN_HOIST_COL_INIT,
        _fused_delta_row_GDN_HOIST_DELTA,
        _fused_delta_row_GDN_HOIST_SCALAR_INIT,
    )

    assert packed_qkv and not normalized, "Column recurrence requires raw packed QKV"
    row_major = q.layout == ttnn.ROW_MAJOR_LAYOUT
    assert not (row_major and normalized)
    assert not packed_qkv or row_major
    assert state.dtype in (ttnn.float32, ttnn.bfloat16)
    state_bf16 = state.dtype == ttnn.bfloat16
    state_bytes = 2048 if state_bf16 else 4096
    capacity = state.shape[0] // 12
    batch = capacity if logical_batch is None else logical_batch
    if type(batch) is not int or not 1 <= batch <= capacity <= 32:
        raise ValueError(f"Recurrence logical batch {batch!r} must be within state capacity {capacity}")
    assert tuple(state.shape) == (capacity * 12, 128, 128)
    assert q.dtype == k.dtype == (ttnn.float32 if normalized else ttnn.bfloat16)
    assert v.dtype == ttnn.bfloat16
    assert tuple(q.shape) == tuple(k.shape)
    if packed_qkv:
        assert tuple(q.shape) == tuple(k.shape) == tuple(v.shape) == (batch, 1, 2560)
        assert q.buffer_address() == k.buffer_address() == v.buffer_address()
    elif normalized:
        assert tuple(q.shape) == (batch, 4, 128)
    else:
        assert q.shape[0] == batch and q.shape[-1] == 512 and q.shape[1] in (1, 32)
    if not packed_qkv:
        assert v.shape[0] == batch and v.shape[-1] == 1536 and v.shape[1] in (1, 32)
    assert tuple(g.shape) == tuple(beta.shape) == (batch, 1, 12)
    assert tuple(g.padded_shape) == tuple(beta.padded_shape) == (batch, 32, 32)
    inputs = [q, k, v, g, beta, state]
    for tensor in inputs:
        assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    for tensor in (q, k, v):
        assert tensor.layout == (ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT)
        if row_major:
            assert tensor.shape[1] == 1
            assert tuple(tensor.padded_shape) == tuple(tensor.shape)
    for tensor in (g, beta, state):
        assert tensor.layout == ttnn.TILE_LAYOUT
    if output is None:
        output = ttnn.empty(
            (1, batch * 12, 128),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=state.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (1, batch * 12, 128)
    grid = state.device().compute_with_storage_grid_size()
    # Partition only logical prefix heads, retaining the full state allocation.
    # Reader/writer state pages are h*16+[0,16), with h < batch*12; inactive
    # users are unreachable. The writer's BH below is logical too: last-head
    # padding writes only the separate output tensor, never recurrent state.
    # column_sources restricts both useful and padding output writes to the
    # owning columns, so partitioned owners cannot race on the final head.
    group_qk, partition = column_partition(batch, grid.x * grid.y, cores, grouped_qk, grain=grain)
    if source_overrides is not None or variant != "baseline":
        assert packed_qkv and not normalized
    if source_overrides is None:
        from .multichip_decoder import _LOCAL12_READER
        from .optimized_decoder import _fused_delta_row_GDN_COMPACT_COMPUTE, _fused_delta_row_GDN_COMPACT_WRITER
        from .recurrent_variants import sources

        reader_source, compute_source = sources(
            _LOCAL12_READER, _fused_delta_row_GDN_COMPACT_COMPUTE, variant, grouped=group_qk
        )
        reader_source, compute_source, writer_source = column_sources(
            reader_source, compute_source, _fused_delta_row_GDN_COMPACT_WRITER
        )
    else:
        assert len(source_overrides) == 3
        reader_source, compute_source, writer_source = source_overrides
    ncores = len(partition)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for c, (start, count, first_n, last_n) in zip(coords, partition):
        rr[c.x][c.y] = [t.buffer_address() for t in inputs] + [start, count, first_n, last_n]
        wr[c.x][c.y] = [state.buffer_address(), output.buffer_address(), start, count, first_n, last_n]
        cr[c.x][c.y] = [count, start, first_n, last_n]
    # Raw q/k/v paddedS32 occupies one tile row per batch.
    rct = [
        int(normalized),
        int(g.dtype == ttnn.float32),
        int(beta.dtype == ttnn.float32),
        16,
        48,
        int(row_major),
        int(packed_qkv),
        state_bytes,
    ]
    for t in inputs:
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [state_bytes]
    for t in (state, output):
        wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct.append(batch * 12)

    def kernel(source, config, rt, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    kernels = [
        kernel(reader_source, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(writer_source, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(
            compute_source,
            _fused_delta_row_compact_fp32_compute_config(state_bf16),
            cr,
            [
                int(normalized),
                int(state_bf16),
                int(group_qk),
                int(_fused_delta_row_GDN_FUSE_ACCUMULATOR),
                int(_fused_delta_row_GDN_HOIST_DELTA),
                int(_fused_delta_row_GDN_HOIST_SCALAR_INIT),
                int(_fused_delta_row_GDN_HOIST_COL_INIT),
                int(_fused_delta_row_GDN_FUSE_DECAY_PREDICT),
            ],
        ),
    ]
    sizes = {
        0: 8,
        1: 8,
        2: 8,
        3: 2,
        4: 2,
        5: 32,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 32,
        12: 8,
        13: 1,
        14: 1,
        15: 1,
        16: 1,
        17: 4,
        18: 1,
    }
    cbs = []
    for i, n in sizes.items():
        is_state = state_bf16 and i in (5, 11)
        page_size = 2048 if is_state else 4096
        data_format = ttnn.bfloat16 if is_state else ttnn.float32
        cbs.append(
            ttnn.CBDescriptor(
                total_size=n * page_size,
                core_ranges=ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=data_format, page_size=page_size)
                ],
            )
        )
    ttnn.generic_op(inputs + [output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output
