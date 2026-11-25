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


def live_writer(writer, *, live_arg=6, offset_arg=None):
    """perf/multiuser: skip recurrent-state persistence for inactive users.

    Reads the decode position vector (int32 [32], -1 = inactive row) once and
    only persists a user's updated state when its position is >= 0. Output
    (readout) writes are unchanged; inactive rows' outputs are never consumed.
    This replaces the host-side clone/where/copy masking of the full state.
    ``offset_arg`` (speculative verify) reads row ``user+offset`` of the vector.
    """
    off = "+get_arg_val<uint32_t>(%d)" % offset_arg if offset_arg is not None else ""
    writer = replace_once(
        writer,
        "    Noc noc; CircularBuffer updated(11),out(12);",
        "    Noc noc; CircularBuffer updated(11),out(12);\n"
        "    constexpr auto la=TensorAccessorArgs<oa.next_compile_time_args_offset()+1>();\n"
        "    const auto lv=TensorAccessor(la,get_arg_val<uint32_t>(%d),128);\n"
        "    CircularBuffer livecb(19); livecb.reserve_back(1);\n"
        "    noc.async_read(lv,livecb,128,{.page_id=0,.offset_bytes=0},{.offset_bytes=0});\n"
        "    noc.async_read_barrier();\n"
        "    volatile tt_l1_ptr int32_t* live_pos=reinterpret_cast<volatile tt_l1_ptr int32_t*>(livecb.get_write_ptr());"
        % live_arg,
    )
    writer = replace_once(
        writer,
        "        for(uint32_t m=0;m<4;m++) for(uint32_t n=n_begin;n<n_end;n++) {\n"
        "            const uint32_t i=m*4+n;\n"
        "            noc.async_write(",
        "        if(live_pos[h/12%s]>=0) for(uint32_t m=0;m<4;m++) for(uint32_t n=n_begin;n<n_end;n++) {\n"
        "            const uint32_t i=m*4+n;\n"
        "            noc.async_write(" % off,
    )
    return writer


def spec_writer(writer):
    """perf/spec: output head offset (runtime arg 6) so W per-token calls fill one [1,B*W*12,128] readout."""
    writer = replace_once(
        writer,
        "    Noc noc; CircularBuffer updated(11),out(12);",
        "    Noc noc; CircularBuffer updated(11),out(12);\n    const uint32_t ooff=get_arg_val<uint32_t>(6);",
    )
    writer = replace_once(writer, "            const uint32_t row=h%32;", "            const uint32_t row=(h+ooff)%32;")
    assert writer.count("{.page_id=(h/32)*4+i,.offset_bytes=dest});") == 2
    writer = writer.replace(
        "{.page_id=(h/32)*4+i,.offset_bytes=dest});", "{.page_id=((h+ooff)/32)*4+i,.offset_bytes=dest});"
    )
    writer = replace_once(writer, "if(h+1==BH && BH%32) {", "if(h+1==BH && (BH+ooff)%32) {")
    writer = replace_once(writer, "for(uint32_t row=BH%32;row<32;row++)", "for(uint32_t row=(BH+ooff)%32;row<32;row++)")
    return live_writer(writer, live_arg=7, offset_arg=8)


def spec_reader(reader):
    """perf/spec: per-user source-slot selection and gate row offset.

    Runtime args 10..14: gate/beta user offset, selector (int32[32], slot index per
    logical row), slot1..slot3 state addresses (slot0 = the ordinary state input).
    """
    reader = replace_once(
        reader,
        "    Noc noc; CircularBuffer scratch(15);",
        "    Noc noc; CircularBuffer scratch(15);\n"
        "    const uint32_t goff=get_arg_val<uint32_t>(10);\n"
        "    constexpr auto sela=TensorAccessorArgs<sa.next_compile_time_args_offset()>();\n"
        "    const auto selv=TensorAccessor(sela,get_arg_val<uint32_t>(11),128);\n"
        "    const auto s1=TensorAccessor(sa,get_arg_val<uint32_t>(12),STATE_BYTES);\n"
        "    const auto s2=TensorAccessor(sa,get_arg_val<uint32_t>(13),STATE_BYTES);\n"
        "    const auto s3=TensorAccessor(sa,get_arg_val<uint32_t>(14),STATE_BYTES);\n"
        "    CircularBuffer selcb(20); selcb.reserve_back(1);\n"
        "    noc.async_read(selv,selcb,128,{.page_id=0,.offset_bytes=0},{.offset_bytes=0});\n"
        "    noc.async_read_barrier();\n"
        "    volatile tt_l1_ptr int32_t* selp=reinterpret_cast<volatile tt_l1_ptr int32_t*>(selcb.get_write_ptr());",
    )
    reader = replace_once(
        reader,
        "        const uint32_t user=h/12,hv=h%12,hk=hv/3;",
        "        const uint32_t user=h/12,hv=h%12,hk=hv/3;\n"
        "        const int32_t sl=selp[user];\n"
        "        const auto& src=sl==1?s1:(sl==2?s2:(sl==3?s3:s));",
    )
    reader = replace_once(reader, "noc.async_read(s,state,STATE_BYTES,", "noc.async_read(src,state,STATE_BYTES,")
    reader = replace_once(reader, "gate(g,user,hv,", "gate(g,user+goff,hv,")
    reader = replace_once(reader, "gate(b,user,hv,", "gate(b,user+goff,hv,")
    return reader


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
    live=None,
    spec=None,
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
    if spec is None:
        assert tuple(g.shape) == tuple(beta.shape) == (batch, 1, 12)
        assert tuple(g.padded_shape) == tuple(beta.padded_shape) == (batch, 32, 32)
    else:
        assert tuple(g.shape) == tuple(beta.shape) and tuple(g.shape)[1:] == (1, 12)
        assert tuple(g.padded_shape)[1:] == (32, 32) and spec["gate_offset"] + batch <= g.shape[0]
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
    if spec is None:
        assert tuple(output.shape) == (1, batch * 12, 128)
    else:
        assert output.shape[0] == 1 and output.shape[2] == 128
        assert spec["out_offset"] + batch * 12 <= output.shape[1]
        read_state = spec.get("read_state")
        read_state = state if read_state is None else read_state
        assert tuple(read_state.shape)[1:] == tuple(state.shape)[1:] and read_state.dtype == state.dtype
        assert read_state.shape[0] >= batch * 12
        inputs[5] = read_state
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
    if spec is not None:
        # perf/spec: dict(read_state, select, slots, gate_offset, out_offset, live_offset).
        # Reads state from spec["read_state"] (or per-row spec["slots"][select[row]]),
        # writes `state`, gates writes by live[row+live_offset], writes readout heads
        # at out_offset. `output` must then be the shared [1, B*W*12, 128] readout.
        assert live is not None and output is not None and source_overrides is None
        reader_source = spec_reader(reader_source)
        writer_source = spec_writer(writer_source)
    elif live is not None:
        writer_source = live_writer(writer_source)
    ncores = len(partition)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    extra_inputs = []
    if spec is not None:
        slots = list(spec["slots"]) + [state] * (4 - len(spec["slots"]))
        assert len(slots) == 4 and all(tuple(t.shape)[1:] == tuple(state.shape)[1:] for t in slots)
        reader_tail = [spec["gate_offset"], spec["select"].buffer_address()] + [t.buffer_address() for t in slots[1:]]
        writer_tail = [spec["out_offset"], live.buffer_address(), spec["live_offset"]]
        seen = {t.buffer_address() for t in inputs}
        for t in [spec["select"], state, *slots[1:]]:
            if t.buffer_address() not in seen:
                seen.add(t.buffer_address())
                extra_inputs.append(t)
    for c, (start, count, first_n, last_n) in zip(coords, partition):
        rr[c.x][c.y] = (
            [t.buffer_address() for t in inputs]
            + [start, count, first_n, last_n]
            + (reader_tail if spec is not None else [])
        )
        wr[c.x][c.y] = [state.buffer_address(), output.buffer_address(), start, count, first_n, last_n] + (
            writer_tail if spec is not None else ([live.buffer_address()] if live is not None else [])
        )
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
    if spec is not None:
        sel = spec["select"]
        assert sel.dtype == ttnn.int32 and sel.layout == ttnn.ROW_MAJOR_LAYOUT and tuple(sel.shape) == (32,)
        rct.extend(ttnn.TensorAccessorArgs(sel).get_compile_time_args())
    wct = [state_bytes]
    for t in (state, output):
        wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct.append(batch * 12)
    if live is not None:
        assert live.dtype == ttnn.int32 and live.layout == ttnn.ROW_MAJOR_LAYOUT and tuple(live.shape) == (32,)
        wct.extend(ttnn.TensorAccessorArgs(live).get_compile_time_args())

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
    if live is not None:
        sizes[19] = 1
    if spec is not None:
        sizes[20] = 1
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
    ttnn.generic_op(inputs + extra_inputs + [output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output
