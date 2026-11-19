# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only source transformations/plans; no TTNN import or default wiring.

Pass the already-selected cumulative_exp_scalar reader/compute sources to
stream_state_sources. The caller must use all three returned sources together.
This changes publication order, not recurrence arithmetic or state geometry.
"""


def _replace(source, before, after):
    assert source.count(before) == 1, (before, source.count(before))
    return source.replace(before, after, 1)


def stream_state_sources(reader, compute, writer):
    """Stream four BF16/FP32 state tiles per value-column to the writer.

    CB11 retains its existing capacity/format. Four matching reserve/push and
    wait/pop groups replace one head group. CB12 readout publication stays
    head-granular. Every state transfer completes before its CB slots retire.
    """
    compute = _replace(
        compute,
        """        updated.reserve_back(16); out.reserve_back(4);
        for(uint32_t n=0;n<4;n++) {
            tile_regs_acquire();""",
        """        out.reserve_back(4);
        for(uint32_t n=0;n<4;n++) {
            updated.reserve_back(4);
            tile_regs_acquire();""",
    )
    compute = _replace(
        compute,
        "for(uint32_t m=0;m<4;m++) pack_tile<true>(m,11,m*4+n);",
        "for(uint32_t m=0;m<4;m++) pack_tile<true>(m,11,m);",
    )
    compute = _replace(
        compute,
        """            tile_regs_release();
        }
        updated.push_back(16); out.push_back(4);""",
        """            tile_regs_release();
            updated.push_back(4);
        }
        out.push_back(4);""",
    )
    writer = _replace(
        writer,
        """        updated.wait_front(16); out.wait_front(4);
        // Waiting for output guarantees compute finished reading updated CB.
        for(uint32_t i=0;i<16;i++)
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(updated),s,STATE_BYTES,{.offset_bytes=i*STATE_BYTES},{.page_id=h*16+i});""",
        """        // Compute retains live state in DST, never rereads CB11. Each
        // published column is complete; input CB5 already holds the old head.
        for(uint32_t n=0;n<4;n++) {
            updated.wait_front(4);
            for(uint32_t m=0;m<4;m++)
                noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(updated),s,STATE_BYTES,
                    {.offset_bytes=m*STATE_BYTES},{.page_id=h*16+m*4+n});
            noc.async_write_barrier();
            updated.pop_front(4);
        }
        out.wait_front(4);""",
    )
    writer = _replace(
        writer,
        "noc.async_write_barrier(); updated.pop_front(16); out.pop_front(4);",
        "noc.async_write_barrier(); out.pop_front(4);",
    )
    return reader, compute, writer


def value_column_partition(batch, available_cores=110):
    """Plan only: contiguous independent (head,n) ranges, no device wiring.

    Returns (first_head, head_count, first_n, last_n_exclusive). The first and
    last masks apply only to their respective boundary heads. Every interior
    head owns all four value columns. No K-row split or cross-core reduction.
    """
    assert 1 <= batch <= 32 and available_cores >= 1
    work = batch * 12 * 4
    owners = min(work, available_cores)
    result = []
    start = 0
    for core in range(owners):
        count = work // owners + (core < work % owners)
        end = start + count
        first_head, first_n = divmod(start, 4)
        last_head = (end - 1) // 4
        result.append((first_head, last_head - first_head + 1, first_n, (end - 1) % 4 + 1))
        start = end
    assert start == work
    return result


def local_conv_sources(reader, compute, writer, *, poison_dead_padding=False):
    """Adapt frozen untilize-once convolution to 2560 local channels.

    Three chunks contain 1024/1024/512 BF16 elements. The last chunk transfers
    only 1024 bytes; its unused CB half cannot reach a useful output because
    this compute has only elementwise arithmetic. No math source is changed.
    """
    for source, expected in ((reader, 6), (writer, 2)):
        assert source.count(",20480)") == expected
    reader = reader.replace(",20480)", ",5120)")
    writer = writer.replace(",20480)", ",5120)")
    sources = []
    for source, transfers in ((reader, 6), (writer, 2)):
        source = _replace(source, "user=task/10,chunk=task%10", "user=task/3,chunk=task%3,bytes=(chunk==2?1024:2048)")
        assert source.count(",2048,") == transfers
        source = source.replace(",2048,", ",bytes,")
        sources.append(source)
    if poison_dead_padding:
        sources[0] = _replace(
            sources[0],
            "        for(uint32_t row=0;row<3;row++) {",
            """        if(chunk==2) for(uint32_t id=0;id<8;id++) {
            CircularBuffer cb(id);
            volatile tt_l1_ptr uint32_t* dead=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr()+1024);
            for(uint32_t i=0;i<256;i++) dead[i]=0x7fc17fc1u;
        }
        for(uint32_t row=0;row<3;row++) {""",
        )
    return sources[0], compute, sources[1]


def local_conv(token, history, taps, *, output=None, cores=None, poison_dead_padding=False, logical_batch=None):
    """Opt-in device wrapper; importing this module alone never imports TTNN.

    Parent hook: model._local_batched_conv = lambda q,h: local_conv(q,h,model.taps_rm).
    First compare convolution output AND in-place raw history bit-for-bit.
    """
    import ttnn

    from .optimized_decoder import (
        _conv107__CONV_RM_PIPE2_COMPUTE,
        _conv107__CONV_RM_PIPE2_READER,
        _conv107__CONV_RM_PIPE2_WRITER,
    )

    capacity = history.shape[0]
    batch = capacity if logical_batch is None else logical_batch
    if type(batch) is not int or not 1 <= batch <= capacity <= 32:
        raise ValueError(f"Convolution logical batch {batch!r} must be within state capacity {capacity}")
    assert tuple(history.shape) == (capacity, 3, 2560)
    assert tuple(token.shape) == (1, batch, 2560), "Candidate requires compact decode input"
    assert token.layout == ttnn.TILE_LAYOUT and history.layout == ttnn.ROW_MAJOR_LAYOUT
    assert len(taps) == 4
    for tap in taps:
        assert tuple(tap.shape) == (1, 1, 2560) and tap.layout == ttnn.ROW_MAJOR_LAYOUT
    raw_token = ttnn.to_layout(token, ttnn.ROW_MAJOR_LAYOUT)
    inputs = [raw_token, history, *taps]
    for tensor in inputs:
        assert tensor.dtype == ttnn.bfloat16 and tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    if output is None:
        output = ttnn.empty(
            (batch, 1, 2560),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=token.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (batch, 1, 2560) and output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.ROW_MAJOR_LAYOUT and output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert output.buffer_address() not in [t.buffer_address() for t in inputs]
    grid = token.device().compute_with_storage_grid_size()
    # The history allocation/address is unchanged. Each task owns one of
    # three channel chunks for user=task//3; its reads/writes address only
    # history rows user*3+[0,3). The final 512-channel chunk transfers 1024
    # bytes, so neither tail padding nor a prefix batch reaches another user.
    tasks = batch * 3
    ncores = min(tasks, cores or grid.x * grid.y)
    assert 1 <= ncores <= grid.x * grid.y
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    start = 0
    for i, coord in enumerate(coords):
        count = tasks // ncores + (i < tasks % ncores)
        rr[coord.x][coord.y] = [t.buffer_address() for t in inputs] + [start, count]
        wr[coord.x][coord.y] = [history.buffer_address(), output.buffer_address(), start, count]
        cr[coord.x][coord.y] = [count]
        start += count
    rct, wct = [], []
    for tensor in inputs:
        rct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    for tensor in (history, output):
        wct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())

    def kernel(source, config, args, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=args,
            config=config,
        )

    reader, compute, writer = local_conv_sources(
        _conv107__CONV_RM_PIPE2_READER,
        _conv107__CONV_RM_PIPE2_COMPUTE,
        _conv107__CONV_RM_PIPE2_WRITER,
        poison_dead_padding=poison_dead_padding,
    )
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    kernels = [
        kernel(reader, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(writer, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(compute, compute_config, cr, []),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in (*range(8), 9, 10)
    ]
    ttnn.generic_op(inputs + [output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output
