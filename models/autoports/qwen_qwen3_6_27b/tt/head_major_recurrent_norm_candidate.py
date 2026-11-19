# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in coherent recurrence -> head-major readout -> fused norm/gate.

Only movement/ownership changes: recurrence's reader, compute, state stores,
CBs and head partition are inherited; norm/gate COMPUTE and WRITER are imported
unchanged. No runtime source-file/AST reading, monkey-patching, layout restore,
state arithmetic change or device import at module load.

Head-major output is FP32[12,B,128], padded[12,32,128]. Each (user,head)
recurrence owner writes its own logical row. The last user of each head owns
all that head's padding. B1 writes all four reduced tiles directly. The native
REDUCE_COL default pack mask passes only row0 in faces0/1 and zeros all other
rows/faces (Blackhole llk_pack_common.h:212). B>1 zero-padding copies use the
fully masked face2, never a live source row. Thus every output byte has exactly
one owner without an extra clear kernel, overlap or cross-core synchronization.

The consumer reads four whole FP32 tiles/head directly and coalesces the same
gate-padding zero bytes into contiguous per-face spans. Its native ordered
RMS and all four BF16 boundaries remain literal. B1 expands readout allocation
from16KiB to192KiB; B32 stays192KiB. The complete chain must be timed separately
at each batch; this source-only rationale establishes no speed or exactness.
"""

import math
import struct
from .local_norm_gate_candidate import CB_LAYOUT, READER, COMPUTE, WRITER, task_partition


def _once(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new, 1)


def head_major_writer_source(writer):
    """Retain state movement and queue barriers; replace only readout writes."""
    begin = "        for(uint32_t i=0;i<4;i++) for(uint32_t face=0;face<2;face++) {"
    end = "        noc.async_write_barrier(); updated.pop_front(16); out.pop_front(4);"
    assert writer.count(begin) == writer.count(end) == 1
    prefix, rest = writer.split(begin, 1)
    _, suffix = rest.split(end, 1)
    body = r"""        constexpr uint32_t B=BH/12;
        const uint32_t user=h/12,head=h%12;
        if constexpr(B==1) {
            // REDUCE_COL pack mask already zeroes every non-row0 element.
            for(uint32_t i=0;i<4;i++)
                noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),o,4096,
                    {.offset_bytes=i*4096},{.page_id=head*4+i});
        } else {
            for(uint32_t i=0;i<4;i++) for(uint32_t face=0;face<2;face++) {
                const uint32_t dest=((user/16)*512+face*256+(user%16)*16)*4;
                noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),o,64,
                    {.offset_bytes=i*4096+face*1024},
                    {.page_id=head*4+i,.offset_bytes=dest});
            }
            if(user+1==B && B<32) {
                // A complete lower face is zero by the native reduce pack mask.
                // Last user owns padding; no full-tile clear may race live rows.
                for(uint32_t row=B;row<32;) {
                    const uint32_t limit=(row/16+1)*16;
                    for(uint32_t i=0;i<4;i++) for(uint32_t face=0;face<2;face++) {
                        const uint32_t dest=((row/16)*512+face*256+(row%16)*16)*4;
                        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),o,(limit-row)*64,
                            {.offset_bytes=i*4096+2048},
                            {.page_id=head*4+i,.offset_bytes=dest});
                    }
                    row=limit;
                }
            }
        }
"""
    return prefix + body + end + suffix


def head_major_norm_reader_source(reader):
    """Replace head-strided64B gathers with four contiguous tile reads."""
    reader = _once(
        reader,
        "        // Initialize padding before ordinary reads; BH zero uses read-loopback.\n        noc.async_write_zeros(raw,4*4096);noc.write_zeros_l1_barrier();\n",
        "",
    )
    old = """            for(uint32_t user=0;user<B;user++) for(uint32_t face=0;face<2;face++) {
                const uint32_t row=user*12+head;
                const uint32_t src=(row%32/16)*2048+face*1024+(row%16)*64;
                const uint32_t dst=j*4096+(user/16)*2048+face*1024+(user%16)*64;
                noc.async_read(r,raw,64,{.page_id=(row/32)*4+j,.offset_bytes=src},{.offset_bytes=dst});
            }
"""
    reader = _once(reader, old, "            noc.async_read(r,raw,4096,{.page_id=head*4+j},{.offset_bytes=j*4096});\n")
    # Coalesce exactly the old gate-padding byte set, especially important at B1.
    old = """        for(uint32_t j=0;j<4;j++) for(uint32_t user=B;user<32;user++) for(uint32_t face=0;face<2;face++)
            noc.async_write_zeros(gate,32,{.offset_bytes=j*2048+(user/16)*1024+face*512+(user%16)*32});
"""
    new = """        for(uint32_t j=0;j<4;j++) for(uint32_t half=0;half<2;half++) {
            const uint32_t first=B>half*16?B:half*16,end=(half+1)*16;
            if(first<end) for(uint32_t face=0;face<2;face++)
                noc.async_write_zeros(gate,(end-first)*32,
                    {.offset_bytes=j*2048+half*1024+face*512+(first%16)*32});
        }
"""
    return _once(reader, old, new)


def local12_recurrent_head_major(
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
):
    """Accept raw RM[B,1,C] directly, or the original TILE interface."""
    import ttnn
    from .multichip_decoder import _LOCAL12_READER, _local12_head_partition
    from .optimized_decoder import (
        _fused_delta_row_GDN_COMPACT_COMPUTE,
        _fused_delta_row_GDN_COMPACT_WRITER,
        _fused_delta_row_compact_fp32_compute_config,
        _fused_delta_row_GDN_FUSE_ACCUMULATOR,
        _fused_delta_row_GDN_HOIST_DELTA,
        _fused_delta_row_GDN_HOIST_SCALAR_INIT,
        _fused_delta_row_GDN_HOIST_COL_INIT,
        _fused_delta_row_GDN_FUSE_DECAY_PREDICT,
    )

    assert packed_qkv and not normalized, "Candidate scope is raw packed QKV"
    row_major = q.layout == ttnn.ROW_MAJOR_LAYOUT
    assert not (row_major and normalized)
    assert not packed_qkv or row_major
    assert state.dtype in (ttnn.float32, ttnn.bfloat16)
    state_bf16 = state.dtype == ttnn.bfloat16
    state_bytes = 2048 if state_bf16 else 4096
    batch = state.shape[0] // 12
    assert 1 <= batch <= 32
    assert tuple(state.shape) == (batch * 12, 128, 128)
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
            (12, batch, 128),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=state.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (12, batch, 128)
    assert tuple(output.padded_shape) == (12, 32, 128)
    assert output.dtype == ttnn.float32 and output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG and output.device() == state.device()
    assert output.buffer_address() not in [t.buffer_address() for t in inputs]
    grid = state.device().compute_with_storage_grid_size()
    group_qk, partition = _local12_head_partition(batch, grid.x * grid.y, cores, grouped_qk)
    if source_overrides is not None or variant != "baseline":
        assert packed_qkv and not normalized
    if source_overrides is None:
        from .recurrent_variants import sources

        reader_source, compute_source = sources(
            _LOCAL12_READER, _fused_delta_row_GDN_COMPACT_COMPUTE, variant, grouped=group_qk
        )
    else:
        reader_source, compute_source = source_overrides
    ncores = len(partition)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for c, (start, count) in zip(coords, partition):
        rr[c.x][c.y] = [t.buffer_address() for t in inputs] + [start, count]
        wr[c.x][c.y] = [state.buffer_address(), output.buffer_address(), start, count]
        cr[c.x][c.y] = [count, start]
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
        kernel(head_major_writer_source(_fused_delta_row_GDN_COMPACT_WRITER), ttnn.WriterConfigDescriptor(), wr, wct),
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


def local_norm_gate_head_major(
    readout_fp32, weight, packed, *, epsilon=1e-6, ownership="head", output=None, cores=None
):
    """FP32 [12,B,128] -> gated BF16 [1,B,1536], compact decode only."""
    import ttnn

    assert math.isfinite(epsilon) and epsilon > 0
    assert len(readout_fp32.shape) == 3 and readout_fp32.shape[0] == 12 and readout_fp32.shape[2] == 128
    batch = readout_fp32.shape[1]
    assert 1 <= batch <= 32 and tuple(weight.shape) == (1, 1, 128)
    assert tuple(readout_fp32.padded_shape) == (12, 32, 128)
    assert len(packed.shape) == 3 and tuple(packed.shape)[:2] == (1, batch) and packed.shape[2] >= 4096
    assert tuple(weight.padded_shape) == (1, 32, 128)
    assert packed.padded_shape[-2] == 32 and packed.padded_shape[-1] % 32 == 0
    assert readout_fp32.dtype == ttnn.float32 and weight.dtype == packed.dtype == ttnn.bfloat16
    device = readout_fp32.device()
    for t in (readout_fp32, weight, packed):
        assert t.layout == ttnn.TILE_LAYOUT and t.memory_config() == ttnn.DRAM_MEMORY_CONFIG and t.device() == device
    if output is None:
        output = ttnn.empty(
            (1, batch, 1536),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (1, batch, 1536) and tuple(output.padded_shape) == (1, 32, 1536)
    assert output.dtype == ttnn.bfloat16 and output.layout == ttnn.TILE_LAYOUT
    assert output.device() == device and output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert output.buffer_address() not in [t.buffer_address() for t in (readout_fp32, weight, packed)]
    grid = device.compute_with_storage_grid_size()
    partition = task_partition(ownership, grid.x * grid.y, cores)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(partition))]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for c, (start, count) in zip(coords, partition):
        rr[c.x][c.y] = [t.buffer_address() for t in (readout_fp32, weight, packed)] + [batch, start, count]
        wr[c.x][c.y] = [output.buffer_address(), start, count]
        cr[c.x][c.y] = [start, count]
    flag = int(ownership == "tile")
    rct = [flag, struct.unpack("<I", struct.pack("<f", epsilon))[0]]
    for t in (readout_fp32, weight, packed):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [flag, *ttnn.TensorAccessorArgs(output).get_compile_time_args()]
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    # Native RMS intermediates use ordinary UnpackToSrc, not direct-DST aliases.
    config.unpack_to_dest_mode = modes

    def kernel(source, cfg, rt, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            runtime_args=rt,
            compile_time_args=ct,
            config=cfg,
        )

    kernels = [
        kernel(head_major_norm_reader_source(READER), ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(WRITER, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(COMPUTE, config, cr, [flag]),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=count * page,
            core_ranges=ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=i, data_format=ttnn.float32 if page == 4096 else ttnn.bfloat16, page_size=page
                )
            ],
        )
        for i, count, page in CB_LAYOUT
    ]
    ttnn.generic_op(
        [readout_fp32, weight, packed, output], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    )
    return output
