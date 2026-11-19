# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in recurrent value-column ownership; CPU-safe source/host preparation.

The caller supplies already-selected cumulative_exp_scalar source strings.
Each owner retains every K-row of its value columns. Full-size CB groups stay
unchanged, but unowned state/output slots are never read or written to DRAM.
"""

import ast
from functools import partial
from types import FunctionType, SimpleNamespace


def replace_once(source, before, after):
    assert source.count(before) == 1, (source.count(before), before)
    return source.replace(before, after, 1)


def column_partition(batch, available_cores, cores=None, grouped_qk=True, *, grain=1):
    """Contiguous ownership in units of 1, 2 or 4 complete value columns."""
    assert 1 <= batch <= 32 and available_cores >= 1 and grain in (1, 2, 4)
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


def host_source(model_source):
    """Private host-wrapper snapshot with boundary runtime args in all kernels."""
    tree = ast.parse(model_source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_local12_recurrent")
    source = ast.get_source_segment(model_source, node)
    source = replace_once(
        source,
        "group_qk, partition = _local12_head_partition(batch, grid.x * grid.y, cores, grouped_qk)",
        "group_qk, partition = _column_partition(batch, grid.x * grid.y, cores, grouped_qk)",
    )
    source = replace_once(
        source,
        """    if source_overrides is None:
        from .recurrent_variants import sources

        reader_source, compute_source = sources(
            _LOCAL12_READER, _fused_delta_row_GDN_COMPACT_COMPUTE, variant, grouped=group_qk
        )
    else:
        reader_source, compute_source = source_overrides""",
        "    assert packed_qkv and not normalized and source_overrides is not None\n"
        "    assert len(source_overrides) == 3\n"
        "    reader_source, compute_source, writer_source = source_overrides",
    )
    source = replace_once(
        source,
        """    for c, (start, count) in zip(coords, partition):
        rr[c.x][c.y] = [t.buffer_address() for t in inputs] + [start, count]
        wr[c.x][c.y] = [state.buffer_address(), output.buffer_address(), start, count]
        cr[c.x][c.y] = [count, start]""",
        """    for c, (start, count, first_n, last_n) in zip(coords, partition):
        rr[c.x][c.y] = [t.buffer_address() for t in inputs] + [start, count, first_n, last_n]
        wr[c.x][c.y] = [state.buffer_address(), output.buffer_address(), start, count, first_n, last_n]
        cr[c.x][c.y] = [count, start, first_n, last_n]""",
    )
    source = replace_once(
        source,
        "kernel(_fused_delta_row_GDN_COMPACT_WRITER, ttnn.WriterConfigDescriptor(), wr, wct)",
        "kernel(writer_source, ttnn.WriterConfigDescriptor(), wr, wct)",
    )
    return source


def bind_local(local, writer, model_source):
    """Fixed independent functions: one original and one for each ownership grain."""
    assert getattr(local, "_fused_delta_row_GDN_COMPACT_WRITER") == writer
    original = local._local12_recurrent
    baseline_globals = dict(original.__globals__)
    baseline_globals["_fused_delta_row_GDN_COMPACT_WRITER"] = writer
    baseline = FunctionType(
        original.__code__, baseline_globals, original.__name__, original.__defaults__, original.__closure__
    )
    baseline.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    code = compile(host_source(model_source), "<column-recurrent-host-snapshot>", "exec")
    functions = {}
    for name, grain in (("columns", 1), ("pairs", 2)):
        namespace = dict(original.__globals__)
        namespace["_column_partition"] = partial(column_partition, grain=grain)
        exec(code, namespace)
        functions[name] = namespace["_local12_recurrent"]
    return SimpleNamespace(
        _local12_head_partition=local._local12_head_partition,
        _local12_recurrent=baseline,
        column_functions=functions,
        writer=writer,
    )
