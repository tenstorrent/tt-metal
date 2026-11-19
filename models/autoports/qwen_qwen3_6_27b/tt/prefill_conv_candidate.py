# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in batched local GDN prefill convolution; lazy TTNN imports only.

No default integration. FP32 product/add -> BF16 -> native BF16 SiLU reuses
validated decode arithmetic; native prefill numerical equivalence needs the
separate probe because its diagonal-matmul / pack-SiLU arithmetic differs.
"""


def _replace(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new, 1)


def prefill_sources(reader, compute, writer, *, poison_dead_padding=False):
    from .recurrent_next_candidate import local_conv_sources

    reader, compute, writer = local_conv_sources(reader, compute, writer, poison_dead_padding=poison_dead_padding)
    reader = _replace(
        reader,
        "const uint32_t start=get_arg_val<uint32_t>(6),count=get_arg_val<uint32_t>(7);",
        "const uint32_t start=get_arg_val<uint32_t>(6),count=get_arg_val<uint32_t>(7),sequence=get_arg_val<uint32_t>(8);",
    )
    reader = _replace(
        reader,
        "user=task/3,chunk=task%3,bytes=(chunk==2?1024:2048)",
        "user=task/(sequence*3),time=(task/3)%sequence,chunk=task%3,bytes=(chunk==2?1024:2048)",
    )
    reader = _replace(
        reader,
        "noc.async_read(history,cb,bytes,{.page_id=user*3+row,.offset_bytes=chunk*2048},{.offset_bytes=0});",
        """const int32_t source_time=int32_t(time)+int32_t(row)-3;
            if(source_time<0) {
                noc.async_read(history,cb,bytes,{.page_id=user*3+uint32_t(source_time+3),.offset_bytes=chunk*2048},{.offset_bytes=0});
            } else {
                noc.async_read(x,cb,bytes,{.page_id=user*sequence+uint32_t(source_time),.offset_bytes=chunk*2048},{.offset_bytes=0});
            }""",
    )
    reader = _replace(
        reader, "{.page_id=user,.offset_bytes=chunk*2048}", "{.page_id=user*sequence+time,.offset_bytes=chunk*2048}"
    )
    writer = _replace(
        writer,
        "const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);",
        "const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3),sequence=get_arg_val<uint32_t>(4);",
    )
    writer = _replace(
        writer,
        "user=task/3,chunk=task%3,bytes=(chunk==2?1024:2048),slot=(task-start)%2",
        "user=task/(sequence*3),time=(task/3)%sequence,chunk=task%3,bytes=(chunk==2?1024:2048),slot=(task-start)%2",
    )
    writer = _replace(
        writer, "{.page_id=user,.offset_bytes=chunk*2048}", "{.page_id=user*sequence+time,.offset_bytes=chunk*2048}"
    )
    writer = _replace(
        writer, "for(uint32_t row=0;row<3;row++) {", "if(time+1==sequence) for(uint32_t row=0;row<3;row++) {"
    )
    # Writer's history accessor receives a DISTINCT allocation. Other cores can
    # still be reading the input history when the final-time task completes.
    return reader, compute, writer


def batched_prefill_conv(qkv, history, taps, *, cores=None, poison_dead_padding=False):
    """Return padded TILE Q/K/V and update stable raw history after one kernel.

    Public input [B,S,2560], 1<=B<=32, arbitrary positive S; history [B,3,2560]
    row major. Input and history are immutable during the generic kernel.
    Each output lane sums taps in chronological order with no cross-user reads.
    Output dead sequence rows are explicitly zeroed, and never enter history.
    """
    import ttnn
    from .optimized_decoder import (
        _conv107__CONV_RM_PIPE2_READER,
        _conv107__CONV_RM_PIPE2_COMPUTE,
        _conv107__CONV_RM_PIPE2_WRITER,
    )

    batch, seq, channels = tuple(qkv.shape)
    assert 1 <= batch <= 32 and seq > 0 and channels == 2560
    assert batch * seq * 3 < 2**32
    assert tuple(history.shape) == (batch, 3, channels)
    assert qkv.layout == ttnn.TILE_LAYOUT and history.layout == ttnn.ROW_MAJOR_LAYOUT
    assert len(taps) == 4
    raw = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)
    inputs = [raw, history, *taps]
    for value in inputs:
        assert value.layout == ttnn.ROW_MAJOR_LAYOUT and value.dtype == ttnn.bfloat16
        assert value.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert all(tuple(t.shape) == (1, 1, channels) for t in taps)

    def allocate(shape):
        return ttnn.empty(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=qkv.device(),
        )

    result = allocate((batch, seq, channels))
    carry = allocate((batch, 3, channels))
    grid = qkv.device().compute_with_storage_grid_size()
    tasks = batch * seq * 3
    ncores = min(tasks, cores or grid.x * grid.y)
    assert 1 <= ncores <= grid.x * grid.y
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    start = 0
    for i, coord in enumerate(coords):
        count = tasks // ncores + (i < tasks % ncores)
        rr[coord.x][coord.y] = [t.buffer_address() for t in inputs] + [start, count, seq]
        wr[coord.x][coord.y] = [carry.buffer_address(), result.buffer_address(), start, count, seq]
        cr[coord.x][coord.y] = [count]
        start += count
    rct, wct = [], []
    for value in inputs:
        rct.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    for value in (carry, result):
        wct.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())

    def kernel(source, config, args, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=args,
            config=config,
        )

    reader, compute, writer = prefill_sources(
        _conv107__CONV_RM_PIPE2_READER,
        _conv107__CONV_RM_PIPE2_COMPUTE,
        _conv107__CONV_RM_PIPE2_WRITER,
        poison_dead_padding=poison_dead_padding,
    )
    kernels = [
        kernel(reader, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(writer, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(
            compute,
            ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
            ),
            cr,
            [],
        ),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in (*range(8), 9, 10)
    ]
    ttnn.generic_op(inputs + [result, carry], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    # Dispatch ordering supplies the cross-core barrier; never alias input and
    # output history inside the kernel. Stable state allocation remains intact.
    ttnn.copy(carry, history)
    result = ttnn.to_layout(result, ttnn.TILE_LAYOUT)
    padded = ((seq + 31) // 32) * 32
    if padded != seq:
        result = ttnn.pad(result, ((0, 0), (0, padded - seq), (0, 0)), value=0.0)
    return result[:, :, :512], result[:, :, 512:1024], result[:, :, 1024:2560]


def install(layer, *, cores=None):
    """Install optional helper after parent adds the documented _gdn hook."""
    assert layer.kind == "linear_attention"
    assert "_prefill_batched_conv" in layer._gdn.__code__.co_names, "Add the documented _gdn hook first"
    assert not hasattr(layer, "_prefill_batched_conv")
    layer._prefill_batched_conv = lambda q, h: batched_prefill_conv(q, h, layer.taps_rm, cores=cores)
    layer.policy["batched_prefill_conv_candidate"] = True
    return layer


def cpu_check():
    """Check source transformation and exact task addressing without TTNN."""
    import ast
    import json
    from pathlib import Path

    tree = ast.parse(Path(__file__).with_name("optimized_decoder.py").read_text())
    names = ["_conv107__CONV_RM_PIPE2_" + suffix for suffix in ("READER", "COMPUTE", "WRITER")]
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names
    }
    for poison in (False, True):
        reader, compute, writer = prefill_sources(*(values[n] for n in names), poison_dead_padding=poison)
        assert compute == values[names[1]]
        assert "if(time+1==sequence)" in writer and "source_time<0" in reader
    cases = 0
    for batch in (1, 2, 3, 17, 32):
        for seq in (1, 2, 3, 17, 31, 32, 33, 65, 257, 513):
            writes = set()
            for task in range(batch * seq * 3):
                user, time, chunk = task // (seq * 3), (task // 3) % seq, task % 3
                assert (user, time, chunk) not in writes
                writes.add((user, time, chunk))
                reads = [("h", user, time + j) if time + j < 3 else ("x", user, time + j - 3) for j in range(4)]
                assert all(0 <= row < (3 if kind == "h" else seq) for kind, _, row in reads)
                if time == seq - 1:
                    expected = [("h", user, seq + j) if seq + j < 3 else ("x", user, seq + j - 3) for j in range(3)]
                    assert reads[1:] == expected
            assert len(writes) == batch * seq * 3
            cases += 1
    print(json.dumps({"status": "pass", "scope": "CPU source and address model only", "cases": cases}))


if __name__ == "__main__":
    import sys

    assert sys.argv[1:] == ["--cpu-check"]
    cpu_check()
