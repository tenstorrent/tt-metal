# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact cumulative recurrence versus column/pair ownership, isolated from model.

Retains the recurrent probe's BF16/FP32 output/state checks, zero controls,
changed-input traces and matched timing. --cpu-check never imports TTNN.
"""

import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from models.autoports.qwen_qwen3_6_27b.tests import probe_multichip_stream_state as stream_probe
from models.autoports.qwen_qwen3_6_27b.tt import recurrent_column_candidate as candidate

original = stream_probe.original


def bind_local(local, frozen):
    writer = original.constant(frozen["optimized_decoder"], stream_probe.WRITER)
    return candidate.bind_local(local, writer, (original.ROOT / "tt/multichip_decoder.py").read_text())


def adapted_main_source():
    source = stream_probe.adapted_main_source()
    edits = (
        (
            'choices=["stream_state"], default=["stream_state"]',
            'choices=["columns", "pairs"], default=["columns", "pairs"]',
        ),
        (
            'group, partition = local._local12_head_partition(batch, grid.x * grid.y, cores, args.grouped_qk == "on")',
            'group, baseline_partition = local._local12_head_partition(batch, grid.x * grid.y, cores, args.grouped_qk == "on")\n'
            '        _, partition = _column_partition(batch, grid.x * grid.y, cores, args.grouped_qk == "on", grain=1 if variant == "columns" else 2)',
        ),
        (
            "candidate = _stream_state_sources(*baseline, local.writer)",
            "candidate = _column_sources(*baseline, local.writer)",
        ),
        (
            '"grouped_qk": group, "partition": partition,',
            '"grouped_qk": group, "partition": partition, "baseline_partition": baseline_partition,',
        ),
        (
            "return local._local12_recurrent(qkv, qkv, qkv, g, beta, states[index], packed_qkv=True,",
            "operation = local._local12_recurrent if index == 0 else local.column_functions[variant]\n"
            "            return operation(qkv, qkv, qkv, g, beta, states[index], packed_qkv=True,",
        ),
        (
            "cumulative_exp_scalar versus same plus streamed state persistence",
            "cumulative_exp_scalar versus identical math with value-column ownership",
        ),
    )
    for before, after in edits:
        source = candidate.replace_once(source, before, after)
    return source


def build_main():
    source = adapted_main_source()
    namespace = dict(vars(original))
    namespace.update(
        __name__=__name__ + "._snapshot",
        __doc__=__doc__,
        _bind_local=bind_local,
        _baseline_sources=stream_probe.baseline_sources,
        _column_sources=candidate.column_sources,
        _column_partition=candidate.column_partition,
        _adapter_metadata={
            "generated_main_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "host_snapshot_sha256": hashlib.sha256(
                candidate.host_source((original.ROOT / "tt/multichip_decoder.py").read_text()).encode()
            ).hexdigest(),
            "source_sha256": {
                str(path.relative_to(original.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    Path(__file__),
                    Path(stream_probe.__file__),
                    original.ROOT / "tt/recurrent_column_candidate.py",
                    original.ROOT / "tt/recurrent_next_candidate.py",
                )
            },
            "binding": "fixed independent host snapshots; no model globals or policies modified",
            "allocation_lifetime": "all tensor allocations and both warmups precede either trace capture",
        },
    )
    exec(compile(source, str(Path(__file__)) + ":adapted_main", "exec"), namespace)
    return namespace["main"]


def check_host_contract(model_source, frozen, reader, compute, writer):
    """Execute the real host snapshot against descriptor-only CPU stand-ins."""
    programs = []

    class Tensor:
        def __init__(self, shape, dtype, layout, address, padded=None):
            self.shape, self.dtype, self.layout, self.address = shape, dtype, layout, address
            self.padded_shape = padded or shape

        def memory_config(self):
            return "dram"

        def buffer_address(self):
            return self.address

        def device(self):
            return SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))

    class Kernel(SimpleNamespace):
        SourceType = SimpleNamespace(SOURCE_CODE="source")

    standin = SimpleNamespace(
        ROW_MAJOR_LAYOUT="rm",
        TILE_LAYOUT="tile",
        DRAM_MEMORY_CONFIG="dram",
        bfloat16="bf16",
        float32="fp32",
        CoreCoord=lambda x, y: SimpleNamespace(x=x, y=y),
        CoreRange=lambda a, b: (a, b),
        CoreRangeSet=lambda x: x,
        RuntimeArgs=lambda: defaultdict(dict),
        TensorAccessorArgs=lambda t: SimpleNamespace(get_compile_time_args=lambda: [t.address]),
        KernelDescriptor=Kernel,
        ReaderConfigDescriptor=lambda: "reader",
        WriterConfigDescriptor=lambda: "writer",
        CBDescriptor=SimpleNamespace,
        CBFormatDescriptor=SimpleNamespace,
        ProgramDescriptor=SimpleNamespace,
        generic_op=lambda tensors, program: programs.append(program),
    )
    env = {
        stream_probe.WRITER: writer,
        "ttnn": standin,
        "_fused_delta_row_compact_fp32_compute_config": lambda bf16: ("compute", bf16),
    }
    for suffix in ("FUSE_ACCUMULATOR", "HOIST_DELTA", "HOIST_SCALAR_INIT", "HOIST_COL_INIT", "FUSE_DECAY_PREDICT"):
        name = "_fused_delta_row_GDN_" + suffix
        env[name] = original.constant(frozen["optimized_decoder"], name)
    exec("def fake(*a, **kw):\n    return None", env)
    local = SimpleNamespace(
        **{stream_probe.WRITER: writer}, _local12_recurrent=env["fake"], _local12_head_partition=None
    )
    proxy = candidate.bind_local(local, writer, model_source)
    checked = 0
    for batch in (1, 3, 32):
        for cores in (None, 1, 7):
            for dtype in ("bf16", "fp32"):
                for name, grain in (("columns", 1), ("pairs", 2)):
                    grouped, plan = candidate.column_partition(batch, 110, cores, grain=grain)
                    src = candidate.column_sources(
                        *stream_probe.baseline_sources(reader, compute, grouped=grouped), writer
                    )
                    q = Tensor((batch, 1, 2560), "bf16", "rm", 1)
                    g = Tensor((batch, 1, 12), "fp32", "tile", 2, (batch, 32, 32))
                    beta = Tensor((batch, 1, 12), "bf16", "tile", 3, (batch, 32, 32))
                    state = Tensor((batch * 12, 128, 128), dtype, "tile", 4)
                    output = Tensor((1, batch * 12, 128), "fp32", "tile", 5)
                    result = proxy.column_functions[name](
                        q, q, q, g, beta, state, output=output, cores=cores, packed_qkv=True, source_overrides=src
                    )
                    assert result is output
                    program = programs.pop()
                    rr, wr, cr = [kernel.runtime_args for kernel in program.kernels]
                    assert [kernel.kernel_source for kernel in program.kernels] == [src[0], src[2], src[1]]
                    for core, (head, count, first, last) in enumerate(plan):
                        x, y = core % 11, core // 11
                        assert rr[x][y] == [1, 1, 1, 2, 3, 4, head, count, first, last]
                        assert wr[x][y] == [4, 5, head, count, first, last]
                        assert cr[x][y] == [count, head, first, last]
                    cbs = {cb.format_descriptors[0].buffer_index: cb for cb in program.cbs}
                    state_bytes = 2048 if dtype == "bf16" else 4096
                    assert cbs[5].total_size == cbs[11].total_size == 32 * state_bytes
                    assert cbs[12].total_size == 8 * 4096
                    assert program.kernels[1].compile_time_args[0] == state_bytes
                    checked += 1
    return checked


def cpu_check():
    reader, compute, frozen = original.frozen_sources(original.FROZEN)
    writer = original.constant(frozen["optimized_decoder"], stream_probe.WRITER)
    for grouped in (False, True):
        for poison in (False, True):
            baseline = stream_probe.baseline_sources(reader, compute, grouped=grouped, poison_dead_padding=poison)
            rr, cc, ww = candidate.column_sources(*baseline, writer)
            # Removing the two deliberate bounds changes restores the complete
            # compute string, proving every arithmetic/packing instruction stays.
            restored = cc.replace(
                "    const uint32_t first_n=get_arg_val<uint32_t>(2),last_n=get_arg_val<uint32_t>(3);\n", ""
            ).replace(
                "        const uint32_t n_begin=head==0?first_n:0,n_end=head+1==count?last_n:4;\n"
                "        for(uint32_t n=n_begin;n<n_end;n++) {",
                "        for(uint32_t n=0;n<4;n++) {",
            )
            assert restored == baseline[1]
            assert "state.reserve_back(16)" in rr and "state.push_back(16)" in rr
            assert "updated.wait_front(16); out.wait_front(4);" in ww
            assert "updated.pop_front(16); out.pop_front(4);" in ww
            assert ww.count("for(uint32_t i=n_begin;i<n_end;i++) for(uint32_t face=0;face<2;face++)") == 2
    checked = 0
    for batch in range(1, 33):
        for cores in (1, 3, 7, 48, 110):
            for grain in (1, 2, 4):
                grouped, plan = candidate.column_partition(batch, 110, cores, True, grain=grain)
                columns, state_pages, output_bytes = set(), set(), set()
                for start, heads, first_n, last_n in plan:
                    assert heads >= 1 and 0 <= first_n < 4 and 1 <= last_n <= 4
                    for h in range(start, start + heads):
                        begin = first_n if h == start else 0
                        end = last_n if h + 1 == start + heads else 4
                        assert begin < end
                        for n in range(begin, end):
                            assert (h, n) not in columns
                            columns.add((h, n))
                            for m in range(4):
                                page = h * 16 + m * 4 + n
                                assert page not in state_pages
                                state_pages.add(page)
                            rows = [h % 32]
                            if h + 1 == batch * 12 and batch * 12 % 32:
                                rows.extend(range(batch * 12 % 32, 32))
                            for row in rows:
                                for face in range(2):
                                    address = ((h // 32) * 4 + n, ((row // 16) * 512 + face * 256 + row % 16 * 16) * 4)
                                    assert address not in output_bytes and address[1] % 64 == 0
                                    output_bytes.add(address)
                assert len(columns) == batch * 48
                assert state_pages == set(range(batch * 12 * 16))
                assert len(output_bytes) == ((batch * 12 + 31) // 32) * 32 * 4 * 2
                assert grouped == (batch >= 3)
                checked += 1
    model_source = (original.ROOT / "tt/multichip_decoder.py").read_text()
    fake_globals = {stream_probe.WRITER: writer}
    exec("def fake(*a, **kw):\n    return " + stream_probe.WRITER, fake_globals)
    fake = SimpleNamespace(
        **{stream_probe.WRITER: writer},
        _local12_recurrent=fake_globals["fake"],
        _local12_head_partition=lambda *a: None,
    )
    proxy = candidate.bind_local(fake, writer, model_source)
    assert proxy._local12_recurrent() == writer
    for name, grain in (("columns", 1), ("pairs", 2)):
        actual = proxy.column_functions[name].__globals__["_column_partition"](32, 110)
        assert actual == candidate.column_partition(32, 110, grain=grain)
    fake_globals[stream_probe.WRITER] = "sentinel"
    assert proxy._local12_recurrent() == writer
    host_cases = check_host_contract(model_source, frozen, reader, compute, writer)
    build_main()
    assert "ttnn" not in sys.modules
    print(
        f"PASS: {checked} ownership and {host_cases} host-argument cases, exact math sources, disjoint writes; no TTNN import"
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--cpu-check"]:
        cpu_check()
    else:
        build_main()()
