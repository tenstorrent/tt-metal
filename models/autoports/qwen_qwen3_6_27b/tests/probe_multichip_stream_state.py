# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated cumulative_exp_scalar versus streamed-state recurrence probe.

Reuses the existing recurrent probe's fixtures, exactness gates and timings in
a private compiled namespace. Both writer bindings are fixed function-global
snapshots; no model globals, decoder files or existing probes are modified.
--cpu-check validates adaptation and snapshots without importing TTNN.
"""

import ast
import hashlib
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

from models.autoports.qwen_qwen3_6_27b.tests import probe_multichip_recurrent as original
from models.autoports.qwen_qwen3_6_27b.tt.recurrent_next_candidate import stream_state_sources
from models.autoports.qwen_qwen3_6_27b.tt.recurrent_variants import sources

WRITER = "_fused_delta_row_GDN_COMPACT_WRITER"


def replace_once(source, old, new):
    assert source.count(old) == 1, (source.count(old), old)
    return source.replace(old, new, 1)


def baseline_sources(reader, compute, *, grouped, poison_dead_padding=False):
    return sources(reader, compute, "cumulative_exp_scalar", grouped=grouped, poison_dead_padding=poison_dead_padding)


def snapshot_function(function, writer):
    """Clone code/defaults with a private, fixed writer binding, once at setup."""
    globals_snapshot = dict(function.__globals__)
    globals_snapshot[WRITER] = writer
    snapshot = FunctionType(
        function.__code__, globals_snapshot, function.__name__, function.__defaults__, function.__closure__
    )
    snapshot.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return snapshot


def bind_local(local, frozen):
    """Build immutable baseline/streaming writer snapshots; select by tuple arity."""
    writer = original.constant(frozen["optimized_decoder"], WRITER)
    reader = original.constant(frozen["multichip_decoder"], "_LOCAL12_READER")
    compute = original.constant(frozen["optimized_decoder"], "_fused_delta_row_GDN_COMPACT_COMPUTE")
    assert getattr(local, WRITER) == writer, "Frozen/current writer drift"
    _, _, streamed_writer = stream_state_sources(*baseline_sources(reader, compute, grouped=False), writer)
    baseline_function = snapshot_function(local._local12_recurrent, writer)
    streaming_function = snapshot_function(local._local12_recurrent, streamed_writer)

    def dispatch(*args, source_overrides, **kwargs):
        assert len(source_overrides) in (2, 3)
        if len(source_overrides) == 3:
            assert source_overrides[2] == streamed_writer, "Wrong writer for streaming compute"
            return streaming_function(*args, source_overrides=source_overrides[:2], **kwargs)
        return baseline_function(*args, source_overrides=source_overrides, **kwargs)

    return SimpleNamespace(
        _local12_head_partition=local._local12_head_partition,
        _local12_recurrent=dispatch,
        writer=writer,
        streamed_writer=streamed_writer,
    )


def adapted_main_source():
    """Fail on source drift instead of silently retaining the old comparison."""
    text = Path(original.__file__).read_text()
    tree = ast.parse(text)
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    source = ast.get_source_segment(text, main_node)
    source = replace_once(
        source,
        "choices=VARIANTS, default=list(VARIANTS[1:7])",
        'choices=["stream_state"], default=["stream_state"]',
    )
    source = replace_once(
        source,
        "from models.autoports.qwen_qwen3_6_27b.tt import multichip_decoder as local",
        "from models.autoports.qwen_qwen3_6_27b.tt import multichip_decoder as local\n"
        "    local = _bind_local(local, frozen)",
    )
    source = replace_once(
        source,
        "candidate = sources(reader, compute, variant, grouped=group, poison_dead_padding=args.poison_dead_padding)",
        "baseline = _baseline_sources(reader, compute, grouped=group, poison_dead_padding=args.poison_dead_padding)\n"
        "        candidate = _stream_state_sources(*baseline, local.writer)",
    )
    source = replace_once(
        source,
        "source_overrides=(reader, compute) if index == 0 else candidate",
        "source_overrides=baseline if index == 0 else candidate",
    )
    source = replace_once(
        source,
        '"grouped_qk": group, "partition": partition, "candidate_sha256": [digest(s) for s in candidate],',
        '"grouped_qk": group, "partition": partition, "candidate_sha256": [digest(s) for s in candidate],\n'
        '                 "baseline_sha256": [digest(s) for s in (*baseline, local.writer)],',
    )
    source = replace_once(
        source,
        'report = {"status": "running",',
        'report = {"comparison": "cumulative_exp_scalar versus same plus streamed state persistence",\n'
        '              "stream_adapter": _adapter_metadata, "status": "running",',
    )
    source = replace_once(
        source,
        "                ttnn.synchronize_device(mesh)\n                tid = ttnn.begin_trace_capture(mesh, cq_id=0)",
        "                ttnn.synchronize_device(mesh)\n"
        "            # Warm both fixed kernels before either trace is active.\n"
        "            for index in range(2):\n"
        "                tid = ttnn.begin_trace_capture(mesh, cq_id=0)",
    )
    return source


def build_main():
    source = adapted_main_source()
    namespace = dict(vars(original))
    namespace.update(
        __name__=__name__ + "._snapshot",
        __doc__=__doc__,
        _bind_local=bind_local,
        _baseline_sources=baseline_sources,
        _stream_state_sources=stream_state_sources,
        _adapter_metadata={
            "generated_main_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "source_sha256": {
                str(path.relative_to(original.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), original.ROOT / "tt/recurrent_next_candidate.py")
            },
            "writer_binding": "two fixed private function-global snapshots; original module untouched",
            "allocation_lifetime": "all device tensors and both kernel warmups precede both captures",
        },
    )
    exec(compile(source, str(Path(__file__)) + ":adapted_main", "exec"), namespace)
    return namespace["main"]


def cpu_check():
    """Check real source transforms and snapshot isolation without TTNN."""
    reader, compute, frozen = original.frozen_sources(original.FROZEN)
    writer = original.constant(frozen["optimized_decoder"], WRITER)
    for grouped in (False, True):
        for poison in (False, True):
            baseline = baseline_sources(reader, compute, grouped=grouped, poison_dead_padding=poison)
            changed = stream_state_sources(*baseline, writer)
            assert changed[0] == baseline[0]
            assert changed[1].split("void kernel_main()", 1)[0] == baseline[1].split("void kernel_main()", 1)[0]
            assert "updated.reserve_back(16)" not in changed[1]
            assert "updated.push_back(16)" not in changed[1]
            assert "updated.wait_front(16)" not in changed[2]
            assert "updated.pop_front(16)" not in changed[2]
            assert "out.wait_front(4)" in changed[2] and "state.pop_front(16)" in changed[1]
    # A complete four-page column stays contiguous in the 32-page CB. Its
    # original matrix page mapping covers every head's sixteen pages once.
    for heads in (1, 2, 3, 12, 36, 384):
        pages = []
        for head in range(heads):
            for column in range(4):
                write_slot = ((head * 4 + column) * 4) % 32
                assert write_slot + 4 <= 32
                pages.extend(head * 16 + row * 4 + column for row in range(4))
        assert sorted(pages) == list(range(heads * 16))
    fake_globals = {WRITER: writer}
    exec("def fake(*args, **kwargs):\n    return " + WRITER + ", kwargs['source_overrides']", fake_globals)
    fake = SimpleNamespace(
        **{WRITER: writer}, _local12_recurrent=fake_globals["fake"], _local12_head_partition=lambda *a: None
    )
    proxy = bind_local(fake, frozen)
    baseline = baseline_sources(reader, compute, grouped=False)
    candidate = stream_state_sources(*baseline, writer)
    assert proxy._local12_recurrent(source_overrides=baseline) == (writer, baseline)
    assert proxy._local12_recurrent(source_overrides=candidate) == (candidate[2], candidate[:2])
    assert fake_globals[WRITER] == writer
    # Mutating the source namespace afterwards cannot change either snapshot.
    fake_globals[WRITER] = "sentinel"
    assert proxy._local12_recurrent(source_overrides=baseline)[0] == writer
    assert proxy._local12_recurrent(source_overrides=candidate)[0] == candidate[2]
    build_main()
    assert "ttnn" not in sys.modules
    print("PASS: frozen transforms, writer snapshot isolation, CB wraps/page mapping, adapted main; no TTNN import")


if __name__ == "__main__":
    if sys.argv[1:] == ["--cpu-check"]:
        cpu_check()
    else:
        build_main()()
