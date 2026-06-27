# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-lang kernel lever — compile-error detection, repair budget, lever indexing, availability gate."""

from agent import probes, states
from agent.handlers import route as route_mod


def test_tt_lang_compile_error_detected_as_crash():
    # a kernel that fails to compile/lower must be detected so it routes to REPAIR_CODE
    for log in [
        "=========== 1 error in 3.2s ===========\nE   ttl.CompilationError: cannot lower matmul tile\n",
        'FAILED tests/e2e/test_perf.py::test_prefill_perf\nloc("k.py":12:3): error: unsupported op\n',
        "=== 1 failed ===\ntt-lang lowering Error: dataflow buffer overflow\n",
        "=== 1 error ===\nfailed to compile kernel for grid (8,8)\n",
    ]:
        got = probes.detect_perf_crash(log)
        assert got, f"tt-lang compile error not detected: {log!r}"


def test_benign_perf_assert_still_not_a_crash():
    # a model that ran fully but failed only a perf-threshold assert is NOT a crash (no regression)
    log = "=========== 1 failed in 5s ===========\nE   AssertionError: device_ms 41.0 > target 39.0\n"
    assert probes.detect_perf_crash(log) is None


def test_kernel_lever_gets_largest_repair_budget():
    assert states.code_fix_budget(states.KERNEL_LEVER) == states.MAX_CODE_FIX_KERNEL
    assert states.MAX_CODE_FIX_KERNEL > states.MAX_CODE_FIX_PRINCIPLES > states.MAX_CODE_FIX
    # other levers unchanged
    assert states.code_fix_budget(states.FROM_PRINCIPLES) == states.MAX_CODE_FIX_PRINCIPLES
    assert states.code_fix_budget("qkv-program-config") == states.MAX_CODE_FIX


def test_kernel_lever_is_routable():
    from agent.router import build_index, read_section, route

    idx = build_index()
    hit = [e for e in idx if e["id"] == states.KERNEL_LEVER]
    assert hit, "#tt-lang-kernel lever not indexed by the router"
    assert hit[0]["lever_type"] == "structural"
    # routes when matmul is the bottleneck (and other compute classes)
    assert any(e["id"] == states.KERNEL_LEVER for e in route(idx, {"op_class": "matmul", "rank": "time"}))
    assert any(e["id"] == states.KERNEL_LEVER for e in route(idx, {"op_class": "attention", "rank": "time"}))
    # section the structural agent reads carries the ttl API recipe
    sec = read_section(states.KERNEL_LEVER)
    assert "ttl" in sec and "@ttl.operation" in sec and "I/O" in sec


def test_availability_gate_helper():
    # the gate must reflect reality: True iff the ttl toolchain is importable in this env.
    import importlib.util

    expected = importlib.util.find_spec("ttl") is not None
    assert route_mod._tt_lang_available() is expected
