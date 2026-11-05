"""CPU tests of tt/dram_levers.py (copy guard + budget arithmetic) and tools/memreport_check.py. No device."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DL = _load("dram_levers", "tt/dram_levers.py")
MC = _load("memreport_check", "tools/memreport_check.py")

R1_LINES = """\
2026-10-30 INFO ... - MEMREPORT after the S_pad 1048576 chunk trace: DRAM allocated 3850.0 MiB/bank, free 81.4, largest free block 76.4; trace region allocated 8.1 MiB/bank, free 134.9, largest free block 134.9
2026-10-30 INFO ... - MEMREPORT after the model warm-up (prefill chunk + decode traces captured): DRAM allocated 3852.0 MiB/bank, free 79.4, largest free block 76.4; trace region allocated 21.0 MiB/bank, free 122.0, largest free block 122.0
2026-10-30 INFO ... - MEMREPORT after the vision traces (end of the build): DRAM allocated 3853.6 MiB/bank, free 77.8, largest free block 75.7; trace region allocated 82.4 MiB/bank, free 60.7, largest free block 60.7
""".splitlines()


def test_guard_off_keeps_everything():
    env = {}
    for U in (2, 4):
        assert DL.want_attn_front_copy(U, env) and DL.want_skinny_wo_copy(U, env) and DL.want_skinny_shared_copy(U, env)


def test_guard_b8_keeps_all_three():
    env = {"DSV41_COPY_GUARD": "1", "DSV41_VLLM_SPEC_KS": "3,5"}
    assert DL.verify_rows(2, env) == [8, 12]
    assert DL.want_attn_front_copy(2, env) and DL.want_skinny_wo_copy(2, env) and DL.want_skinny_shared_copy(2, env)


def test_guard_b16_drops_af_and_shared_keeps_wo_at_k3():
    env = {"DSV41_COPY_GUARD": "1"}
    assert DL.verify_rows(4, env) == [16, 24]
    assert not DL.want_attn_front_copy(4, env)
    assert not DL.want_skinny_shared_copy(4, env)
    assert DL.want_skinny_wo_copy(4, env)  # k=3 -> T=16
    assert not DL.want_skinny_wo_copy(4, {**env, "DSV41_VLLM_SPEC_KS": "5"})  # k=5 only -> T=24


def test_guard_env_overrides_follow_the_t16_24_extensions():
    env = {"DSV41_COPY_GUARD": "1", "DSV41_COPY_GUARD_AF_T": "8,12,16", "DSV41_COPY_GUARD_SHARED_MAX": "24"}
    assert DL.want_attn_front_copy(4, env) and DL.want_skinny_shared_copy(4, env)


def test_budget_matches_the_measured_points():
    a, f = DL.budget(8)
    assert abs(a - 3744.2) < 0.05 and abs(f - 186.0) < 2.0  # measured free 187.2 (the 3930 usable is rounded)
    a, f = DL.budget(16)
    assert -156 < f < -144  # PLAN.md: about -150 (OOM at boot)


def test_recommended_combinations_reach_40_free():
    # combination 1: guard (af + skinny shared) + no paged APC stores at B=16 (unsupported at Up < U) + trace region 1.0e9 + embedding shard
    _, f1 = DL.budget(16, copies_on=("skinny_wo",), levers=("embed_shard", "trace_1e9"))
    assert f1 >= 150
    # and still >= 40 if the T=16/24 work makes every fused copy live again (paged APC stays off)
    _, f1b = DL.budget(16, copies_on=("attn_front", "skinny_wo", "skinny_shared"), levers=("embed_shard", "trace_1e9"))
    assert f1b >= 40
    # without the embedding shard the python-only levers are NOT enough
    _, f0 = DL.budget(16, copies_on=("skinny_wo",), levers=("trace_1e9",))
    assert 20 < f0 < 40
    # combination 2: bfp4 experts in place instead of the embedding shard
    _, f2 = DL.budget(16, copies_on=("skinny_wo",), levers=("bfp4_experts", "trace_1e9"))
    assert f2 >= 1000


def test_memreport_parse_and_check():
    rows = MC.parse(R1_LINES)
    assert [r["alloc"] for r in rows] == [3850.0, 3852.0, 3853.6]
    assert rows[-1]["trace_used"] == 82.4
    ok, msg = MC.check(rows, 40, 40)
    assert ok and "PASS" in msg
    ok, _ = MC.check(rows, 100, 40)
    assert ok is False
    assert MC.check([], 40, 40)[0] is None


def test_memreport_cli(tmp_path):
    p = tmp_path / "server.log"
    p.write_text("\n".join(R1_LINES))
    assert MC.main(["x", str(p), "--min-free", "40"]) == 0
    assert MC.main(["x", str(p), "--min-free", "90"]) == 1
