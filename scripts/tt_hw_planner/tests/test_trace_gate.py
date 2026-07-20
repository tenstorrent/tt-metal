import json
import os

from scripts.tt_hw_planner import trace_gate as tg


def test_trace_policy_all_graduated():
    pol = tg.trace_policy({"a": "sharded", "b": "native"})
    assert pol["required"] is True
    assert pol["all_graduated"] is True
    assert pol["eager_eligible_modules"] == set()


def test_trace_policy_some_ungraduated():
    pol = tg.trace_policy({"a": "sharded", "b": None})
    assert pol["required"] is False
    assert pol["eager_eligible_modules"] == {"b"}
    assert pol["graduated_modules"] == {"a"}


def test_trace_policy_empty():
    pol = tg.trace_policy({})
    assert pol["required"] is False
    assert pol["all_graduated"] is False


def test_trace_engaged():
    assert tg.trace_engaged({"trace_2cq": True, "trace_1cq": False}) is True
    assert tg.trace_engaged({"trace_2cq": False, "trace_1cq": True}) is True
    assert tg.trace_engaged({"trace_2cq": False, "trace_1cq": False}) is False
    assert tg.trace_engaged(None) is False


def test_valid_overflow_proof():
    assert tg.valid_overflow_proof({"required_bytes": 100, "budget_bytes": 50}) is True
    assert tg.valid_overflow_proof({"required_bytes": 50, "budget_bytes": 100}) is False
    assert tg.valid_overflow_proof({"required_bytes": 100}) is False
    assert tg.valid_overflow_proof("nope") is False


def test_verdict_pass_when_trace_engaged():
    pol = tg.trace_policy({"a": "sharded"})
    v, _ = tg.classify_trace_verdict({"trace_2cq": True}, pol)
    assert v == "PASS"


def test_verdict_fail_all_graduated_no_trace_no_proof():
    pol = tg.trace_policy({"a": "sharded", "b": "native"})
    v, r = tg.classify_trace_verdict({"trace_2cq": False, "trace_1cq": False}, pol)
    assert v == "FAIL"
    assert "eager not permitted" in r


def test_verdict_eager_waived_with_valid_proof():
    pol = tg.trace_policy({"a": "sharded"})
    v, r = tg.classify_trace_verdict(
        {"trace_2cq": False},
        pol,
        allow_no_trace=True,
        overflow_proof={"required_bytes": 999, "budget_bytes": 10},
    )
    assert v == "EAGER_WAIVED"
    assert "verified physical overflow" in r


def test_verdict_fail_when_proof_flag_but_no_real_overflow():
    pol = tg.trace_policy({"a": "sharded"})
    v, _ = tg.classify_trace_verdict(
        {"trace_2cq": False},
        pol,
        allow_no_trace=True,
        overflow_proof={"required_bytes": 10, "budget_bytes": 999},
    )
    assert v == "FAIL"


def test_verdict_eager_ok_when_ungraduated_present():
    pol = tg.trace_policy({"a": "sharded", "b": None})
    v, r = tg.classify_trace_verdict({"trace_2cq": False}, pol)
    assert v == "EAGER_WAIVED"
    assert "b" in r


def test_glue_host_op_in_traced_step():
    src = (
        "class P:\n"
        "    def _forward_from_hidden(self, h):\n"
        "        x = ttnn.from_torch(h)\n"
        "        return ttnn.matmul(x, self.w)\n"
    )
    v = tg.glue_trace_violations(src)
    assert any("host-op" in x and "_forward_from_hidden" in x for x in v)


def test_glue_layout_churn_in_traced_step():
    src = (
        "class P:\n"
        "    def decode_step(self):\n"
        "        h = ttnn.tilize(self._h)\n"
        "        return ttnn.matmul(h, self.w)\n"
    )
    v = tg.glue_trace_violations(src)
    assert any("layout-churn" in x for x in v)


def test_glue_clean_traced_step_no_violation():
    src = (
        "class P:\n"
        "    def _forward_from_hidden(self, h):\n"
        "        h = self._layer0(h)\n"
        "        return ttnn.matmul(h, self.w)\n"
    )
    assert tg.glue_trace_violations(src) == []


def test_glue_docstring_mentioning_from_torch_is_not_flagged():
    src = (
        "class P:\n"
        "    def _forward_from_hidden(self, h):\n"
        '        """Pure-ttnn forward. No from_torch / no build -> trace-capturable."""\n'
        "        h = self._layer0(h)\n"
        "        return ttnn.matmul(h, self.w)\n"
    )
    assert tg.glue_trace_violations(src) == []


def test_glue_ignores_setup_functions():
    src = (
        "class P:\n"
        "    def decode_write_inputs(self, ids):\n"
        "        return ttnn.from_torch(torch.full((1, 32), 0))\n"
    )
    assert tg.glue_trace_violations(src) == []


def test_decode_repin_violation_detected():
    src = (
        "class P:\n"
        "    def decode_write_inputs(self, ids):\n"
        "        return self._pin_hidden(ids, self._decode_C)\n"
    )
    assert tg.decode_repin_violation(src) is not None


def _make_demo(tmp_path, statuses, snapshots, pipeline_src):
    demo = tmp_path / "demo_model"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "tt").mkdir(parents=True)
    comps = [{"name": n} for n in statuses]
    (demo / "bringup_status.json").write_text(json.dumps({"components": comps}))
    for name, kind in snapshots.items():
        stub = demo / "_stubs" / (name + ".py")
        stub.write_text("import ttnn\n")
        if kind:
            stub.with_suffix(".py.last_good_" + kind).write_text("import ttnn\n")
    (demo / "tt" / "pipeline.py").write_text(pipeline_src)
    return demo


def test_read_graduation_and_evaluate_all_graduated_fail(tmp_path, monkeypatch):
    import scripts.tt_hw_planner.bringup_loop as bl

    monkeypatch.setattr(
        bl,
        "_stub_has_graduated_any",
        lambda p: p.with_suffix(".py.last_good_native").is_file() or p.with_suffix(".py.last_good_sharded").is_file(),
    )
    monkeypatch.setattr(bl, "_safe_id", lambda n: n)
    demo = _make_demo(
        tmp_path,
        ["a", "b"],
        {"a": "sharded", "b": "native"},
        "class P:\n    def _forward_from_hidden(self, h):\n        return ttnn.matmul(h, self.w)\n",
    )
    grad = tg.read_graduation(demo)
    assert grad == {"a": "sharded", "b": "native"}
    res = tg.evaluate_trace_gate(demo, {"trace_2cq": False, "trace_1cq": False})
    assert res["verdict"] == "FAIL"
    assert res["policy"]["required"] is True


def test_evaluate_ungraduated_allows_eager(tmp_path, monkeypatch):
    import scripts.tt_hw_planner.bringup_loop as bl

    monkeypatch.setattr(
        bl,
        "_stub_has_graduated_any",
        lambda p: p.with_suffix(".py.last_good_native").is_file() or p.with_suffix(".py.last_good_sharded").is_file(),
    )
    monkeypatch.setattr(bl, "_safe_id", lambda n: n)
    demo = _make_demo(
        tmp_path,
        ["a", "b"],
        {"a": "sharded", "b": None},
        "class P:\n    def _forward_from_hidden(self, h):\n        return ttnn.matmul(h, self.w)\n",
    )
    res = tg.evaluate_trace_gate(demo, {"trace_2cq": False, "trace_1cq": False})
    assert res["verdict"] == "EAGER_WAIVED"
    assert res["reasons"] == []


def test_task_of_is_model_agnostic():
    assert tg._task_of("test_gen_text_perf.py") == "gen_text"
    assert tg._task_of("test_main_perf.py") == "main"
    assert tg._task_of("test_image_classification_perf.py") == "image_classification"


def _make_caps_demo(tmp_path):
    demo = tmp_path / "d"
    e2e = demo / "tests" / "e2e"
    e2e.mkdir(parents=True)
    (demo / "tt").mkdir(parents=True)
    (demo / "tt" / "pipeline.py").write_text("x = 1\n")
    caps = e2e / "test_gen_text_perf.py.trace_caps.json"
    caps.write_text(json.dumps({"trace_1cq": False, "trace_2cq": False}))
    return demo, caps


def test_caps_stale_when_missing(tmp_path):
    demo = tmp_path / "d"
    (demo / "tests" / "e2e").mkdir(parents=True)
    assert tg.caps_stale(demo) is True


def test_caps_fresh_when_caps_newer_than_pipeline(tmp_path):
    demo, caps = _make_caps_demo(tmp_path)
    os.utime(demo / "tt" / "pipeline.py", (1000, 1000))
    os.utime(caps, (2000, 2000))
    assert tg.caps_stale(demo) is False


def test_caps_stale_when_pipeline_newer(tmp_path):
    demo, caps = _make_caps_demo(tmp_path)
    os.utime(caps, (1000, 1000))
    os.utime(demo / "tt" / "pipeline.py", (2000, 2000))
    assert tg.caps_stale(demo) is True


def test_run_fresh_no_perf_test(tmp_path):
    demo = tmp_path / "d"
    (demo / "tests" / "e2e").mkdir(parents=True)
    caps, detail = tg.run_fresh_trace_capture(demo)
    assert caps is None
    assert "no perf test" in detail


def test_run_fresh_capture_full_body_runs(tmp_path, monkeypatch):
    demo = tmp_path / "d"
    e2e = demo / "tests" / "e2e"
    e2e.mkdir(parents=True)
    perf = e2e / "test_gen_text_perf.py"
    perf.write_text("def test_gen_text_perf():\n    pass\n")
    caps_file = e2e / "test_gen_text_perf.py.trace_caps.json"

    import sys
    import types

    fake = types.ModuleType("models.experimental.perf_automation.agent.perf_test_gen")

    def _fake_validate(out_path, task, component=False):
        assert task == "gen_text"
        caps_file.write_text(json.dumps({"trace_1cq": False, "trace_2cq": False, "eager_terminal": False}))
        return "invalid", "pipeline could not trace at all"

    fake.validate_generated_perf_test = _fake_validate
    monkeypatch.setitem(sys.modules, "models.experimental.perf_automation.agent.perf_test_gen", fake)
    caps, detail = tg.run_fresh_trace_capture(demo)
    assert caps == {"trace_1cq": False, "trace_2cq": False, "eager_terminal": False}
    assert "could not trace" in detail


def test_explicit_caps_skips_fresh_capture(tmp_path, monkeypatch):
    import scripts.tt_hw_planner.bringup_loop as bl

    monkeypatch.setattr(bl, "_stub_has_graduated_any", lambda p: False)
    monkeypatch.setattr(bl, "_safe_id", lambda n: n)

    def _boom(*a, **k):
        raise AssertionError("fresh capture must NOT run when caps passed explicitly")

    monkeypatch.setattr(tg, "run_fresh_trace_capture", _boom)
    demo = _make_demo(tmp_path, ["a"], {"a": None}, "class P:\n    pass\n")
    res = tg.evaluate_trace_gate(demo, {"trace_2cq": True})
    assert res["verdict"] == "PASS"
    assert res["capture_detail"] is None


def test_glue_ops_from_stream_subtraction():
    op_stream = ["Matmul", "Tilize", "RMSNorm", "Untilize"]
    sigs = {"attn": {"Matmul"}, "norm": {"RMSNorm"}}
    glue = tg.glue_ops_from_stream(op_stream, sigs)
    assert glue == ["Tilize", "Untilize"]


def test_glue_ops_from_stream_empty_sigs():
    assert tg.glue_ops_from_stream(["A", "B"], {}) == ["A", "B"]


def test_is_overflow_marker():
    assert tg._is_overflow("trace region overflow: need more space") is True
    assert tg._is_overflow("Out Of Memory allocating") is True
    assert tg._is_overflow("could not trace at all (path=None)") is False


def test_overflow_fix_loop_resolves_after_growing():
    calls = {"n": 0}

    def _cap(demo):
        calls["n"] += 1
        if calls["n"] < 2:
            return {"trace_2cq": False, "trace_1cq": False}, "trace region overflow"
        return {"trace_2cq": True, "trace_1cq": False}, "ok"

    res = tg.overflow_fix_loop("x", capture_fn=_cap, max_rounds=4)
    assert res["resolved"] is True
    assert res["caps"]["trace_2cq"] is True


def test_overflow_fix_loop_blocker_with_proof():
    def _cap(demo):
        return {"trace_2cq": False, "trace_1cq": False}, "trace region overflow persists"

    res = tg.overflow_fix_loop("x", capture_fn=_cap, max_rounds=3, base_region=1000)
    assert res["resolved"] is False
    assert res["proof"]["required_bytes"] == 8000
    assert res["proof"]["rounds"] == 3


def test_overflow_fix_loop_non_overflow_stops_early():
    calls = {"n": 0}

    def _cap(demo):
        calls["n"] += 1
        return {"trace_2cq": False, "trace_1cq": False}, "could not trace at all (path=None)"

    res = tg.overflow_fix_loop("x", capture_fn=_cap, max_rounds=5)
    assert res["resolved"] is False
    assert res["proof"] is None
    assert calls["n"] == 1


def test_build_fix_directive_repin():
    res = {"verdict": "FAIL", "repin_violation": "decode re-pin ...", "glue_violations": []}
    d = tg.build_fix_directive(res)
    assert "KV-cache single-token decode_step" in d


def test_build_fix_directive_glue():
    res = {"verdict": "FAIL", "repin_violation": None, "glue_violations": ["glue host-op in `f`: torch.full"]}
    d = tg.build_fix_directive(res)
    assert "Port to on-device ttnn" in d


def test_build_fix_directive_none_on_pass():
    assert tg.build_fix_directive({"verdict": "PASS"}) is None


def test_record_trace_verdict_writes_section(tmp_path):
    demo = tmp_path / "d"
    demo.mkdir()
    res = {
        "verdict": "FAIL",
        "reason": "trace did not engage",
        "policy": {"graduated_modules": {"a", "b"}, "eager_eligible_modules": set()},
        "reasons": ["G6 trace-gate: X"],
        "capture_detail": "invalid could not trace",
        "repin_violation": "decode re-pin",
        "glue_violations": [],
    }
    p = tg.record_trace_verdict(demo, res)
    assert p is not None
    txt = (demo / "RUN_REPORT.md").read_text()
    assert "<!-- BEGIN trace-gate -->" in txt
    assert "verdict: **FAIL**" in txt
    assert "fix directive:" in txt


def test_is_l1_overflow_detects_signature():
    assert tg.is_l1_overflow("Circular buffer size 2MB grow to beyond max L1") is True
    assert tg.is_l1_overflow("max l1 size of 1.5MB exceeded") is True
    assert tg.is_l1_overflow("could not trace at all (path=None)") is False
    assert tg.is_l1_overflow("trace region overflow") is False
    assert tg.is_l1_overflow(None) is False


def test_l1_overflow_reason_has_shrink_guidance():
    r = tg.l1_overflow_reason()
    assert "L1_OVERFLOW" in r and "in0_block_w" in r


def test_build_fix_directive_l1_first():
    res = {"verdict": "FAIL", "l1_overflow": True, "repin_violation": None, "glue_violations": []}
    d = tg.build_fix_directive(res)
    assert "Reduce the L1 footprint" in d


def test_evaluate_l1_overflow_resets_and_flags(tmp_path, monkeypatch):
    import scripts.tt_hw_planner.bringup_loop as bl

    monkeypatch.setattr(
        bl,
        "_stub_has_graduated_any",
        lambda p: p.with_suffix(".py.last_good_native").is_file() or p.with_suffix(".py.last_good_sharded").is_file(),
    )
    monkeypatch.setattr(bl, "_safe_id", lambda n: n)
    reset_calls = {"n": 0}
    monkeypatch.setattr(tg, "reclaim_mesh", lambda: reset_calls.__setitem__("n", reset_calls["n"] + 1) or True)
    monkeypatch.setattr(
        tg,
        "run_fresh_trace_capture",
        lambda d, timeout_s=900: (
            {"trace_1cq": False, "trace_2cq": False},
            "invalid circular buffer grow to beyond max L1",
        ),
    )
    demo = _make_demo(
        tmp_path,
        ["a"],
        {"a": "sharded"},
        "class P:\n    def _forward_from_hidden(self, h):\n        return ttnn.matmul(h, self.w)\n",
    )
    res = tg.evaluate_trace_gate(demo, fresh=True)
    assert res["l1_overflow"] is True
    assert reset_calls["n"] == 1
    assert res["verdict"] == "FAIL"
    assert any("L1_OVERFLOW" in r for r in res["reasons"])
