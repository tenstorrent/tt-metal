"""A model with several pipelines (demo/demo_<task>.py each with its own e2e PCC test) shares its model code
between them, so the correctness gate has to run EVERY pipeline's e2e PCC test on every edit -- also when
--pcc-test pins one of them. Devstral 2026-10-13..22: six optimize runs gated on text_generation alone while
image_text_to_text's prefill hidden PCC fell from 0.9954 to 0.9904 under the same 0.995 floor (no hardware)."""

from pathlib import Path

from agent import pcc_runner
from agent.before_loop import _pinned_pipelines
from agent.pcc_runner import _gates_verdict, gate_nodes


# ---- before_loop: the pinned gate keeps the other pipelines as gate-only entries -----------------------
def test_pinned_gate_keeps_the_other_pipelines_gate_only():
    discovered = [
        {
            "task": "image_text_to_text",
            "perf_test": "tests/e2e/test_image_perf.py",
            "pcc_test": "tests/e2e/test_e2e_image.py::test_e2e_image",
        },
        {
            "task": "text_generation",
            "perf_test": "tests/e2e/test_text_perf.py",
            "pcc_test": "tests/e2e/test_e2e_text.py::test_e2e_text",
        },
    ]
    out = _pinned_pipelines(
        discovered, "main", "tests/e2e/test_text_perf.py::test_perf", "tests/e2e/test_e2e_text.py::test_e2e_text"
    )
    assert out[0] == {
        "task": "main",
        "perf_test": "tests/e2e/test_text_perf.py::test_perf",
        "pcc_test": "tests/e2e/test_e2e_text.py::test_e2e_text",
    }
    # the image pipeline stays, without a perf test: not optimized, still gated
    assert out[1:] == [
        {"task": "image_text_to_text", "perf_test": None, "pcc_test": "tests/e2e/test_e2e_image.py::test_e2e_image"}
    ]


def test_pinned_gate_with_a_single_pipeline_is_unchanged():
    out = _pinned_pipelines(
        [{"task": "main", "perf_test": "p.py", "pcc_test": "t.py::test_e2e"}], "main", "perf.py::t", "./t.py::test_e2e"
    )
    assert out == [{"task": "main", "perf_test": "perf.py::t", "pcc_test": "./t.py::test_e2e"}]


def test_gate_only_pipelines_are_not_optimized():
    import sys

    perf = Path(__file__).resolve().parents[1]
    for p in (str(perf), str(perf.parent.parent.parent), str(perf / "cc_optimize")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from cc_optimize.run import pipelines_from_manifest

    manifest = {
        "pathmap": {
            "pipelines": _pinned_pipelines(
                [
                    {
                        "task": "image_text_to_text",
                        "perf_test": "x.py",
                        "pcc_test": "tests/e2e/test_e2e_image.py::test_e2e_image",
                    }
                ],
                "main",
                "tests/e2e/test_text_perf.py::test_perf",
                "tests/e2e/test_e2e_text.py::test_e2e_text",
            )
        }
    }
    pipes = pipelines_from_manifest(manifest, "models/demos/m")
    assert [p["task"] for p in pipes] == ["main"]  # the gate-only image pipeline is gated, not optimized


# ---- pcc_runner: every gate node of the manifest runs, each against its own threshold ------------------
class _Ctx:
    def __init__(self, root, pipelines):
        self._root = root
        self.manifest = {
            "pathmap": {
                "pcc": {"end_to_end": {"path": "tests/e2e/test_text.py::test_e2e_text", "threshold": 0.995}},
                "pipelines": pipelines,
            },
            "config": {},
        }

    def model_root(self):
        return self._root


def _model(tmp_path):
    e2e = tmp_path / "tests" / "e2e"
    e2e.mkdir(parents=True)
    (e2e / "test_text.py").write_text("def test_e2e_text():\n    assert pcc >= 0.995\n")
    (e2e / "test_image.py").write_text("def test_e2e_image():\n    assert pcc >= 0.97\n")
    return tmp_path


def test_gate_nodes_lists_every_pipelines_test_with_its_own_threshold(monkeypatch, tmp_path):
    from agent import gitio

    root = _model(tmp_path)
    monkeypatch.setattr(gitio, "repo_root", lambda p: root)
    pipelines = [
        {
            "task": "main",
            "perf_test": "tests/e2e/test_text_perf.py::t",
            "pcc_test": "tests/e2e/test_text.py::test_e2e_text",
        },
        {"task": "image_text_to_text", "perf_test": None, "pcc_test": "tests/e2e/test_image.py::test_e2e_image"},
    ]
    nodes = gate_nodes(_Ctx(root, pipelines))
    assert [Path(n["test"].partition("::")[0]).name for n in nodes] == ["test_text.py", "test_image.py"]
    assert [n["test"].partition("::")[2] for n in nodes] == ["test_e2e_text", "test_e2e_image"]
    assert [n["threshold"] for n in nodes] == [0.995, 0.97]


def test_gate_nodes_dedupes_the_pinned_test(monkeypatch, tmp_path):
    from agent import gitio

    root = _model(tmp_path)
    monkeypatch.setattr(gitio, "repo_root", lambda p: root)
    nodes = gate_nodes(
        _Ctx(root, [{"task": "main", "perf_test": "p.py", "pcc_test": "tests/e2e/test_text.py::test_e2e_text"}])
    )
    assert len(nodes) == 1


def _nodes():
    return [
        {"test": "/m/tests/e2e/test_text.py::test_e2e_text", "threshold": 0.995},
        {"test": "/m/tests/e2e/test_image.py::test_e2e_image", "threshold": 0.97},
    ]


_RUN = (
    "tests/e2e/test_text.py::test_e2e_text [text] prefill hidden PCC min over rows={text}\n"
    "PASSED\n"
    "tests/e2e/test_image.py::test_e2e_image [image] prefill hidden PCC min over rows={image}\n"
    "{image_status}\n"
    "{summary}"
)


def test_both_gates_clear_their_own_floors_is_ok():
    out = _RUN.format(text=0.9955, image=0.980, image_status="PASSED", summary="===== 2 passed in 400.0s =====")
    v = _gates_verdict(out, _nodes())
    assert v["status"] == "ok"
    assert v["pcc"] == 0.980  # the worst reading is the one banked
    assert [g["pcc"] for g in v["gates"]] == [0.9955, 0.980]


def test_a_second_gate_under_its_own_floor_is_pcc_low_even_when_the_pinned_one_clears():
    out = _RUN.format(
        text=0.9955,
        image=0.9604,
        image_status="FAILED",
        summary="=========================== FAILURES ===========================\n"
        "E   AssertionError: image: prefill hidden state off level\n"
        "=========================== short test summary info ===========================\n"
        "FAILED tests/e2e/test_image.py::test_e2e_image - AssertionError\n"
        "===== 1 failed, 1 passed in 400.0s =====",
    )
    v = _gates_verdict(out, _nodes())
    assert v["status"] == "pcc_low"
    assert "test_image.py::test_e2e_image" in v["error"] and "0.9700" in v["error"]


def test_the_loosest_gate_does_not_fail_the_strict_one_by_the_whole_run_number():
    # the image gate (floor 0.97) reads 0.980 -- the worst number of the run -- and that must NOT be judged
    # against the text gate's 0.995 floor
    out = _RUN.format(text=0.9960, image=0.980, image_status="PASSED", summary="===== 2 passed in 400.0s =====")
    assert _gates_verdict(out, _nodes())["status"] == "ok"


def test_a_gate_that_prints_no_pcc_makes_the_run_judge_the_whole_transcript_against_the_strictest_floor():
    out = (
        "tests/e2e/test_text.py::test_e2e_text [text] prefill hidden PCC min over rows=0.980\nPASSED\n"
        "tests/e2e/test_image.py::test_e2e_image no number here\nPASSED\n===== 2 passed in 1.0s ====="
    )
    v = _gates_verdict(out, _nodes())
    assert v["status"] == "pcc_low" and v["threshold"] == 0.995
    assert [g["pcc"] for g in v["gates"]] == [None, None]


def test_run_pcc_runs_every_gate_node_in_one_pytest_run(monkeypatch, tmp_path):
    from agent import gitio, probes

    root = _model(tmp_path)
    monkeypatch.setattr(gitio, "repo_root", lambda p: root)
    monkeypatch.setattr(probes, "wait_for_memory_headroom_before_device_work", lambda *a, **k: None)
    seen = {}

    def _fake_execute(cmd, cwd, env, timeout_s, log_path, **kw):
        seen["cmd"] = list(cmd)
        Path(log_path).write_text(
            _RUN.format(text=0.9955, image=0.980, image_status="PASSED", summary="===== 2 passed in 400.0s =====")
        )
        return 0

    monkeypatch.setattr(probes, "_execute", _fake_execute)
    pipelines = [
        {"task": "main", "perf_test": "p.py::t", "pcc_test": "tests/e2e/test_text.py::test_e2e_text"},
        {"task": "image_text_to_text", "perf_test": None, "pcc_test": "tests/e2e/test_image.py::test_e2e_image"},
    ]
    v = pcc_runner.run_pcc(_Ctx(root, pipelines))
    nodes = [a for a in seen["cmd"] if "::test_e2e_" in a]
    assert [n.rpartition("/")[2] for n in nodes] == ["test_text.py::test_e2e_text", "test_image.py::test_e2e_image"]
    assert v["status"] == "ok" and v["pcc"] == 0.980 and len(v["gates"]) == 2
