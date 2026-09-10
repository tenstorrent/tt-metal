"""Direct tt-metal optimization: --pcc-test supplies the e2e correctness gate to the NORMAL LLM
discovery (it still maps the model), and the PCC test may live OUTSIDE the model code dir.

- resolve_pcc_node: a tt-root-relative/absolute PCC node -> model-root-relative (with '..' when the
  test is in a sibling folder) + threshold lifted from the test text.
- read_model_files(pcc_override=...): discovery still runs; the pinned PCC is injected as the e2e
  gate so a scattered/undiscovered PCC test no longer fails validation.
"""

import json

import pytest

from agent.model_files import ModelFilesError, read_model_files, resolve_pcc_node


def _tree(tmp_path):
    tt = tmp_path
    model_root = tt / "models" / "demos" / "mymodel"
    model_root.mkdir(parents=True)
    (model_root / "model.py").write_text("import ttnn\ndef forward(x):\n    return x\n")
    pcc = tt / "models" / "demos" / "wormhole" / "mymodel" / "tests" / "pcc"
    pcc.mkdir(parents=True)
    (pcc / "test_pcc.py").write_text("def test_x():\n    assert_with_pcc(a, b, 0.98)\n")
    return tt, model_root


def test_resolve_pcc_outside_model_root_uses_dotdot_and_lifts_threshold(tmp_path):
    tt, model_root = _tree(tmp_path)
    node, thr, abs_path = resolve_pcc_node(
        model_root, "models/demos/wormhole/mymodel/tests/pcc/test_pcc.py::test_x", tt
    )
    assert node == "../wormhole/mymodel/tests/pcc/test_pcc.py::test_x"
    assert thr == 0.98  # lifted from assert_with_pcc(..., 0.98)
    assert abs_path.name == "test_pcc.py"


def test_resolve_explicit_threshold_overrides(tmp_path):
    tt, model_root = _tree(tmp_path)
    node, thr, _ = resolve_pcc_node(model_root, "models/demos/wormhole/mymodel/tests/pcc/test_pcc.py::test_x", tt, 0.95)
    assert thr == 0.95


def test_resolve_missing_file_raises(tmp_path):
    tt, model_root = _tree(tmp_path)
    with pytest.raises(ModelFilesError):
        resolve_pcc_node(model_root, "models/demos/nope/test_x.py::t", tt)


def test_discovery_still_runs_and_injects_pinned_gate(tmp_path):
    """read_model_files runs the (mock) discovery sub-agent, but the e2e gate is the pinned --pcc-test
    even though the sub-agent reported a different (or no) PCC. The model_files it discovered survive."""
    tt, model_root = _tree(tmp_path)
    (model_root / "tests").mkdir()
    (model_root / "tests" / "test_perf.py").write_text("def test_perf():\n    pass\n")
    node, thr, _ = resolve_pcc_node(model_root, "models/demos/wormhole/mymodel/tests/pcc/test_pcc.py::test_x", tt)

    def fake_discovery(_prompt):
        # the sub-agent maps the model BUT also reports OTHER tests it found scattered OUTSIDE
        # model_root (a perf test + a per-component PCC) — these must NOT fail validation in this mode.
        return json.dumps(
            {
                "pcc": {
                    "end_to_end": {"path": "tests/test_perf.py", "threshold": 0.5},
                    "attention": {"path": "../wormhole/mymodel/tests/pcc/test_attn.py::t", "threshold": 0.99},
                },
                "perf_test": {"path": "../wormhole/mymodel/tests/perf/test_perf.py", "case": "test_perf"},
                "model_files": ["model.py"],
                "flags": [],
            }
        )

    pm = read_model_files(model_root, fake_discovery, pcc_override={"path": node, "threshold": thr})
    assert pm["pcc"]["end_to_end"]["path"] == "../wormhole/mymodel/tests/pcc/test_pcc.py::test_x"
    assert pm["pcc"]["end_to_end"]["threshold"] == 0.98  # pinned gate, not the sub-agent's 0.5
    assert "model.py" in pm["model_files"]  # discovery's mapping survives
    assert list(pm["pcc"].keys()) == ["end_to_end"]  # scattered component PCC dropped
    assert pm["components"] == {}  # not needed; gate is the pinned e2e PCC only
