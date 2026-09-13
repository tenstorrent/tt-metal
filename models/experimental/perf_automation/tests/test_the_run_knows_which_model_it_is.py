"""The process that writes the FINAL report has to know which model it is, or it reads nothing.

perf_mcp keys every per-run artifact on (model, task) and resolves the model as
`PERF_MCP_MODEL_ROOT or manifest.config.model_root or "."`. run.py put that variable into the env
dict it hands the perf-mcp SERVER, and never onto its own environment -- so anything the run itself
rendered resolved `Path(".").name`, which is "", which fell through to the literal "model".

Measured on voxtral_mini_3b_2507 (2026-09-05). The end-of-run report looked for
perf_mcp_stage_ms_model_main.json, a path no run writes, got {}, and printed a roofline with no
batch, every stage relabelled per-token, memory falsely binding on a compute-bound stage, and an
empty fidelity ladder -- while the LIVE reports, written by the server, were correct all run. Two
reports of one run disagreeing, and the wrong one was the final. The stray
tt_device_recovery_model_main.json, written the same minute, is the fallback's fingerprint.

The same shape was fixed once already for the matmul sweep, whose comment sits directly above the
line this adds to: "nothing ever set that variable, so it landed on '.' ... every lookup returned
None and the whole pre-pass was invisible". One consumer was fixed; the rest were not.

So the fallback goes too. A name that cannot be resolved now refuses, reusing the predicate the
ledger already refuses on, instead of quietly keying a file that cannot exist.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
_CC = PERF / "cc_optimize"
for _p in (str(PERF), str(PERF.parent.parent.parent), str(_CC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _code_of(fn: str, path: Path) -> str:
    """A function's CODE: no docstring, no comments.

    These checks kept failing on prose. The paragraph explaining why the literal fallback was wrong
    contains the word it forbids, and the one naming the models it was measured on contains those.
    What is asserted is what the function DOES.
    """
    src = path.read_text(encoding="utf-8")
    i = src.index("def %s" % fn)
    body = src[i : src.index("\ndef ", i + 10)]
    q = body.find('"""')
    body = body[body.index('"""', q + 3) + 3 :] if q != -1 else body
    return "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))


def _pm(monkeypatch, tmp_path, *, named: bool):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    for _k in ("PERF_MCP_MODEL_NAME", "PERF_MCP_LEDGER", "PERF_MCP_RUN_ID"):
        monkeypatch.delenv(_k, raising=False)
    root = tmp_path / "some_model"
    root.mkdir(exist_ok=True)
    if named:
        monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(root))
    else:
        monkeypatch.delenv("PERF_MCP_MODEL_ROOT", raising=False)
    # the doc this run wrote, keyed on the real name
    (tmp_path / "perf_mcp_stage_ms_some_model_main.json").write_text(
        json.dumps({"run": "a-run", "stages": {"s": 1.0}, "batch": 8})
    )
    spec = importlib.util.spec_from_file_location("pmcp_model_id_%s" % named, str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_a_run_that_knows_its_model_reads_its_own_measurements(monkeypatch, tmp_path):
    m = _pm(monkeypatch, tmp_path, named=True)
    assert m.read_stage_batch() == 8
    assert m.read_stage_ms() == {"s": 1.0}


def test_a_run_that_does_not_know_its_model_reads_nothing(monkeypatch, tmp_path):
    """Not the literal "model" -- nothing. A blank is honest; a wrong key renders as confident."""
    m = _pm(monkeypatch, tmp_path, named=False)
    assert m.read_stage_batch() == 0
    assert m.read_stage_ms() == {}


def test_the_unresolved_name_never_keys_a_file(monkeypatch, tmp_path):
    """perf_mcp_stage_ms_model_main.json is a path no run writes; asking for it can only mislead."""
    m = _pm(monkeypatch, tmp_path, named=False)
    m.read_stage_ms()
    m.read_stage_batch()
    assert not (tmp_path / "perf_mcp_stage_ms_model_main.json").exists()


def test_an_explicit_model_is_enough_on_its_own(monkeypatch, tmp_path):
    """The caller naming it counts as identified, however the environment is set."""
    m = _pm(monkeypatch, tmp_path, named=False)
    assert m.read_stage_ms(model="some_model") == {"s": 1.0}


def test_the_refusal_reuses_the_ledger_predicate():
    """One definition of "is this model known". The ledger already refuses on it."""
    code = _code_of("_read_stage_doc", _CC / "perf_mcp.py")
    assert "is_identified(model)" in code
    assert '"model"' not in code, "the literal fallback is back"


def test_the_run_sets_the_model_root_on_its_own_environment():
    """The env dict is for the server. The run renders reports too."""
    src = (_CC / "run.py").read_text(encoding="utf-8")
    i = src.index('env["PERF_MCP_MODEL_ROOT"] = str((Path(repo_root) / _mrel).resolve())')
    seg = src[i : i + 1400]
    assert 'os.environ["PERF_MCP_MODEL_ROOT"]' in seg, "only the subprocess is told where the run is"


def test_no_model_name_is_typed_into_the_resolution():
    code = _code_of("_read_stage_doc", _CC / "perf_mcp.py")
    for typed in ("voxtral", "gemma", "llama", "nemotron"):
        assert typed not in code.lower(), typed
