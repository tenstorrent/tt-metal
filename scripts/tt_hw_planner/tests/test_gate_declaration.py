"""Pin: an emitted package must NAME the one test that is its correctness gate.

Downstream, optimize runs a SINGLE test node after every change and reverts whatever fails it, so
one test decides whether hours of optimization are correct. On 2026-09-23 nothing said which, an
operator picked by name, and the pick was a final-output PCC whose reference had been teacher-forced
onto the pipeline's own output -- a gate that cannot fail for the stages that generate that output.
The stricter test sat unused in the same file for the whole run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.commands.emit_e2e import (
    _declared_gate,
    _gate_declaration_gate,
    _gate_node_ids,
)

_DECLARED = (
    'E2E_CORRECTNESS_GATE = "test_codes_match_the_reference"\n'
    "\n"
    "def test_codes_match_the_reference(evidence):\n"
    "    assert tt_codes == ref_codes\n"
    "\n"
    "def test_final_pcc(evidence):\n"
    "    assert pcc >= 0.99\n"
)
_UNDECLARED = "def test_final_pcc(evidence):\n    assert pcc >= 0.99\n"
_DANGLING = 'E2E_CORRECTNESS_GATE = "test_that_was_renamed"\n\ndef test_final_pcc(e):\n    assert pcc >= 0.99\n'


def _demo(tmp_path: Path, test_src: str | None, name: str = "test_e2e.py", signal: bool = True) -> Path:
    d = tmp_path / "demo"
    (d / "tests" / "e2e").mkdir(parents=True)
    (d / "tt").mkdir(parents=True)
    # A pipeline declares itself a signal renderer by returning the rate it renders at; only
    # those are asked for a gate declaration.
    (d / "tt" / "pipeline.py").write_text(
        "def run(s):\n    return {'waveform': w, 'sampling_rate': r}\n"
        if signal
        else "def run(s):\n    return {'token_ids': ids}\n"
    )
    if test_src is not None:
        (d / "tests" / "e2e" / name).write_text(test_src)
    return d


def test_declared_gate_passes(tmp_path: Path) -> None:
    assert _gate_declaration_gate(_demo(tmp_path, _DECLARED)) is None


def test_undeclared_gate_fails(tmp_path: Path) -> None:
    r = _gate_declaration_gate(_demo(tmp_path, _UNDECLARED))
    assert r and "gate-declaration" in r
    assert "E2E_CORRECTNESS_GATE" in r


def test_a_gate_naming_a_missing_function_fails(tmp_path: Path) -> None:
    r = _gate_declaration_gate(_demo(tmp_path, _DANGLING))
    assert r and "does not exist" in r


def test_reason_says_a_self_referential_test_is_not_the_gate(tmp_path: Path) -> None:
    r = _gate_declaration_gate(_demo(tmp_path, _UNDECLARED))
    assert "reference is built from the pipeline's own output" in r


def test_node_id_is_reported_for_the_operator(tmp_path: Path) -> None:
    nodes = _gate_node_ids(_demo(tmp_path, _DECLARED))
    assert nodes == ["test_e2e.py::test_codes_match_the_reference"]


def test_node_ids_empty_when_nothing_declared(tmp_path: Path) -> None:
    assert _gate_node_ids(_demo(tmp_path, _UNDECLARED)) == []


def test_declaration_is_read_without_importing(tmp_path: Path) -> None:
    src = 'import ttnn  # would need a device\nE2E_CORRECTNESS_GATE = "test_x"\n\ndef test_x(d):\n    assert 1\n'
    d = _demo(tmp_path, src)
    assert _declared_gate(d / "tests" / "e2e" / "test_e2e.py") == "test_x"


def test_unparseable_file_is_safe(tmp_path: Path) -> None:
    assert _gate_declaration_gate(_demo(tmp_path, "def broken(:\n")) is not None


def test_no_tests_dir_is_safe(tmp_path: Path) -> None:
    (tmp_path / "demo").mkdir()
    assert _gate_declaration_gate(tmp_path / "demo") is None


def test_empty_tests_dir_is_safe(tmp_path: Path) -> None:
    assert _gate_declaration_gate(_demo(tmp_path, None)) is None


def test_a_token_output_model_is_never_asked(tmp_path: Path) -> None:
    """Scoped like the signal-quality gate: a model whose output is tokens is compared against an
    independent reference anyway, so it is not asked to declare a gate."""
    assert _gate_declaration_gate(_demo(tmp_path, _UNDECLARED, signal=False)) is None
