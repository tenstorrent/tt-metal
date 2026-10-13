# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A gate that keeps its assertions in a helper of its own package still declares its threshold.

WHAT HAPPENED (Devstral-Small-2, 2026-10-13). The supplied gate, tests/e2e/test_e2e_text_generation.py,
calls check_against_golden() from tests/e2e/common.py, and that helper is where `PCC >= 0.995` and
`assert e2e >= 0.99` live. The policy scanned the gate file alone, found no number, and printed

    correctness gate WARNING: the supplied gate declares no PCC threshold. It is being used as
    given, but if it does not print `PCC: <float>` every edit will be rejected as unverified.

while the gate printed `e2e PCC=0.9998` on every run and had already banked 13 wins. The threshold
check_pcc was given came from the extractor's 0.99 FALLBACK, not from the gate.

THE RULE. The text that declares a gate's threshold is the gate file plus the modules of ITS OWN
package it imports (transitively, in import order). The policy's two readers -- "is a threshold
declared?" and "which number?" -- read that same corpus, so a gate cannot be "declared" and then
handed the fallback.

WHAT MUST NOT CHANGE. A shared library module does not vouch for a gate: models/utility_functions.py
carries comp_pcc's 0.99 default, so a token-accuracy test that imports it would otherwise read as a
PCC gate -- the llama3_1_8b_p150 failure the declared-check exists to catch. Only a module that
resolves to a file in the gate's own directory counts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import pcc_gate_policy
from agent.model_files import _extract_pcc_threshold, gate_package_modules, pcc_threshold_is_declared

PKG = ("models", "demos", "a_model", "tests", "e2e")


def _tree(tmp_path: Path, gate_src: str, **siblings: str) -> Path:
    """A model tree with the gate at models/demos/a_model/tests/e2e/test_gate.py and the given siblings."""
    d = tmp_path.joinpath(*PKG)
    d.mkdir(parents=True)
    for name, src in siblings.items():
        d.joinpath(name + ".py").write_text(src)
    gate = d / "test_gate.py"
    gate.write_text(gate_src)
    return gate


def _no_hf(monkeypatch):
    monkeypatch.setattr(pcc_gate_policy, "hf_reference_available", lambda model_id: False)


def test_a_threshold_in_the_gate_file_itself_is_still_found(tmp_path):
    gate = _tree(tmp_path, "def test_x():\n    assert pcc >= 0.97\n")
    assert pcc_threshold_is_declared(gate)
    assert _extract_pcc_threshold(gate) == pytest.approx(0.97)
    assert gate_package_modules(gate) == [gate]


@pytest.mark.parametrize(
    "import_line",
    [
        "from models.demos.a_model.tests.e2e.common import check_against_golden",
        "from .common import check_against_golden",
        "from . import common",
        "import models.demos.a_model.tests.e2e.common",
    ],
)
def test_a_threshold_declared_in_an_imported_sibling_helper_counts(tmp_path, monkeypatch, import_line):
    gate = _tree(
        tmp_path,
        import_line + "\n\n\ndef test_x(pipe):\n    check_against_golden(pipe)\n",
        common="def check_against_golden(pipe):\n    e2e = pipe.pcc()\n    assert e2e >= 0.995, 'PCC >= 0.995'\n",
    )
    assert gate_package_modules(gate) == [gate, gate.parent / "common.py"]
    assert pcc_threshold_is_declared(gate)
    # the number comes from the helper, not from the extractor's fallback
    assert _extract_pcc_threshold(gate, default=0.5) == pytest.approx(0.995)
    _no_hf(monkeypatch)
    d = pcc_gate_policy.decide(str(gate) + "::test_x", "org/a-model")
    assert d["action"] == pcc_gate_policy.USE_SUPPLIED
    assert "warning" not in d, d


def test_the_gate_file_is_read_before_its_helpers(tmp_path):
    gate = _tree(
        tmp_path,
        "from .common import check\n\n\ndef test_x():\n    assert pcc >= 0.93\n",
        common="def check():\n    assert pcc >= 0.995\n",
    )
    assert _extract_pcc_threshold(gate) == pytest.approx(0.93)


def test_helpers_are_followed_transitively_within_the_package(tmp_path):
    gate = _tree(
        tmp_path,
        "from .common import check\n",
        common="from .floors import PCC_FLOOR\n\n\ndef check():\n    return PCC_FLOOR\n",
        floors="PCC_FLOOR = 0.985  # pcc floor\n",
    )
    d = gate.parent
    assert gate_package_modules(gate) == [gate, d / "common.py", d / "floors.py"]
    assert pcc_threshold_is_declared(gate)
    assert _extract_pcc_threshold(gate) == pytest.approx(0.985)


def test_a_shared_library_module_does_not_vouch_for_a_threshold_less_gate(tmp_path, monkeypatch):
    lib = tmp_path / "models" / "utility_functions.py"
    lib.parent.mkdir(parents=True)
    lib.write_text("def comp_pcc(golden, calculated, pcc=0.99):\n    return True, pcc\n")
    gate = _tree(
        tmp_path,
        "from models.utility_functions import comp_pcc\n\n\ndef test_x(out):\n    assert out.argmax() == 3\n",
    )
    assert gate_package_modules(gate) == [gate]
    assert not pcc_threshold_is_declared(gate)
    assert _extract_pcc_threshold(gate, default=0.5) == 0.5
    _no_hf(monkeypatch)
    d = pcc_gate_policy.decide(str(gate) + "::test_x", "org/a-model")
    assert d["action"] == pcc_gate_policy.USE_SUPPLIED
    assert "declares no PCC threshold" in d["warning"]


def test_a_sibling_outside_the_gates_directory_is_not_followed(tmp_path):
    gate = _tree(tmp_path, "from ..pcc.common import check\nfrom .sub.mod import f\n")
    pcc = tmp_path.joinpath(*PKG[:-1], "pcc")
    pcc.mkdir()
    (pcc / "common.py").write_text("def check():\n    assert pcc >= 0.995\n")
    sub = gate.parent / "sub"
    sub.mkdir()
    (sub / "mod.py").write_text("def f():\n    assert pcc >= 0.995\n")
    assert gate_package_modules(gate) == [gate]
    assert not pcc_threshold_is_declared(gate)


def test_a_package_name_must_spell_the_gates_package_not_just_share_a_basename(tmp_path):
    gate = _tree(
        tmp_path,
        "from models.demos.other_model.tests.e2e.common import check\n",
        common="def check():\n    assert pcc >= 0.995\n",
    )
    assert gate_package_modules(gate) == [gate]
    assert not pcc_threshold_is_declared(gate)


def test_a_gate_that_does_not_parse_contributes_its_text_but_no_imports(tmp_path):
    gate = _tree(
        tmp_path,
        "from .common import check\ndef broken(:\n    assert pcc >= 0.91\n",
        common="def check():\n    assert pcc >= 0.995\n",
    )
    assert gate_package_modules(gate) == [gate]
    assert pcc_threshold_is_declared(gate)
    assert _extract_pcc_threshold(gate) == pytest.approx(0.91)
