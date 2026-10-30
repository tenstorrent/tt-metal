# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""An emitted demo is a package all the way down to every `__init__.py` the builder left under it.

The builder chooses its own layout each run: Kolibri-1's put the golden under tests/e2e and made that a
package (tests/__init__.py, tests/e2e/__init__.py) with nothing at the demo root. Under pytest's prepend
import mode -- in force wherever the tool clears addopts -- that chain names the test `tests.e2e.<file>`,
which the repo's own top-level `tests` package shadows: 0 collected at optimize's final check. Every
upstream demo that carries tests/__init__.py also carries the root marker; the gate runner now completes
the chain deterministically instead of trusting the agent's layout.
"""

from __future__ import annotations

from pathlib import Path

from scripts.tt_hw_planner.commands import emit_e2e as E
from scripts.tt_hw_planner.op_emitter import _SPDX_HEADER


def _demo(tmp_path: Path, *markers: str) -> Path:
    demo = tmp_path / "some_demo"
    for m in markers:
        p = demo / m / "__init__.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    demo.mkdir(parents=True, exist_ok=True)
    return demo


def test_the_chain_is_completed_up_to_the_demo_root(tmp_path):
    """THE CASE: tests/ and tests/e2e/ are packages, the root is not."""
    demo = _demo(tmp_path, "tests", "tests/e2e")
    written = E._ensure_package_markers(demo)
    assert [p.relative_to(demo).as_posix() for p in written] == ["__init__.py"]
    assert (demo / "__init__.py").read_text() == _SPDX_HEADER, "the tool's own header, from one constant"


def test_a_gap_in_the_middle_is_filled_too(tmp_path):
    demo = _demo(tmp_path, "tests/e2e")  # tests/ itself has no marker
    written = sorted(p.relative_to(demo).as_posix() for p in E._ensure_package_markers(demo))
    assert written == ["__init__.py", "tests/__init__.py"]


def test_a_demo_with_no_markers_is_left_alone(tmp_path):
    demo = _demo(tmp_path)
    (demo / "tests" / "e2e").mkdir(parents=True)
    (demo / "tests" / "e2e" / "test_x.py").write_text("def test_x():\n    pass\n")
    assert E._ensure_package_markers(demo) == []
    assert not (demo / "__init__.py").exists()


def test_a_complete_chain_is_untouched_and_the_step_is_idempotent(tmp_path):
    demo = _demo(tmp_path, "", "tests", "tests/e2e")
    before = (demo / "__init__.py").read_text()
    assert E._ensure_package_markers(demo) == []
    assert (demo / "__init__.py").read_text() == before
    assert (
        E._ensure_package_markers(_demo(tmp_path / "again", "tests", "tests/e2e"))
        and E._ensure_package_markers(tmp_path / "again" / "some_demo") == []
    )


def test_the_gate_runner_completes_the_chain_before_it_looks_at_the_tests(tmp_path, capsys):
    """Behavioural: the runner is the one entry point every builder round passes through."""
    demo = _demo(tmp_path, "tests", "tests/e2e")
    ok, reasons = E._run_deterministic_gates(demo, pcc=0.99, timeout_s=1)
    assert not ok and any("no tests/e2e/test_*.py" in r for r in reasons)  # nothing to run, as expected
    assert (demo / "__init__.py").exists(), "the marker is written before any gate reads the layout"
    assert "package marker written: __init__.py" in capsys.readouterr().out


def test_it_names_no_directory():
    """Whatever marker exists, the chain above it is completed; nothing keys on `tests` or any other name."""
    import inspect

    src = inspect.getsource(E._ensure_package_markers)
    code = "".join(src.replace('"""', "\x00").split("\x00")[::2])
    code = "\n".join(ln for ln in code.splitlines() if not ln.strip().startswith("#"))
    assert '"tests"' not in code and "'tests'" not in code and '"e2e"' not in code
