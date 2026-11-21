"""Every site that clears pytest.ini's addopts starts from probes.pytest_argv, which keeps the repo's
`--import-mode=importlib`. Clearing addopts alone dropped it: an emitted demo with tests/__init__.py and no
demo-root __init__.py was then named `tests.e2e.<file>` and shadowed by tt-metal's top-level `tests`
package -- "perf test selects no cases ... 0 collected" at optimize's final check (Kolibri-1, 2026-10-08).
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

from agent import pcc_gate_gen, pcc_runner, probes


def test_the_argv_clears_addopts_and_keeps_the_import_mode():
    argv = probes.pytest_argv("x.py", "-q")
    assert argv[:3] == [sys.executable, "-m", "pytest"]
    assert argv[argv.index("-o") + 1] == "addopts="
    assert "--import-mode=importlib" in argv
    assert argv[-2:] == ["x.py", "-q"]
    assert probes.pytest_argv() is not probes.pytest_argv(), "a fresh list per call: callers append to it"


def test_every_site_that_clears_addopts_goes_through_the_one_argv():
    """No second spelling of the prefix: the literal `addopts=` lives in pytest_argv and nowhere else."""
    for mod in (probes, pcc_runner, pcc_gate_gen):
        src = inspect.getsource(mod)
        own = inspect.getsource(probes.pytest_argv) if mod is probes else ""
        assert src.replace(own, "").count('"addopts="') == 0, f"{mod.__name__} builds its own pytest prefix"
    for fn in (probes.collect_cases, probes.preflight_collect):
        assert "pytest_argv(" in inspect.getsource(fn), fn.__name__
    assert "probes.pytest_argv(" in inspect.getsource(pcc_runner)
    assert "probes.pytest_argv(" in inspect.getsource(pcc_gate_gen)


def _shadowed_layout(root: Path) -> Path:
    """tt-metal's shape in miniature: a top-level `tests` package, and a demo whose tests tree is a package
    chain that stops short of the demo root."""
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("")
    # tt-metal's root conftest imports from its `tests` package before any test module is collected, so the
    # name is already bound in sys.modules when the demo's own `tests` would be looked up.
    (root / "conftest.py").write_text("import tests  # noqa: F401\n")
    e2e = root / "models" / "demos" / "some_demo" / "tests" / "e2e"
    e2e.mkdir(parents=True)
    (e2e.parent / "__init__.py").write_text("")
    (e2e / "__init__.py").write_text("")
    test = e2e / "test_main_perf.py"
    test.write_text("def test_main_perf():\n    assert True\n")
    return test


def _collected(cmd, cwd) -> int:
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
    text = out.stdout + out.stderr
    import re

    m = re.search(r"(\d+)\s+tests? collected", text)
    return int(m.group(1)) if (m and out.returncode == 0) else 0


def test_the_shadowed_layout_collects_with_the_argv_and_not_without_the_import_mode(tmp_path):
    """THE CASE, reproduced end to end with the real pytest: the same file, the same cwd."""
    test = _shadowed_layout(tmp_path)
    rel = str(test.relative_to(tmp_path))
    assert _collected(probes.pytest_argv(rel, "--collect-only", "-q", "-p", "no:cacheprovider"), tmp_path) == 1
    bare = [sys.executable, "-m", "pytest", "-o", "addopts=", rel, "--collect-only", "-q", "-p", "no:cacheprovider"]
    assert _collected(bare, tmp_path) == 0, "without the import mode the demo's `tests` is shadowed -- the bug"


def test_preflight_collect_sees_the_case_in_the_shadowed_layout(tmp_path):
    test = _shadowed_layout(tmp_path)
    n = probes.preflight_collect(tmp_path, str(test.relative_to(tmp_path)), "test_main_perf")
    assert n == 1
