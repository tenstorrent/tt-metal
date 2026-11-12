# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Import-dependency tests for models.common.utility_functions.

The implementation lives in ``tt_py_test_utils_common/utility_functions.py``; ``models/common/utility_functions.py``
is a ``sys.modules`` alias onto it. The source-level checks therefore parse the implementation file, and the alias
contract itself is pinned so a future shim rewrite cannot quietly turn it back into a snapshot.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SHIM_PATH = _REPO_ROOT / "models" / "common" / "utility_functions.py"
_IMPL_PATH = _REPO_ROOT / "tt_py_test_utils_common" / "utility_functions.py"


def _top_level_pytest_imports(path):
    syntax_tree = ast.parse(path.read_text())
    return [
        node
        for node in syntax_tree.body
        if (isinstance(node, ast.Import) and any(alias.name == "pytest" for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module == "pytest")
    ]


@pytest.mark.parametrize("path", [_IMPL_PATH, _SHIM_PATH], ids=["implementation", "shim"])
def test_pytest_is_not_imported_at_module_scope(path):
    assert not _top_level_pytest_imports(path), "pytest must remain an optional test-only dependency"


def test_ti_skip_imports_pytest_lazily():
    syntax_tree = ast.parse(_IMPL_PATH.read_text())
    ti_skip = next(node for node in syntax_tree.body if isinstance(node, ast.FunctionDef) and node.name == "ti_skip")

    assert any(
        isinstance(node, ast.Import) and any(alias.name == "pytest" for alias in node.names) for node in ti_skip.body
    ), "ti_skip must import pytest when the test helper is used"


@pytest.mark.parametrize(
    "shim_name, impl_name",
    [
        ("models.common.utility_functions", "tt_py_test_utils_common.utility_functions"),
        ("models.common.tensor_utils", "tt_py_test_utils_common.tensor_utils"),
        ("models.perf.perf_utils", "tt_py_test_utils_common.perf.perf_utils"),
        ("models.perf.device_perf_utils", "tt_py_test_utils_common.perf.device_perf_utils"),
        ("models.perf.benchmarking_utils", "tt_py_test_utils_common.perf.benchmarking_utils"),
        ("tests.ttnn.utils_for_testing", "tt_py_test_utils_common.utils_for_testing"),
    ],
)
def test_legacy_import_path_is_an_alias_of_the_implementation(shim_name, impl_name):
    """The old import paths must resolve to the very same module object, so attribute patches made through either
    path (e.g. ``monkeypatch.setattr(models.perf.device_perf_utils, "run_device_profiler", ...)``) reach the code
    that runs. A ``globals().update()`` snapshot would pass an attribute-equality check but fail this one."""
    for dep in (
        "torch",
        "loguru",
        "ttnn.device",
    ):  # the implementation modules import ttnn; a bare `ttnn` dir resolves as a namespace package
        pytest.importorskip(dep)
    import importlib

    shim = importlib.import_module(shim_name)
    impl = importlib.import_module(impl_name)
    assert shim is impl, f"{shim_name} must alias {impl_name}, got two distinct module objects"


@pytest.mark.parametrize(
    "script",
    ["models/perf/merge_device_perf_results.py", "models/perf/merge_perf_results.py"],
    ids=["merge_device_perf_results", "merge_perf_results"],
)
def test_merge_report_scripts_still_run_as_clis(script):
    """CI invokes these files as scripts (REPORT / CHECK steps of the device-perf pipelines). Importing the
    implementation and aliasing the module must not swallow the ``__main__`` entry point."""
    for dep in ("pandas", "git", "loguru", "tracy"):
        pytest.importorskip(dep)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(_REPO_ROOT), os.environ.get("PYTHONPATH")])))
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / script), "--help"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout, f"{script} did not run its CLI entry point:\n{result.stdout}\n{result.stderr}"
