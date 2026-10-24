# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Exercise import routing without importing TTNN or opening devices."""

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "filename,new_path,old_path,symbol,missing",
    [
        ("decoder.py", "models.common.tt_ccl", "models.common.modules.tt_ccl", "TT_CCL", "models.common.tt_ccl"),
        (
            "generator.py",
            "tt_transformers.modules.sampling.sampling_1d",
            "models.common.modules.sampling.sampling_1d",
            "Sampling1D",
            "tt_transformers",
        ),
    ],
)
@pytest.mark.parametrize("layout", ["new", "old", "broken-dependency"])
def test_import_paths_preserve_dependency_errors(monkeypatch, filename, new_path, old_path, symbol, missing, layout):
    path = Path(__file__).parents[1] / "tt" / filename
    tree = ast.parse(path.read_text())
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.Try) and isinstance(node.body[0], ast.ImportFrom) and node.body[0].module == new_path
    )
    expected = object()
    imports = []

    def load(name, *args, **kwargs):
        imports.append(name)
        if name == new_path and layout != "new":
            raise ModuleNotFoundError(name=missing if layout == "old" else "broken_dependency")
        assert name == (old_path if layout == "old" else new_path)
        return SimpleNamespace(**{symbol: expected})

    code = compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec")
    namespace = {}
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", load)
        if layout == "broken-dependency":
            try:
                exec(code, namespace)
            except ModuleNotFoundError as error:
                assert error.name == "broken_dependency"
            else:
                pytest.fail("Missing transitive dependency must propagate")
        else:
            exec(code, namespace)
            assert namespace[symbol] is expected
    assert imports == ([new_path, old_path] if layout == "old" else [new_path])
