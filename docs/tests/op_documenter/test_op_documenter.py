# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Run with pytest --confcutdir=docs/tests docs/tests/op_documenter.

Requires Sphinx (docs/requirements-docs.txt), pytest, nanobind==2.14.0 and a
C++17 compiler. A tiny real nanobind extension replaces the device runtime.
"""

from __future__ import annotations

import importlib
import io
import pathlib
import shutil
import subprocess
import sys
import sysconfig
from types import SimpleNamespace

import pytest

HERE = pathlib.Path(__file__).resolve().parent
EXTENSIONS = HERE.parents[1] / "source" / "ttnn" / "_ext"


@pytest.fixture(scope="module")
def bindings_dir(tmp_path_factory):
    import nanobind

    directory = tmp_path_factory.mktemp("nanobind")
    compiler = shutil.which("c++")
    assert compiler, "The signature tests require a C++17 compiler"
    nb = pathlib.Path(nanobind.__file__).parent
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-fvisibility=hidden",
            "-O0",
            f"-I{nb / 'include'}",
            f"-I{nb / 'ext' / 'robin_map' / 'include'}",
            f"-I{sysconfig.get_path('include')}",
            str(HERE / "bindings.cpp"),
            str(nb / "src" / "nb_combined.cpp"),
            "-o",
            str(directory / f"_op_documenter_bindings{sysconfig.get_config_var('EXT_SUFFIX')}"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return directory


@pytest.fixture(scope="module")
def documenter(bindings_dir):
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(bindings_dir))
        patch.syspath_prepend(str(HERE / "fixture"))
        patch.syspath_prepend(str(EXTENSIONS))
        module = importlib.import_module("op_documenter")
        yield module
        for name in ("op_documenter", "ttnn", "ttnn.decorators"):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("name", ["FastOperationDocumenter", "OperationDocumenter", "OperationDataDocumenter"])
def test_documenters_use_real_binding_names_types_and_defaults(documenter, name):
    import ttnn

    instance = object.__new__(getattr(documenter, name))
    instance.object = ttnn.layer_norm
    instance.get_doc = lambda: [instance.object.__doc__.splitlines()]
    assert instance.format_signature() == (
        "(input_tensor: ttnn.Tensor, *, epsilon: float = 1e-12, "
        "compute_kernel_config: object | None = None, recip_tensor: object | None = None) -> ttnn.Tensor"
    )


def test_every_overload_is_rendered_even_with_an_overloaded_docstring(documenter):
    import ttnn

    assert documenter._binding_signature(ttnn.overloaded).splitlines() == [
        "(value: int = 0) -> int",
        "(value: str = 'literal\\\\0') -> str",
    ]


def test_explicit_default_expression_is_not_quoted(documenter):
    from _op_documenter_bindings import custom_default

    assert (
        documenter._binding_signature(SimpleNamespace(function=custom_default)) == "(value: int = DEFAULT_VALUE) -> int"
    )


def test_python_signature_does_not_depend_on_docstring_layout(documenter):
    import ttnn

    assert documenter._binding_signature(ttnn.transpose) == "(input_tensor, dim1, dim2, *, memory_config=None)"


def test_default_substitution_preserves_literal_backslashes_and_multi_digit_indices(documenter):
    function = SimpleNamespace(
        __nb_signature__=(
            ("def many(" + ", ".join(f"x{i} = \\{i}" for i in range(12)) + ")", None, ("\\1",) + tuple(range(1, 12))),
        )
    )
    assert documenter._binding_signature(SimpleNamespace(function=function)) == (
        "(x0 = '\\\\1', " + ", ".join(f"x{i} = {i}" for i in range(1, 12)) + ")"
    )


def test_sphinx_emits_complete_signatures_for_all_documenter_kinds(bindings_dir, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "conf.py").write_text(
        "import sys\n"
        f"sys.path[:0] = {[str(HERE / 'fixture'), str(bindings_dir), str(EXTENSIONS)]!r}\n"
        "extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'op_documenter']\n"
        "project = 'operation-signatures'\n"
        "html_theme = 'basic'\n"
    )
    (source / "index.rst").write_text(
        "Operation signatures\n====================\n\n"
        ".. autofastoperation:: ttnn.layer_norm\n\n"
        ".. autooperation:: ttnn.overloaded\n\n"
        ".. autooperationdata:: ttnn.transpose\n"
    )
    output = tmp_path / "html"
    from sphinx.application import Sphinx

    status, warnings = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "path", sys.path.copy())
        app = Sphinx(
            str(source),
            str(source),
            str(output),
            str(tmp_path / "doctrees"),
            "html",
            status=status,
            warning=warnings,
            freshenv=True,
            warningiserror=True,
        )
        app.build(force_all=True)
    assert app.statuscode == 0, warnings.getvalue() or status.getvalue()

    from lxml import html

    page = html.parse(str(output / "index.html"))
    signatures = ["".join(node.itertext()) for node in page.xpath('//dt[contains(@class, "sig")]')]
    assert len(signatures) == 4, signatures
    assert "recip_tensor" in signatures[0] and "epsilon" in signatures[0] and "1e-12" in signatures[0]
    assert "int" in signatures[1] and "str" in signatures[2]
    assert "dim1" in signatures[3] and "dim2" in signatures[3] and "memory_config" in signatures[3]
    assert all("self" not in signature and "_ttnn" not in signature for signature in signatures)
