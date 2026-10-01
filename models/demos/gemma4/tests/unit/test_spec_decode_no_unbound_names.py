# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""No method in the speculative serving path reads a name it never binds.

This is a static check because the alternative is a device run. Extracting
``_srv_upload_iter`` out of ``serving_step`` left ``K`` behind in the caller, so
every MTP decode raised

    NameError: name 'K' is not defined

at the first serving iteration -- which cost a full 12B benchmark leg to find,
and which no host test could reach because the method's body is ttnn uploads
against a captured trace. A NameError needs no device to detect.

The check is deliberately narrow: only module-level names, builtins, imports,
parameters and local assignments count as bound, so a genuinely free variable
is reported while normal code is not.
"""

import ast
import builtins
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "tt" / "spec_decode.py"


def _module():
    return ast.parse(SOURCE.read_text())


def _bound_at_module_level(tree):
    names = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _free_names(fn, module_names):
    """Names a function reads without binding them anywhere in its own body."""
    bound = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
    if fn.args.vararg:
        bound.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        bound.add(fn.args.kwarg.arg)
    loads = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, (ast.Store, ast.Del)) else loads).add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return loads - bound - module_names


def _functions(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def test_srv_upload_iter_binds_every_name_it_reads():
    """The method that actually regressed, named so the failure is unmistakable."""
    tree = _module()
    module_names = _bound_at_module_level(tree)
    target = [fn for fn in _functions(tree) if fn.name == "_srv_upload_iter"]
    assert target, "_srv_upload_iter not found in spec_decode.py"
    assert _free_names(target[0], module_names) == set()


@pytest.mark.parametrize(
    "name",
    [
        "serving_step",
        "serving_reseed",
        "serving_setup",
        "serving_warmup_widths",
        "_srv_upload_iter",
        "_srv_capture_width",
        "srv_width_for",
        "refresh_page_tables",
    ],
)
def test_serving_path_methods_bind_every_name_they_read(name):
    tree = _module()
    module_names = _bound_at_module_level(tree)
    for fn in _functions(tree):
        if fn.name == name:
            free = _free_names(fn, module_names)
            assert free == set(), f"{name} reads unbound name(s): {sorted(free)}"
            return
    pytest.skip(f"{name} is not defined in spec_decode.py")
