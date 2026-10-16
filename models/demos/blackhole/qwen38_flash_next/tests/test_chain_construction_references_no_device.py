# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every module attribute the chain's construction path names exists (no device).

The served chain is built by ``Qwen38TracedChain.open`` / ``construct_chain`` and driven by the server's ``main``; a
name they reach through an imported module (``sampling_step.x``, ``mtp_v2.y``, ``resident_decode.z``) resolves only at
that call, on a device.  A re-export pruned from a module (an import cleaner's doing) then dies at chain open, past
every static pin.  This test walks those functions' ASTs and asserts that each attribute reached through a module
bound in their globals exists on that module, and that each bare global name is bound."""

from __future__ import annotations

import ast
import builtins
import inspect
import types

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as step

CONSTRUCTION = (
    session.Qwen38TracedChain.open,
    session.construct_chain,
    session.Qwen38TracedChain.mtp_enter,
    session.Qwen38TracedChain.close,
    session.Qwen38ChatSession.complete,
    session.Qwen38ChatSession._generate_mtp,
    step.Qwen38SamplingChainExtension.__init__,
    server.main,
)


def _locals_of(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def _unresolved(function) -> list[str]:
    source = inspect.getsource(function)
    tree = ast.parse(source if not source[0].isspace() else "if True:\n" + source)
    globals_ = function.__globals__
    bound = _locals_of(tree)
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            root = node.value.id
            if root in bound or root not in globals_:
                continue
            target = globals_[root]
            if isinstance(target, types.ModuleType) and not hasattr(target, node.attr):
                problems.append(f"{root}.{node.attr}")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in bound or node.id in globals_ or hasattr(builtins, node.id):
                continue
            problems.append(node.id)
    return sorted(set(problems))


@pytest.mark.parametrize("function", CONSTRUCTION, ids=lambda f: f.__qualname__)
def test_every_module_attribute_the_construction_path_names_exists(function) -> None:
    assert _unresolved(function) == []


def test_the_walk_catches_a_pruned_re_export() -> None:
    def broken():  # pragma: no cover - the walk reads it, never runs it
        return step.this_name_was_pruned_by_an_import_cleaner(step.WARM_POLICY)

    def clean():  # pragma: no cover - the walk reads it, never runs it
        return step.WARM_POLICY, session.MTP_DRAFTS, server.MTP_SAMPLED_VARIABLE

    assert _unresolved(broken) == ["step.this_name_was_pruned_by_an_import_cleaner"]
    assert _unresolved(clean) == []  # the walk reads nested bodies too: this test's own body names the pruned one
