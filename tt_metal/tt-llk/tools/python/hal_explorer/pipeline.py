# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Entry point shared by the CLI and the server: compile, analyse, compare."""

import os
import re

from hal_explorer.analysis import compare
from hal_explorer.analysis.function import Context, analyse
from hal_explorer.gimple import reader
from hal_explorer.isa import instructions
from hal_explorer.isa.origin import SourceIndex
from hal_explorer.model import Result
from hal_explorer.toolchain import build

MODES = ("code", "body")
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
TYPE = re.compile(r"[A-Za-z_][\w:<>, ]*[\s*&]*")


class InputError(Exception):
    """The request itself is invalid; nothing was compiled."""


def load_tables(llk_root=build.LLK_ROOT):
    return instructions.load_tables(llk_root)


def check_variables(variables):
    for index, variable in enumerate(variables):
        type_text = variable.get("type", "").strip()
        name = variable.get("name", "").strip()
        if not TYPE.fullmatch(type_text) or not IDENTIFIER.fullmatch(name):
            raise InputError(
                f"variable {index + 1}: '{type_text} {name}' is not a type followed by a name"
            )


def run(mode, source, tables, variables=(), llk_root=build.LLK_ROOT):
    """Result for one input. mode "code": top-level functions; "body": a replay lambda body."""
    if mode not in MODES:
        raise InputError(f"mode must be one of {', '.join(MODES)}")
    variables = [v for v in variables if v.get("name", "").strip()]
    check_variables(variables)
    if variables and mode != "body":
        raise InputError(
            "variables apply to body mode; in code mode use function parameters"
        )
    try:
        wrapped = (
            build.wrap_body(source, variables)
            if mode == "body"
            else build.wrap_code(source)
        )
        dump, diagnostics, errors = build.compile_source(wrapped, llk_root)
    except build.BuildError as error:
        return Result(False, mode, str(error))
    if dump is None:
        return Result(False, mode, diagnostics, errors)

    functions = reader.parse_functions(dump)
    if mode == "body":
        functions = [f for f in functions if f.name == build.BODY_FUNCTION]
    else:
        # Functions emitted from included headers only have statements located in headers.
        functions = [f for f in functions if build.INPUT_FILE in f.files or not f.files]
    if not functions:
        message = (
            "No function was emitted. Define at least one function that is neither "
            "inline nor static, for example: void my_kernel() { ... }"
        )
        return Result(False, mode, (diagnostics + "\n" + message).strip(), errors)

    sources = SourceIndex(llk_root, build.INPUT_FILE, source)
    root = os.path.realpath(llk_root) + "/"
    contexts, results = [], {}
    for function in functions:
        context = Context(function, tables, sources)
        contexts.append(context)
        results[function.mangled] = analyse(context, function.text.replace(root, ""))
    if mode == "body":
        results[functions[0].mangled].name = "replay body"
        results[functions[0].mangled].signature = "replay body"
    comparisons = []
    if mode == "code":
        pairs, comparisons = compare.pair(contexts)
        for actual, reference in pairs:
            comparisons.append(
                compare.compare(
                    actual,
                    reference,
                    results[actual.function.mangled],
                    results[reference.function.mangled],
                )
            )
    return Result(True, mode, diagnostics, errors, list(results.values()), comparisons)
