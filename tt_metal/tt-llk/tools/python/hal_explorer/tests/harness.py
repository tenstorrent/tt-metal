# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check the explorer against the object file for every function of a source."""

import tempfile
from dataclasses import dataclass, field
from typing import List

import oracle
from hal_explorer.analysis import samples as sampling
from hal_explorer.analysis.function import Context, analyse
from hal_explorer.analysis.interp import Interpreter
from hal_explorer.gimple import reader
from hal_explorer.gimple.types import Pointer, type_info
from hal_explorer.isa.origin import SourceIndex
from hal_explorer.model import Instruction
from hal_explorer.toolchain import build

INTERNAL_WARNINGS = ("internal check failed", "a concrete run took a path")


@dataclass
class Report:
    functions: int = 0
    vectors: int = 0
    skipped: int = 0
    mismatches: List[str] = field(default_factory=list)
    internal: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def with_aliases(source):
    """Older inputs were written against a prelude that declared the HAL aliases."""
    declared = "namespace cfg" in source or "using namespace ckernel" in source
    return source if declared else build.ALIASES + source


def check(source, tables, vector_limit=48):
    report = Report()
    wrapped = build.wrap_code(source)
    dump, diagnostics, _ = build.compile_source(wrapped)
    if dump is None:
        raise AssertionError(f"compile failed:\n{diagnostics}")
    with tempfile.TemporaryDirectory() as tmp:
        code = oracle.functions(oracle.compile_object(wrapped, tmp))
    sources = SourceIndex(build.LLK_ROOT, build.INPUT_FILE, source)
    for function in reader.parse_functions(dump):
        if build.INPUT_FILE not in function.files:
            continue
        report.functions += 1
        context = Context(function, tables, sources)
        result = analyse(context)
        for warning in result.warnings:
            internal = warning["message"].startswith(INTERNAL_WARNINGS)
            target = report.internal if internal else report.warnings
            target.append(f"{function.signature}: {warning['message']}")
        if any(not w["message"].startswith(INTERNAL_WARNINGS) for w in result.warnings):
            report.skipped += 1  # the explorer already says this function is incomplete
            continue
        types = [type_info(t) for t, _ in function.parameters]
        if any(t.pointer or t.bits > 32 for t in types) or function.mangled not in code:
            report.skipped += 1
            continue
        for vector in sampling.vectors(context.input_space(), vector_limit):
            vector = {
                k: (v if isinstance(v, Pointer) else 0) if k in context.globals else v
                for k, v in vector.items()
            }
            arguments = [vector[name] for _, name in function.parameters]
            try:
                expected = oracle.run(code[function.mangled], arguments)
            except oracle.EmulationError:
                report.skipped += 1
                continue
            candidates = []
            # The emulator reads every MMIO register as zero; read the same here.
            interpreter = Interpreter(
                function,
                context.loops,
                tables,
                sources,
                context.definitions,
                read_value=0,
            )
            for trace in interpreter.run(vector):
                items = [i for i in trace.items if isinstance(i, Instruction)]
                if trace.complete and not any(i.runtime_mask for i in items):
                    candidates.append([i.word for i in items])
            if not candidates:
                report.skipped += 1
                continue
            report.vectors += 1
            if expected not in candidates:
                shown = ", ".join(
                    f"{k}={v}" for k, v in vector.items() if not isinstance(v, Pointer)
                )
                report.mismatches.append(
                    f"{function.signature} [{shown}]: object {[hex(w) for w in expected]} "
                    f"explorer {[[hex(w) for w in c] for c in candidates][:2]}"
                )
    return report
