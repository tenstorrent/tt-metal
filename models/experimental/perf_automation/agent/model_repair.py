# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bring a model into the shape optimize requires — for the clauses that are mechanically repairable.

REPAIR IS TWO JOBS, NOT ONE, and conflating them would produce something worse than the defect.

Sweeping 87 models showed the split plainly. The PORTING findings -- PIPELINE_STAGES, the per-stage
trace hooks, trace_capture_selftest, host_op_selftest -- fire on essentially every hand-written
model, because they are emit-e2e's OUTPUT shape. Producing them from an arbitrary model is a
port: it needs the model's stage decomposition, its resident buffers, its reference outputs. That is
what emit-e2e already does, and a second generator would duplicate it badly and diverge.

The COMPATIBILITY findings are different in kind. They are small, local, and mean the model actively
fights the harness -- a trace gate the harness cannot reach, a depth cap of 0 that a builder reads as
"zero layers", a factory that runs the model instead of returning it. Across 87 models these amount
to a handful of edits, and each is the same edit every time, because the requirement is the same
every time.

So: this repairs compatibility, and refers porting to emit-e2e. It does not attempt to write a
pipeline.

WHAT IT WILL NOT DO. It does not write to the model unless told to. A tool that silently edits source
it did not author is worse than the bug it fixes: the next reader inherits a change nobody made, in a
file they own, with no record of why. The default produces the edits for inspection; `apply=True` is
a decision someone takes.

AND IT VERIFIES ITSELF. After applying, the contract is re-checked -- a repair that does not clear
the clause it targeted is a failed repair, and it says so rather than reporting success and leaving
the run to discover it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

# What the harness sets to demand eager execution. The gate must consult BOTH: the profiler cannot
# attribute per-op time inside a trace, and TT_PERF_TRACE=0 is the harness saying so directly.
_GUARD_SRC = (
    "import os as _os\n"
    "\n"
    'if _os.environ.get("TT_METAL_DEVICE_PROFILER") == "1" or _os.environ.get("TT_PERF_TRACE") == "0":\n'
    "    # The harness asked for EAGER. The device profiler attributes per-op time from eager\n"
    "    # dispatch -- a traced region runs as one fused program and emits none, and synchronising\n"
    "    # inside a capture is fatal (`Event Synchronization is not supported during trace\n"
    "    # capture`). So nothing is traceable while it is on, and the profiler wins: the eager path\n"
    "    # can be measured and the traced one cannot.\n"
    "    return {empty}\n"
)

# The empty answer per gate, by what the function is contracted to return.
_EMPTY_FOR = {
    "get_trace_prefill_supported_seq_lens": "[]",
    "can_enable_trace": "False",
}


@dataclass
class Edit:
    """One repair. `after` is the full new file text, so applying is a single write and reviewing is
    a single diff -- no partial application, no half-repaired file if something raises."""

    path: Path
    clause: str
    what: str
    before: str
    after: str

    @property
    def applied_ok(self) -> bool:
        return self.before != self.after


def _body_insert_point(fn):
    """(line, indent) for the guard: immediately before the first real statement of the body.

    Both come from the AST, not from reading the text. The first version took the line AFTER the
    docstring and read its leading whitespace -- which on a function whose docstring is followed by a
    BLANK line gave an indent of "\n" and an insertion into the gap, producing a file that did not
    parse. The failure was then swallowed by the `except SyntaxError: continue` below and the repair
    silently reported nothing to do, on precisely the two models it exists for.

    A statement's own col_offset is its indentation, exactly, whatever the surrounding blank lines
    look like. Inserting before the docstring would turn it into a bare string expression and delete
    the documentation; inserting after the first real statement would let that statement run first,
    which for a gate is the thing being prevented.
    """
    body = fn.body
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
        and len(body) > 1
    ):
        first = body[1]
    return first.lineno, " " * first.col_offset


_FAILED: list = []


def _repair_trace_authority(model_root: Path) -> list:
    """Make the model's own trace gate consult the harness.

    THE EDIT THAT WAS MADE BY HAND FOR gemma-3, and the one the sweep found still needed on gpt_oss
    and llama3_1_8b_p150 -- both of which have a demo and/or a PCC test, so both are optimizable
    today and both would fail the same way: the harness goes eager, the model traces prefill anyway,
    and the profiled run dies with no data after the weights have loaded.
    """
    edits = []
    for path in sorted(model_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(errors="ignore")
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        if "TT_METAL_DEVICE_PROFILER" in text and "TT_PERF_TRACE" in text:
            continue  # already consults the harness
        lines = text.splitlines(keepends=True)
        # Deepest-first, so an earlier insertion cannot shift a later one's line number.
        targets = sorted(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in _EMPTY_FOR
            ),
            key=lambda n: n.lineno,
            reverse=True,
        )
        if not targets:
            continue
        out = list(lines)
        for fn in targets:
            at, indent = _body_insert_point(fn)
            guard = _GUARD_SRC.format(empty=_EMPTY_FOR[fn.name])
            block = "".join((indent + ln if ln.strip() else ln) + "\n" for ln in guard.split("\n")[:-1])
            out.insert(at - 1, block)
        after = "".join(out)
        try:
            ast.parse(after)  # never hand back a file that does not parse
        except SyntaxError as exc:
            # LOUD, NOT SILENT. This `continue` once hid a bug in the insertion itself and the
            # repair reported "nothing mechanically repairable" for the two models it exists for.
            # A repair that cannot be made is a fact worth stating.
            _FAILED.append((path, "%s line %s" % (exc.msg, exc.lineno)))
            continue
        edits.append(
            Edit(
                path=path,
                clause="trace-authority",
                what="%s: consult TT_METAL_DEVICE_PROFILER / TT_PERF_TRACE before allowing a trace"
                % ", ".join(sorted({f.name for f in targets})),
                before=text,
                after=after,
            )
        )
    return edits


# Only clauses whose repair is the SAME edit every time belong here. Anything needing a judgement
# about this particular model is a port, and goes to emit-e2e.
REPAIRS = {"trace-authority": _repair_trace_authority}


def plan(model_root) -> list:
    """The edits that would make this model compatible. Writes nothing."""
    root = Path(model_root)
    _FAILED.clear()
    out = []
    for _clause, fn in sorted(REPAIRS.items()):
        try:
            out.extend(fn(root))
        except Exception:  # noqa: BLE001 -- one unrepairable file must not lose the others
            continue
    return [e for e in out if e.applied_ok]


def apply(model_root, edits=None) -> dict:
    """Write the edits, then RE-CHECK. A repair that does not clear its clause is a failed repair.

    Returns {"written": [paths], "cleared": bool, "remaining": [findings]}.
    """
    from .model_contract import check

    root = Path(model_root)
    edits = plan(root) if edits is None else edits
    written = []
    for e in edits:
        try:
            e.path.write_text(e.after)
            written.append(str(e.path))
        except OSError:
            continue
    remaining = [f for f in check(root) if f.blocking]
    return {"written": written, "cleared": not remaining, "remaining": remaining}


def report(edits, model_root) -> str:
    if not edits:
        why = ("; could not patch: " + ", ".join("%s (%s)" % (p.name, w) for p, w in _FAILED)) if _FAILED else ""
        return "  [repair] %s: nothing mechanically repairable%s" % (Path(model_root).name, why)
    head = "  [repair] %s: %d edit(s) would make it compatible" % (Path(model_root).name, len(edits))
    return "\n".join([head] + ["    %s  [%s] %s" % (e.path.name, e.clause, e.what) for e in edits])
