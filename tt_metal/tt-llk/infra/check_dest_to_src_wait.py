#!/usr/bin/env python3
"""Warn when a Dest->Src move is issued with no source-bank wait ahead of it.

`MOVD2A` / `MOVD2B` write SrcA / SrcB from Dest. Neither waits for the Matrix Unit to own the
bank it writes (`SrcA/SrcB[MatrixUnit.Src?Bank].AllowedClient == MatrixUnit`): the hardware's
Src auto-wait covers only instructions that READ a source register. Software has to wait on the
bank's valid bit first -- `SRCA_VLD` before `MOVD2A`, `SRCB_VLD` before `MOVD2B` -- or the move
can overwrite a bank the unpacker still owns.

Scope is one function body (see check_mutex_balance.functions). A move is covered when, earlier
in the same body, there is a `STALLWAIT` whose arguments name the matching `SRC?_VLD`, or a call
to a function whose body contains such a wait (`srca_bank_wait()`, `move_d2a_fixed_face()`).
Moves inside a file-local `#define` count at the macro's use site.

This is a WARNING, never a failure: the exit status is always 0. A wait can legitimately live
in a caller, or precede a MOP/replay run that holds the move, and a lexical check cannot see
either. Moves that are only recorded into a replay buffer (`load_replay_buf` / `lltt::record`
without `Exec`) and `TT_OP_MOVD2A` / `TT_OP_MOVD2B` instruction words built for a MOP are not
issued where they are written, so they are not checked: the wait belongs before the replay or
MOP run.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_mutex_balance import _blank_noncode, _signature, functions  # noqa: E402

LLK = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TREE_GLOB = os.path.join(LLK, "tt_llk_*", "**", "*.h")

# (move, the valid bit that must be waited on)
PAIRS = (("MOVD2A", "SRCA_VLD"), ("MOVD2B", "SRCB_VLD"))
_CALL = r"\s*\("
_DEFINE = re.compile(
    r"(?m)^[ \t]*#[ \t]*define[ \t]+(\w+)(?:\([^)]*\))?((?:[^\n]*\\\n)*[^\n]*)"
)
_FN_NAME = re.compile(r"(\w+)\s*(?:<[^;{}()]*>)?\s*\([^;{}]*$")
_RECORD = re.compile(
    r"\b(?P<fn>lltt::record|load_replay_buf)\s*(?:<(?P<targs>[^;{}()]*(?:\([^()]*\))?[^;{}()]*)>)?\s*\("
)
# A recorder that also issues what it records: `<lltt::Exec>`, `ExecBool(Exec)`, or Quasar's
# `exec_while_loading = true` passed as a template or call argument.
_EXECUTES = re.compile(r"(?<!No)\bExec\b|\btrue\b")


def _move_re(move, macros):
    names = [rf"TTI?_{move}"] + [re.escape(m) for m in sorted(macros)]
    return re.compile(r"\b(?:" + "|".join(names) + r")" + _CALL)


def _wait_re(vld, providers):
    stall = rf"\bTTI?_STALLWAIT\s*\([^;]*\b{vld}\b[^;]*\)"
    if not providers:
        return re.compile(stall)
    calls = r"\b(?:" + "|".join(re.escape(p) for p in sorted(providers)) + r")" + _CALL
    return re.compile(f"{stall}|{calls}")


def _move_macros(code, move):
    """File-local macros that expand to the move, followed transitively through other macros."""
    defs = {m.group(1): m.group(2) for m in _DEFINE.finditer(code)}
    found = set()
    while True:
        pat = _move_re(move, found)
        new = {n for n, body in defs.items() if n not in found and pat.search(body)}
        if not new:
            return found
        found |= new


def _paren_end(text, open_idx):
    """Index just past the `)` matching the `(` at open_idx (or len(text) if unbalanced)."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def _recorded_spans(body):
    """Spans of body whose instructions go into a replay buffer without being issued.

    Both recorders default to load-only (`ExecBool E = NoExec`; Quasar's `exec_while_loading =
    false`). `load_replay_buf` records what its callable emits, so the span is the call's
    argument list. `lltt::record` records the next N instructions, which a lexical check cannot
    count, so the span runs to the end of the enclosing block.
    """
    spans = []
    for m in _RECORD.finditer(body):
        targs = m.group("targs") or ""
        open_idx = m.end() - 1
        end = _paren_end(body, open_idx)
        args = body[open_idx:end]
        if _EXECUTES.search(targs) or (
            m.group("fn") == "load_replay_buf" and _EXECUTES.search(args)
        ):
            continue
        spans.append(
            (
                m.start(),
                end if m.group("fn") == "load_replay_buf" else _block_end(body, end),
            )
        )
    return spans


def _block_end(text, start):
    """Index of the `}` closing the block that contains start (or len(text))."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return i
            depth -= 1
    return len(text)


def _name(lines, brace_line):
    head = " ".join(lines[max(brace_line - 3, 0) : brace_line + 1])
    head = head.split("{")[0]
    m = _FN_NAME.search(head)
    return m.group(1) if m else None


def wait_providers(paths):
    """Function names whose body waits on each valid bit -- a call to one covers a later move."""
    providers = {vld: set() for _, vld in PAIRS}
    for path in paths:
        src = open(path, errors="ignore").read()
        if "STALLWAIT" not in src or "_VLD" not in src:
            continue
        lines = _blank_noncode(src).split("\n")
        for brace_line, _end, body, _guard in functions(src):
            for _move, vld in PAIRS:
                if re.search(rf"\bTTI?_STALLWAIT\s*\([^;]*\b{vld}\b", body):
                    name = _name(lines, brace_line)
                    if name:
                        providers[vld].add(name)
    return providers


def scan(path, providers):
    """Yield (line, move, vld, signature) for each move with no matching wait earlier in its body."""
    src = open(path, errors="ignore").read()
    if "MOVD2" not in src:
        return
    code = _blank_noncode(src)
    lines = src.split("\n")
    for move, vld in PAIRS:
        move_pat = _move_re(move, _move_macros(code, move))
        wait_pat = _wait_re(vld, providers[vld])
        for brace_line, _end, body, _guard in functions(src):
            spans = _recorded_spans(body)
            moves = [
                m
                for m in move_pat.finditer(body)
                if not any(a <= m.start() < b for a, b in spans)
            ]
            if not moves:
                continue
            first_wait = wait_pat.search(body)
            first_move = moves[0]
            if first_wait and first_wait.start() < first_move.start():
                continue
            line = brace_line + body.count("\n", 0, first_move.start())
            sig = _signature(lines, brace_line)
            yield line + 1, move, vld, lines[sig].strip()[:76]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "files", nargs="*", help="headers to check (default: the whole LLK tree)"
    )
    args = ap.parse_args()

    tree = sorted(glob.glob(TREE_GLOB, recursive=True))
    files = args.files or tree
    # Wait helpers live in shared headers a commit seldom touches, so find them in the whole tree.
    providers = wait_providers(sorted(set(tree) | set(files)))

    n = 0
    for path in files:
        if not path.endswith((".h", ".hpp")):
            continue
        for line, move, vld, head in scan(path, providers):
            n += 1
            print(
                f"{path}:{line}: warning: {move} with no {vld} wait before it in this function"
            )
            print(f"  in: {head}")
            print(
                f"  {move} does not wait for the Matrix Unit to own the bank it writes. Add"
                f" TTI_STALLWAIT(p_stall::STALL_MATH, ... {vld}) before it, or confirm the"
                " caller waits.\n"
            )
    if n:
        print(
            f"{n} Dest->Src move(s) with no source-bank wait in the same function (warning only)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
