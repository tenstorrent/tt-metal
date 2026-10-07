#!/usr/bin/env python3
"""Fail on an instruction argument drawn from the wrong constant family.

Two checks, both lexical and both silent in the compiler, because every constant involved is a
plain integer:

1. STALLWAIT / SEMWAIT argument roles. The first argument is the stall class (what to stall:
   `p_stall::STALL_*`); the rest are wait resources or, for SEMWAIT, the semaphore condition
   (`p_stall::STALL_ON_ZERO` / `STALL_ON_MAX`). Stall-class and wait-resource masks overlap
   bitwise, so a swapped or misplaced constant still assembles -- into a different wait.
   Argument layout per architecture (ckernel_ops.h):

     STALLWAIT  WH/BH  (stall_res, wait_res)
                Quasar (stall_res, wait_res_idx_2, wait_res_idx_1, wait_res_idx_0)
     SEMWAIT    WH/BH  (stall_res, sem_sel, wait_sem_cond)
                Quasar (stall_res, wait_sem_cond, sem_bank_sel, sem_sel)

   The stall-class slot must name only `STALL_*` classes (no wait resource, no `NONE` /
   `NOTHING`, no bare integer literal); the condition slot only `STALL_ON_*`; no other slot
   any `STALL_*` name.

2. Instruction-scoped modifier constants. A constant whose family names one instruction --
   sfpi's `SFP<X>_MOD<k>_*`, or a `p_<instr>::` struct listed in INSTR_FAMILIES -- encodes
   that instruction's field, so it must not be an argument of a different instruction. The
   innermost call around the constant decides: an instruction macro (`TT_OP_` / `TTI_` /
   `TT_`) is checked; a constant passed to any other function -- including sfpi's
   `__builtin_rvtt_*` intrinsics, whose `sfpx*` forms expand to several instructions -- is
   forwarded, and not checked. A cast (`static_cast<T>(c)`, `(T)c`, `T(c)`) is not a call: it
   keeps the value, so the constant is still the enclosing instruction's argument. Families
   that several instructions share by design (`p_setrwc`, `p_unpacr`, `p_movd2a`, `p_mova2d`,
   ...) are deliberately not listed.

The verdict depends on this file, on the lexer in check_mutex_balance.py and on the
`TT_OP_*` definitions in `ckernel*_ops.h`; a commit touching any of those rescans every header.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_mutex_balance import _blank_noncode  # noqa: E402

# `p_<instr>::` structs whose every constant belongs to that one instruction (measured: no use
# inside any other instruction anywhere in the LLK trees). Value: accepted instruction names.
INSTR_FAMILIES = {
    "p_adddmareg": {"ADDDMAREG"},
    "p_cleardvalid": {"CLEARDVALID"},
    "p_mop": {"MOP"},
    "p_movb2a": {"MOVB2A"},
    "p_movb2d": {"MOVB2D"},
    "p_reg2flop": {"REG2FLOP", "REG2FLOP_COMMON"},
    "p_setdmareg": {"SETDMAREG"},
    "p_sfpconfig": {"SFPCONFIG"},
    "p_sfpexexp": {"SFPEXEXP"},
    "p_sfpgt": {"SFPGT"},
    "p_sfpiadd": {"SFPIADD"},
    "p_sfploadi": {"SFPLOADI"},
    "p_sfpnonlinear": {"SFPNONLINEAR"},
    "p_sfpshft2": {"SFPSHFT2"},
    "p_sfpswap": {"SFPSWAP"},
    "p_unpacr_nop": {"UNPACR_NOP"},
    "p_zeroacc": {"ZEROACC"},
    "p_zerosrc": {"ZEROSRC"},
}

_ROLE_CALL = re.compile(r"\b(?:TT_OP|TTI|TT)_(STALLWAIT|SEMWAIT)\s*\(")
_STALL_NAME = re.compile(r"\bp_stall::(\w+)")
_INT_LITERAL = re.compile(r"^(?:0[xX][0-9a-fA-F']+|0[bB][01']+|\d[\d']*)[uUlL]*$")
_CALL_NAME = re.compile(r"((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)\s*$")
_INSN = re.compile(r"^(?:TT_OP_|TTI_|TT_)([A-Z][A-Z0-9_]*)$")
_HERE = os.path.dirname(os.path.abspath(__file__))
_LLK = os.path.normpath(os.path.join(_HERE, ".."))
_OPS_HEADER = re.compile(r"^ckernel\w*_ops\.h$")
# Keywords whose `(` is grouping, and conversions that keep the value: a constant inside them is
# still the enclosing instruction's argument.
_GROUPING = {"if", "while", "for", "switch", "return", "sizeof", "decltype"}
_CASTS = {"static_cast", "reinterpret_cast", "const_cast", "dynamic_cast"}
_TYPES = {
    "bool",
    "char",
    "short",
    "int",
    "long",
    "unsigned",
    "size_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "uint",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
}


def _instruction_names():
    """Every `TT_OP_<name>` the LLK op headers define: only these are instruction calls."""
    names = set()
    for path in glob.glob(
        os.path.join(_LLK, "tt_llk_*", "common", "inc", "ckernel*_ops.h")
    ):
        with open(path) as f:
            names |= set(re.findall(r"#\s*define\s+TT_OP_(\w+)\s*\(", f.read()))
    if not names:
        sys.exit(
            "check_instr_encoding_args: no TT_OP_* definitions found under "
            f"{_LLK}/tt_llk_*/common/inc/ckernel*_ops.h -- refusing to pass while checking nothing"
        )
    return names


INSTRUCTIONS = _instruction_names()
_CONST = re.compile(r"\b(SFP[A-Z0-9]+)_MOD\d_\w+|\b(p_[a-z0-9_]+)::\w+")


def _line(code, pos):
    return code.count("\n", 0, pos) + 1


def _split_args(code, open_idx):
    """Top-level arguments of the call whose `(` is at open_idx, with each one's start offset."""
    depth, start, out = 0, open_idx + 1, []
    for i in range(open_idx, len(code)):
        c = code[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                out.append((code[start:i], start))
                return out
        elif c == "," and depth == 1:
            out.append((code[start:i], start))
            start = i + 1
    return None


def _role_slots(insn, nargs):
    """(stall-class slot, condition slot or None), or None for an arity this check doesn't know."""
    if insn == "STALLWAIT" and nargs in (2, 4):
        return 0, None
    if insn == "SEMWAIT" and nargs == 3:
        return 0, 2
    if insn == "SEMWAIT" and nargs == 4:
        return 0, 1
    return None


def check_roles(code):
    for m in _ROLE_CALL.finditer(code):
        insn = m.group(1)
        args = _split_args(code, m.end() - 1)
        slots = _role_slots(insn, len(args)) if args else None
        if not slots:
            continue
        stall_slot, cond_slot = slots
        for k, (text, start) in enumerate(args):
            names = _STALL_NAME.findall(text)
            if k == stall_slot:
                if _INT_LITERAL.match(text.strip()):
                    yield _line(code, start), (
                        f"{insn} stall class is the literal `{text.strip()}`; name a"
                        " p_stall::STALL_* class"
                    )
                bad = [
                    n
                    for n in names
                    if not n.startswith("STALL_") or n.startswith("STALL_ON_")
                ]
                what = "the stall class (argument 1)"
            elif k == cond_slot:
                bad = [n for n in names if not n.startswith("STALL_ON_")]
                what = f"the semaphore condition (argument {k + 1})"
            else:
                bad = [n for n in names if n.startswith("STALL_")]
                what = f"a wait/semaphore argument (argument {k + 1})"
            for n in bad:
                yield _line(code, start), f"{insn}: p_stall::{n} is not valid as {what}"


def _norm(name):
    return name.replace("_", "").upper()


# Inside a template argument list only type-like text appears: names, `::`, digits, commas,
# `*`, a single `&`, nested `<>`. Anything else (an operator, `?`, a paren) means the `<` was
# a comparison.
_TEMPLATE_BODY = re.compile(r"^(?:[\w\s:,*<>]|&(?!&))*$")


def _template_spans(code):
    """(start, end) of every `<...>` template argument list, so a constant inside one is a
    template argument (forwarded), while a `<` that is a comparison opens nothing."""
    spans = []
    for m in re.finditer(r"\w\s*(<)(?![<=])", code):
        start = m.start(1)
        depth = 0
        for j in range(start, len(code)):
            c = code[j]
            if c == "<":
                depth += 1
            elif c == ">":
                depth -= 1
                if depth == 0:
                    body = code[start + 1 : j]
                    nxt = code[j + 1 :].lstrip()[:2]
                    if _TEMPLATE_BODY.match(body) and (
                        nxt[:1] in ("(", "{", ",", ")", ";", ">", "") or nxt == "::"
                    ):
                        spans.append((start, j))
                    break
            elif not (c.isalnum() or c in "_ \t\n:,*&"):
                break
    return spans


def _template_callee(code, i):
    """For a `(` that follows `name<...>`, the name; None if there is no such template."""
    j = i - 1
    while j >= 0 and code[j].isspace():
        j -= 1
    if j < 0 or code[j] != ">":
        return None
    depth = 0
    while j >= 0:
        if code[j] == ">":
            depth += 1
        elif code[j] == "<":
            depth -= 1
            if depth == 0:
                break
        j -= 1
    m = _CALL_NAME.search(code, max(0, j - 80), j)
    return m.group(1) if m else None


def _opener(code, i):
    """What the `(` at i opens: an instruction name, "" for any other call, None for grouping
    (control keywords, parenthesized expressions and value-preserving casts)."""
    m = _CALL_NAME.search(code, max(0, i - 80), i)
    if not m:
        callee = _template_callee(code, i)
        if callee is None:
            return None  # a parenthesized expression or a C-style cast `(T)c`
        return None if callee.split("::")[-1] in _CASTS else ""
    name = m.group(1)
    base = name.split("::")[-1]
    ins = _INSN.match(base)
    # TT_<X>_VALID range-checks X's operands, so its arguments are X's
    insn = ins and re.sub(r"_VALID$", "", ins.group(1))
    if insn in INSTRUCTIONS:
        return insn
    if base in _GROUPING or base in _TYPES:
        return None  # grouping, or a functional cast `T(c)`
    return ""


def check_families(code):
    opener = {i: _opener(code, i) for i, c in enumerate(code) if c == "("}
    forwarded = _template_spans(code)
    consts = {m.start(): m for m in _CONST.finditer(code)}
    stack = []  # (opener, offset of the `(`)
    for i, c in enumerate(code):
        if c == "(":
            stack.append((opener[i], i))
        elif c == ")" and stack:
            stack.pop()
        m = consts.get(i)
        if not m:
            continue
        inner = next((x for x, _ in reversed(stack) if x is not None), None)
        if not inner or any(a < i < b for a, b in forwarded):
            continue  # not an instruction argument, or forwarded as a template argument
        if m.group(1):
            if _norm(m.group(1)) != _norm(inner):
                yield _line(code, i), (
                    f"{m.group(0)} is an {m.group(1)} modifier, passed to {inner}"
                )
        elif m.group(2) in INSTR_FAMILIES and inner not in INSTR_FAMILIES[m.group(2)]:
            msg = f"{m.group(0)} belongs to {m.group(2)}, passed to {inner}"
            yield _line(code, i), msg


def scan(path):
    code = _blank_noncode(open(path, errors="ignore").read())
    yield from check_roles(code)
    yield from check_families(code)


_RESCAN_TRIGGERS = {
    os.path.abspath(__file__),
    os.path.join(
        _HERE, "check_mutex_balance.py"
    ),  # the lexer every verdict goes through
}


def rescans_all(paths):
    """A change to the checker, its lexer or an instruction-definition header moves the verdict
    on every header, not just the touched ones."""
    return any(
        os.path.abspath(p) in _RESCAN_TRIGGERS or _OPS_HEADER.match(os.path.basename(p))
        for p in paths
    )


def all_headers():
    repo = os.path.normpath(os.path.join(_LLK, "..", ".."))
    trees = (
        "tt_metal/tt-llk/tt_llk_*/**/*.h",
        "tt_metal/tt-llk/tt_llk_*/**/*.hpp",
        "tt_metal/hw/ckernels/*/metal/**/*.h",
    )
    return {
        os.path.relpath(p)
        for t in trees
        for p in glob.glob(os.path.join(repo, t), recursive=True)
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    files = [f for f in args.files if f.endswith((".h", ".hpp"))]
    if rescans_all(args.files):
        files = sorted(set(files) | set(all_headers()))
    n = 0
    for path in files:
        for line, msg in scan(path):
            n += 1
            print(f"{path}:{line}: {msg}")
    if n:
        print(f"{n} instruction argument(s) from the wrong constant family.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
