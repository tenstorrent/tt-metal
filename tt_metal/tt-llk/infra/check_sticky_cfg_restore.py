#!/usr/bin/env python3
"""Check that writes to long-lived config fields are put back.

The fields and their policy live in sticky_cfg_fields.yaml:

  restore -- hardware-configure sets the field once and later ops assume that value. A write of a
             fixed value must be undone: by a later write of a different value (or of the
             configured value) in the same function, by the matching `_uninit_`, or by a site the
             table names under `sites:`.
  tracked -- the math thread caches the field in a software tracker. A raw write outside the
             tracker's writer must be followed by the invalidate call, in the same function or in
             a function of the same file that calls it.

A write whose value is not a literal cannot be judged and is reported as a note, never a failure.
Pre-existing findings go in a baseline keyed by path, function and field.

Known limit: the restore check is lexical, so a branch that clears a field and a sibling branch that
sets it back read as restored.

Scope matches the other LLK hooks: run with no files and the whole tree is checked; from the hook,
only the touched headers are, unless a verdict input (this script, the table, the baseline) is
touched, in which case the whole tree is.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_mutex_balance import _blank_noncode, _signature, functions  # noqa: E402

INFRA = os.path.dirname(os.path.abspath(__file__))
LLK = os.path.normpath(os.path.join(INFRA, ".."))
REPO = os.path.normpath(os.path.join(LLK, "..", ".."))
TABLE = os.path.join(INFRA, "sticky_cfg_fields.yaml")
DEFINES = {
    "wormhole_b0": "tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/cfg_defines.h",
    "blackhole": "tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h",
}
FIELD_KEYS = {
    "arches",
    "policy",
    "configured_by",
    "configuring_functions",
    "tracker_writer",
    "invalidate",
}
POLICY_KEYS = {
    "restore": {"configured_by", "configuring_functions"},
    "tracked": {"tracker_writer", "invalidate"},
}
SITE_KEYS = {"path", "function", "field", "restored_by", "reason"}
_LITERAL = {"0": 0, "1": 1, "0x0": 0, "0x1": 1, "true": 1, "false": 0, "0u": 0, "1u": 1}
_NAME = re.compile(r"(~?\w+)\s*(?:<[^;{}()]*>)?\s*\([^;{}]*$")


class TableError(Exception):
    pass


def tree_files(arch):
    pats = [
        os.path.join(LLK, f"tt_llk_{arch}", "**", "*.h"),
        os.path.join(REPO, "tt_metal", "hw", "ckernels", arch, "metal", "**", "*.h"),
    ]
    return sorted({p for pat in pats for p in glob.glob(pat, recursive=True)})


def arch_of(path):
    p = os.path.abspath(path)
    for arch in DEFINES:
        if (
            f"{os.sep}tt_llk_{arch}{os.sep}" in p
            or f"{os.sep}ckernels{os.sep}{arch}{os.sep}" in p
        ):
            return arch
    return None


def load_table(path=TABLE, repo=REPO):
    import yaml

    try:
        doc = yaml.safe_load(open(path)) or {}
    except yaml.YAMLError as e:
        raise TableError(f"not valid YAML: {e}")
    if not isinstance(doc, dict):
        raise TableError("top level must be a mapping")
    extra = set(doc) - {"fields", "sites"}
    if extra:
        raise TableError(f"unknown top-level key(s): {sorted(extra)}")
    fields = doc.get("fields") or {}
    for name, spec in fields.items():
        if not isinstance(spec, dict):
            raise TableError(f"{name}: entry must be a mapping")
        unknown = set(spec) - FIELD_KEYS
        if unknown:
            raise TableError(f"{name}: unknown key(s) {sorted(unknown)}")
        policy = spec.get("policy")
        if policy not in POLICY_KEYS:
            raise TableError(f"{name}: policy must be one of {sorted(POLICY_KEYS)}")
        missing = POLICY_KEYS[policy] - set(spec)
        wrong = (set(spec) - {"arches", "policy"}) - POLICY_KEYS[policy]
        if missing or wrong:
            raise TableError(
                f"{name}: policy {policy} needs {sorted(POLICY_KEYS[policy])}"
            )
        arches = spec.get("arches") or []
        if not arches or any(a not in DEFINES for a in arches):
            raise TableError(
                f"{name}: arches must be a non-empty subset of {sorted(DEFINES)}"
            )
        for a in arches:
            defs = open(os.path.join(repo, DEFINES[a])).read()
            if not re.search(rf"#define\s+{re.escape(name)}_ADDR32\b", defs):
                raise TableError(f"{name}: not defined in {DEFINES[a]}")
    sites = doc.get("sites") or []
    for s in sites:
        if not isinstance(s, dict) or set(s) != SITE_KEYS:
            raise TableError(f"site entry must have exactly {sorted(SITE_KEYS)}: {s}")
        if s["field"] not in fields:
            raise TableError(f"site names unknown field {s['field']}")
        if not str(s["reason"]).strip():
            raise TableError(f"site {s['path']}:{s['function']} needs a reason")
    return fields, sites


def _paren_arg(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    return text[open_idx + 1 :]


def _value(expr):
    e = expr.strip()
    while e.startswith("(") and e.endswith(")"):
        e = e[1:-1].strip()
    return _LITERAL.get(e.lower(), e)


def writes(body, field):
    """(offset, value) for each write of field in body; value is 0/1 or the source text."""
    out = []
    for m in re.finditer(rf"\bcfg_reg_rmw_tensix\s*<\s*{field}_RMW\s*>\s*\(", body):
        out.append((m.start(), _value(_paren_arg(body, m.end() - 1))))
    for m in re.finditer(rf"\.\s*{field}\s*=(?!=)\s*([^;]+);", body):
        out.append((m.start(), _value(m.group(1))))
    return sorted(out, key=lambda w: w[0])


_ATTR = re.compile(r"__attribute__\s*\(\((?:[^()]|\([^()]*\))*\)\)|\[\[[^\]]*\]\]")


def _fn_name(code_lines, brace_line, body):
    first = body.split("\n")[0]
    col = code_lines[brace_line].find(first)
    head = (
        " ".join(code_lines[max(brace_line - 4, 0) : brace_line])
        + " "
        + code_lines[brace_line][:col]
    )
    head = re.split(r"[;{}]", _ATTR.sub(" ", head))[-1]
    m = _NAME.search(head)
    return m.group(1) if m else None


def bodies(path):
    """(name, start_line_1based, body) per function, from blanked source."""
    src = open(path, errors="ignore").read()
    code_lines = _blank_noncode(src).split("\n")
    lines = src.split("\n")
    for brace_line, _end, body, _guard in functions(src):
        yield _fn_name(code_lines, brace_line, body), _signature(
            lines, brace_line
        ), brace_line, body


def uninit_index(fields, extra=()):
    """(arch, function name) -> set of fields each _uninit_ writes, over every tree."""
    idx = {}
    paths = {p for a in DEFINES for p in tree_files(a)} | {
        os.path.abspath(e) for e in extra
    }
    for path in sorted(paths):
        arch = arch_of(path)
        if arch:
            src = open(path, errors="ignore").read()
            if not any(f in src for f in fields):
                continue
            for name, _sig, _bl, body in bodies(path):
                if name and "uninit" in name:
                    for f in fields:
                        if writes(body, f):
                            idx.setdefault((arch, name), set()).add(f)
    return idx


def _uninit_name(name):
    return re.sub(r"_init(?=_|$)", "_uninit", name, count=1) if name else None


def _caller_invalidates(fns, callee, invalidate):
    """True when a function in the same file calls callee and then the invalidate function."""
    call = re.compile(rf"\b{re.escape(callee)}\s*(?:<[^;{{}}()]*>)?\s*\(")
    inv = re.compile(rf"\b{re.escape(invalidate)}\s*\(")
    for name, _sig, _bl, body in fns:
        if name == callee:
            continue
        calls = [m.start() for m in call.finditer(body)]
        if calls and any(m.start() > calls[-1] for m in inv.finditer(body)):
            return True
    return False


def scan(path, fields, sites, uninits):
    """Yield (line, function, field, kind, message, blocking) for one header."""
    arch = arch_of(path)
    if arch is None:
        return
    src = open(path, errors="ignore").read()
    active = {f: s for f, s in fields.items() if arch in s["arches"] and f in src}
    if not active:
        return
    rel = os.path.relpath(os.path.abspath(path), REPO)
    fns = list(bodies(path))
    for name, _sig, brace_line, body in fns:
        for field, spec in active.items():
            ws = writes(body, field)
            if not ws:
                continue

            def line_of(off):
                return brace_line + body.count("\n", 0, off) + 1

            covered = any(
                s["path"] == rel and s["function"] == name and s["field"] == field
                for s in sites
            )
            if spec["policy"] == "tracked":
                if name == spec["tracker_writer"] or covered:
                    continue
                if name and _caller_invalidates(fns, name, spec["invalidate"]):
                    continue
                last = ws[-1][0]
                inv = [
                    m.start()
                    for m in re.finditer(rf"\b{spec['invalidate']}\s*\(", body)
                ]
                if not any(i > last for i in inv):
                    yield (
                        line_of(last),
                        name,
                        field,
                        "tracked",
                        f"raw write in {name}() bypasses the math-thread tracker; call "
                        f"{spec['invalidate']}() after it (math thread) or go through the setter",
                        True,
                    )
                continue

            if name and ("uninit" in name or name in spec["configuring_functions"]):
                continue  # where the value is set or put back
            pending = None
            for off, val in ws:
                if isinstance(val, int):
                    if pending is None:
                        pending = (off, val)
                    elif val != pending[1]:
                        pending = None
                elif pending is not None:
                    pending = None  # a computed value: assume it re-establishes the configuration
                elif val not in spec["configured_by"]:
                    yield (
                        line_of(off),
                        name,
                        field,
                        "unresolved",
                        f"value `{val}` cannot be judged in {name}()",
                        False,
                    )
            if pending is None or covered:
                continue
            un = _uninit_name(name)
            if un and un != name and field in uninits.get((arch, un), ()):
                continue
            yield (
                line_of(pending[0]),
                name,
                field,
                "restore",
                f"{name}() leaves it at {pending[1]}; restore it in this function, in "
                f"{un or 'its _uninit_'}(), or name the restoring site in sticky_cfg_fields.yaml",
                True,
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*")
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--baseline")
    args = ap.parse_args()

    try:
        fields, sites = load_table(args.table)
    except TableError as e:
        print(f"{args.table}: {e}")
        return 1

    headers = [f for f in args.files if f.endswith((".h", ".hpp"))]
    whole_tree = not args.files or len(headers) != len(args.files)
    targets = (
        sorted({p for a in DEFINES for p in tree_files(a)}) if whole_tree else headers
    )
    uninits = uninit_index(fields, targets)

    accepted = set()
    if args.baseline and os.path.exists(args.baseline):
        accepted = {
            l.split("#")[0].strip()
            for l in open(args.baseline)
            if l.strip() and not l.startswith("#")
        }

    seen, blocking, notes = set(), [], []
    for path in targets:
        for line, fn, field, kind, msg, block in scan(path, fields, sites, uninits):
            key = f"{os.path.relpath(os.path.abspath(path), REPO)}:{fn}:{field}"
            seen.add(key)
            if key in accepted:
                continue
            (blocking if block else notes).append((path, line, field, kind, msg))

    for path, line, field, kind, msg in blocking:
        print(f"{path}:{line}: [{kind}] {field}")
        print(f"  {msg}\n")
    if notes:
        print(
            f"note: {len(notes)} write(s) whose value could not be judged (not a failure):"
        )
        for path, line, field, _kind, msg in notes:
            print(f"  {path}:{line}: {field}: {msg}")
    if whole_tree and accepted:
        stale = sorted(accepted - seen)
        if stale:
            print(
                f"note: {len(stale)} baseline entr(y/ies) no longer match; remove them:"
            )
            for k in stale:
                print(f"  {k}")
    if blocking:
        print(f"{len(blocking)} config write(s) not restored or not tracked.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
