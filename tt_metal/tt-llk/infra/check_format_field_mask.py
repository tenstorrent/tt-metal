#!/usr/bin/env python3
"""Check that a DataFormat value is masked before it is shifted into a 4-bit config field.

The DataFormat enum carries software-only bits above the hardware code: UInt8 is 30 and UInt32 is
24, with bit 4 set, while a format field of the config space is 4 bits wide (DATA_FORMAT_BIT_COUNT).
A writer that shifts the raw value in lands bit 4 on the bit just above the field. For a write
whose mask covers only that field (`cfg_reg_rmw_tensix<FIELD_RMW>(v)`) the register-side mask drops
the spilled bit, so nothing is wrong. For a write whose mask also covers the bit above the field --
a multi-field RMW or a full-word write -- the spilled bit is programmed into the neighbouring field.
That is the case this check fails on, and the only one: a format value shifted unmasked into a
4-bit field, reaching a write whose mask includes the bit the spill lands on.

What makes a finding certain:
  * the shifted operand is a format value -- a non-constant identifier whose name contains
    `format`, such as `srca_data_format`, with no masking definition in the function
    (`masked_data_format(...)`, `& DATA_FORMAT_CONFIG_MASK`, `& 0xf`, or a `_masked` name);
  * the field is exactly 4 bits wide in that arch's cfg_defines.h (`FIELD_MASK >> FIELD_SHAMT == 0xF`);
  * the write's mask, resolved from `*_MASK` names and the function's `constexpr` masks, has the bit
    at `FIELD_SHAMT + 4` set, or the write is a full-word `cfg[ADDR] = ...`.
A write whose mask cannot be resolved is not a finding. Quasar's format fields are 5 bits wide and
its writers go through bitfields, so the check has nothing to say there and skips that tree.

The fix is to shift `masked_data_format(x)` instead of `x`, as the unpacker does for its THCON
format fields.

Scope matches the other LLK hooks: run with no files and the whole tree is checked; from the hook,
only the touched headers are, unless the checker itself is touched, in which case the whole tree is.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_mutex_balance import _signature, functions  # noqa: E402

INFRA = os.path.dirname(os.path.abspath(__file__))
LLK = os.path.normpath(os.path.join(INFRA, ".."))
REPO = os.path.normpath(os.path.join(LLK, "..", ".."))
DEFINES = {
    "wormhole_b0": "tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/cfg_defines.h",
    "blackhole": "tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h",
}
FIELD_BITS = 4  # DATA_FORMAT_BIT_COUNT on Wormhole and Blackhole
FULL_WORD = 0xFFFFFFFF

_DEFINE = re.compile(r"^\s*#define\s+(\w+)_(SHAMT|MASK)\s+(0x[0-9a-fA-F]+|\d+)\b", re.M)
# `X << F_SHAMT` with X a bare identifier (a constant, cast or call before `<<` is not a bare name).
_SHIFT = re.compile(r"(?<![\w.>\)])([A-Za-z_]\w*)\s*<<\s*(\w+)_SHAMT\b")
_CONSTEXPR = re.compile(r"\bconstexpr\s+(?:std::)?u?int\d*_t\s+(\w+)\s*=\s*([^;]+);")
_ASSIGN = re.compile(r"(?:\b(?:std::)?u?int\d*_t\s+)?\b(\w+)\s*(=|\|=)\s*([^;]+);")
_RMW_FULL = re.compile(
    r"cfg_reg_rmw_tensix\s*<\s*(\w+)\s*,\s*([^,>]+?)\s*,\s*([^>]+?)\s*>\s*\("
)
_CFG_STORE = re.compile(r"\bcfg\s*\[[^\]]+\]\s*=\s*(?!=)")
_FORMAT_NAME = re.compile(r"format", re.I)
_IDENT = re.compile(r"\b([A-Za-z_]\w*)\b")


def tree_files(arch):
    pats = [
        os.path.join(LLK, f"tt_llk_{arch}", "**", "*.h"),
        os.path.join(LLK, f"tt_llk_{arch}", "**", "*.hpp"),
        os.path.join(REPO, "tt_metal", "hw", "ckernels", arch, "metal", "**", "*.h"),
        os.path.join(REPO, "tt_metal", "hw", "ckernels", arch, "metal", "**", "*.hpp"),
    ]
    return sorted({p for pat in pats for p in glob.glob(pat, recursive=True)})


def arch_of(path):
    m = re.search(r"tt_llk_(\w+?)/|/ckernels/(\w+?)/", path.replace(os.sep, "/"))
    if not m:
        return None
    arch = m.group(1) or m.group(2)
    return arch if arch in DEFINES else None


def load_defines(path):
    """{field: (shamt, mask)} for every field with both a _SHAMT and a _MASK define."""
    shamt, mask = {}, {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for m in _DEFINE.finditer(f.read()):
            (shamt if m.group(2) == "SHAMT" else mask)[m.group(1)] = int(m.group(3), 0)
    return {k: (shamt[k], mask[k]) for k in shamt if k in mask}


def _paren_arg(text, open_idx):
    """Text of the parenthesised argument that opens at open_idx (exclusive of the parens)."""
    depth, i = 0, open_idx
    while i < len(text):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
        i += 1
    return text[open_idx + 1 :]


def _mask_value(expr, fields, local, depth=0):
    """Numeric value of a mask expression built from literals, *_MASK names, local constexprs
    and `|`; None when any part is unknown."""
    if depth > 8:
        return None
    value = 0
    for part in expr.split("|"):
        part = part.strip().strip("()").strip()
        if not part:
            return None
        if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", part):
            value |= int(part, 0)
        elif part.endswith("_MASK") and part[:-5] in fields:
            value |= fields[part[:-5]][1]
        elif part in local:
            sub = _mask_value(local[part], fields, local, depth + 1)
            if sub is None:
                return None
            value |= sub
        else:
            return None
    return value


def _masked_names(body):
    """Identifiers the function defines from a masked expression; shifting one of them is fine."""
    names = set()
    for m in _ASSIGN.finditer(body):
        rhs = m.group(3)
        if (
            "masked_data_format(" in rhs
            or "DATA_FORMAT_CONFIG_MASK" in rhs
            or re.search(r"&\s*0x0*[fF]\b", rhs)
        ):
            names.add(m.group(1))
    names.update(re.findall(r"\b(\w+_masked)\b", body))
    return names


def _shift_terms(expr, fields, masked):
    """[(name, field)] for each unmasked format identifier shifted into a 4-bit field."""
    out = []
    for m in _SHIFT.finditer(expr):
        name, field = m.group(1), m.group(2)
        if field not in fields:
            continue
        shamt, mask = fields[field]
        if (mask >> shamt) != (1 << FIELD_BITS) - 1:
            continue
        if not _FORMAT_NAME.search(name) or name in masked:
            continue
        if re.fullmatch(r"[A-Z0-9_]+", name):  # a constant, not a runtime value
            continue
        out.append((name, field))
    return out


def _expand(expr, assigns, depth=0):
    """Inline the function's assignments into expr so a shift behind a temporary is seen."""
    if depth > 4:
        return expr
    for ident in set(_IDENT.findall(expr)):
        if ident in assigns:
            expr = re.sub(
                rf"\b{re.escape(ident)}\b",
                "(" + " | ".join(assigns[ident]) + ")",
                expr,
            )
            expr = _expand(
                expr, {k: v for k, v in assigns.items() if k != ident}, depth + 1
            )
    return expr


def scan_function(body, fields):
    """Findings in one function body: (offset, name, field, write_kind, mask)."""
    local = {m.group(1): m.group(2) for m in _CONSTEXPR.finditer(body)}
    assigns = {}
    for m in _ASSIGN.finditer(body):
        if m.group(1) in local:
            continue
        assigns.setdefault(m.group(1), []).append(m.group(3).strip())
    masked = _masked_names(body)
    findings = []

    def check(expr, write_mask, kind, offset):
        for name, field in _shift_terms(_expand(expr, assigns), fields, masked):
            spill = fields[field][0] + FIELD_BITS
            if write_mask is not None and (write_mask >> spill) & 1:
                hit = (offset, name, field, kind, write_mask)
                # A temporary assigned in two branches expands to both; one finding per write.
                if hit not in findings:
                    findings.append(hit)

    for m in _RMW_FULL.finditer(body):
        arg = _paren_arg(body, m.end() - 1)
        # The write's mask applies after the template shift; a non-zero shift moves every
        # field, so only the plain `0` shift form is resolved.
        if m.group(2).strip() != "0":
            continue
        check(arg, _mask_value(m.group(3), fields, local), "multi-field RMW", m.start())
    # A single-field `cfg_reg_rmw_tensix<FIELD_RMW>(v)` masks at the register: the spilled bit
    # lands outside its mask, so that form is never a finding.
    for m in _CFG_STORE.finditer(body):
        rest = body[m.end() :]
        expr = rest[: rest.find(";")] if ";" in rest else rest
        check(expr, FULL_WORD, "full-word write", m.start())
    return findings


def scan(path, fields):
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    lines = src.split("\n")
    out = []
    for fn_line, _end, body, _guard in functions(src):
        for offset, name, field, kind, mask in scan_function(body, fields):
            line = fn_line + body[:offset].count("\n")
            sig = lines[_signature(lines, fn_line)].strip()
            out.append((path, line + 1, sig, name, field, kind, mask))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*")
    ap.add_argument(
        "--repo", default=REPO, help="repository root holding the cfg_defines headers"
    )
    args = ap.parse_args()

    files = [os.path.abspath(f) for f in args.files]
    whole_tree = not files or any(
        os.path.basename(f) == os.path.basename(__file__) for f in files
    )
    if whole_tree:
        files = [p for arch in DEFINES for p in tree_files(arch)]
    files = [f for f in files if f.endswith((".h", ".hpp"))]

    fields_by_arch = {}
    findings = []
    for path in files:
        arch = arch_of(path)
        if arch is None:
            continue
        if arch not in fields_by_arch:
            defs = os.path.join(args.repo, DEFINES[arch])
            if not os.path.exists(defs):
                print(f"error: {defs} not found", file=sys.stderr)
                return 1
            fields_by_arch[arch] = load_defines(defs)
        findings.extend(scan(path, fields_by_arch[arch]))

    for path, line, sig, name, field, kind, mask in findings:
        rel = os.path.relpath(path, args.repo)
        print(
            f"{rel}:{line}: `{name}` is shifted unmasked into the 4-bit field {field} and the "
            f"{kind} (mask 0x{mask:08x}) covers the bit its spill lands on; shift "
            f"masked_data_format({name}) instead.\n    in {sig}"
        )
    if findings:
        print(
            f"\n{len(findings)} finding(s): a DataFormat value such as UInt8 (30) or UInt32 (24) "
            "sets bit 4, which these writes program into the neighbouring field."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
