#!/usr/bin/env python3
"""
Static metrics of the source code written by the agent (benchmark specification, 4.3):
  - lines of code (excluding blank lines and comments)
  - "magic constants": hex literals, decimal literals >= 16, numeric shifts (<< N, >> N)
  - raw instructions (TTI_*, TT_OP_*), HAL usage (hal::), NOT_IN_HAL marks

The rules are fixed and written down here (before the runs) so that the numbers are comparable between arms.
Comments and strings are not counted.

Usage:
  codestats.py <dir or file> [...] [--json out.json] [--md out.md]
"""
import json
import re
import sys
from pathlib import Path

HEX = re.compile(r"\b0[xX][0-9a-fA-F']+[uUlL]*\b")
DEC = re.compile(r"(?<![\w.])(\d[\d']*)[uUlL]*\b(?!\.)")
SHIFT = re.compile(r"(<<|>>)\s*\d+")
RAW = re.compile(r"\b(TTI_\w+|TT_OP_\w+)")
HAL = re.compile(r"\bhal::")
NOT_IN_HAL = re.compile(r"NOT_IN_HAL")


def strip_comments(src: str) -> str:
    out, i, n = [], 0, len(src)
    while i < n:
        if src.startswith("//", i):
            j = src.find("\n", i); j = n if j == -1 else j
            out.append(" " * (j - i)); i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2); j = n if j == -1 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j])); i = j
        elif src[i] in "\"'" and not (src[i] == "'" and i > 0 and src[i - 1].isalnum()):
            q, j = src[i], i + 1
            while j < n and src[j] != q and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(" " * (j - i)); i = j
        else:
            out.append(src[i]); i += 1
    return "".join(out)


def stats(path: Path):
    src = path.read_text(errors="replace")
    code = strip_comments(src)
    lines = [ln for ln in code.split("\n") if ln.strip() and not ln.lstrip().startswith("#")]
    dec = [m for m in DEC.findall(code) if int(m.replace("'", "")) >= 16]
    return {
        "loc": len(lines),
        "hex_literals": len(HEX.findall(code)),
        "decimal_literals_ge16": len(dec),
        "numeric_shifts": len(SHIFT.findall(code)),
        "raw_instructions": len(RAW.findall(code)),
        "hal_uses": len(HAL.findall(code)),
        "not_in_hal_marks": len(NOT_IN_HAL.findall(src)),
    }


def main():
    args = sys.argv[1:]
    out_json = out_md = None
    if "--json" in args:
        i = args.index("--json"); out_json = args[i + 1]; del args[i:i + 2]
    if "--md" in args:
        i = args.index("--md"); out_md = args[i + 1]; del args[i:i + 2]
    if not args:
        print(__doc__); sys.exit(1)
    files = []
    for a in args:
        p = Path(a)
        files += sorted(p.rglob("*.h")) + sorted(p.rglob("*.cpp")) if p.is_dir() else [p] if p.exists() else []
    per_file = {str(f): stats(f) for f in files}
    keys = ["loc", "hex_literals", "decimal_literals_ge16", "numeric_shifts", "raw_instructions", "hal_uses", "not_in_hal_marks"]
    total = {k: sum(v[k] for v in per_file.values()) for k in keys}
    md = ["| File | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for f, v in per_file.items():
        md.append(f"| {Path(f).name} | " + " | ".join(str(v[k]) for k in keys) + " |")
    md.append("| **total** | " + " | ".join(f"**{total[k]}**" for k in keys) + " |")
    md += ["", "_Comments and strings are not counted. Magic constants: hex literals, decimal literals >= 16, "
               "numeric shifts (`<< N`, `>> N`)._"]
    text = "\n".join(md)
    print(text)
    if out_json:
        Path(out_json).write_text(json.dumps({"total": total, "per_file": per_file}, indent=2))
    if out_md:
        Path(out_md).write_text(text + "\n")


if __name__ == "__main__":
    main()
