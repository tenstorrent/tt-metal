#!/usr/bin/env python3
"""
Creates a "spec" version of a C++ header for mode B: keeps comments, #include/#pragma,
namespaces, enum/struct/using declarations and function SIGNATURES with doc comments,
and replaces function bodies with ';'. Also removes macros / constants that reveal
raw instruction encodings (TT_OP_*, TTI_*, ...).

Usage:
  make_spec.py <src_dir> <dst_dir> [--exclude name.h ...] [--trivial]
Processes all *.h files directly in src_dir (not recursively).

--trivial: "trivial agent" for validating the test. Everything stays as in the original
(including raw encodings), only function bodies become empty ('{}' or '{ return {}; }').
"""
import re
import sys
from pathlib import Path

RAW_MARKERS = re.compile(r"\b(TT_OP_\w+|TTI_\w+|TT_\w+\s*\(|INSTRUCTION_WORD|TRISC_OP_SWIZZLE|ckernel_template)\b")
TYPE_KEYWORDS = re.compile(r"^\s*(struct|class|enum|union|namespace)\b")
FUNC_TAIL = re.compile(r"\)\s*(const|noexcept(\s*\(.*?\))?|override|final|mutable|->\s*[\w:<>,\s\*&]+|\s)*$", re.S)


def strip_comments_and_strings(s: str) -> str:
    """Returns the same text with comments and strings replaced by spaces (same length)."""
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if s.startswith("//", i):
            j = s.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i)); i = j
        elif s.startswith("/*", i):
            j = s.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(re.sub(r"[^\n]", " ", s[i:j])); i = j
        elif c in "\"'":
            j = i + 1
            while j < n and s[j] != c:
                j += 2 if s[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(" " * (j - i)); i = j
        else:
            out.append(c); i += 1
    return "".join(out)


def mask_preprocessor(clean: str) -> str:
    """Preprocessor lines (with continuations) become spaces so they do not disturb the parser."""
    lines = clean.split("\n")
    res, cont = [], False
    for ln in lines:
        if cont or ln.lstrip().startswith("#"):
            cont = ln.rstrip().endswith("\\")
            res.append(" " * len(ln))
        else:
            res.append(ln)
    return "\n".join(res)


def strip_template_prefix(s: str) -> str:
    """Removes a leading 'template <...>' (with nested <>) so '=' in default parameters does not confuse the parser."""
    s = s.lstrip()
    while s.startswith("template"):
        k = s.find("<")
        if k == -1:
            break
        depth = 0
        for m in range(k, len(s)):
            if s[m] == "<":
                depth += 1
            elif s[m] == ">":
                depth -= 1
                if depth == 0:
                    s = s[m + 1:].lstrip()
                    break
        else:
            break
    return s


def find_matching(clean: str, i: int) -> int:
    depth = 0
    for j in range(i, len(clean)):
        if clean[j] == "{":
            depth += 1
        elif clean[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return len(clean) - 1


NEVER = re.compile(r"(?!x)x")


def trivial_body(seg: str) -> str:
    """Empty function body; for a non-void return type returns a value-initialized result."""
    head = seg.split("(")[0].rstrip()
    prefix = re.sub(r"[\w:~]+$", "", head)                      # everything before the function name
    prefix = re.sub(r"\[\[.*?\]\]|__attribute__\s*\(\(.*?\)\)", " ", prefix)
    prefix = re.sub(r"\b(inline|static|constexpr|consteval|extern|ALWI|FORCE_INLINE|__inline__)\b", " ", prefix)
    ret = prefix.strip()
    if not ret or re.fullmatch(r"(void|auto)", ret):
        return "{}"
    return "{ return {}; }"


def process(src: str, trivial: bool = False) -> str:
    raw = NEVER if trivial else RAW_MARKERS
    clean = mask_preprocessor(strip_comments_and_strings(src))
    out = []          # output pieces
    out_len = 0       # total output length
    def emit(t):
        nonlocal out_len
        out.append(t); out_len += len(t)
    def truncate(pos):
        nonlocal out, out_len
        text = "".join(out)[:pos]
        out, out_len = [text], len(text)

    i, n = 0, len(src)
    seg_start = 0       # start of the current top-level statement in the source
    seg_out = 0         # position in the output that corresponds to seg_start
    paren = 0
    def new_segment(pos):
        nonlocal seg_start, seg_out
        seg_start, seg_out = pos, out_len

    while i < n:
        c = clean[i]
        if c == "(":
            paren += 1
        elif c == ")":
            paren -= 1
        elif c == ";" and paren == 0:
            seg = clean[seg_start:i]
            if raw.search(seg) and "=" in seg:
                # top-level constant with a raw encoding: drop it
                lead = re.match(r"\s*", src[seg_start:i]).group(0)
                truncate(seg_out)
                emit(lead + "/* [spec] removed constant with raw encoding */")
            emit(";"); i += 1; new_segment(i)
            continue
        elif c == "{" and paren == 0:
            seg_stripped = strip_template_prefix(clean[seg_start:i].strip())
            if re.search(r"\bnamespace\b[^;]*$", seg_stripped) or re.search(r"extern\s*$", seg_stripped):
                emit("{"); i += 1; new_segment(i)       # namespace: transparent
                continue
            j = find_matching(clean, i)
            body_clean = clean[i:j + 1]
            if TYPE_KEYWORDS.search(seg_stripped) or "=" in seg_stripped.split("(")[0]:
                # struct/class/enum/union or initializer: keep the block
                if raw.search(body_clean) and not re.search(r"\benum\b", seg_stripped):
                    emit("{ /* [spec] body removed */ }")
                else:
                    emit(src[i:j + 1])
                i = j + 1
                continue
            if "(" in seg_stripped and FUNC_TAIL.search(seg_stripped):
                # function body -> ';'
                text = "".join(out).rstrip()
                out, out_len = [text], len(text)
                emit((" " + trivial_body(seg_stripped)) if trivial else ";"); i = j + 1; new_segment(i)
                continue
            emit("{ /* [spec] removed */ }" if raw.search(body_clean) else src[i:j + 1])
            i = j + 1
            continue
        elif c == "}" and paren == 0:
            emit("}"); i += 1; new_segment(i)
            continue
        emit(src[i]); i += 1

    text = "".join(out)
    # macros that define raw instructions
    lines = []
    skip_cont = False
    for ln in text.split("\n"):
        if skip_cont:
            skip_cont = ln.rstrip().endswith("\\")
            continue
        if ln.lstrip().startswith("#define") and raw.search(ln):
            skip_cont = ln.rstrip().endswith("\\")
            lines.append("// [spec] removed macro with raw instruction")
            continue
        lines.append(ln)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    header = ("// [trivial] Generated for test validity check: function bodies are empty.\n" if trivial else
              "// [spec] Generated for the HAL experiment (mode B): function bodies and raw encodings have been removed.\n")
    return header + text


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    src_dir, dst_dir = Path(sys.argv[1]), Path(sys.argv[2])
    trivial = "--trivial" in sys.argv
    args = [a for a in sys.argv if a != "--trivial"]
    exclude = set(args[args.index("--exclude") + 1:]) if "--exclude" in args else set()
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_dir.glob("*.h")):
        if f.name in exclude:
            continue
        (dst_dir / f.name).write_text(process(f.read_text(errors="replace"), trivial))
        count += 1
    print(f"[{'trivial' if trivial else 'spec'}] {count} headers: {src_dir} -> {dst_dir}")


if __name__ == "__main__":
    main()
