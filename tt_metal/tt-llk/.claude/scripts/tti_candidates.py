"""List TTI_ call sites whose operands come from a function parameter.

A TTI_ macro bakes its operands into the instruction word, so every operand must
fold at compile time. When an operand is a function parameter the site compiles
only while the compiler happens to see a constant there. Those are the sites
#62420 makes safe to demote to TT_, and the ones that fail sporadically today.

Template parameters are left out: they are compile-time by definition.

    tti_candidates.py <repo root> [--arch blackhole|wormhole_b0|quasar]
"""

import re
import sys
from pathlib import Path

ARCHES = ("blackhole", "wormhole_b0", "quasar")
IDENT = re.compile(r"[A-Za-z_]\w*")
KEYWORDS = {
    "const",
    "constexpr",
    "static",
    "inline",
    "return",
    "if",
    "else",
    "for",
    "while",
    "sizeof",
    "int",
    "uint",
    "bool",
    "true",
    "false",
    "void",
    "auto",
    "unsigned",
    "signed",
    "char",
    "short",
    "long",
    "float",
    "double",
    "std",
    "uint32_t",
    "uint16_t",
    "uint8_t",
    "int32_t",
    "ckernel",
}


def strip_noise(text):
    """Blank out comments and string bodies, keeping every newline so line numbers hold."""

    def blank(m):
        return "\n" * m.group(0).count("\n")

    text = re.sub(r"/\*.*?\*/", blank, text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r'"(\\.|[^"\\])*"', '""', text)


def split_args(s):
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return [a.strip() for a in out]


def balanced(text, open_at):
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def params_of(signature):
    """Parameter names declared in a function signature, template params excluded."""
    tmpl = set()
    m = re.search(r"template\s*<(.*?)>\s*$", signature, flags=re.S)
    head = signature
    if m:
        for piece in split_args(m.group(1)):
            names = IDENT.findall(piece)
            if names:
                tmpl.add(names[-1])
        head = signature[: m.start()]
    open_at = head.rfind("(")
    if open_at == -1:
        return set(), tmpl
    close = balanced(head, open_at)
    if close == -1:
        return set(), tmpl
    names = set()
    for piece in split_args(head[open_at + 1 : close]):
        piece = piece.split("=")[0].strip()
        found = [n for n in IDENT.findall(piece) if n not in KEYWORDS]
        if len(found) >= 2:  # a type and a name
            names.add(found[-1])
    return names, tmpl


def functions(text):
    """(body start, body end, signature) per function body.

    The LLK headers wrap everything in a namespace, so brace depth never returns
    to zero between functions. Instead: a body opens at a '{' whose preceding
    code is a ')' with optional trailing qualifiers.
    """
    out = []
    for m in re.finditer(r"\)\s*(?:const\s*)?(?:noexcept\s*)?\{", text):
        brace = text.index("{", m.start())
        paren_close = text.rindex(")", m.start(), brace + 1)
        depth, open_at = 0, None
        for i in range(paren_close, -1, -1):
            if text[i] == ")":
                depth += 1
            elif text[i] == "(":
                depth -= 1
                if depth == 0:
                    open_at = i
                    break
        if open_at is None:
            continue
        cut = max(text.rfind(c, 0, open_at) for c in ";{}")
        signature = text[cut + 1 : paren_close + 1]
        depth, end = 0, None
        for i in range(brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is not None:
            out.append((brace, end, signature))
    return out


DECL = r"(?:^|[;{}\n])\s*(?P<quals>(?:(?:static|inline|volatile|constexpr|const)\s+)*)[\w:<>,\s\*&]*?\b%s\s*(?:=|\{|\[)"


def classify(name, body, params, tmpl):
    """How constant is this name at the call site?"""
    if name in tmpl:
        return None  # compile-time by definition
    if name in params:
        return "parameter"
    import re as _re

    m = _re.search(DECL % _re.escape(name), body, flags=_re.M)
    if m is None:
        return None  # declared elsewhere; assume a constant
    return "local" if "constexpr" not in m.group("quals") else None


def scan(root, arch):
    hits = []
    base = Path(root, "tt_metal/tt-llk", f"tt_llk_{arch}")
    for path in sorted(base.rglob("*.h")):
        if path.name == "ckernel_ops.h":  # the macro definitions themselves
            continue
        text = strip_noise(path.read_text(errors="ignore"))
        funcs = functions(text)
        for m in re.finditer(r"\bTTI_([A-Z0-9_]+)\s*\(", text):
            close = balanced(text, m.end() - 1)
            if close == -1:
                continue
            args = split_args(text[m.end() : close])
            fn = next((f for f in funcs if f[0] < m.start() < f[1]), None)
            if fn is None:
                continue
            params, tmpl = params_of(fn[2])
            body = text[fn[0] : fn[1]]
            for pos, arg in enumerate(args):
                scoped = set(re.findall(r"\w+\s*::\s*(\w+)|(\w+)\s*::", arg))
                scoped = {x for pair in scoped for x in pair if x}
                found = []
                for n in IDENT.findall(arg):
                    if n in KEYWORDS or n in scoped or n.isupper():
                        continue
                    kind = classify(n, body, params, tmpl)
                    if kind:
                        found.append(f"{n} ({kind})")
                if found:
                    hits.append(
                        (
                            str(path.relative_to(Path(root, "tt_metal/tt-llk"))),
                            text[: m.start()].count("\n") + 1,
                            f"TTI_{m.group(1)}",
                            pos,
                            arg.strip(),
                            found,
                        )
                    )
    return hits


def main():
    root = sys.argv[1]
    want = sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else None
    for arch in ([want] if want else ARCHES):
        hits = scan(root, arch)
        by_macro, by_file = {}, {}
        for path, line, macro, pos, arg, names in hits:
            by_macro.setdefault(macro, []).append((path, line, pos, arg, names))
            by_file[path] = by_file.get(path, 0) + 1
        print(f"\n######## {arch}: {len(hits)} operand(s) in {len(by_file)} file(s)\n")
        for macro, rows in sorted(by_macro.items(), key=lambda kv: -len(kv[1])):
            print(f"  {macro}  ({len(rows)})")
            for path, line, pos, arg, names in rows[:4]:
                print(f"      {path}:{line}  arg{pos} = {arg!r}  <- {', '.join(names)}")
            if len(rows) > 4:
                print(f"      ... {len(rows) - 4} more")
        print("\n  busiest files:")
        for path, n in sorted(by_file.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {n:4d}  {path}")


main()
