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
    """(function parameter names, template parameter names) from a signature.

    The template clause leads the signature, so it is matched from the front and
    its angle brackets are balanced; anchoring at the end misses every function
    that has a body after it.
    """
    tmpl = set()
    head = signature
    m = re.search(r"\btemplate\s*<", signature)
    if m:
        depth, close = 0, None
        for i in range(m.end() - 1, len(signature)):
            if signature[i] == "<":
                depth += 1
            elif signature[i] == ">":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close is not None:
            for piece in split_args(signature[m.end() : close]):
                names = IDENT.findall(piece)
                if names:
                    tmpl.add(names[-1])
            head = signature[close + 1 :]
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
        # Two or more names means a type and a name; a name already declared as a
        # template parameter is the type's argument, not the parameter's name.
        if len(found) >= 2 and found[-1] not in tmpl:
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


ASSIGN = re.compile(r"\b(?P<name>[A-Za-z_]\w*)(?:\.\w+)*\s*(?:=[^=]|\+=|-=|\|=|&=)")
COND = re.compile(r"\b(?:if|while|for|switch)\s*\(")


def _block_after(body, at):
    """The brace-delimited block that follows position `at`, if there is one."""
    brace = body.find("{", at)
    if brace == -1 or body[at:brace].strip() not in ("", ")"):
        nl = body.find("\n", at)
        return body[at : nl if nl != -1 else len(body)]
    depth = 0
    for i in range(brace, len(body)):
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
            if depth == 0:
                return body[brace : i + 1]
    return body[brace:]


def tainted_names(body, params):
    """Names whose value follows a function parameter, by data or by control flow.

    A local assigned a literal is still tainted when the branch that assigns it is
    chosen by a parameter: that is how face_r_dim reaches row_set_mapping_1.
    """
    tainted = set(params)
    for _ in range(6):  # a fixed point; the bodies are shallow
        before = len(tainted)
        for m in ASSIGN.finditer(body):
            rhs = body[m.end() - 1 : body.find(";", m.end())]
            if {n for n in IDENT.findall(rhs)} & tainted:
                tainted.add(m.group("name"))
        for m in COND.finditer(body):
            close = balanced(body, m.end() - 1)
            if close == -1:
                continue
            if not ({n for n in IDENT.findall(body[m.end() : close])} & tainted):
                continue
            block = _block_after(body, close + 1)
            tainted |= {a.group("name") for a in ASSIGN.finditer(block)}
        if len(tainted) == before:
            break
    return tainted


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
            tainted = tainted_names(body, params)
            for pos, arg in enumerate(args):
                scoped = set(re.findall(r"\w+\s*::\s*(\w+)|(\w+)\s*::", arg))
                scoped = {x for pair in scoped for x in pair if x}
                found = []
                for n in IDENT.findall(arg):
                    if n in KEYWORDS or n in scoped or n.isupper():
                        continue
                    if n in tmpl or n not in tainted:
                        continue
                    found.append(
                        f"{n} ({'parameter' if n in params else 'follows a parameter'})"
                    )
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
