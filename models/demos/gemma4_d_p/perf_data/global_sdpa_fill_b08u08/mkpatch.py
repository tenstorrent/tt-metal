"""Edit helpers for the zones worktree; each experiment = edit, `git diff > patch`, `git checkout .`"""

import re, sys


def read(p):
    return open(p).read()


def write(p, s):
    open(p, "w").write(s)


def stmt_end(s, i):
    """Index just past the ';' that ends the statement starting at i (paren/angle-aware via parens only)."""
    depth = 0
    while True:
        c = s[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ";" and depth == 0:
            return i + 1
        i += 1


def wrap_stmt(p, anchor, zone, nth=0):
    """Wrap the statement beginning at the nth occurrence of anchor in { DeviceZoneScopedN(zone); ... }."""
    s = read(p)
    idx = -1
    for _ in range(nth + 1):
        idx = s.index(anchor, idx + 1)
    line_start = s.rfind("\n", 0, idx) + 1
    indent = s[line_start:idx]
    end = stmt_end(s, idx)
    s = s[:idx] + '{ DeviceZoneScopedN("' + zone + '");\n' + indent + s[idx:end] + " }" + s[end:]
    write(p, s)


def insert_after(p, anchor, text, nth=0):
    s = read(p)
    idx = -1
    for _ in range(nth + 1):
        idx = s.index(anchor, idx + 1)
    idx += len(anchor)
    write(p, s[:idx] + text + s[idx:])


def replace_once(p, old, new):
    s = read(p)
    assert s.count(old) >= 1, old
    write(p, s.replace(old, new, 1))
