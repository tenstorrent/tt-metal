# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint of the code a TILE_LOOP zone executes, independent of the address it is linked at.

The window is every instruction of run_kernel reachable from the start of the TILE_LOOP zone up to its end, plus the
functions those instructions call. The zone is found by its id: either the zone_reserve/zone_record calls or the
inline lui of its START/END entry word. Branch targets inside the window are written as
positions in the window and jumps out of it as "out", so moving code leaves the fingerprint unchanged and a change
to an executed instruction changes it.
"""

import hashlib
import os
import re
import subprocess
from pathlib import Path

_HEADER = re.compile(r"^([0-9a-f]+) <(.+)>:$")
_INSN = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
_TARGET = re.compile(r"\b([0-9a-f]+) <([^>+]+)(?:\+0x([0-9a-f]+))?>")
_CALLS = ("jal", "call", "tail", "j")
_BRANCH = re.compile(r"^(b[a-z]*|j|c\.j|c\.b[a-z]*)$")
_ENDS = ("ret", "jr", "mret", "c.jr")
_TILE_LOOP = re.compile(rb"LLK_PROFILER:[^\x00]*?:TILE_LOOP\x00")
_CACHE = {}


def _functions(text):
    funcs, current = {}, None
    for line in text.splitlines():
        m = _HEADER.match(line)
        if m:
            current = m.group(2)
            funcs[current] = []
            continue
        m = _INSN.match(line)
        if m and current is not None:
            funcs[current].append(
                (int(m.group(1), 16), m.group(2), m.group(3).split("#")[0].strip())
            )
    return funcs


def _hash16(text):
    h = 2166136261
    for c in text:
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return (h ^ (h >> 16)) & 0xFFFF


def _is_call(mnemonic, operands, symbol):
    return (mnemonic in _CALLS or mnemonic.startswith("jal")) and symbol in operands


def _a0_before(insns, i):
    hi = lo = None
    for _, mnemonic, operands in reversed(insns[max(0, i - 6) : i]):
        args = [a.strip() for a in operands.split(",")]
        if not args or args[0] != "a0":
            continue
        if mnemonic == "addi" and len(args) == 3 and args[1] == "a0" and lo is None:
            lo = int(args[2], 0)
        elif mnemonic == "lui" and hi is None:
            hi = int(args[1], 0) << 12
            break
        elif mnemonic == "li":
            return int(args[1], 0) & 0xFFFF
    return None if hi is None else (hi + (lo or 0)) & 0xFFFF


def _zone_lui(mnemonic, operands):
    """(kind, id) for the inline zone entry word: lui of (0xA or 0xB) << 16 | id, else None."""
    if mnemonic not in ("lui", "c.lui"):
        return None
    try:
        imm = int(operands.split(",")[-1].strip(), 0)
    except ValueError:
        return None
    return {0xA: "start", 0xB: "end"}.get(imm >> 16), imm & 0xFFFF


def _starts(insns, k):
    _, mnemonic, operands = insns[k]
    lui = _zone_lui(mnemonic, operands)
    return _is_call(mnemonic, operands, "zone_reserve") or (lui and lui[0] == "start")


def _ends(insns, k, ids):
    _, mnemonic, operands = insns[k]
    if _is_call(mnemonic, operands, "zone_record"):
        return _a0_before(insns, k) in ids
    lui = _zone_lui(mnemonic, operands)
    return lui is not None and lui[0] == "end" and lui[1] in ids


def _region(insns, start, ids):
    """Instructions on a path from the zone start to its end, in walk order.

    Paths that leave the zone without its end (for example to the function's epilogue) are not part of it.
    """
    index = {a: k for k, (a, _, _) in enumerate(insns)}
    order, seen, todo, succ, ends = [], set(), [start], {}, set()
    while todo:
        k = todo.pop()
        while k < len(insns) and k not in seen:
            if k != start and _starts(insns, k):
                break
            seen.add(k)
            order.append(k)
            succ[k] = []
            if _ends(insns, k, ids):
                ends.add(k)
                break
            _, mnemonic, operands = insns[k]
            target = _TARGET.search(operands)
            t = index.get(int(target.group(1), 16)) if target else None
            if mnemonic in _ENDS or (mnemonic in ("j", "c.j", "tail") and t is None):
                break
            if _BRANCH.match(mnemonic) and t is not None:
                succ[k].append(t)
                if mnemonic in ("j", "c.j"):
                    k = t
                    continue
                todo.append(t)
            succ[k].append(k + 1)
            k += 1
    pred = {}
    for k, targets in succ.items():
        for t in targets:
            pred.setdefault(t, []).append(k)
    keep, todo = set(ends), list(ends)
    while todo:
        for p in pred.get(todo.pop(), []):
            if p not in keep:
                keep.add(p)
                todo.append(p)
    return [k for k in order if k in keep]


def _window(insns, elf_bytes):
    ids = {_hash16(m.group(0)[:-1]) for m in _TILE_LOOP.finditer(elf_bytes)}
    found = []
    for k in range(len(insns)):
        if not _starts(insns, k):
            continue
        region = _region(insns, k, ids)
        if any(_ends(insns, x, ids) for x in region):
            found.append(region)
    return min(found, key=len) if found else None


def _callees(insns):
    out = []
    for _, mnemonic, operands in insns:
        if mnemonic in _CALLS or mnemonic.startswith("jal"):
            m = _TARGET.search(operands)
            if m:
                out.append(m.group(2))
    return out


def _normalise(mnemonic, operands, function, positions):
    def repl(m):
        addr, symbol = int(m.group(1), 16), m.group(2)
        if symbol == function:
            return f"#{positions[addr]}" if addr in positions else "#out"
        return f"{symbol}+{m.group(3) or '0'}"

    return f"{mnemonic} {_TARGET.sub(repl, operands)}"


def _compute(objdump, elf_path):
    result = subprocess.run(
        [objdump, "-d", "--no-show-raw-insn", str(elf_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "no-objdump"
    funcs = _functions(result.stdout)
    kernel = next((name for name in funcs if "run_kernel" in name), None)
    if kernel is None:
        return "no-kernel"
    insns = funcs[kernel]
    region = _window(insns, Path(elf_path).read_bytes())
    if region is None:
        return "no-window"
    window = [insns[k] for k in region]
    positions = {insns[k][0]: n for n, k in enumerate(region)}
    lines = [_normalise(m, o, kernel, positions) for _, m, o in window]
    seen = set()
    todo = [c for c in _callees(window) if c != kernel and "zone_" not in c]
    while todo and len(seen) < 64:
        function = todo.pop(0)
        if function in seen or function not in funcs:
            continue
        seen.add(function)
        body = funcs[function]
        offsets = {a: a - body[0][0] for a, _, _ in body}
        lines.append(f"<{function}>")
        lines += [_normalise(m, o, function, offsets) for _, m, o in body]
        todo += [
            c
            for c in _callees(body)
            if c not in seen and c != kernel and "zone_" not in c
        ]
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()[:12]


def fingerprint(objdump, elf_path):
    digest = hashlib.sha1(Path(elf_path).read_bytes()).hexdigest()
    if digest not in _CACHE:
        _CACHE[digest] = _compute(objdump, elf_path)
    return _CACHE[digest]


def enabled():
    return os.environ.get("LLK_WINDOW_HASH") == "1"


def write_rows(run_dir, test_name, sweep, rows):
    import pandas as pd

    if not rows:
        return
    frame = sweep.merge(
        pd.DataFrame(rows, columns=["window_run_type", "window_thread", "window_hash"]),
        how="cross",
    )
    frame["nodeid"] = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" ", 1)[0]
    run_dir.mkdir(parents=True, exist_ok=True)
    name = test_name.replace("/", "_").replace(".", "_")
    # Tests that share a source can have different columns; each column set gets its own file.
    columns = hashlib.sha1(",".join(frame.columns).encode()).hexdigest()[:8]
    path = run_dir / f"{name}.{columns}.{os.getpid()}.windowhash"
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
