# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Experiment: fingerprint of the code a TILE_LOOP zone executes, independent of where it sits.

The window is run_kernel's code between the TILE_LOOP zone's zone_reserve call and its zone_record call, plus the
functions it calls. Branch and call targets are written relative to the window start (inside run_kernel) or as
symbol + offset (elsewhere), so moving the whole kernel by a NOP leaves the fingerprint unchanged and a change to an
executed instruction changes it.
"""

import hashlib
import re
import subprocess
from pathlib import Path

_HEADER = re.compile(r"^([0-9a-f]+) <(.+)>:$")
_INSN = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
_TARGET = re.compile(r"\b([0-9a-f]+) <([^>+]+)(?:\+0x([0-9a-f]+))?>")
_CALLS = ("jal", "call", "tail", "j")


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


def _norm(mnemonic, operands, base_sym, base_addr):
    def repl(m):
        addr, sym = int(m.group(1), 16), m.group(2)
        if sym == base_sym:
            return f"@{addr - base_addr}"
        return f"{sym}+{m.group(3) or '0'}"

    return f"{mnemonic} {_TARGET.sub(repl, operands)}"


def _callees(insns):
    out = []
    for _, mnemonic, operands in insns:
        if mnemonic in _CALLS or mnemonic.startswith("jal"):
            m = _TARGET.search(operands)
            if m:
                out.append(m.group(2))
    return out


_CACHE = {}


def _fingerprint_bytes(objdump, elf_path):
    result = subprocess.run(
        [objdump, "-d", "--no-show-raw-insn", elf_path], capture_output=True, text=True
    )
    if result.returncode != 0:
        return "objdump-error", ""
    funcs = _functions(result.stdout)
    kernel = next((name for name in funcs if "run_kernel" in name), None)
    if kernel is None:
        kernel = "main" if "main" in funcs else None
    if kernel is None:
        return "no-kernel", ""
    insns = funcs[kernel]
    opens = [
        i
        for i, (_, mn, op) in enumerate(insns)
        if "zone_reserve" in op and (mn in _CALLS or mn.startswith("jal"))
    ]
    closes = [
        i
        for i, (_, mn, op) in enumerate(insns)
        if "zone_record" in op and (mn in _CALLS or mn.startswith("jal"))
    ]
    if not opens or not closes:
        return "no-zones", ""
    start = end = None
    for o in reversed(opens):
        c = next((c for c in closes if c > o), None)
        if c is not None and not any(o < x < c for x in opens):
            start, end = o, c
            break
    if start is None:
        return "no-window", ""
    window = insns[start : end + 1]
    base = window[0][0]
    lines = [_norm(mn, op, kernel, base) for _, mn, op in window]
    seen, todo = set(), [
        c for c in _callees(window) if c != kernel and "zone_" not in c
    ]
    while todo and len(seen) < 64:
        fn = todo.pop(0)
        if fn in seen or fn not in funcs:
            continue
        seen.add(fn)
        body = funcs[fn]
        fbase = body[0][0] if body else 0
        lines.append(f"<{fn}>")
        lines += [_norm(mn, op, fn, fbase) for _, mn, op in body]
        todo += [
            c
            for c in _callees(body)
            if c not in seen and c != kernel and "zone_" not in c
        ]
    joined = "\n".join(lines)
    return hashlib.sha1(joined.encode()).hexdigest()[:12], joined


def fingerprint(objdump: str, elf_path: Path):
    digest = hashlib.sha1(Path(elf_path).read_bytes()).hexdigest()
    if digest not in _CACHE:
        _CACHE[digest] = _fingerprint_bytes(objdump, str(elf_path))
    return _CACHE[digest]


_DUMPED = set()


def write_rows(run_dir: Path, test_name: str, sweep, rows):
    import os

    import pandas as pd

    if not rows:
        return
    frame = sweep.merge(
        pd.DataFrame(
            [r[:3] for r in rows],
            columns=["window_run_type", "window_thread", "window_hash"],
        ),
        how="cross",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    safe = test_name.replace("/", "_").replace(".", "_")
    path = run_dir / f"{safe}.{os.getpid()}.windowhash"
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
    dump_dir = run_dir / "windows"
    dump_dir.mkdir(exist_ok=True)
    for _, _, digest, text in rows:
        if digest not in _DUMPED and len(_DUMPED) < 400 and text:
            _DUMPED.add(digest)
            (dump_dir / f"{digest}.txt").write_text(text)
