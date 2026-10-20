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
_BRANCH = re.compile(r"^(b[a-z]*|j|c\.j|c\.b[a-z]*)$")
_ENDS = ("ret", "jr", "mret", "c.jr")
_TILE_LOOP = re.compile(rb"LLK_PROFILER:[^\x00]*?:TILE_LOOP\x00")


def _hash16(text):
    h = 2166136261
    for c in text:
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return (h ^ (h >> 16)) & 0xFFFF


def _a0_before(insns, i):
    hi = lo = None
    for _, mn, op in reversed(insns[max(0, i - 6) : i]):
        args = [a.strip() for a in op.split(",")]
        if not args or args[0] != "a0":
            continue
        if mn == "addi" and len(args) == 3 and args[1] == "a0" and lo is None:
            lo = int(args[2], 0)
        elif mn == "lui" and hi is None:
            hi = int(args[1], 0) << 12
            break
        elif mn == "li":
            return int(args[1], 0) & 0xFFFF
    return None if hi is None else (hi + (lo or 0)) & 0xFFFF


def _is_call(mn, op, sym):
    return (mn in _CALLS or mn.startswith("jal")) and sym in op


def _region(insns, start, stop_ids):
    index = {a: k for k, (a, _, _) in enumerate(insns)}
    order, seen, todo = [], set(), [start]
    while todo:
        k = todo.pop()
        while k is not None and k < len(insns) and k not in seen:
            if k != start and _is_call(insns[k][1], insns[k][2], "zone_reserve"):
                break
            seen.add(k)
            order.append(k)
            a, mn, op = insns[k]
            if _is_call(mn, op, "zone_record") and _a0_before(insns, k) in stop_ids:
                break
            target = _TARGET.search(op)
            t = index.get(int(target.group(1), 16)) if target else None
            if mn in _ENDS or (mn in ("j", "c.j", "tail") and t is None):
                break
            if _BRANCH.match(mn) and t is not None:
                if mn in ("j", "c.j"):
                    k = t
                    continue
                todo.append(t)
            k += 1
    return order


def _window(insns, data):
    ids = {_hash16(m.group(0)[:-1]) for m in _TILE_LOOP.finditer(data)}
    if not ids:
        return None
    reserves = [
        k for k, (_, mn, op) in enumerate(insns) if _is_call(mn, op, "zone_reserve")
    ]
    found = []
    for k in reserves:
        reg = _region(insns, k, ids)
        if any(
            _is_call(insns[x][1], insns[x][2], "zone_record")
            and _a0_before(insns, x) in ids
            for x in reg
        ):
            found.append(reg)
    return min(found, key=len) if found else None


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
    reg = _window(insns, Path(elf_path).read_bytes())
    if reg is None:
        dump = "\n".join(f"{a:x} {mn} {op}" for a, mn, op in insns)
        return "no-window-" + hashlib.sha1(dump.encode()).hexdigest()[:8], dump
    pos = {insns[k][0]: n for n, k in enumerate(reg)}

    def norm(mn, op, fn):
        def repl(m):
            a, sym = int(m.group(1), 16), m.group(2)
            if sym == fn:
                return f"#{pos[a]}" if a in pos else "#out"
            return f"{sym}+{m.group(3) or '0'}"

        return f"{mn} {_TARGET.sub(repl, op)}"

    window = [insns[k] for k in reg]
    lines = [norm(mn, op, kernel) for _, mn, op in window]
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
