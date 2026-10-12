#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Experiment CI legs (calleepin): the out of line functions each TILE_LOOP and INIT window can run, per module and
thread, from a Wormhole perf build. TILE_LOOP: test_config.py's own analysis on every kept thread assembly
(obj/<thread>.s.gz): the functions it places in .text.llk_loop_callee.*, the external ones (libraries) the window calls,
and the fixed harness ones. INIT: the functions placed in the INIT region of every INIT measurement ELF (symbols between
zone_record and the trampoline). usage: exp_callees.py RUNNER_TEMP OUT.json.gz"""
import sys, os, glob, gzip, json, collections, struct
from pathlib import Path

sys.path.insert(0, os.environ.get("EXP_PT") or str(Path(__file__).resolve().parent / "python_tests"))
from helpers import test_config as tc  # noqa: E402

rt, out = sys.argv[1], sys.argv[2]
# libgcc's integer and soft float helpers
LIBGCC = __import__('re').compile(r"__(float|fix|add|sub|mul|div|mod|udiv|umod|neg|cmp|eq|ne|lt|le|gt|ge|un|ash|lsh|clz|ctz|pop|extend|trunc|bswap|powi)\w*$")
tl = collections.defaultdict(collections.Counter)
nvar = collections.Counter()
for p in sorted(glob.glob(f"{rt}/tt-llk-build/sources/*/*/obj/*.s.gz")):
    parts = p.split("/")
    key = f"{parts[-4]}|{parts[-1].split('.')[0]}"
    nvar[key] += 1
    try:
        lines = gzip.open(p, "rt").read().split("\n")
        functions, _ = tc._asm_functions(lines)
        kernel = next((f for n, f in functions.items() if n.startswith("_Z10run_kernel")), None)
        if kernel is None:
            tl[key]["!no run_kernel"] += 1
            continue
        libs = {"memset", "memcpy", "memmove", "memcmp"}
        lib = lambda t: t in libs or bool(LIBGCC.match(t))
        names = type("N", (), {"__contains__": lambda self, t: t in functions or lib(t)})()
        todo = tc._loop_window_calls(lines, kernel, names)
        if not todo:
            tl[key]["!no window calls"] += 1
        seen = set(todo)
        while todo:
            name = todo.pop(0)
            if name not in functions:
                tl[key][f"external {name}"] += 1
                continue
            start, label, end, section, _ = functions[name]
            if section == ".llk_init_text" or name.startswith("_Z10run_kernel"):
                continue
            tl[key][f"{'moved' if section.startswith('.text.llk_loop_callee') else 'fixed ' + section} {name}"] += 1
            for line in lines[label + 1: end]:
                m = tc._CALL.match(line)
                refs = [m.group(1)] if m else []
                if not m and (i := tc._ASM_INSTRUCTION.match(line)):
                    refs = tc._ASM_SYMBOL.findall(i.group(2) or "")
                for c in refs:
                    if c not in seen and c in names:
                        seen.add(c)
                        todo.append(c)
    except Exception as e:  # noqa: BLE001
        tl[key][f"!error {e!r}"[:200]] += 1


def symbols(path):
    """(sorted function symbols (address, size, name), read(address, n) -> bytes) of an ELF"""
    data = Path(path).read_bytes()
    (shoff,) = struct.unpack_from("<I", data, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x2E)
    heads = [struct.unpack_from("<IIIIIIIIII", data, shoff + i * shentsize) for i in range(shnum)]
    name = lambda off, tab: data[tab[4] + off: data.index(b"\0", tab[4] + off)].decode()
    out, secs = [], []
    for h in heads:
        if h[2] & 2 and h[1] != 8:
            secs.append((h[3], data[h[4]: h[4] + h[5]]))
        if h[1] == 2:
            for off in range(h[4], h[4] + h[5], 16):
                st_name, st_value, st_size, st_info = struct.unpack_from("<IIIB", data, off)
                if st_info & 0xF == 2:  # STT_FUNC
                    out.append((st_value, st_size, name(st_name, heads[h[6]])))

    def read(a, n):
        for b, d in secs:
            if b <= a and a + n <= b + len(d):
                return d[a - b: a - b + n]
        return b""
    return sorted(out), read


def jal_targets(code, base):
    out = []
    for i in range(0, len(code) - 3, 4):
        w = struct.unpack_from("<I", code, i)[0]
        if w & 0x7F == 0x6F:
            off = ((w >> 31) & 1) << 20 | ((w >> 12) & 0xFF) << 12 | ((w >> 20) & 1) << 11 | ((w >> 21) & 0x3FF) << 1
            off -= (1 << 21) if off & (1 << 20) else 0
            out.append((base + i + off) & 0xFFFFFFFF)
    return out


init = collections.defaultdict(collections.Counter)
ninit = collections.Counter()
for e in sorted(glob.glob(f"{rt}/tt-llk-build/sources/*/init_sol/*/init_elf/*.elf")):
    parts = e.split("/")
    key = f"{parts[-5]}|{parts[-1].split('.')[0]}"
    ninit[key] += 1
    try:
        syms, read = symbols(e)
        tramp = [(a, s_) for a, s_, n in syms if n.startswith("_ZN8llk_perf9init_zone")]
        rec = [a + s_ for a, s_, n in syms if n.startswith("_ZN12llk_profiler11zone_record")]
        if not tramp:
            init[key]["!no trampoline"] += 1
            continue
        t0, tsz = tramp[0]
        lo = rec[0] if rec and rec[0] <= t0 else 0
        at = {a: n for a, s_, n in syms}
        for a, s_, n in syms:
            if lo <= a <= t0 and s_ and not n.startswith("_ZN12llk_profiler"):
                init[key][n] += 1
                for t in jal_targets(read(a, s_), a):
                    if not lo <= t < t0 + tsz and not at.get(t, "").startswith("_ZN12llk_profiler"):
                        init[key][f"external {at.get(t, hex(t))}"] += 1
    except Exception as ex:  # noqa: BLE001
        init[key][f"!error {ex!r}"[:200]] += 1

res = {k: {"variants": nvar[k], "init_elfs": ninit[k], "tile_loop": dict(tl[k]), "init": dict(init[k])}
       for k in sorted(set(nvar) | set(ninit))}
with gzip.open(out, "wt") as f:
    json.dump(res, f)
print(f"exp_callees: {sum(nvar.values())} assemblies, {sum(ninit.values())} INIT ELFs -> {out}")
