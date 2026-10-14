#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Experiment CI legs (cpin2): static signature of the TILE_LOOP window of every Wormhole perf TILE_LOOP ELF (variant
and layout copies), per thread. Window = run_kernel's instructions reachable from the TILE_LOOP restart (the first
instruction after an ebreak followed by straight-line code to a wall clock read), both branch ways, calls falling
through, up to an ebreak, a jalr or a jump out of run_kernel; plus every function those code paths call (jal), whole.
abs: sha1 of the (address, word) pairs; rel: the same with addresses relative to the restart; mset: sha1 of the sorted
words with branch / jal offsets masked (the same code in any order or place). usage: exp_tilewin.py RUNNER_TEMP OUT.json.gz"""
import sys, os, json, gzip, struct, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
M32 = 0xFFFFFFFF
EBREAK = 0x00100073
FILL = (0x00000013, 0x00000000)
RUN_KERNEL = "_Z10run_kernelRK13RuntimeParams"


def sext(x, b):
    return x - (1 << b) if x & (1 << (b - 1)) else x


def jal_off(w):
    return sext(((w >> 31) & 1) << 20 | ((w >> 12) & 0xFF) << 12 | ((w >> 20) & 1) << 11 | ((w >> 21) & 0x3FF) << 1, 21)


def br_off(w):
    return sext(((w >> 31) & 1) << 12 | ((w >> 7) & 1) << 11 | ((w >> 25) & 0x3F) << 5 | ((w >> 8) & 0xF) << 1, 13)


def mask(w):
    op = w & 0x7F
    return w & ~0xFE000F80 & M32 if op == 0x63 else (w & 0xFFF if op == 0x6F else w)


def load(path):
    data = Path(path).read_bytes()
    (shoff,) = struct.unpack_from("<I", data, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x2E)
    heads = [struct.unpack_from("<IIIIIIIIII", data, shoff + i * shentsize) for i in range(shnum)]
    name = lambda off, tab: data[tab[4] + off: data.index(b"\0", tab[4] + off)].decode()
    words, funcs, rk = {}, {}, None
    for h in heads:
        if h[2] & 4 and h[2] & 2 and h[1] != 8:  # SHF_EXECINSTR + SHF_ALLOC, not NOBITS
            for i in range(0, h[5] - 3, 4):
                words[h[3] + i] = struct.unpack_from("<I", data, h[4] + i)[0]
        if h[1] == 2:
            for off in range(h[4], h[4] + h[5], 16):
                st_name, st_value, st_size, st_info = struct.unpack_from("<IIIB", data, off)
                if st_info & 0xF == 2 and st_size:  # STT_FUNC
                    nm = name(st_name, heads[h[6]])
                    funcs[st_value] = (nm, st_size)
                    if nm == RUN_KERNEL:
                        rk = (st_value, st_value + st_size)
    return words, funcs, rk


def restart_of(words, rk):
    for e in sorted(a for a in range(rk[0], rk[1], 4) if words.get(a) == EBREAK):
        a = e + 4
        while a < rk[1] and words.get(a) in FILL:
            a += 4
        r = a
        while a < rk[1]:
            w = words.get(a)
            if w is None or w & 0x7F in (0x63, 0x67, 0x6F):
                break
            if w & 0x707F == 0x2003 and sext(w >> 20, 12) == 496:
                return r
            a += 4
    return None


def sig(path):
    try:
        words, funcs, rk = load(path)
        if rk is None:
            return {"err": "no run_kernel"}
        restart = restart_of(words, rk)
        if restart is None:
            return {"err": "no restart"}
        seen, todo, callees = set(), [restart], set()
        while todo:
            a = todo.pop()
            if a in seen or not (rk[0] <= a < rk[1]):
                continue
            seen.add(a)
            w = words.get(a)
            if w is None or w == EBREAK:
                continue
            op = w & 0x7F
            if op == 0x63:
                todo += [a + 4, (a + br_off(w)) & M32]
            elif op == 0x6F:
                t = (a + jal_off(w)) & M32
                if (w >> 7) & 31 == 0:
                    todo.append(t)
                else:
                    callees.add(t)
                    todo.append(a + 4)
            elif op == 0x67:
                continue
            else:
                todo.append(a + 4)
        # callees, whole, and what they call
        done, ctodo, cseen = set(), list(callees), set()
        while ctodo:
            t = ctodo.pop()
            if t in done or t not in funcs or rk[0] <= t < rk[1]:
                continue
            done.add(t)
            for a in range(t, t + funcs[t][1], 4):
                w = words.get(a)
                cseen.add(a)
                if w is not None and w & 0x7F == 0x6F and (w >> 7) & 31:
                    ctodo.append((a + jal_off(w)) & M32)
        win = sorted((a, words.get(a, 0)) for a in seen)
        cal = sorted((a, words.get(a, 0)) for a in cseen)
        h = lambda x: hashlib.sha1(repr(x).encode()).hexdigest()[:16]
        # the window's fall-through chains (consecutive reachable words, cut after an unconditional jump): the same
        # chains in another order or place is a move of the same code, a chain whose words differ is a code change
        chains, cur, prev = [], [], None
        for a, w in win:
            if prev is not None and a != prev + 4 and cur:
                chains.append(tuple(cur))
                cur = []
            cur.append(mask(w))
            prev = a
            if (w & 0x7F == 0x6F and (w >> 7) & 31 == 0) or w & 0x7F == 0x67:
                chains.append(tuple(cur))
                cur = []
        if cur:
            chains.append(tuple(cur))
        # per callee: name, address, its words with offsets masked (a callee moved as a whole compares by content)
        clist = sorted([funcs[t][0], t, h([mask(words.get(a, 0)) for a in range(t, t + funcs[t][1], 4)])] for t in done)
        return {
            "restart": restart,
            "n": len(win),
            "nc": len(cal),
            # run_kernel's window: words at their addresses / relative to the restart (offsets masked, so a move of
            # the window alone keeps it) / as a multiset (the same code anywhere)
            "abs": h(win),
            "rel": h([(a - restart, mask(w)) for a, w in win]),
            "mset": h(sorted(mask(w) for _, w in win)),
            "cset": h(sorted(chains)),
            # the functions it calls: at their addresses / as a multiset
            "cabs": h(cal),
            "cmset": h(sorted(mask(w) for _, w in cal)),
            "callees": clist,
        }
    except Exception as e:  # noqa
        return {"err": repr(e)[:160]}


def main(rt, out):
    root = Path(rt) / "tt-llk-build" / "sources"
    elfs = []
    for t in ("unpack", "math", "pack"):
        elfs += [p for p in root.glob(f"*/*/elf/{t}.elf")] + [p for p in root.glob(f"*/*/layout_*/elf/{t}.elf")]
    with ProcessPoolExecutor(int(os.environ.get("EXP_TILEWIN_JOBS", "12"))) as ex:
        res = dict(zip((str(p.relative_to(root)) for p in elfs), ex.map(sig, elfs, chunksize=64)))
    json.dump(res, gzip.open(out, "wt"))
    print(f"{len(res)} TILE_LOOP ELFs, {sum('err' in v for v in res.values())} without a window -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
