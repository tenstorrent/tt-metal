#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Router bring-up tools (temporary: drop before merge; see README.md).

elf_size.py [--kernel NAME] [--detail] [--entry-sp IMAGE=HEX ...] [--limit BYTES] DIR_OR_ELF [DIR_OR_ELF ...]:
code size and static stack depth of router ERISC images.

Inputs: kernel caches (the router's images are found under .../kernels/<kernel>/<hash>/<image>/), offline builds
(offline_build.py), or ELF files. Per image (Wormhole: erisc; Blackhole: active_erisc = E0, subordinate_active_erisc
= E1), the largest and smallest over the configs found:
  code    loadable bytes (PT_LOAD file sizes: .text plus initialised data), what the kernel config buffer must hold
  .text   bytes of .text
  stack   deepest static stack use from _start: each function's frame from its sp adjustments, the call graph from
          jal (a call adds the caller's frame; a tail jump does not). An indirect call (jalr) is flagged, not followed
          (on a core whose base firmware stays resident, a jalr can enter it: add its depth by hand).
  free    with --entry-sp IMAGE=HEX (the stack pointer at kernel entry for that image, measured or from the
          firmware's memory map): entry sp minus the end of the image's data/bss (__ldm_bss_end) minus the stack depth
--limit BYTES flags images whose code is larger (e.g. the arch's ERISC kernel config buffer size).
--detail prints the deepest call chain of the deepest image per image kind.
Static numbers: they say whether and where code or stack grew, not what it costs at run time.
"""
import argparse
import collections
import concurrent.futures
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_tools_common import ERISC_IMAGES, ERISC_TAG, DEFAULT_KERNEL, router_images, toolchain_prefix  # noqa: E402

BIN = toolchain_prefix()
LEAVES = {"memset", "memcpy", "memmove"}  # their jalr is a computed jump inside the function


def run(*a):
    return subprocess.run([BIN + a[0], *a[1:]], capture_output=True, text=True).stdout


def code_bytes(elf):
    return sum(
        int(m.group(1), 16)
        for m in re.finditer(r"^\s+LOAD\s+\S+\s+\S+\s+\S+\s+(0x[0-9a-f]+)", run("readelf", "-lW", elf), re.M)
    )


def text_bytes(elf):
    m = re.search(r"^\s*\[\s*\d+\]\s+\.text\s+\S+\s+\S+\s+\S+\s+([0-9a-f]+)", run("readelf", "-SW", elf), re.M)
    return int(m.group(1), 16) if m else 0


def data_end(elf):
    m = re.search(r"^([0-9a-f]+) \w __ldm_bss_end$", run("nm", elf), re.M)
    return int(m.group(1), 16) if m else None


def functions(elf):
    f, cur = {}, None
    for line in run("objdump", "-d", "--no-show-raw-insn", elf).splitlines():
        m = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if m:
            if m.group(2).startswith(".L"):
                continue
            cur = int(m.group(1), 16)
            f[cur] = (m.group(2), [])
            continue
        m = re.match(r"^\s+[0-9a-f]+:\s+(\S+)\s*(.*)$", line)
        if m and cur is not None:
            f[cur][1].append((m.group(1), m.group(2)))
    return f


def analyse(fn, ins):
    depth = frame = 0
    calls, indirect, dyn = [], 0, False
    for mn, ops in ins:
        o = ops.replace(" ", "")
        m = re.match(r"sp,sp,(-?\d+)", o)
        if mn == "addi" and m:
            depth = max(0, depth - int(m.group(1)))
            frame = max(frame, depth)
        elif mn in ("add", "sub") and o.startswith("sp,sp,"):
            dyn = True
        m = re.match(r"([0-9a-f]+)<", o)
        if mn == "jal" and m:
            calls.append((int(m.group(1), 16), False))
        elif mn == "j" and m:
            calls.append((int(m.group(1), 16), True))
        elif mn == "jalr" and fn not in LEAVES:
            indirect += 1
    return frame, calls, indirect, dyn


def stack(elf):
    fs = functions(elf)
    info = {a: (n, *analyse(n, ins)) for a, (n, ins) in fs.items()}
    memo, flags = {}, set()

    def deep(a, path):
        if a in memo:
            return memo[a]
        if a in path:
            flags.add("recursion")
            return 0, []
        n, frame, calls, indirect, dyn = info[a]
        if dyn:
            flags.add("dynamic sp")
        if indirect:
            flags.add("indirect call")
        best, chain = frame, []
        for t, tail in calls:
            if t not in info or t == a:
                continue
            d, c = deep(t, path | {a})
            d += 0 if tail else frame
            if d > best:
                best, chain = d, c
        memo[a] = (best, [(n, frame)] + chain)
        return memo[a]

    starts = [a for a, (n, _) in fs.items() if n == "_start"]
    if not starts:
        return None, [], ["no _start"]
    d, chain = deep(starts[0], frozenset())
    return d, chain, sorted(flags)


def one(job):
    h, proc, elf, entry_sp = job
    d, chain, flags = stack(elf)
    r = {
        "hash": h,
        "image": proc,
        "elf": elf,
        "code": code_bytes(elf),
        "text": text_bytes(elf),
        "stack": d,
        "flags": flags,
        "chain": [f"{n} {f}" for n, f in chain],
    }
    if entry_sp is not None and d is not None:
        end = data_end(elf)
        r["free"] = entry_sp - end - d if end is not None else None
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--kernel", default=DEFAULT_KERNEL)
    ap.add_argument("--entry-sp", action="append", default=[], help="IMAGE=HEX, e.g. erisc=0xFFB01FC0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    sp = {k: int(v, 16) for k, v in (x.split("=", 1) for x in a.entry_sp)}
    jobs = []
    for inp in a.inputs:
        if os.path.isfile(inp):
            proc = os.path.basename(inp).split(".")[0]
            jobs.append(("-", proc, inp, sp.get(proc)))
        else:
            jobs += [(h, p, e, sp.get(p)) for h, p, e in router_images(inp, a.kernel)]
    if not jobs:
        sys.exit(f"no {a.kernel} images found (images: {', '.join(ERISC_IMAGES)})")
    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, jobs))
    by = collections.defaultdict(list)
    for r in res:
        by[r["image"]].append(r)
    print(
        f"{'image':26s} {'n':>4s} {'code min..max':>16s} {'.text min..max':>16s} {'stack min..max':>15s} "
        f"{'free min':>9s}  flags"
    )
    for proc in ERISC_IMAGES + tuple(sorted(set(by) - set(ERISC_IMAGES))):
        rs = by.get(proc)
        if not rs:
            continue
        st = [r["stack"] for r in rs if r["stack"] is not None]
        fr = [r["free"] for r in rs if r.get("free") is not None]
        fl = sorted({f for r in rs for f in r["flags"]})
        over = [r for r in rs if a.limit and r["code"] > a.limit]
        print(
            f"{proc + ' (' + ERISC_TAG.get(proc, proc) + ')':26s} {len(rs):4d} "
            f"{min(r['code'] for r in rs):7d}..{max(r['code'] for r in rs):<7d} "
            f"{min(r['text'] for r in rs):7d}..{max(r['text'] for r in rs):<7d} "
            f"{(str(min(st)) + '..' + str(max(st))) if st else '-':>15s} {min(fr) if fr else '-':>9}  "
            f"{', '.join(fl)}{'; ' + str(len(over)) + ' over --limit' if over else ''}"
        )
        if a.detail and st:
            deepest = max((r for r in rs if r["stack"] is not None), key=lambda r: r["stack"])
            print(f"   deepest: {deepest['hash']} {deepest['elf']}")
            for c in deepest["chain"]:
                print(f"      {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
