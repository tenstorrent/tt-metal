#!/usr/bin/env python3
"""dyn_size.py ARM=CACHE[,CACHE...] ...: code size and stack of the router ELFs device runs built, per slot arm.

The code-size check of the dynamic slot allocator's A/B (SLOT_ALLOCATOR.md): dyn_p150_runs.sh keeps one kernel cache
per suite and arm (/home/aagarwal/.tt-metal-cache-dyn-<suite>-<arm>); every v2-slot router ELF in an arm's caches is
measured as size_report.py does (loadable bytes, deepest static stack from _start, free stack), and the arms compared:
per arm and ERISC the largest code and the least free stack, and E0 + E1 (an upper bound for one router) against the
26,624 B kernel config buffer. Non-power-of-2 depths add a little code (index wrap, partial completion groups), and
deeper tabulated sender channels add stack (MAX_TABULATED_SLOTS = 32 slots per channel are stored).

  dyn_size.py tier=/home/aagarwal/.tt-metal-cache-dyn-2d3k-tier dyn=/home/aagarwal/.tt-metal-cache-dyn-2d3k-dyn
  dyn_size.py 'tier=SCR/sizevet/tier/*/cache_*' 'dyn=SCR/sizevet/dyn/*/cache_*'   (globs: mock-vet caches)
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import size_report as sr  # noqa: E402


def elfs(cache):
    for proc in sr.ENTRY_SP:
        for e in glob.glob(os.path.join(cache, "*", "kernels", "fabric_erisc_router_v2", "*", proc, f"{proc}.elf")):
            yield proc, e


def main():
    arms = []
    for a in sys.argv[1:]:
        name, _, caches = a.partition("=")
        if not caches:
            sys.exit(__doc__)
        arms.append((name, [c for pat in caches.split(",") for c in (sorted(glob.glob(pat)) or [pat])]))
    if not arms:
        sys.exit(__doc__)
    print(f"| arm | ELFs | E0 code | E1 code | E0+E1 (vs {sr.KERNEL_CONFIG}) | least free stack E0 | E1 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, caches in arms:
        found = [(p, e) for c in caches for p, e in elfs(c)]
        res = [(sr.ERISC[p], sr.one(e, p)) for p, e in found]
        code = {x: max((r["code"] for k, r in res if k == x), default=0) for x in ("E0", "E1")}
        free = {x: min((r["free"] for k, r in res if k == x), default=0) for x in ("E0", "E1")}
        print(f"| {name} | {len(res)} | {code['E0']} | {code['E1']} | {code['E0'] + code['E1']} | "
              f"{free['E0']} | {free['E1']} |")


if __name__ == "__main__":
    main()
