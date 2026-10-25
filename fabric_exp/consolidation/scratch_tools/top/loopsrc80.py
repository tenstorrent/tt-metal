# Loop instructions / sp loads / sp stores grouped by router source function region, for s79m vs s80 ELFs.
# usage: loopsrc80.py <variant-src> <elf> [<variant-src> <elf> ...]
import sys, re
sys.path.insert(0, "/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad")
from linemap import parse
from loop79 import loop
from collections import Counter
def regions(src):
    L = open(src).read().split("\n")
    marks = []
    for pat, name in [("FORCE_INLINE bool sender_channel_try_send(", "try_send"),
                      ("FORCE_INLINE void sender_channel_process_completions(", "process_completions"),
                      ("    run_worker_sender_step(", "worker_step"),
                      ("    run_forwarding_sender_step(", "fwd_step(templated)"),
                      ("struct ForwardingSenderChannelState {", "fwd_state/init"),
                      ("FORCE_INLINE bool run_forwarding_sender_step_runtime(", "fwd_step_runtime"),
                      ("    run_sender_channel_step(", "run_sender_channel_step"),
                      ("FORCE_INLINE void run_fabric_edm_main_loop(", "main_loop")]:
        for i, l in enumerate(L):
            if pat in l:
                marks.append((i + 1, name)); break
    marks.sort()
    def reg(ln):
        r = "other"
        for a, n in marks:
            if ln >= a: r = n
        return r
    return reg
args = sys.argv[1:]
for src, elf in zip(args[::2], args[1::2]):
    reg = regions(src)
    body = loop(elf); a, b = body[0][0], body[-1][0]
    c = Counter(); ld = Counter(); st = Counter()
    for addr, op, arg, (f, ln) in parse(elf):
        if a <= addr <= b:
            k = reg(ln) if f == "fabric_erisc_router_v2.cpp" else f
            c[k] += 1
            if "(sp)" in arg and op in ("lw", "lbu", "lhu", "lb", "lh"): ld[k] += 1
            if "(sp)" in arg and op in ("sw", "sb", "sh"): st[k] += 1
    print(elf.split("/")[-3], elf.split("/")[-2], f"total {sum(c.values())}")
    for k in sorted(c, key=lambda k: -c[k]):
        print(f"   {k:34s} insns {c[k]:4d}  sp-ld {ld[k]:3d}  sp-st {st[k]:3d}")
