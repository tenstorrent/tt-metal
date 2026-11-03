#!/usr/bin/env python3
"""loopbody.py BASE NEW LIST [--offsets]: for each image in LIST, the loop of each run function (sender::run, receiver::run,
single::run: the largest strongly connected region, as loopvars.py finds it) and the whole of each one-ERISC pass
function (single::sender_pass / receiver_pass), compared between two cov_m builds instruction by instruction, with
branch and call targets by symbol+offset and nothing else normalised (--offsets: also stack offsets, `N(sp)` and
`addi R,sp,N`, for a frame-layout-only check). Prints, per function, how many images have an
identical loop and how many do not (and the first few of those)."""
import collections, concurrent.futures as cf, re, subprocess, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import loopvars as L
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
B = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
OFFSETS = "--offsets" in sys.argv
# Mangled-name prefixes: a substring match would also take a function instantiated with run()'s lambdas (e.g.
# telemetry::round_end_impl<..., receiver::run(...)::{lambda()#1}, ...> on telemetry images).
RUNS = {"_ZN13fabric_router6sender3runEv": "sender::run", "_ZN13fabric_router8receiver3runE": "receiver::run",
        "_ZN13fabric_router6single3runE": "single::run"}
PASSES = {"_ZN13fabric_router6single11sender_passE": "single::sender_pass",
          "_ZN13fabric_router6single13receiver_passE": "single::receiver_pass"}


def text(elf):
    out = subprocess.run([B + "objdump", "-d", "-C", "--no-show-raw-insn", "--section=.text", elf],
                         capture_output=True, text=True).stdout
    t = {}
    for l in out.splitlines():
        m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", l)
        if m:
            ops = re.sub(r"\s#.*", "", m.group(3))
            ops = re.sub(r"\b[0-9a-f]{3,8} <(.*)>\s*$", r"<\1>", ops)
            ops = re.sub(r"<\.L[^>]*>", "<L>", ops)
            ops = ops.strip()
            if OFFSETS:
                ops = re.sub(r"-?\d+\(sp\)", "K(sp)", ops)
                if m.group(2) == "addi":
                    ops = re.sub(r"^(\w+),sp,-?\d+$", r"\1,sp,K", ops)
            t[int(m.group(1), 16)] = m.group(2) + " " + ops
    return t


def loops(elf):
    fns, t, res = L.functions(elf), text(elf), {}
    for name, body in fns.items():
        for key, short in list(RUNS.items()) + list(PASSES.items()):
            if name.startswith(key):
                addrs = [a for a, *_ in body]
                if short in RUNS.values():
                    succ, _ = L.cfg(body)
                    sel = sorted(L.largest_scc(addrs, succ))
                else:
                    sel = addrs
                res[short] = [t[a] for a in sel]
    return res


def one(rel):
    return rel, loops(f"{CB}/{sys.argv[1]}/{rel}"), loops(f"{CB}/{sys.argv[2]}/{rel}")


cnt = collections.Counter()
bad = collections.defaultdict(list)
with cf.ProcessPoolExecutor(16) as ex:
    for rel, a, b in ex.map(one, open(sys.argv[3]).read().split()):
        for f in sorted(set(a) | set(b)):
            same = a.get(f) == b.get(f)
            cnt[(f, same)] += 1
            if not same:
                bad[f].append(rel)
for (f, same), n in sorted(cnt.items()):
    print(f"{n:5d}  {f:24s} {'loop identical' if same else 'LOOP DIFFERS'}")
for f, rels in bad.items():
    print(f"  {f}: e.g. {', '.join(r.split('/')[0] for r in rels[:4])}")
