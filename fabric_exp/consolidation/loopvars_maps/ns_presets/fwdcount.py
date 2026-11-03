#!/usr/bin/env python3
"""fwdcount.py BUILD VARIANT CONFIG_DIR...: every inlined forward (Downstreams::forward, Relay::forward) in the receiver
loop of each image: its command part (the NoC commands, between the slot read and the slot advance) split into NoC
register writes, other stores, command-buffer ready reads, other loads, branches and the rest; and its bookkeeping part.
Static counts of executed-path instructions are NOT computed: a copy's count includes its not-ready wait loops (listed
separately as 'wait' = instructions on the wait lines)."""
import collections, os, re, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import loopvars as L

CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
BASE = "/home/aagarwal/.tt-metal-p1-base-m"
build, variant = sys.argv[1:3]
cfgs = sys.argv[3:]

_index = {}


def find(basename):
    """The source file the build compiled for `basename`: the variant's copy first, then the offline base."""
    if basename in _index:
        return _index[basename]
    hit = None
    for root in (f"{CB}/{build}/overlay", BASE + "/tt_metal"):
        cands = []
        for dp, _, fs in os.walk(root):
            if basename in fs and "/build" not in dp[len(root):] and "/.git" not in dp[len(root):]:
                cands.append(os.path.join(dp, basename))
        if cands:
            cands.sort(key=lambda p: (("blackhole" not in p and "tt-1xx" in p), "quasar" in p or "tt-2xx" in p, len(p)))
            hit = cands[0]
            break
    _index[basename] = open(hit).read().splitlines() if hit else []
    return _index[basename]


def text(loc):
    if ":" not in loc:
        return ""
    f, ln = loc.rsplit(":", 1)
    lines = find(f)
    n = int(ln)
    return lines[n - 1] if 0 < n <= len(lines) else ""


rd = find("router_downstream.hpp")


def span(start_pat, end_pat):
    a = next(i for i, l in enumerate(rd) if start_pat in l) + 1
    b = next(i for i, l in enumerate(rd) if i >= a and end_pat in l) + 1
    return a, b


SPANS = [span("const uint32_t dst = progress.next_slot_addr;", "progress.next_slot_addr = slot_after"),
         span("const uint32_t dst = downstream.next_slot_addr;", "downstream.next_slot_addr = slot_after")]
# The forward functions' bodies (Relay::forward, Downstreams::forward): a frame there is a forward copy.
BODIES = [span("FORCE_INLINE void forward(uint32_t src_addr", "increment_local_update_ptr_val")
          for _ in range(1)]
BODIES = []
for i, l in enumerate(rd):
    if "FORCE_INLINE void forward(uint32_t src_addr" in l:
        b = next(j for j in range(i, len(rd)) if "increment_local_update_ptr_val" in rd[j])
        BODIES.append((i + 1, b + 1))


def in_body(loc):
    if not loc.startswith("router_downstream.hpp:"):
        return False
    n = int(loc.split(":")[1])
    return any(a <= n <= b for a, b in BODIES)


def classify(op, chain, fwd_line):
    inner_fn, inner_loc = chain[0]
    t = text(inner_loc)
    in_cmd = any(a < fwd_line < b for a, b in SPANS)
    part = "cmd" if in_cmd else "book"
    if op in ("sw", "sh", "sb"):
        kind = "reg_w" if inner_fn == "NOC_CMD_BUF_WRITE_REG" else "st"
    elif op in ("lw", "lh", "lhu", "lb", "lbu"):
        kind = "ready_r" if inner_fn == "NOC_CMD_BUF_READ_REG" else "ld"
    elif op.startswith("b") or op in ("j", "jal", "jalr", "ret") or op.startswith("c.b") or op.startswith("c.j"):
        kind = "br"
    else:
        kind = "alu"
    wait = any("while (!noc_cmd_buf_ready" in text(loc) for _, loc in chain)
    return part, kind, wait


def image(cfg):
    elf = f"{CB}/{build}/{cfg}/subordinate_active_erisc/subordinate_active_erisc.elf"
    if not os.path.exists(elf):
        elf = next(os.path.join(dp, f) for dp, _, fs in os.walk(f"{CB}/{build}/{cfg}") for f in fs if f.endswith(".elf"))
    fns = L.functions(elf)
    names = [k for k in fns if "8receiver3run" in k] or [k for k in fns if "6single13receiver_pass" in k]
    body = fns[names[0]]
    succ, _ = L.cfg(body)
    loop = L.largest_scc([a for a, *_ in body], succ) if "run" in names[0] else {a for a, *_ in body}
    ops = {a: op for a, op, _ in body}
    ch = L.chains(elf, sorted(loop))
    copies = collections.defaultdict(collections.Counter)
    for a in sorted(loop):
        chain = ch.get(a, [])
        k = next((i for i, (fn, loc) in enumerate(chain) if in_body(loc)), None)
        if k is None:
            continue
        sig = tuple(loc for _, loc in chain[k + 1:])
        part, kind, wait = classify(ops[a], chain, int(chain[k][1].split(":")[1]))
        c = copies[sig]
        c[part] += 1
        c[f"{part}.{kind}"] += 1
        if wait:
            c[f"{part}.wait"] += 1
    return len(loop), copies


for cfg in cfgs:
    n, copies = image(cfg)
    print(f"{cfg}: loop {n}, forward copies {len(copies)}")
    for sig, c in sorted(copies.items(), key=lambda kv: kv[0]):
        where = " < ".join(s for s in sig if s.startswith("router_"))[:110]
        print(f"  path {c['cmd'] - c['cmd.wait']:3d} cmd {c['cmd']:3d} (reg_w {c['cmd.reg_w']:2d} st {c['cmd.st']:2d} ready_r {c['cmd.ready_r']:2d} "
              f"ld {c['cmd.ld']:2d} br {c['cmd.br']:2d} alu {c['cmd.alu']:2d}; wait {c['cmd.wait']:2d})  "
              f"book {c['book']:3d}  @ {where}")
