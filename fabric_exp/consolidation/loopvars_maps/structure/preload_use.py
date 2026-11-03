#!/usr/bin/env python3
"""preload_use.py BUILD [CASE,...]: in each router image's receiver loop function (two ERISCs: receiver::run; one:
single::receiver_pass), the route loads and where their values are first used. A load is a route load when its
addr2line inline chain passes through the codec's load_route_bytes (2D) or Routing1D's load (1D); it is a preload when
the chain's router_receiver.hpp frame is in step (issued after the packet), load-now when it is in receive. For each
load: the instructions on every control-flow path from the load to the first instruction that reads its register
(shortest / longest path), and that instruction (a store into channel state, a stack store = spill, or the decode).
Counts are static instructions, not cycles."""
import collections, glob, os, re, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import loopvars as L
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
LOADS = ("lbu", "lb", "lhu", "lh", "lw")


def first_uses(body, i, reg):
    """[(distance, instruction)] over the paths from body[i] to the first reader of reg (a redefinition ends a path)."""
    succ, _ = L.cfg(body)
    idx = {a: k for k, (a, _, _) in enumerate(body)}
    out, seen = [], set()
    stack = [(s, 1) for s in succ.get(body[i][0], [])]
    while stack:
        a, d = stack.pop()
        if (a, d) in seen or d > 200 or a not in idx:
            continue
        seen.add((a, d))
        k = idx[a]
        op, raw = body[k][1], body[k][2]
        w, r, call = L.defuse(op, raw)
        if reg in r:
            out.append((d, f"{op} {raw}"))
            continue
        if reg in w:
            out.append((d, f"(dead: {op} {raw})"))
            continue
        stack.extend((s, d + 1) for s in succ.get(a, []))
    return out


def analyse(elf):
    fns = L.functions(elf)
    names = [n for n in fns if "8receiver3run" in n] or [n for n in fns if "6single13receiver_pass" in n]
    if not names:
        return None
    body = fns[names[0]]
    loads = [(k, a) for k, (a, op, raw) in enumerate(body) if op in LOADS]
    ch = L.chains(elf, [a for _, a in loads])
    res = []
    for k, a in loads:
        c = ch.get(a, [])
        fnames = [f for f, _ in c]
        if not any(f.startswith(("load_route_bytes", "load_route_words")) for f in fnames) and not (
                "load" in fnames and any(fl.startswith("router_routing.hpp") for _, fl in c)):
            continue
        where = next((f for f, fl in c if fl.startswith("router_receiver.hpp") and f.split("<")[0] in ("step", "receive")), "?")
        reg = body[k][2].split(",")[0]
        uses = first_uses(body, k, reg)
        res.append({"kind": "preload" if where.startswith("step") else "load-now" if where.startswith("receive") else where,
                    "insn": f"{body[k][1]} {body[k][2]}", "uses": uses})
    return names[0], res


if __name__ == "__main__":
    build = sys.argv[1]
    cases = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    fam = collections.defaultdict(list)
    for elf in sorted(glob.glob(f"{CB}/{build}/*/subordinate_active_erisc/*.elf")):
        key = elf.split("/")[-3]
        case = key.split("__")[0]
        if cases and case not in cases:
            continue
        r = analyse(elf)
        if r:
            fam[case].append(r[1])
    for case, imgs in sorted(fam.items()):
        for kind in ("preload", "load-now"):
            ds = [d for img in imgs for x in img if x["kind"] == kind for d, _ in x["uses"]]
            spills = sum(1 for img in imgs for x in img if x["kind"] == kind for _, u in x["uses"] if "(sp)" in u and u.startswith("s"))
            kinds = collections.Counter(re.sub(r"\s.*", "", u) for img in imgs for x in img if x["kind"] == kind for _, u in x["uses"])
            n = sum(1 for img in imgs for x in img if x["kind"] == kind)
            if n:
                print(f"{case:28s} {kind:8s} loads {n:4d}  first use after {min(ds) if ds else '-'}..{max(ds) if ds else '-'} insns"
                      f"  spill stores {spills}  first-use ops {dict(kinds.most_common(4))}")
