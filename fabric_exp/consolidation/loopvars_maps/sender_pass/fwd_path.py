#!/usr/bin/env python3
"""fwd_path.py BUILD_A BUILD_B REL...: the receiver loop of the given E1 images (cov_m builds/<name>/<REL>) in two builds,
per innermost source function (instructions, stack loads/stores, other loads/stores), and the forward's ready-read
region: each command-buffer register read in the loop (NOC_CMD_BUF_READ_REG), its distance in instructions to the first
use of its destination register, and whether the two reads of a forward issue back to back. Study analysis only."""
import collections, re, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import loopvars as L  # noqa: E402

CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
MEM = re.compile(r"(\w+),(-?\d+)\((\w+)\)")
REG = re.compile(r"\b([astx]\d+|ra|sp|gp|tp|zero)\b")


def analyse(elf):
    a = L.analyse(elf, "receiver", detail=True)
    loop = set(a["loop_addrs"])
    ch = L.chains(elf, sorted(loop))
    body = [(addr, op, raw) for addr, op, raw in a["body"] if addr in loop]
    per = collections.defaultdict(collections.Counter)
    reads = []
    for i, (addr, op, raw) in enumerate(body):
        chain = ch.get(addr) or [("?", "?")]
        fn = L.short(chain[0][0])
        c = per[fn]
        c["n"] += 1
        m = MEM.fullmatch(raw)
        if m and (op in L.LOADS or op in L.STORES):
            c[("s" if m.group(3) == "sp" else "m") + ("L" if op in L.LOADS else "S")] += 1
            if op in L.LOADS and "NOC_CMD_BUF_READ_REG" in fn and any("Forwards" in f for f, _ in chain):
                dst = m.group(1)
                use = next((j - i for j in range(i + 1, min(i + 40, len(body)))
                            if dst in REG.findall(body[j][2].split(",", 1)[1] if body[j][1] not in L.STORES and "," in body[j][2] else body[j][2])), None)
                reads.append((addr, dst, use))
    return a, per, reads


def main():
    ba, bb, rels = sys.argv[1], sys.argv[2], sys.argv[3:]
    for rel in rels:
        print(f"== {rel}")
        res = {}
        for b in (ba, bb):
            res[b] = analyse(f"{CB}/{b}/{rel}")
        for b in (ba, bb):
            a, per, reads = res[b]
            tot = collections.Counter()
            for c in per.values():
                tot.update(c)
            print(f"  {b:22s} loop {a['loop_insns']:4d}  stack L{tot['sL']} S{tot['sS']}  other L{tot['mL']} S{tot['mS']}"
                  f"  forward ready reads {len(reads)}: " + ", ".join(f"{hex(x)} {d} use+{u}" for x, d, u in reads))
        fa, fb = res[ba][1], res[bb][1]
        for f in sorted(set(fa) | set(fb), key=lambda f: -(fa[f]["n"] + fb[f]["n"])):
            if fa[f] != fb[f]:
                fmt = lambda c: f"{c['n']:3d} sL{c['sL']} sS{c['sS']} mL{c['mL']} mS{c['mS']}" if c["n"] else "-"
                print(f"    {f[:44]:44s} {fmt(fa[f]):>24s} -> {fmt(fb[f]):>24s}")


if __name__ == "__main__":
    main()
