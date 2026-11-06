#!/usr/bin/env python3
"""dio_sites.py PAIRS_JSON [KEY ...]: per E1 image pair (pair_images.py; a = noinline, b = inline), receiver::run()'s
instructions by site (addr2line inline chain), with loads, stack loads / stores (sp-relative), stores and NoC command
register writes; deliver_other's out-of-line body is counted under its own site in arm a. Also run()'s stack frame and
callee-saved saves, and a normalized multiset diff (registers and constants masked) of every site outside deliver_other."""
import collections, json, re, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation/reconcile_device/write_gap")
from annot import annotate, func_range
SITES = [("local_inline_write", "inline write (deliver_other)"), ("deliver_other", "deliver_other (dispatch, impl)"),
         ("execute_chip_unicast_to_local_chip_impl", "deliver_other (dispatch, impl)"),
         ("local_write<true>", "local write (preset)"), ("local_write<false>", "local write (whole)"),
         ("count_local_writes", "scatter (counts, chunks)"), ("shift_to_next_chunk", "scatter (counts, chunks)"),
         ("local_atomic_inc", "atomic"), ("forward", "forward"), ("decode_action", "route decode"),
         ("admit", "admission"), ("accumulate_completion_credit", "credit accumulate"),
         ("credit_completions", "completion proof / due"), ("send_credit_frame", "credit frame"),
         ("retire", "retire"), ("deliver", "deliver (type dispatch)"), ("receive", "receive (header)")]
REG = r"\b(zero|ra|sp|gp|tp|t[0-6]|s[0-9]|s1[01]|a[0-7])\b"
def norm(t): return re.sub(r"-?\b0x[0-9a-f]+\b|-?\b[0-9a-f]{3,}\b|-?\b\d+\b", "N", re.sub(REG, "R", t))
def site(ch):
    for f, _ in ch:
        for pat, name in SITES:
            if f.startswith(pat) or (pat == "forward" and "forward" in f):
                return name
    return "loop (other)"
def count(elf):
    rows = annotate(elf, r"receiver::run[<(]")[1]
    if func_range(elf, r"deliver_other"):
        rows = rows + [(a, t, ch + [("deliver_other", "")]) for a, t, ch in annotate(elf, r"deliver_other")[1]]
    c = collections.defaultdict(collections.Counter); ops = collections.defaultdict(collections.Counter)
    for a, t, ch in rows:
        s = site(ch); op = t.split()[0]; c[s]["i"] += 1; ops[s][norm(t)] += 1
        if op in ("lw", "lh", "lhu", "lb", "lbu"): c[s]["sp-ld" if "(sp)" in t else "ld"] += 1
        if op in ("sw", "sh", "sb"): c[s]["sp-st" if "(sp)" in t else "st"] += 1
        if any(f.startswith("NOC_CMD_BUF_WRITE_REG") for f, _ in ch): c[s]["cmd"] += 1
    frame = next((t for a, t, ch in rows[:3] if t.startswith("addi\tsp,sp,-")), "?")
    saves = sum(1 for a, t, ch in rows[:40] if t.startswith("sw\t") and re.match(r"sw\t(ra|s\d+),\d+\(sp\)", t))
    return c, ops, frame, saves, len(rows)
pairs = [p for p in json.load(open(sys.argv[1])) if p["proc"] == "subordinate_active_erisc"]
keys = sys.argv[2:] or [max(pairs, key=lambda p: p["size_a"])["key"]]
for k in keys:
    p = [p for p in pairs if p["key"] == k][0]
    (ca, oa, fa, sa, na), (cb, ob, fb, sb, nb) = count(p["a"]), count(p["b"])
    print(f"== E1 {k}: image {p['size_a']} -> {p['size_b']} B; run()+deliver_other {na} -> {nb} insns; "
          f"frame '{fa}' -> '{fb}', callee saves {sa} -> {sb}")
    f = lambda x: f"{x['i']:3d} i {x['ld']:2d} ld {x['sp-ld']:2d} sp-ld {x['st']:2d} st {x['sp-st']:2d} sp-st {x['cmd']:2d} cmd"
    for s in sorted(set(ca) | set(cb)):
        same = "" if s.startswith("deliver_other") or s.startswith("inline write") else \
            ("  same code (normalized)" if oa[s] == ob[s] else
             f"  differs: -{sum((oa[s] - ob[s]).values())} / +{sum((ob[s] - oa[s]).values())} normalized insns")
        print(f"   {s:32s} noinline {f(ca[s])} | inline {f(cb[s])}{same}")
