#!/usr/bin/env python3
"""size_report.py [--opt O3|O2|Os] [--cases C,...] [--detail] NAME [NAME ...]: code size and stack use of router builds.

Reads coverage builds COV/builds/<NAME>_<opt>/ (coverage.py build; the first NAME is the reference) and, per router ELF:
  code     loadable bytes (PT_LOAD file sizes: .text plus initialised data), what the kernel config buffer holds
  stack    deepest static stack use from the kernel's _start: frames from the sp adjustments of each function, the call
           graph from jal (calls add the caller's frame, tail jumps do not). An indirect call (jalr) on E0 enters the
           base firmware (context switch, link status, Ethernet messages), which goes at most BASE_FW_DEPTH below the
           caller's frame (RESULTS §18, stack paint); elsewhere it is flagged.
  free     stack available (stack pointer at kernel entry minus the end of the kernel's data and bss) minus stack use
Stack pointer at kernel entry (RESULTS §18, measured; E1 matches the firmware's __stack_top - _start - main frames):
  E0 (active_erisc, base firmware resident) 0xFFB01EF0, E1 (subordinate_active_erisc) 0xFFB01FC0.

Prints, per case (production cases by default) and ERISC, the largest code and the least free stack over the case's
configs, and E0 + E1 (largest of each, an upper bound for one router) against the 26,624 B kernel config buffer
(MEM_ERISC_KERNEL_CONFIG_SIZE, which also holds runtime args). Then the worst case per router over the cases every
named build has.
Writes COV/builds/<NAME>_<opt>/size_report.json. --detail prints the deepest call chain of each router's least-free
ELF per ERISC."""
import argparse
import collections
import concurrent.futures
import json
import os
import re
import subprocess

COV = os.environ.get("TT_COV_DIR", "/home/aagarwal/.tt-metal-v3r-builds/cov_m")
BIN = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
ENTRY_SP = {"active_erisc": 0xFFB01EF0, "subordinate_active_erisc": 0xFFB01FC0}
ERISC = {"active_erisc": "E0", "subordinate_active_erisc": "E1"}
BASE_FW_DEPTH = 68
KERNEL_CONFIG = 26 * 1024
LEAVES = {"memset", "memcpy", "memmove"}  # their jalr is a computed jump inside the function, no frame
# Study-only cases (instrumentation, stream credits, synthesized configs main's router cannot build): not counted by default
STUDY = re.compile(r"_(delay|passdelay|allinstr|codeprof|sc)$|^bh_2d_udm$|_datacache$|_spinwait$|_nofuse$")


def run(*a):
    return subprocess.run([BIN + a[0], *a[1:]], capture_output=True, text=True).stdout


def code_bytes(elf):
    return sum(int(m.group(1), 16) for m in re.finditer(r"^\s+LOAD\s+\S+\s+\S+\s+\S+\s+(0x[0-9a-f]+)",
                                                         run("readelf", "-lW", elf), re.M))


def data_end(elf):
    m = re.search(r"^([0-9a-f]+) \w __ldm_bss_end$", run("nm", elf), re.M)
    return int(m.group(1), 16)


def functions(elf):
    """{start address: (name, [(mnemonic, operands)])} from the disassembly."""
    f, cur = {}, None
    for line in run("objdump", "-d", "--no-show-raw-insn", elf).splitlines():
        m = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if m:
            if m.group(2).startswith(".L"):  # local label inside the current function
                continue
            cur = int(m.group(1), 16)
            f[cur] = (m.group(2), [])
            continue
        m = re.match(r"^\s+[0-9a-f]+:\s+(\S+)\s*(.*)$", line)
        if m and cur is not None:
            f[cur][1].append((m.group(1), m.group(2)))
    return f


def analyse(fn, ins):
    """frame (largest running sp depth), calls [(target, is_tail)], indirect call count, dynamic sp flag."""
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


def stack(elf, proc):
    fs = functions(elf)
    info = {a: (n, *analyse(n, ins)) for a, (n, ins) in fs.items()}
    memo, flags = {}, set()

    def deep(a, path):
        """(deepest use from function a's entry, chain of (name, frame))."""
        if a in memo:
            return memo[a]
        if a in path:
            flags.add("recursion")
            return 0, []
        n, frame, calls, indirect, dyn = info[a]
        if dyn:
            flags.add("dynamic sp")
        best, chain = frame, []
        if indirect:
            if proc == "active_erisc":
                best, chain = frame + BASE_FW_DEPTH, [("base firmware", BASE_FW_DEPTH)]
            else:
                flags.add("indirect call")
        for t, tail in calls:
            if t not in info or t == a:  # a jump to a local label
                continue
            d, c = deep(t, path | {a})
            d += 0 if tail else frame
            if d > best:
                best, chain = d, c
        memo[a] = (best, [(n, frame)] + chain)
        return memo[a]

    start = next(a for a, (n, _) in fs.items() if n == "_start")
    d, chain = deep(start, frozenset())
    return d, chain, sorted(flags)


def one(elf, proc):
    d, chain, flags = stack(elf, proc)
    avail = ENTRY_SP[proc] - data_end(elf)
    return {"code": code_bytes(elf), "stack": d, "avail": avail, "free": avail - d, "flags": flags,
            "chain": [f"{n} {f}" for n, f in chain]}


def load(name, opt, cases):
    d = os.path.join(COV, "builds", f"{name}_{opt}")
    if not os.path.isdir(d):
        return None
    elfs = []
    for key in sorted(os.listdir(d)):
        case = key.split("__")[0]
        if "__" not in key or (cases and case not in cases) or (not cases and STUDY.search(case)):
            continue
        for proc in ENTRY_SP:
            e = os.path.join(d, key, proc, f"{proc}.elf")
            if os.path.exists(e):
                elfs.append((key, proc, e))
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        res = dict(zip((f"{k}/{p}" for k, p, _ in elfs), ex.map(lambda x: one(x[2], x[1]), elfs)))
    json.dump(res, open(os.path.join(d, "size_report.json"), "w"), indent=0)
    return res


def per_case(res):
    """{case: {"E0"/"E1": (largest code, least free, key of the least free)}}"""
    out = collections.defaultdict(dict)
    for k, r in res.items():
        case, e = k.split("__")[0], ERISC[k.split("/")[1]]
        c, f, fk = out[case].get(e, (0, 1 << 30, None))
        out[case][e] = (max(c, r["code"]), min(f, r["free"]), k if r["free"] < f else fk)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--opt", default="O3")
    ap.add_argument("--cases", default="")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    cases = set(a.cases.split(",")) if a.cases else set()
    data = {n: load(n, a.opt, cases) for n in a.names}
    names = [n for n in a.names if data[n]]
    for n in a.names:
        if not data[n]:
            print(f"(no build {n}_{a.opt})")
    pc = {n: per_case(data[n]) for n in names}
    every = sorted(set().union(*(pc[n] for n in names)))
    print(f"## -{a.opt}: code bytes (largest per ERISC; E0+E1 vs {KERNEL_CONFIG}) and least free stack bytes")
    print(f"{'case':24}" + "".join(f" | {n:>8} E0  E1  E0+E1  free E0  E1" for n in names))
    for case in every:
        row = f"{case:24}"
        for n in names:
            c = pc[n].get(case, {})
            e0, e1 = c.get("E0"), c.get("E1")
            tot = (e0[0] if e0 else 0) + (e1[0] if e1 else 0)
            fmt = lambda x, i: f"{x[i]:6d}" if x else "     -"
            row += f" | {fmt(e0, 0)} {fmt(e1, 0)} {tot:6d}  {fmt(e0, 1)} {fmt(e1, 1)}"
        print(row)
    common = set.intersection(*(set(pc[n]) for n in names))
    print(f"\n## -{a.opt}: worst over the {len(common)} cases every build has"
          + (f" (left out: {', '.join(sorted(set(every) - common))})" if set(every) - common else ""))
    for n in names:
        rs = {k: r for k, r in data[n].items() if k.split("__")[0] in common}
        pc[n] = {c: v for c, v in pc[n].items() if c in common}
        e = {x: [r for k, r in rs.items() if ERISC[k.split("/")[1]] == x] for x in ("E0", "E1")}
        tot = max(sum(v[0] for v in c.values()) for c in pc[n].values())
        totc = max(pc[n], key=lambda c: sum(v[0] for v in pc[n][c].values()))
        print(f"{n:10} ELFs {len(rs):5}  largest code E0 {max((r['code'] for r in e['E0']), default=0):6} "
              f"E1 {max((r['code'] for r in e['E1']), default=0):6}  E0+E1 {tot} ({totc}, "
              f"{KERNEL_CONFIG - tot:+d} left)  least free E0 {min((r['free'] for r in e['E0']), default=0)} "
              f"E1 {min((r['free'] for r in e['E1']), default=0)}  deepest E0 "
              f"{max((r['stack'] for r in e['E0']), default=0)} E1 {max((r['stack'] for r in e['E1']), default=0)}")
        flags = collections.Counter(f for r in rs.values() for f in r["flags"])
        if flags:
            print(f"{'':10} flags: {dict(flags)}")
        if a.detail:
            for x in ("E0", "E1"):
                ks = [k for k in rs if ERISC[k.split("/")[1]] == x]
                if ks:
                    k = min(ks, key=lambda k: rs[k]["free"])
                    r = rs[k]
                    print(f"{'':10} {x} least free {k}: avail {r['avail']}, stack {r['stack']}: "
                          + " > ".join(r["chain"]))


if __name__ == "__main__":
    main()
