#!/usr/bin/env python3
"""rc_diff.py [--rc DIR] [CONFIG ...]: what the study and port hosts pass the router, per mock config
(rc_capture.sh's captures in DIR/study/<config> and DIR/port/<config>).

Per side and config, from every router build in the run's kernel cache (study: fabric_erisc_router_v2, the v2 slot the
lead is swapped into; port: fabric_erisc_router): the named compile-time args (named_ct_arg_map_generated.h), the -D
defines and -O level of its compile command (the run log's compile commands, TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1),
the positional compile-time arg count; and the slot rows the host logged (TT_FABRIC_EXP_LOG_SLOTS=1).
Builds are grouped by processor and MY_DIRECTION (E, W, N, S, Z). Per group: the named args whose value sets differ,
and the args only one side passes; then the defines and opt levels. Prints markdown.
"""
import argparse
import collections
import glob
import os
import re

CLEAN = re.compile(r"\x1b\[[0-9;]*m")
DIRS = {0: "E", 1: "W", 2: "N", 3: "S", 4: "Z"}


def builds(run_dir, kernel):
    out = []
    logs = glob.glob(os.path.join(run_dir, "*", "*.log"))
    cmds = {}
    for log in logs:
        for line in open(log, errors="replace"):
            line = CLEAN.sub("", line)
            m = re.search(r"g\+\+ compile cmd: (.*)$", line)
            if m and f"/kernels/{kernel}/" in line:
                d = re.search(rf"(\S*/kernels/{kernel}/\d+)/(\w+)/", m.group(1))
                if d:
                    cmds[(os.path.basename(d.group(1)), d.group(2))] = m.group(1)
    for hdr in glob.glob(os.path.join(run_dir, "*", "cache_*", "*", "kernels", kernel, "*", "named_ct_arg_map_generated.h")):
        gen = os.path.dirname(hdr)
        named = {k: int(v) for k, v in re.findall(r'\{"([A-Z0-9_a-z]+)",(\d+)\}', open(hdr).read())}
        for proc in sorted(p for p in os.listdir(gen) if os.path.isdir(os.path.join(gen, p))):
            cmd = cmds.get((os.path.basename(gen), proc), "")
            defines = sorted(t for t in cmd.split() if t.startswith("-D") and "/" not in t and
                             not t.startswith("-DKERNEL_COMPILE_TIME_ARGS=") and not t.startswith("-DKERNEL_COMPILE_TIME_ARG_MAP"))
            opt = sorted({t for t in cmd.split() if re.fullmatch(r"-O[0-3sz]|-Ofast", t)})
            ct = re.search(r"-DKERNEL_COMPILE_TIME_ARGS=([0-9,]+)", cmd)
            ctv = tuple(int(x) for x in ct.group(1).split(",")) if ct else ()
            out.append({"proc": proc, "dir": DIRS.get(named.get("MY_DIRECTION", -1), "?"), "named": named,
                        "defines": defines, "opt": opt, "nct": len(ctv) if ct else None, "ct": ctv})
    return out


def slot_rows(run_dir):
    rows = collections.Counter()
    for log in glob.glob(os.path.join(run_dir, "*", "*.log")):
        for line in open(log, errors="replace"):
            m = re.search(r"(VC\d sender slots \{[^}]*\} receiver slots \{[^}]*\} \(channel buffer \d+ B\))", CLEAN.sub("", line))
            if m:
                rows[m.group(1)] += 1
    return rows


def vals(v):
    v = sorted(v)
    return ",".join(map(str, v)) if len(v) <= 6 else f"{','.join(map(str, v[:5]))},.. ({len(v)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc", default="/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/rc")
    ap.add_argument("configs", nargs="*")
    a = ap.parse_args()
    configs = a.configs or sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.rc, "study", "*")) if os.path.isdir(p))
    for cfg in configs:
        s = builds(os.path.join(a.rc, "study", cfg), "fabric_erisc_router_v2")
        p = builds(os.path.join(a.rc, "port", cfg), "fabric_erisc_router")
        print(f"## {cfg}: {len(s)} study router builds, {len(p)} port\n")
        if not s or not p:
            print("(no capture on one side)\n")
            continue
        rs, rp = slot_rows(os.path.join(a.rc, "study", cfg)), slot_rows(os.path.join(a.rc, "port", cfg))
        print("slot rows (count of allocator logs): study " + "; ".join(f"{k} x{v}" for k, v in sorted(rs.items())))
        print("\nport " + "; ".join(f"{k} x{v}" for k, v in sorted(rp.items())) + "\n")
        for side, b in (("study", s), ("port", p)):
            print(f"{side}: opt {sorted({o for x in b for o in x['opt']})}, positional CT args {sorted({x['nct'] for x in b})}")
        ds = collections.Counter(d for x in s for d in x["defines"])
        dp = collections.Counter(d for x in p for d in x["defines"])
        only_s = sorted(set(ds) - set(dp))
        only_p = sorted(set(dp) - set(ds))
        print(f"\ndefines only on the study side: {only_s or 'none'}; only on the port side: {only_p or 'none'}\n")
        groups = sorted({(x["proc"], x["dir"]) for x in s} | {(x["proc"], x["dir"]) for x in p})
        names_s = set().union(*(x["named"] for x in s))
        names_p = set().union(*(x["named"] for x in p))
        def only(names, b):
            return ", ".join(f"{n}={vals({x['named'][n] for x in b if n in x['named']})}" for n in sorted(names)) or "none"
        print(f"named args only the study passes ({len(names_s - names_p)}): {only(names_s - names_p, s)}\n")
        print(f"named args only the port passes ({len(names_p - names_s)}): {only(names_p - names_s, p)}\n")
        print("| processor | dir | builds (study / port) | named arg | study values | port values |")
        print("| --- | --- | --- | --- | --- | --- |")
        for proc, d in groups:
            gs = [x for x in s if x["proc"] == proc and x["dir"] == d]
            gp = [x for x in p if x["proc"] == proc and x["dir"] == d]
            for name in sorted((names_s & names_p)):
                vs = {x["named"].get(name) for x in gs}
                vp = {x["named"].get(name) for x in gp}
                if vs != vp and gs and gp:
                    print(f"| {proc} | {d} | {len(gs)} / {len(gp)} | {name} | {vals(v for v in vs if v is not None)} | "
                          f"{vals(v for v in vp if v is not None)} |")
            if not gs or not gp:
                print(f"| {proc} | {d} | {len(gs)} / {len(gp)} | (group only on one side) | | |")
                continue
            # positional compile-time args (the channel allocation entries among them): positions whose value sets differ
            n = max(len(x["ct"]) for x in gs + gp)
            for i in range(n):
                vs = {x["ct"][i] for x in gs if i < len(x["ct"])}
                vp = {x["ct"][i] for x in gp if i < len(x["ct"])}
                if vs != vp:
                    print(f"| {proc} | {d} | {len(gs)} / {len(gp)} | CT[{i}] | {vals(vs)} | {vals(vp)} |")
        print()


if __name__ == "__main__":
    main()
