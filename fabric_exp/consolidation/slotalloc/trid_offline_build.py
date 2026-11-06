#!/usr/bin/env python3
"""trid_offline_build.py LOG... --tree TREE --out DIR [--rule new|uniform]: rebuild captured router configs offline
with the per-channel transaction-ID named args added (TRID_ALLOCATION.md), before a host emits them.

LOG: a mock or device run's log with TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 (BuildKernels "g++ compile cmd" /
"g++ link cmd" lines). Each fabric_erisc_router_v2 config's generated directory is copied to DIR, its
named_ct_arg_map_generated.h gets RX_CH_<c>_COMPLETION_GROUP_SIZE / _NUM_TRIDS / _TRID_START (the rule applied to the
config's own receiver slots, from its channel-allocation CT args, and its ENABLE_FIRST_LEVEL_ACK_VC<n> /
ENABLE_DEADLOCK_AVOIDANCE args), and the compile and link commands run again with the source tree replaced by TREE and
the output paths by DIR. Prints per config the receivers, the IDs and the result; then code size and free stack per
ERISC (size_report.py's measures). CPU only.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import size_report as sr  # noqa: E402

TAG = 0xABCD1234


def group_size(slots, fla, group=0):
    if fla:
        return 1
    if group:
        return group  # TT_FABRIC_EXP_COMPLETION_GROUP (SLOT_ALLOCATOR.md section 12.10)
    return 1 if slots < 8 else (2 if slots < 16 else 4)


def assign(rx, rule, da, group=0):
    """rx: [(slots, fla)] -> [(G, ids, start)]: the host's rule (fabric_trid_allocator.cpp), new or uniform; group:
    the completion-group override of every receiver without first-level acks (0: from the slot count)."""
    gs = [group_size(s, f, group) for s, f in rx]
    groups = [(s + g - 1) // g for (s, _), g in zip(rx, gs)]
    if rule == "uniform":
        n = 16 if (len(rx) == 1 and not da) else (8 if da else 4)
        ids = [n] * len(rx)
    else:
        ids = [max(1, g) for g in groups]
        while sum(ids) > 16:
            i = max(range(len(ids)), key=lambda k: (ids[k], -k))
            n = ids[i]
            if n & (n - 1) == 0:
                ids[i] = n // 2  # a power of 2 halves
            else:
                p = 1
                while p * 2 <= n:
                    p *= 2
                ids[i] = p  # else round down to a power of 2 first (20 -> 16)
    out, start = [], 0
    for g, n in zip(gs, ids):
        out.append((g, n, start))
        start += n
    return out


def parse_log(path):
    """{generated dir: [(compile cmd, link cmd, proc)]} for fabric_erisc_router_v2 configs."""
    clean = re.compile(r"\x1b\[[0-9;]*m")
    compiles, links = {}, {}
    for line in open(path, errors="replace"):
        line = clean.sub("", line)
        m = re.search(r"g\+\+ (compile|link) cmd: (.*)$", line)
        if not m or "fabric_erisc_router_v2/" not in line:
            continue
        cmd = re.sub(r"\s+\(\w+\.cpp:\d+\)$", "", m.group(2).strip())  # the logger's source location
        d = re.search(r"(\S*/kernels/fabric_erisc_router_v2/\d+)/(\w+)/", cmd)
        if not d:
            continue
        key = (d.group(1), d.group(2))
        (compiles if m.group(1) == "compile" else links)[key] = cmd
    out = {}
    for key, c in compiles.items():
        if key in links:
            out.setdefault(key[0], []).append((c, links[key], key[1]))
    return out


def named_args(header):
    return {k: int(v) for k, v in re.findall(r'\{"([A-Z0-9_]+)",(\d+)\}', open(header).read())}


def receivers(gen_dir, compile_cmd, named):
    ct = [int(x) for x in re.search(r"-DKERNEL_COMPILE_TIME_ARGS=([0-9,]+)", compile_cmd).group(1).split(",")]
    i = ct.index(TAG)
    n_entries = ct[i + 1]
    entries = [ct[i + 2 + 4 * k : i + 6 + 4 * k] for k in range(n_entries)]
    n_rx = named["NUM_RECEIVER_CHANNELS"]
    slots = [e[1] for e in entries[n_entries - n_rx :]]
    fla = [bool(named.get(f"ENABLE_FIRST_LEVEL_ACK_VC{c}", 0)) for c in range(n_rx)]
    return list(zip(slots, fla)), bool(named.get("ENABLE_DEADLOCK_AVOIDANCE", 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", default="new", choices=["new", "uniform"])
    a = ap.parse_args()
    configs = {}
    for log in a.logs:
        configs.update(parse_log(log))
    os.makedirs(a.out, exist_ok=True)
    sizes, failed = [], 0
    for gen, procs in sorted(configs.items()):
        named = named_args(os.path.join(gen, "named_ct_arg_map_generated.h"))
        rx, da = receivers(gen, procs[0][0], named)
        ids = assign(rx, a.rule, da)
        dst = os.path.join(a.out, os.path.basename(gen))
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(gen, dst, ignore=shutil.ignore_patterns("*.o", "*.elf", "*.d"))
        add = {}
        for c in range(3):
            g, n, s = ids[c] if c < len(ids) else (1, 0, 0)
            add[f"RX_CH_{c}_COMPLETION_GROUP_SIZE"], add[f"RX_CH_{c}_NUM_TRIDS"], add[f"RX_CH_{c}_TRID_START"] = g, n, s
        h = os.path.join(dst, "named_ct_arg_map_generated.h")
        pairs = {**named, **add}
        body = ",".join('{"%s",%d}' % (k, pairs[k]) for k in sorted(pairs))
        text = open(h).read()
        text = re.sub(r"#define KERNEL_COMPILE_TIME_ARG_MAP .*", "#define KERNEL_COMPILE_TIME_ARG_MAP " + body, text)
        open(h, "w").write(text)
        root = re.search(r"(\S+)/runtime/sfpi/compiler/bin/riscv-tt-elf-g\+\+", procs[0][0]).group(1)
        result = []
        for compile_cmd, link_cmd, proc in procs:
            os.makedirs(os.path.join(dst, proc), exist_ok=True)
            cc = compile_cmd.replace(gen, dst).replace(root, a.tree)
            lc = link_cmd.replace(gen, dst).replace(root, a.tree)
            # The JIT compiles from the ERISC's output directory: -I. -I.. and -include named_ct_arg_map_generated.h.
            r = subprocess.run(cc, shell=True, capture_output=True, text=True, cwd=os.path.join(dst, proc))
            if r.returncode == 0:
                r = subprocess.run(lc, shell=True, capture_output=True, text=True)
            if r.returncode != 0:
                failed += 1
                err = [l for l in (r.stderr + r.stdout).splitlines() if "error" in l][:3]
                result.append(f"{proc} FAILED: " + " | ".join(err))
                continue
            elf = re.search(r"\s-o\s+(\S+\.elf)", lc).group(1)  # the JIT links to a temporary name
            m = sr.one(elf, proc)
            sizes.append((sr.ERISC[proc], m))
            result.append(f"{sr.ERISC[proc]} code {m['code']} free {m['free']}")
        rx_text = " | ".join(f"{s}{'a' if f else ''}" for s, f in rx)
        id_text = " | ".join(f"{g}/{n}@{s}" for g, n, s in ids)
        print(f"{os.path.basename(gen)}: rx {rx_text} da {int(da)} -> {id_text}: {'; '.join(result)}")
    for e in ("E0", "E1"):
        ms = [m for k, m in sizes if k == e]
        if ms:
            print(f"{e}: {len(ms)} ELFs, largest code {max(m['code'] for m in ms)}, least free {min(m['free'] for m in ms)}")
    print("FAILED" if failed else "ALL_BUILT")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
