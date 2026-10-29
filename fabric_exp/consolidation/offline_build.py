#!/usr/bin/env python3
"""Offline router builds: rebuild the fabric router ELFs for any source variant without the device.

  collect                      read the JIT's logged compile/link commands (TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1)
                               from the Phase 0 and VC1 logs into offline/cmds.json, keyed by opt level, config hash,
                               ERISC (VC1 configs: vc1_runs.sh; their hashes list in offline/vc1_hashes.txt)
  build --variant DIR --out DIR [--opt O3] [--hashes H,H|all] [--flags "-f..."] [--jobs N]
                               overlay DIR's files on a copy of tt_metal/fabric and rebuild every logged config at
                               that opt level (--set-opt rebuilds the --opt configs at another -O level)

A variant DIR holds replacement files: either mirrored tree paths (DIR/tt_metal/fabric/...) or bare file names, which
are matched by name under tt_metal/fabric (they must be unique there). New files, and files under tests/, need mirrored paths. A bare
fabric_erisc_router_v3.cpp replaces fabric_erisc_router_v2.cpp (Phase 1 runs v3 candidates in v2's place). The generated per-config headers
(named_ct_arg_map_generated.h etc.) and the firmware link objects come from the Phase 0 kernel cache
(/home/aagarwal/.tt-metal-cache-p0), which must stay in place. OFFLINE_SFPI=<sfpi root> swaps the toolchain.
"""
import argparse, concurrent.futures, glob, json, os, re, shlex, shutil, subprocess, sys

ROOT = "/home/aagarwal/tt-metal"
HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "offline", "cmds.json")
KDIR = "kernels/fabric_erisc_router_v2/"
EDM = ROOT + "/tt_metal/fabric/impl/kernels/edm_fabric"
V2_PATH = "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp"
# Overlay base: a snapshot of tt_metal/fabric taken with the tree at s79mc (2026-10-24), so offline builds do not see
# files a hardware job has swapped into the live tree. Variants overlay it; everything else comes from the tree.
# OFFLINE_BASE picks another snapshot (the merged tree: /home/aagarwal/.tt-metal-p1-base-m, 2026-10-28).
BASE = os.environ.get("OFFLINE_BASE", "/home/aagarwal/.tt-metal-p1-base")


def wellformed(kind, cmd):
    """A logged command that was not cut by another process's output."""
    if "<stdout>" in cmd or "<stderr>" in cmd or "\x1b" in cmd:
        return False
    a = cmd.split(" ")
    hashes = set(re.findall(KDIR + r"(\d+)/", cmd))
    if len(hashes) != 1 or cmd.count("riscv-tt-elf-g++") != 1:
        return False
    if kind == "compile":
        return (a[0].endswith("riscv-tt-elf-g++") and a.count("-c") == 1 and a.count("-MF") == 1
                and sum(x.startswith("-DFULL_KERNEL_NAME=") for x in a) == 1 and a[-1].endswith(".d")
                and "-I" + ROOT + "/tt_metal/hw/inc/internal/tt-1xx/blackhole" in a)
    return cmd.startswith("cd ") and cmd.count(" -o ") == 1 and cmd.rstrip().endswith(".elf")


def collect(args):
    db = {}
    logs = sorted(glob.glob(os.path.join(ROOT, "fabric_exp/results/p0o*_13129088_*.log")))
    # VC1 split (vc1_runs.sh, vc1_lib.sh). Two MPI ranks write to one log, so a line can be cut by the other rank's
    # output; only well-formed commands are kept (see `wellformed`), the first one per kernel.
    for pat in ("vc1*_13129088_vc1.log", "p*vc1*_13129088_vc1.log"):
        logs += sorted(glob.glob(os.path.join(ROOT, "fabric_exp/results", pat)))
    for log in logs:
        for line in open(log, errors="replace"):
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)  # VC1 logs: colour codes and MPI rank tags around each line
            m = re.search(r"g\+\+ (compile|link) cmd: (.*?)\s*\(build\.cpp:\d+\)\s*$", line)
            if not m or KDIR not in m.group(2):
                continue
            kind, cmd = m.group(1), m.group(2)
            h = re.search(KDIR + r"(\d+)/(\w+)/", cmd)
            opt = re.search(r"\s-(O\w+)\s", cmd).group(1)
            if not wellformed(kind, cmd):
                continue
            e = db.setdefault(opt, {}).setdefault(h.group(1), {}).setdefault(h.group(2), {})
            e.setdefault(kind, cmd)
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    json.dump(db, open(DB, "w"), indent=0)
    vc1 = set()
    for log in logs:
        if "/vc1" in log:
            for line in open(log, errors="replace"):
                if "g++ compile cmd" in line and KDIR in line:
                    vc1.add(re.search(KDIR + r"(\d+)/", line).group(1))
    open(os.path.join(os.path.dirname(DB), "vc1_hashes.txt"), "w").write("\n".join(sorted(vc1)) + "\n")
    for opt, hs in db.items():
        n = sum(1 for h in hs.values() for p in h.values() if "compile" in p and "link" in p)
        print(f"{opt}: {len(hs)} configs, {n} complete (config, ERISC) command pairs")


def resolve(variant):
    """[(source file, tree-relative destination)] for a variant dir (see the module docstring)."""
    by_name = {}
    for dp, _, fs in os.walk(ROOT + "/tt_metal/fabric"):
        for f in fs:
            by_name.setdefault(f, []).append(os.path.relpath(os.path.join(dp, f), ROOT))
    out = []
    for dp, _, fs in os.walk(variant):
        if "/elf" in dp[len(variant):]:
            continue
        for f in fs:
            src = os.path.join(dp, f)
            rel = os.path.relpath(src, variant)
            if f in ("fabric_erisc_router_v3.cpp", "fabric_erisc_router_v4.cpp") and rel == f:
                dst = V2_PATH  # Phase 1: a v3 candidate is built and run in v2's place (same host args, selector "2")
            elif rel.startswith(("tt_metal/", "tests/")):
                dst = rel
            elif f.endswith((".cpp", ".hpp", ".h")):
                cands = by_name.get(f, [])
                if len(cands) != 1:
                    sys.exit(f"variant file {rel}: {len(cands)} matches under tt_metal/fabric")
                dst = cands[0]
            else:
                continue
            out.append((src, dst))
    return out


def make_overlay(variant, out):
    ov = os.path.join(out, "overlay")
    if os.path.exists(ov):
        shutil.rmtree(ov)
    shutil.copytree(BASE + "/tt_metal/fabric", ov + "/tt_metal/fabric", symlinks=True)
    for src, dst in resolve(variant):
        os.makedirs(os.path.dirname(os.path.join(ov, dst)), exist_ok=True)
        shutil.copy(src, os.path.join(ov, dst))
    return ov


def prep_headers(h, cmd, ov, out):
    cache_hash_dir = re.search(r"(\S+/" + KDIR + h + ")/", cmd).group(1)
    hd = os.path.join(out, h)
    os.makedirs(hd, exist_ok=True)
    for f in glob.glob(cache_hash_dir + "/*.h") + glob.glob(cache_hash_dir + "/*.hpp"):
        shutil.copy(f, hd)
    ki = os.path.join(hd, "kernel_includes.hpp")
    s = open(ki).read().replace(EDM, ov + "/tt_metal/fabric/impl/kernels/edm_fabric")
    open(ki, "w").write(s)
    add_host_setting_args(hd)


def add_host_setting_args(hd):
    """Named args the host gained after the configs were logged, with the values today's host would pass.
    ETH_TXQ_SEQ_UPDATE_TIMEOUT (the keepalive, RESULTS §95): the study's host override passes
    TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT through, and that is the logged EXP_SEQ_UPDATE_TIMEOUT."""
    m = os.path.join(hd, "named_ct_arg_map_generated.h")
    t = open(m).read()
    if '{"ETH_TXQ_SEQ_UPDATE_TIMEOUT",' not in t:
        v = re.search(r'\{"EXP_SEQ_UPDATE_TIMEOUT",(\d+)\}', t)
        t = t.replace("#define KERNEL_COMPILE_TIME_ARG_MAP ",
                      '#define KERNEL_COMPILE_TIME_ARG_MAP {"ETH_TXQ_SEQ_UPDATE_TIMEOUT",%s},' % (v.group(1) if v else 0), 1)
        open(m, "w").write(t)


# OFFLINE_SFPI=<sfpi root> builds with another toolchain (compiler, its headers and libraries) than the one the
# commands were logged with, e.g. a newer sfpi release unpacked outside the tree, to compare its code generation.
# The logged root differs by source (the tree, or coverage.py's runtime root of symlinks), so any .../runtime/sfpi.
LOGGED_SFPI = re.compile(r"[^\s'\"]*/runtime/sfpi(?=/)")
OFFLINE_SFPI = os.environ.get("OFFLINE_SFPI")


def build_one(h, proc, cmds, ov, out, flags, set_opt):
    if OFFLINE_SFPI:
        cmds = {k: LOGGED_SFPI.sub(OFFLINE_SFPI, v) for k, v in cmds.items()}
    hd = os.path.join(out, h)
    wd = os.path.join(hd, proc)
    os.makedirs(wd, exist_ok=True)
    obj, elf = os.path.join(wd, "k.o"), os.path.join(wd, proc + ".elf")

    a = cmds["compile"].split(" ")
    o = []
    i = 0
    while i < len(a):
        x = a[i]
        if x == "-o":
            o += ["-o", obj]; i += 2; continue
        if x == "-MF":
            o += ["-MF", os.path.join(wd, "k.d")]; i += 2; continue
        if x == "-I..":
            o += [x, "-I" + ov, "-I" + ov + "/tt_metal"]; i += 1; continue
        if x == "-I" + EDM:
            o.append("-I" + ov + "/tt_metal/fabric/impl/kernels/edm_fabric"); i += 1; continue
        # A variant that overlays files outside tt_metal/fabric (e.g. NoC API headers under tt_metal/hw/inc): its copy
        # of the include directory goes first; files it does not overlay still come from the tree's directory.
        m = re.fullmatch(r"-I(?:\S*/)?(tt_metal/(?!fabric)\S*?)/?", x)
        if m and os.path.isdir(os.path.join(ov, m.group(1))):
            o += ["-I" + os.path.join(ov, m.group(1)), x]; i += 1; continue
        if set_opt and re.fullmatch(r"-O\w+", x):
            o.append("-" + set_opt); i += 1; continue
        if x == "-c":
            o += flags
        o.append(x); i += 1
    r = subprocess.run(o, cwd=wd, capture_output=True, text=True)
    if r.returncode:
        return h, proc, "compile failed:\n" + r.stderr[-3000:]

    link = cmds["link"]
    lo = re.search(r"(\S+\.o)\s+-o\s+(\S+\.elf)", link)
    link = link.replace(lo.group(1), obj).replace(lo.group(2), elf)
    link = re.sub(r"^cd \S+ &&", f"cd {shlex.quote(wd)} &&", link)
    if set_opt:
        link = re.sub(r"(\s)-O\w+(\s)", r"\1-" + set_opt + r"\2", link, count=1)
    if flags:
        link = link.replace(" -o " + elf, " " + " ".join(flags) + " -o " + elf)
    r = subprocess.run(link, shell=True, capture_output=True, text=True)
    if r.returncode:
        return h, proc, "link failed:\n" + r.stderr[-3000:]
    return h, proc, None


def build(args):
    db = json.load(open(DB))[args.opt]
    hashes = list(db) if args.hashes == "all" else args.hashes.split(",")
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    ov = make_overlay(os.path.abspath(args.variant), out)
    flags = args.flags.split() if args.flags else []
    jobs = [(h, p, c) for h in hashes for p, c in db[h].items() if "compile" in c and "link" in c]
    for h in hashes:
        prep_headers(h, next(iter(db[h].values()))["compile"], ov, out)
    bad = 0
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        for h, p, err in ex.map(lambda j: build_one(j[0], j[1], j[2], ov, out, flags, args.set_opt), jobs):
            if err:
                bad += 1
                print(f"{h}/{p}: {err}")
    print(f"built {len(jobs) - bad}/{len(jobs)} ELFs into {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("collect")
    b = sp.add_parser("build")
    b.add_argument("--variant", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--opt", default="O3")
    b.add_argument("--set-opt", default=None)
    b.add_argument("--hashes", default="all")
    b.add_argument("--flags", default="")
    b.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    sys.exit(collect(a) if a.cmd == "collect" else build(a))
