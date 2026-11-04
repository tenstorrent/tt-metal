#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Router bring-up tools (temporary: drop before merge; see README.md).

offline_build.py: rebuild the fabric router's ERISC ELFs from the JIT's logged commands, without a device, for any
source variant (a changed file, a compiler flag, another -O level). Use it to see what a source change does to the
generated code before running it, or to bisect a code-shape regression.

  collect --out cmds.json [--kernel fabric_erisc_router] LOG [LOG ...]
      Read the compile and link commands the JIT logged (run the test with TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1;
      the lines are "g++ compile cmd: ..." / "g++ link cmd: ...") for KERNEL's builds, keyed by -O level, config hash
      and ERISC image (Wormhole: erisc; Blackhole: active_erisc / subordinate_active_erisc). Keeps the first
      well-formed pair of each.
  build --cmds cmds.json --variant DIR --out DIR [--opt O3] [--set-opt O2] [--hashes H,H|all] [--flags "-f..."]
        [--root TREE] [--toolchain-root SFPI] [--jobs N]
      Overlay DIR's files on a copy of TREE/tt_metal/fabric (TREE: the tree the commands were logged from, or --root)
      and rebuild every logged config at that -O level into OUT/<hash>/<image>/<image>.elf.
      DIR holds replacement files, either at mirrored tree paths (DIR/tt_metal/..., DIR/tests/...) or as bare file names
      matched by name under tt_metal/fabric (they must be unique there). An empty DIR rebuilds the tree as logged.
      --set-opt rebuilds the --opt configs at another -O level; --flags adds compiler flags; --toolchain-root swaps
      the sfpi toolchain (another release, to compare its code generation).

Inputs that must still exist: the kernel cache the commands point into (its per-config generated headers, e.g.
named_ct_arg_map_generated.h, and the firmware objects the link uses) and the logged tree. Outputs: ELFs only; compare
them with elf_size.py, codegen_diff.py, line_compare.py, loopvars.py.
"""
import argparse
import concurrent.futures
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

LOG_LINE = re.compile(r"g\+\+ (compile|link) cmd: (.*?)\s*(?:\(build\.cpp:\d+\))?\s*$")
EDM_REL = "tt_metal/fabric/impl/kernels/edm_fabric"


def kdir(kernel):
    return f"kernels/{kernel}/"


def wellformed(kind, cmd, kernel):
    """A logged command that was not cut by another process's output (multi-process runs share one log)."""
    if "<stdout>" in cmd or "<stderr>" in cmd or "\x1b" in cmd:
        return False
    if len(set(re.findall(kdir(kernel) + r"(\d+)/", cmd))) != 1 or len(re.findall(r"\S*g\+\+\s", cmd + " ")) != 1:
        return False
    a = cmd.split(" ")
    if kind == "compile":
        return (
            re.search(r"g\+\+$", a[0]) is not None
            and a.count("-c") == 1
            and a.count("-MF") == 1
            and sum(x.startswith("-DFULL_KERNEL_NAME=") for x in a) == 1
            and a[-1].endswith(".d")
        )
    return cmd.startswith("cd ") and cmd.count(" -o ") == 1 and cmd.rstrip().endswith(".elf")


def tree_of(cmd):
    """The tree a compile command was logged from (its -I<tree>/tt_metal/... include paths)."""
    m = re.search(r"-I(\S+?)/tt_metal/hw/inc(?:\s|/)", cmd)
    return m.group(1) if m else None


def collect(a):
    db = {}
    for log in a.logs:
        for line in open(log, errors="replace"):
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = LOG_LINE.search(line)
            if not m or kdir(a.kernel) not in m.group(2):
                continue
            kind, cmd = m.group(1), m.group(2)
            if not wellformed(kind, cmd, a.kernel):
                continue
            h = re.search(kdir(a.kernel) + r"(\d+)/(\w+)/", cmd)
            opt = re.search(r"\s-(O\w+)\s", cmd + " ")
            if not h or not opt:
                continue
            e = db.setdefault(opt.group(1), {}).setdefault(h.group(1), {}).setdefault(h.group(2), {})
            e.setdefault(kind, cmd)
    json.dump({"kernel": a.kernel, "configs": db}, open(a.out, "w"), indent=0)
    for opt, hs in db.items():
        n = sum(1 for h in hs.values() for p in h.values() if "compile" in p and "link" in p)
        print(f"{opt}: {len(hs)} configs, {n} complete (config, image) command pairs")
    if not db:
        print(f"no {a.kernel} commands found: were the logs written with TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1?")
        return 1
    return 0


def resolve(variant, root):
    """[(source file, tree-relative destination)] for a variant dir."""
    by_name = {}
    for dp, _, fs in os.walk(os.path.join(root, "tt_metal/fabric")):
        for f in fs:
            by_name.setdefault(f, []).append(os.path.relpath(os.path.join(dp, f), root))
    out = []
    for dp, _, fs in os.walk(variant):
        for f in fs:
            src = os.path.join(dp, f)
            rel = os.path.relpath(src, variant)
            if rel.startswith(("tt_metal/", "tests/")):
                dst = rel
            elif f.endswith((".cpp", ".hpp", ".h")):
                cands = by_name.get(f, [])
                if len(cands) != 1:
                    sys.exit(f"variant file {rel}: {len(cands)} matches under tt_metal/fabric (use a mirrored path)")
                dst = cands[0]
            else:
                continue  # notes, READMEs
            out.append((src, dst))
    return out


def make_overlay(variant, root, out):
    ov = os.path.join(out, "overlay")
    if os.path.exists(ov):
        shutil.rmtree(ov)
    shutil.copytree(os.path.join(root, "tt_metal/fabric"), os.path.join(ov, "tt_metal/fabric"), symlinks=True)
    for src, dst in resolve(variant, root):
        os.makedirs(os.path.dirname(os.path.join(ov, dst)), exist_ok=True)
        shutil.copy(src, os.path.join(ov, dst))
    return ov


def prep_headers(h, cmd, kernel, root, ov, out):
    """The config's generated headers, from the kernel cache, with the router source pointed at the overlay."""
    cache_hash_dir = re.search(r"(\S+/" + kdir(kernel) + h + ")/", cmd).group(1)
    hd = os.path.join(out, h)
    os.makedirs(hd, exist_ok=True)
    for f in glob.glob(cache_hash_dir + "/*.h") + glob.glob(cache_hash_dir + "/*.hpp"):
        shutil.copy(f, hd)
    ki = os.path.join(hd, "kernel_includes.hpp")
    if os.path.exists(ki):
        s = open(ki).read().replace(os.path.join(root, EDM_REL), os.path.join(ov, EDM_REL))
        open(ki, "w").write(s)


LOGGED_SFPI = re.compile(r"[^\s'\"]*/(?:runtime/sfpi|tenstorrent/sfpi)(?=/)")


def build_one(h, proc, cmds, root, ov, out, flags, set_opt, sfpi):
    if sfpi:
        cmds = {k: LOGGED_SFPI.sub(sfpi, v) for k, v in cmds.items()}
    wd = os.path.join(out, h, proc)
    os.makedirs(wd, exist_ok=True)
    obj, elf = os.path.join(wd, "k.o"), os.path.join(wd, proc + ".elf")
    edm = os.path.join(root, EDM_REL)
    a = cmds["compile"].split(" ")
    o, i = [], 0
    while i < len(a):
        x = a[i]
        if x == "-o":
            o += ["-o", obj]
            i += 2
            continue
        if x == "-MF":
            o += ["-MF", os.path.join(wd, "k.d")]
            i += 2
            continue
        if x == "-I..":
            o += [x, "-I" + ov, "-I" + ov + "/tt_metal"]
            i += 1
            continue
        if x == "-I" + edm:
            o.append("-I" + os.path.join(ov, EDM_REL))
            i += 1
            continue
        # A variant that overlays files outside tt_metal/fabric (e.g. NoC API headers under tt_metal/hw/inc): its copy
        # of the include directory goes first; files it does not overlay still come from the tree's directory.
        m = re.fullmatch(r"-I(?:\S*/)?(tt_metal/(?!fabric)\S*?)/?", x)
        if m and os.path.isdir(os.path.join(ov, m.group(1))):
            o += ["-I" + os.path.join(ov, m.group(1)), x]
            i += 1
            continue
        if set_opt and re.fullmatch(r"-O\w+", x):
            o.append("-" + set_opt)
            i += 1
            continue
        if x == "-c":
            o += flags
        o.append(x)
        i += 1
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


def build(a):
    d = json.load(open(a.cmds))
    kernel, db = d["kernel"], d["configs"]
    if a.opt not in db:
        sys.exit(f"no -{a.opt} configs in {a.cmds} (have {', '.join(db)})")
    db = db[a.opt]
    hashes = list(db) if a.hashes == "all" else a.hashes.split(",")
    first = next(iter(next(iter(db.values())).values()))["compile"]
    root = os.path.abspath(a.root) if a.root else tree_of(first)
    if not root or not os.path.isdir(os.path.join(root, "tt_metal/fabric")):
        sys.exit(f"logged tree {root} not found: give --root")
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ov = make_overlay(os.path.abspath(a.variant), root, out)
    flags = a.flags.split() if a.flags else []
    jobs = [(h, p, c) for h in hashes for p, c in db[h].items() if "compile" in c and "link" in c]
    for h in hashes:
        prep_headers(h, next(iter(db[h].values()))["compile"], kernel, root, ov, out)
    bad = 0
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        for h, p, err in ex.map(
            lambda j: build_one(j[0], j[1], j[2], root, ov, out, flags, a.set_opt, a.toolchain_root), jobs
        ):
            if err:
                bad += 1
                print(f"{h}/{p}: {err}")
    print(f"built {len(jobs) - bad}/{len(jobs)} ELFs into {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--kernel", default="fabric_erisc_router")
    c.add_argument("logs", nargs="+")
    b = sp.add_parser("build")
    b.add_argument("--cmds", required=True)
    b.add_argument("--variant", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--opt", default="O3")
    b.add_argument("--set-opt", default=None)
    b.add_argument("--hashes", default="all")
    b.add_argument("--flags", default="")
    b.add_argument("--root", default=None)
    b.add_argument("--toolchain-root", default=None)
    b.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    sys.exit(collect(args) if args.cmd == "collect" else build(args))
