#!/usr/bin/env python3
"""vc2l_offline.py: offline router builds for the VC2 line-mode bring-up (VC2_LINE_BRINGUP.md), in this worktree only.

The study's coverage tooling (fabric_exp/consolidation/coverage.py, offline_build.py) hard-codes the main tree and its
coverage directory. This wraps the worktree's copies of those modules with every path pointed at this worktree and a
coverage directory of its own (COV below); the main tree's post-merge captures (CAPTURES_M) are only read.

  variant NAME              snapshot this worktree's router sources (every file under tt_metal/fabric/hw,
                            tt_metal/fabric/impl/kernels, tt_metal/hw/inc and tt_metal/hostdevcommon that differs from
                            checkpoint 12, plus the untracked router/ headers) into fabric_exp/variants/NAME
  mock CASE ...             CPU-only mock-cluster capture with this worktree's host and router (coverage.py's cases, plus
                            the VC2 line cases below) into COV/cases/CASE
  build --variant DIR --name N [--opt O3] [--cases C,...] [--source main|own|both]
                            rebuild one config per distinct (ERISC, CT args, defines) of each case: the main tree's
                            captures (source main), this worktree's (own), or both; COV/builds/N_<opt>
  cmp A B                   per case: ELFs with byte-identical .text and loadable bytes in builds A and B
Captures from before this work lack VC2 line mode's named args; they get neighbour mode's values (add_vc2_line_args).
"""
import argparse, collections, concurrent.futures, glob, json, os, re, shutil, subprocess, sys

WT = "/home/aagarwal/tt-metal-vc2line"
COV = os.environ.setdefault("TT_COV_DIR", "/home/aagarwal/.tt-metal-vc2line-cov")
CAPTURES_M = "/home/aagarwal/.tt-metal-v3r-builds/cov_m"  # read only
# The overlay base: tt_metal/fabric at checkpoint 12 (git archive 5578d6292a6), which the main tree's committed files are.
os.environ.setdefault("OFFLINE_BASE", COV + "/base_ckpt12")

sys.path.insert(0, WT + "/fabric_exp/consolidation")
import offline_build as ob  # noqa: E402
import coverage as cov  # noqa: E402

ob.ROOT = WT
ob.BASE = os.environ["OFFLINE_BASE"]
cov.TREE = WT
cov.COV = COV
cov.ROOT = os.path.join(COV, "root")
cov.BIN = WT + "/build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric"
cov.GTEST_BIN = WT + "/build_Release/test/tt_metal/tt_fabric/fabric_unit_tests"
cov.BH = WT + "/fabric_exp/a08u24_cluster_desc.yaml"
cov.GLX = (WT + "/tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/"
           "single_bh_galaxy_clus_desc.yaml")
cov.TREE_EDM = WT + "/tt_metal/fabric/impl/kernels/edm_fabric"
OC = WT + "/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy"
RE = WT + "/runtime/sfpi/compiler/bin/riscv-tt-elf-readelf"

# VC2 line mode cases: VC2 on, line mode on (TT_METAL_FABRIC_VC2_LINE), on the P150x8 2D mesh and the Galaxy 2D mesh,
# both ERISC modes.
VC2L = {"TT_METAL_ENABLE_FABRIC_VC2": "1", "TT_METAL_FABRIC_VC2_LINE": "1"}
cov.CASES.update({
    "bh_2d_vc2line": (cov.BH, cov.test_yaml("Mesh"), VC2L, False),
    "bh_2d_vc2line_1erisc": (cov.BH, cov.test_yaml("Mesh"), {**VC2L, **cov.ONE}, False),
    "bh_2d_vc2line_allinstr": (cov.BH, cov.test_yaml("Mesh", cov.TRIM), {**VC2L, **cov.ALL_INSTR}, False),
    "glx_2d_vc2line": (cov.GLX, cov.test_yaml("Mesh"), VC2L, False),
    "glx_2d_vc2line_1erisc": (cov.GLX, cov.test_yaml("Mesh"), {**VC2L, **cov.ONE}, False),
    # Phase 2, VC2 rings: line mode on tori (rings on torus axes, lines on the other), and neighbour mode on a torus
    "glx_torus_vc2line": (cov.GLX, cov.test_yaml("Torus", "      torus_config: XY\n"), VC2L, False),
    "glx_torus_x_vc2line": (cov.GLX, cov.test_yaml("Torus", "      torus_config: X\n"), VC2L, False),
    "glx_torus_y_vc2line": (cov.GLX, cov.test_yaml("Torus", "      torus_config: Y\n"), VC2L, False),
    "glx_torus_vc2line_1erisc": (cov.GLX, cov.test_yaml("Torus", "      torus_config: XY\n"), {**VC2L, **cov.ONE}, False),
    "bh_torus_vc2line": (cov.BH, cov.test_yaml("Torus", "      torus_config: XY\n"), VC2L, False),
    "bh_torus_vc2line_allinstr": (cov.BH, cov.test_yaml("Torus", "      torus_config: XY\n" + cov.TRIM),
                                  {**VC2L, **cov.ALL_INSTR}, False),
    "glx_torus_vc2": (cov.GLX, cov.test_yaml("Torus", "      torus_config: XY\n"), {"TT_METAL_ENABLE_FABRIC_VC2": "1"}, False),
    # Phase 3, VC2 fan-in into VC1: multi-mesh splits (two ranks, one 2x2 mesh each; split "bh" = the P150x8 split of
    # fabric_exp/vc1, "glx" = the Galaxy split of glx_2x2_split_*), VC2 line mode, neighbour mode and VC2 off
    "bh_vc1_vc2line": (cov.BH, None, VC2L, "bh"),
    "bh_vc1_vc2line_1erisc": (cov.BH, None, {**VC2L, **cov.ONE}, "bh"),
    "bh_vc1_vc2line_allinstr": (cov.BH, None, {**VC2L, **cov.ALL_INSTR}, "bh"),
    "glx_vc1_vc2line": (cov.GLX, None, VC2L, "glx"),
    "glx_vc1_vc2line_1erisc": (cov.GLX, None, {**VC2L, **cov.ONE}, "glx"),
    "glx_vc1_vc2": (cov.GLX, None, {"TT_METAL_ENABLE_FABRIC_VC2": "1"}, "glx"),
    "glx_vc1": (cov.GLX, None, {}, "glx"),
    # the same at -O3 (the host's default for VC1 routers is -Os): the size check at program finalize
    "glx_vc1_vc2line_o3": (cov.GLX, None, {**VC2L, "TT_METAL_FABRIC_OPT_LEVEL": "O3"}, "glx"),
    # Neighbour mode with this worktree's host, to compare its named args with the main tree's captures
    "bh_2d_vc2_wt": (cov.BH, cov.test_yaml("Mesh"), {"TT_METAL_ENABLE_FABRIC_VC2": "1"}, False),
    "glx_2d_vc2_wt": (cov.GLX, cov.test_yaml("Mesh"), {"TT_METAL_ENABLE_FABRIC_VC2": "1"}, False),
    "glx_2d_wt": (cov.GLX, cov.test_yaml("Mesh"), {}, False),
})

# Named args the host gained in this work, with the values it passes in neighbour mode (and on routers without VC2).
VC2_LINE_ARGS = {
    "VC2_LINE_MODE": 0,
    "NUM_DOWNSTREAM_SENDERS_VC2": 0,
    "VC2_DOWNSTREAM_EDM_SIZE": 0,
    "OPEN_DOWNSTREAM_VC2_CONNECTIONS": 0,
    "VC2_FREE_SLOTS_FROM_DOWNSTREAM_STREAM_ID": 32,  # k_unused_stream_id
    "ENABLE_FIRST_LEVEL_ACK_VC2": 0,  # phase 2: VC2 rings
}
_add_host_setting_args = ob.add_host_setting_args


def add_vc2_line_args(hd):
    _add_host_setting_args(hd)
    m = os.path.join(hd, "named_ct_arg_map_generated.h")
    t = open(m).read()
    extra = "".join('{"%s",%d},' % (k, v) for k, v in VC2_LINE_ARGS.items() if '{"%s",' % k not in t)
    if extra:
        t = t.replace("#define KERNEL_COMPILE_TIME_ARG_MAP ", "#define KERNEL_COMPILE_TIME_ARG_MAP " + extra, 1)
        open(m, "w").write(t)


ob.add_host_setting_args = add_vc2_line_args

# Multi-mesh splits for mock captures: rank bindings (paths in them relative to the tree) and the test yaml.
SPLITS = {
    "bh": (WT + "/fabric_exp/vc1/p150x8_2x2_split_rank_bindings.yaml", WT + "/fabric_exp/vc1/vc1_sanity.yaml"),
    "glx": (WT + "/fabric_exp/vc2line/glx_2x2_split_rank_bindings.yaml", WT + "/fabric_exp/vc1/vc1_sanity.yaml"),
}


def run_mock(name):
    """coverage.run_mock, with this worktree's splits (two mock ranks, each the whole cluster descriptor)."""
    desc, yaml, extra, split = cov.CASES[name]
    if not split:
        return cov.run_mock(name)
    d = os.path.join(COV, "cases", name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    env = {k: v for k, v in os.environ.items() if k not in ("TT_METAL_SLOW_DISPATCH_MODE",)}
    env.update(cov.BASE_ENV)
    for k, v in extra.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    env.update({"TT_METAL_RUNTIME_ROOT": cov.ROOT, "TT_METAL_HOME": cov.ROOT, "TT_METAL_LOGS_PATH": d,
                "TT_METAL_CACHE": os.path.join(d, "cache")})
    bindings, test = SPLITS[split]
    mapping = os.path.join(d, "mock_mapping.yaml")
    open(mapping, "w").write(f'rank_to_cluster_mock_cluster_desc:\n  0: "{desc}"\n  1: "{desc}"\n')
    rb = os.path.join(d, "rank_bindings.yaml")
    open(rb, "w").write(open(bindings).read().replace('"fabric_exp/', f'"{WT}/fabric_exp/'))
    shutil.copy(yaml if yaml else test, os.path.join(d, "test.yaml"))
    cmd = ["timeout", "1200", "nice", "-n", "10", "taskset", "-c", cov.CPUS, WT + "/python_env/bin/python",
           WT + "/ttnn/ttnn/distributed/ttrun.py", "--mock-cluster-rank-binding", mapping, "--rank-binding", rb,
           "--mpi-args", "--oversubscribe", cov.BIN, "--test_config", os.path.join(d, "test.yaml")]
    with open(os.path.join(d, "mock.log"), "w") as log:
        r = subprocess.run(cmd, cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
    cmds = cov.collect_case(name)
    print(f"{name}: exit {r.returncode}, {len(cmds)} router configs ({sum(len(p) for p in cmds.values())} ELFs logged)")


SOURCE_DIRS = ["tt_metal/fabric/hw", "tt_metal/fabric/impl/kernels", "tt_metal/hw/inc", "tt_metal/hostdevcommon"]


def make_variant(name):
    out = os.path.join(WT, "fabric_exp/variants", name)
    if os.path.exists(out):
        shutil.rmtree(out)
    changed = subprocess.run(["git", "-C", WT, "diff", "--name-only", "5578d6292a6", "--", *SOURCE_DIRS],
                             capture_output=True, text=True, check=True).stdout.split()
    changed += subprocess.run(["git", "-C", WT, "ls-files", "--others", "--exclude-standard", "--", *SOURCE_DIRS],
                              capture_output=True, text=True, check=True).stdout.split()
    for rel in sorted(set(changed)):
        if not os.path.exists(os.path.join(WT, rel)):
            continue
        os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
        shutil.copy2(os.path.join(WT, rel), os.path.join(out, rel))
        print("  ", rel)
    print(f"{name}: {len(set(changed))} files")


def case_dirs(source):
    dirs = {}
    if source in ("main", "both"):
        for d in sorted(glob.glob(CAPTURES_M + "/cases/*")):
            dirs[os.path.basename(d)] = d
    if source in ("own", "both"):
        for d in sorted(glob.glob(COV + "/cases/*")):
            dirs[os.path.basename(d)] = d
    return dirs


# The router directory as each capture's compile commands name it: the main tree's coverage root, this worktree's
# coverage root, the main tree itself (device-log captures).
EDM_ROOTS = [CAPTURES_M + "/root/tt_metal/fabric/impl/kernels/edm_fabric",
             COV + "/root/tt_metal/fabric/impl/kernels/edm_fabric",
             "/home/aagarwal/tt-metal/tt_metal/fabric/impl/kernels/edm_fabric",
             WT + "/tt_metal/fabric/impl/kernels/edm_fabric"]


def build(args):
    out = os.path.abspath(os.path.join(COV, "builds", f"{args.name}_{args.opt}"))
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    ov = ob.make_overlay(os.path.abspath(args.variant), out)
    ov_edm = ov + "/tt_metal/fabric/impl/kernels/edm_fabric"
    dirs = case_dirs(args.source)
    names = args.cases.split(",") if args.cases else sorted(dirs)
    jobs, meta = [], {}
    for case in names:
        cp = os.path.join(dirs.get(case, "/nonexistent"), "cmds.json")
        if not os.path.exists(cp):
            continue
        db = json.load(open(cp))
        seen = set()
        for h, procs in sorted(db.items()):
            for p, c in sorted(procs.items()):
                s = cov.signature(c["compile"], h, p)
                if s in seen:
                    continue
                seen.add(s)
                key = f"{case}__{h}"
                cdir = os.path.join(out, key)
                os.makedirs(cdir, exist_ok=True)
                for f in glob.glob(cov.cache_dir(c["compile"], h) + "/*.h*"):
                    shutil.copy(f, cdir)
                ki = os.path.join(cdir, "kernel_includes.hpp")
                t = open(ki).read()
                for r in EDM_ROOTS:
                    t = t.replace(r, ov_edm)
                open(ki, "w").write(t)
                add_vc2_line_args(cdir)
                cc = {k: v.replace(f"{cov.KDIR}{h}/", f"{cov.KDIR}{key}/") for k, v in c.items()}
                # map the command's router directory onto ob.EDM, which build_one replaces with the overlay's
                for r in EDM_ROOTS:
                    cc = {k: v.replace("-I" + r + " ", "-I" + ob.EDM + " ") for k, v in cc.items()}
                jobs.append((key, p, cc))
                meta[f"{key}/{p}"] = {"case": case}
    print(f"{len(jobs)} ELFs over {len(names)} cases")

    def one(j):
        key, p, c = j
        return ob.build_one(key, p, c, ov, out, [], args.opt)

    res = {}
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        for key, p, err in ex.map(one, jobs):
            res[f"{key}/{p}"] = {"case": meta[f"{key}/{p}"]["case"], "ok": err is None, "err": cov.first_error(err)}
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    # Keep the ELFs only (disk): objects, dependency files.
    for f in glob.glob(f"{out}/*/*/k.o") + glob.glob(f"{out}/*/*/k.d"):
        os.remove(f)
    bad = [k for k, r in res.items() if not r["ok"]]
    by_case = collections.Counter(res[k]["case"] for k in bad)
    print(f"built {len(res) - len(bad)}/{len(res)} into {out}; failures by case: {dict(by_case)}")
    for k in bad[:5]:
        print("  ", k, res[k]["err"])


def section_bytes(elf):
    text = subprocess.run([OC, "-O", "binary", "--only-section=.text", elf, "/dev/stdout"], capture_output=True).stdout
    load = subprocess.run([OC, "-O", "binary", elf, "/dev/stdout"], capture_output=True).stdout
    return text, load


def cmp(a, b):
    da, db_ = (os.path.join(COV, "builds", x) for x in (a, b))
    pairs = []
    for e in glob.glob(f"{da}/*/*/*.elf"):
        rel = os.path.relpath(e, da)
        if os.path.exists(f"{db_}/{rel}"):
            pairs.append((rel.split("__")[0], rel, e, f"{db_}/{rel}"))
    with concurrent.futures.ThreadPoolExecutor(48) as ex:
        res = list(ex.map(lambda p: (p[0], p[1], section_bytes(p[2]), section_bytes(p[3])), pairs))
    c = collections.defaultdict(lambda: [0, 0, 0])
    differ = []
    for case, rel, (ta, la), (tb, lb) in res:
        c[case][0] += 1
        c[case][1] += ta == tb
        c[case][2] += la == lb
        if la != lb:
            differ.append(rel)
    tot = [0, 0, 0]
    for case in sorted(c):
        print(f"{case:28s} ELFs {c[case][0]:4d}  .text same {c[case][1]:4d}  loadable same {c[case][2]:4d}")
        tot = [x + y for x, y in zip(tot, c[case])]
    print(f"{'total':28s} ELFs {tot[0]:4d}  .text same {tot[1]:4d}  loadable same {tot[2]:4d}")
    for rel in differ[:20]:
        print("  differs:", rel)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("variant")
    v.add_argument("name")
    m = sp.add_parser("mock")
    m.add_argument("cases", nargs="+")
    b = sp.add_parser("build")
    b.add_argument("--variant", required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--opt", default="O3")
    b.add_argument("--cases", default="")
    b.add_argument("--source", default="both")
    b.add_argument("--jobs", type=int, default=48)
    c = sp.add_parser("cmp")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "variant":
        make_variant(a.name)
    elif a.cmd == "mock":
        cov.ensure_root()
        for case in a.cases:
            run_mock(case)
    elif a.cmd == "build":
        build(a)
    else:
        cmp(a.a, a.b)
