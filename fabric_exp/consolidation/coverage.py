#!/usr/bin/env python3
"""coverage.py: compile a router variant for production config types the logged hardware runs never built.

  mock [CASE ...]           CPU-only mock-cluster runs (TT_METAL_MOCK_CLUSTER_DESC_PATH; no device): test_tt_fabric
                            JIT-builds the fabric router for every link, logging each compile and link command. The
                            run then fails in result validation (mock reads), which does not matter here.
  ingest LOG CASE           take a device run's log (cov_device_runs.sh) as case CASE; its kernel cache must still exist
  collect [CASE ...]        re-read the cases' logs into their cmds.json (mock does this after each run)
  synth                     derive the config types no host path produces, from mock configs, by editing named CT args
                            and defines exactly as the builder would emit them (see SYNTH)
  build --variant DIR --opt O3|O2|Os --name N [--cases C,...]
                            rebuild one config per distinct (ERISC, CT args, defines) of each case from DIR (offline_build's
                            overlay; v3 runs in v2's place); results in COV/builds/N_<opt>/results.json
  report NAME ...           the coverage table (case x level), with the first error of each failure

COV (outside the repo) holds a runtime root of symlinks into the tree with its own generated/ directory, so a mock run
writes nothing into the tree; per case: the yaml, the log, the kernel cache and cmds.json. Each mock case runs with the
logged suites' settings (TT_FABRIC_KERNEL_VERSION=2, keepalive 1000, flags 13129088, BW telemetry off) plus its own.
The JIT build is limited to 16 CPUs (taskset) at nice 10. Compile commands are logged at the case's default level and
rebuilt at the requested one (offline_build's --set-opt).
"""
import argparse, concurrent.futures, glob, hashlib, json, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_build as ob  # noqa: E402

TREE = "/home/aagarwal/tt-metal"
COV = os.environ.get("TT_COV_DIR", "/home/aagarwal/.tt-metal-v3r-builds/cov")
ROOT = os.path.join(COV, "root")
BIN = TREE + "/build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric"
BH = TREE + "/fabric_exp/a08u24_cluster_desc.yaml"
WH_T3K = TREE + "/tt_metal/third_party/umd/tests/cluster_descriptor_examples/t3k_cluster_desc.yaml"
GLX = TREE + "/tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml"
EXPRESS_8X4 = {"TT_MESH_GRAPH_DESC_PATH": TREE + "/tests/tt_metal/tt_fabric/custom_mesh_descriptors/"
                                          "express_links_8x4_mesh_graph_descriptor.textproto"}
KDIR = "kernels/fabric_erisc_router_v2/"
TREE_EDM = TREE + "/tt_metal/fabric/impl/kernels/edm_fabric"
CPUS = os.environ.get("TT_COV_CPUS", "48-63")

BASE_ENV = {
    "TT_FABRIC_KERNEL_VERSION": "2",
    "TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT": "1000",
    "TT_FABRIC_EXP_FLAGS": "13129088",
    "TT_FABRIC_NO_BW_TELEMETRY": "1",
    "TT_METAL_LOG_KERNELS_COMPILE_COMMANDS": "1",
}


def test_yaml(topology, extra=""):
    return f"""Tests:
  - name: "Cov{topology}"
    fabric_setup:
      topology: {topology}
{extra}    defaults:
      ftype: unicast
      ntype: unicast_write
      size: 1024
      num_packets: 10
    patterns:
      - type: full_device_random_pairing
"""


MUX = "      fabric_tensix_config: Mux\n"
TEL = {"TT_METAL_FABRIC_TELEMETRY": "1"}
BW = {"TT_METAL_FABRIC_BW_TELEMETRY": "1", "TT_FABRIC_NO_BW_TELEMETRY": None}
PROF = {"TT_FABRIC_PROFILE_RX_CH_FWD": "1"}
# Channel trimming: test_tt_fabric sets the capture/replay rtoptions itself per test (setup_channel_trimming_rtoptions),
# from the yaml key, and runs each test twice: capture, then replay of the captured profile.
TRIM = "      enable_channel_trimming: true\n"
ONE = {"TT_METAL_DISABLE_FABRIC_TWO_ERISC": "1"}
ALL_INSTR = {**TEL, **BW, **PROF}

# name: (cluster descriptor, yaml, extra env, VC1 split over two mock ranks)
CASES = {
    "bh_1d": (BH, test_yaml("Linear"), {}, False),
    "bh_ring": (BH, test_yaml("Ring"), {}, False),
    "bh_2d": (BH, test_yaml("Mesh"), {}, False),
    "bh_torus": (BH, test_yaml("Torus", "      torus_config: XY\n"), {}, False),
    "bh_2d_telemetry": (BH, test_yaml("Mesh"), TEL, False),
    "bh_1d_telemetry": (BH, test_yaml("Linear"), TEL, False),
    "bh_2d_bwtelemetry": (BH, test_yaml("Mesh"), BW, False),
    "bh_2d_codeprof": (BH, test_yaml("Mesh"), PROF, False),
    "bh_2d_trim": (BH, test_yaml("Mesh", TRIM), {}, False),
    "bh_1d_trim": (BH, test_yaml("Linear", TRIM), {}, False),
    "bh_2d_allinstr": (BH, test_yaml("Mesh", TRIM), ALL_INSTR, False),
    "bh_1d_allinstr": (BH, test_yaml("Linear", TRIM), ALL_INSTR, False),
    "bh_ring_allinstr": (BH, test_yaml("Ring", TRIM), ALL_INSTR, False),
    "bh_2d_vc2": (BH, test_yaml("Mesh"), {"TT_METAL_ENABLE_FABRIC_VC2": "1"}, False),
    "bh_2d_mux": (BH, test_yaml("Mesh", MUX), {}, False),
    "bh_1d_mux": (BH, test_yaml("Linear", MUX), {}, False),
    "bh_2d_1erisc": (BH, test_yaml("Mesh"), ONE, False),
    "bh_1d_1erisc": (BH, test_yaml("Linear"), ONE, False),
    "bh_vc1": (BH, None, {}, "bh"),
    "bh_vc1_1erisc": (BH, None, ONE, "bh"),
    "bh_vc1_allinstr": (BH, None, ALL_INSTR, "bh"),
    "wh_vc1": (WH_T3K, None, {}, "wh"),
    # Single BH Galaxy (8x4); express = the 8x4 torus with one dim-0 express pattern (5 VC0 sender channels, Z routers)
    "glx_2d": (GLX, test_yaml("Mesh"), {}, False),
    "glx_1d": (GLX, test_yaml("Linear"), {}, False),
    "glx_ring": (GLX, test_yaml("Ring"), {}, False),
    "glx_torus": (GLX, test_yaml("Torus", "      torus_config: XY\n"), {}, False),
    "glx_express": (GLX, test_yaml("Torus", "      torus_config: XY\n"), EXPRESS_8X4, False),
    "wh_1d": (WH_T3K, test_yaml("Linear"), {}, False),
    "wh_ring": (WH_T3K, test_yaml("Ring"), {}, False),
    "wh_2d": (WH_T3K, test_yaml("Mesh"), {}, False),
    "wh_1d_allinstr": (WH_T3K, test_yaml("Linear", TRIM), ALL_INSTR, False),
    "wh_2d_allinstr": (WH_T3K, test_yaml("Mesh", TRIM), ALL_INSTR, False),
    "wh_1d_mux": (WH_T3K, test_yaml("Linear", MUX), {}, False),
    "wh_2d_mux": (WH_T3K, test_yaml("Mesh", MUX), {}, False),
}


def ensure_root():
    if os.path.isdir(ROOT):
        return
    os.makedirs(os.path.join(ROOT, "generated"))
    for e in os.listdir(TREE):
        if e != "generated":
            os.symlink(os.path.join(TREE, e), os.path.join(ROOT, e))


def run_mock(name):
    desc, yaml, extra, vc1 = CASES[name]
    d = os.path.join(COV, "cases", name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    env = {k: v for k, v in os.environ.items() if k not in ("TT_METAL_SLOW_DISPATCH_MODE",)}
    env.update(BASE_ENV)
    for k, v in extra.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    env.update({"TT_METAL_RUNTIME_ROOT": ROOT, "TT_METAL_HOME": ROOT, "TT_METAL_LOGS_PATH": d,
                "TT_METAL_CACHE": os.path.join(d, "cache")})
    pre = ["timeout", "1200", "nice", "-n", "10", "taskset", "-c", CPUS]
    if not vc1:
        open(os.path.join(d, "test.yaml"), "w").write(yaml)
        env["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = desc
        cmd = pre + [BIN, "--test_config", os.path.join(d, "test.yaml")]
    else:
        # Two ranks, one mesh each, both mocking the same host: BH is vc1_mock.sh's split of this host into two 2x2
        # meshes; WH is upstream's T3K 2x2 multi-mesh test setup.
        v = TREE + "/fabric_exp/vc1"
        mapping = os.path.join(d, "mock_mapping.yaml")
        open(mapping, "w").write(f'rank_to_cluster_mock_cluster_desc:\n  0: "{desc}"\n  1: "{desc}"\n')
        rb = os.path.join(d, "rank_bindings.yaml")
        if vc1 == "bh":
            rbt = open(v + "/p150x8_2x2_split_rank_bindings.yaml").read().replace('"fabric_exp/vc1/', f'"{v}/')
        else:
            rbt = open(TREE + "/tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml").read()
            rbt = rbt.replace('"tests/', f'"{TREE}/tests/')
        open(rb, "w").write(rbt)
        shutil.copy(v + "/vc1_sanity.yaml", os.path.join(d, "test.yaml"))
        cmd = pre + [TREE + "/python_env/bin/python", TREE + "/ttnn/ttnn/distributed/ttrun.py",
                     "--mock-cluster-rank-binding", mapping, "--rank-binding", rb, "--mpi-args", "--oversubscribe",
                     BIN, "--test_config", os.path.join(d, "test.yaml")]
    with open(os.path.join(d, "mock.log"), "w") as log:
        r = subprocess.run(cmd, cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT,
                           preexec_fn=lambda: __import__("resource").setrlimit(
                               __import__("resource").RLIMIT_CORE, (0, 0)))
    cmds = collect_case(name)
    n = sum(len(p) for p in cmds.values())
    print(f"{name}: exit {r.returncode}, {len(cmds)} router configs ({n} ELFs logged)")


def collect_case(name):
    d = os.path.join(COV, "cases", name)
    db = {}
    for line in open(os.path.join(d, "mock.log"), errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"g\+\+ (compile|link) cmd: (.*?)\s*\(build\.cpp:\d+\)\s*$", line)
        if not m or KDIR not in m.group(2):
            continue
        kind, cmd = m.group(1), m.group(2)
        # Two MPI ranks write one log (VC1 split): keep only commands no other line cut into.
        if ("<stdout>" in cmd or "<stderr>" in cmd or cmd.count("riscv-tt-elf-g++") != 1
                or len(set(re.findall(KDIR + r"(\d+)/", cmd))) != 1):
            continue
        if kind == "compile" and not (cmd.startswith((ROOT + "/", TREE + "/")) and cmd.count(" -c ") == 1
                                      and cmd.endswith(".d")):
            continue
        if kind == "link" and not (cmd.startswith("cd ") and cmd.count(" -o ") == 1 and cmd.endswith(".elf")):
            continue
        h, proc = re.search(KDIR + r"(\d+)/(\w+)/", cmd).groups()
        db.setdefault(h, {}).setdefault(proc, {}).setdefault(kind, cmd)
    db = {h: {p: c for p, c in ps.items() if "compile" in c and "link" in c} for h, ps in db.items()}
    db = {h: ps for h, ps in db.items() if ps}
    json.dump(db, open(os.path.join(d, "cmds.json"), "w"), indent=0)
    return db


def cache_dir(cmd, h):
    return re.search(r"(\S+/" + KDIR + h + ")/", cmd).group(1)


def named_args(path):
    return dict(re.findall(r'\{"(\w+)",(\d+)\}', open(path).read()))


# Per-link values: configs that differ only in these compile to the same code shape.
PER_LINK = {"MY_ETH_CHANNEL", "EDM_CHANNELS_MASK", "LOCAL_HANDSHAKE_MASTER_ETH_CHAN", "IS_LOCAL_HANDSHAKE_MASTER",
            "NUM_LOCAL_EDMS", "IS_HANDSHAKE_SENDER", "PEER_ERISC_NOC_X", "PEER_ERISC_NOC_Y"}


def signature(cmd, h, proc):
    n = named_args(os.path.join(cache_dir(cmd, h), "named_ct_arg_map_generated.h"))
    defs = sorted(x for x in cmd.split() if x.startswith("-D") and not x.startswith("-DFULL_KERNEL_NAME"))
    return proc, tuple(sorted((k, v) for k, v in n.items() if k not in PER_LINK)), tuple(defs)


# Config types no host path produces (each is a constant in the builder, erisc_datamover_builder.cpp):
#   datacache       enable_risc_cpu_data_cache = false (constexpr, :1134)
#   spinwait_send   eth_txq_spin_wait_send_next_data = false (:1076)
#   spinwait_ack    eth_txq_spin_wait_receiver_send_completion_ack = false (:1077)
#   nofuse          fuse_receiver_flush_and_completion_ptr = true (member default, never assigned, .hpp:545)
# and one the host produces only through FabricTensixConfig::UDM, which test_tt_fabric cannot select (its yaml mapper
# has Default and Mux): udm = the 2D mux config with what connect_to_local_tensix_builder and the fabric defines change
# (UDM_MODE named arg 1, LOCAL_RELAY_NUM_BUFFERS, NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS 1, -DUDM_MODE=1).
SYNTH = {
    "bh_2d_datacache": ("bh_2d", {"ENABLE_RISC_CPU_DATA_CACHE": "1"}, []),
    "bh_1d_datacache": ("bh_1d", {"ENABLE_RISC_CPU_DATA_CACHE": "1"}, []),
    "bh_vc1_datacache": ("bh_vc1", {"ENABLE_RISC_CPU_DATA_CACHE": "1"}, []),
    "bh_2d_spinwait": ("bh_2d", {"ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA": "1",
                                 "ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK": "1"}, []),
    "bh_1d_spinwait": ("bh_1d", {"ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA": "1",
                                 "ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK": "1"}, []),
    "bh_2d_nofuse": ("bh_2d", {"FUSE_RECEIVER_FLUSH_AND_COMPLETION_PTR": "0"}, []),
    "bh_1d_nofuse": ("bh_1d", {"FUSE_RECEIVER_FLUSH_AND_COMPLETION_PTR": "0"}, []),
    "bh_vc1_nofuse": ("bh_vc1", {"FUSE_RECEIVER_FLUSH_AND_COMPLETION_PTR": "0"}, []),
    "bh_2d_udm": ("bh_2d_mux", {"UDM_MODE": "1", "LOCAL_RELAY_NUM_BUFFERS": "8",
                                "NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS": "1"}, ["-DUDM_MODE=1"]),
    "wh_1d_nofuse": ("wh_1d", {"FUSE_RECEIVER_FLUSH_AND_COMPLETION_PTR": "0"}, []),
}


def synth():
    for name, (base, edits, defines) in SYNTH.items():
        bdir = os.path.join(COV, "cases", base)
        bcmds = json.load(open(os.path.join(bdir, "cmds.json")))
        d = os.path.join(COV, "cases", name)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
        out = {}
        for h, procs in bcmds.items():
            src = cache_dir(next(iter(procs.values()))["compile"], h)
            nh = str(int(hashlib.sha1((name + h).encode()).hexdigest()[:15], 16))
            dst = os.path.join(d, "cache", KDIR, nh)
            os.makedirs(dst)
            for f in glob.glob(src + "/*.h") + glob.glob(src + "/*.hpp"):
                shutil.copy(f, dst)
            m = os.path.join(dst, "named_ct_arg_map_generated.h")
            t = open(m).read()
            for k, v in edits.items():
                t, n = re.subn(r'\{"%s",\d+\}' % k, '{"%s",%s}' % (k, v), t)
                if n != 1:
                    sys.exit(f"{name}: named arg {k} found {n} times in {m}")
            open(m, "w").write(t)
            out[nh] = {}
            for p, c in procs.items():
                cc = {k: v.replace(src, dst) for k, v in c.items()}
                if defines:
                    cc["compile"] = cc["compile"].replace(" -c ", " " + " ".join(defines) + " -c ", 1)
                out[nh][p] = cc
        json.dump(out, open(os.path.join(d, "cmds.json"), "w"), indent=0)
        print(f"{name}: {len(out)} configs from {base}")


def build(args):
    ob.EDM = ROOT + "/tt_metal/fabric/impl/kernels/edm_fabric"  # mock commands use the symlinked root
    out = os.path.abspath(os.path.join(COV, "builds", f"{args.name}_{args.opt}"))
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    ov = ob.make_overlay(os.path.abspath(args.variant), out)
    names = args.cases.split(",") if args.cases else sorted(os.listdir(os.path.join(COV, "cases")))
    jobs, meta = [], {}
    for case in names:
        cp = os.path.join(COV, "cases", case, "cmds.json")
        if not os.path.exists(cp):
            continue
        db = json.load(open(cp))
        seen = set()
        for h, procs in sorted(db.items()):
            for p, c in sorted(procs.items()):
                s = signature(c["compile"], h, p)
                if s in seen:
                    continue
                seen.add(s)
                key = f"{case}__{h}"
                cdir = os.path.join(out, key)
                os.makedirs(cdir, exist_ok=True)
                for f in glob.glob(cache_dir(c["compile"], h) + "/*.h*"):
                    shutil.copy(f, cdir)
                ki = os.path.join(cdir, "kernel_includes.hpp")
                t = open(ki).read().replace(ob.EDM, ov + "/tt_metal/fabric/impl/kernels/edm_fabric")
                t = t.replace(TREE_EDM, ov + "/tt_metal/fabric/impl/kernels/edm_fabric")
                open(ki, "w").write(t)
                ob.add_host_setting_args(cdir)
                # build_one keys its work dir by hash: point the command at this case's copy. A device log (ingest)
                # names the tree itself; map its router directory onto the root's so the overlay replaces it.
                cc = {k: v.replace(f"{KDIR}{h}/", f"{KDIR}{key}/").replace(TREE_EDM + " ", ob.EDM + " ")
                      for k, v in c.items()}
                jobs.append((key, p, cc))
                meta[f"{key}/{p}"] = {"case": case}
    print(f"{len(jobs)} ELFs ({len(meta)} distinct configs) over {len(names)} cases")

    def one(j):
        key, p, c = j
        return ob.build_one(key, p, c, ov, out, [], args.opt)

    res = {}
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        for key, p, err in ex.map(one, jobs):
            res[f"{key}/{p}"] = {"case": meta[f"{key}/{p}"]["case"], "ok": err is None, "err": first_error(err)}
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    bad = sum(not r["ok"] for r in res.values())
    print(f"built {len(res) - bad}/{len(res)} into {out}")


def first_error(err):
    if not err:
        return None
    for line in err.splitlines():
        if "error:" in line:
            return re.sub(r"^\S*/", "", line.strip())[:300]
    return err.strip().splitlines()[-1][:300] if err.strip() else "unknown"


def report(names):
    rows = {}
    for n in names:
        for opt in ("O3", "O2", "Os"):
            p = os.path.join(COV, "builds", f"{n}_{opt}", "results.json")
            if not os.path.exists(p):
                continue
            for k, r in json.load(open(p)).items():
                e = rows.setdefault((r["case"], n), {}).setdefault(opt, [0, 0, set()])
                e[0] += r["ok"]
                e[1] += 1
                if r["err"]:
                    e[2].add(r["err"])
    for (case, n), lv in sorted(rows.items()):
        cells = " | ".join(f"{o} {lv[o][0]}/{lv[o][1]}" if o in lv else f"{o} -" for o in ("O3", "O2", "Os"))
        print(f"{case:22s} {n:8s} {cells}")
        errs = set().union(*(lv[o][2] for o in lv))
        for e in sorted(errs)[:4]:
            print(f"    {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("mock")
    m.add_argument("cases", nargs="*")
    i = sp.add_parser("ingest")
    i.add_argument("log")
    i.add_argument("case")
    c = sp.add_parser("collect")
    c.add_argument("cases", nargs="*")
    sp.add_parser("synth")
    b = sp.add_parser("build")
    b.add_argument("--variant", required=True)
    b.add_argument("--opt", required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--cases", default="")
    b.add_argument("--jobs", type=int, default=16)
    r = sp.add_parser("report")
    r.add_argument("names", nargs="+")
    a = ap.parse_args()
    if a.cmd == "mock":
        ensure_root()
        for c in a.cases or list(CASES):
            run_mock(c)
    elif a.cmd == "ingest":
        d = os.path.join(COV, "cases", a.case)
        os.makedirs(d, exist_ok=True)
        shutil.copy(a.log, os.path.join(d, "mock.log"))
        print(a.case, len(collect_case(a.case)), "configs")
    elif a.cmd == "collect":
        for c in a.cases or [c for c in CASES if os.path.isdir(os.path.join(COV, "cases", c))]:
            print(c, len(collect_case(c)), "configs")
    elif a.cmd == "synth":
        synth()
    elif a.cmd == "build":
        build(a)
    else:
        report(a.names)


if __name__ == "__main__":
    main()
