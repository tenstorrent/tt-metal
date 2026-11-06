#!/usr/bin/env python3
"""cov.py: offline build and byte-identity gate (copied from the port agent's tt-metal-mainline/.scratch/cov.py for the
receiver tuning sweep, RECEIVER_TUNING.md; defaults point at /home/aagarwal/tt-metal-port-tune and its own COV dir).

Adapted from the study's coverage.py / offline_build.py / bytecmp.py; nothing here reads or writes the study tree.

  mock [CASE ...]        CPU-only mock-cluster runs of the worktree's binaries (TT_METAL_MOCK_CLUSTER_DESC_PATH): the JIT
                         builds every kernel and firmware the run needs, logging each compile and link command
                         (TT_METAL_LOG_KERNELS_COMPILE_COMMANDS). The run itself then fails in result checks (mock reads);
                         that does not matter. Each case keeps its log, cache and cmds.json under COV/cases/CASE.
  build --name N [--cases C,..] [--router-only]
                         re-run every logged compile + link command (one per distinct kernel build: the JIT's hash
                         directory) against the live worktree into COV/builds/N; records HEAD and whether the tracked
                         tree was clean. Firmware is rebuilt as well (its ELF is linked against by kernels: the rebuild
                         links kernels against the case's cached firmware, so a firmware change shows on the firmware
                         ELF only).
  cmp A B                compare the loadable bytes (objcopy -O binary) of every ELF two builds share; exit 0 when all
                         are identical and both have the same set. `cmp jit:CASE B` compares B against the JIT's own ELFs.
  report N               build failures by case, first error each
  fw N                   regenerate the precompiled firmware from the live tree (precompile_fw) and compare its ELFs
                         with the copy saved at the base commit (COV/fw/base)

COV is /home/aagarwal/.tt-metal-mainline-cov: a runtime root of symlinks into the worktree (its own generated/), the
cases and the builds. Mock runs are limited to TT_COV_CPUS (default 48-63) at nice 10.
"""
import argparse, collections, concurrent.futures, glob, json, os, re, shutil, subprocess, sys

TREE = os.environ.get("TT_COV_TREE", "/home/aagarwal/tt-metal-port-tune")  # a scratch host tree for O3 slot-row cases
HERE = os.path.dirname(os.path.abspath(__file__))
COV = os.environ.get("TT_COV_DIR", "/home/aagarwal/.tt-metal-port-tune-cov")
ROOT = os.path.join(COV, "root")
BIN = TREE + "/build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric"
GTEST_BIN = TREE + "/build/test/tt_metal/tt_fabric/fabric_unit_tests"
OC = TREE + "/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy"
CPUS = os.environ.get("TT_COV_CPUS", "48-63")

BH = HERE + "/p150x8_cluster_desc.yaml"
GLX = TREE + "/tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml"
EXPRESS_8X4 = {"TT_MESH_GRAPH_DESC_PATH": TREE + "/tests/tt_metal/tt_fabric/custom_mesh_descriptors/"
                                          "express_links_8x4_mesh_graph_descriptor.textproto"}

BASE_ENV = {"TT_METAL_LOG_KERNELS_COMPILE_COMMANDS": "1"}


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
TRIM = "      enable_channel_trimming: true\n"
VC2 = "      use_vc2: true\n"
TORUS_XY = "      torus_config: XY\n"
TORUS_Y = "      torus_config: Y\n"
ONE = {"TT_METAL_DISABLE_FABRIC_TWO_ERISC": "1"}
TEL = {"TT_METAL_FABRIC_TELEMETRY": "1"}
BW = {"TT_METAL_FABRIC_BW_TELEMETRY": "1"}
PROF = {"TT_FABRIC_PROFILE_RX_CH_FWD": "1"}
NBR = test_yaml("NeighborExchange").replace("type: full_device_random_pairing", "type: neighbor_exchange")

# Two ranks, one 2x2 mesh each, on a mock of the P150x8 (VC1: intermesh links between the meshes); ttrun.py from the
# worktree, run by a Python that has its dependencies (click, loguru, pydantic, yaml).
VC1 = "vc1"
PY = os.environ.get("TT_COV_PYTHON", "/home/aagarwal/tt-metal/python_env/bin/python")
VC1_DIR = HERE + "/vc1"
VC2_ENV = {"TT_METAL_ENABLE_FABRIC_VC2": "1"}

# Multi-rank mesh graphs, launched as CI's CPU-only fabric tests launch them (tt-run's discovery mode:
# --mesh-graph-descriptor with --mock-cluster-rank-binding, one rank per mesh): MGD(mapping, mgd, ranks). A mapping of
# None gives every rank the single BH Galaxy's descriptor.
class MGD(tuple):
    pass


CUSTOM_MGD = TREE + "/tests/tt_metal/tt_fabric/custom_mesh_descriptors/"
DUAL_GLX_MAPPING = TREE + "/tt_metal/third_party/tt-cluster-descriptors/blackhole/dual_bh_galaxy_experimental/" \
                          "dual_bh_galaxy_experimental_cluster_desc_mapping.yaml"
DUAL_GLX = MGD((DUAL_GLX_MAPPING, CUSTOM_MGD + "dual_bh_galaxy_experimental_mesh_graph_descriptor.textproto", 2))
GLX_SPLIT3 = MGD((None, CUSTOM_MGD + "bh_galaxy_split_4x4_2x4_3_mesh.textproto", 3))
GLX_SPLIT3_2X8 = MGD((None, CUSTOM_MGD + "bh_galaxy_split_2x8_2x4_3_mesh.textproto", 3))
GLX_ZRING = MGD((None, CUSTOM_MGD + "single_galaxy_2x2_z_ring_4stage_ring_mesh_graph_descriptor.textproto", 4))
MM_SANITY = "file:" + TREE + "/tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_multi_mesh_sanity_common.yaml"
GLX_TORUS_YX = MGD((None, CUSTOM_MGD + "single_galaxy_per_mesh_torus_yx_mesh_graph_descriptor.textproto", 3))
HETERO_YAML = "file:" + TREE + "/tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_per_mesh_heterogeneous.yaml"
PASS_THROUGH_YAML = "file:" + TREE + "/tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_mesh_passthrough.yaml"
NOVC = {"TT_METAL_FABRIC_DISABLE_INTERMESH_VC": "1"}
STREAMING = {"TT_METAL_STREAMING_PROFILER": "1"}
PASS = {"TT_METAL_ENABLE_FABRIC_MESH_PASS_THROUGH": "1"}
EXPRESS_EW = {"TT_MESH_GRAPH_DESC_PATH": HERE + "/mgd/express_links_4x8_ew_mesh_graph_descriptor.textproto"}
GLX_1X32 = {"TT_MESH_GRAPH_DESC_PATH": HERE + "/mgd/bh_galaxy_1x32_mesh_graph_descriptor.textproto"}
GLX_1X32_RING = {"TT_MESH_GRAPH_DESC_PATH": HERE + "/mgd/bh_galaxy_1x32_ring_mesh_graph_descriptor.textproto"}

# name: (cluster descriptor, yaml or "gtest:<filter>" or VC1, extra env)
CASES = {
    "bh_1d": (BH, test_yaml("Linear"), {}),
    "bh_ring": (BH, test_yaml("Ring"), {}),
    "bh_2d": (BH, test_yaml("Mesh"), {}),
    "bh_torus": (BH, test_yaml("Torus", TORUS_XY), {}),
    "bh_1d_1erisc": (BH, test_yaml("Linear"), ONE),
    "bh_ring_1erisc": (BH, test_yaml("Ring"), ONE),
    "bh_2d_1erisc": (BH, test_yaml("Mesh"), ONE),
    "bh_torus_1erisc": (BH, test_yaml("Torus", TORUS_XY), ONE),
    "bh_neighbor": (BH, NBR, {}),
    "bh_neighbor_1erisc": (BH, NBR, ONE),
    "bh_1d_mux": (BH, test_yaml("Linear", MUX), {}),
    "bh_2d_mux": (BH, test_yaml("Mesh", MUX), {}),
    "bh_ring_mux": (BH, test_yaml("Ring", MUX), {}),
    "bh_2d_vc2": (BH, test_yaml("Mesh", VC2), {}),
    "bh_2d_vc2_1erisc": (BH, test_yaml("Mesh", VC2), ONE),
    "bh_torus_vc2": (BH, test_yaml("Torus", TORUS_Y + VC2), {}),
    "bh_2d_telemetry": (BH, test_yaml("Mesh"), TEL),
    "bh_1d_telemetry": (BH, test_yaml("Linear"), TEL),
    "bh_2d_bwtelemetry": (BH, test_yaml("Mesh"), BW),
    "bh_2d_codeprof": (BH, test_yaml("Mesh"), PROF),
    "bh_2d_trim": (BH, test_yaml("Mesh", TRIM), {}),
    "bh_1d_trim": (BH, test_yaml("Linear", TRIM), {}),
    "bh_2d_udm_gtest": (BH, "gtest:Fabric2DUDMModeFixture.TestUDMFabricUnicastWriteEast", {}),
    "bh_trim_gtest": (BH, "gtest:ChannelTrimming*", {}),
    "glx_1d": (GLX, test_yaml("Linear"), {}),
    "glx_2d": (GLX, test_yaml("Mesh"), {}),
    "glx_ring": (GLX, test_yaml("Ring"), {}),
    "glx_torus": (GLX, test_yaml("Torus", TORUS_XY), {}),
    "glx_express": (GLX, test_yaml("Torus", TORUS_XY), EXPRESS_8X4),
    "glx_express_1erisc": (GLX, test_yaml("Torus", TORUS_XY), {**EXPRESS_8X4, **ONE}),
    "bh_vc1": (BH, VC1, {}),
    "bh_vc1_1erisc": (BH, VC1, ONE),
    "bh_vc1_vc2": (BH, VC1, VC2_ENV),
}

# Configurations main's router supports that the new router had never built (PARITY.md section 3).
NEW_CASES = {
    "glx2_mesh": (DUAL_GLX, test_yaml("Mesh"), {}),
    "glx2_mesh_novc": (DUAL_GLX, test_yaml("Mesh"), NOVC),
    "glx2_mesh_novc_1erisc": (DUAL_GLX, test_yaml("Mesh"), {**NOVC, **ONE}),
    "glx3_mesh": (GLX_SPLIT3, test_yaml("Mesh"), {}),
    "glx3_passthrough": (GLX_SPLIT3, PASS_THROUGH_YAML, PASS),
    "glx3_torus_yx": (GLX_TORUS_YX, HETERO_YAML, {}),
    "glx_express_ew": (GLX, test_yaml("Torus", TORUS_XY), EXPRESS_EW),
    "glx_1x32_line": (GLX, test_yaml("Linear"), GLX_1X32),
    "glx_1x32_ring": (GLX, test_yaml("Ring"), GLX_1X32_RING),
    # The Galaxy device rows' own MGD (CI's "2x2 multi-mesh sanity (inter-mesh over Z)"), with and without the
    # intermesh VC.
    "glxz_mesh": (GLX_ZRING, MM_SANITY, {}),
    "glxz_mesh_novc": (GLX_ZRING, MM_SANITY, NOVC),
    # The streaming profiler's clock sync (PROFILE_STREAMING): ERISC 0 of each synced link compiles its port.
    "bh_2d_streaming": (BH, test_yaml("Mesh"), STREAMING),
    "bh_2d_1erisc_streaming": (BH, test_yaml("Mesh"), {**STREAMING, **ONE}),
}
CASES.update(NEW_CASES)

# VC1 + VC2 at -O3 (the host's TT_METAL_FABRIC_OPT_LEVEL override), with the JIT's program-size telemetry logged at
# exit (each core type's least config-buffer headroom and its breakdown). TT_PORT_SLOT_ROW (a scratch host only) sets
# the slot row.
LOGSZ = {"TT_METAL_LOG_JIT_TELEMETRY": "1", "TT_PORT_LOG_PROGRAM_SIZE": "1"}  # the second: scratch host only
O3 = {"TT_METAL_FABRIC_OPT_LEVEL": "O3", **LOGSZ}
O3_CASES = {
    "bh_vc1_vc2_O3": (BH, VC1, {**VC2_ENV, **O3}),
    "bh_vc1_vc2_1erisc_O3": (BH, VC1, {**VC2_ENV, **ONE, **O3}),
    "glx2_mesh_vc2_O3": (DUAL_GLX, test_yaml("Mesh"), {**VC2_ENV, **O3}),
    "glx3_mesh_vc2_O3": (GLX_SPLIT3, test_yaml("Mesh"), {**VC2_ENV, **O3}),
    "glx3_passthrough_vc2_O3": (GLX_SPLIT3, PASS_THROUGH_YAML, {**PASS, **VC2_ENV, **O3}),
    "glx3_torus_yx_vc2_O3": (GLX_TORUS_YX, HETERO_YAML, {**VC2_ENV, **O3}),
    "glxz_mesh_vc2_O3": (GLX_ZRING, MM_SANITY, {**VC2_ENV, **O3}),
    "glxz_mesh_vc2_1erisc_O3": (GLX_ZRING, MM_SANITY, {**VC2_ENV, **ONE, **O3}),
}
CASES.update(O3_CASES)
# The dynamic allocator's three-VC row for the P150 split (17,8,16,5,4 + 3,2); needs TT_COV_TREE's scratch host.
ROW = {"TT_PORT_SLOT_ROW": "17,8,16,5,4,3,2"}
CASES.update({
    "bh_vc1_vc2_O3_row": (BH, VC1, {**VC2_ENV, **O3, **ROW}),
    "bh_vc1_vc2_1erisc_O3_row": (BH, VC1, {**VC2_ENV, **ONE, **O3, **ROW}),
    "bh_vc1_vc2_Os_row": (BH, VC1, {**VC2_ENV, **ROW, **LOGSZ}),
    "bh_vc1_vc2_Os_log": (BH, VC1, {**VC2_ENV, **LOGSZ}),
    "bh_vc1_vc2_1erisc_Os_log": (BH, VC1, {**VC2_ENV, **ONE, **LOGSZ}),
    "glx2_mesh_vc2_Os": (DUAL_GLX, test_yaml("Mesh"), {**VC2_ENV, **LOGSZ}),
    "glx3_mesh_vc2_Os": (GLX_SPLIT3, test_yaml("Mesh"), {**VC2_ENV, **LOGSZ}),
    "glx3_passthrough_vc2_Os": (GLX_SPLIT3, PASS_THROUGH_YAML, {**PASS, **VC2_ENV, **LOGSZ}),
    "glxz_mesh_vc2_Os": (GLX_ZRING, MM_SANITY, {**VC2_ENV, **LOGSZ}),
    "glx3_torus_yx_vc2_Os": (GLX_TORUS_YX, HETERO_YAML, {**VC2_ENV, **LOGSZ}),
    # CI's G0-mm-2x8 pass-through line (main: 26,720 B, 96 B over): VC2 off as CI runs it (the host then picks -O3).
    "glx3_2x8_passthrough": (GLX_SPLIT3_2X8, PASS_THROUGH_YAML, {**PASS, **LOGSZ}),
    "glx3_2x8_passthrough_Os": (GLX_SPLIT3_2X8, PASS_THROUGH_YAML, {**PASS, **LOGSZ, "TT_METAL_FABRIC_OPT_LEVEL": "Os"}),
    "glx3_2x8_passthrough_vc2_Os": (GLX_SPLIT3_2X8, PASS_THROUGH_YAML, {**PASS, **VC2_ENV, **LOGSZ}),
    "glx3_2x8_passthrough_vc2_O3": (GLX_SPLIT3_2X8, PASS_THROUGH_YAML, {**PASS, **VC2_ENV, **O3}),
    # The multi-mesh configs as CI runs them (VC2 off: the host picks -O3), program sizes logged.
    "glx3_2x8_mesh_sz": (GLX_SPLIT3_2X8, test_yaml("Mesh"), LOGSZ),
    "glx3_mesh_sz": (GLX_SPLIT3, test_yaml("Mesh"), LOGSZ),
    "glx3_passthrough_sz": (GLX_SPLIT3, PASS_THROUGH_YAML, {**PASS, **LOGSZ}),
    "glx2_mesh_sz": (DUAL_GLX, test_yaml("Mesh"), LOGSZ),
    "glxz_mesh_sz": (GLX_ZRING, MM_SANITY, LOGSZ),
    "glx3_torus_yx_sz": (GLX_TORUS_YX, HETERO_YAML, LOGSZ),
    "bh_vc1_sz": (BH, VC1, LOGSZ),
})

# The CI multi-mesh configs as CI runs them (galaxy_e2e_tests.yaml; VC2 off), program sizes logged: ci_<cfg> at the
# host's own opt level, ci_<cfg>_Os forced -Os. Run with TT_COV_TREE=<scratch host>; for main's sizes, a scratch host
# at the base commit (the same case names, its own COV dir).
STAB_YAML = "file:" + TREE + "/tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_multi_mesh_stability_short_running.yaml"
TRIM_MM_YAML = "file:" + TREE + "/tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_multi_mesh_channel_trimming.yaml"
CI_CFG = {
    "z": (GLX_ZRING, MM_SANITY, {}),
    "nsew": (MGD((None, CUSTOM_MGD + "single_galaxy_2x2_ring_4stage_ring_mesh_graph_descriptor.textproto", 4)), MM_SANITY, {}),
    "trim_z": (GLX_ZRING, TRIM_MM_YAML, {}),
    "trim_nsew": (MGD((None, CUSTOM_MGD + "single_galaxy_2x2_ring_4stage_ring_mesh_graph_descriptor.textproto", 4)), TRIM_MM_YAML, {}),
    "het_bigmesh": (MGD((None, CUSTOM_MGD + "single_galaxy_bigmesh_multirank_combo_mesh_graph_descriptor.textproto", 4)), HETERO_YAML, {}),
    "het_combo": (MGD((None, CUSTOM_MGD + "single_galaxy_per_mesh_config_combo_mesh_graph_descriptor.textproto", 4)), HETERO_YAML, {}),
    "het_torus": (GLX_TORUS_YX, HETERO_YAML, {}),
    "het_2x4": (MGD((None, CUSTOM_MGD + "single_galaxy_split_2x4_per_mesh_mesh_graph_descriptor.textproto", 3)), HETERO_YAML, {}),
    "4x4": (GLX_SPLIT3, STAB_YAML, {}),
    "4x4_pt": (GLX_SPLIT3, PASS_THROUGH_YAML, PASS),
    "2x8": (GLX_SPLIT3_2X8, STAB_YAML, {}),
    "2x8_pt": (GLX_SPLIT3_2X8, PASS_THROUGH_YAML, PASS),
    "dual": (DUAL_GLX, test_yaml("Mesh"), {}),
    "p150_vc1": (BH, VC1, {}),
}
for _k, (_d, _y, _e) in CI_CFG.items():
    CASES["ci_" + _k] = (_d, _y, {**_e, **LOGSZ})
    CASES["ci_" + _k + "_Os"] = (_d, _y, {**_e, **LOGSZ, "TT_METAL_FABRIC_OPT_LEVEL": "Os"})


CMD_RE = re.compile(r"g\+\+ (compile|link) cmd: (.*?)\s*\(build\.cpp:\d+\)\s*$")


def ensure_root():
    if os.path.isdir(ROOT):
        return
    os.makedirs(os.path.join(ROOT, "generated"))
    for e in os.listdir(TREE):
        if e not in ("generated", ".git", ".scratch"):
            os.symlink(os.path.join(TREE, e), os.path.join(ROOT, e))


def run_mock(name):
    desc, yaml, extra = CASES[name]
    d = os.path.join(COV, "cases", name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TT_METAL_", "TT_FABRIC_", "TT_MESH_"))}
    env.update(BASE_ENV)
    env.update(json.loads(os.environ.get("TT_COV_EXTRA_ENV", "{}")))  # e.g. the study's builder settings for its tree
    env.update(extra)
    env.update({"TT_METAL_RUNTIME_ROOT": ROOT, "TT_METAL_HOME": ROOT, "TT_METAL_LOGS_PATH": d,
                "TT_METAL_CACHE": os.path.join(d, "cache"), "TT_METAL_MOCK_CLUSTER_DESC_PATH": desc})
    pre = ["timeout", "1500", "nice", "-n", "10", "taskset", "-c", CPUS]
    if yaml.startswith("file:"):
        yaml_path = yaml[len("file:"):]
    else:
        yaml_path = os.path.join(d, "test.yaml")
        if yaml != VC1 and not yaml.startswith("gtest:"):
            open(yaml_path, "w").write(yaml)
    if isinstance(desc, MGD):
        mapping, mgd, ranks = desc
        env.pop("TT_METAL_MOCK_CLUSTER_DESC_PATH")
        env["OMPI_MCA_mpi_yield_when_idle"] = "1"
        if mapping is None:
            mapping = os.path.join(d, "mock_mapping.yaml")
            open(mapping, "w").write("rank_to_cluster_mock_cluster_desc:\n" +
                                     "".join(f'  {r}: "{GLX}"\n' for r in range(ranks)))
        cmd = pre + [PY, TREE + "/ttnn/ttnn/distributed/ttrun.py", "--mesh-graph-descriptor", mgd,
                     "--mock-cluster-rank-binding", mapping, "--mpi-args", "--oversubscribe", "--force-rediscovery",
                     BIN, "--test_config", yaml_path]
    elif yaml == VC1:
        env.pop("TT_METAL_MOCK_CLUSTER_DESC_PATH")
        mapping = os.path.join(d, "mock_mapping.yaml")
        open(mapping, "w").write(f'rank_to_cluster_mock_cluster_desc:\n  0: "{desc}"\n  1: "{desc}"\n')
        cmd = pre + [PY, TREE + "/ttnn/ttnn/distributed/ttrun.py", "--mock-cluster-rank-binding", mapping,
                     "--rank-binding", VC1_DIR + "/p150x8_2x2_split_rank_bindings.yaml", "--mpi-args", "--oversubscribe",
                     BIN, "--test_config", VC1_DIR + "/vc1_sanity.yaml"]
    elif yaml.startswith("gtest:"):
        cmd = pre + [GTEST_BIN, "--gtest_filter=" + yaml[len("gtest:"):]]
    else:
        cmd = pre + [BIN, "--test_config", yaml_path]
    with open(os.path.join(d, "mock.log"), "w") as log:
        r = subprocess.run(cmd, cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT,
                           preexec_fn=lambda: __import__("resource").setrlimit(
                               __import__("resource").RLIMIT_CORE, (0, 0)))
    db = collect_case(name)
    routers = sum(1 for k in db if "/fabric_erisc_router/" in k)
    print(f"{name}: exit {r.returncode}, {len(db)} builds logged ({routers} router ERISC builds)", flush=True)
    if os.environ.get("TT_COV_DROP_CACHE"):  # size runs: keep the log only (disk)
        shutil.rmtree(os.path.join(d, "cache"), ignore_errors=True)


def wellformed(kind, cmd):
    """A logged command no other process's output cut into (two MPI ranks write one log)."""
    if "<stdout>" in cmd or "<stderr>" in cmd or "\x1b" in cmd or cmd.count("riscv-tt-elf-g++") != 1:
        return False
    a = cmd.split(" ")
    if len(set(re.findall(r"/kernels/[^/\s]+/(\d+)/", cmd))) > 1:
        return False
    if kind == "compile":
        return (a[0].endswith("riscv-tt-elf-g++") and os.path.exists(a[0]) and "//" not in a[0] and a.count("-c") == 1 and a.count("-MF") == 1 and a.count("-o") == 1
                and sum(x.startswith("-DFULL_KERNEL_NAME=") for x in a) == 1 and a[-1].endswith(".d")
                and any(x.startswith("-mcpu=") for x in a))
    return cmd.startswith("cd ") and cmd.count(" -o ") == 1 and cmd.rstrip().endswith(".elf")


def build_key(cmd):
    """The JIT output directory of a command, relative to the cache: kernels/<name>/<hash>/<proc> or firmware/<proc>."""
    m = re.search(r"/(kernels/[^/\s]+/\d+/\w+)/", cmd) or re.search(r"/(firmware/\w+)/", cmd)
    return m.group(1) if m else None


def collect_case(name):
    d = os.path.join(COV, "cases", name)
    db = {}
    for line in open(os.path.join(d, "mock.log"), errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = CMD_RE.search(line)
        if not m:
            continue
        kind, cmd = m.group(1), m.group(2)
        if not wellformed(kind, cmd):
            continue
        k = build_key(cmd)
        if k is None:
            continue
        # One command of each kind per build: two ranks log the same build twice (different temporary names).
        db.setdefault(k, {}).setdefault(kind, [cmd])
    # Builds with a logged compile and link only: the JIT logs firmware links but not their compiles, so firmware is
    # checked separately (the host build's precompiled firmware).
    db = {k: v for k, v in db.items() if "link" in v and "compile" in v}
    json.dump(db, open(os.path.join(d, "cmds.json"), "w"), indent=0)
    return db


def cache_dir_of(cmd, key):
    m = re.search(r"(\S*/" + re.escape(key) + r")/", cmd)
    return m.group(1)


EXTRA_DEFINES = []  # build --defines "-DNAME=VALUE ...": added to every compile (port_tune: the sweep's arms)
ALT_TREE = None
ALT_ROOT = None  # build --alt-tree: the logged runtime root replaced by a root of symlinks into another tree


def alt_root_for(tree):
    """A runtime root of symlinks into `tree`, its path as long as ROOT's (no path length changes in the build).
    The name is a stable digest of the tree path (str hash() is salted per process: an earlier version reused another
    tree's root), and an existing root is rebuilt unless every link points into `tree`."""
    import hashlib
    alt = ROOT[:-4] + "rtx" + str(int(hashlib.md5(tree.encode()).hexdigest(), 16) % 10)
    if os.path.isdir(alt):
        links = [os.path.join(alt, e) for e in os.listdir(alt) if os.path.islink(os.path.join(alt, e))]
        if not links or any(not os.readlink(l).startswith(tree.rstrip("/") + "/") for l in links):
            shutil.rmtree(alt)
    if not os.path.isdir(alt):
        os.makedirs(os.path.join(alt, "generated"))
        for e in os.listdir(tree):
            if e not in ("generated", ".git", ".scratch"):
                os.symlink(os.path.join(tree, e), os.path.join(alt, e))
    assert os.readlink(os.path.join(alt, "tt_metal")) == os.path.join(tree, "tt_metal"), alt
    print(f"alt root {alt}: tt_metal -> {os.readlink(os.path.join(alt, 'tt_metal'))}", flush=True)
    return alt


def alt_cmds(case, key, cmds, out):
    """The build's commands against ALT_ROOT: its generated headers copied (paths rewritten) next to a work directory
    of its own, the root replaced in the commands; the compile then runs in that work directory."""
    src = cache_dir_of(cmds["link"][0], key)
    gen = os.path.join(out, case + "__" + key.replace("/", "__"), "gen")
    os.makedirs(os.path.join(gen, os.path.basename(src)), exist_ok=True)
    for f in glob.glob(os.path.dirname(src) + "/*.h") + glob.glob(os.path.dirname(src) + "/*.hpp"):
        open(os.path.join(gen, os.path.basename(f)), "w").write(open(f).read().replace(ROOT, ALT_ROOT))
    new_src = os.path.join(gen, os.path.basename(src))
    sub = lambda c: c.replace(ROOT + "/", ALT_ROOT + "/")
    return {k: [sub(c) for c in v] for k, v in cmds.items()} | {"_cwd": new_src}


def build_one(case, key, cmds, out):
    """Rebuild one JIT build (all its compiles, then the link) into out/<case>__<key flattened>/."""
    try:
        if ALT_ROOT:
            cmds = alt_cmds(case, key, cmds, out)
        return build_one_(case, key, cmds, out)
    except Exception as e:  # a command the log garbled: report it as this build's failure
        return os.path.join(out, case + "__" + key.replace("/", "__")), f"build error: {e}"


def build_one_(case, key, cmds, out):
    wd = os.path.join(out, case + "__" + key.replace("/", "__"))
    os.makedirs(wd, exist_ok=True)
    # Compiles run in the JIT's own directory: its -I. and -I.. name the generated per-build headers (named args,
    # kernel includes, defines) in the cache, which are read in place and not modified (an --alt-tree build: copies).
    src = cmds.get("_cwd") or cache_dir_of(cmds["link"][0], key)
    objs = {}
    for c in cmds.get("compile", []):
        a = c.split(" ")
        o = []
        i = 0
        while i < len(a):
            x = a[i]
            if x == "-o":
                obj = os.path.join(wd, os.path.basename(a[i + 1]))
                objs[a[i + 1]] = obj
                o += ["-o", obj]
                i += 2
                continue
            if x == "-MF":
                o += ["-MF", os.path.join(wd, os.path.basename(a[i + 1]))]
                i += 2
                continue
            o.append(x)
            i += 1
        if EXTRA_DEFINES:  # build --defines: an arm's kernel defines, as the host adds them (port_tune.patch)
            o = o[:1] + EXTRA_DEFINES + o[1:]
        r = subprocess.run(o, cwd=src, capture_output=True, text=True)
        if r.returncode:
            return wd, "compile failed:\n" + r.stderr[-4000:]
    link = cmds["link"][0]
    for old, new in objs.items():
        link = link.replace(old, new)
    if len(objs) == 1:
        # Two ranks can log a build's compile and link from different processes (different temporary object names):
        # the link's object in the build's directory is this rebuild's.
        link = re.sub(r"\S+/" + re.escape(key) + r"/[^/\s]+\.o(?=\s)", next(iter(objs.values())), link)
    lm = re.search(r"\s-o\s+(\S+\.elf)", link)
    elf = os.path.join(wd, os.path.basename(lm.group(1)).split(".")[0] + ".elf")
    link = link[: lm.start(1)] + elf + link[lm.end(1):]
    # Objects the link names that this rebuild did not compile (none expected) stay the cache's.
    r = subprocess.run(link, shell=True, capture_output=True, text=True)
    if r.returncode:
        return wd, "link failed:\n" + r.stderr[-4000:]
    return wd, None


def tree_state():
    tree = ALT_TREE or TREE
    head = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", tree, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    return head, dirty


def source_identity():
    """What a build compiles: the tree the runtime root actually links to, its HEAD and a digest of its uncommitted
    tracked changes (two builds with the same HEAD and digest compiled the same sources)."""
    import hashlib
    root = ALT_ROOT or ROOT
    tree = os.path.dirname(os.path.realpath(os.path.join(root, "tt_metal")))
    head = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "-C", tree, "diff", "HEAD", "--", "tt_metal", "ttnn", "tt_stl"], capture_output=True,
                          text=True).stdout
    return {"tree": tree, "head": head, "diff_md5": hashlib.md5(diff.encode()).hexdigest()}


def build(args):
    out = os.path.join(COV, "builds", args.name)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    head, dirty = tree_state()
    names = args.cases.split(",") if args.cases else sorted(os.listdir(os.path.join(COV, "cases")))
    jobs, seen = [], set()
    for case in names:
        p = os.path.join(COV, "cases", case, "cmds.json")
        if not os.path.exists(p):
            continue
        for key, cmds in sorted(json.load(open(p)).items()):
            if args.router_only and "/fabric_erisc_router/" not in key:
                continue
            # One rebuild per JIT build: the hash directory names the build's inputs. Firmware per case (the cache's
            # firmware directory carries no hash, and cases may differ in firmware build options).
            ident = key if key.startswith("kernels/") else case + ":" + key
            if ident in seen:
                continue
            seen.add(ident)
            jobs.append((case, key, cmds))
    print(f"{len(jobs)} builds over {len(names)} cases; HEAD {head[:11]}{' (tracked changes!)' if dirty else ''}")
    res = {}
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(build_one, c, k, cm, out): (c, k) for c, k, cm in jobs}
        for f in concurrent.futures.as_completed(futs):
            c, k = futs[f]
            wd, err = f.result()
            res[c + "__" + k] = {"case": c, "key": k, "ok": err is None, "err": first_error(err)}
    ident = source_identity()
    print(f"compiled {ident['tree']} at {ident['head'][:11]} (uncommitted diff md5 {ident['diff_md5'][:8]})")
    ident["defines"] = EXTRA_DEFINES
    json.dump({"head": head, "dirty": dirty, "source": ident, "results": res}, open(os.path.join(out, "results.json"), "w"),
              indent=1)
    bad = sum(not r["ok"] for r in res.values())
    print(f"built {len(res) - bad}/{len(res)} into {out}")


def first_error(err):
    if not err:
        return None
    for line in err.splitlines():
        if "error:" in line:
            return re.sub(r"^\S*/", "", line.strip())[:300]
    return err.strip().splitlines()[-1][:300] if err.strip() else "unknown"


def image(elf):
    """The ELF's loadable bytes: each allocated section with contents (readelf), its address and its bytes (objcopy).
    Per section, not one objcopy -O binary of the whole file: firmware places sections far apart, which would make a
    flat image gigabytes long."""
    import tempfile
    rd = OC.replace("objcopy", "readelf")
    secs = []
    for line in subprocess.run([rd, "-S", "-W", elf], capture_output=True, text=True, check=True).stdout.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9a-f]+)\s+[0-9a-f]+\s+([0-9a-f]+)\s+\S+\s+(\S*)", line)
        if m and "A" in m.group(5) and m.group(2) != "NOBITS" and int(m.group(4), 16) > 0:
            secs.append((m.group(1), m.group(3)))
    out = b""
    with tempfile.TemporaryDirectory() as td:
        for name, addr in secs:
            f = os.path.join(td, "s.bin")
            r = subprocess.run([OC, "-O", "binary", "-j", name, elf, f], capture_output=True)
            if r.returncode:
                raise RuntimeError(f"objcopy failed on {elf} {name}: {r.stderr.decode()[:200]}")
            out += name.encode() + b"@" + addr.encode() + b":" + open(f, "rb").read()
    return out


def elfs_of(name):
    if name.startswith("jit:"):
        case = name[4:]
        db = json.load(open(os.path.join(COV, "cases", case, "cmds.json")))
        out = {}
        for key, cmds in db.items():
            # The JIT links to a temporary name, then renames it to <proc>.elf.
            m = re.search(r"\s-o\s+(\S+\.elf)", cmds["link"][0])
            elf = m and os.path.join(os.path.dirname(m.group(1)), os.path.basename(m.group(1)).split(".")[0] + ".elf")
            if elf and os.path.exists(elf):
                out[case + "__" + key.replace("/", "__")] = elf
        return out
    b = os.path.join(COV, "builds", name)
    out = {}
    for e in glob.glob(b + "/*/*.elf"):
        out[os.path.basename(os.path.dirname(e))] = e
    return out


def build_source(name):
    try:
        return json.load(open(os.path.join(COV, "builds", name, "results.json"))).get("source")
    except (OSError, ValueError):
        return None


def cmp(args):
    sa, sb = build_source(args.a), build_source(args.b)
    for n, s_ in ((args.a, sa), (args.b, sb)):
        if s_:
            print(f"{n}: {s_['tree']} at {s_['head'][:11]} (diff md5 {s_['diff_md5'][:8]})")
    if sa and sb and sa["head"] == sb["head"] and sa["diff_md5"] == sb["diff_md5"] and not args.same_ok:
        print(f"REFUSED: {args.a} and {args.b} compiled the same sources (pass --same-ok to compare anyway)")
        sys.exit(2)
    a, b = elfs_of(args.a), elfs_of(args.b)
    shared = sorted(set(a) & set(b))
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        same = list(ex.map(lambda k: image(a[k]) == image(b[k]), shared))
    per = collections.defaultdict(lambda: [0, 0])
    diff = []
    for k, s in zip(shared, same):
        kind = "router" if "__fabric_erisc_router__" in k else ("firmware" if "__firmware__" in k else "other")
        per[(k.split("__")[0], kind)][0 if s else 1] += 1
        if not s:
            diff.append(k)
    for (case, kind), (i, d) in sorted(per.items()):
        print(f"{case:24s} {kind:9s} identical {i:5d}  differ {d:4d}")
    tot = collections.Counter()
    for (case, kind), (i, d) in per.items():
        tot[kind + " identical"] += i
        tot[kind + " differ"] += d
    print("totals:", dict(sorted(tot.items())))
    print(f"shared {len(shared)}  identical {len(shared) - len(diff)}  differ {len(diff)}  "
          f"only in {args.a} {len(set(a) - set(b))}  only in {args.b} {len(set(b) - set(a))}")
    for k in diff[:30]:
        print("  differs:", k)
    sys.exit(0 if not diff and set(a) == set(b) else 1)


def fw(args):
    """Regenerate the precompiled firmware from the live tree (the host build's precompile_fw, into
    tt_metal/pre-compiled; its JIT cache under COV/fw/cache_NAME), keep a copy as COV/fw/NAME and compare every firmware
    ELF's loadable bytes with COV/fw/base (the host build's own output at the base commit)."""
    head, dirty = tree_state()
    print(f"HEAD {head[:11]}{' (tracked changes!)' if dirty else ''}")
    cache = os.path.join(COV, "fw", "cache_" + args.name)
    shutil.rmtree(cache, ignore_errors=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TT_METAL_", "TT_FABRIC_"))}
    env["TT_METAL_CACHE"] = cache
    r = subprocess.run(["nice", "-n", "10", TREE + "/build/tools/precompile_fw"], cwd=TREE, env=env,
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit("precompile_fw failed:\n" + (r.stdout + r.stderr)[-3000:])
    dst = os.path.join(COV, "fw", args.name)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(TREE + "/tt_metal/pre-compiled", dst)
    base = os.path.join(COV, "fw", "base")
    rel = lambda root: {os.path.relpath(e, root) for e in glob.glob(root + "/*/**/*.elf", recursive=True)}
    a, b = rel(base), rel(dst)
    shared = sorted(a & b)
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        same = list(ex.map(lambda k: image(os.path.join(base, k)) == image(os.path.join(dst, k)), shared))
    diff = [k for k, s in zip(shared, same) if not s]
    print(f"firmware ELFs shared {len(shared)}  identical {len(shared) - len(diff)}  differ {len(diff)}  "
          f"only in base {len(a - b)}  only in {args.name} {len(b - a)}")
    for k in diff[:20]:
        print("  differs:", k)
    sys.exit(0 if not diff and a == b else 1)


def report(args):
    r = json.load(open(os.path.join(COV, "builds", args.name, "results.json")))
    print("HEAD", r["head"], "dirty" if r["dirty"] else "clean")
    per = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for v in r["results"].values():
        kind = "router" if "/fabric_erisc_router/" in v["key"] else ("firmware" if v["key"].startswith("firmware/") else "other")
        e = per[(v["case"], kind)]
        e[0] += v["ok"]
        e[1] += 1
        if v["err"]:
            e[2][v["err"]] += 1
    for (case, kind), (ok, n, errs) in sorted(per.items()):
        print(f"{case:24s} {kind:9s} {ok}/{n}")
        for e, c in errs.most_common(3):
            print(f"    x{c} {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("mock")
    m.add_argument("cases", nargs="*")
    m.add_argument("--jobs", type=int, default=1)
    c = sp.add_parser("collect")
    c.add_argument("cases", nargs="*")
    b = sp.add_parser("build")
    b.add_argument("--name", required=True)
    b.add_argument("--cases", default="")
    b.add_argument("--router-only", action="store_true")
    b.add_argument("--jobs", type=int, default=32)
    b.add_argument("--alt-tree", default="", help="build another tree's sources (a root of symlinks into it)")
    b.add_argument("--defines", default="", help="extra -D flags for every compile (an arm's kernel defines)")
    k = sp.add_parser("cmp")
    k.add_argument("a")
    k.add_argument("b")
    k.add_argument("--same-ok", action="store_true", help="compare two builds of the same sources (a determinism check)")
    r = sp.add_parser("report")
    r.add_argument("name")
    f = sp.add_parser("fw")
    f.add_argument("name")
    a = ap.parse_args()
    if a.cmd == "mock":
        ensure_root()
        names = a.cases or list(CASES)
        with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(run_mock, names))
    elif a.cmd == "collect":
        for c in a.cases or sorted(os.listdir(os.path.join(COV, "cases"))):
            print(c, len(collect_case(c)), "builds")
    elif a.cmd == "build":
        global ALT_ROOT, ALT_TREE, EXTRA_DEFINES
        EXTRA_DEFINES = a.defines.split()
        assert all(x.startswith("-D") for x in EXTRA_DEFINES), a.defines
        if a.alt_tree:
            ALT_TREE = os.path.abspath(a.alt_tree)
            ALT_ROOT = alt_root_for(ALT_TREE)
            print("alt root", ALT_ROOT, "->", a.alt_tree)
        build(a)
    elif a.cmd == "cmp":
        cmp(a)
    elif a.cmd == "fw":
        fw(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
