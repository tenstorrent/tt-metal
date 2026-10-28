#!/usr/bin/env python3
"""argdump.py: which router runtime args vary between routers (shape inventory, RESULTS §98).

  run [CASE ...]     CPU-only mock-cluster runs (no device) with TT_FABRIC_EXP_DUMP_ARGS: the host writes one JSON line
                     per router ERISC with its CT args, named CT args, defines and runtime args (compute_mesh_router_builder)
  report [CASE ...]  per case: routers, distinct binaries (ERISC + CT args + named CT args + defines), and for each named
                     runtime-arg field the number of distinct values over all routers and whether it is constant within
                     every binary; then how many binaries there would be if the varying fields were compile-time

Runtime-arg layout (StaticSizedChannelConnectionWriterAdapter::pack_inbound_channel_rt_args, get_runtime_args):
  10 sender-channel connection semaphore ids/addresses, 10 downstream buffer-index semaphore ids, then per VC
  (VC0, and VC1 when the router has a VC1 receiver):
    2D: has-downstream mask m, then for each set bit of m (dense): buffer base; then packed NoC x, packed NoC y; then
        dense registration addresses, location-info addresses, buffer-index semaphore addresses
    1D: has-connection, buffer base, NoC x, NoC y, registration, location-info, buffer-index semaphore address
  then 10 downstream teardown semaphores, then the relay (UDM) block: 0 when not connected.
Cases run kernel v1 (main's router): the args come from the host builder, which v3 shares.
"""
import argparse, collections, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coverage as cov  # noqa: E402

TREE = cov.TREE
OUT = os.environ.get("TT_ARGDUMP_DIR", "/home/aagarwal/.tt-metal-v3r-builds/argdump")
DESC = TREE + "/tt_metal/third_party/tt-cluster-descriptors/blackhole"
GLX = DESC + "/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml"
GLX_XYZ = DESC + "/bh_galaxy_xyz_cluster_desc/bh_galaxy_xyz_cluster_desc.yaml"
MGD = TREE + "/tests/tt_metal/tt_fabric/custom_mesh_descriptors"
TORUS = "      torus_config: XY\n"

# name: (cluster descriptor, topology, extra yaml under fabric_setup, extra env)
CASES = {
    "glx_mesh": (GLX, "Mesh", "", {}),
    "glx_linear": (GLX, "Linear", "", {}),
    "glx_ring": (GLX, "Ring", "", {}),
    "glx_torus": (GLX, "Torus", TORUS, {}),
    "glx_express": (GLX, "Torus", TORUS, {"TT_MESH_GRAPH_DESC_PATH": MGD + "/express_links_8x4_mesh_graph_descriptor.textproto"}),
    "glx_z": (GLX_XYZ, "Mesh", "", {"TT_MESH_GRAPH_DESC_PATH": MGD + "/bh_galaxy_4x4_z_mesh_graph_descriptor.textproto"}),
    "p150_mesh": (cov.BH, "Mesh", "", {}),
}
NUM_MAX_SENDER_CHANNELS = 10


def run(name):
    desc, topo, extra_yaml, extra_env = CASES[name]
    d = os.path.join(OUT, name)
    subprocess.run(["rm", "-rf", d], check=True)
    os.makedirs(d)
    open(os.path.join(d, "test.yaml"), "w").write(cov.test_yaml(topo, extra_yaml))
    env = {k: v for k, v in os.environ.items() if k not in ("TT_METAL_SLOW_DISPATCH_MODE",)}
    env.update({"TT_FABRIC_KERNEL_VERSION": "1", "TT_FABRIC_EXP_FLAGS": "0", "TT_FABRIC_NO_BW_TELEMETRY": "1",
                "TT_METAL_MOCK_CLUSTER_DESC_PATH": desc, "TT_FABRIC_EXP_DUMP_ARGS": os.path.join(d, "args.jsonl"),
                "TT_METAL_RUNTIME_ROOT": cov.ROOT, "TT_METAL_HOME": cov.ROOT, "TT_METAL_LOGS_PATH": d,
                "TT_METAL_CACHE": os.path.join(d, "cache")})
    env.update(extra_env)
    cov.ensure_root()
    cmd = ["timeout", "1200", "nice", "-n", "10", "taskset", "-c", cov.CPUS, cov.BIN, "--test_config",
           os.path.join(d, "test.yaml")]
    with open(os.path.join(d, "mock.log"), "w") as log:
        rc = subprocess.run(cmd, env=env, cwd=cov.ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    n = sum(1 for _ in open(os.path.join(d, "args.jsonl"))) if os.path.exists(os.path.join(d, "args.jsonl")) else 0
    print(f"{name}: exit {rc}, {n} router ERISCs dumped")


def parse_rt(rt, is_2d):
    """Runtime args -> [(field, value)], fields named by role (downstream fields by compact slot)."""
    f = [(f"conn_sem[{i}]", rt[i]) for i in range(10)] + [(f"ds_bufidx_sem_id[{i}]", rt[10 + i]) for i in range(10)]
    p = 20
    rest_after_vcs = NUM_MAX_SENDER_CHANNELS + 1  # teardown semaphores + relay flag (UDM relay not expected here)
    for vc in (0, 1):
        if len(rt) - p <= rest_after_vcs:
            break
        if is_2d:
            m = rt[p]
            slots = [k for k in range(4) if m >> k & 1]
            f.append((f"vc{vc}.mask", m))
            p += 1
            for k in slots:
                f.append((f"vc{vc}.buffer_base[{k}]", rt[p])); p += 1
            f.append((f"vc{vc}.noc_x_packed", rt[p])); f.append((f"vc{vc}.noc_y_packed", rt[p + 1])); p += 2
            for name in ("registration_addr", "location_info_addr", "bufidx_sem_addr"):
                for k in slots:
                    f.append((f"vc{vc}.{name}[{k}]", rt[p])); p += 1
        else:
            for name in ("has_connection", "buffer_base", "noc_x", "noc_y", "registration_addr", "location_info_addr",
                         "bufidx_sem_addr"):
                f.append((f"vc{vc}.{name}", rt[p])); p += 1
    for i in range(NUM_MAX_SENDER_CHANNELS):
        f.append((f"teardown_sem[{i}]", rt[p + i]))
    p += NUM_MAX_SENDER_CHANNELS
    f.append(("relay", tuple(rt[p:])))
    return f


def report(name):
    path = os.path.join(OUT, name, "args.jsonl")
    if not os.path.exists(path):
        print(f"\n## {name}: no dump")
        return
    rows = [json.loads(l) for l in open(path)]
    # A run builds the fabric once per test; keep one entry per router ERISC (the last).
    by_router = {}
    for r in rows:
        by_router[(r["device"], r["eth_chan"], r["risc"])] = r
    rows = list(by_router.values())
    binary = lambda r: json.dumps([r["risc"], r["ct_args"], r["named_ct_args"], r["defines"]], sort_keys=True)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[binary(r)].append(r)
    is_2d = lambda r: "FABRIC_2D" in r["defines"] or r["named_ct_args"].get("IS_2D_FABRIC", 0) == 1
    fields = collections.OrderedDict()
    parsed = {}
    for r in rows:
        pf = parse_rt(r["rt_args"], is_2d(r))
        parsed[id(r)] = dict(pf)
        for k, _ in pf:
            fields.setdefault(k, None)
    devices = len({r["device"] for r in rows})
    print(f"\n## {name}: {devices} chips, {len(rows)} router ERISCs, {len(groups)} distinct binaries"
          f" (E0 {len({binary(r) for r in rows if r['risc'] == 0})}, E1 {len({binary(r) for r in rows if r['risc'] == 1})})")
    print(f"{'field':34} {'distinct':>8} {'within binary':>14}  values")
    varying = []
    for k in fields:
        vals = [parsed[id(r)].get(k) for r in rows]
        distinct = sorted({str(v) for v in vals})
        within = all(len({str(parsed[id(r)].get(k)) for r in g}) == 1 for g in groups.values())
        if len(distinct) > 1 and not within:
            varying.append(k)
        shown = ", ".join(distinct[:4]) + (" ..." if len(distinct) > 4 else "")
        print(f"{k:34} {len(distinct):>8} {'constant' if within else 'VARIES':>14}  {shown}")
    if varying:
        split = {binary(r) + json.dumps([str(parsed[id(r)].get(k)) for k in varying]) for r in rows}
        print(f"fields that vary within a binary: {', '.join(varying)}")
        print(f"binaries if they were compile-time: {len(split)} (now {len(groups)})")
    else:
        print("every runtime-arg field is constant within each binary: all could be compile-time at no extra binaries")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "report"):
        p = sp.add_parser(c)
        p.add_argument("cases", nargs="*", default=list(CASES))
    a = ap.parse_args()
    for c in a.cases:
        run(c) if a.cmd == "run" else report(c)


if __name__ == "__main__":
    main()
