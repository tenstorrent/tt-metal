#!/usr/bin/env python3
"""alloc_mock.py: the offline half of the QB2 torus regression A/B (staged slot allocator, 07322400fd6 vs its parent
57c55f98118; coordinator 2026-11-05; local, never committed). Mock clusters only, on a09u32.

  alloc_mock.py mock ARM [CASE...]   port_tune/cov/cov.py mock runs of the cases below with ARM's host build and JIT
                                     (ARM a = /home/aagarwal/tt-metal-alloc-57c5, b = /home/aagarwal/tt-metal-alloc-0732;
                                     COV dirs /home/aagarwal/.tt-metal-alab-<arm>-cov)
  alloc_mock.py slots [CASE...]      every router build's slot counts per arm (port_tune/slot_probe.py's probe), with its
                                     direction, slot size, deadlock avoidance and bubble threshold (named CT args)
  alloc_mock.py yaml ARM YAML [ENV=V ...]
                                     mock run of one device yaml as the device job runs it (the yaml vetting)

Cases: one 2x2 of the P150x8 (TT_VISIBLE_DEVICES=0,1,2,3, cluster-descriptor chip ids, as the split harness), MGD
tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto (2x2, 2 channels), topology Torus with
torus_config XY (FABRIC_2D_TORUS_XY) or Mesh (FABRIC_2D), max_packet_size 14400 (the CI test's 14 * 1024 + 64) or the
default."""
import collections, glob, importlib.util, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
COVPY = os.path.join(HERE, "..", "..", "port_tune", "cov", "cov.py")
TREES = {"a": "/home/aagarwal/tt-metal-alloc-57c5", "b": "/home/aagarwal/tt-metal-alloc-0732"}
# ALAB_TREES="NAME=TREE ...": other arms (e.g. nofix=/home/aagarwal/tt-metal-bfix-de8a fix=/home/aagarwal/tt-metal-bfix-6a4a)
if os.environ.get("ALAB_TREES"):
    TREES = dict(x.split("=", 1) for x in os.environ["ALAB_TREES"].split())
COVS = {a: f"/home/aagarwal/.tt-metal-alab-{a}-cov" for a in TREES}
MGD = "tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"


def load_cov(arm):
    os.environ["TT_COV_TREE"], os.environ["TT_COV_DIR"] = TREES[arm], COVS[arm]
    sys.argv = ["cov"]
    spec = importlib.util.spec_from_file_location(f"cov_{arm}", COVPY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    env = {"TT_MESH_GRAPH_DESC_PATH": os.path.join(m.TREE, MGD), "TT_VISIBLE_DEVICES": "0,1,2,3"}
    torus, mpk = "      torus_config: XY\n", "      max_packet_size: 14400\n"
    m.CASES.update({
        "qb_torus_14400": (m.BH, m.test_yaml("Torus", torus + mpk), env),
        "qb_torus_4352": (m.BH, m.test_yaml("Torus", torus), env),
        "qb_mesh_14400": (m.BH, m.test_yaml("Mesh", mpk), env),
        "qb_mesh_4352": (m.BH, m.test_yaml("Mesh"), env),
    })
    return m


CASES = ["qb_torus_14400", "qb_torus_4352", "qb_mesh_14400", "qb_mesh_4352"]


def named(hdir):
    f = os.path.join(hdir, "named_ct_arg_map_generated.h")
    return dict((k, int(v)) for k, v in re.findall(r'\{"([A-Z0-9_]+)",(\d+)\}', open(f).read())) if os.path.exists(f) else {}


def slots(cases):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "port_tune"))
    import tempfile
    import slot_probe as sp
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "tail.h"), "w").write(sp.TAIL)
        ns = sp.TAIL.replace("SENDER_NUM_BUFFERS_ARRAY", "tt::tt_fabric::SENDER_NUM_BUFFERS_ARRAY")
        ns = ns.replace("RECEIVER_NUM_BUFFERS_ARRAY", "tt::tt_fabric::RECEIVER_NUM_BUFFERS_ARRAY")
        open(os.path.join(tmp, "tail_ns.h"), "w").write(ns)
        for case in cases:
            for arm in TREES:
                d = os.path.join(COVS[arm], "cases", case)
                if not os.path.exists(os.path.join(d, "cmds.json")):
                    print(f"== {case} arm {arm}: no capture")
                    continue
                cmds = json.load(open(os.path.join(d, "cmds.json")))
                rows = collections.Counter()
                for key, v in sorted(cmds.items()):
                    if "/fabric_erisc_router" not in key or not v.get("compile"):
                        continue
                    kd = glob.glob(os.path.join(d, "cache", "*", key))
                    if not kd:
                        continue
                    tx, rx = sp.probe(v["compile"][0], kd[0], tmp)
                    n = named(os.path.dirname(kd[0]))
                    rows[(key.rsplit("/", 1)[1], "EWNS"[n.get("MY_DIRECTION", 0)] if "MY_DIRECTION" in n else "?",
                          n.get("CHANNEL_BUFFER_SIZE"), n.get("ENABLE_DEADLOCK_AVOIDANCE"),
                          n.get("BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS"),
                          n.get("NUM_SENDER_CHANNELS"), tx, rx)] += 1
                print(f"== {case} arm {arm} ({os.path.basename(TREES[arm])}): {sum(rows.values())} router builds")
                print(f"   {'erisc':25} {'dir':3} {'n':>3} {'slot B':>6} {'DA':>2} {'bubble':>6} {'tx ch':>5}  TX slots / RX slots")
                for (proc, dr, size, da, bub, ntx, tx, rx), c in sorted(rows.items()):
                    print(f"   {proc:25} {dr:3} {c:3d} {size!s:>6} {da!s:>2} {bub!s:>6} {ntx!s:>5}  TX {tx:22s} RX {rx}")


def mock_yaml(arm, yaml, env):
    """One device yaml as a mock case named after it (the yaml vetting)."""
    m = load_cov(arm)
    name = "yaml_" + os.path.basename(yaml).replace(".yaml", "") + ("_" + "_".join(e.split("=")[0][-6:] for e in env) if env else "")
    extra = {"TT_MESH_GRAPH_DESC_PATH": os.path.join(m.TREE, MGD), "TT_VISIBLE_DEVICES": "0,1,2,3"}
    extra.update(dict(e.split("=", 1) for e in env))
    m.CASES[name] = (m.BH, "file:" + os.path.abspath(yaml), extra)
    m.ensure_root()
    m.run_mock(name)
    log = os.path.join(m.COV, "cases", name, "mock.log")
    t = re.sub(r"\x1b\[[0-9;]*m", "", open(log, errors="replace").read())
    for pat in (r"Running Test[^\n]*", r"TT_FATAL[^\n]*", r"TT_THROW[^\n]*", r"what\(\):[^\n]*", r"Test groups:[^\n]*"):
        hits = re.findall(pat, t)
        if hits:
            print(f"   {len(hits):4d} x {hits[0][:200]}" + (f"  ... last: {hits[-1][:160]}" if len(hits) > 1 else ""))


def main():
    what = sys.argv[1]
    if what == "mock":
        arm, cases = sys.argv[2], sys.argv[3:] or CASES
        m = load_cov(arm)
        m.ensure_root()
        for c in cases:
            m.run_mock(c)
    elif what == "slots":
        slots(sys.argv[2:] or CASES)
    elif what == "yaml":
        mock_yaml(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
