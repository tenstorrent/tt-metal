#!/usr/bin/env python3
"""mk_pr_gap_yamls.py: the PR perf runs that fill the review page's gaps (coordinator, 2026-11-05), stock test infra
only (no stream keys). Rows, settings and packet counts as their source suites; only the sizes (and row selection) change.

400G (P150x8 LoudBox, pr_perf_runs.sh), at 256, 512, 1024, 2048, 3072, 4096, 4352 B unless noted:
  pr_ptype_sweep.yaml     2D mesh, two ERISCs (as pr_size_sweep.yaml's Mesh1HopEW, 50k packets):
                          Mesh1HopEW scatter_write and fused_atomic_inc (write is in pr_size_sweep.yaml);
                          Mesh1HopEWAtomic / Mesh1HopNSAtomic: atomic_inc (no payload; one nominal size, 256);
                          MeshFwd: 2 hops merging with a 1-hop flow ([0,0] -> [0,2] and [0,1] -> [0,2], write), as
                          galaxy/glx_mesh.yaml's GlxMeshFwd
  pr_txy_wrap_2x4.yaml    XY torus 2x4 (auto-discovered; X = ring): torus/torusxy_2x4_sizes.yaml's TorusXYWrap1HopEW
                          (column 3 -> column 0 across the wrap), write
  pr_txy_wrap_4x2.yaml    XY torus 4x2 (study descriptor torus/p150_x8_4x2_torus_xy_mgd.textproto; Y = ring):
                          torus/torusxy_4x2_sizes.yaml's TorusXY4x2_Wrap1HopNS (row 3 -> row 0 across the wrap), write
  pr_ring1_sweep.yaml     ring, two ERISCs: v4/exp_ring.yaml's Ring1HopEW, write
200G (BH Galaxy, frs_glx_pr.sbatch), at 512 and 4096 B only (the glx_sizes suites ran 256 / 1024 / 2048 / 3072 / 4352):
  pr_glx_k4_mesh.yaml     glx_sizes_mesh.yaml: GlxMesh1HopNS, GlxMesh1HopEWScatter, GlxMesh1HopEWFused,
                          GlxMesh1HopEWAtomic, GlxMesh3HopEW, GlxMeshMulticast
  pr_glx_k4_linear.yaml   glx_sizes_linear.yaml: GlxLinear3HopEW, GlxLinearMulticast
  pr_glx_k4_ring.yaml     glx_sizes_ring.yaml: GlxRing1HopEW, GlxFullRingMulticast
  pr_glx_k4_torus.yaml    glx_sizes_torus.yaml: GlxTorusWrapEW, GlxTorusWrapNS (XY torus across the wrap)
  pr_glx_nx_sweep.yaml    NeighborExchange write, 1 and 2 links, 256-4352 B (pr_size_sweep.yaml's NeighborExchange row
                          with pr_glx_size_sweep.yaml's 200,000 packets); a separate step (released by the user)
Latency (coordinator 2026-11-05): the 400G LoudBox runs fabric_exp/exp_latency.yaml as is (the same-job set of
report/data/latency_samejob.json, job 556); the Galaxy runs
  pr_glx_latency.yaml     exp_latency.yaml's rows on the Galaxy's 8x4 mesh (rows of 4 chips: [0,0] -> [0,1] 1 hop E/W,
                          [0,4] 1 hop N/S, [0,3] 3 hops E/W) plus Lat7HopNS ([0,0] -> [0,28], column 0)
Each output is checked to parse and to hold exactly the expected rows and sizes."""
import copy, os, yaml

FE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
LB_SIZES = [256, 512, 1024, 2048, 3072, 4096, 4352]
GLX_SIZES = [512, 4096]


def load(p):
    return yaml.safe_load(open(os.path.join(FE, p)))


def pick(src, names, sizes, ntypes=None):
    d = load(src)
    by = {t["name"]: t for t in d["Tests"]}
    out = []
    for n in names:
        t = copy.deepcopy(by[n])
        pp = t["parametrization_params"]
        pp["size"] = list(sizes)
        if ntypes:
            pp["ntype"] = list(ntypes)
        out.append(t)
    return d.get("allocation_policies"), out


def write(dst, note, policies, tests, want_rows, want_sizes):
    d = {"allocation_policies": policies, "Tests": tests} if policies else {"Tests": tests}
    text = "".join(f"# {line}\n" for line in note.splitlines()) + yaml.safe_dump(d, sort_keys=False)
    open(os.path.join(HERE, dst), "w").write(text)
    back = yaml.safe_load(text)
    names = [t["name"] for t in back["Tests"]]
    assert names == want_rows, (dst, names)
    for t in back["Tests"]:
        assert t["parametrization_params"]["size"] in want_sizes, (dst, t["name"])
    print(f"{dst}: {len(names)} tests: " + ", ".join(
        f"{t['name']} {t['parametrization_params']['ntype']} {t['parametrization_params']['size']}" for t in back["Tests"]))


# ---- 400G
pol, base = pick("pr_perf/pr_size_sweep.yaml", ["Mesh1HopEW"], LB_SIZES)
ew = copy.deepcopy(base[0])
ew["parametrization_params"]["ntype"] = ["unicast_scatter_write", "fused_atomic_inc"]
atom_ew = copy.deepcopy(base[0])
atom_ew["name"] = "Mesh1HopEWAtomic"
atom_ew["parametrization_params"].update(ntype=["atomic_inc"], size=[256])
_, ns = pick("exp_2d_4352.yaml", ["Mesh1HopNS"], [256], ["atomic_inc"])
atom_ns = ns[0]
atom_ns["name"] = "Mesh1HopNSAtomic"
fwd = copy.deepcopy(base[0])
fwd["name"] = "MeshFwd"
fwd["senders"] = [{"device": [0, 0], "patterns": [{"destination": {"device": [0, 2]}}]},
                  {"device": [0, 1], "patterns": [{"destination": {"device": [0, 2]}}]}]
fwd["parametrization_params"]["ntype"] = ["unicast_write"]
write("pr_ptype_sweep.yaml",
      "PR perf gap runs (mk_pr_gap_yamls.py): 2D 1-hop E/W by packet type (scatter_write, fused_atomic_inc) at 256-4352 B,\n"
      "atomic_inc (no payload, one nominal size) on 1-hop E/W and N/S, and 2 hops merging with a 1-hop flow (MeshFwd).",
      pol, [ew, atom_ew, atom_ns, fwd], ["Mesh1HopEW", "Mesh1HopEWAtomic", "Mesh1HopNSAtomic", "MeshFwd"],
      [LB_SIZES, [256]])
pol, t = pick("torus/torusxy_2x4_sizes.yaml", ["TorusXYWrap1HopEW"], LB_SIZES, ["unicast_write"])
write("pr_txy_wrap_2x4.yaml", "PR perf gap runs (mk_pr_gap_yamls.py): XY torus 2x4 (auto-discovered), 1 hop E/W across the "
      "wrap,\nwrite, 256-4352 B (torus/torusxy_2x4_sizes.yaml's TorusXYWrap1HopEW).", pol, t, ["TorusXYWrap1HopEW"], [LB_SIZES])
pol, t = pick("torus/torusxy_4x2_sizes.yaml", ["TorusXY4x2_Wrap1HopNS"], LB_SIZES, ["unicast_write"])
write("pr_txy_wrap_4x2.yaml", "PR perf gap runs (mk_pr_gap_yamls.py): XY torus 4x2 (TT_MESH_GRAPH_DESC_PATH="
      "fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto),\n1 hop N/S across the wrap, write, 256-4352 B "
      "(torus/torusxy_4x2_sizes.yaml's TorusXY4x2_Wrap1HopNS).", pol, t, ["TorusXY4x2_Wrap1HopNS"], [LB_SIZES])
pol, t = pick("v4/exp_ring.yaml", ["Ring1HopEW"], LB_SIZES, ["unicast_write"])
write("pr_ring1_sweep.yaml", "PR perf gap runs (mk_pr_gap_yamls.py): ring 1 hop E/W, two ERISCs, write, 256-4352 B "
      "(v4/exp_ring.yaml's Ring1HopEW).", pol, t, ["Ring1HopEW"], [LB_SIZES])

# ---- 200G
for suite, rows in (("mesh", ["GlxMesh1HopNS", "GlxMesh1HopEWScatter", "GlxMesh1HopEWFused", "GlxMesh1HopEWAtomic",
                              "GlxMesh3HopEW", "GlxMeshMulticast"]),
                    ("linear", ["GlxLinear3HopEW", "GlxLinearMulticast"]),
                    ("ring", ["GlxRing1HopEW", "GlxFullRingMulticast"]),
                    ("torus", ["GlxTorusWrapEW", "GlxTorusWrapNS"])):
    pol, t = pick(f"galaxy/glx_sizes_{suite}.yaml", rows, GLX_SIZES)
    write(f"pr_glx_k4_{suite}.yaml", f"PR perf gap runs, Galaxy 200G (mk_pr_gap_yamls.py): galaxy/glx_sizes_{suite}.yaml's "
          f"{', '.join(rows)}\nat 512 and 4096 B only (the sizes the 200G size suites did not run).", pol, t, rows,
          [GLX_SIZES])

# NeighborExchange at 200G (held step): pr_size_sweep.yaml's NeighborExchange row, Galaxy sweep packet count.
pol, t = pick("pr_perf/pr_size_sweep.yaml", ["NeighborExchange"], LB_SIZES)
t[0]["name"] = "GlxNeighborExchange"
t[0]["defaults"]["num_packets"] = load("pr_perf/pr_glx_size_sweep.yaml")["Tests"][0]["defaults"]["num_packets"]
t[0]["parametrization_params"]["num_links"] = [1, 2]  # 2 links: mock-mapped on single_bh_galaxy (both trees)
write("pr_glx_nx_sweep.yaml", "PR perf gap runs, Galaxy 200G (mk_pr_gap_yamls.py): NeighborExchange write, 1 and 2 links, 256-4352 B "
      "(pr_size_sweep.yaml's row,\npr_glx_size_sweep.yaml's packet count).",
      pol, t, ["GlxNeighborExchange"], [LB_SIZES])

# Galaxy latency: exp_latency.yaml's rows (chip ids hold on the Galaxy's rows of 4) plus a 7-hop N/S row down column 0.
d = load("exp_latency.yaml")
lat = d["Tests"]
assert [t["name"] for t in lat] == ["Lat1HopEW", "Lat1HopNS", "Lat3HopEW"]
l7 = copy.deepcopy(lat[1])
l7["name"] = "Lat7HopNS"
l7["senders"][0]["patterns"][0]["destination"]["device"] = [0, 28]
write("pr_glx_latency.yaml", "PR perf, Galaxy 200G latency (mk_pr_gap_yamls.py): exp_latency.yaml's rows (one-packet round trips, 1000 "
      "samples,\n16 and 4096 B, write and fused_atomic_inc) plus Lat7HopNS ([0,0] -> [0,28], column 0).", None, lat + [l7],
      ["Lat1HopEW", "Lat1HopNS", "Lat3HopEW", "Lat7HopNS"], [[16, 4096]])

# 200G NX A/B (nxab/frs_glx_nxab.sbatch, coordinator 2026-11-05): pr_glx_nx_sweep.yaml's row at 1 link, wire-bound sizes.
d = load("pr_perf/pr_glx_nx_sweep.yaml")
t = d["Tests"][0]
t["parametrization_params"].update(num_links=[1], size=[3072, 4096, 4352])
write("pr_glx_nxab.yaml", "200G NeighborExchange A/B (mk_pr_gap_yamls.py): pr_glx_nx_sweep.yaml's row at 1 link, 3072 / 4096 / 4352 B.",
      d.get("allocation_policies"), [t], ["GlxNeighborExchange"], [[3072, 4096, 4352]])

# IDLE_PASSES_BEFORE_PARTIAL_CREDIT sweep, Galaxy 200G rows besides NX (pr_glx_nxab.yaml) (coordinator 2026-11-05):
# mesh 1-hop E/W 4096 B, mesh 3-hop N/S 1024 / 4096 B ([0,0] -> [0,12], down column 0), linear 3-hop E/W 4096 B, and
# the ring controls (groups of 1): ring 1-hop E/W 4096 B, full-ring multicast 4352 B.
pol, t = pick("galaxy/glx_sizes_mesh.yaml", ["GlxMesh1HopEW", "GlxMesh1HopNS"], [4096])
ns3 = t[1]
ns3["name"] = "GlxMesh3HopNS"
ns3["senders"][0]["patterns"][0]["destination"]["device"] = [0, 12]
ns3["parametrization_params"]["size"] = [1024, 4096]
write("pr_glx_idle_mesh.yaml", "IDLE_PASSES sweep, Galaxy 200G (mk_pr_gap_yamls.py): mesh 1-hop E/W 4096 B and 3-hop N/S "
      "([0,0] -> [0,12]) 1024 / 4096 B.", pol, t, ["GlxMesh1HopEW", "GlxMesh3HopNS"], [[4096], [1024, 4096]])
pol, t = pick("galaxy/glx_sizes_linear.yaml", ["GlxLinear3HopEW"], [4096])
write("pr_glx_idle_linear.yaml", "IDLE_PASSES sweep, Galaxy 200G (mk_pr_gap_yamls.py): linear 3-hop E/W 4096 B.", pol, t,
      ["GlxLinear3HopEW"], [[4096]])
pol, t = pick("galaxy/glx_sizes_ring.yaml", ["GlxRing1HopEW", "GlxFullRingMulticast"], [4096])
t[1]["parametrization_params"]["size"] = [4352]
write("pr_glx_idle_ring.yaml", "IDLE_PASSES sweep, Galaxy 200G (mk_pr_gap_yamls.py): ring controls (groups of 1): ring 1-hop "
      "E/W 4096 B, full-ring multicast 4352 B.", pol, t, ["GlxRing1HopEW", "GlxFullRingMulticast"], [[4096], [4352]])
