#!/usr/bin/env python3
"""mk_rf_yamls.py: the yamls of the report's consolidated data refresh (2026-11-05): main as shipped against the port
(PR tip) at 400G (P150x8 LoudBox, rf_lb_runs.sh) and 200G (BH Galaxy, rf_glx.sbatch). Stock test infra keys only,
except where noted. Rows, settings and packet counts are their source suites'; only the sizes and the row selection
change. Writes fabric_exp/report/refresh/yamls/; each output is read back and checked.

400G size sweep (sizes LB_SIZES), from the census suites:
  rf_2d       exp_2d_4352.yaml (1 hop E/W, N/S, both ways; 3 hops E/W; multicast), write + scatter; Mesh1HopEW also
              fused_atomic_inc; MeshFwd (2 hops merging with a 1-hop flow, pr_ptype_sweep.yaml's row) write, scatter,
              fused; pure atomics Mesh1HopEWAtomic / Mesh1HopNSAtomic / MeshFwdAtomic (atomic_inc, no payload: one
              nominal size, 256)
  rf_1d       exp_1d_4352.yaml (lines, half ring, NeighborExchange), write + scatter
  rf_ring     v4/exp_ring.yaml (no flow-control full-ring row in it), write + scatter
  rf_torusy   v4/exp_torus_y.yaml
  rf_txy24    torus/torusxy_2x4_sizes.yaml (XY torus 2x4, auto-discovered)
  rf_txy42    torus/torusxy_4x2_sizes.yaml (XY torus 4x2, the study descriptor torus/p150_x8_4x2_torus_xy_mgd.textproto)
  rf_ka       the keepalive table's one-way rows at 4352 B (main at firmware keepalive, main at 1000 cycles, port):
              Linear1HopEW / Linear1HopNS / Linear3HopEW and Ring1HopEW / Ring1HopNS (write, scatter), MeshFwd (write,
              fused)
  rf_bd_ew1024  breakdown point: pcs3/ew4352.yaml's row at 1024 B, 100M packets (long enough to sample)
200G (sizes GLX_SIZES): the page's missing cells, as the glx_sizes_* rows they extend
  rfg_linear  GlxLinear1HopEW scatter_write
  rfg_ring    GlxRing3HopEW ([0,0] -> [0,3], write), GlxRing1HopEW scatter_write
  rfg_torus   XY torus: GlxTorus3HopNS ([0,0] -> [0,12], 3 hops down an 8-chip column ring; E/W has no 3-hop route
              on the 4-chip row ring), GlxTorusWrapEW scatter_write, GlxTorusRowMcast ([0,1] multicast E 3 hops around
              its 4-chip row ring, as the P150's TorusXYRowMcast), write
  rfg_nx      GlxNeighborExchange scatter_write, 1 link, 200,000 packets (pr_glx_nx_sweep.yaml's row)
"""
import copy, os, yaml

FE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yamls")
GY = "/data/aagarwal/frs-glx/yamls/pr_e7fe"
LB_SIZES = [256, 1024, 2048, 3072, 4096, 4352]
GLX_SIZES = [256, 512, 1024, 2048, 3072, 4096, 4352]
W, S, F, A = "unicast_write", "unicast_scatter_write", "fused_atomic_inc", "atomic_inc"


def load(p):
    return yaml.safe_load(open(p if p.startswith("/") else os.path.join(FE, p)))


def rows(src, names=None, sizes=None, ntypes=None):
    d = load(src)
    by = {t["name"]: t for t in d["Tests"]}
    out = []
    for n in names or list(by):
        t = copy.deepcopy(by[n])
        pp = t.setdefault("parametrization_params", {})
        if sizes is not None:
            pp["size"] = list(sizes)
        if ntypes is not None:
            pp["ntype"] = list(ntypes)
        out.append(t)
    return d.get("allocation_policies"), out


def write(name, note, policies, tests, expect):
    d = {"allocation_policies": policies, "Tests": tests} if policies else {"Tests": tests}
    text = "".join(f"# {line}\n" for line in note.splitlines()) + yaml.safe_dump(d, sort_keys=False)
    path = os.path.join(OUT, name)
    open(path, "w").write(text)
    back = yaml.safe_load(text)["Tests"]
    got = [(t["name"], tuple(t["parametrization_params"]["ntype"]), tuple(t["parametrization_params"]["size"]))
           for t in back]
    assert got == expect, (name, got, expect)
    n = sum(len(t["parametrization_params"]["ntype"]) * len(t["parametrization_params"]["size"])
            * len(t["parametrization_params"].get("num_links", [1])) for t in back)
    print(f"{name}: {len(back)} rows, {n} tests")


def exp(tests):
    return [(t["name"], tuple(t["parametrization_params"]["ntype"]), tuple(t["parametrization_params"]["size"]))
            for t in tests]


os.makedirs(OUT, exist_ok=True)
HDR = "Report data refresh 2026-11-05 (fabric_exp/report/refresh/mk_rf_yamls.py): "

# ---- 400G
pol, t2d = rows("exp_2d_4352.yaml", sizes=LB_SIZES)
for t in t2d:
    if t["name"] == "Mesh1HopEW":
        t["parametrization_params"]["ntype"] = [W, S, F]
_, ptype = rows("pr_perf/pr_ptype_sweep.yaml")
by = {t["name"]: t for t in ptype}
fwd = copy.deepcopy(by["MeshFwd"])
fwd["parametrization_params"].update(ntype=[W, S, F], size=LB_SIZES)
atoms = [copy.deepcopy(by["Mesh1HopEWAtomic"]), copy.deepcopy(by["Mesh1HopNSAtomic"])]
fwd_atom = copy.deepcopy(by["MeshFwd"])
fwd_atom["name"] = "MeshFwdAtomic"
fwd_atom["parametrization_params"].update(ntype=[A], size=[256])
tests = t2d + [fwd] + atoms + [fwd_atom]
write("rf_2d.yaml", HDR + "exp_2d_4352.yaml's rows at " + str(LB_SIZES) + " B (Mesh1HopEW also fused_atomic_inc), "
      "pr_ptype_sweep.yaml's MeshFwd (write, scatter, fused) and pure atomics (atomic_inc at a nominal 256 B; "
      "MeshFwdAtomic = MeshFwd's two flows as atomic_inc).", pol, tests, exp(tests))

pol, t1d = rows("exp_1d_4352.yaml", sizes=LB_SIZES)
write("rf_1d.yaml", HDR + "exp_1d_4352.yaml's rows at " + str(LB_SIZES) + " B.", pol, t1d, exp(t1d))
pol, tr = rows("v4/exp_ring.yaml", sizes=LB_SIZES)
for t in tr:
    assert not t["fabric_setup"].get("enable_flow_control"), t["name"]  # user rule: no flow-control full ring on LBs
write("rf_ring.yaml", HDR + "v4/exp_ring.yaml's rows at " + str(LB_SIZES) + " B (no flow-control row).", pol, tr, exp(tr))
pol, tty = rows("v4/exp_torus_y.yaml", sizes=LB_SIZES)
write("rf_torusy.yaml", HDR + "v4/exp_torus_y.yaml's rows at " + str(LB_SIZES) + " B.", pol, tty, exp(tty))
pol, t24 = rows("torus/torusxy_2x4_sizes.yaml", sizes=LB_SIZES)
write("rf_txy24.yaml", HDR + "torus/torusxy_2x4_sizes.yaml's rows at " + str(LB_SIZES) + " B (XY torus 2x4, "
      "auto-discovered shape).", pol, t24, exp(t24))
pol, t42 = rows("torus/torusxy_4x2_sizes.yaml", sizes=LB_SIZES)
write("rf_txy42.yaml", HDR + "torus/torusxy_4x2_sizes.yaml's rows at " + str(LB_SIZES) + " B (XY torus 4x2: run with "
      "TT_MESH_GRAPH_DESC_PATH=fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto).", pol, t42, exp(t42))

pol, ka_lin = rows("exp_1d_4352.yaml", ["Linear1HopEW", "Linear1HopNS", "Linear3HopEW"], [4352], [W, S])
_, ka_ring = rows("v4/exp_ring.yaml", ["Ring1HopEW", "Ring1HopNS"], [4352], [W, S])
ka_fwd = copy.deepcopy(by["MeshFwd"])
ka_fwd["parametrization_params"].update(ntype=[W, F], size=[4352])
tests = ka_lin + ka_ring + [ka_fwd]
write("rf_ka.yaml", HDR + "the keepalive table's one-way rows at 4352 B (1D line, ring, 2D merge), for main at the "
      "firmware keepalive, main at 1000 cycles and the port in one job.", pol, tests, exp(tests))

d = load("pcs3/ew4352.yaml")
t = copy.deepcopy(d["Tests"][0])
t["name"] = "PcsEW1024"
t["parametrization_params"]["size"] = [1024]
t["defaults"]["num_packets"] = 100_000_000
write("rf_bd_ew1024.yaml", HDR + "breakdown point, 2D 1-hop E/W write at 1024 B: pcs3/ew4352.yaml's row with 100M "
      "packets (~10 s on the port, ~15 s on main), long enough for the PC sampler.", d["allocation_policies"], [t],
      exp([t]))

# ---- 200G
pol, gl = rows(f"{GY}/glx_sizes_linear.yaml", ["GlxLinear1HopEW"], GLX_SIZES, [S])
write("rfg_linear.yaml", HDR + "200G missing cells: glx_sizes_linear.yaml's GlxLinear1HopEW as scatter_write.", pol, gl,
      exp(gl))
pol, g1 = rows(f"{GY}/glx_sizes_ring.yaml", ["GlxRing1HopEW"], GLX_SIZES, [S])
g3 = copy.deepcopy(g1[0])
g3["name"] = "GlxRing3HopEW"
g3["parametrization_params"]["ntype"] = [W]
g3["senders"][0]["patterns"][0]["destination"]["device"] = [0, 3]
tests = [g3] + g1
write("rfg_ring.yaml", HDR + "200G missing cells: GlxRing3HopEW ([0,0] -> [0,3], as the P150's Ring3HopEW), "
      "glx_sizes_ring.yaml's GlxRing1HopEW as scatter_write.", pol, tests, exp(tests))
pol, gw = rows(f"{GY}/glx_sizes_torus.yaml", ["GlxTorusWrapEW"], GLX_SIZES, [S])
t3 = copy.deepcopy(gw[0])
t3["name"] = "GlxTorus3HopNS"
t3["parametrization_params"]["ntype"] = [W]
t3["senders"][0]["patterns"][0]["destination"]["device"] = [0, 12]
rm = copy.deepcopy(gw[0])
rm["name"] = "GlxTorusRowMcast"
rm["parametrization_params"]["ntype"] = [W]
rm["defaults"] = {"ftype": "mcast", "num_packets": 20000}
rm["senders"] = [{"device": [0, 1], "patterns": [{"destination": {"hops": {"E": 3}}}]}]
tests = [t3] + gw + [rm]
write("rfg_torus.yaml", HDR + "200G missing cells, XY torus (8x4, both axes rings): GlxTorus3HopNS ([0,0] -> [0,12]: "
      "3 hops down column 0's 8-chip ring; on the 4-chip row ring [0,0] -> [0,3] is the 1-hop wrap), "
      "glx_sizes_torus.yaml's GlxTorusWrapEW as scatter_write, GlxTorusRowMcast ([0,1] multicasting E 3 hops around its "
      "row: cols 2, 3 and, across the wrap, 0; as the P150's TorusXYRowMcast; 20,000 packets as the multicast rows).",
      pol, tests, exp(tests))
pol, gn = rows(f"{GY}/pr_glx_nx_sweep.yaml", ["GlxNeighborExchange"], GLX_SIZES, [S])
gn[0]["parametrization_params"]["num_links"] = [1]
write("rfg_nx.yaml", HDR + "200G missing cells: pr_glx_nx_sweep.yaml's GlxNeighborExchange as scatter_write, 1 link.",
      pol, gn, exp(gn))
