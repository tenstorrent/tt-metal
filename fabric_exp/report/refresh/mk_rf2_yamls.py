#!/usr/bin/env python3
"""mk_rf2_yamls.py: fused write + atomic (fused_atomic_inc, the stock test infra's default flush, benchmark mode) for the
report's topology x packet-type matrix (coordinator, 2026-11-06): 1D line, ring, XY torus and neighbor exchange, each as
the row shape of that topology's existing 1-hop write cell, at 1,024 / 4,096 / 4,352 B. Packet counts as the source
rows. Writes fabric_exp/report/refresh/yamls/; each output is read back and checked.
  rf2_fused_p150.yaml   400G: exp_1d_4352.yaml's Linear1HopEW and NeighborExchange (1 link), v4/exp_ring.yaml's
                        Ring1HopEW, torus/torusxy_2x4_sizes.yaml's TorusXYWrap1HopEW (XY torus 2x4, column 3 -> 0
                        across the wrap)
  rf2g_<linear|ring|torus|nx>.yaml   200G: glx_sizes_linear's GlxLinear1HopEW, glx_sizes_ring's GlxRing1HopEW,
                        glx_sizes_torus's GlxTorusWrapEW, pr_glx_nx_sweep's GlxNeighborExchange (1 link)"""
import copy, os, yaml

FE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yamls")
GY = "/data/aagarwal/frs-glx/yamls/pr_e7fe"
SIZES = [1024, 4096, 4352]
F = "fused_atomic_inc"


def pick(src, name, links=None):
    d = yaml.safe_load(open(src if src.startswith("/") else os.path.join(FE, src)))
    t = copy.deepcopy(next(x for x in d["Tests"] if x["name"] == name))
    pp = t["parametrization_params"]
    pp.update(ntype=[F], size=list(SIZES))
    if links:
        pp["num_links"] = links
    assert t.get("benchmark_mode") is True, name
    assert not t.get("fabric_setup", {}).get("enable_flow_control"), name
    return d.get("allocation_policies"), t


def write(name, note, pol, tests):
    d = {"allocation_policies": pol, "Tests": tests}
    text = "".join(f"# {l}\n" for l in note.splitlines()) + yaml.safe_dump(d, sort_keys=False)
    open(os.path.join(OUT, name), "w").write(text)
    for t in yaml.safe_load(text)["Tests"]:
        pp = t["parametrization_params"]
        assert pp["ntype"] == [F] and pp["size"] == SIZES and t["benchmark_mode"], (name, t["name"])
    print(name, [t["name"] for t in tests])


HDR = ("Report refresh 2 (2026-11-06, fabric_exp/report/refresh/mk_rf2_yamls.py): fused write + atomic (fused_atomic_inc, "
       "the stock test infra's default flush), benchmark mode only, at 1,024 / 4,096 / 4,352 B; ")
pol, lin = pick("exp_1d_4352.yaml", "Linear1HopEW")
_, nx = pick("exp_1d_4352.yaml", "NeighborExchange", [1])
_, ring = pick("v4/exp_ring.yaml", "Ring1HopEW")
_, tor = pick("torus/torusxy_2x4_sizes.yaml", "TorusXYWrap1HopEW")
write("rf2_fused_p150.yaml", HDR + "400G rows as the matrix's 1-hop write cells: Linear1HopEW, Ring1HopEW, "
      "TorusXYWrap1HopEW (XY torus 2x4), NeighborExchange (1 link).", pol, [lin, ring, tor, nx])
for suite, src, name, links in (("linear", "glx_sizes_linear.yaml", "GlxLinear1HopEW", None),
                                ("ring", "glx_sizes_ring.yaml", "GlxRing1HopEW", None),
                                ("torus", "glx_sizes_torus.yaml", "GlxTorusWrapEW", None),
                                ("nx", "pr_glx_nx_sweep.yaml", "GlxNeighborExchange", [1])):
    pol, t = pick(f"{GY}/{src}", name, links)
    write(f"rf2g_{suite}.yaml", HDR + f"200G row as the matrix's 1-hop write cell: {name} ({src}).", pol, [t])
