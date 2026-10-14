"""Dump flat_routed_expert's core layout for the DRISC benchmark: argv H I E NG M out.json."""
import json, sys
import ttnn
H, I, E, NG, M, out = (int(v) for v in sys.argv[1:6]) + (sys.argv[6],) if False else (*map(int, sys.argv[1:6]), sys.argv[6])
m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
p = dict(ttnn._ttnn.operations.bringup.flat_routed_expert_plan(m, H, I, E, NG, M))
cv = lambda v: [cv(x) for x in v] if isinstance(v, (list, tuple)) else ([v.x, v.y] if hasattr(v, "x") else v)
p = {k: cv(v) for k, v in p.items()}
keep = {k: p[k] for k in ("np", "g", "rg", "nk_gu", "kblk", "nd", "pcds", "readers", "gu", "down", "relays", "rdown", "n_rdn", "pcd_r")}
keep.update(H=H, I=I, E=E, tile_bytes=576)
json.dump(keep, open(out, "w"))
print("dumped", out, "gu", len(p["gu"]), "down", p["nd"], "readers", len(p["readers"]), "pcds", sorted(set(p["pcds"])), "rdown", p["rdown"])
ttnn.close_mesh_device(m)
