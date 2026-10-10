import ttnn, json, sys
m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
H, I, E, NG, M, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
p = dict(ttnn._ttnn.operations.bringup.flat_routed_expert_plan(m, H, I, E, NG, M))
def cv(v):
    if isinstance(v, (list, tuple)): return [cv(x) for x in v]
    if hasattr(v, "x"): return [v.x, v.y]
    return v
json.dump({k: cv(v) for k, v in p.items()}, open(out, "w"))
ttnn.close_mesh_device(m)
