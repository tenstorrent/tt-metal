import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
d = md
try:
    info = ttnn._ttnn.reports.get_device_info(d)
    print("L1PROBE", info, [(a, getattr(info, a)) for a in dir(info) if not a.startswith("_")], flush=True)
except Exception as e:
    print("L1PROBE err", e, [a for a in dir(ttnn._ttnn.reports)], flush=True)
ttnn.close_mesh_device(md)
