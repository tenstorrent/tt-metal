import json, os, ttnn
d0 = ttnn.open_device(device_id=0)
p = ttnn.bringup.flat_routed_expert_plan(d0, 3584, 3072, 56, 896, 8192, False, 1, False, False)
d = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(p).items()}
keys = [k for k in ("np", "g", "mt", "nd", "n_rd", "rdown", "n_rdn", "pcd_r", "drisc", "dual", "drisc_n", "nh", "readers", "relays", "rects") if k in d]
print("PLAN", os.environ.get("TAG"), json.dumps({k: d[k] for k in keys}, default=lambda o: str(o)))
print("PLAN_KEYS", sorted(d.keys()))
ttnn.close_device(d0)
