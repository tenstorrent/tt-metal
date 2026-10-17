import json, ttnn
d0 = ttnn.open_device(device_id=0)
p = dict(ttnn.bringup.flat_routed_expert_plan(d0, 3584, 3072, 1, 896, 8192, False, 1, False, False))
roles = {}
for role in ("readers", "relays", "gu", "down"):
    for c in p[role]:
        q = d0.worker_core_from_logical_core(ttnn.CoreCoord(c[0], c[1]))
        roles[f"{q.x},{q.y}"] = role
json.dump(roles, open("/tmp/claude-1211409210/-localdev-mstaletovic-tt-metal/ffa7931a-b2b5-4d2e-96ab-b8cae53082ad/scratchpad/roles_e1.json", "w"))
print("ROLES", {r: sum(1 for v in roles.values() if v == r) for r in ("readers", "relays", "gu", "down")})
ttnn.close_device(d0)
