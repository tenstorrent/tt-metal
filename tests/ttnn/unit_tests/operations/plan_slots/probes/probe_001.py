import json, ttnn
d0 = ttnn.open_device(device_id=0)
p = dict(ttnn.bringup.flat_routed_expert_plan(d0, 3584, 3072, 1, 896, 8192, False, 1, False, False))
phys = lambda c: (lambda q: f"{q.x},{q.y}")(d0.worker_core_from_logical_core(ttnn.CoreCoord(c[0], c[1])))
out = {"rg": p["rg"], "n_rd": p["n_rd"], "gu": [phys(c) for c in p["gu"]], "readers": [phys(c) for c in p["readers"]],
       "down": [phys(c) for c in p["down"]], "relays": [phys(c) for c in p["relays"]]}
json.dump(out, open("/tmp/claude-1211409210/-localdev-mstaletovic-tt-metal/ffa7931a-b2b5-4d2e-96ab-b8cae53082ad/scratchpad/slots_e1.json", "w"))
print("SLOTS rg", p["rg"], "n_rd", p["n_rd"], "gu", len(p["gu"]))
ttnn.close_device(d0)
