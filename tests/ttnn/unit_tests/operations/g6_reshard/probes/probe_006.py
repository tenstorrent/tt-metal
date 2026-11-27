# Route-model loads (tiles, unweighted) of each policy setting's plan: busiest link, per-NoC busiest link,
# busiest NIU ingress/egress, cuts, issuers whose main NoC is NoC0, and whether the plan equals balanced's.
import collections, torch, ttnn
import ttnn.operations.g6_reshard.g6_reshard_program_descriptor as pd
from tests.ttnn.unit_tests.operations.g6_reshard.test_g6_reshard import SCENARIOS, _configs, _make_input
from tests.ttnn.unit_tests.operations.g6_reshard.test_g6_reshard_noc_policy_perf import SETTINGS

device = ttnn.open_device(device_id=0)
try:
    grid = pd.NOC_GRID["BLACKHOLE"]
    cache = {}

    def xy(core):
        if core not in cache:
            v = device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
            cache[core] = (v.x, v.y)
        return cache[core]

    saved = {k: getattr(pd, k) for k in ("NOC_POLICY", "SPLIT_SHAPE", "ISSUERS_PER_CORE", "SPLIT_WRAP_WEIGHT")}
    for name in SCENARIOS:
        in_cfg, out_cfg = _configs(device, name)
        x, _ = _make_input(device, in_cfg)
        in_p = pd.placement_of(x, x.memory_config())
        plans = {}
        for s, knobs in SETTINGS.items():
            if knobs.get("ISSUERS_PER_CORE") == 1:
                continue
            for k, v in saved.items():
                setattr(pd, k, v)
            for k, v in knobs.items():
                setattr(pd, k, v)
            plan = pd.describe_plan(x, out_cfg)
            plans[s] = [tuple(iss["segments"]) for role in plan["roles"] for iss in role["issuers"]]
            load = collections.Counter()
            for role in plan["roles"]:
                for iss in role["issuers"]:
                    for b, e, c0, c1, noc in iss["segments"]:
                        for ri, ci, pr, pw in pd.iter_pieces(
                            b, e, c0, c1, in_p.block_height_tiles, in_p.block_width_tiles
                        ):
                            for link in pd._route_links(noc, xy(in_p.core_of(ri, ci)), xy(role["core"]), grid):
                                load[link] += pr * pw
            links = {k: v for k, v in load.items() if k[0] not in ("egress", "ingress")}
            l0 = max([v for k, v in links.items() if k[0] == 0], default=0)
            l1 = max([v for k, v in links.items() if k[0] == 1], default=0)
            ing = max(v for k, v in load.items() if k[0] == "ingress")
            eg = max(v for k, v in load.items() if k[0] == "egress")
            cuts = sum(1 for role in plan["roles"] for iss in role["issuers"] if iss["cut"] is not None)
            n0 = sum(1 for role in plan["roles"] for iss in role["issuers"] if iss["noc"] == 0)
            same = "=balanced" if s != "balanced2" and plans.get("balanced2") == plans[s] else ""
            print(
                f"{name:3s} {s:17s} max_link={max(links.values(), default=0):4d} noc0={l0:4d} noc1={l1:4d} ingress={ing:3d} egress={eg:3d} cuts={cuts:3d} main_noc0={n0:3d} {same}"
            )
    for k, v in saved.items():
        setattr(pd, k, v)
finally:
    ttnn.close_device(device)
