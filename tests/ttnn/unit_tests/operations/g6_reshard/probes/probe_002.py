# Route-model loads (tiles) of each NoC-policy setting's plan, segments included: busiest
# inter-router link, busiest NIU ingress / egress port, per-NoC busiest link.
import collections, torch, ttnn
import ttnn.operations.g6_reshard.g6_reshard_program_descriptor as pd
from tests.ttnn.unit_tests.operations.g6_reshard.test_g6_reshard import SCENARIOS, _configs, _make_input

SETTINGS = {
    "fixed2": {"NOC_POLICY": "fixed"},
    "wrapfree2": {"NOC_POLICY": "wrap_free"},
    "balanced2": {"NOC_POLICY": "balanced"},
    "split_issuer2": {"NOC_POLICY": "split", "SPLIT_SHAPE": "issuer"},
    "split_cut2": {"NOC_POLICY": "split", "SPLIT_SHAPE": "cut"},
}
device = ttnn.open_device(device_id=0)
try:
    grid = pd.NOC_GRID["BLACKHOLE"]
    cache = {}

    def xy(core):
        if core not in cache:
            v = device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
            cache[core] = (v.x, v.y)
        return cache[core]

    saved = {k: getattr(pd, k) for k in ("NOC_POLICY", "SPLIT_SHAPE")}
    for name in SCENARIOS:
        in_cfg, out_cfg = _configs(device, name)
        x, _ = _make_input(device, in_cfg)
        in_p = pd.placement_of(x, x.memory_config())
        for s, knobs in SETTINGS.items():
            for k, v in saved.items():
                setattr(pd, k, v)
            for k, v in knobs.items():
                setattr(pd, k, v)
            plan = pd.describe_plan(x, out_cfg)
            load = collections.Counter()
            per_issuer = []
            for role in plan["roles"]:
                for iss in role["issuers"]:
                    t = 0
                    for b, e, c0, c1, noc in iss["segments"]:
                        for ri, ci, pr, pw in pd.iter_pieces(
                            b, e, c0, c1, in_p.block_height_tiles, in_p.block_width_tiles
                        ):
                            t += pr * pw
                            for link in pd._route_links(noc, xy(in_p.core_of(ri, ci)), xy(role["core"]), grid):
                                load[link] += pr * pw
                    per_issuer.append(t)
            links = {k: v for k, v in load.items() if k[0] not in ("egress", "ingress")}
            l0 = max([v for k, v in links.items() if k[0] == 0], default=0)
            l1 = max([v for k, v in links.items() if k[0] == 1], default=0)
            ing = max(v for k, v in load.items() if k[0] == "ingress")
            eg = max(v for k, v in load.items() if k[0] == "egress")
            cuts = sum(1 for role in plan["roles"] for iss in role["issuers"] if iss["cut"] is not None)
            n0 = sum(1 for role in plan["roles"] for iss in role["issuers"] if iss["noc"] == 0)
            print(
                f"{name:3s} {s:14s} max_link={max(links.values(), default=0):4d} noc0_link={l0:4d} noc1_link={l1:4d} ingress={ing:3d} egress={eg:3d} cuts={cuts:3d} main_noc0={n0}"
            )
finally:
    ttnn.close_device(device)
