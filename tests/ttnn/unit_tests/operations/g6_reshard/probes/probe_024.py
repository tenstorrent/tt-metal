# Route-model link loads (tiles) of the shipped plan per placement: busiest inter-router link,
# busiest NIU ingress/egress port, and issuer NoC split.
import collections, torch, ttnn
import ttnn.operations.g6_reshard.g6_reshard_program_descriptor as pd
from tests.ttnn.unit_tests.operations.g6_reshard.test_g6_reshard import SCENARIOS, _configs, _make_input

device = ttnn.open_device(device_id=0)
try:
    grid = pd.NOC_GRID["BLACKHOLE"]
    cache = {}

    def xy(core):
        if core not in cache:
            v = device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
            cache[core] = (v.x, v.y)
        return cache[core]

    for name in SCENARIOS:
        in_cfg, out_cfg = _configs(device, name)
        x, _ = _make_input(device, in_cfg)
        plan = pd.describe_plan(x, out_cfg)
        in_p = pd.placement_of(x, x.memory_config())
        load = collections.Counter()
        local = 0
        nocs = collections.Counter()
        for role in plan["roles"]:
            for k, iss in enumerate(role["issuers"]):
                nocs[iss["noc"]] += 1
                b, e = iss["rows"]
                for ri, ci, pr, pw in pd.iter_pieces(
                    b, e, role["block_cols"][0], role["block_cols"][1], in_p.block_height_tiles, in_p.block_width_tiles
                ):
                    src = in_p.core_of(ri, ci)
                    if src == role["core"]:
                        local += pr * pw
                    for link in pd._route_links(iss["noc"], xy(src), xy(role["core"]), grid):
                        load[link] += pr * pw
        links = {k: v for k, v in load.items() if k[0] not in ("egress", "ingress")}
        ing = {k: v for k, v in load.items() if k[0] == "ingress"}
        eg = {k: v for k, v in load.items() if k[0] == "egress"}
        runs = [iss["runs"] for role in plan["roles"] for iss in role["issuers"]]
        print(
            f"{name} max_link={max(links.values()) if links else 0} max_ingress_port={max(ing.values())} max_egress_port={max(eg.values())} local_tiles={local} issuers_noc0/noc1={nocs[0]}/{nocs[1]} runs/issuer max={max(runs)}"
        )
finally:
    ttnn.close_device(device)
