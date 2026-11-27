# Host-only: lowest busiest link the per-issuer ("issuer" shape) p-norm search reaches from random
# starting NoC assignments, vs the shipped start (balanced plan).
import collections, random, time, torch, ttnn
import ttnn.operations.g6_reshard.g6_reshard_program_descriptor as pd
from tests.ttnn.unit_tests.operations.g6_reshard.test_g6_reshard import _configs, _make_input

device = ttnn.open_device(device_id=0)
try:
    grid = pd.NOC_GRID["BLACKHOLE"]
    model = pd._route_model(device)
    _, xy = model
    for name in ("P", "O1", "I5"):
        in_cfg, out_cfg = _configs(device, name)
        x, _ = _make_input(device, in_cfg)
        in_p = pd.placement_of(x, x.memory_config())
        pd.NOC_POLICY, pd.SPLIT_SHAPE = "balanced", "issuer"
        plan = pd.describe_plan(x, out_cfg)
        out_p = pd.placement_of(x, out_cfg)
        # rebuild roles with piece lists (describe_plan drops them)
        roles = []
        for role in plan["roles"]:
            r = dict(role)
            r["issuers"] = []
            for iss in role["issuers"]:
                b, e = iss["rows"]
                d = dict(iss)
                d["piece_list"] = list(
                    pd.iter_pieces(
                        b,
                        e,
                        role["block_cols"][0],
                        role["block_cols"][1],
                        in_p.block_height_tiles,
                        in_p.block_width_tiles,
                    )
                )
                r["issuers"].append(d)
            roles.append(r)
        base = [[iss["noc"] for iss in role["issuers"]] for role in plan["roles"]]

        def evaluate(enc):
            load = collections.Counter()
            for role, row in zip(roles, enc):
                for iss, (noc, cut) in zip(role["issuers"], row):
                    for ri, ci, pr, pw in iss["piece_list"]:
                        for link in pd._route_links(noc, xy(in_p.core_of(ri, ci)), xy(role["core"]), grid):
                            load[link] += pr * pw
            links = [v for k, v in load.items() if k[0] not in pd._NIU]
            niu = max(v for k, v in load.items() if k[0] in pd._NIU)
            return max(links, default=0), niu

        t = time.time()
        shipped = pd.split_nocs(roles, in_p, base, device, "issuer")
        print(f"{name} shipped start: {evaluate(shipped)} ({time.time()-t:.1f}s)")
        rng = random.Random(0)
        best = None
        for trial in range(12):
            start = [[rng.randint(0, 1) for _ in row] for row in base]
            enc = pd.split_nocs(roles, in_p, start, device, "issuer")
            v = evaluate(enc)
            if best is None or v < best:
                best = v
        print(f"{name} best of 12 random restarts: {best}")
finally:
    ttnn.close_device(device)
