# Host-only estimate: lowest busiest-link load (route model of g6_reshard_program_descriptor)
# reachable when each piece (not each issuer) picks its NoC. Greedy local search, same objective as balance_nocs.
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
        in_p = pd.placement_of(x, x.memory_config())
        out_p = pd.placement_of(x, out_cfg)
        units = []  # one per piece: per NoC, Counter of link -> tiles
        for r in range(out_p.num_block_rows):
            rb = r * out_p.block_height_tiles
            re_ = min(out_p.tensor_height_tiles, rb + out_p.block_height_tiles)
            for c in range(out_p.num_block_cols):
                cb = c * out_p.block_width_tiles
                ce = min(out_p.tensor_width_tiles, cb + out_p.block_width_tiles)
                dst = xy(out_p.core_of(r, c))
                for ri, ci, pr, pw in pd.iter_pieces(rb, re_, cb, ce, in_p.block_height_tiles, in_p.block_width_tiles):
                    src = xy(in_p.core_of(ri, ci))
                    per = []
                    for noc in (0, 1):
                        acc = collections.Counter()
                        for link in pd._route_links(noc, src, dst, grid):
                            acc[link] += pr * pw
                        per.append(acc)
                    units.append(per)

        def run(start):
            flat = list(start)
            load = collections.Counter()
            for k, n in enumerate(flat):
                load.update(units[k][n])
            score = lambda: (
                (
                    max(v for kk, v in load.items() if kk[0] not in ("egress", "ingress"))
                    if any(kk[0] not in ("egress", "ingress") for kk in load)
                    else 0
                ),
                max(load.values()),
                sum(v * v for v in load.values()),
            )
            best = score()
            for _ in range(32):
                imp = False
                for k in range(len(flat)):
                    o, nw = flat[k], 1 - flat[k]
                    load.subtract(units[k][o])
                    load.update(units[k][nw])
                    cand = score()
                    if cand < best:
                        best, flat[k], imp = cand, nw, True
                    else:
                        load.subtract(units[k][nw])
                        load.update(units[k][o])
                if not imp:
                    break
            ing = max(v for kk, v in load.items() if kk[0] == "ingress")
            return best[0], ing, flat.count(0), flat.count(1)

        r1 = run([1] * len(units))
        r0 = run([0] * len(units))
        print(
            f"{name:3s} pieces={len(units)} per-piece greedy from all-NoC1: max_link={r1[0]} max_ingress_port={r1[1]} noc0/noc1={r1[2]}/{r1[3]} | from all-NoC0: max_link={r0[0]} max_ingress_port={r0[1]}"
        )
finally:
    ttnn.close_device(device)
