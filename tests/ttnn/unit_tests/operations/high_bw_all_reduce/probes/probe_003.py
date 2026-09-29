from ttexalens.tt_exalens_init import init_ttexalens

ctx = init_ttexalens()
conns = ctx.cluster_descriptor.get_ethernet_connections()
for dev in ctx.devices.values():
    print("DEV", dev.id)
    types = sorted(dev._block_locations.keys())
    print("OTHER", dev.id, {t: len(dev._block_locations[t]) for t in types})
    w = dev.get_block_locations("functional_workers")
    lx = sorted({c.to("noc0")[0] for c in w})
    ly = sorted({c.to("noc0")[1] for c in w})
    print("WORK", dev.id, "noc0 x", lx, "y", ly)
    for c in w:
        n = c.to("noc0")
        l = c.to("logical")
        if n[1] == 2:
            print("WORK", dev.id, "noc0", n, "logical", l, "translated", c.to("translated"))
    for t in types:
        if "harvest" in t or t in ("dram", "arc", "pcie", "router_only", "security", "l2cpu"):
            print(
                "HARV" if "harvest" in t else "OTHER",
                dev.id,
                t,
                sorted({c.to("noc0") for c in dev._block_locations[t]})[:40],
            )
    eth = dev.get_block_locations("eth")
    for chan, c in enumerate(eth):
        peer = conns.get(dev.id, {}).get(chan)
        print("ETH", dev.id, "chan", chan, "noc0", c.to("noc0"), "->", peer)
