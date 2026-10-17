import ttnn
d0 = ttnn.open_device(device_id=0)
g = d0.compute_with_storage_grid_size()
for E in (56, 12):
    p = dict(ttnn.bringup.flat_routed_expert_plan(d0, 3584, 3072, E, 896, 8192, False, 1, False, False))
    grid = [["." for _ in range(g.x)] for _ in range(g.y)]
    cnt = {}
    for role, ch in (("readers", "R"), ("relays", "X"), ("gu", "G"), ("down", "D"), ("rdn", "r")):
        cs = p.get(role, [])
        cnt[role] = len(cs)
        for c in cs:
            grid[c[1]][c[0]] = ch
    used = sum(1 for row in grid for v in row if v != ".")
    print(f"MAP E{E}: grid {g.x}x{g.y}={g.x*g.y}; " + ", ".join(f"{k} {v}" for k, v in cnt.items()) + f"; used {used}, free {g.x*g.y-used}; np {p['np']} g {p['g']} mt {p['mt']} nd {p['nd']}")
    print("MAP    x " + "".join(f"{x%10}" for x in range(g.x)))
    for y in range(g.y):
        print(f"MAP  y{y} " + "".join(grid[y]))
ttnn.close_device(d0)
