import json, csv

FREQ = 1.35e9
BYTES = 2  # bf16
DRAM_PEAK = 512e9  # bytes/s
LINK_PEAK = 50e9   # bytes/s per unidirectional ethernet link
CYCLES = {"HiFi2": 2, "HiFi4": 4, "LoFi": 1, "HiFi3": 3}

# achievable fractions of peak for the best-case projection
EFF_FLOP = 0.50
EFF_DRAM = 0.90
EFF_FABRIC = 0.80

d = json.load(open("agmm/agmm_instances.json"))
rows = []
for r in d:
    gx, gy = (int(x) for x in r["grid"].split("-"))
    cores = gx * gy
    cyc = CYCLES[r["math_fidelity"]]
    peak_flops = cores * (4096 / cyc) * FREQ  # flops/s

    M, Kg, N = r["M"], r["K_gathered"], r["N"]
    flops = 2 * M * Kg * N
    t = r["min_time_us"] * 1e-6  # s (fastest device)

    ach_flops = flops / t
    flop_util = ach_flops / peak_flops

    bytes_read = BYTES * (M * Kg + Kg * N)
    ach_bw = bytes_read / t
    dram_util = ach_bw / DRAM_PEAK

    # --- fabric (bidirectional ring all-gather) ---
    R = r["ring_size"]
    nlinks = r["num_links"]
    K_local = r["K_local"]
    shard = BYTES * M * K_local                       # this device's contributed shard
    bytes_per_link = (R - 1) * shard / (2 * nlinks)   # bidirectional split, per link
    ach_link_bw = bytes_per_link / t
    fabric_util = ach_link_bw / LINK_PEAK

    # --- best-case projection: time each resource needs at its achievable ceiling ---
    # compute/DRAM/fabric overlap within AGMM, so ideal time = max(bottleneck)
    t_compute = flops / (EFF_FLOP * peak_flops)
    t_dram = bytes_read / (EFF_DRAM * DRAM_PEAK)
    t_fabric = bytes_per_link / (EFF_FABRIC * LINK_PEAK)
    limiter, t_ideal = max(
        [("compute", t_compute), ("dram", t_dram), ("fabric", t_fabric)],
        key=lambda p: p[1])
    speedup = t / t_ideal

    rows.append({
        "stage": r["stage"], "id": r["instance"],
        "M": M, "K_gathered": Kg, "N": N,
        "fused": ("addcmul" if r["has_addcmul"] else "") + (" chunks=%d" % r["chunks"] if r["chunks"] > 1 else ""),
        "GFLOPs": flops / 1e9,
        "time_us": r["min_time_us"],
        "TFLOPs_ach": ach_flops / 1e12,
        "flop_util_%": flop_util * 100,
        "MB_read": bytes_read / 1e6,
        "DRAM_GBps": ach_bw / 1e9,
        "dram_util_%": dram_util * 100,
        "shard_MB": shard / 1e6,
        "fabric_MB_per_link": bytes_per_link / 1e6,
        "fabric_GBps_per_link": ach_link_bw / 1e9,
        "fabric_util_%": fabric_util * 100,
        "bound": max(("compute", flop_util), ("dram", dram_util), ("fabric", fabric_util),
                     key=lambda p: p[1])[0],
        "t_ideal_us": t_ideal * 1e6,
        "limiter": limiter,
        "speedup": speedup,
    })

# save
with open("agmm/agmm_roofline.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for x in rows:
        w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})

# print table
h = ["stg", "id", "M", "Kgat", "N", "fused", "t_us", "FLOP%", "DRAM%", "FAB%",
     "ideal_us", "limiter", "speedup"]
print("{:>4} {:>3} {:>5} {:>5} {:>5} {:>12} {:>8} {:>6} {:>6} {:>6} {:>9} {:>8} {:>8}".format(*h))
print("-" * 120)
for x in rows:
    print("{:>4} {:>3} {:>5} {:>5} {:>5} {:>12} {:>8.1f} {:>5.1f}% {:>5.1f}% {:>5.1f}% {:>9.1f} {:>8} {:>7.2f}x".format(
        x["stage"], x["id"], x["M"], x["K_gathered"], x["N"], x["fused"] or "-",
        x["time_us"], x["flop_util_%"], x["dram_util_%"], x["fabric_util_%"],
        x["t_ideal_us"], x["limiter"], x["speedup"]))
print("-" * 120)
print(f"Peak: compute {108*2048*FREQ/1e12:.1f} TFLOP/s (HiFi2,108c@1.35GHz) | DRAM 512 GB/s | "
      f"fabric 50 GB/s per unidir link (ring={d[0]['ring_size']}, links={d[0]['num_links']}/dir)")
print(f"Achievable ceilings: {EFF_FLOP:.0%} FLOP util, {EFF_DRAM:.0%} DRAM BW, {EFF_FABRIC:.0%} fabric BW")
tot_meas = sum(x["time_us"] for x in rows)
tot_ideal = sum(x["t_ideal_us"] for x in rows)
print(f"Aggregate (sum of all 44): measured {tot_meas/1000:.1f} ms -> ideal {tot_ideal/1000:.1f} ms "
      f"= {tot_meas/tot_ideal:.2f}x")
