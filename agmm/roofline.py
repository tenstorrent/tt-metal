import json, csv, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roofline_lib import compute_roofline, FREQ, EFF_FLOP, EFF_DRAM, EFF_FABRIC

d = json.load(open("agmm/agmm_instances.json"))
rows = []
for r in d:
    gx, gy = (int(x) for x in r["grid"].split("-"))
    M, Kg, N = r["M"], r["K_gathered"], r["N"]

    rl = compute_roofline(
        M,
        Kg,
        N,
        ring_size=r["ring_size"],
        num_links=r["num_links"],
        grid=(gx, gy),
        math_fidelity=r["math_fidelity"],
        time_us=r["min_time_us"],
    )

    rows.append(
        {
            "stage": r["stage"],
            "id": r["instance"],
            "M": M,
            "K_gathered": Kg,
            "N": N,
            "fused": ("addcmul" if r["has_addcmul"] else "") + (" chunks=%d" % r["chunks"] if r["chunks"] > 1 else ""),
            "GFLOPs": rl["gflops"],
            "time_us": r["min_time_us"],
            "TFLOPs_ach": rl["tflops_ach"],
            "flop_util_%": rl["flop_util"] * 100,
            "MB_read": rl["mb_read"],
            "DRAM_GBps": rl["dram_gbps"],
            "dram_util_%": rl["dram_util"] * 100,
            "shard_MB": rl["shard_mb"],
            "fabric_MB_per_link": rl["fabric_mb_per_link"],
            "fabric_GBps_per_link": rl["fabric_gbps_per_link"],
            "fabric_util_%": rl["fabric_util"] * 100,
            "bound": max(
                ("compute", rl["flop_util"]),
                ("dram", rl["dram_util"]),
                ("fabric", rl["fabric_util"]),
                key=lambda p: p[1],
            )[0],
            "t_ideal_us": rl["ideal_us"],
            "limiter": rl["limiter"],
            "speedup": rl["speedup"],
        }
    )

# save
with open("agmm/agmm_roofline.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for x in rows:
        w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})

# print table
h = ["stg", "id", "M", "Kgat", "N", "fused", "t_us", "FLOP%", "DRAM%", "FAB%", "ideal_us", "limiter", "speedup"]
print("{:>4} {:>3} {:>5} {:>5} {:>5} {:>12} {:>8} {:>6} {:>6} {:>6} {:>9} {:>8} {:>8}".format(*h))
print("-" * 120)
for x in rows:
    print(
        "{:>4} {:>3} {:>5} {:>5} {:>5} {:>12} {:>8.1f} {:>5.1f}% {:>5.1f}% {:>5.1f}% {:>9.1f} {:>8} {:>7.2f}x".format(
            x["stage"],
            x["id"],
            x["M"],
            x["K_gathered"],
            x["N"],
            x["fused"] or "-",
            x["time_us"],
            x["flop_util_%"],
            x["dram_util_%"],
            x["fabric_util_%"],
            x["t_ideal_us"],
            x["limiter"],
            x["speedup"],
        )
    )
print("-" * 120)
print(
    f"Peak: compute {108*2048*FREQ/1e12:.1f} TFLOP/s (HiFi2,108c@1.35GHz) | DRAM 512 GB/s | "
    f"fabric 50 GB/s per unidir link (ring={d[0]['ring_size']}, links={d[0]['num_links']}/dir)"
)
print(f"Achievable ceilings: {EFF_FLOP:.0%} FLOP util, {EFF_DRAM:.0%} DRAM BW, {EFF_FABRIC:.0%} fabric BW")
tot_meas = sum(x["time_us"] for x in rows)
tot_ideal = sum(x["t_ideal_us"] for x in rows)
print(
    f"Aggregate (sum of all 44): measured {tot_meas/1000:.1f} ms -> ideal {tot_ideal/1000:.1f} ms "
    f"= {tot_meas/tot_ideal:.2f}x"
)
