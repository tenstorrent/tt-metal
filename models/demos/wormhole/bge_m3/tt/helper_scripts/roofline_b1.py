"""B1 is the tuned reference. Its achieved utilization per op is the empirical
ceiling this codebase reaches on this silicon. Apply that ceiling to B8/B16 to
estimate what a comparable tuning effort would buy, instead of assuming 100%.

Floors use the same roofline as roofline.py: LoFi 719 TFLOPs (130 cores),
HiFi2 359, DRAM 512 GB/s, floor = max(T_math, T_comms).
"""

CORES = 130
PEAK_LOFI = 4096 * 1.35 / 1000 * CORES
PEAK_HIFI2 = PEAK_LOFI / 2
DRAM = 512e9
SEQ = 512
BYTES = {"BF16": 2.0, "BFP8": 1.0625}


def mm_floor(B, K, N, in0, w="BFP8", out="BFP8", peak=PEAK_LOFI):
    flops = 2.0 * B * SEQ * K * N
    byt = B * SEQ * K * BYTES[in0] + K * N * BYTES[w] + B * SEQ * N * BYTES[out]
    return max(flops / (peak * 1e12), byt / DRAM) * 1e6


def sdpa_floor(B, H=16, Dh=64, dt="BFP8", peak=PEAK_HIFI2):
    flops = 4.0 * B * H * SEQ * SEQ * Dh
    byt = 4 * B * H * SEQ * Dh * BYTES[dt]
    return max(flops / (peak * 1e12), byt / DRAM) * 1e6


# measured us/call from the tt-perf-report CSVs
MEAS = {
    1: dict(qkv=19.6, ao=9.8, wi=27.1, wo=31.3, sdpa=28.3),
    8: dict(qkv=147.0, ao=40.1, wi=103.9, wo=106.7, sdpa=304.9),
    16: dict(qkv=208.4, ao=90.4, wi=279.3, wo=226.8, sdpa=343.8),
}
SHAPES = {
    "qkv": ("QKV  1024->3072", 1024, 3072, "BF16"),
    "ao": ("attn-out 1024->1024", 1024, 1024, "BFP8"),
    "wi": ("MLP wi 1024->4096", 1024, 4096, "BFP8"),
    "wo": ("MLP wo 4096->1024", 4096, 1024, "BFP8"),
}

# 1. B1 achieved utilization = ceiling
print("=== B1: achieved utilization per op (the empirical ceiling) ===")
ceil = {}
for k, (nm, K, N, i0) in SHAPES.items():
    fl = mm_floor(1, K, N, i0)
    ceil[k] = fl / MEAS[1][k]
    print("  %-22s floor %5.1f us  meas %5.1f us  util %4.0f%%" % (nm, fl, MEAS[1][k], 100 * ceil[k]))
fl = sdpa_floor(1)
ceil["sdpa"] = fl / MEAS[1]["sdpa"]
print("  %-22s floor %5.1f us  meas %5.1f us  util %4.0f%%" % ("SDPA", fl, MEAS[1]["sdpa"], 100 * ceil["sdpa"]))

# 2. Project B8/B16 at B1's per-op utilization
H200 = {8: 4.111, 16: 8.055}
BASE_WALL = {8: 23.699, 16: 38.963}
for B in (8, 16):
    print("\n=== B%d: today vs 'tuned to B1 utilization' ===" % B)
    print("  %-22s %8s %8s %8s %9s %8s" % ("op", "floor", "today", "util", "@B1 util", "saved"))
    tot_today = tot_proj = tot_floor = 0.0
    for k, (nm, K, N, i0) in SHAPES.items():
        fl = mm_floor(B, K, N, i0)
        today = MEAS[B][k]
        proj = fl / ceil[k]
        tot_today += today
        tot_proj += proj
        tot_floor += fl
        print("  %-22s %7.1f  %7.1f  %6.0f%%  %8.1f  %7.1f" % (nm, fl, today, 100 * fl / today, proj, today - proj))
    fl = sdpa_floor(B)
    today = MEAS[B]["sdpa"]
    proj = fl / ceil["sdpa"]
    tot_today += today
    tot_proj += proj
    tot_floor += fl
    print("  %-22s %7.1f  %7.1f  %6.0f%%  %8.1f  %7.1f" % ("SDPA", fl, today, 100 * fl / today, proj, today - proj))
    saved_ms = 24 * (tot_today - tot_proj) / 1000
    proj_wall = BASE_WALL[B] - saved_ms
    print(
        "  %-22s %7.1f  %7.1f  %6.0f%%  %8.1f  %7.1f   (per layer, us)"
        % ("TOTAL", tot_floor, tot_today, 100 * tot_floor / tot_today, tot_proj, tot_today - tot_proj)
    )
    print(
        "  x24 layers: hot-op time %.2f ms -> %.2f ms, saves %.2f ms"
        % (24 * tot_today / 1000, 24 * tot_proj / 1000, saved_ms)
    )
    print(
        "  wall %.2f ms -> ~%.2f ms   = %.2fx H200 (today %.2fx; 3x target %.2f ms, 2x %.2f ms)"
        % (BASE_WALL[B], proj_wall, proj_wall / H200[B], BASE_WALL[B] / H200[B], 3 * H200[B], 2 * H200[B])
    )
