"""Roofline for BGE-M3 on Blackhole p150, corrected for where the operands live.

The first version assumed every operand came from DRAM. It does not: at B8 and
B32 the model keeps the MLP and attention-output activations in L1, and only the
weights and a few interleaved tensors come from DRAM. A DRAM-traffic model
therefore described a configuration the model does not run, which is why it
predicted savings that measured zero:

  item 2  halve the QKV activation read      predicted 0.23 ms   measured 0.00 ms
  item 3  shard the LayerNorm                predicted 1.0 ms    measured -4.76 ms
  item 5  retune the SDPA chunk              predicted 2.0 ms    measured -0.2 ms
  item 1  retune the MLP blocking            predicted 2.3 ms    the incumbent ranks 1st of 120

What follows reports two floors per op:

  T_math   the arithmetic alone, at the LoFi peak. Nothing that keeps this
           kernel and this dtype can beat it.
  T_dram   only the bytes that genuinely cross DRAM, decided per batch from
           the model's own memory configs.

The floor is max(T_math, T_dram). Where the operands sit in L1 the floor is
T_math, because L1 bandwidth is an order above DRAM and is not the constraint.

Peaks: tt-perf-report's Blackhole spec, whose tflops_lofi is per core, times the
130 cores this p150 reports. DRAM 512 GB/s.
"""

CORES = 130
PEAK_LOFI = 4096 * 1.35 / 1000 * CORES  # 719 TFLOPs
PEAK_HIFI2 = PEAK_LOFI / 2
DRAM = 512e9
SEQ = 512
HEADS, HEAD_DIM = 16, 64
LAYERS = 24
BF8 = 1.0625

# Where each activation lives, read from Optimizations.build on the board.
# in0 is the residual stream; out is this op's output.
RESIDENCY = {
    8: {"qkv": ("DRAM", "DRAM"), "ao": ("DRAM", "L1"), "wi": ("L1", "L1"), "wo": ("L1", "L1")},
    16: {"qkv": ("DRAM", "DRAM"), "ao": ("DRAM", "DRAM"), "wi": ("DRAM", "DRAM"), "wo": ("DRAM", "DRAM")},
    32: {"qkv": ("DRAM", "DRAM"), "ao": ("L1", "L1"), "wi": ("L1", "L1"), "wo": ("L1", "L1")},
}
SHAPES = {
    "qkv": ("QKV 1024->3072", 1024, 3072),
    "ao": ("attn-out 1024->1024", 1024, 1024),
    "wi": ("MLP wi 1024->4096", 1024, 4096),
    "wo": ("MLP wo 4096->1024", 4096, 1024),
}
# Measured us/call from the B8/B16 Tracy captures. B8 is post-bf8 (commit
# 04ff550d35); B16 is the pre-existing configuration.
MEASURED = {
    8: {"qkv": 102.8, "ao": 34.6, "wi": 103.9, "wo": 106.6, "sdpa": 181.7},
    16: {"qkv": 208.4, "ao": 90.4, "wi": 279.3, "wo": 226.8, "sdpa": 343.8},
}
WALL = {8: 18.285, 16: 38.976, 32: None}
H200 = {8: 4.111, 16: 8.055, 32: 16.656}


def matmul_floor(batch, k, n, in0_where, out_where):
    m = batch * SEQ
    flops = 2.0 * m * k * n
    t_math = flops / (PEAK_LOFI * 1e12) * 1e6
    # Weights always cross DRAM. Activations only when the config says DRAM.
    dram_bytes = k * n * BF8
    if in0_where == "DRAM":
        dram_bytes += m * k * BF8
    if out_where == "DRAM":
        dram_bytes += m * n * BF8
    t_dram = dram_bytes / DRAM * 1e6
    return t_math, t_dram


def sdpa_floor(batch):
    flops = 4.0 * batch * HEADS * SEQ * SEQ * HEAD_DIM
    t_math = flops / (PEAK_HIFI2 * 1e12) * 1e6
    t_dram = 4 * batch * HEADS * SEQ * HEAD_DIM * BF8 / DRAM * 1e6
    return t_math, t_dram


print(
    "Blackhole p150: LoFi %.0f TFLOPs, HiFi2 %.0f, DRAM %.0f GB/s, %d cores"
    % (PEAK_LOFI, PEAK_HIFI2, DRAM / 1e9, CORES)
)
print("chip intensity %.0f FLOPs/byte\n" % (PEAK_LOFI * 1e12 / DRAM))

for batch in (8, 16, 32):
    res = RESIDENCY[batch]
    print("=== B%d / S512 ===" % batch)
    print("  %-20s %5s %8s %8s %8s %9s %7s" % ("op", "where", "T_math", "T_dram", "floor", "measured", "util"))
    floor_sum = meas_sum = 0.0
    for key, (name, k, n) in SHAPES.items():
        in0_where, out_where = res[key]
        t_math, t_dram = matmul_floor(batch, k, n, in0_where, out_where)
        floor = max(t_math, t_dram)
        meas = MEASURED.get(batch, {}).get(key)
        floor_sum += floor
        util = ""
        if meas:
            meas_sum += meas
            util = "%6.0f%%" % (100 * floor / meas)
        print(
            "  %-20s %5s %8.1f %8.1f %8.1f %9s %7s"
            % (name, in0_where[:4] + "/" + out_where[:4], t_math, t_dram, floor, ("%.1f" % meas) if meas else "-", util)
        )
    t_math, t_dram = sdpa_floor(batch)
    floor = max(t_math, t_dram)
    floor_sum += floor
    meas = MEASURED.get(batch, {}).get("sdpa")
    util = ""
    if meas:
        meas_sum += meas
        util = "%6.0f%%" % (100 * floor / meas)
    print(
        "  %-20s %5s %8.1f %8.1f %8.1f %9s %7s"
        % ("SDPA (HiFi2)", "DRAM", t_math, t_dram, floor, ("%.1f" % meas) if meas else "-", util)
    )

    print(
        "  %-20s %5s %8s %8s %8.1f %9s"
        % ("per-layer sum", "", "", "", floor_sum, ("%.1f" % meas_sum) if meas_sum else "-")
    )
    hot_floor_ms = LAYERS * floor_sum / 1000
    hot_meas_ms = LAYERS * meas_sum / 1000 if meas_sum else None
    print("  x%d layers: hot-op floor %.2f ms" % (LAYERS, hot_floor_ms), end="")
    if hot_meas_ms:
        print("   measured %.2f ms   headroom %.2fx" % (hot_meas_ms, hot_meas_ms / hot_floor_ms))
    else:
        print()
    wall = WALL[batch]
    if wall:
        # The wall carries ops this model does not floor: LayerNorm, the head
        # split, the casts. Hold them at their measured cost.
        other = wall - hot_meas_ms
        best = hot_floor_ms + other
        print("  wall %.3f ms = %.2f hot + %.2f other" % (wall, hot_meas_ms, other))
        print(
            "  perfect hot ops -> %.2f ms = %.2fx H200 (%.3f ms).  today %.2fx"
            % (best, best / H200[batch], H200[batch], wall / H200[batch])
        )
        print("  3x H200 needs %.2f ms; 2x needs %.2f ms" % (3 * H200[batch], 2 * H200[batch]))
    print()
