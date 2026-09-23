"""Roofline for BGE-M3 matmuls and SDPA on Blackhole p150, following
https://jax-ml.github.io/scaling-book/roofline/ .

T_math  = FLOPs / peak_FLOPs_per_s
T_comms = bytes / DRAM_bandwidth
T_lower = max(T_math, T_comms)         the roofline floor
compute-bound when intensity(op) = FLOPs/bytes  >  intensity(chip) = peak/bw

Peaks follow tt-perf-report's Blackhole spec. Its tflops_lofi is PER CORE
(tflops_per_core), so multiply by the worker-core count. This p150 reports
130 cores (tt-perf-report assumes 110 for headroom):
  LoFi per core = 4096 * 1.35 GHz / 1000 = 5.53 TFLOPs
  LoFi chip     = 5.53 * 130             = 719 TFLOPs
  HiFi2 chip    = 719 / 2                = 359 TFLOPs
  DRAM          = 512 GB/s
The measured us/call comes from the tt-perf-report CSVs for B8 and B16.
"""

CORES = 130
PEAK_LOFI_CORE = 4096 * 1.35 / 1000  # TFLOPs per core, tt-perf-report's constant
PEAK_LOFI = PEAK_LOFI_CORE * CORES
PEAK_HIFI2 = PEAK_LOFI / 2
DRAM_GBS = 512.0
CHIP_INTENSITY = PEAK_LOFI * 1e12 / (DRAM_GBS * 1e9)  # FLOPs per byte

SEQ = 512
BYTES = {"BF16": 2.0, "BFP8": 1.0625}  # bfp8 is 1 B/elem + 1/16 shared exponent


def matmul(name, B, M, K, N, measured_us, in0="BFP8", w="BFP8", out="BFP8", peak=PEAK_LOFI):
    flops = 2.0 * B * M * K * N
    bytes_ = B * M * K * BYTES[in0] + K * N * BYTES[w] + B * M * N * BYTES[out]
    t_math = flops / (peak * 1e12) * 1e6
    t_comm = bytes_ / (DRAM_GBS * 1e9) * 1e6
    floor = max(t_math, t_comm)
    bound = "compute" if t_math > t_comm else "memory"
    inten = flops / bytes_
    print(
        "  %-22s B%-2d  meas %7.1f us | T_math %6.1f  T_comm %6.1f  floor %6.1f | %s-bound  I=%6.0f  util %4.0f%%  headroom %4.1fx"
        % (name, B, measured_us, t_math, t_comm, floor, bound, inten, 100 * floor / measured_us, measured_us / floor)
    )
    return floor, measured_us


def sdpa(B, H, S, Dh, measured_us, dt="BFP8", peak=PEAK_HIFI2):
    # Q@K^T: 2*B*H*S*S*Dh ; softmax@V: 2*B*H*S*S*Dh. Softmax exp/sum ignored (SFPU, small).
    flops = 4.0 * B * H * S * S * Dh
    # Flash-style: read Q,K,V once, write O once. Scores never touch DRAM.
    bytes_ = 4 * B * H * S * Dh * BYTES[dt]
    t_math = flops / (peak * 1e12) * 1e6
    t_comm = bytes_ / (DRAM_GBS * 1e9) * 1e6
    floor = max(t_math, t_comm)
    bound = "compute" if t_math > t_comm else "memory"
    print(
        "  %-22s B%-2d  meas %7.1f us | T_math %6.1f  T_comm %6.1f  floor %6.1f | %s-bound  I=%6.0f  util %4.0f%%  headroom %4.1fx"
        % (
            "SDPA (HiFi2)",
            B,
            measured_us,
            t_math,
            t_comm,
            floor,
            bound,
            flops / bytes_,
            100 * floor / measured_us,
            measured_us / floor,
        )
    )
    return floor, measured_us


print(
    "Blackhole p150 roofline   peak LoFi %.2f TFLOPs (130 cores)  HiFi2 %.2f  DRAM %.0f GB/s"
    % (PEAK_LOFI, PEAK_HIFI2, DRAM_GBS)
)
print("chip intensity (LoFi)     %.0f FLOPs/byte  -> ops above this are compute-bound\n" % CHIP_INTENSITY)

for B, meas in (
    (8, dict(qkv=147.0, wi=103.9, wo=106.7, ao=40.1, sdpa=304.9)),
    (16, dict(qkv=208.4, wi=279.3, wo=226.8, ao=90.4, sdpa=343.8)),
):
    print("=== B%d / S512, per layer (x24) ===" % B)
    tot_floor = tot_meas = 0.0
    for nm, (K, N, us, i0, o) in {
        "QKV  1024->3072": (1024, 3072, meas["qkv"], "BF16", "BFP8"),
        "attn-out 1024->1024": (1024, 1024, meas["ao"], "BFP8", "BFP8"),
        "MLP wi 1024->4096": (1024, 4096, meas["wi"], "BFP8", "BFP8"),
        "MLP wo 4096->1024": (4096, 1024, meas["wo"], "BFP8", "BFP8"),
    }.items():
        f, m = matmul(nm, B, SEQ, K, N, us, in0=i0, out=o)
        tot_floor += f
        tot_meas += m
    f, m = sdpa(B, 16, SEQ, 64, meas["sdpa"])
    tot_floor += f
    tot_meas += m
    print(
        "  %-22s      meas %7.1f us   floor %6.1f us   -> x24 layers: meas %6.2f ms  floor %5.2f ms  headroom %.1fx\n"
        % ("LAYER TOTAL", tot_meas, tot_floor, 24 * tot_meas / 1000, 24 * tot_floor / 1000, tot_meas / tot_floor)
    )

print("Token count for compute-bound (book: B_tokens > peak/bw):  %.0f tokens" % CHIP_INTENSITY)
print("  B8  = %d tokens   B16 = %d   B32 = %d" % (8 * SEQ, 16 * SEQ, 32 * SEQ))
