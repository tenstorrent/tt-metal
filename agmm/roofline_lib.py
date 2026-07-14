# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pure roofline math for the fused AllGather+Matmul (AGMM) op.

Single source of truth shared by:
  - roofline.py         (per-instance table from agmm_instances.json)
  - run_sweeps.py       (ideal/limiter/speedup columns in the perf history)

Given a shape (M, K_gathered, N) and the collective/grid config, computes the
best-case (ideal) time under fixed per-resource efficiency ceilings, the binding
resource (limiter), and — if a measured time is supplied — achieved utilizations
and speedup-to-ideal. See AGMM_roofline_analysis.md for the derivation.
"""

FREQ = 1.35e9  # Hz, Blackhole Tensix
BYTES = 2  # bf16
DRAM_PEAK = 512e9  # bytes/s
LINK_PEAK = 50e9  # bytes/s per unidirectional ethernet link
CYCLES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}  # cycles per 8x16x16 MAC block

# Achievable fractions of peak for the best-case projection.
EFF_FLOP = 0.50
EFF_DRAM = 0.90
EFF_FABRIC = 0.80


def compute_roofline(
    M,
    K_gathered,
    N,
    *,
    ring_size=4,
    num_links=2,
    grid=(12, 9),
    math_fidelity="HiFi2",
    time_us=None,
):
    """Return a dict of roofline metrics for one AGMM instance.

    Args:
        M, K_gathered, N: matmul dims. K_gathered is the post-all-gather K
            (K_local * ring_size); it matches the `K` field in sweep_shapes.json.
        ring_size, num_links: collective config along the cluster axis.
        grid: (grid_x, grid_y) compute grid — cores = grid_x * grid_y.
        math_fidelity: one of CYCLES keys.
        time_us: measured (fastest-device) time in microseconds. When given,
            achieved utilizations and speedup-to-ideal are added.

    Always present: ideal_us, limiter, t_compute_us, t_dram_us, t_fabric_us,
        gflops, mb_read, fabric_mb_per_link.
    Added when time_us is not None: flop_util, dram_util, fabric_util, speedup.
    """
    cores = grid[0] * grid[1]
    if math_fidelity not in CYCLES:
        raise ValueError(f"Unknown math_fidelity '{math_fidelity}'. Known: {sorted(CYCLES)}")
    cyc = CYCLES[math_fidelity]
    peak_flops = cores * (4096 / cyc) * FREQ

    flops = 2 * M * K_gathered * N
    bytes_read = BYTES * (M * K_gathered + K_gathered * N)

    K_local = K_gathered / ring_size
    shard = BYTES * M * K_local  # this device's contributed shard
    bytes_per_link = (ring_size - 1) * shard / (2 * num_links)  # bidirectional split, per link

    # compute/DRAM/fabric overlap within AGMM, so ideal time = max(bottleneck)
    t_compute = flops / (EFF_FLOP * peak_flops)
    t_dram = bytes_read / (EFF_DRAM * DRAM_PEAK)
    t_fabric = bytes_per_link / (EFF_FABRIC * LINK_PEAK)
    limiter, t_ideal = max(
        [("compute", t_compute), ("dram", t_dram), ("fabric", t_fabric)],
        key=lambda p: p[1],
    )

    result = {
        "ideal_us": t_ideal * 1e6,
        "limiter": limiter,
        "t_compute_us": t_compute * 1e6,
        "t_dram_us": t_dram * 1e6,
        "t_fabric_us": t_fabric * 1e6,
        "peak_tflops": peak_flops / 1e12,
        "gflops": flops / 1e9,
        "mb_read": bytes_read / 1e6,
        "shard_mb": shard / 1e6,
        "fabric_mb_per_link": bytes_per_link / 1e6,
    }

    if time_us is not None:
        t = time_us * 1e-6
        result["tflops_ach"] = (flops / t) / 1e12
        result["dram_gbps"] = (bytes_read / t) / 1e9
        result["fabric_gbps_per_link"] = (bytes_per_link / t) / 1e9
        result["flop_util"] = (flops / t) / peak_flops
        result["dram_util"] = (bytes_read / t) / DRAM_PEAK
        result["fabric_util"] = (bytes_per_link / t) / LINK_PEAK
        result["speedup"] = t / t_ideal

    return result
