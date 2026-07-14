#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""BGE-M3 B12/S8192 TP=2 (seq-parallel, 1x N300) roofline model.

Purpose
-------
Estimate two different lower bounds for this exact model on one N300:

  1. A strict but unattainable hardware-only bound that treats SDPA as pure
     matrix work and excludes its required softmax/SFPU cost.
  2. A practical, empirically calibrated floor using a 55-TFLOP/s effective
     SDPA rate. This is a planning assumption, not a hard physical limit.

AUDIT CORRECTION (2026-07-14)
--------------------------------
The original model used 148 TFLOP/s as the LoFi peak, undercounted head-split
traffic by 2x, omitted concat-head traffic, and selected firmware duration
instead of kernel duration in the CSV overlay. Those issues made the utilization
and "hard floor" claims internally inconsistent. The corrected constants and
counts below match tt-perf-report's Wormhole model and the signed TP2 profile.

Hardware constants (Wormhole b0, per single chip)
-------------------------------------------------
  Tensix cores            : 64
  AI clock                : 1.0 GHz
  LoFi peak               : 262/72 * 64 = 232.89 TFLOP/s
  HiFi2 peak              : 148/72 * 64 = 131.56 TFLOP/s
  HiFi4 peak              :  74/72 * 64 =  65.78 TFLOP/s
  DRAM bandwidth          : 288 GB/s
  Eth user bandwidth      : 12.5 GB/s / direction (100 Gbps)

Usage
-----
  # analytic ceiling only (no device needed):
  python models/demos/wormhole/bge_m3/tests/sweeps/roofline_ceiling_tp2.py

  # overlay measured device times from a profiled CSV (start/stop signposts):
  python .../roofline_ceiling_tp2.py path/to/ops_perf_results.csv
"""

from __future__ import annotations

import os
import sys

# ── N300 / Wormhole b0 hardware constants (per single chip) ──────────────────
NUM_CORES = 64
CLOCK_GHZ = 1.0
DRAM_GBPS = 288.0  # off-chip GDDR6, per chip
ETH_USER_GBPS = 12.5  # per direction, per chip, after dispatcher reserve
# AUDIT CORRECTION: tt-perf-report models a 72-core Wormhole as
# LoFi=262, HiFi2=148, HiFi4=74 TFLOP/s. N300 exposes 64 worker cores per ASIC,
# so scale each figure by 64/72. The former 148/74/37 table was shifted by one
# fidelity level and was contradicted by measured LoFi matmuls above 150 TFLOP/s.
PEAK_TFLOPS = {
    "LoFi": 262.0 / 72.0 * NUM_CORES,
    "HiFi2": 148.0 / 72.0 * NUM_CORES,
    "HiFi4": 74.0 / 72.0 * NUM_CORES,
}
DTYPE_BYTES = {"bf8": 1.0625, "bf16": 2.0, "bf4": 0.5625, "fp32": 4.0, "u32": 4.0}

# ── Model dimensions (BGE-M3) ────────────────────────────────────────────────
B = 12
S = 8192
S_LOCAL = S // 2  # seq-parallel: each chip owns half the tokens
D = 1024
H = 16
DH = 64
INTER = 4096  # MLP intermediate
LAYERS = 24
FID = "LoFi"  # all matmuls + SDPA run LoFi bf8


def gib(x):
    return x / 1e9


def matmul_time_ideal(m, k, n, fid=FID, in_bytes=DTYPE_BYTES["bf8"], out_bytes=DTYPE_BYTES["bf8"]):
    """Roofline-ideal time (us) for one M×K×N matmul: max(compute, dram)."""
    flops = 2.0 * m * k * n
    t_compute = flops / (PEAK_TFLOPS[fid] * 1e12)
    # bytes: read act (m*k) + read weight (k*n) + write out (m*n)
    bytes_moved = (m * k + k * n) * in_bytes + (m * n) * out_bytes
    t_dram = bytes_moved / (DRAM_GBPS * 1e9)
    return max(t_compute, t_dram) * 1e6, t_compute * 1e6, t_dram * 1e6


# Flash-attention includes QK^T/PV matrix work plus softmax/SFPU work that is not
# represented by the matmul FLOP count. There is no independent SFPU roofline in
# this model. 55 TFLOP/s is therefore an empirical planning assumption, chosen
# slightly above the measured ~50.5 TFLOP/s; it MUST NOT be described as a hard
# physical ceiling.
SDPA_EFFECTIVE_TFLOPS = 55.0


def sdpa_time_ideal(b, h, sq, sk, dh, effective_tflops=SDPA_EFFECTIVE_TFLOPS):
    """Calibrated time (us) for non-causal SDPA at ``effective_tflops``."""
    flops = 2.0 * (2.0 * b * h * sq * sk * dh)  # QK^T + PV
    t_compute = flops / (effective_tflops * 1e12)
    bf8 = DTYPE_BYTES["bf8"]
    bytes_moved = (b * h * sq * dh + 2 * b * h * sk * dh + b * h * sq * dh) * bf8
    t_dram = bytes_moved / (DRAM_GBPS * 1e9)
    return max(t_compute, t_dram) * 1e6, t_compute * 1e6, t_dram * 1e6


def allgather_time_ideal(bytes_local):
    """Roofline-ideal time (us) for a 2-chip all-gather: each chip sends its
    local shard across the eth link (bytes_local) at the user link bandwidth."""
    t_eth = bytes_local / (ETH_USER_GBPS * 1e9)
    return t_eth * 1e6


def layernorm_time_ideal(rows, d, in_bytes=DTYPE_BYTES["bf8"]):
    """LN is bandwidth-bound: read x + residual + write out."""
    bytes_moved = (rows * d) * (in_bytes * 2 + in_bytes)  # x, residual, out
    return bytes_moved / (DRAM_GBPS * 1e9) * 1e6


def main():
    print("=" * 78)
    print("BGE-M3 B12/S8192 TP=2 (seq-parallel, 1x N300) — ROOFLINE CEILING")
    print("=" * 78)
    print(
        f"Per chip: {NUM_CORES} Tensix @ {CLOCK_GHZ}GHz | DRAM {DRAM_GBPS} GB/s | "
        f"ETH user {ETH_USER_GBPS} GB/s/dir | matmul/SDPA {FID}={PEAK_TFLOPS[FID]} TFLOP/s"
    )
    print(f"Seq-parallel: each chip owns S_local={S_LOCAL} of S={S} tokens\n")

    # Per-layer op ideal times (us)
    M = B * S_LOCAL  # 49152 local rows

    qkv_i, qkv_c, qkv_d = matmul_time_ideal(M, D, 3 * D)
    ao_i, ao_c, ao_d = matmul_time_ideal(M, D, D)
    wi_i, wi_c, wi_d = matmul_time_ideal(M, D, INTER)
    wo_i, wo_c, wo_d = matmul_time_ideal(M, INTER, D)
    sdpa_i, sdpa_c, sdpa_d = sdpa_time_ideal(B, H, S_LOCAL, S, DH)
    kv_local = B * H * S_LOCAL * DH * DTYPE_BYTES["bf8"]
    ag_i = allgather_time_ideal(kv_local)  # per gather (K, then V)
    ln_i = layernorm_time_ideal(M, D)
    hidden_bytes = M * D * DTYPE_BYTES["bf8"]
    # AUDIT CORRECTION: fused QKV is 3*hidden and the three outputs total
    # another 3*hidden, so head-split traffic is 6*hidden (not 3*hidden).
    heads_i = 6 * hidden_bytes / (DRAM_GBPS * 1e9) * 1e6
    # Context concat reads and writes one hidden tensor.
    concat_i = 2 * hidden_bytes / (DRAM_GBPS * 1e9) * 1e6

    rows = [
        ("QKV matmul", qkv_i, "compute" if qkv_c > qkv_d else "dram", 1),
        ("AttnOut matmul", ao_i, "compute" if ao_c > ao_d else "dram", 1),
        ("MLP wi matmul", wi_i, "compute" if wi_c > wi_d else "dram", 1),
        ("MLP wo matmul", wo_i, "compute" if wo_c > wo_d else "dram", 1),
        ("SDPA", sdpa_i, "compute" if sdpa_c > sdpa_d else "dram", 1),
        ("AllGather K", ag_i, "eth-link", 1),
        ("AllGather V", ag_i, "eth-link", 1),
        ("LayerNorm x2", ln_i, "dram", 2),
        ("QKV head-split", heads_i, "dram", 1),
        ("Concat heads", concat_i, "dram", 1),
    ]

    print(f"{'op':18} {'ideal_us':>9} {'bound':>9} {'x/layer':>7} {'layer_us':>9}")
    print("-" * 60)
    per_layer = 0.0
    for name, us, bound, cnt in rows:
        layer_us = us * cnt
        per_layer += layer_us
        print(f"{name:18} {us:9.1f} {bound:>9} {cnt:7} {layer_us:9.1f}")
    print("-" * 60)
    total_ideal_ms = per_layer * LAYERS / 1000
    # Strict hardware-only bound: replace the empirical SDPA rate with LoFi
    # matrix peak. This intentionally omits mandatory softmax/SFPU cost and is
    # shown only to demonstrate what the hardware math does (and does not) prove.
    sdpa_hw_i, _, _ = sdpa_time_ideal(B, H, S_LOCAL, S, DH, PEAK_TFLOPS[FID])
    strict_hw_ms = (per_layer - sdpa_i + sdpa_hw_i) * LAYERS / 1000
    print(f"{'PER-LAYER ideal':18} {per_layer:9.1f} us")
    print(f"{'CALIBRATED FLOOR':18} {total_ideal_ms:9.2f} ms  ({LAYERS} layers)")
    print(f"{'HW-ONLY LOWER BOUND':18} {strict_hw_ms:9.2f} ms  (softmax cost omitted)\n")

    # Compute the bound breakdown
    comp = sum(us * cnt for n, us, b, cnt in rows if b == "compute")
    dram = sum(us * cnt for n, us, b, cnt in rows if b == "dram")
    eth = sum(us * cnt for n, us, b, cnt in rows if b == "eth-link")
    print("Per-layer ideal by bound:")
    print(f"  compute-bound : {comp:8.1f} us ({100*comp/per_layer:.0f}%)")
    print(f"  dram-bound    : {dram:8.1f} us ({100*dram/per_layer:.0f}%)")
    print(f"  eth-link-bound: {eth:8.1f} us ({100*eth/per_layer:.0f}%)")

    print("\nNOTE: calibrated floor assumes serialized ops and a 55-TFLOP/s empirical SDPA rate.")
    print("It excludes embeddings/initial LayerNorm and trace wall overhead.")
    print("Measured best trace replay = 1560ms; signed device-kernel sum is ~1461ms.")

    # Optional: overlay measured CSV
    # Require an explicit CSV. Automatically selecting the newest report can
    # accidentally overlay an isolated sweep instead of a full-model profile.
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    if csv and os.path.exists(csv):
        _overlay_measured(csv, total_ideal_ms)


def _overlay_measured(csv, ceiling_ms):
    import pandas as pd

    df = pd.read_csv(csv)
    if "OP TYPE" in df.columns:
        sp = df[df["OP TYPE"] == "signpost"]["OP CODE"] if "OP CODE" in df.columns else []
        try:
            i0 = sp[sp == "start"].index[0]
            i1 = sp[sp == "stop"].index[0]
            df = df.iloc[i0 + 1 : i1]
        except Exception:
            print("\n(no start/stop signposts in CSV; refusing to overlay a mixed-phase report)")
            return
    dev0 = df[df["DEVICE ID"].astype(str).isin(["0", "0.0"])] if "DEVICE ID" in df.columns else df
    # AUDIT CORRECTION: the CSV places DEVICE FW DURATION before DEVICE KERNEL
    # DURATION. The previous substring search selected firmware duration and
    # overstated the signed profile (1645ms instead of ~1461ms).
    dur_col = next((c for c in df.columns if "DEVICE KERNEL DURATION" in c), None)
    if dur_col is None:
        dur_col = next((c for c in df.columns if "DEVICE FW DURATION" in c), None)
    if dur_col is None:
        print("\n(no duration column in CSV; skipping measured overlay)")
        return
    code_col = "OP CODE" if "OP CODE" in df.columns else None

    def _f(v):
        try:
            return float(str(v).split("[")[0])
        except (ValueError, TypeError):
            return 0.0

    from collections import defaultdict

    agg = defaultdict(float)
    for _, r in dev0.iterrows():
        dt = _f(r[dur_col]) / 1e3  # ns -> us
        if dt <= 0:
            continue
        code = str(r[code_col]) if code_col else "op"
        key = (
            "SDPA"
            if "SDPA" in code
            else (
                "AllGather"
                if "AllGather" in code
                else (
                    "Matmul"
                    if "Matmul" in code
                    else (
                        "LayerNorm"
                        if "LayerNorm" in code
                        else "HeadSplit" if "Generic" in code else "ConcatHeads" if "ConcatHeads" in code else "other"
                    )
                )
            )
        )
        agg[key] += dt
    print("\n" + "=" * 60)
    print(f"MEASURED (device 0, signed device time from {dur_col})")
    print("=" * 60)
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"  {k:12} {v/1000:8.2f} ms")
    meas_ms = sum(agg.values()) / 1000
    print(f"  {'TOTAL':12} {meas_ms:8.2f} ms")
    if meas_ms:
        print(
            f"\nCalibrated-floor efficiency = {ceiling_ms:.2f} / {meas_ms:.2f} = " f"{100 * ceiling_ms / meas_ms:.0f}%"
        )


if __name__ == "__main__":
    main()
