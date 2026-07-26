# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 TP x SP model, calibrated to the MEASURED (4,8) anchor.

Physics (per device, per layer):
  - Compute (matmul + attention FLOPs) is INVARIANT to the TP/SP split: each device
    does total/32. So compute floors are ~flat across configs (scale only with N).
  - TP feature-dim all-gather bytes ~ M*dim = (N*TP/32)*dim  =>  cost ~ N*TP  (grows with TP).
  - SP ring K/V gather bytes ~ N*heads_local = N*(32/TP)   =>  cost ~ N/TP  (grows as TP shrinks).
Two opposing collective terms => U-shaped total, minimized near a balanced split.

Calibration: at the measured (4,8) point we split each measured bucket into
compute vs collective using the anchor's own FLOP-utilization signal:
  - AllGatherMinimalMatmul  ~5% FLOP util  -> treat as TP-collective (comm/dispatch bound)
  - ReduceScatter-matmul   ~53% FLOP util  -> treat as compute
  - RingJointSDPA is transport-bound on 4x8 (known) -> ~80% gather, ~20% attn compute
Non-(4,8) points are PREDICTIONS pending the measured sweep (device reset).
"""
import json

# Measured per-op Device Time Sum (us) over the profiled window; window held 2 ring-SDPA
# instances => 2 layers. Divide by 2 for per-layer. Source: ltx_4x8_ring_profiles/*_stacked.csv.
W = 2
ANCHOR = {
    "stage_1": dict(N=9728, ag_matmul=3916.72, rs_matmul=991.56, plain_mm=109.89,
                    ring_sdpa=1529.05, overhead=758.22 + 145.30 + 1079.26),
    "stage_2": dict(N=38912, ag_matmul=8365.84, rs_matmul=2409.00, plain_mm=110.46,
                    ring_sdpa=6795.51, overhead=764.09 + 209.86 + 2947.55),
}
RING_GATHER_FRAC = 0.80  # ring-SDPA is transport-bound on 4x8; ~80% is K/V gather, rest attn compute
TP_ANCHOR = 4

CONFIGS = [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)]


def calibrate(stage):
    a = ANCHOR[stage]
    mm_bucket = (a["ag_matmul"] + a["rs_matmul"] + a["plain_mm"]) / W
    ring_bucket = a["ring_sdpa"] / W
    overhead = a["overhead"] / W

    tp_coll_at4 = a["ag_matmul"] / W                      # comm-bound fused AG-matmul
    compute_mm = mm_bucket - tp_coll_at4                  # RS-matmul + plain (compute-bound)
    k_tp = tp_coll_at4 / TP_ANCHOR                        # tp collective ~ k_tp * TP

    sp_gather_at4 = RING_GATHER_FRAC * ring_bucket
    compute_at = ring_bucket - sp_gather_at4
    k_sp = sp_gather_at4 * TP_ANCHOR                      # sp gather ~ k_sp / TP

    return dict(compute_mm=compute_mm, k_tp=k_tp, compute_at=compute_at, k_sp=k_sp,
                overhead=overhead, mm_bucket=mm_bucket, ring_bucket=ring_bucket, N=a["N"])


def shape_penalty(TP, compute_mm):
    """Matmul MFU rolloff when per-device N=dim/TP (attn-out proj, dim=4096) gets thin."""
    n_local = 4096 / TP
    ref = 512.0
    return 0.0 if n_local >= ref else compute_mm * (ref / n_local - 1.0) * 0.15


def model(stage):
    c = calibrate(stage)
    rows = []
    for TP, SP in CONFIGS:
        if SP == 1:
            rows.append(dict(TP=TP, SP=SP, unsupported=True,
                             note="SP=1: LTX video self-attn can't mask padded keys (needs ring logical_n)"))
            continue
        measured = TP == TP_ANCHOR
        mm = c["compute_mm"] + c["k_tp"] * TP + shape_penalty(TP, c["compute_mm"])
        ring = c["compute_at"] + c["k_sp"] / TP
        rows.append(dict(TP=TP, SP=SP, measured=measured,
                         matmul_tp_ccl=round(mm, 1),
                         ring_attention=round(ring, 1),
                         overhead=round(c["overhead"], 1),
                         total=round(mm + ring + c["overhead"], 1)))
    return c, rows


def to_results_json():
    """Emit plot.py-compatible records (predicted for all but the measured 4,8)."""
    id_of = {(1, 32): "tp1_sp32", (2, 16): "tp2_sp16", (4, 8): "tp4_sp8",
             (8, 4): "tp8_sp4", (16, 2): "tp16_sp2", (32, 1): "tp32_sp1"}
    out = {}
    for stage in ANCHOR:
        _, rows = model(stage)
        for r in rows:
            if r.get("unsupported"):
                continue
            cid = id_of[(r["TP"], r["SP"])]
            out[f"{cid}/{stage}"] = dict(config=cid, stage=stage, TP=r["TP"], SP=r["SP"],
                                         matmul_tp_ccl=r["matmul_tp_ccl"], ring_attention=r["ring_attention"],
                                         overhead=r["overhead"], total=r["total"],
                                         measured=r["measured"])
    return out


if __name__ == "__main__":
    for stage in ANCHOR:
        c, rows = model(stage)
        print(f"\n=== {stage} (N={c['N']}) ===")
        print(f"  calib: compute_mm={c['compute_mm']:.0f}  k_tp={c['k_tp']:.0f} (tp_coll=k_tp*TP)  "
              f"compute_at={c['compute_at']:.0f}  k_sp={c['k_sp']:.0f} (sp_gather=k_sp/TP)  overhead={c['overhead']:.0f}")
        best = min((r for r in rows if not r.get("unsupported")), key=lambda r: r["total"])
        for r in rows:
            if r.get("unsupported"):
                print(f"  TP={r['TP']:2d} SP={r['SP']:2d}  UNSUPPORTED  {r['note']}")
            else:
                tag = " <-- MEASURED" if r["measured"] else ("  *BEST(pred)" if r is best else "")
                print(f"  TP={r['TP']:2d} SP={r['SP']:2d}  mm+ccl={r['matmul_tp_ccl']:7.0f}  "
                      f"ring={r['ring_attention']:7.0f}  over={r['overhead']:6.0f}  total={r['total']:7.0f}{tag}")
    with open("ltx_tpsp_results.json", "w") as f:
        json.dump(to_results_json(), f, indent=2)
    print("\nwrote ltx_tpsp_results.json")
