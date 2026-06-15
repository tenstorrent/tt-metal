# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Extract per-signpost matmul_decode + sharding-glue device time from a tracy
ops_perf_results CSV. All numbers SOLELY from DEVICE FW DURATION [ns].

Per signpost region (MMD_SigLIP_T{T} / MMD_VLM_T{T}):
  (a) matmul_decode  = sum DEVICE FW DURATION over MatmulDecodeDeviceOperation rows
  (b) glue           = sum over InterleavedToSharded + ShardedToInterleaved rows
Region attribution: rows between this signpost marker and the next.

Forward-count normalization: each named region contains its N_ITERS measured
forwards PLUS the single warm-up forward of the NEXT region (the next region's
warm-up runs before its signpost fires). Solve per-single-forward from
  region_obs = N_ITERS * m[this] + 1 * m[next]
using the matmul_decode CALL COUNT as the anchor (calls/fwd is an exact integer:
n_chunks * sum_over_projections(ceil(T/32) * device_calls_per_mtile)).
The last region of each test has no following warm-up inside it (the next test's
per-test signpost T_<node> fires first), so next-contribution = 0.
"""
from __future__ import annotations

import csv
import math
import os
import sys

N_ITERS = int(os.environ.get("N_ITERS", "5"))

# deep-plan_14 §9.1: additive METRIC env switch selecting which device-time column the
# per-forward solver consumes. DEFAULT = FW (byte-identical to all prior behavior).
# METRIC=KERNEL selects "DEVICE KERNEL DURATION [NS]" (col 20, the compute-kernel time;
# the verdict column for the strict-beat campaign). The switch lives ONLY in parse() +
# parse_mmsweep() -- report_natexp() has no column of its own (it CALLS parse()), so
# switching parse() automatically covers natexp+mmsweep. The EXACT candidate string is
# used FIRST in _find_col so the exact-match pass hits col 20 before the fuzzy fallback
# (which would otherwise match the 7 "...KERNEL DURATION..." columns 20-31).
_METRIC = os.environ.get("METRIC", "FW").upper()
if _METRIC == "KERNEL":
    _DUR_CANDS = ("DEVICE KERNEL DURATION [NS]", "DEVICE KERNEL DURATION")
else:
    _DUR_CANDS = ("DEVICE FW DURATION [NS]", "DEVICE FW DURATION")

MMD = "MatmulDecodeDeviceOperation"
MM_NATIVE = "MatmulDeviceOperation"  # deep-plan_8: native matmul op (Leg N verdict column)
I2S = "InterleavedToShardedDeviceOperation"
S2I = "ShardedToInterleavedDeviceOperation"

# deep-plan_8 Target-1 denoise reuse signposts (one region per CSV) -- UNTOUCHED.
DENOISE_MARKERS = ("DENOISE_NATIVE", "DENOISE_RESIDENT", "DENOISE_REREAD")

# deep-plan_10 §5: NATIVE region markers (one per stage). The native leg sums
# MatmulDeviceOperation DEVICE FW into the SAME per-block-forward unit as mmd via the
# sibling solver solve_native_per_forward. Native is unchunked (one GEMM/proj/forward).
NATIVE_MARKERS = ("NATIVE_SigLIP", "NATIVE_VLM", "NATIVE_DENOISE")
# expected native matmul-op CALLS/forward = N_proj per stage (the §6c drift anchor).
NATIVE_CALLS_EXPECTED = {"SigLIP": 4, "VLM": 5, "DENOISE": 3}

# deep-plan_11 §5.5: WS2D (native 2D-mcast, resident-WIDTH_SHARDED-L1 in1) + NAT2D
# (same explicit 2D config, interleaved-DRAM in1) region prefixes. BOTH legs emit the
# SAME op-code (MatmulDeviceOperation, == MM_NATIVE) -- separated by DISTINCT signposts.
# Region forms: WS2D_SigLIP_T{T} / WS2D_VLM_T{T} / NAT2D_SigLIP_T{T} / NAT2D_VLM_T{T}.
WS2D_PREFIXES = ("WS2D_", "NAT2D_")

# device matmul_decode CALLS per single projection M-tile (from the plan):
#   FULL plain      -> 1 call            (1 N-chunk, G=1)
#   FULL K-split G  -> G calls
#   PARTIAL nchunks -> nchunks calls
# These are fixed per projection (independent of T). calls_per_mtile below.
SIGLIP_PROJ = {  # role: (calls_per_mtile, K_for_n_mtiles_note)
    "qkv": None, "o": None, "fc1": None, "fc2": None,
}


def _find_col(header, *cands):
    for c in cands:
        for i, h in enumerate(header):
            if h.strip().upper() == c.upper():
                return i
    # fuzzy
    for i, h in enumerate(header):
        hu = h.strip().upper()
        if all(t in hu for t in cands[0].upper().split()):
            return i
    raise KeyError(cands)


def parse(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    ci_dur = _find_col(header, *_DUR_CANDS)
    ci_op = _find_col(header, "OP CODE", "OP TYPE")
    # signpost / marker column: tracy writes a row with the signpost name; detect by
    # an op-code that is empty + a column containing the header text. Simpler: many
    # tracy CSVs carry signposts inline in OP CODE as the header text. We scan the
    # whole row text for our markers.
    regions = []  # (name, start_idx)
    # deep-plan_10 §5.1: ADDITIVELY open NATIVE_* and MMD_DENOISE_T regions. KEEP
    # T_test_ (autouse warm-up region, load-bearing for the warm-up subtraction model).
    markers = ("MMD_SigLIP_T", "MMD_VLM_T", "MMD_DENOISE_T",
               "NATIVE_SigLIP", "NATIVE_VLM", "NATIVE_DENOISE", "T_test_",
               # deep-plan_11 §5.5: ADDITIVE -- WS2D_/NAT2D_ only OPEN MORE regions; they
               # never appear in pre-existing CSVs so MMD_/NATIVE_ boundaries are unaffected.
               "WS2D_SigLIP_T", "WS2D_VLM_T", "NAT2D_SigLIP_T", "NAT2D_VLM_T",
               # deep-plan_12 S2: ADDITIVE -- best-explicit native sweep markers
               # (NATEXP_<stage>_<proj>_<cfgid>) + RESID_ resident-leg markers. Never in
               # pre-existing CSVs, so DEFAULT-mode boundaries stay byte-identical.
               "NATEXP_", "RESID_")
    for idx, r in enumerate(rows[1:], start=1):
        joined = " ".join(r)
        for m in markers:
            if m in joined:
                # extract the token containing the marker
                tok = next((c for c in r if m in c), None)
                if tok:
                    regions.append((tok.strip(), idx))
                break
    # accumulate per region
    out = {}
    for ri, (name, start) in enumerate(regions):
        end = regions[ri + 1][1] if ri + 1 < len(regions) else len(rows)
        mmd_us = glue_us = 0.0
        mmd_calls = i2s_calls = s2i_calls = 0
        mm_native_us = 0.0
        mm_native_calls = 0  # deep-plan_10 §5: native matmul op (additive; 0 in mmd regions)
        for r in rows[start:end]:
            if len(r) <= max(ci_dur, ci_op):
                continue
            op = r[ci_op].strip()
            try:
                dur = float(r[ci_dur])
            except (ValueError, IndexError):
                continue
            if op == MMD:
                mmd_us += dur / 1000.0
                mmd_calls += 1
            elif op == MM_NATIVE:
                mm_native_us += dur / 1000.0
                mm_native_calls += 1
            elif op == I2S:
                glue_us += dur / 1000.0
                i2s_calls += 1
            elif op == S2I:
                glue_us += dur / 1000.0
                s2i_calls += 1
        out[name] = dict(mmd_us=mmd_us, glue_us=glue_us, mmd_calls=mmd_calls,
                         i2s_calls=i2s_calls, s2i_calls=s2i_calls,
                         mm_native_us=mm_native_us, mm_native_calls=mm_native_calls)
    return out, regions


# Per-single-forward solve: region_obs = N_ITERS*m[this] + 1*m[next_warmup].
# The "next warm-up" of a region is the warm-up forward of the NEXT T in the
# SAME test (it runs before that T's signpost fires). For the LAST T in a test,
# the next marker is the following test's per-test signpost, whose region holds
# only that test's first warm-up -- so next-contribution = 0 for the last T.
#
# Device matmul_decode calls per SINGLE forward at chunk T (S, calls/mtile fixed):
#   forward_calls(T) = ceil(S/T) * ceil(T/32) * CPM
# where CPM (calls per M-tile, T-invariant) = sum over projections of device
# calls per M-tile. We use the observed call counts to solve; the model below is
# only a cross-check.
def solve_per_forward(regions, res, S_by_test):
    """Return {region_name: (mmd_us_per_fwd, glue_us_per_fwd, mmd_calls_per_fwd)}.
    Solves region_obs = 10*this + next_warmup using the next region's per-forward
    value as the warm-up unit. next_warmup is 1 forward of the NEXT T."""
    names = [n for n, _ in regions]
    out = {}
    # group the MMD_ regions per stage in order
    for i, name in enumerate(names):
        if not name.startswith("MMD_"):
            continue
        d = res[name]
        # find next MMD_ region in the SAME stage (same prefix before _T)
        stage = name.rsplit("_T", 1)[0]
        nxt = None
        for j in range(i + 1, len(names)):
            if names[j].startswith(stage + "_T"):
                nxt = names[j]
                break
        out[name] = dict(region=d, next=nxt)
    # solve from LAST T (next=None -> per_fwd = region/10) backward
    per = {}
    for name in reversed([n for n in names if n.startswith("MMD_")]):
        d = res[name]
        nxt = out[name]["next"]
        if nxt is None:
            mmd_pf = d["mmd_us"] / N_ITERS
            glue_pf = d["glue_us"] / N_ITERS
            calls_pf = d["mmd_calls"] / N_ITERS
        else:
            # next region's solved per-forward IS the warm-up unit
            mmd_pf = (d["mmd_us"] - per[nxt]["mmd_us"]) / N_ITERS
            glue_pf = (d["glue_us"] - per[nxt]["glue_us"]) / N_ITERS
            calls_pf = (d["mmd_calls"] - per[nxt]["mmd_calls"]) / N_ITERS
        per[name] = dict(mmd_us=mmd_pf, glue_us=glue_pf, mmd_calls=calls_pf)
    return per


# --------------------------------------------------------------------------- #
# deep-plan_10 §5.3: NATIVE per-forward solver (sibling of solve_per_forward).
# Native regions are isolated -- the per-test autouse T_test_<node> signpost fires
# BEFORE the next native warm-up, so each NATIVE_* region holds exactly N_ITERS *
# N_proj MatmulDeviceOperation rows with NO trailing warm-up contamination. So
# per-fwd = region_sum / N_ITERS, in the EXACT same per-block-forward unit as mmd.
# --------------------------------------------------------------------------- #
def solve_native_per_forward(regions, res):
    """Return {NATIVE_region: dict(matmul_us, glue_us, matmul_calls)} per single
    forward, summing MatmulDeviceOperation DEVICE FW within each NATIVE_* region."""
    out = {}
    for name, _ in regions:
        if name not in NATIVE_MARKERS:
            continue
        d = res[name]
        out[name] = dict(
            matmul_us=d["mm_native_us"] / N_ITERS,
            glue_us=d["glue_us"] / N_ITERS,
            matmul_calls=d["mm_native_calls"] / N_ITERS,
        )
    return out


def _stage_of_native(name):
    return name.split("NATIVE_", 1)[1]  # "SigLIP" | "VLM" | "DENOISE"


def _stage_of_mmd(name):
    # MMD_SigLIP_T256 -> ("SigLIP", 256)
    base = name[len("MMD_"):]
    stage, _, t = base.rpartition("_T")
    return stage, int(t)


def merge_table(mmd_per, native_per):
    """deep-plan_10 §5.4: ONE table keyed by stage with native + mmd in the SAME
    per-forward frame. Returns {stage: {native_us_fwd, mmd_by_T:{T:us}, best_T,
    best_us_fwd, ratio, verdict, native_calls}}."""
    table = {}
    for nname, nd in native_per.items():
        stage = _stage_of_native(nname)
        table.setdefault(stage, {})
        table[stage]["native_us_fwd"] = nd["matmul_us"]
        table[stage]["native_calls"] = nd["matmul_calls"]
    for mname, md in mmd_per.items():
        stage, T = _stage_of_mmd(mname)
        # normalize stage casing: MMD uses SigLIP/VLM/DENOISE
        table.setdefault(stage, {})
        table[stage].setdefault("mmd_by_T", {})
        table[stage]["mmd_by_T"][T] = dict(us_fwd=md["mmd_us"], calls=md["mmd_calls"])
    for stage, d in table.items():
        by_T = d.get("mmd_by_T", {})
        if by_T:
            best_T = min(by_T, key=lambda t: by_T[t]["us_fwd"])
            d["best_T"] = best_T
            d["best_us_fwd"] = by_T[best_T]["us_fwd"]
            nat = d.get("native_us_fwd")
            if nat:
                d["ratio"] = d["best_us_fwd"] / nat
                d["verdict"] = "mmd WINS" if d["best_us_fwd"] < nat else "mmd LOSES"
    return table


# --------------------------------------------------------------------------- #
# deep-plan_8 Target-1: denoise reuse three-leg extraction.
# Verdict bucket = matmul-op DEVICE FW sum ALONE (MatmulDecodeDeviceOperation for
# the resident/re-read legs; MatmulDeviceOperation for the native leg). Glue
# (per-step A-I2S + output-S2I + once-per-instance weight-stage I2S) is CONTEXT.
# N_PROJ=3 (gate/up/down); N_STEPS=10; N_ITERS replay forwards in the region.
# --------------------------------------------------------------------------- #
def parse_denoise(csv_path, n_proj=3, n_steps=10, n_iters=None):
    n_iters = N_ITERS if n_iters is None else n_iters
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    ci_dur = _find_col(header, "DEVICE FW DURATION [NS]", "DEVICE FW DURATION")
    ci_op = _find_col(header, "OP CODE", "OP TYPE")
    # find the single DENOISE region marker
    start = None
    name = None
    for idx, r in enumerate(rows[1:], start=1):
        joined = " ".join(r)
        for m in DENOISE_MARKERS:
            if m in joined:
                start = idx
                name = m
                break
        if start is not None:
            break
    if start is None:
        return None
    mmd_us = mm_native_us = 0.0
    mmd_calls = mm_native_calls = 0
    i2s_us = s2i_us = 0.0
    i2s_calls = s2i_calls = 0
    for r in rows[start:]:
        if len(r) <= max(ci_dur, ci_op):
            continue
        op = r[ci_op].strip()
        try:
            dur = float(r[ci_dur])
        except (ValueError, IndexError):
            continue
        us = dur / 1000.0
        if op == MMD:
            mmd_us += us; mmd_calls += 1
        elif op == MM_NATIVE:
            mm_native_us += us; mm_native_calls += 1
        elif op == I2S:
            i2s_us += us; i2s_calls += 1
        elif op == S2I:
            s2i_us += us; s2i_calls += 1
    # verdict matmul-op column: native leg uses MatmulDeviceOperation, mmd legs use MatmulDecode.
    # deep-plan_15 §3/§7.3: the gather_in0 reuse legs (DENOISE_RESIDENT/REREAD) are Matmul1D ->
    # MatmulDeviceOperation (MM_NATIVE), NOT MatmulDecodeDeviceOperation. DENOISE_GATHER=1 tells
    # parse_denoise to sum MM_NATIVE for those non-native legs too (the native leg path is
    # byte-identical -- it already keys on MM_NATIVE). Default (unset) = the deep-plan_8
    # matmul_decode reuse behavior (MMD for non-native legs).
    is_native = (name == "DENOISE_NATIVE")
    _gather_legs = os.environ.get("DENOISE_GATHER", "0") == "1"
    if is_native or _gather_legs:
        matmul_op_us = mm_native_us
        matmul_calls = mm_native_calls
    else:
        matmul_op_us = mmd_us
        matmul_calls = mmd_calls
    # CONTEXT bucket split (I2S = per-step A-I2S + once-per-instance weight-stage):
    #   resident leg: weight already staged in warm-up (outside signpost) -> in-region
    #     weight-stage I2S should be ~0; all in-region I2S = per-step A-I2S.
    #   re-read leg: weight re-staged every call -> weight-stage I2S = n_iters*n_steps*n_proj,
    #     but the same op-code also covers per-step A-I2S. We report total I2S + the
    #     EXPECTED per-leg op-counts for falsifiability; do not over-attribute.
    per_loop = lambda v: v / n_iters  # one 10-step denoise loop
    out = dict(
        region=name,
        matmul_op_us_region=matmul_op_us,
        matmul_op_us_per_loop=per_loop(matmul_op_us),
        matmul_calls_region=matmul_calls,
        # context buckets (per-loop)
        a_i2s_plus_out_s2i_us_per_loop=per_loop(i2s_us + s2i_us),  # bucket 2 (context)
        i2s_us_per_loop=per_loop(i2s_us),
        s2i_us_per_loop=per_loop(s2i_us),
        i2s_calls_region=i2s_calls,
        s2i_calls_region=s2i_calls,
        glue_us_per_loop=per_loop(i2s_us + s2i_us),
        total_us_per_loop=per_loop(matmul_op_us + i2s_us + s2i_us),  # CONTEXT total
        is_native=is_native,
    )
    return out


def report_denoise(paths):
    print("=== deep-plan_8 TARGET-1 DENOISE REUSE (matmul-op = VERDICT; glue = CONTEXT) ===")
    legs = {}
    for p in paths:
        d = parse_denoise(p)
        if d is None:
            print(f"  [skip] no DENOISE region in {p}")
            continue
        legs[d["region"]] = d
        print(f"\n-- {d['region']}  ({os.path.basename(p)})")
        print(f"   VERDICT matmul-op us/loop = {d['matmul_op_us_per_loop']:.2f}  "
              f"(region {d['matmul_op_us_region']:.2f} over {N_ITERS} loops, "
              f"{d['matmul_calls_region']} matmul rows)")
        print(f"   [CONTEXT] per-step A-I2S + out-S2I us/loop = "
              f"{d['a_i2s_plus_out_s2i_us_per_loop']:.2f} "
              f"(I2S {d['i2s_us_per_loop']:.2f} / S2I {d['s2i_us_per_loop']:.2f}; "
              f"I2S rows={d['i2s_calls_region']} S2I rows={d['s2i_calls_region']})")
        print(f"   [CONTEXT] TOTAL us/loop (matmul+glue, NON-DECIDING) = "
              f"{d['total_us_per_loop']:.2f}")
    if "DENOISE_NATIVE" in legs and "DENOISE_RESIDENT" in legs:
        nv = legs["DENOISE_NATIVE"]["matmul_op_us_per_loop"]
        rv = legs["DENOISE_RESIDENT"]["matmul_op_us_per_loop"]
        ratio = rv / nv if nv else float("nan")
        if rv < nv * 0.97:
            verdict = "WIN (resident matmul-op < native)"
        elif rv > nv * 1.03:
            verdict = "matmul-op LOSS (native < resident)"
        else:
            verdict = "TIE (within +-3%)"
        print("\n=== TARGET-1 VERDICT (matmul-op DEVICE FW per 10-step loop) ===")
        print(f"   native-10x  MatmulDeviceOperation     = {nv:.2f} us/loop")
        print(f"   resident-10x MatmulDecodeDeviceOp     = {rv:.2f} us/loop")
        if "DENOISE_REREAD" in legs:
            xv = legs["DENOISE_REREAD"]["matmul_op_us_per_loop"]
            print(f"   reread-10x   MatmulDecodeDeviceOp (ctx)= {xv:.2f} us/loop")
        print(f"   resident/native ratio = {ratio:.4f}  -> {verdict}")
        print("   NOTE: glue does NOT decide the verdict (binding addendum).")

        # deep-plan_15 §7.3: SEPARATE, distinctly-labeled SECONDARY block on the TOTAL loop
        # time (matmul+staging+glue). Here the staging/glue IS the subject -- the OPPOSITE of
        # the matmul-op verdict above (where glue does NOT decide). This NEVER alters the
        # KERNEL verdict; it is clearly-labeled CONTEXT over the 10-step replay loop.
        nT = legs["DENOISE_NATIVE"]["total_us_per_loop"]
        rT = legs["DENOISE_RESIDENT"]["total_us_per_loop"]
        ratioT = rT / nT if nT else float("nan")
        noise = float(os.environ.get("DENOISE_NOISE", "0.02"))
        if rT < nT * (1 - 2 * noise) and ratioT < 0.98:
            vT = "REUSE WIN (resident TOTAL < native TOTAL)"
        elif rT > nT * 1.02:
            vT = "LOSS (residency saving offset by glue/underfill)"
        else:
            vT = "TIE (honest null)"
        print("\n=== SECONDARY REUSE-TOTAL CONTEXT (10-step loop; NOT the KERNEL verdict) ===")
        print("   this ratio is over the TOTAL loop time (matmul+staging+glue); staging/glue is")
        print("   the SUBJECT here, the OPPOSITE of the matmul-op verdict above where glue does")
        print("   NOT decide. SECONDARY CONTEXT -- does NOT alter the KERNEL verdict.")
        print(f"   resident-10x-TOTAL = {rT:.2f} us/loop vs native-10x-TOTAL = {nT:.2f} us/loop, "
              f"ratio {ratioT:.4f} (noise +-{noise*100:.0f}%, n={N_ITERS}) -> {vT}")
        if "DENOISE_REREAD" in legs:
            xT = legs["DENOISE_REREAD"]["total_us_per_loop"]
            rg = legs["DENOISE_RESIDENT"]["glue_us_per_loop"]
            xg = legs["DENOISE_REREAD"]["glue_us_per_loop"]
            print(f"   reread-10x-TOTAL  = {xT:.2f} us/loop  "
                  f"(R-vs-X glue delta = {xg - rg:.2f} us/loop = residency-saved weight re-stage)")
            print(f"   R-vs-X I2S delta (region rows) = "
                  f"{legs['DENOISE_REREAD']['i2s_calls_region'] - legs['DENOISE_RESIDENT']['i2s_calls_region']} "
                  f"(expected ~150 = N_ITERS*N_STEPS*N_PROJ weight re-stages if cache OFF)")
        print(f"   deep-plan_8 prior (dirty fork e7023ed9): resident-TOTAL 1905.90 / native 1204.94 "
              f"/ reread 2249.42; I2S 0/150/300 (residency LOST on TOTAL there).")


# --------------------------------------------------------------------------- #
# deep-plan_8 Target-2: sequential full-M=288 VLM block extraction.
# Region marker MMD_VLM_FULLSEQ_T288[_<lever>]. matmul-op (VERDICT) = sum over
# MatmulDecodeDeviceOperation; glue (CONTEXT) = I2S + S2I. per-fwd = region / N_ITERS.
# --------------------------------------------------------------------------- #
NATIVE_TARGET_US = 1069.55
NATIVE_BAND = (1037.0, 1102.0)


def parse_fullseq(csv_path, n_iters=None):
    n_iters = N_ITERS if n_iters is None else n_iters
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    ci_dur = _find_col(header, "DEVICE FW DURATION [NS]", "DEVICE FW DURATION")
    ci_op = _find_col(header, "OP CODE", "OP TYPE")
    start = None
    name = None
    for idx, r in enumerate(rows[1:], start=1):
        joined = " ".join(r)
        if "MMD_VLM_FULLSEQ_T" in joined:
            tok = next((c for c in r if "MMD_VLM_FULLSEQ_T" in c), None)
            start = idx
            name = tok.strip() if tok else "MMD_VLM_FULLSEQ"
            break
    if start is None:
        return None
    mmd_us = i2s_us = s2i_us = 0.0
    mmd_calls = i2s_calls = s2i_calls = 0
    for r in rows[start:]:
        if len(r) <= max(ci_dur, ci_op):
            continue
        op = r[ci_op].strip()
        try:
            dur = float(r[ci_dur])
        except (ValueError, IndexError):
            continue
        us = dur / 1000.0
        if op == MMD:
            mmd_us += us; mmd_calls += 1
        elif op == I2S:
            i2s_us += us; i2s_calls += 1
        elif op == S2I:
            s2i_us += us; s2i_calls += 1
    return dict(
        region=name,
        matmul_op_us_per_fwd=mmd_us / n_iters,
        matmul_calls_per_fwd=mmd_calls / n_iters,
        glue_us_per_fwd=(i2s_us + s2i_us) / n_iters,
        total_us_per_fwd=(mmd_us + i2s_us + s2i_us) / n_iters,
        matmul_op_us_region=mmd_us, mmd_calls_region=mmd_calls,
        i2s_calls_region=i2s_calls, s2i_calls_region=s2i_calls,
    )


def report_fullseq(paths):
    print("=== deep-plan_8 TARGET-2 SEQUENTIAL FULL-M=288 VLM BLOCK "
          "(matmul-op = VERDICT vs native 1069.55; glue = CONTEXT) ===")
    for p in paths:
        d = parse_fullseq(p)
        if d is None:
            print(f"  [skip] no MMD_VLM_FULLSEQ region in {p}")
            continue
        mm = d["matmul_op_us_per_fwd"]
        gap = (mm - NATIVE_TARGET_US) / NATIVE_TARGET_US * 100.0
        in_band = NATIVE_BAND[0] <= mm <= NATIVE_BAND[1]
        print(f"\n-- {d['region']}  ({os.path.basename(p)})")
        print(f"   VERDICT matmul-op us/fwd = {mm:.2f}  "
              f"(region {d['matmul_op_us_region']:.2f} over {N_ITERS} fwds, "
              f"{d['matmul_calls_per_fwd']:.1f} matmul calls/fwd)")
        print(f"   gap vs native 1069.55 = {gap:+.2f}%  -> "
              f"{'+-3% TIE (PASS)' if in_band else 'OUT OF BAND'}")
        print(f"   [CONTEXT] glue us/fwd = {d['glue_us_per_fwd']:.2f} "
              f"(I2S rows={d['i2s_calls_region']} S2I rows={d['s2i_calls_region']}); "
              f"TOTAL us/fwd = {d['total_us_per_fwd']:.2f} (NON-DECIDING)")


def report_merge(paths):
    """deep-plan_10 §5.4/§6c: parse ALL given CSVs (mmd MMD_* + native NATIVE_*),
    solve both legs into the SAME per-forward frame, drift-audit observed vs expected
    call counts, and print the unified per-stage native-vs-best-T table."""
    mmd_per = {}
    native_per = {}
    for path in paths:
        res, regions = parse(path)
        print(f"== {path}")
        print("ORDERED REGIONS:", [r[0] for r in regions])
        per = solve_per_forward(regions, res, {})
        for n, d in per.items():
            mmd_per[n] = d
            print(f"  [mmd] {n}: mmd_us/fwd={d['mmd_us']:.2f} "
                  f"mmd_calls/fwd={d['mmd_calls']:.1f} glue_us/fwd={d['glue_us']:.2f}")
        nat = solve_native_per_forward(regions, res)
        for n, d in nat.items():
            native_per[n] = d
            print(f"  [native] {n}: matmul_us/fwd={d['matmul_us']:.2f} "
                  f"calls/fwd={d['matmul_calls']:.1f} glue_us/fwd={d['glue_us']:.2f}")
        print()
    # drift audit (native expected anchors)
    print("=== NATIVE DRIFT AUDIT (observed vs expected calls/fwd) ===")
    for n, d in native_per.items():
        stage = _stage_of_native(n)
        exp = NATIVE_CALLS_EXPECTED.get(stage)
        obs = round(d["matmul_calls"])
        flag = "OK" if exp is None or obs == exp else f"DRIFT! exp={exp}"
        print(f"  {n}: observed={obs} expected={exp} -> {flag}")
    table = merge_table(mmd_per, native_per)
    print("\n=== UNIFIED PER-STAGE NATIVE-vs-BEST-T TABLE (matmul-op DEVICE FW us/fwd) ===")
    for stage in ("SigLIP", "VLM", "DENOISE"):
        if stage not in table:
            continue
        d = table[stage]
        nat = d.get("native_us_fwd")
        print(f"\n-- {stage}: native_us/fwd={nat:.2f} (calls/fwd={d.get('native_calls', 0):.1f})"
              if nat is not None else f"\n-- {stage}: native=N/A")
        for T in sorted(d.get("mmd_by_T", {})):
            m = d["mmd_by_T"][T]
            r = (m["us_fwd"] / nat) if nat else float("nan")
            print(f"   T={T:<4} mmd_us/fwd={m['us_fwd']:8.2f}  calls/fwd={m['calls']:5.1f}  "
                  f"ratio(mmd/native)={r:.3f}")
        if "best_T" in d:
            print(f"   BEST T={d['best_T']} mmd_us/fwd={d['best_us_fwd']:.2f}  "
                  f"ratio={d.get('ratio', float('nan')):.3f}  -> {d.get('verdict', '?')}")
    return table


# --------------------------------------------------------------------------- #
# deep-plan_11 §5.5: WS2D / NAT2D per-forward solver + unified verdict table.
# Both legs emit MatmulDeviceOperation (MM_NATIVE). Per-T regions live in ONE test
# (one stage), each region holds N_ITERS measured forwards + the NEXT T's single
# warm-up; the LAST T's next marker is the following test's T_test_ signpost so its
# next-contribution = 0. Sibling of solve_per_forward, keyed on MM_NATIVE rows.
# --------------------------------------------------------------------------- #
def _is_ws2d(name):
    return any(name.startswith(p) for p in WS2D_PREFIXES)


def _stage_of_ws2d(name):
    """WS2D_SigLIP_T128 -> ('WS2D', 'SigLIP', 128); NAT2D_VLM_T32 -> ('NAT2D','VLM',32)."""
    leg = "WS2D" if name.startswith("WS2D_") else "NAT2D"
    base = name[len(leg) + 1:]
    stage, _, t = base.rpartition("_T")
    return leg, stage, int(t)


def solve_ws2d_per_forward(regions, res):
    """Return {region_name: dict(mm_us, mm_calls)} per single forward for every WS2D_/
    NAT2D_ region, using the warm-up-subtraction model (next same-leg-same-stage region's
    solved per-forward IS the warm-up unit)."""
    names = [n for n, _ in regions]
    ws_names = [n for n in names if _is_ws2d(n)]
    nxt_of = {}
    for i, name in enumerate(names):
        if not _is_ws2d(name):
            continue
        leg, stage, _ = _stage_of_ws2d(name)
        prefix = f"{leg}_{stage}_T"
        nxt = None
        for j in range(i + 1, len(names)):
            if names[j].startswith(prefix):
                nxt = names[j]
                break
        nxt_of[name] = nxt
    per = {}
    for name in reversed(ws_names):
        d = res[name]
        nxt = nxt_of[name]
        if nxt is None:
            mm_pf = d["mm_native_us"] / N_ITERS
            calls_pf = d["mm_native_calls"] / N_ITERS
        else:
            mm_pf = (d["mm_native_us"] - per[nxt]["mm_us"]) / N_ITERS
            calls_pf = (d["mm_native_calls"] - per[nxt]["mm_calls"]) / N_ITERS
        per[name] = dict(mm_us=mm_pf, mm_calls=calls_pf)
    return per


# Canonical auto-config baseline (profile_native_unchunked_stages, UNCHANGED).
CANONICAL_US = {"SigLIP": 304.35, "VLM": 1068.42}


def report_ws2d(paths):
    """deep-plan_11 §5.5/§7.4: parse ALL CSVs, solve WS2D_/NAT2D_ legs + NATIVE_*
    (canonical) into the SAME per-forward frame, and print the unified per-(stage,T)
    verdict table (ws2d-resident vs nat2d-DRAM airtight ratio + canonical ratio).

    eps band, marginal-beat replication, and strict-beat/not-lose gating are applied by
    the report writer using the printed numbers; this prints the raw solved per-forward
    table + ratios so the verdict is fully auditable."""
    ws2d_per = {}   # WS2D_<stage>_T<T> -> per-fwd
    nat2d_per = {}  # NAT2D_<stage>_T<T> -> per-fwd
    native_canon = {}  # stage -> canonical us/fwd (from NATIVE_* regions if present)
    for path in paths:
        res, regions = parse(path)
        print(f"== {path}")
        print("ORDERED REGIONS:", [r[0] for r in regions])
        per = solve_ws2d_per_forward(regions, res)
        for n, d in per.items():
            leg, stage, T = _stage_of_ws2d(n)
            (ws2d_per if leg == "WS2D" else nat2d_per)[(stage, T)] = d
            print(f"  [{leg}] {stage} T{T}: mm_us/fwd={d['mm_us']:.2f} "
                  f"calls/fwd={d['mm_calls']:.1f}")
        nat = solve_native_per_forward(regions, res)
        for n, d in nat.items():
            stage = _stage_of_native(n)
            native_canon[stage] = d["matmul_us"]
            print(f"  [canonical NATIVE] {stage}: matmul_us/fwd={d['matmul_us']:.2f} "
                  f"calls/fwd={d['matmul_calls']:.1f}")
        print()
    # eps band
    eps = float(os.environ.get("WS2D_EPS", "0.01"))
    print(f"=== UNIFIED WS2D PER-(stage,T) VERDICT TABLE (matmul-op DEVICE FW us/fwd) ===")
    print(f"    eps band = {eps:.4f}  (strict-beat iff ratio<1-eps; tie iff |r-1|<=eps; lose iff r>1+eps)")
    stages = sorted({s for (s, _) in ws2d_per} | {s for (s, _) in nat2d_per})
    table = {}
    for stage in stages:
        canon_lit = CANONICAL_US.get(stage)
        canon_meas = native_canon.get(stage)
        canon = canon_meas if canon_meas else canon_lit
        print(f"\n-- {stage}: canonical(literal)={canon_lit}  "
              f"canonical(measured)={canon_meas if canon_meas else 'N/A'}  "
              f"using={canon}")
        Ts = sorted({t for (s, t) in ws2d_per if s == stage}
                    | {t for (s, t) in nat2d_per if s == stage})
        table[stage] = {}
        for T in Ts:
            ws = ws2d_per.get((stage, T))
            na = nat2d_per.get((stage, T))
            ws_us = ws["mm_us"] if ws else float("nan")
            na_us = na["mm_us"] if na else float("nan")
            ratio = (ws_us / na_us) if (na and na_us) else float("nan")
            rcanon = (ws_us / canon) if canon else float("nan")
            if ratio != ratio:
                verdict = "N/A"
            elif ratio < 1.0 - eps:
                verdict = "STRICT-BEAT"
            elif abs(ratio - 1.0) <= eps:
                verdict = "TIE"
            else:
                verdict = "LOSS"
            not_lose_canon = (rcanon <= 1.0 + eps) if rcanon == rcanon else None
            table[stage][T] = dict(ws_us=ws_us, na_us=na_us, ratio=ratio,
                                   rcanon=rcanon, verdict=verdict,
                                   not_lose_canon=not_lose_canon)
            print(f"   T={T:<4} ws2d_res={ws_us:8.2f}  nat2d_dram={na_us:8.2f}  "
                  f"ratio(ws/nat2d)={ratio:.4f}  ratio_canon={rcanon:.4f}  "
                  f"-> {verdict}  not_lose_canon={not_lose_canon}")
    return table


def report_natexp(paths):
    """deep-plan_12 S2: best-explicit native per (stage, proj) = min us/CALL over the
    swept configs. Each NATEXP_<stage>_<proj>_<cfgid> region holds N_ITERS native
    MatmulDeviceOperation rows (one call/forward). Also reports any RESID_ regions
    (resident-leg gather_in0) as us/call for the tie comparison."""
    best = {}  # (stage, proj) -> (cfgid, us_per_call)
    resid = {}  # region -> us_per_call
    for path in paths:
        res, regions = parse(path)
        for name, _ in regions:
            d = res[name]
            if name.startswith("NATEXP_"):
                # NATEXP_<stage>_<proj>_<cfgid>
                body = name[len("NATEXP_"):]
                parts = body.split("_")
                stage = parts[0]
                proj = parts[1]
                cfgid = "_".join(parts[2:])
                calls = d["mm_native_calls"]
                if calls <= 0:
                    continue
                upc = d["mm_native_us"] / calls
                key = (stage, proj)
                if key not in best or upc < best[key][1]:
                    best[key] = (cfgid, upc)
            elif name.startswith("RESID_"):
                calls = d["mm_native_calls"]
                if calls > 0:
                    resid[name] = d["mm_native_us"] / calls
    print("=== BEST-EXPLICIT NATIVE (min us/call per stage.proj) ===")
    stages = {}
    for (stage, proj), (cfgid, upc) in sorted(best.items()):
        print(f"  {stage}.{proj:<5} best_cfg={cfgid:<20} us/call={upc:.3f}")
        stages.setdefault(stage, 0.0)
        stages[stage] += upc
    print("=== STAGE TOTAL (sum of per-proj best-explicit us/forward) ===")
    for stage, tot in sorted(stages.items()):
        print(f"  {stage}: best-explicit native total/forward = {tot:.3f} us")
    if resid:
        print("=== RESIDENT LEG (gather_in0) us/call ===")
        for name, upc in sorted(resid.items()):
            print(f"  {name}: us/call={upc:.3f}")
    return best, resid, stages


# --------------------------------------------------------------------------- #
# deep-plan_13 §10.6: per-projection UNIFIED-op (MatmulDecodeLinear) leg.
# Regions are MMD_<stage>_<proj>[_r<n>] from profile_unified_mmsweep_stages.py.
# Each region holds N_ITERS full-seq block-forwards; warm-up is OUTSIDE the
# signpost and each (stage,proj) is in its OWN tracy subprocess, so there is NO
# trailing warm-up contamination: per-forward = region MMD-FW sum / N_ITERS
# (the same isolated-region solve as solve_native_per_forward). Tracy-only.
# --------------------------------------------------------------------------- #
def parse_mmsweep(csv_path, n_iters=5):
    """Sum the matmul-op + glue DEVICE-time (METRIC col) per MMD_<stage>_<proj>[_r] region.

    deep-plan_15 §3: the matmul op-code summed is selected by MMSWEEP_OP. DEFAULT
    (unset/"mmd") sums MatmulDecodeDeviceOperation (MMD) -- byte-identical to all prior
    behavior. MMSWEEP_OP=native sums MatmulDeviceOperation (MM_NATIVE) -- the gather_in0
    ws2d path is Matmul1D -> MatmulDeviceOperation (NATIVE), not MMD; without this the
    ws2d rows sum to 0 us. A calls/fwd==0 result is the STOP sentinel (wrong op-code branch)."""
    _mm_op = MM_NATIVE if os.environ.get("MMSWEEP_OP", "mmd").lower() == "native" else MMD
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    ci_dur = _find_col(header, *_DUR_CANDS)
    ci_op = _find_col(header, "OP CODE", "OP TYPE")
    regions = []
    for idx, r in enumerate(rows[1:], start=1):
        tok = next((c for c in r if c.strip().startswith("MMD_")), None)
        if tok:
            regions.append((tok.strip(), idx))
    out = {}
    for ri, (name, start) in enumerate(regions):
        end = regions[ri + 1][1] if ri + 1 < len(regions) else len(rows)
        mmd_us = glue_us = 0.0
        mmd_calls = 0
        for r in rows[start:end]:
            if len(r) <= max(ci_dur, ci_op):
                continue
            op = r[ci_op].strip()
            try:
                dur = float(r[ci_dur])
            except (ValueError, IndexError):
                continue
            if op == _mm_op:
                mmd_us += dur / 1000.0
                mmd_calls += 1
            elif op in (I2S, S2I):
                glue_us += dur / 1000.0
        out[name] = dict(mmd_us_per_fwd=mmd_us / n_iters,
                         glue_us_per_fwd=glue_us / n_iters,
                         mmd_calls_per_fwd=mmd_calls / n_iters)
    return out


def report_mmsweep(paths, n_iters=5):
    """Print per-(stage,proj) unified-op mmd DEVICE FW us/block-forward. Handles _r<n>
    repeat regions for the epsilon noise-floor (groups them, reports spread)."""
    agg = {}  # (stage,proj) -> list of per-forward us (one per CSV / repeat region)
    glue = {}
    calls = {}
    for path in paths:
        d = parse_mmsweep(path, n_iters)
        for name, v in d.items():
            body = name[len("MMD_"):]
            # strip _r<n> repeat suffix
            if "_r" in body and body.rsplit("_r", 1)[1].isdigit():
                body = body.rsplit("_r", 1)[0]
            parts = body.split("_")
            stage, proj = parts[0], "_".join(parts[1:])
            key = (stage, proj)
            agg.setdefault(key, []).append(v["mmd_us_per_fwd"])
            glue[key] = v["glue_us_per_fwd"]
            calls[key] = v["mmd_calls_per_fwd"]
    print("=== UNIFIED OP (MatmulDecodeLinear) mmd DEVICE FW us/block-forward ===")
    for (stage, proj), vals in sorted(agg.items()):
        mn, mx = min(vals), max(vals)
        spread = (mx - mn) / mn if mn else 0.0
        nrep = len(vals)
        rep = f"  [reps={nrep} spread={spread*100:.3f}%]" if nrep > 1 else ""
        print(f"  {stage}.{proj:<6} mmd_us/fwd={sum(vals)/len(vals):.3f}  "
              f"calls/fwd={calls[(stage,proj)]:.1f}  glue_us/fwd={glue[(stage,proj)]:.2f}{rep}")
    return agg, glue, calls


if __name__ == "__main__":
    paths = sys.argv[1:]
    if os.environ.get("EXTRACT_MODE") == "mmsweep":
        report_mmsweep(paths, int(os.environ.get("N_ITERS", "5")))
        sys.exit(0)
    if os.environ.get("EXTRACT_MODE") == "natexp":
        report_natexp(paths)
        sys.exit(0)
    if os.environ.get("EXTRACT_MODE") == "ws2d":
        report_ws2d(paths)
        sys.exit(0)
    if os.environ.get("EXTRACT_MODE") == "merge":
        report_merge(paths)
        sys.exit(0)
    # OLD deep-plan_8 denoise reuse path: require the legacy DENOISE_ markers (the new
    # deep-plan_10 sweep CSVs use MMD_DENOISE_T and route through the default/merge path).
    if (os.environ.get("EXTRACT_MODE") == "denoise") or \
       any(("denoise" in os.path.basename(p).lower() and "sweep" not in os.path.basename(p).lower()
            and "mtdirect" not in os.path.basename(p).lower()) for p in paths):
        report_denoise(paths)
        sys.exit(0)
    if any("fullseq" in os.path.basename(p).lower() or "vlm_lever" in os.path.basename(p).lower()
           for p in paths) or (os.environ.get("EXTRACT_MODE") == "fullseq"):
        report_fullseq(paths)
        sys.exit(0)
    merged = {}
    region_order = []
    for path in paths:
        res, regions = parse(path)
        print(f"== {path}")
        print("ORDERED REGIONS:", [r[0] for r in regions])
        for name, d in res.items():
            print(f"  {name}: mmd_us={d['mmd_us']:.2f} mmd_calls={d['mmd_calls']} "
                  f"glue_us={d['glue_us']:.2f} i2s={d['i2s_calls']} s2i={d['s2i_calls']}")
        per = solve_per_forward(regions, res, {})
        merged.update(per)
        for n, _ in regions:
            if n.startswith("MMD_") and n not in region_order:
                region_order.append(n)
        print()
    print("=== PER SINGLE FORWARD (solved) ===")
    for name in region_order:
        p = merged[name]
        total = p["mmd_us"] + p["glue_us"]
        upc = p["mmd_us"] / p["mmd_calls"] if p["mmd_calls"] else 0.0
        print(f"{name}: mmd_us/fwd={p['mmd_us']:.2f}  mmd_calls/fwd={p['mmd_calls']:.1f}  "
              f"us/call={upc:.2f}  glue_us/fwd={p['glue_us']:.2f}  total_us/fwd={total:.2f}")
