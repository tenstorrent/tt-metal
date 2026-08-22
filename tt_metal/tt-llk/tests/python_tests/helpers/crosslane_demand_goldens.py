# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cross-lane arsenal: DEMAND-KERNEL golden fixtures (lane FB).

CRAQ-checkable reference goldens for the cross-lane cores of the target
kernels (the X7 migration lane's acceptance inputs).  Everything derives
from helpers/crosslane_oracle.py; fixtures are deterministic (splitmix32
seeds), stored as JSON with hex-u32 values under
tests/python_tests/crosslane_fixtures/.

Scope decisions (per lane-EX's topk_xl phase table and the EY-R design
input):
  - softmax_k       : lane-masked max fold over the first k of 8 columns,
                      recorded in BOTH formulations (SFPCONFIG lane-mask vs
                      computed vConstTileId predicate) -- they must agree
                      (the lane-EX dissolution claim).
  - moe_gate_topk   : top-k selection over 32 lanes under the sign-magnitude
                      total order (values + source indices), tie-free keys.
  - ema             : register-chain recurrence (scan-free reformulation --
                      consensus verdict: no scan primitive).  TWO arithmetic
                      contracts recorded ('fma' single-rounded, 'mul_add'
                      per-op rounded); the lowering picks one, a third value
                      on sim/silicon is a finding.
  - cumsum          : register-chain inclusive prefix, int32 (exact) and
                      fp32 (per-add rounding, serial order pinned).
  - bitonic stages  : per-stage traces of the bitonic networks for 8 and 32,
                      asc + desc, values-only (tie-free stimuli) and a KV
                      variant; scoped to the EX-liftable phases (merge +
                      phases 11-13 class) -- NO fixture depends on the FPU
                      Dst-face transpose (P21).
  - tie behavior    : a dedicated fixture carrying BOTH tie-mode variants
                      (doc vs pinned-sim) of an equal-key compare-exchange --
                      the recorded SFPSWAP tie divergence; consumers must not
                      bake either in until silicon adjudicates.
"""

from __future__ import annotations

import hashlib
import json
import os

from . import crosslane_oracle as co

FIXDIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "crosslane_fixtures")

LANES = co.LANES
M32 = co.M32


def _hex(v):
    if isinstance(v, list):
        return [_hex(x) for x in v]
    return f"{v & M32:08x}"


def _fkeys(seed, n=LANES, distinct=True):
    """Deterministic fp32-bit keys, finite, tie-free when distinct."""
    out = []
    seen = set()
    i = 0
    while len(out) < n:
        r = co.splitmix32(seed * 7919 + i)
        i += 1
        x = ((r % 2000) - 1000) + (r >> 20) / 4096.0
        b = co.f32_to_bits(x if r & 1 else -x)
        if distinct and b in seen:
            continue
        seen.add(b)
        out.append(b)
    return out


def gen_softmax_k():
    cases = []
    for k in range(1, 9):
        for seed in (11, 12):
            v = _fkeys(100 + seed)
            mask_form, pred_form = co.softmax_k_masked_fold(v, k, "max")
            assert mask_form == pred_form, "dissolution equivalence broke"
            cases.append({
                "k": k, "seed": seed,
                "input": _hex(v),
                "masked_max_all_lanes": _hex(mask_form),
                "formulations": "lane-mask == vConstTileId-predicate "
                                "(asserted equal at generation)",
            })
    return {
        "kernel": "softmax_k",
        "core": "per-row-of-8 masked max fold, first k columns participate; "
                "all lanes of the row receive the fold; identity -inf "
                "(0xFF800000); fold tree order = rotate 1,2,4 (pinned)",
        "cases": cases,
    }


def gen_moe_gate_topk():
    cases = []
    for k in (1, 2, 4, 8):
        for seed in (21, 22):
            keys = _fkeys(200 + seed)
            vals, idxs = co.topk_select(keys, k, "desc")
            cases.append({
                "k": k, "seed": seed,
                "keys": _hex(keys),
                "topk_values_desc": _hex(vals),
                "topk_indices_desc": idxs,
            })
    return {
        "kernel": "moe_gate_topk",
        "core": "top-k over 32 lanes, sign-magnitude total order "
                "(-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN), "
                "descending; keys tie-free (tie companions ride the "
                "tie_behavior fixture until silicon adjudication)",
        "cases": cases,
    }


def gen_ema():
    alpha = co.f32_to_bits(0.7)
    cases = []
    for seed in (31, 32):
        x_rows = [_fkeys(300 + seed * 10 + r) for r in range(8)]
        y0 = _fkeys(390 + seed)
        outs = co.ema_rowchain(x_rows, alpha, y0)
        cases.append({
            "seed": seed,
            "alpha": _hex(alpha),
            "x_rows": _hex(x_rows),
            "y0": _hex(y0),
            "out_rows_fma": _hex(outs["fma"]),
            "out_rows_mul_add": _hex(outs["mul_add"]),
        })
    return {
        "kernel": "ema",
        "core": "y_i = alpha*x_i + (1-alpha)*y_{i-1} along the 8-register "
                "chain (scan-free reformulation; no scan primitive exists). "
                "TWO arithmetic contracts recorded; the lowering pins one -- "
                "a third value observed on sim/silicon is a finding",
        "cases": cases,
    }


def gen_cumsum():
    cases = []
    for seed in (41, 42):
        int_rows = [[co.splitmix32(seed * 100 + r * 37 + l)
                     for l in range(LANES)] for r in range(8)]
        fp_rows = [_fkeys(400 + seed * 10 + r) for r in range(8)]
        cases.append({
            "seed": seed,
            "int_rows": _hex(int_rows),
            "int_prefix": _hex(co.cumsum_rowchain(int_rows, "int")),
            "fp_rows": _hex(fp_rows),
            "fp_prefix": _hex(co.cumsum_rowchain(fp_rows, "fp32")),
        })
    return {
        "kernel": "cumsum",
        "core": "inclusive prefix along the 8-register chain; int32 exact "
                "mod 2^32; fp32 serial low-to-high, one rounding per add "
                "(order is CONTRACT)",
        "cases": cases,
    }


def gen_bitonic():
    cases = []
    for n in (8, 32):
        for order in ("asc", "desc"):
            for seed in (51, 52):
                vals = _fkeys(500 + seed * 10 + n)[:n]
                out, trace = co.bitonic_sort_trace(vals, order)
                cases.append({
                    "n": n, "order": order, "seed": seed, "kv": False,
                    "input": _hex(vals),
                    "stages": _hex(trace),
                    "sorted": _hex(out),
                })
    # KV variant (tie-free keys -> tie divergence cannot bite)
    for order in ("asc", "desc"):
        keys = _fkeys(600)
        pay = [0xC0DE0000 | i for i in range(LANES)]
        ks, ps, trace = co.bitonic_sort_kv_trace(keys, pay, order)
        cases.append({
            "n": LANES, "order": order, "seed": 600, "kv": True,
            "keys": _hex(keys), "payloads": _hex(pay),
            "stages": [{"keys": _hex(k), "payloads": _hex(p)}
                       for k, p in trace],
            "sorted_keys": _hex(ks), "sorted_payloads": _hex(ps),
        })
    return {
        "kernel": "bitonic_compare_exchange_stages",
        "core": "standard bitonic network (stage list = "
                "bitonic_network_stages(n)); compare-exchange = SFPSWAP "
                "sign-magnitude semantics; per-stage traces let X7 pinpoint "
                "the first diverging stage; stimuli tie-free by construction",
        "cases": cases,
    }


def gen_tie_behavior():
    pos = [co.f32_to_bits(2.5)] * LANES
    neg = [co.f32_to_bits(-2.5)] * LANES
    ca = co.lane_id_sentinels(1)
    cb = co.lane_id_sentinels(2)
    out = {}
    for name, keys in (("equal_positive", pos), ("equal_negative", neg)):
        for mod in (1, 9):
            for tie in ("doc", "sim"):
                r = co.sfpswap_indexed(keys, keys, ca, cb, mod, tie=tie)
                out[f"{name}_mod{mod}_{tie}"] = {
                    "companion_vc_out": _hex(r[2]),
                    "companion_vd_out": _hex(r[3]),
                }
    return {
        "kernel": "sfpswap_tie_behavior",
        "core": "UNRESOLVED doc-vs-sim divergence (lane FB, 2026-08-21): "
                "SFPSWAP.md keys tie-swaps on SIGN (min lanes swap equal "
                "negatives, max lanes equal positives); pinned craq-sim "
                "9f324140 uses min:c<d / max:c>=d (no sign arm).  Visible "
                "only via ENABLE_DEST_INDEX companions (argmin/argmax "
                "ties).  Consumers MUST NOT depend on tie companion "
                "movement until silicon adjudicates; both variants below.",
        "companions_in": {"vc": _hex(ca), "vd": _hex(cb)},
        "variants": out,
    }


GENERATORS = {
    "softmax_k": gen_softmax_k,
    "moe_gate_topk": gen_moe_gate_topk,
    "ema": gen_ema,
    "cumsum": gen_cumsum,
    "bitonic_stages": gen_bitonic,
    "tie_behavior": gen_tie_behavior,
}


def generate_all():
    return {name: gen() for name, gen in GENERATORS.items()}


def fixture_path(name):
    return os.path.join(FIXDIR, f"{name}.json")


def write_fixtures():
    os.makedirs(FIXDIR, exist_ok=True)
    sums = []
    for name, data in generate_all().items():
        blob = json.dumps(data, indent=1, sort_keys=True) + "\n"
        with open(fixture_path(name), "w") as f:
            f.write(blob)
        sums.append((name, hashlib.sha256(blob.encode()).hexdigest()))
    with open(os.path.join(FIXDIR, "SHA256SUMS"), "w") as f:
        for name, h in sums:
            f.write(f"{h}  {name}.json\n")
    return sums


if __name__ == "__main__":
    for name, h in write_fixtures():
        print(f"{h}  {name}.json")
