# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Timing harness for 02b. Copy it, fill the two model hooks, do not touch the protocol.

  CHALLENGER_MODEL_DIR=<md> python harness_template.py --label incumbent \
      --out models/autoports/<md>/doc/advisor_challenger/incumbent.json

One process measures ONE configuration. That is deliberate: a candidate timed after the incumbent in the
same process is simply warmer, and the non-overlap rule assumes exchangeable samples. Run it again, in a
fresh process, for each candidate.

Why the protocol is fixed. Every cell in the reference corpus wrote its own harness, and the reported noise
floors ranged from 0.03 % to 1.37 % of the measured time -- a 45x spread driven entirely by protocol, not by
the hardware. The single cell that averaged inside each timed block is the single cell that found a material
win; two cells whose floor exceeded everything the advisor proposed could not have measured a contribution at
all. Two rules produce almost all of that difference:

  WARMUP >= 10   One corpus harness did exactly 1, and its first timed repeat then carried 73 % of the whole
                 reported spread -- a settling ramp misread as run-to-run variance. Discarding that one
                 sample moved the cell from unmeasurable to measurable.
  ITERS  >= 50   Each timed block reports the MEAN of ITERS replays, so the spread between blocks is the
                 spread of means, roughly sqrt(ITERS) tighter than single-shot timing.

incumbent_ms is the MEDIAN of the block means. Not the min: min-of-n is biased low by an amount that grows
with n, so cells with different n stop being comparable. All nine corpus cells recorded the min.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone

# ttnn is imported inside measure(), not here, so a bad protocol or a missing policy fails immediately
# instead of after the device is open.

# ---- THE PROTOCOL. Raise these if you like; the guards below refuse to go under. ----------------
WARMUP = int(os.environ.get("CHALLENGER_WARMUP", "10"))
REPEATS = int(os.environ.get("CHALLENGER_REPEATS", "5"))
ITERS = int(os.environ.get("CHALLENGER_ITERS", "50"))
MIN_WARMUP, MIN_REPEATS, MIN_ITERS = 10, 5, 50

MODEL_DIR = os.environ["CHALLENGER_MODEL_DIR"]
BATCH = int(os.environ["CHALLENGER_DECODE_BATCH"])       # the ONE batch this whole stage runs at
REQUESTED_BATCH = int(os.environ.get("CHALLENGER_REQUESTED_DECODE_BATCH", str(BATCH)))
# What the numbers below actually cover, recorded so nobody has to guess later. It is unset in all nine
# corpus cells, and they do not all measure the same thing: most time one decoder layer, while one reports
# a per-model composite computed as sum(layer_count x per-layer median) -- 937 ms of arithmetic, not a
# measurement. A derived metric's spread is the spread of medians and is not comparable to a real one.
HARNESS_SCOPE = os.environ.get(
    "CHALLENGER_HARNESS_SCOPE",
    f"one decoder layer, traced decode replay, batch {BATCH}, measured end to end on the host")


# ---- FILL 1: build the decoder under test ------------------------------------------------------
def build(device, policy: dict):
    """Return whatever `decode()` needs. Construct with `policy` -- the SHIPPED policy, never class
    defaults -- so the thing being timed is the thing that ships."""
    import torch
    import ttnn
    from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
        LAYER_IDX, _config, _page_table, _positions, _synthetic_state, _to_tt_decode,
    )
    from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import (
        OptimizationPolicy, OptimizedDecoder,
    )

    dtype_by_name = {
        "BFLOAT4_B": ttnn.bfloat4_b,
        "BFLOAT8_B": ttnn.bfloat8_b,
        "BFLOAT16": ttnn.bfloat16,
    }
    optimization_policy = OptimizationPolicy(
        attention_weight_dtype=dtype_by_name[policy["attention_weight_dtype"]],
        mlp_gate_up_weight_dtype=dtype_by_name[policy["mlp_gate_up_weight_dtype"]],
        mlp_down_weight_dtype=dtype_by_name[policy["mlp_down_weight_dtype"]],
        advisor_rope_l1=os.environ.get("CHALLENGER_ADVISOR_ROPE_L1", ""),
        advisor_sdpa_concat_l1=os.environ.get("CHALLENGER_ADVISOR_SDPA_CONCAT_L1", "0") == "1",
        advisor_norm_cores=int(os.environ.get("CHALLENGER_ADVISOR_NORM_CORES", "0")),
    )
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config), hf_config=config, layer_idx=LAYER_IDX,
        mesh_device=device, batch=BATCH, max_context=128,
        optimization_policy=optimization_policy,
    )
    hidden = torch.randn(BATCH, 1, config.hidden_size, generator=torch.Generator().manual_seed(9132)).bfloat16()
    key_cache, value_cache = decoder.create_paged_kv_cache()
    return {
        "decoder": decoder,
        "hidden": _to_tt_decode(hidden, device),
        "key_cache": key_cache,
        "value_cache": value_cache,
        "page_table": _page_table(BATCH, 128, device, permute=True),
        "positions": _positions([0] * BATCH, device),
    }


# ---- FILL 2: one decode step -------------------------------------------------------------------
def decode(state):
    """Run exactly one decode step. No host syncs inside; the caller brackets them."""
    return state["decoder"].decode_forward(
        state["hidden"], key_cache=state["key_cache"], value_cache=state["value_cache"],
        page_table=state["page_table"], current_positions=state["positions"], use_long_rope=False,
    )


def measure(label: str, out_path: str, policy_path: str) -> dict:
    if WARMUP < MIN_WARMUP or REPEATS < MIN_REPEATS or ITERS < MIN_ITERS:
        raise SystemExit(f"protocol floor: WARMUP>={MIN_WARMUP} REPEATS>={MIN_REPEATS} ITERS>={MIN_ITERS}; "
                         f"got {WARMUP}/{REPEATS}/{ITERS}. These are why the corpus floors differed 45x.")

    frozen = json.load(open(policy_path))
    for k in ("shipped_policy", "shipped_weight_dtypes"):
        if k not in frozen:
            raise SystemExit(
                f"{policy_path} has no {k!r}. Before timing anything, write a small JSON holding the policy "
                "that EXECUTED -- read it off the final tt-perf-report CSV or the selected candidate JSON "
                "from the datatype sweep, never resolved_policy.constructor_defaults:\n"
                '  {"shipped_policy": {...}, "shipped_weight_dtypes": {"attention": "BFLOAT8_B", ...},\n'
                '   "shipped_policy_source": "doc/datatype_sweep/selected_precision_config.json"}')
    policy = frozen["shipped_policy"]

    import ttnn
    try:
        from tracy import signpost
    except ImportError:                                 # profiling not built in; timing still works
        def signpost(*_a, **_k):
            pass

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        state = build(mesh, policy)

        decode(state)                                   # once eagerly, so compilation is not traced
        ttnn.synchronize_device(mesh)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        decode(state)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)

        for _ in range(WARMUP):                         # UNTIMED. See the module docstring.
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)

        repeats_ms = []
        for _ in range(REPEATS):
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(ITERS):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            repeats_ms.append((time.perf_counter() - start) * 1000.0 / ITERS)

        # A SEPARATE signposted replay for the op-level profile. Bounding exactly one replay is what keeps
        # tt-perf-report from covering several: three corpus cells committed unbounded reports (2x, 2x and
        # 19.1x), and every op share computed from those is smaller than the truth by that factor.
        ttnn.synchronize_device(mesh)
        signpost(header="PERF_DECODE")
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        signpost(header="PERF_DECODE_END")
    finally:
        ttnn.close_mesh_device(mesh)

    record = {
        "label": label,
        "decode_batch": BATCH, "requested_decode_batch": REQUESTED_BATCH,
        "warmup_replays": WARMUP, "iters_per_repeat": ITERS,
        "repeats_ms": repeats_ms,
        "median_ms": statistics.median(repeats_ms),              # MEDIAN, not min
        "noise_floor_ms": max(repeats_ms) - min(repeats_ms),
        "harness": os.path.relpath(__file__), "harness_scope": HARNESS_SCOPE,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shipped_policy": policy,
        "shipped_weight_dtypes": frozen["shipped_weight_dtypes"],
        "shipped_policy_source": frozen.get("shipped_policy_source") or policy_path,
        "signposts": ["PERF_DECODE", "PERF_DECODE_END"],
    }
    if label == "incumbent":
        record["incumbent_ms"] = record["median_ms"]   # the name the gate and reconcile.py read
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(record, open(out_path, "w"), indent=2)

    floor_us = record["noise_floor_ms"] * 1000
    print(f"{label}: median {record['median_ms']:.6f} ms over {REPEATS} blocks of {ITERS} "
          f"(+{WARMUP} warm-up), floor {floor_us:.3f} us "
          f"({100 * record['noise_floor_ms'] / record['median_ms']:.3f} %) -> {out_path}")
    print("   next: bound tt-perf-report with --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END, "
          "then reconcile.py --incumbent this file. A candidate goes in a FRESH process.")
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="incumbent", help="incumbent, or the candidate/chain being measured")
    ap.add_argument("--out", required=True, help="incumbent.json, or measurements/<candidate>.json")
    ap.add_argument("--policy", default=None,
                    help="JSON holding shipped_policy and shipped_weight_dtypes as they EXECUTED. Required "
                         "for the incumbent run; candidates default to the cell's frozen incumbent.json so "
                         "they cannot drift from the control.")
    a = ap.parse_args()
    default_policy = f"models/autoports/{MODEL_DIR}/doc/advisor_challenger/incumbent.json"
    if a.label == "incumbent" and not a.policy:
        raise SystemExit("--policy is required for the incumbent run: name the artifact the shipped policy "
                         "came from. A control built from class defaults measures the wrong decoder.")
    measure(a.label, a.out, a.policy or default_policy)
