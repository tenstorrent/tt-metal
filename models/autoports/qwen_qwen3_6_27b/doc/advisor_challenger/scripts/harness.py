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

AND THROW THE FIRST PROCESS OF A SESSION AWAY. The floor above is a within-process quantity, but a large part
of the real one is not: the first harness process of a session recorded 11.838 us where the identical
configuration in a later process recorded 0.196 us -- 60x, from JIT-cache warmth BETWEEN processes, which no
per-process warm-up can touch. Since the floor decides feasibility.verdict, a cell whose control was the first
thing it ran silently changed what it was allowed to screen. Run once with --label warmup_discard and delete
the output; process_ordinal below records where each measurement sat in the session.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import replace
import time
from datetime import datetime, timezone

import torch
from transformers import AutoConfig

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
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import (
        LAYER as FULL_LAYER,
        _state as full_state,
    )
    from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import (
        LAYER as LINEAR_LAYER,
        _state as linear_state,
    )
    from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, _to_device
    from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder, resolve_policy

    kind = os.environ["CHALLENGER_LAYER_KIND"]
    assert kind in ("full_attention", "linear_attention"), kind
    config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True).text_config
    layer = FULL_LAYER if kind == "full_attention" else LINEAR_LAYER
    state_dict = full_state(config) if kind == "full_attention" else linear_state(config)
    candidate = policy.get("candidate")
    assert candidate, "frozen policy must name the executed candidate"
    frozen_policy = resolve_policy(candidate, kind)
    policy_updates = policy.get("policy_updates", {})
    executed_policy = replace(frozen_policy, **policy_updates)
    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=layer,
        mesh_device=device,
        batch=BATCH,
        max_context=64,
        page_size=64,
        candidate=candidate,
        policy_override=executed_policy,
    )
    torch.manual_seed(20260810)
    hidden = (torch.randn(BATCH, 1, config.hidden_size) * 0.2).bfloat16()
    hidden_tt = _to_device(hidden.reshape(1, 1, BATCH, config.hidden_size), mesh_device=device)
    page_table = _to_device(
        torch.arange(BATCH, dtype=torch.int32).reshape(BATCH, 1),
        mesh_device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    positions = _to_device(
        torch.zeros(BATCH, dtype=torch.uint32),
        mesh_device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    return {"decoder": decoder, "hidden": hidden_tt, "page_table": page_table, "positions": positions}


# ---- FILL 2: one decode step -------------------------------------------------------------------
def decode(state):
    """Run exactly one decode step. No host syncs inside; the caller brackets them."""
    return state["decoder"].decode_forward(
        hidden_states=state["hidden"],
        page_table=state["page_table"],
        current_positions=state["positions"],
    )


def _process_ordinal(out_path: str) -> int:
    """Which harness process of this session this is, counted in a marker file beside the artifacts.

    One line per process, so the count survives crashes and is auditable. The point is a single number in
    every record: a floor measured as ordinal 1 is not comparable with one measured later.
    """
    marker = os.path.join(os.path.dirname(out_path) or ".", ".harness_session")
    os.makedirs(os.path.dirname(marker) or ".", exist_ok=True)
    with open(marker, "a+") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {os.getpid()}\n")
        fh.flush()
        fh.seek(0)
        return sum(1 for _ in fh)


def _device_users() -> dict:
    """Which processes hold a Tenstorrent device open, right now.

    Hosts are shared and `tt-smi` reports board presence rather than utilisation, so there is NO
    retrospective evidence that a measurement had the device to itself -- two cells of the reference corpus
    predate this instrumentation and can never be shown clean. Sampled at both ends of every measurement so
    the question is answerable later instead of being argued about.

    Read from /proc rather than by shelling out, so it costs nothing and cannot fail the run.
    """
    users = []
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit() or int(pid) == os.getpid():
                continue
            fd_dir = f"/proc/{pid}/fd"
            try:
                for fd in os.listdir(fd_dir):
                    target = os.readlink(f"{fd_dir}/{fd}")
                    if "tenstorrent" in target:
                        with open(f"/proc/{pid}/cmdline", "rb") as fh:
                            cmd = fh.read().replace(b"\0", b" ").decode(errors="replace").strip()
                        users.append({"pid": int(pid), "device": target, "cmd": cmd[:160]})
                        break
            except (PermissionError, FileNotFoundError, OSError):
                continue          # a process that exited, or one owned by someone else -- not our business
    except Exception as exc:
        return {"error": f"{exc}", "note": "device users could not be sampled; state that in the README"}
    return {"count": len(users), "processes": users}


def measure(label: str, out_path: str, policy_path: str) -> dict:
    if WARMUP < MIN_WARMUP or REPEATS < MIN_REPEATS or ITERS < MIN_ITERS:
        raise SystemExit(f"protocol floor: WARMUP>={MIN_WARMUP} REPEATS>={MIN_REPEATS} ITERS>={MIN_ITERS}; "
                         f"got {WARMUP}/{REPEATS}/{ITERS}. These are why the corpus floors differed 45x.")

    ordinal = _process_ordinal(out_path)
    device_users_before = _device_users()
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
        if os.environ.get("CHALLENGER_PROFILE_EAGER") == "1":
            # Current profiler builds attribute trace-contained device ops to capture-time host
            # timestamps, outside these signposts. Execute the identical single decode graph eagerly
            # only for the op-level accounting window; all latency decisions above remain trace replay.
            decode(state)
        else:
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        signpost(header="PERF_DECODE_END")
        device_users_after = _device_users()
    finally:
        ttnn.close_mesh_device(mesh)

    record = {
        "label": label,
        "decode_batch": BATCH, "requested_decode_batch": REQUESTED_BATCH,
        "warmup_replays": WARMUP, "iters_per_repeat": ITERS,
        "process_ordinal": ordinal,
        # EXCLUSIVITY, sampled rather than assumed. Anything here beyond this process means the number
        # shared the device -- including an advisor capture, since the container running ttnn-advise maps
        # the same device. Sequence them; do not capture during a timed run.
        "device_users_at_start": device_users_before,
        "device_users_at_end": device_users_after,
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
    for when, du in (("start", device_users_before), ("end", device_users_after)):
        if du.get("count"):
            print(f"   !! NOT EXCLUSIVE at {when}: {du['count']} other process(es) hold a device open -- "
                  + "; ".join(f"{u['pid']} {u['cmd'][:60]}" for u in du["processes"][:3]))
    if ordinal == 1:
        print("   !! this is the FIRST harness process of the session, so this floor carries cross-process "
               "JIT-cache warm-up (measured 60x on one cell). Re-run with --label warmup_discard first and "
               "delete its output, then measure the incumbent.")
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
