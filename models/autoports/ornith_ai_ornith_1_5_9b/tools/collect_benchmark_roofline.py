#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collect measured full-phase host intervals and modeled Ornith TP4 work."""

import argparse
import hashlib
import importlib.util
import json
import math
import socket
from collections import Counter
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ornith_benchmark_timing", MODEL / "tt/benchmark_timing.py")
_timing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_timing)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    # Identical canonical representation to benchmark_stage.subsets.digest.
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def snapshot(path):
    path = Path(path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(30)
        client.connect(_timing.socket_address(path))
        client.sendall(b"snapshot\n")
        answer = client.recv(1024)
        if answer.strip() != b"OK":
            raise RuntimeError(f"Phase snapshot failed: {answer!r}")
    return json.loads(path.read_text())


def stored(record):
    return sum(record["stored_bytes_by_rank"])


def identify_requests(raw, events):
    starts = raw.get("start_times")
    if not starts or len(starts) != raw["completed"]:
        raise ValueError("Detailed client start_times required for exclusion of warmups and initial probe")
    first = min(starts)
    # vLLM stores perf_counter in start_times. A wall-clock or relative timestamp
    # cannot be silently converted into a guessed server window.
    if not events or abs(first - max(e["end"] for e in events)) > 86400:
        raise ValueError("Client/server detailed timestamps do not share the host monotonic clock")
    e2els = raw.get("e2els") or []
    last = max(s + t for s, t in zip(starts, e2els)) if len(e2els) == len(starts) else max(e["end"] for e in events)
    ids = set()
    prompt_lens = {}
    for event in events:
        if event["phase"] != "prefill" or not first <= event["start"] <= last:
            continue
        for req, start, end, original in zip(
            event["request_ids"], event["starts"], event["ends"], event["original_prompt_lens"]
        ):
            if start == 0 and end > 0:
                ids.add(req)
                prompt_lens[req] = original
    if len(ids) != raw["completed"]:
        raise ValueError(f"Measured request mapping found {len(ids)} unique admissions, expected {raw['completed']}")
    if sum(prompt_lens.values()) != raw["total_input_tokens"]:
        raise ValueError("Server prompt lengths differ from performance input-token total")
    selected = []
    for event in events:
        if not ids.intersection(event["request_ids"]):
            continue
        if set(event["request_ids"]) - ids:
            raise ValueError("Measured work overlaps requests outside the measured client set")
        selected.append(event)
    if not any(e["phase"] == "decode" for e in selected):
        raise ValueError("No completed decode evidence")
    outputs = Counter()
    for event in selected:
        if event["phase"] == "decode":
            outputs.update(event["request_ids"])
    expected_per_request = raw["total_output_tokens"] // raw["completed"] - 1
    if any(outputs[req] < expected_per_request for req in ids):
        raise ValueError(f"Missing decode completions; expected at least {expected_per_request} each: {dict(outputs)}")
    return selected, {
        "client_first_start": first,
        "client_last_completion": last,
        "request_ids": sorted(ids),
        "prompt_lengths": prompt_lens,
        "executed_decode_steps_by_request": dict(outputs),
        "delivered_decode_tokens_per_request": expected_per_request,
        "mapping": "Same-host perf_counter interval plus captured scheduler request IDs; initial probe and warmups precede min detailed start_times.",
    }


def phase_intervals(events):
    ordered = sorted(events, key=lambda e: (e["start"], e["id"]))
    prefills = []
    pending_start = None
    for event in ordered:
        if event["phase"] == "prefill":
            if pending_start is None:
                pending_start = event["start"]
            if event["completed"]:
                prefills.append([pending_start, event["end"]])
                pending_start = None
        elif pending_start is not None:
            raise ValueError(
                "Decode overlaps unfinished chunked prefill: no independent completion boundary; refuse guessed attribution"
            )
    if pending_start is not None:
        raise ValueError("Uncompleted prefill chunks in measured work")
    decodes = [e for e in ordered if e["phase"] == "decode"]
    if not prefills or not decodes:
        raise ValueError("Both complete phases are required")
    for pstart, pend in prefills:
        if any(d["start"] < pend and d["end"] > pstart for d in decodes):
            raise ValueError("Overlapping prefill/decode completion intervals cannot be independently attributed")
    # Include all scheduler gaps on a single non-overlapping timeline. A gap
    # between consecutive prefill calls with no decode submission belongs to
    # prefill. Gaps before/between/after decode submissions belong to decode.
    for i in range(len(prefills) - 1):
        left, right = prefills[i][1], prefills[i + 1][0]
        if not any(left <= d["start"] < right for d in decodes):
            prefills[i][1] = right
    begin = min(e["start"] for e in ordered)
    end = max(e["end"] for e in ordered)
    decode_intervals = []
    cursor = begin
    for start, stop in prefills:
        if start < cursor:
            raise ValueError("Overlapping prefill intervals")
        if start > cursor:
            decode_intervals.append([cursor, start])
        cursor = stop
    if cursor < end:
        decode_intervals.append([cursor, end])
    prefill_seconds = sum(b - a for a, b in prefills)
    decode_seconds = sum(b - a for a, b in decode_intervals)
    if prefill_seconds <= 0 or decode_seconds <= 0:
        raise ValueError("Non-positive phase time")
    assert abs(prefill_seconds + decode_seconds - (end - begin)) < 1e-6
    return {
        "prefill": prefills,
        "decode": decode_intervals,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_wall_seconds": end - begin,
        "method": "Host perf_counter: prefill adapter entry through terminal host readback; decode through existing async completion and host materialization. Union/complement on one timeline includes scheduler gaps; no added synchronization.",
    }


def account(metadata, events):
    layers = metadata["layers"]
    cfg = metadata["config"]
    policy = metadata["precision"]["requested"]
    if policy["id"] != "bfp8_all_lofi" or any(v != "LoFi" for v in policy["compute_fidelities"].values()):
        raise ValueError("Collector requires the measured selected bfp8_all_lofi policy")
    ranks = len(metadata["participating_devices"])
    if ranks != 4 or len(layers) != 32:
        raise ValueError("Collector requires all32 TP4")
    weights = sum(stored(weight) for layer in layers for weight in layer["weights"].values()) + stored(metadata["head"])
    norms = sum(stored(weight) for layer in layers for weight in layer["norms"].values())
    linear_layers = [layer for layer in layers if layer["kind"] == "linear_attention"]
    full_layers = len(layers) - len(linear_layers)
    recurrent = sum(stored(layer["recurrent"]) for layer in linear_layers)
    conv = sum(stored(layer["conv"]) for layer in linear_layers)
    conv_constants = sum(stored(tensor) for layer in linear_layers for tensor in layer.get("conv_constants", []))
    projection_parameters = 0
    for layer in layers:
        for name, weight in layer["weights"].items():
            for shape in weight["shard_shapes"]:
                k, n = shape[-2:]
                # Two padded 32-wide gates contain only eight useful columns.
                if name == "packed_input" and layer["kind"] == "linear_attention":
                    n -= 48
                projection_parameters += k * n
    prefill_by_kind = Counter()
    decode_bytes = Counter()
    per_step = []
    for event in events:
        if event["phase"] == "prefill":
            for start, end, original, discard in zip(
                event["starts"], event["ends"], event["original_prompt_lens"], event["discarded"]
            ):
                length = end - start
                prefill_by_kind["LoFi_projection"] += 2 * length * projection_parameters
                if not discard and end:
                    prefill_by_kind["LoFi_head"] += 2 * cfg["hidden_size"] * cfg["vocab_size"]
                # Useful semantic causal attention and recurrent contractions.
                causal_pairs = (end * (end + 1) - start * (start + 1)) // 2
                prefill_by_kind["HiFi2_attention"] += (
                    full_layers * 4 * cfg["num_attention_heads"] * cfg["head_dim"] * causal_pairs
                )
                prefill_by_kind["HiFi4_recurrent"] += (
                    len(linear_layers)
                    * 6
                    * cfg["linear_num_value_heads"]
                    * cfg["linear_key_head_dim"]
                    * cfg["linear_value_head_dim"]
                    * length
                )
        else:
            active_positions = [p for p, active in zip(event["positions"], event["active"]) if active]
            inactive = len(event["active"]) - sum(event["active"])
            capacity = len(event["active"])
            # FP8 tile has 1024 mantissas plus 64 exponent bytes. Logical paged
            # read rounds to 32 tokens; KV update writes one tile in each plane.
            kv_read = (
                sum(math.ceil((p + 1) / 32) * 32 for p in active_positions)
                * full_layers
                * 2
                * cfg["num_key_value_heads"]
                * cfg["head_dim"]
                * (17 / 16)
            )
            kv_write = (
                len(active_positions) * full_layers * 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * 32 * (17 / 16)
            )
            parts = {
                "projection_and_head_weights": weights,
                "norm_weights": norms,
                "kv_page_reads": kv_read,
                "kv_tile_writes": kv_write,
                "fp32_recurrent_read_write": 2 * recurrent,
                "bf16_conv_read_write": 2 * conv,
                "inactive_recurrent_snapshot_restore": 4 * recurrent * inactive / capacity,
                "masked_conv_snapshot_merge": 7 * conv if inactive else 0,
                "fp32_convolution_constants": conv_constants,
                "masked_conv_flags": len(linear_layers) * ranks * 2048 if inactive else 0,
                "embedding_rows": capacity * cfg["hidden_size"] * 2,
                # Head logits are BF16 tile-padded at B<=32: one write, one
                # sampler read. Sampling workspace is L1; no vocab gather.
                "bf16_logits_write_and_sample": 2 * 32 * cfg["vocab_size"] * 2,
                "sample_readback": 32 * 4,
            }
            decode_bytes.update(parts)
            per_step.append(
                {
                    "id": event["id"],
                    "submission_id": event.get("submission_id"),
                    "active_positions": active_positions,
                    "capacity": capacity,
                    "inactive_slots": inactive,
                    "dram_bytes": sum(parts.values()),
                }
            )
    flops = sum(prefill_by_kind.values())
    weighted_flops = sum(
        value * (4 if key.startswith("HiFi4") else 2 if key.startswith("HiFi2") else 1)
        for key, value in prefill_by_kind.items()
    )
    base_peak = ranks * math.prod(metadata["compute_grid"]) * 4096 * metadata["clock_hz"]
    return {
        "prefill_flops": flops,
        "prefill_flops_by_kind": dict(prefill_by_kind),
        "prefill_peak_flops_per_second": base_peak * flops / weighted_flops if weighted_flops else base_peak,
        "LoFi_peak_flops_per_second": base_peak,
        "decode_dram_bytes": sum(decode_bytes.values()),
        "decode_bytes_by_kind": dict(decode_bytes),
        "decode_peak_dram_bytes_per_second": ranks * metadata["dram_bytes_per_second_per_chip"],
        "decode_step_accounting": per_step,
        "work_method": "TP4 aggregate. Useful projection/head and causal-attention/recurrent contraction FLOPs, excluding tile padding, discarded work, scalar/SFPU ops and redundant chunk rebuilds. Peak is work-weighted harmonic LoFi/HiFi2/HiFi4 from observed compute grid at published 1.35GHz. Decode reads actual stored projection/head/norm bytes once per fixed-capacity step (batch shared), logical page-rounded BF8 KV read/write, FP32 recurrent/conv traffic including inactive snapshot/restore, embedding/logit/sample traffic. Includes all executed steps, including asynchronous pipeline tail beyond delivered tokens. DRAM is a logical traffic estimate, not physical bus transactions; repeated SDPA tile fetches and small scalar/dispatch traffic are not modeled.",
        "peak_source": metadata["peak_source"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--action", choices=("check", "collect"), required=True)
    args = parser.parse_args()
    run = args.run_dir
    server = json.loads((run / f"perf-b{args.concurrency}-server.json").read_text())
    data = snapshot(server["timing_snapshot"])
    metadata = data["metadata"]
    if metadata["layer_count"] != 32 or metadata["configured_layer_count"] != 32:
        raise ValueError("Timing instrumentation loaded a reduced layer model")
    if metadata["max_num_seqs"] != args.concurrency or metadata["max_model_len"] != server["max_model_len"]:
        raise ValueError("Timing instrumentation capacity differs from observed server")
    if "models.autoports.ornith_ai_ornith_1_5_9b" not in metadata["generator_module"]:
        raise ValueError("Wrong imported generator")
    # Validate accounting before accuracy, even when no warmup has run yet.
    account(metadata, [])
    if args.action == "check":
        print(f"READY b{args.concurrency}: buffered phases, all32 TP4, loaded tensor storage and exact import identity")
        return
    perf = run / f"perf-b{args.concurrency}.json"
    raw = json.loads(perf.read_text())
    if any(length != 4096 for length in raw["input_lens"]):
        raise ValueError("Performance transport did not preserve exact 4096-token prompts; repair before accuracy")
    selected, mapping = identify_requests(raw, data["events"])
    if any(set(event["request_ids"]).intersection(mapping["request_ids"]) for event in data.get("pending", [])):
        raise ValueError(
            "Measured requests still have an uncompleted async submission; retain snapshot and collect after normal server drain"
        )
    intervals = phase_intervals(selected)
    accounting = account(metadata, selected)
    artifact = run / f"roofline-b{args.concurrency}-evidence.json"
    artifact.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "request_mapping": mapping,
                "events": selected,
                "intervals": intervals,
                "accounting": accounting,
            },
            indent=2,
        )
        + "\n"
    )
    # Preserve the complete raw buffer so warmup/probe exclusion remains auditable.
    (run / f"roofline-b{args.concurrency}-raw.json").write_text(json.dumps(data, separators=(",", ":")))
    shared = {
        "timing_scope": "full_phase_wall_time",
        "timing_method": intervals["method"],
        "work_method": accounting["work_method"],
        "peak_source": accounting["peak_source"],
        "evidence": artifact.name,
        "evidence_sha256": sha(artifact),
    }
    entry = {
        "performance_sha256": sha(perf),
        "server_identity_sha256": digest(server),
        "requests": raw["completed"],
        "input_tokens": raw["total_input_tokens"],
        "output_tokens": raw["total_output_tokens"],
        "prefill": {
            **shared,
            "flops": accounting["prefill_flops"],
            "seconds": intervals["prefill_seconds"],
            "peak_flops_per_second": accounting["prefill_peak_flops_per_second"],
        },
        "decode": {
            **shared,
            "dram_bytes": accounting["decode_dram_bytes"],
            "seconds": intervals["decode_seconds"],
            "peak_dram_bytes_per_second": accounting["decode_peak_dram_bytes_per_second"],
        },
    }
    roofline = run / "roofline.json"
    results = json.loads(roofline.read_text()) if roofline.exists() else {}
    results[str(args.concurrency)] = entry
    roofline.write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"COLLECTED b{args.concurrency}: {entry['requests']} requests, {len(selected)} phase events, {intervals['total_wall_seconds']:.3f}s"
    )


if __name__ == "__main__":
    main()
