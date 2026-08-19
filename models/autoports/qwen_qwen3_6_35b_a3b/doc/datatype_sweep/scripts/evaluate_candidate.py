# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate one Qwen3.6-35B-A3B datatype-sweep candidate."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import build_generator
from models.common.readiness_check.mesh_device import (
    add_mesh_device_args,
    close_readiness_mesh_device,
    open_readiness_mesh_device,
)
from models.common.readiness_check.run_prefill_check import _run_one_entry_prefill
from models.common.readiness_check.run_teacher_forcing import _run_one_entry
from models.common.readiness_check.schema import Reference, load_reference
from models.common.readiness_check.teacher_forcing import TokenAccuracy


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(row["total"]) for row in rows)
    if total == 0:
        return {}
    out: dict[str, Any] = {
        "top1": sum(row["matches_top1"] for row in rows) / total,
        "top5": sum(row["matches_top5"] for row in rows) / total,
        "top100": sum(row["matches_top100"] for row in rows) / total,
        "matches_top1": sum(row["matches_top1"] for row in rows),
        "matches_top5": sum(row["matches_top5"] for row in rows),
        "matches_top100": sum(row["matches_top100"] for row in rows),
        "total": total,
        "k": rows[0]["k"],
    }
    elapsed_s = sum(float(row.get("elapsed_s", 0.0)) for row in rows)
    if elapsed_s > 0:
        out["elapsed_s"] = elapsed_s
        out["e2e_t_s_u"] = total / elapsed_s
    ttft = [float(row["ttft_ms"]) for row in rows if row.get("ttft_ms") is not None]
    if ttft:
        out["ttft_ms"] = sum(ttft) / len(ttft)
    decode_tokens = sum(float(row.get("decode_tokens", 0.0)) for row in rows)
    decode_elapsed_s = sum(float(row.get("decode_elapsed_s", 0.0)) for row in rows)
    if decode_elapsed_s > 0:
        out["decode_tokens"] = decode_tokens
        out["decode_elapsed_s"] = decode_elapsed_s
        out["decode_t_s_u"] = decode_tokens / decode_elapsed_s
    return out


def _reference_summary(reference: Reference, reference_path: Path) -> dict[str, Any]:
    return {
        "path": str(reference_path),
        "hf_model_id": reference.hf_model_id,
        "k": reference.k,
        "entries": len(reference.entries),
        "prompt_lengths": [int(entry.prompt_tokens.shape[1]) for entry in reference.entries],
        "generated_token_lengths": [int(entry.generated_tokens.shape[1]) for entry in reference.entries],
        "chat_template": "AIME24 main readiness reference",
    }


def _print_accuracy(label: str, row: dict[str, Any]) -> None:
    print(
        f"{label:<20} top1={row['top1']:.3f} ({row['matches_top1']}/{row['total']})  "
        f"top5={row['top5']:.3f} ({row['matches_top5']}/{row['total']})  "
        f"top{row['k']}={row['top100']:.3f} ({row['matches_top100']}/{row['total']})"
    )


def _run_prefill(generator, reference: Reference) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for entry_idx, entry in enumerate(reference.entries):
        if entry_idx > 0:
            generator.reset()
        row = _run_one_entry_prefill(generator=generator, entry=entry, reference=reference)
        rows.append(row)
        _print_accuracy(f"prefill[{entry_idx}]", row)
    aggregate = _aggregate(rows)
    _print_accuracy("prefill agg", aggregate)
    return rows, aggregate


def _run_teacher_forcing(generator, reference: Reference) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    acc = TokenAccuracy(reference)
    rows = []
    for entry_idx in range(acc.num_entries):
        if entry_idx > 0:
            generator.reset()
        row = _run_one_entry(generator=generator, acc=acc, entry_idx=entry_idx)
        rows.append(row)
        _print_accuracy(f"teacher[{entry_idx}]", row)
    aggregate = _aggregate(rows)
    _print_accuracy("teacher agg", aggregate)
    return rows, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--precision-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--top5-threshold", type=float, default=0.98)
    add_mesh_device_args(parser)
    args = parser.parse_args()

    reference = load_reference(args.reference)
    command = " ".join(shlex.quote(part) for part in sys.argv)
    start = time.perf_counter()
    mesh_device = open_readiness_mesh_device(args.mesh_device, args.fabric_config)
    try:
        hardware = {
            "mesh_device": args.mesh_device,
            "fabric_config": args.fabric_config,
            "mesh_shape": [int(dim) for dim in tuple(getattr(mesh_device, "shape", ()))],
            "num_devices": int(mesh_device.get_num_devices()),
            "product": "Blackhole p300c",
        }
        generator = build_generator(
            model_dir=args.model_dir.resolve(),
            mesh_device=mesh_device,
            precision_config=args.precision_config.resolve(),
        )
        try:
            runtime_policy = generator.model.describe_precision_policy()
            prefill_rows, prefill_aggregate = _run_prefill(generator, reference)
            generator.reset()
            teacher_rows, teacher_aggregate = _run_teacher_forcing(generator, reference)
            trace_counters = dict(generator.last_trace_counters)
        finally:
            teardown = getattr(generator, "teardown", None)
            if callable(teardown):
                teardown()
    finally:
        close_readiness_mesh_device(mesh_device, args.fabric_config)

    elapsed_s = time.perf_counter() - start
    trace_verified = bool(
        trace_counters.get("trace_replays", 0) > 0
        and trace_counters.get("host_sampling") is False
        and trace_counters.get("full_logits_readbacks", 0) == 0
    )
    passed = bool(
        teacher_aggregate.get("top1", 0.0) >= args.top1_threshold
        and teacher_aggregate.get("top5", 0.0) >= args.top5_threshold
        and trace_verified
    )
    payload = {
        "config_id": args.config_id,
        "precision_config_path": str(args.precision_config),
        "measurement_regime": "AIME24 chat-template prefill plus trace-verified teacher-forcing decode",
        "reference": _reference_summary(reference, args.reference),
        "prefill": {"per_entry": prefill_rows, "aggregate": prefill_aggregate},
        "teacher_forcing": {
            "per_entry": teacher_rows,
            "aggregate": teacher_aggregate,
            "trace_verified": trace_verified,
            "trace_counters": trace_counters,
        },
        "thresholds": {"top1": args.top1_threshold, "top5": args.top5_threshold},
        "status": "pass" if passed else "fail",
        "runtime_policy_summary": runtime_policy,
        "hardware": hardware,
        "command": command,
        "env": {
            "TT_READINESS_TRACE_REGION_SIZE": os.environ.get("TT_READINESS_TRACE_REGION_SIZE"),
            "TT_METAL_WATCHER_DISABLE_ETH": os.environ.get("TT_METAL_WATCHER_DISABLE_ETH"),
        },
        "elapsed_s": elapsed_s,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
