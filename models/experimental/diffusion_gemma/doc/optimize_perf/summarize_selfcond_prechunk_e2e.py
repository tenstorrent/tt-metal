# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build compact control/candidate/final-default evidence from traced E2E JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


FIELDS = (
    "DG_SELFCOND_PRECHUNK_EMBED",
    "resolved_selfcond_prechunk",
    "trace_region_size_bytes",
    "steps",
    "blocks",
    "prefill_s",
    "ttft_s",
    "per_block_latency_s",
    "sum_block_latency_s",
    "full_generation_s",
    "steady_block_latency_s",
    "tokens_per_block_per_s",
    "denoise_steps_per_block",
    "committed_sha",
    "text_head",
)
REPO_ROOT = Path(__file__).resolve().parents[5]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_provenance() -> dict:
    source_paths = (
        "models/experimental/diffusion_gemma/tt/self_conditioning.py",
        "models/experimental/diffusion_gemma/doc/optimize_perf/bench_lever_e2e.py",
        "models/experimental/diffusion_gemma/doc/optimize_perf/summarize_selfcond_prechunk_e2e.py",
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--", "models/experimental/diffusion_gemma"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    )
    checkpoint = Path("/home/zni/dg_models/diffusiongemma-26B-A4B-it")
    cmake_cache = REPO_ROOT / "build_Release/CMakeCache.txt"
    return {
        "model": "google/diffusiongemma-26B-A4B-it",
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": _sha256(checkpoint / "config.json"),
        "tokenizer_config_sha256": _sha256(checkpoint / "tokenizer_config.json"),
        "mesh": "P150x4",
        "mesh_shape": [1, 4],
        "tensor_parallel": 4,
        "max_seq_len": 1024,
        "canvas_length": 256,
        "prompt": "Explain what a diffusion language model is in one sentence.",
        "seed": 0,
        "source_head": head,
        "source_worktree_dirty": dirty,
        "source_sha256": {path: _sha256(REPO_ROOT / path) for path in source_paths},
        "build_cache_sha256": _sha256(cmake_cache),
        "enable_tracy": False,
        "raw_process_rows": "embedded in this artifact; process labels identify the five serialized outputs",
    }


def _read_one(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{path} must contain exactly one benchmark result")
    return value[0]


def _row(path: str, value: dict) -> dict:
    return {"process_label": Path(path).stem, **{field: value[field] for field in FIELDS}}


def _speedups(control: dict, candidate: dict) -> dict:
    return {
        "steady_block_latency_percent": round(
            (control["steady_block_latency_s"] / candidate["steady_block_latency_s"] - 1.0) * 100.0,
            3,
        ),
        "steady_tokens_per_s_percent": round(
            (candidate["tokens_per_block_per_s"] / control["tokens_per_block_per_s"] - 1.0) * 100.0,
            3,
        ),
        "ttft_percent": round((control["ttft_s"] / candidate["ttft_s"] - 1.0) * 100.0, 3),
        "full_generation_percent": round(
            (control["full_generation_s"] / candidate["full_generation_s"] - 1.0) * 100.0,
            3,
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--default", required=True)
    parser.add_argument("--control-12")
    parser.add_argument("--default-12")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    control = _read_one(args.control)
    candidate = _read_one(args.candidate)
    default = _read_one(args.default)
    rows = (control, candidate, default)
    if [row["resolved_selfcond_prechunk"] for row in rows] != [False, True, True]:
        raise ValueError("expected control/candidate/default resolved selectors [false, true, true]")
    if [row["DG_SELFCOND_PRECHUNK_EMBED"] for row in rows] != ["0", "1", "<unset>"]:
        raise ValueError("expected raw selectors 0, 1, and <unset>")
    for key in ("base_env", "trace_region_size_bytes", "steps", "blocks", "denoise_steps_per_block"):
        if not all(row[key] == control[key] for row in rows[1:]):
            raise ValueError(f"incomparable {key}")
    if not all(row["committed_sha"] == control["committed_sha"] for row in rows[1:]):
        raise ValueError("committed output mismatch")
    if control["steps"] != 48 or control["blocks"] != 3:
        raise ValueError("expected model-faithful 48-step, three-block workload")
    if control["trace_region_size_bytes"] != 10737418240:
        raise ValueError("expected documented 10 GiB trace reservation")

    result = {
        "date": "2026-07-09",
        "gate": "self-conditioning prechunk full-generation traced E2E",
        "sampling_scope": "RUN-first argmax (gumbel_noise=None); traced_denoise.py rejects real Gumbel noise",
        "production_sampler_evidence": {
            "decision_gate": "selfcond_prechunk_gumbel_decisions.json",
            "full_depth_256k": "selfcond_prechunk_256k_chunked.json",
            "traced_throughput_claimed": False,
        },
        "provenance": _source_provenance(),
        "base_env": control["base_env"],
        "control": _row(args.control, control),
        "candidate": _row(args.candidate, candidate),
        "final_default": _row(args.default, default),
        "candidate_speedup_vs_control_percent": _speedups(control, candidate),
        "final_default_speedup_vs_control_percent": _speedups(control, default),
        "committed_output_exact": True,
        "verdict": "PASS",
    }
    if bool(args.control_12) != bool(args.default_12):
        raise ValueError("--control-12 and --default-12 must be supplied together")
    if args.control_12:
        control_12 = _read_one(args.control_12)
        default_12 = _read_one(args.default_12)
        if control_12["DG_SELFCOND_PRECHUNK_EMBED"] != "0" or default_12["DG_SELFCOND_PRECHUNK_EMBED"] != "<unset>":
            raise ValueError("expected 12-step raw selectors 0 and <unset>")
        if control_12["resolved_selfcond_prechunk"] or not default_12["resolved_selfcond_prechunk"]:
            raise ValueError("unexpected 12-step resolved selectors")
        for key in ("base_env", "trace_region_size_bytes", "steps", "blocks", "denoise_steps_per_block"):
            if key == "steps":
                continue
            if control_12[key] != default_12[key]:
                raise ValueError(f"incomparable 12-step {key}")
        if control_12["steps"] != 12 or default_12["steps"] != 12:
            raise ValueError("expected 12-step slope points")
        if control_12["trace_region_size_bytes"] != control["trace_region_size_bytes"]:
            raise ValueError("12-step trace reservation differs from 48-step reservation")
        if control_12["base_env"] != control["base_env"]:
            raise ValueError("12-step base environment differs from 48-step environment")
        if control_12["committed_sha"] != default_12["committed_sha"]:
            raise ValueError("12-step committed output mismatch")
        step_delta = control["steps"] - control_12["steps"]
        control_step_ms = (
            (control["steady_block_latency_s"] - control_12["steady_block_latency_s"]) / step_delta * 1000.0
        )
        default_step_ms = (
            (default["steady_block_latency_s"] - default_12["steady_block_latency_s"]) / step_delta * 1000.0
        )
        result["12_step_slope_points"] = {
            "control": _row(args.control_12, control_12),
            "final_default": _row(args.default_12, default_12),
        }
        result["derived_warmed_traced_step"] = {
            "control_ms": round(control_step_ms, 3),
            "final_default_ms": round(default_step_ms, 3),
            "saving_ms": round(control_step_ms - default_step_ms, 3),
            "speedup_percent": round((control_step_ms / default_step_ms - 1.0) * 100.0, 3),
        }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("SELFCOND_E2E_SUMMARY " + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
