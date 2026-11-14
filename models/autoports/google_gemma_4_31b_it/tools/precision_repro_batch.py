# SPDX-License-Identifier: Apache-2.0
"""Run diagnostic cases serially with one model load; never a serving benchmark."""

import argparse
import json
import time
import zlib
from pathlib import Path

from models.autoports.google_gemma_4_31b_it.tools.precision_repro_probe import (
    close_generator_mesh,
    compare_reports,
    validate_case,
    write_private,
)


def timed_free_generate(generator, prompt, length, sampling, clock=None):
    """Time existing synchronous token readbacks; introduce no device synchronization."""
    clock = clock or time.monotonic
    original = generator._replay
    had_override = "_replay" in generator.__dict__
    state = {"calls": 0, "start": None, "end": None, "before": {}, "after": {}}

    def replay(*args, **kwargs):
        if kwargs.get("host_sampling", False) or not kwargs.get("read_from_device", True):
            raise ValueError("Decode timing requires canonical sampler and existing token readback")
        if state["start"] is None:
            state["before"] = dict(generator.counters)
            state["start"] = clock()
        value = original(*args, **kwargs)
        state["end"] = clock()
        state["after"] = dict(generator.counters)
        state["calls"] += 1
        return value

    generator._replay = replay
    try:
        ids = generator.generate(prompt, length, enable_trace=True, buffered=False, sampling_params=sampling)
    finally:
        if had_override:
            generator._replay = original
        else:
            del generator._replay
    seconds = state["end"] - state["start"] if state["calls"] else None
    metrics = {
        "decode_tokens": state["calls"],
        "decode_wall_s": seconds,
        "decode_tok_s_user": state["calls"] / seconds if seconds and seconds > 0 else None,
        "trace_counters_before_decode": state["before"],
        "trace_counters_after_decode": state["after"],
        "trace_counters_delta": {key: value - state["before"].get(key, 0) for key, value in state["after"].items()},
        "basis": "Batch1 wall from immediately before first _replay to return after last existing synchronous token-out readback; counts completed replay calls, includes host loop gaps, excludes prefill/first token and text decoding. No added device sync or profiler. Traced diagnostic rate, not serving benchmark.",
    }
    return ids, metrics


def run_case(generator, job, output_path, compare_path=None):
    import torch

    case = validate_case(json.loads(Path(job["case"]).read_text()))
    result = {
        "case": job["case"],
        "metadata": case.get("metadata", {}),
        "runtime_precision": generator.model.precision_config,
        "regime": "diagnostic traced TF with host logit readbacks; not serving performance",
        "probabilities": "untempered full-vocabulary softmax",
        "continuations": {},
    }
    started = time.monotonic()
    for name in job.get("branches", ["chosen", "correct"]):
        tokens = case["continuations"][name]
        rows = []

        def collect(step, predicted):
            logits = generator._host_logits(generator.logits).reshape(1, -1)[0]
            lp = torch.log_softmax(logits, dim=-1)
            target, top1 = tokens[step], int(logits.argmax())
            rows.append(
                {
                    "position": step,
                    "forced_token": target,
                    "top1_token": top1,
                    "forced_token_logprob": float(lp[target]),
                    "forced_token_probability": float(lp[target].exp()),
                    "top1_probability": float(lp[top1].exp()),
                }
            )
            if (step + 1) % 256 == 0:
                print(
                    "PRECISION_TF_PROGRESS",
                    job.get("name", "case"),
                    name,
                    step + 1,
                    len(tokens),
                    round(time.monotonic() - branch_start, 2),
                    flush=True,
                )
            return target

        branch_start = time.monotonic()
        generator.generate(
            case["prompt_token_ids"],
            len(tokens),
            next_input=collect,
            enable_trace=True,
            host_sampling=True,
            buffered=False,
        )
        elapsed = time.monotonic() - branch_start
        if len(rows) != len(tokens):
            raise RuntimeError("Incomplete teacher forcing")
        result["continuations"][name] = {
            "positions": rows,
            "tokens": len(rows),
            "conditioning_token_ids": case["prompt_token_ids"],
            "sequence_log_probability": sum(r["forced_token_logprob"] for r in rows),
            "diagnostic_wall_s_including_prefill": elapsed,
            "diagnostic_tokens_per_wall_second": len(tokens) / elapsed,
        }
        write_private(output_path, result)
    if job.get("free_run_tokens"):
        result["free_running"] = []
        metadata = case.get("metadata", {})
        for trial in range(job.get("free_run_trials", 3)):
            params = dict(metadata["sampling"], seed=metadata.get("seed", 42) + trial)
            trial_start = time.monotonic()
            ids, decode_timing = timed_free_generate(
                generator, case["prompt_token_ids"], job["free_run_tokens"], params
            )
            elapsed = time.monotonic() - trial_start
            text = generator.tokenizer.decode(ids, skip_special_tokens=False)
            tail = text[-4096:]
            ratio = len(zlib.compress(tail.encode())) / max(1, len(tail.encode()))
            result["free_running"].append(
                {
                    "sampling": params,
                    "token_ids": ids,
                    "text": text,
                    "tail_zlib_ratio": ratio,
                    "tail_distinct_chars": len(set(tail)),
                    "digest_min_chars_met": len(tail) >= 1000,
                    "flag": len(tail) >= 1000 and (ratio < 0.05 or len(set(tail)) <= 10),
                    "diagnostic_wall_s_including_prefill": elapsed,
                    "decode_timing": decode_timing,
                }
            )
            write_private(output_path, result)
    if compare_path:
        result["comparison"] = compare_reports(json.loads(compare_path.read_text()), result)
    result["diagnostic_total_wall_s"] = time.monotonic() - started
    write_private(output_path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="JSON list of named jobs")
    parser.add_argument("--build-kwargs-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compare-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    jobs = json.loads(args.cases.read_text())
    kwargs = json.loads(args.build_kwargs_file.read_text())
    if kwargs.get("batch_size") != 1:
        raise ValueError("Diagnostic requires batch1")
    names = []
    for job in jobs:
        if not job["name"] or Path(job["name"]).name != job["name"]:
            raise ValueError("Job name must be a filename component")
        names.append(job["name"])
        case = validate_case(json.loads(Path(job["case"]).read_text()))
        if not set(job.get("branches", ["chosen", "correct"])) <= set(case["continuations"]):
            raise ValueError("Missing branch")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate job names")
    if args.validate_only:
        print(f"Validated {len(jobs)} jobs without importing TTNN")
        return
    from readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device

    from models.autoports.google_gemma_4_31b_it.tt.generator import build_generator

    generator = None
    mesh = open_readiness_mesh_device("P300x2", "FABRIC_1D_RING")
    try:
        started = time.monotonic()
        generator = build_generator(model_dir=Path(__file__).parents[1], mesh_device=mesh, **kwargs)
        write_private(args.output_dir / "load.json", {"wall_s": time.monotonic() - started, "build_kwargs": kwargs})
        for job in jobs:
            print("PRECISION_CASE_START", job["name"], flush=True)
            compare_path = args.compare_dir / (job["name"] + ".json") if args.compare_dir else None
            run_case(generator, job, args.output_dir / (job["name"] + ".json"), compare_path)
            print("PRECISION_CASE_DONE", job["name"], flush=True)
    finally:
        close_generator_mesh(generator, mesh, close_readiness_mesh_device)


if __name__ == "__main__":
    main()
