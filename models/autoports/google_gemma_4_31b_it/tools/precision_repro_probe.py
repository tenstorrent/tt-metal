# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only traced teacher forcing with full-vocabulary probabilities.

Input JSON: prompt_token_ids plus continuations mapping names to token-id lists.
Preserve exact failed-call token IDs; do not re-render or silently truncate them.
This adds readbacks for diagnosis and must never supply serving performance rates.
"""

import argparse
import json
from pathlib import Path


def write_private(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", opener=lambda name, flags: __import__("os").open(name, flags, 0o600)) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    path.chmod(0o600)


def close_generator_mesh(generator, mesh, close_mesh):
    """Release this model's traces, and always close its mesh even if release fails."""
    try:
        if generator is not None:
            release = getattr(generator, "release_persistent_capture", None)
            if callable(release):
                release()
            else:
                teardown = getattr(generator, "teardown", None)
                if callable(teardown):
                    teardown()
    finally:
        close_mesh(mesh, "FABRIC_1D_RING")


def validate_case(case):
    sequences = [case.get("prompt_token_ids")]
    continuations = case.get("continuations", {})
    if not isinstance(continuations, dict) or "chosen" not in continuations:
        raise ValueError("continuations must include a chosen token ID list")
    sequences.extend(continuations.values())
    for seq in sequences:
        if not isinstance(seq, list) or not seq or any(type(t) is not int or t < 0 for t in seq):
            raise ValueError("Expected nonempty lists of nonnegative integer token IDs")
    return case


def compare_reports(served, full):
    result = {}
    for name, entry in served["continuations"].items():
        other = full["continuations"][name]
        if entry["conditioning_token_ids"] != other["conditioning_token_ids"]:
            raise ValueError("Comparison requires identical conditioning token IDs")
        a, b = entry["positions"], other["positions"]
        if [(r["position"], r["forced_token"]) for r in a] != [(r["position"], r["forced_token"]) for r in b]:
            raise ValueError("Comparison requires identical continuation windows")
        result[name] = {
            "tokens": len(a),
            "top1_agreement": sum(x["top1_token"] == y["top1_token"] for x, y in zip(a, b)) / len(a),
            "served_sequence_log_probability": entry["sequence_log_probability"],
            "full_sequence_log_probability": other["sequence_log_probability"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--build-kwargs-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--start-token",
        type=int,
        default=0,
        help="Force earlier branch tokens as prompt, score from this reply position",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Explicit diagnostic window; omitted scores the whole continuation"
    )
    parser.add_argument(
        "--compare-report", type=Path, help="After full run, compare against served report over identical windows"
    )
    parser.add_argument(
        "--free-run-tokens", type=int, default=0, help="Also run device-sampled continuations from the saved prompt"
    )
    parser.add_argument(
        "--free-run-trials", type=int, default=3, help="Seeds metadata.seed, metadata.seed+1, ...; default three trials"
    )
    parser.add_argument("--free-run-only", action="store_true")
    args = parser.parse_args()
    if args.free_run_tokens < 0 or args.free_run_trials < 1 or (args.free_run_only and not args.free_run_tokens):
        parser.error("Invalid free-running trial configuration")
    case = validate_case(json.loads(args.case.read_text()))
    if args.start_token < 0 or (args.max_tokens is not None and args.max_tokens <= 0):
        raise ValueError("Invalid diagnostic window")
    if any(args.start_token >= len(t) for t in case["continuations"].values()):
        raise ValueError("Window starts beyond a continuation")
    kwargs = json.loads(args.build_kwargs_file.read_text())
    if kwargs.get("batch_size") != 1:
        raise ValueError("Diagnostic requires batch_size=1")
    if args.validate_only:
        print("case and batch schema valid; no hardware imported")
        return
    import torch
    from readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device

    from models.autoports.google_gemma_4_31b_it.tt.generator import build_generator

    mesh = open_readiness_mesh_device("P300x2", "FABRIC_1D_RING")
    generator = None
    output = {
        "case": str(args.case),
        "build_kwargs": kwargs,
        "regime": "traced diagnostic TF with per-position full-logit readback; not serving performance",
        "probabilities": "untempered full-vocabulary softmax; sequence probability stored as summed log probability",
        "window": {"start_token": args.start_token, "max_tokens": args.max_tokens},
        "prompt_token_ids": case["prompt_token_ids"],
        "continuations": {},
    }
    try:
        generator = build_generator(model_dir=Path(__file__).parents[1], mesh_device=mesh, **kwargs)
        output["runtime_precision"] = generator.model.precision_config
        output["sdpa_kernel_fidelity"] = generator.model.precision_config.get("sdpa_compute_fidelity", "HiFi2")
        if args.free_run_tokens:
            import zlib

            output["free_running"] = []
            metadata = case.get("metadata", {})
            sampling = dict(metadata.get("sampling", {"temperature": 1.0, "top_p": 0.95, "top_k": 20}))
            base_seed = metadata.get("seed", 42)
            for trial in range(args.free_run_trials):
                params = dict(sampling, seed=base_seed + trial)
                ids = generator.generate(
                    case["prompt_token_ids"],
                    args.free_run_tokens,
                    enable_trace=True,
                    buffered=False,
                    sampling_params=params,
                )
                text = generator.tokenizer.decode(ids, skip_special_tokens=False)
                tail = text[-4096:]
                ratio = len(zlib.compress(tail.encode())) / max(1, len(tail.encode()))
                output["free_running"].append(
                    {
                        "sampling": params,
                        "token_ids": ids,
                        "text": text,
                        "tail_zlib_ratio": ratio,
                        "tail_distinct_chars": len(set(tail)),
                        "digest_min_chars_met": len(tail) >= 1000,
                        "flag": len(tail) >= 1000 and (ratio < 0.05 or len(set(tail)) <= 10),
                    }
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_private(args.output, output)
        for name, branch in [] if args.free_run_only else case["continuations"].items():
            prompt = case["prompt_token_ids"] + branch[: args.start_token]
            tokens = branch[args.start_token :]
            if args.max_tokens is not None:
                tokens = tokens[: args.max_tokens]
            rows = []

            def collect(step, predicted):
                logits = generator._host_logits(generator.logits).reshape(1, -1)[0]
                logp = torch.log_softmax(logits, dim=-1)
                token = tokens[step]
                top1 = int(logits.argmax())
                rows.append(
                    {
                        "position": args.start_token + step,
                        "forced_token": token,
                        "top1_token": top1,
                        "forced_token_logprob": float(logp[token]),
                        "forced_token_probability": float(logp[token].exp()),
                        "top1_probability": float(logp[top1].exp()),
                    }
                )
                return token

            generator.generate(
                prompt, len(tokens), next_input=collect, enable_trace=True, host_sampling=True, buffered=False
            )
            if len(rows) != len(tokens):
                raise RuntimeError("Incomplete teacher forcing")
            output["continuations"][name] = {
                "positions": rows,
                "tokens": len(rows),
                "conditioning_token_ids": prompt,
                "sequence_log_probability": sum(row["forced_token_logprob"] for row in rows),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_private(args.output, output)
        if args.compare_report:
            served = json.loads(args.compare_report.read_text())
            output["comparison"] = compare_reports(served, output)
            write_private(args.output, output)
    finally:
        close_generator_mesh(generator, mesh, close_readiness_mesh_device)


if __name__ == "__main__":
    main()
