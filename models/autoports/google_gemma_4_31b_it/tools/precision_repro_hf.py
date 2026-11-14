# SPDX-License-Identifier: Apache-2.0
"""Offline pinned HF CPU contingency using the exact saved diagnostic token IDs."""

import argparse
import copy
import hashlib
import json
import os
import time
import zlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", type=Path)
    group.add_argument("--cases", type=Path, help="Batch job manifest; --output is a directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--prefill-chunk", type=int, default=512)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--compare-dir",
        type=Path,
        action="append",
        default=[],
        help="TT batch result directory; repeat for served/full",
    )
    args = parser.parse_args()
    jobs = [{"name": "case", "case": str(args.case)}] if args.case else json.loads(args.cases.read_text())
    if not jobs:
        raise ValueError("No cases")
    for job in jobs:
        if Path(job["name"]).name != job["name"]:
            raise ValueError("Unsafe job name")
        case = json.loads(Path(job["case"]).read_text())
        tokens = case["prompt_token_ids"]
        if not tokens or any(type(x) is not int or x < 0 for x in tokens):
            raise ValueError("Invalid prompt token IDs")
    if args.tokens < 1 or args.prefill_chunk < 1:
        raise ValueError("Generation and prefill lengths must be positive")
    if args.validate_only:
        print(f"Validated {len(jobs)} saved-token cases; no model loaded")
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    from transformers import AutoTokenizer

    from models.autoports.google_gemma_4_31b_it.tests.generate_full_reference import load_reference
    from models.autoports.google_gemma_4_31b_it.tt.weights import HF_MODEL, HF_REVISION

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION, local_files_only=True)
    model = load_reference()
    assert next(model.parameters()).device.type == "cpu"
    assert next(model.parameters()).dtype == torch.bfloat16
    for job in jobs:
        case_path = Path(job["case"])
        case = json.loads(case_path.read_text())
        output_path = args.output if args.case else args.output / (job["name"] + ".hf.private.json")
        run_case(model, tokenizer, case_path, case, output_path, args, started, HF_REVISION)
        if args.compare_dir:
            report = json.loads(output_path.read_text())
            hf_rows = report["teacher_forced_chosen"]["positions"]
            report["tt_comparisons"] = {}
            for directory in args.compare_dir:
                tt_path = directory / (job["name"] + ".json")
                tt_rows = json.loads(tt_path.read_text())["continuations"]["chosen"]["positions"][: len(hf_rows)]
                if [r["forced_token"] for r in tt_rows] != [r["forced_token"] for r in hf_rows]:
                    raise ValueError("HF/TT teacher forcing tokens differ")
                report["tt_comparisons"][str(directory)] = {
                    "positions": len(hf_rows),
                    "top1_agreement": sum(a["top1_token"] == b["top1_token"] for a, b in zip(hf_rows, tt_rows))
                    / len(hf_rows),
                    "chosen_logprob_delta_hf_minus_tt": [
                        a["forced_token_logprob"] - b["forced_token_logprob"] for a, b in zip(hf_rows, tt_rows)
                    ],
                }
            write_report(output_path, report)


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


def run_case(model, tokenizer, case_path, case, output_path, args, started, revision):
    import torch

    tokens = case["prompt_token_ids"]
    ids = torch.tensor([tokens], dtype=torch.long)
    cache = None
    with torch.inference_mode():
        # Leave the last input token for generate, whose first output samples its logits.
        for start in range(0, len(tokens) - 1, args.prefill_chunk):
            end = min(start + args.prefill_chunk, len(tokens) - 1)
            output = model(
                input_ids=ids[:, start:end],
                attention_mask=torch.ones((1, end), dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            del output
            print("HF_PREFILL", end, len(tokens) - 1, round(time.monotonic() - started, 2), flush=True)
        report = {
            "status": "partial",
            "case": str(case_path),
            "case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
            "revision": revision,
            "prompt_tokens": len(tokens),
            "prompt_source": "Saved prompt_token_ids, no chat rerender",
            "device": "cpu",
            "weights_dtype": "bfloat16",
            "prefill_chunk": args.prefill_chunk,
            "sampling_note": "HF RNG differs from TT. Seeds are diagnostic, native request was unseeded.",
            "case_metadata": case.get("metadata", {}),
            "results": [],
        }
        chosen = case.get("continuations", {}).get("chosen", [])[:64]
        if chosen:
            teacher_cache = copy.deepcopy(cache)
            forced = torch.tensor([tokens[-1:] + chosen[:-1]], dtype=torch.long)
            output = model(
                input_ids=forced,
                attention_mask=torch.ones((1, len(tokens) + len(chosen) - 1), dtype=torch.long),
                past_key_values=teacher_cache,
                use_cache=True,
                logits_to_keep=len(chosen),
            )
            logprobs = torch.log_softmax(output.logits[0].float(), dim=-1)
            report["teacher_forced_chosen"] = {
                "probabilities": "untempered full vocabulary softmax",
                "positions": [
                    {
                        "position": i,
                        "forced_token": target,
                        "top1_token": int(logprobs[i].argmax()),
                        "forced_token_logprob": float(logprobs[i, target]),
                        "forced_token_probability": float(logprobs[i, target].exp()),
                        "top1_probability": float(logprobs[i].max().exp()),
                    }
                    for i, target in enumerate(chosen)
                ],
            }
            del output, logprobs, teacher_cache
            write_report(output_path, report)
        for seed in args.seeds:
            trial_cache = copy.deepcopy(cache)
            torch.manual_seed(seed)
            trial_start = time.monotonic()
            generated = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                past_key_values=trial_cache,
                max_new_tokens=args.tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                logits_to_keep=1,
            )[0, len(tokens) :].tolist()
            del trial_cache
            text = tokenizer.decode(generated, skip_special_tokens=False)
            tail = text[-4096:]
            ratio = len(zlib.compress(tail.encode())) / max(1, len(tail.encode()))
            report["results"].append(
                {
                    "seed": seed,
                    "generated_token_ids": generated,
                    "text": text,
                    "generated_tokens": len(generated),
                    "wall_s": time.monotonic() - trial_start,
                    "tail_ratio": ratio,
                    "tail_distinct": len(set(tail)),
                    "digest_min_chars_met": len(tail) >= 1000,
                    "requested_tokens": args.tokens,
                    "flag": len(tail) >= 1000 and (ratio < 0.05 or len(set(tail)) <= 10),
                }
            )
            report["total_wall_s"] = time.monotonic() - started
            report["status"] = "complete" if len(report["results"]) == len(args.seeds) else "partial"
            write_report(output_path, report)
            print("HF_TRIAL_COMPLETE", seed, len(generated), report["results"][-1]["flag"], flush=True)


if __name__ == "__main__":
    main()
