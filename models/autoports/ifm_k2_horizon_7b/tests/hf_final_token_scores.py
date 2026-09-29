"""Score final TT continuation tokens with one CPU-only, causally aligned HF forward."""

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "IFM/K2-Horizon-7B"
REVISION = "036114ce8d46c32b24c15423211069abb9c5d25e"
DOC = Path("models/autoports/ifm_k2_horizon_7b/doc/full_model")


def load_source(path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def summary(rows):
    misses = [r for r in rows if r["rank"] != 1]
    gaps = torch.tensor([r["gap_to_hf_top1"] for r in rows])
    return {
        "tokens": len(rows),
        "top1_matches": sum(r["rank"] == 1 for r in rows),
        "top1_tied_matches": sum(r["rank_strict"] == 1 for r in rows),
        "top5_matches": sum(r["in_top5"] for r in rows),
        "top100_matches": sum(r["in_top100"] for r in rows),
        "top5_tie_inclusive_matches": sum(r["rank_strict"] <= 5 for r in rows),
        "top100_tie_inclusive_matches": sum(r["rank_strict"] <= 100 for r in rows),
        "mean_target_negative_log_probability": sum(-r["log_probability"] for r in rows) / len(rows),
        "worst_rank": max(r["rank"] for r in rows),
        "gap_quantiles": {
            name: float(torch.quantile(gaps, q)) for name, q in (("median", 0.5), ("p90", 0.9), ("max", 1.0))
        },
        "top1_miss_gap_counts": {
            "total": len(misses),
            **{f"at_most_{bound}": sum(r["gap_to_hf_top1"] <= bound for r in misses) for bound in (0.5, 1.0, 2.0)},
            **{f"at_least_{bound}": sum(r["gap_to_hf_top1"] >= bound for r in misses) for bound in (2.0, 5.0, 10.0)},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tt-source", type=Path, default=DOC / "qualitative_head_bfp8_hifi2.json")
    parser.add_argument("--hf-source", type=Path, default=DOC / "hf_qualitative_extended.json")
    parser.add_argument("--id", default="shared_1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DOC / "hf_final_token_scores.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tt_records, tt_source = load_source(args.tt_source)
    hf_records, hf_source = load_source(args.hf_source)
    tt = {r["id"]: r for r in tt_records}[args.id]
    hf = {r["id"]: r for r in hf_records}[args.id]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
    rendered = tok.apply_chat_template(hf["messages"], tokenize=False, add_generation_prompt=True)
    prompt = tok.apply_chat_template(hf["messages"], tokenize=True, add_generation_prompt=True, return_dict=False)
    assert rendered == hf["rendered_prompt"]
    assert prompt == hf["prompt_token_ids"] == tt["prompt_token_ids"]
    targets = tt["tt_token_ids"]
    assert len(targets) == 512 and tok.eos_token_id not in targets
    prompt_len = len(prompt)
    batch = torch.tensor([prompt + targets], dtype=torch.int64)
    # Logits at absolute position p predict the token at p+1. Keep positions
    # prompt_len-1 through prompt_len+511-1; never score a target against itself.
    prediction_positions = torch.arange(prompt_len - 1, prompt_len + len(targets) - 1)
    assert batch[0, prediction_positions + 1].tolist() == targets
    metadata = {
        "diagnostic": "HF scores under forced final-TT token history; not independent HF generation",
        "hf_model": MODEL,
        "revision": REVISION,
        "layers": 36,
        "sources": {"tt": tt_source, "independent_hf": hf_source},
        "prompt_id": args.id,
        "messages": hf["messages"],
        "rendered_prompt": rendered,
        "prompt_token_ids": prompt,
        "prompt_mode": "chat",
        "chat_template_present": bool(tok.chat_template),
        "tokenizer_class": type(tok).__name__,
        "device": "cpu",
        "offline": True,
        "threads": args.threads,
        "interop_threads": 1,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "forward": {
            "batch_size": 1,
            "input_tokens": batch.shape[1],
            "generated_tokens": len(targets),
            "use_cache": False,
            "generation": False,
            "attention_mask": "all ones",
            "selected_logit_positions_inclusive": [int(prediction_positions[0]), int(prediction_positions[-1])],
        },
        "alignment": "Generated index i is input token at prompt_len+i; scored with logits at prompt_len+i-1.",
        "rank_definition": "1-based descending logit; ascending token ID breaks ties. rank_strict counts only strictly larger logits; rank_tie_end counts greater or equal.",
        "page_size": 32,
        "interpretation_limit": "Token likelihood on a forced TT trajectory does not replace independent or exact-prefix HF controls, explain a root cause, or waive the stress failure.",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": f"python -m models.autoports.ifm_k2_horizon_7b.tests.hf_final_token_scores --threads {args.threads} --id {args.id} --tt-source {args.tt_source} --hf-source {args.hf_source} --output {args.output}",
    }
    print(json.dumps({"event": "load", "metadata": metadata}), flush=True)
    started = time.perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
        .eval()
        .to("cpu")
    )
    assert model.config.num_hidden_layers == len(model.model.layers) == 36
    metadata["model_dtype"] = str(next(model.parameters()).dtype)
    metadata["attention_backend"] = model.config._attn_implementation
    metadata["load_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    with torch.no_grad():
        logits = model(
            batch, attention_mask=torch.ones_like(batch), use_cache=False, logits_to_keep=prediction_positions
        ).logits[0]
    metadata["forward_seconds"] = time.perf_counter() - started
    assert logits.shape[0] == len(targets)
    rows = []
    for i, token_id in enumerate(targets):
        values = logits[i].float()
        assert torch.isfinite(values).all()
        value = values[token_id]
        rank_strict = int((values > value).sum()) + 1
        rank = rank_strict + int((values[:token_id] == value).sum())
        # Include cutoff ties before applying the declared token-ID tie break.
        cutoff = values.topk(5).values[-1]
        candidates = torch.where(values >= cutoff)[0].tolist()
        top5 = sorted(candidates, key=lambda index: (-float(values[index]), index))[:5]
        normalization = float(values.logsumexp(0))
        log_probability = float(value) - normalization
        context_position, target_position = prompt_len + i - 1, prompt_len + i
        rows.append(
            {
                "generated_index": i,
                "target_absolute_position": target_position,
                "context_last_absolute_position": context_position,
                "target_page": target_position // 32,
                "target_page_offset": target_position % 32,
                "context_last_page_offset": context_position % 32,
                "token_id": token_id,
                "token_text": tok.decode([token_id], skip_special_tokens=False),
                "rank": rank,
                "rank_strict": rank_strict,
                "rank_tie_end": int((values >= value).sum()),
                "in_top5": rank <= 5,
                "in_top100": rank <= 100,
                "logit": float(value),
                "log_probability": log_probability,
                "probability": math.exp(log_probability),
                "gap_to_hf_top1": float(values[top5[0]] - value),
                "hf_top1_id": top5[0],
                "hf_top1_text": tok.decode([top5[0]], skip_special_tokens=False),
                "hf_top1_probability": math.exp(float(values[top5[0]]) - normalization),
                "hf_top1_top2_gap": float(values[top5[0]] - values[top5[1]]),
                "hf_top5_ids": top5,
            }
        )
    del logits
    totals = summary(rows)
    first_misses = {f"top{k}": [r for r in rows if r["rank"] > k][:16] for k in (1, 5, 100)}
    worst = sorted(rows, key=lambda r: (-r["gap_to_hf_top1"], r["generated_index"]))[:16]
    windows = [
        {
            "generated_indices_inclusive": [start, start + len(rows[start : start + 32]) - 1],
            **summary(rows[start : start + 32]),
        }
        for start in range(0, len(rows), 32)
    ]
    boundary = {
        str(offset): {
            "tokens": sum(r["target_page_offset"] == offset for r in rows),
            "top1_misses": sum(r["target_page_offset"] == offset and r["rank"] > 1 for r in rows),
            "top5_misses": sum(r["target_page_offset"] == offset and r["rank"] > 5 for r in rows),
            "top100_misses": sum(r["target_page_offset"] == offset and r["rank"] > 100 for r in rows),
        }
        for offset in range(32)
    }
    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    result = {
        "metadata": metadata,
        "summary": totals,
        "first_misses": first_misses,
        "largest_logit_gaps": worst,
        "generated_32token_windows": windows,
        "by_target_page_offset": boundary,
        "token_scores": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Compact JSON stores scalar scores only, never full model logits.
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output),
                "forward_seconds": metadata["forward_seconds"],
                "summary": totals,
                "first_top5_misses": first_misses["top5"][:4],
                "first_top100_misses": first_misses["top100"][:4],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
