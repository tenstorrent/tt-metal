"""CPU HF continuations conditioned on TT prefixes; not independent HF controls."""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# This diagnostic must use the already-cached pinned checkpoint, with no accelerator.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

MODEL = "IFM/K2-Horizon-7B"
REVISION = "036114ce8d46c32b24c15423211069abb9c5d25e"
DOC = Path("models/demos/k2_horizon_7b_qb2/doc/full_model")


def read_source(path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


class Progress(StoppingCriteria):
    def __init__(self, width):
        self.width = width
        self.start = time.perf_counter()

    def __call__(self, input_ids, scores, **kwargs):
        steps = input_ids.shape[1] - self.width
        if steps % 32 == 0:
            print(
                json.dumps({"event": "progress", "new_tokens": steps, "seconds": time.perf_counter() - self.start}),
                flush=True,
            )
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tt-source", type=Path, default=DOC / "qualitative_extended.json")
    parser.add_argument("--hf-source", type=Path, default=DOC / "hf_qualitative_extended.json")
    parser.add_argument("--output", type=Path, default=DOC / "hf_conditional_quality.json")
    parser.add_argument("--ids", nargs="+", default=["shared_1", "shared_3", "shared_4"])
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 4 <= args.threads <= 8 or min(args.prefix_tokens, args.steps) < 1:
        parser.error("use 4..8 threads and positive prefix/continuation token counts")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tt_records, tt_source = read_source(args.tt_source)
    hf_records, hf_source = read_source(args.hf_source)
    tt_by_id = {row["id"]: row for row in tt_records}
    hf_by_id = {row["id"]: row for row in hf_records}
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
    if not tok.chat_template:
        raise ValueError("Expected the pinned K2 chat template")
    prepared = []
    for prompt_id in args.ids:
        tt, hf = tt_by_id[prompt_id], hf_by_id[prompt_id]
        rendered = tok.apply_chat_template(hf["messages"], tokenize=False, add_generation_prompt=True)
        prompt_ids = tok.apply_chat_template(
            hf["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if rendered != hf["rendered_prompt"] or prompt_ids != hf["prompt_token_ids"]:
            raise ValueError(f"HF prompt-format mismatch for {prompt_id}")
        if prompt_ids != tt["prompt_token_ids"]:
            raise ValueError(f"TT/HF prompt-token mismatch for {prompt_id}")
        prefix = tt["tt_token_ids"][: args.prefix_tokens]
        if len(prefix) != args.prefix_tokens or tok.eos_token_id in prefix:
            raise ValueError(f"Need {args.prefix_tokens} TT tokens preceding EOS for {prompt_id}")
        prepared.append(
            {
                "id": prompt_id,
                "messages": hf["messages"],
                "rendered_prompt": rendered,
                "prompt_token_ids": prompt_ids,
                "tt_prefix_token_ids": prefix,
                "tt_prefix_text": tok.decode(prefix, skip_special_tokens=False),
                "tt_original_text": tt["tt_text_through_eos"],
                "independent_hf_text": hf["hf_text"],
            }
        )
    metadata = {
        "diagnostic": "HF continuation conditioned on TT-generated prefixes",
        "interpretation_limit": "Not an independent HF baseline or a stage-pass waiver; the TT prefix fixes an existing trajectory.",
        "hf_model": MODEL,
        "revision": REVISION,
        "tokenizer_class": type(tok).__name__,
        "chat_template_present": bool(tok.chat_template),
        "prompt_mode": "chat",
        "rendering": "apply_chat_template(add_generation_prompt=True, return_dict=False)",
        "sources": {"tt": tt_source, "independent_hf": hf_source},
        "generation": {
            "prefix_tokens": args.prefix_tokens,
            "max_new_tokens": args.steps,
            "do_sample": False,
            "batch_size": len(prepared),
            "padding": "left, masked",
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.eos_token_id,
        },
        "device": "cpu",
        "threads": args.threads,
        "interop_threads": 1,
        "offline": True,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python -m models.demos.k2_horizon_7b_qb2.tests.hf_conditional_quality " + " ".join(sys.argv[1:]),
    }
    print(json.dumps({"event": "load", "metadata": metadata}), flush=True)
    started = time.perf_counter()
    # Match the independent baseline's loading defaults, and explicitly retain CPU placement.
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
        .eval()
        .to("cpu")
    )
    metadata["model_dtype"] = str(next(model.parameters()).dtype)
    metadata["load_seconds"] = time.perf_counter() - started
    width = max(len(row["prompt_token_ids"]) + len(row["tt_prefix_token_ids"]) for row in prepared)
    batch_ids = torch.full((len(prepared), width), tok.eos_token_id, dtype=torch.int64)
    mask = torch.zeros_like(batch_ids)
    for row, record in enumerate(prepared):
        ids = record["prompt_token_ids"] + record["tt_prefix_token_ids"]
        batch_ids[row, -len(ids) :] = torch.tensor(ids)
        mask[row, -len(ids) :] = 1
    print(
        json.dumps(
            {
                "event": "generate",
                "load_seconds": metadata["load_seconds"],
                "dtype": metadata["model_dtype"],
                "batch_shape": list(batch_ids.shape),
            }
        ),
        flush=True,
    )
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            batch_ids,
            attention_mask=mask,
            max_new_tokens=args.steps,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([Progress(width)]),
        )[:, width:]
    metadata["generation_seconds"] = time.perf_counter() - started
    for record, output in zip(prepared, outputs.tolist()):
        eos = output.index(tok.eos_token_id) if tok.eos_token_id in output else None
        if eos is not None:
            output = output[: eos + 1]
        full = record["tt_prefix_token_ids"] + output
        continuation = tok.decode(output, skip_special_tokens=False)
        text = tok.decode(full, skip_special_tokens=False)
        record.update(
            hf_continuation_token_ids=output,
            hf_continuation_text=continuation,
            conditional_full_text=text,
            continuation_tokens=len(output),
            first_eos_index=eos,
            reached_token_limit=eos is None and len(output) == args.steps,
            observations={
                "think_close_present": "</ifm|think>" in text,
                "we_can_also_mention_in_continuation": continuation.lower().count("we can also mention"),
            },
        )
        print(
            json.dumps(
                {"event": "result", "id": record["id"], "continuation_tokens": len(output), "text": continuation}
            ),
            flush=True,
        )
    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata": metadata, "records": prepared}, indent=2) + "\n")
    print(
        json.dumps(
            {"event": "complete", "output": str(args.output), "generation_seconds": metadata["generation_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
