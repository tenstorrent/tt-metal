# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Save a real-checkpoint HF CPU reference without importing/opening TTNN.

This is reference preparation, not TT device validation or a performance
baseline. The HF BF16 weights/cache differ from the selected TT BFP4/BFP8
precision. Keep the matched traced TT baseline as the fusion comparison.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.utils.logging import disable_progress_bar

from models.demos.llama31_8b_qb2.tests.benchmark_megakernel import make_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("LLAMA_MODEL_PATH"))
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-source", type=Path, help="TT evidence.pt whose teacher token stream to follow")
    args = parser.parse_args()
    if args.checkpoint is None or args.context < 1 or args.tokens < 3:
        parser.error("Local checkpoint, context >= 1, tokens >= 3 are required")
    if args.context + args.tokens - 1 > 131072:
        parser.error("Request exceeds Llama 3.1 context")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    disable_progress_bar()
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    prompt = make_prompt(tokenizer, args.context)
    teacher = None
    if args.teacher_source:
        source = torch.load(args.teacher_source, weights_only=True, map_location="cpu")
        assert source["prompt"] == prompt and source["tokens"] == args.tokens
        teacher = source["teacher_tokens"]
    print("Loading real HF BF16 checkpoint on CPU", flush=True)
    model = LlamaForCausalLM.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    assert model.config.num_hidden_layers == 32 and model.config.hidden_size == 4096
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    print(f"Loaded 32 layers in {time.monotonic() - started:.2f}s", flush=True)
    current = torch.tensor([prompt], dtype=torch.long)
    cache = None
    rows, predictions, inputs, lengths = [], [], [], []
    with torch.inference_mode():
        for step in range(args.tokens):
            output = model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            row = output.logits[0, -1].clone()
            assert torch.isfinite(row).all()
            prediction = int(row.argmax())
            token = teacher[step] if teacher is not None else prediction
            rows.append(row)
            predictions.append(prediction)
            inputs.append(token)
            lengths.append(cache.get_seq_length())
            assert lengths[-1] == args.context + step
            current = torch.tensor([[token]], dtype=torch.long)
            print(f"HF reference step {step + 1}/{args.tokens}, cache={lengths[-1]}, token={prediction}", flush=True)
    # Store only outputs and request KV, never checkpoint weights.
    kv = [[layer.keys.detach().clone(), layer.values.detach().clone()] for layer in cache.layers]
    assert len(kv) == 32
    result = {
        "kind": "HF BF16 CPU reference; not TT numerical validation or device latency",
        "checkpoint": str(args.checkpoint),
        "checkpoint_revision": args.checkpoint.name,
        "context": args.context,
        "tokens": args.tokens,
        "batch": 1,
        "prompt": prompt,
        "generated_tokens": predictions,
        "teacher_tokens": inputs,
        "teacher_logits": torch.stack(rows),
        "teacher_cache": kv,
        "cache_lengths": lengths,
    }
    torch.save(result, args.output / "reference.pt")
    summary = {k: v for k, v in result.items() if k not in ("teacher_logits", "teacher_cache")}
    summary.update(
        {
            "reference_preparation_s": time.monotonic() - started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "generated_text": tokenizer.decode(predictions),
            "logits_shape": list(result["teacher_logits"].shape),
            "cache_shape_per_layer": list(kv[0][0].shape),
            "all_outputs_finite": all(torch.isfinite(t).all().item() for pair in kv for t in pair),
        }
    )
    assert summary["all_outputs_finite"]
    (args.output / "reference.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
