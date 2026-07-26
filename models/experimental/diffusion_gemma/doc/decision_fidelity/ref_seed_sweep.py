#!/usr/bin/env python3
"""Is the reference's step-1 entropy on a block-0 reproducer robust across initial canvases?

The comparison that produced "HF 3.75 vs TT 5.10 on the first forward" used a DIFFERENT random
initial canvas on each side -- HF draws its own `torch.randint` canvas, TT draws one from its seed.
So the gap was never separated from canvas variance. This measures that variance directly: same
prompt, same model, N different canvas seeds, recording step-1 entropy and whether the block
converges.

  reference tight around 3.75 across seeds  -> the TT gap is a real numerics difference
  reference sometimes lands near 5.1 and stalls -> block-0 failure is canvas luck, shared with the
                                                   reference, and belongs to the retry policy

Loads the model once and sweeps seeds, so N runs cost one load.
"""
import argparse
import json
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--checkpoint", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="/tmp/ref_seed_sweep.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as G

    prompt = open(args.prompt_file, encoding="utf-8").read()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, enable_thinking=True
    )
    input_ids = torch.tensor([list(ids["input_ids"])], dtype=torch.long)
    print(f"prompt tokens: {input_ids.shape[1]}", flush=True)

    current = {"trace": []}

    class Recording(G.StableAndConfidentStoppingCriteria):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._prev_argmax = None
            self._step = 0

        def __call__(self, argmax_canvas, logits, **kwargs):
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            mismatch = int((argmax_canvas != self._prev_argmax).sum()) if self._prev_argmax is not None else None
            self._prev_argmax = argmax_canvas.clone()
            self._step += 1
            counts = torch.bincount(argmax_canvas.flatten())
            current["trace"].append(
                {
                    "step": self._step,
                    "mean_entropy": round(float(entropy.mean()), 6),
                    "mismatch": mismatch,
                    "distinct": int((counts > 0).sum()),
                    "top_frac": round(float(counts.max()) / argmax_canvas.numel(), 4),
                }
            )
            return super().__call__(argmax_canvas, logits, **kwargs)

    G.StableAndConfidentStoppingCriteria = Recording

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model = model.to(args.device).eval()
    input_ids = input_ids.to(args.device)

    max_steps = model.generation_config.max_denoising_steps
    results = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        current["trace"] = []
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=256, do_sample=True)
        trace = current["trace"]
        # Only the FIRST block's steps are comparable to TT's block-0 telemetry; the criterion is
        # reset per block, so a step numbered 1 after step>1 marks a new block.
        first_block = []
        for entry in trace:
            if entry["step"] == 1 and first_block:
                break
            first_block.append(entry)
        sequences = out.sequences if hasattr(out, "sequences") else out
        text = tokenizer.decode(sequences[0][input_ids.shape[1] :], skip_special_tokens=False)
        row = {
            "seed": seed,
            "step1_entropy": first_block[0]["mean_entropy"] if first_block else None,
            "final_entropy": first_block[-1]["mean_entropy"] if first_block else None,
            "steps": len(first_block),
            "converged": len(first_block) < max_steps,
            "step1_distinct": first_block[0]["distinct"] if first_block else None,
            "final_distinct": first_block[-1]["distinct"] if first_block else None,
            "final_top_frac": first_block[-1]["top_frac"] if first_block else None,
            "trace": first_block,
            "text_head": text[:120],
        }
        results.append(row)
        print(
            f"seed {seed:>3}: step1 H={row['step1_entropy']}  final H={row['final_entropy']}  "
            f"steps={row['steps']}/{max_steps} converged={row['converged']}  "
            f"distinct {row['step1_distinct']}->{row['final_distinct']}  top_frac={row['final_top_frac']}",
            flush=True,
        )

    entropies = [r["step1_entropy"] for r in results if r["step1_entropy"] is not None]
    print(
        f"\nstep-1 entropy across {len(entropies)} seeds: min={min(entropies):.4f} "
        f"max={max(entropies):.4f} spread={max(entropies)-min(entropies):.4f}"
    )
    print(f"converged: {sum(1 for r in results if r['converged'])}/{len(results)}")
    print("\nTT on this prompt (block 0, five configurations) measured 5.09-5.37 and did NOT converge.")
    json.dump({"prompt_file": args.prompt_file, "results": results}, open(args.out, "w"))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
