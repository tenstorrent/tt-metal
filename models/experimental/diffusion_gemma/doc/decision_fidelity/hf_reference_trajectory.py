#!/usr/bin/env python3
"""Record the HF reference's per-step denoise trajectory, in TT's telemetry units.

The question this answers: on a prompt where TT's loop stalls (entropy 5.10 -> ~3.0 with ~100 of 256
positions still flipping at the 48-step cap), does the REFERENCE converge, or does it also stall and
simply commit the unsettled positions?

Those two worlds need different fixes. If the reference converges, TT has a numerics gap to close.
If it stalls too, the difference is only in WHAT the unsettled positions become -- TT collapses them
onto unigram filler ('-', '\\n', '1', ' the'), and the fix belongs at the commit/sampling end.

Instrumentation point: StableAndConfidentStoppingCriteria.__call__ receives the processed logits and
the argmax canvas every step, which is exactly what TT's halt telemetry reads. Wrapping it gives the
same two scalars per step -- mean per-position entropy and the argmax mismatch against the previous
step -- with no changes to the model.
"""
import argparse
import json
import os
import sys
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--checkpoint", default=os.environ.get("DG_CKPT", "google/diffusiongemma-26B-A4B-it"))
    parser.add_argument("--out", default="/tmp/hf_traj.json")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="one canvas = one block")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the initial canvas. REQUIRED for comparing step COUNTS across arms: the canvas is "
        "otherwise redrawn each run, and ref_seed_sweep.py measured 9-15 steps on q106 across 8 "
        "canvases, so an unseeded step count carries more variance than most effects being tested. "
        "Step-1 entropy is far more stable (0.35 nats of spread) but seed it anyway.",
    )
    parser.add_argument(
        "--pad-prompt-to",
        type=int,
        default=0,
        help="Append pad-id-0 tokens until the prompt is a multiple of N, reproducing TT's prefill "
        "(_pad_prompt_tokens_for_prefill appends zeros to a tile multiple, those pad K/V are written "
        "to the cache, and the reveal mask reveals [0:padded_len], so TT's canvas ATTENDS them). This "
        "is the fuller TT geometry: the position gap AND the attended pad keys, which are the canvas's "
        "nearest neighbours in RoPE terms. --position-shift reproduces only the gap.",
    )
    parser.add_argument(
        "--pad-side",
        choices=("right", "left"),
        default="right",
        help="Which side --pad-prompt-to appends on. 'right' is what TT does today and detaches the "
        "canvas from the prompt by the pad count. 'left' keeps the prompt ENDING adjacent to the "
        "canvas, so the geometry is correct with the padding still present -- the candidate fix, "
        "since RoPE is relative and a uniform shift of all positions is harmless.",
    )
    parser.add_argument(
        "--position-shift",
        type=int,
        default=0,
        help="Add N to the canvas position ids only, reproducing TT's padded q_rope_offset. At step 1 "
        "this alone drops the reference's accept count from 5-7 to 1 and wipes the template-prefix "
        "anchor; this flag exists to ask the follow-up question, whether the block still CONVERGES.",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as G

    prompt = open(args.prompt_file, encoding="utf-8").read()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, enable_thinking=True
    )
    input_ids = torch.tensor([list(ids["input_ids"])], dtype=torch.long)
    print(f"prompt tokens: {input_ids.shape[1]}", flush=True)

    if args.pad_prompt_to:
        pad = (-input_ids.shape[1]) % args.pad_prompt_to
        if pad:
            padding = torch.zeros((1, pad), dtype=torch.long)
            parts = [input_ids, padding] if args.pad_side == "right" else [padding, input_ids]
            input_ids = torch.cat(parts, dim=1)
            adjacency = "detached by" if args.pad_side == "right" else "adjacent, pads moved before the prompt;"
            print(
                f"INJECTED prefill padding on the {args.pad_side}: +{pad} pad-id-0 tokens -> "
                f"{input_ids.shape[1]}; canvas {adjacency} {pad if args.pad_side == 'right' else 0} "
                f"positions from the prompt end",
                flush=True,
            )

    trace = []

    class Recording(G.StableAndConfidentStoppingCriteria):
        """Same criterion, plus the two scalars TT's halt telemetry reports."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._prev_argmax = None
            self._step = 0

        def __call__(self, argmax_canvas, logits, **kwargs):
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            mean_entropy = float(entropy.mean())
            mismatch = int((argmax_canvas != self._prev_argmax).sum()) if self._prev_argmax is not None else None
            self._prev_argmax = argmax_canvas.clone()
            self._step += 1
            counts = torch.bincount(argmax_canvas.flatten())
            trace.append(
                {
                    "step": self._step,
                    "mean_entropy": round(mean_entropy, 6),
                    "mismatch": mismatch,
                    "distinct_argmax": int((counts > 0).sum()),
                    "top_frac": round(float(counts.max()) / argmax_canvas.numel(), 4),
                    "top_id": int(counts.argmax()),
                }
            )
            print(
                f"  step {self._step:>3} H={mean_entropy:.5f} mismatch={mismatch} "
                f"distinct={trace[-1]['distinct_argmax']}/256 top_frac={trace[-1]['top_frac']:.3f}",
                flush=True,
            )
            return super().__call__(argmax_canvas, logits, **kwargs)

    G.StableAndConfidentStoppingCriteria = Recording

    print(f"loading {args.checkpoint} as {args.dtype} on {args.device} ...", flush=True)
    t0 = time.time()
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.checkpoint, dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True, local_files_only=True
    )
    model = model.to(args.device).eval()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    if args.position_shift:
        decoder = model.model.decoder
        original_decoder_forward = decoder.forward

        def shifted_decoder_forward(*call_args, **call_kwargs):
            position_ids = call_kwargs.get("decoder_position_ids")
            if position_ids is None:
                raise RuntimeError("decoder_position_ids was None; the shift would silently do nothing")
            call_kwargs["decoder_position_ids"] = position_ids + args.position_shift
            return original_decoder_forward(*call_args, **call_kwargs)

        decoder.forward = shifted_decoder_forward
        print(f"INJECTED TT geometry: canvas position ids shifted by +{args.position_shift}", flush=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)
        print(f"canvas seeded: {args.seed}", flush=True)

    t0 = time.time()
    input_ids = input_ids.to(args.device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=True)
    elapsed = time.time() - t0
    sequences = out.sequences if hasattr(out, "sequences") else out
    text = tokenizer.decode(sequences[0][input_ids.shape[1] :], skip_special_tokens=False)
    print(f"\ngenerated in {elapsed:.0f}s")
    print(f"steps recorded: {len(trace)}")
    print(f"text tail: {text[-200:]!r}")
    json.dump(
        {"prompt_file": args.prompt_file, "dtype": args.dtype, "steps": trace, "elapsed_s": elapsed, "text": text},
        open(args.out, "w"),
    )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
