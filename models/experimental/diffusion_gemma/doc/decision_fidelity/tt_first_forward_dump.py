#!/usr/bin/env python3
"""Dump TT's FIRST denoise forward layer by layer, on the SAME inputs as the reference dump.

Pairs with `ref_first_forward_dump.py` (A100). The two runs share both inputs that could otherwise
differ, which is what makes a layer-by-layer comparison meaningful:

  prompt  -- tokenized with enable_thinking=True, matching the reference's 270 tokens on q106.
             `generate_text` does not forward that flag, so this drives the adapter directly.
  canvas  -- `torch.randint(0, 262144, (1,256))` from a `Generator().manual_seed(0)`: the identical
             construction `make_seeded_host_canvas_init_fn` uses for block 0 and the reference dump
             used, verified byte-identical (first 8 ids
             [199340, 43567, 173685, 117952, 176963, 152315, 95939, 97639]).

Driving `adapter(canvas, step=0)` is the same single-step entry `doc/optimize_perf/prof_denoise_step.py`
uses, so this is one forward with no denoise loop, no halt logic and no commit.

The processed logits are formed here as softcap(z)/0.8 -- an explicit t_max rather than the halt
telemetry or whatever a 1-step config resolves the schedule to -- so they match the reference tap
exactly. Everything derived from them belongs to `first_forward_stats.py`, which owns one
implementation for both sides; this script only prints the accept count and the reference targets so
a run can be read at a glance.
"""
import argparse
import json
import os
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--checkpoint", default="/home/zni/dg_models/diffusiongemma-26B-A4B-it")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--canvas-seed", type=int, default=0)
    parser.add_argument("--canvas-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8, help="t_max, the first-step value")
    parser.add_argument("--out", default="/tmp/tt_first_forward.pt")
    parser.add_argument("--stats-json", default="/tmp/tt_first_forward_stats.json")
    args = parser.parse_args()

    import ttnn
    from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
    from models.experimental.diffusion_gemma.tt import denoise_forward as DF
    from models.experimental.diffusion_gemma.tt.generate import (
        host_canvas_to_device,
        make_generation_logits_fn_builder_from_checkpoint_state,
        prefill_prompt_tokens,
        tokenize_prompt,
    )

    prompt = open(args.prompt_file, encoding="utf-8").read()

    captured = {"layers": {}}
    original_layer_forward = DF._denoise_layer_forward

    def capturing_layer_forward(tt_model, layer_idx, hidden_states, *rest, **kwargs):
        out = original_layer_forward(tt_model, layer_idx, hidden_states, *rest, **kwargs)
        if layer_idx not in captured["layers"]:
            captured["layers"][layer_idx] = ttnn.to_torch(out).to(torch.bfloat16).cpu().clone()
        return out

    DF._denoise_layer_forward = capturing_layer_forward

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=int(os.getenv("DG_TRACE_REGION_SIZE", 0)))
    try:
        # Only pass num_layers when it is set: build_tt_model_from_checkpoint_inputs branches on
        # `"num_layers" in model_kwargs`, so passing None would still switch it to the reduced-layer
        # builder rather than the default full-model path.
        model_kwargs = {"num_layers": args.num_layers} if args.num_layers is not None else {}
        model_inputs = build_tt_model_from_checkpoint_dir(
            mesh,
            args.checkpoint,
            max_batch_size=1,
            max_seq_len=args.max_seq_len,
            create_kv_cache=True,
            tokenizer_kwargs={"local_files_only": True, "trust_remote_code": True},
            **model_kwargs,
        )
        tt_model = model_inputs.tt_model

        prompt_tokens = tokenize_prompt(model_inputs.tokenizer, prompt, enable_thinking=True)
        print(f"prompt tokens: {tuple(prompt_tokens.shape)}", flush=True)
        prefill = prefill_prompt_tokens(tt_model, prompt_tokens)
        print(f"prefill: prompt_len={prefill.prompt_len} cache_len={prefill.cache_len}", flush=True)

        adapter_kwargs = {}
        hf_config = getattr(tt_model, "hf_config", None)
        if hf_config is not None:
            adapter_kwargs["config"] = hf_config
        logits_builder = make_generation_logits_fn_builder_from_checkpoint_state(
            model_inputs.state_dict, **adapter_kwargs
        )
        adapter = logits_builder(tt_model, prompt_tokens=prompt_tokens, prompt_len=prefill.cache_len)

        generator = torch.Generator()
        generator.manual_seed(args.canvas_seed)
        host_canvas = torch.randint(0, 262144, (1, args.canvas_length), dtype=torch.long, generator=generator)
        print(f"canvas: seed={args.canvas_seed} first 8 ids {host_canvas[0, :8].tolist()}", flush=True)
        canvas = host_canvas_to_device(mesh, host_canvas)

        logits_tt = adapter(canvas, step=0)
        logits = ttnn.to_torch(logits_tt).float().cpu().clone()
    finally:
        DF._denoise_layer_forward = original_layer_forward
        ttnn.close_mesh_device(mesh)

    # Statistics live in first_forward_stats.py so both sides share one implementation -- and so the
    # decision-relevant number (the accept count, i.e. how many positions fall under the 0.1-nat
    # bound) is not re-derived here next to the mean entropy, which is the statistic the sampler does
    # NOT read.
    processed = logits.reshape(1, -1, logits.shape[-1]) / args.temperature
    probs = torch.softmax(processed, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    argmax = processed.argmax(dim=-1)
    top2 = processed.topk(2, dim=-1)
    stats = {
        "logit_std": float(processed.std()),
        "logit_max_mean": float(processed.max(dim=-1).values.mean()),
        "logit_min_mean": float(processed.min(dim=-1).values.mean()),
        "top1_minus_top2_mean": float((top2.values[..., 0] - top2.values[..., 1]).mean()),
        "distinct_argmax": int(torch.unique(argmax).numel()),
    }
    ordered = entropy.flatten().sort().values
    exclusive_prefix = torch.cat([torch.zeros(1), ordered.cumsum(0)[:-1]])
    accept = int((exclusive_prefix <= 0.1).sum())
    print(
        f"\nfirst-forward summary (TT): accept_count_step1={accept}  "
        f"positions_below_0.1={int((entropy < 0.1).sum())}  "
        f"entropy median={float(ordered[len(ordered) // 2]):.4f} mean={float(entropy.mean()):.4f}  "
        f"logit_std={stats['logit_std']:.4f}"
    )
    print("reference targets on q106: accept 6, below_0.1 5, median 4.2624, logit_std 7.4553")

    # The reference's confident positions are not scattered: on q106/q096/q095 they are positions 0-4
    # and they carry the SAME tokens every time -- <|channel>(100), thought(45518), \n(107), *(236829),
    # spaces(139) -- the thinking-template prefix that structurally must follow the generation prompt.
    # That handful of near-zero-entropy positions is the entire accept budget the block bootstraps
    # from, so whether TT is confident THERE is the question, not what its average entropy is.
    print("\npositions 0-7 (reference on q106: ids 100/45518/107/236829/139 at H 4e-4/2e-5/0.011/0.009/0.004):")
    for pos in range(8):
        top = processed[0, pos].topk(2)
        print(
            f"  pos {pos:>2}  H={float(entropy[0, pos]):.6f}  argmax={int(argmax[0, pos]):>7}  "
            f"top1-top2={float(top.values[0] - top.values[1]):.4f}"
        )
    print("full comparison: first_forward_stats.py ref_ff_qNNN.pt tt_first_forward.pt")

    torch.save(
        {
            "prompt_file": args.prompt_file,
            "prompt_tokens": prompt_tokens,
            "canvas_ids": host_canvas,
            "canvas_seed": args.canvas_seed,
            "layer_hidden": captured["layers"],
            "logits": logits,
            "entropy": entropy,
            "argmax": argmax,
            "logits_stats": stats,
        },
        args.out,
    )
    json.dump(
        {
            "logits_stats": stats,
            "canvas_seed": args.canvas_seed,
            "canvas_head": host_canvas[0, :16].tolist(),
            "prompt_tokens": int(prompt_tokens.shape[-1]),
        },
        open(args.stats_json, "w"),
        indent=2,
    )
    print(f"\n-> {args.out}\n-> {args.stats_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
