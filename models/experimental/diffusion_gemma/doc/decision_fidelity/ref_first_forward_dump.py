#!/usr/bin/env python3
"""Dump the reference's FIRST denoise forward, layer by layer, for a shared-canvas TT comparison.

Section 9's gap (TT 5.10 vs reference 3.70 mean per-position entropy on q106 block 0) lives inside a
single forward pass: it is there at step 1, before self-conditioning or any canvas feedback exists.
Localising it needs the reference's intermediate tensors on the SAME inputs TT will see, which means
pinning the one input that is otherwise random -- the initial canvas.

`generate(decoder_input_ids=...)` is the supported way in: _prepare_denoiser_inputs pops
`decoder_input_ids` and uses it as the initial canvas instead of `sampler.initialize_canvas()`. So a
canvas drawn here with an explicit generator is both what the reference denoises and what TT can be
handed via `generate.make_host_canvas_init_fn`.

Saves the prompt ids, the canvas ids, every decoder layer's output hidden state, the final normed
hidden, and the processed logits' decision statistics. Layer hiddens are [1, 256, 2816] -- 1.4 MB
each in bf16, 43 MB for all 30 -- so the whole dump is small enough to move between boxes.
"""
import argparse
import json
import sys

import torch


class _StopAfterFirstForward(Exception):
    """Raised once the first denoise forward has been captured, to end generate() early."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--checkpoint", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canvas-seed", type=int, default=0)
    parser.add_argument("--out", default="/tmp/ref_first_forward.pt")
    parser.add_argument("--stats-json", default="/tmp/ref_first_forward_stats.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion

    prompt = open(args.prompt_file, encoding="utf-8").read()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, enable_thinking=True
    )
    input_ids = torch.tensor([list(ids["input_ids"])], dtype=torch.long)
    print(f"prompt tokens: {input_ids.shape[1]}", flush=True)

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model = model.to(args.device).eval()

    text_config = model.config.get_text_config()
    canvas_length = model.config.canvas_length
    vocab_size = text_config.vocab_size

    # The one otherwise-random input, pinned so TT can replay it verbatim.
    generator = torch.Generator().manual_seed(args.canvas_seed)
    canvas = torch.randint(0, vocab_size, (1, canvas_length), dtype=torch.long, generator=generator)
    print(f"canvas: seed={args.canvas_seed} first 8 ids {canvas[0, :8].tolist()}", flush=True)

    captured = {"layers": {}}
    decoder = model.model.decoder

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            if layer_idx in captured["layers"]:
                return  # only the first forward
            hidden = output[0] if isinstance(output, tuple) else output
            captured["layers"][layer_idx] = hidden.detach().to("cpu", torch.bfloat16).clone()

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(decoder.layers)]

    def norm_hook(_module, _inputs, output):
        if "final_hidden" not in captured:
            captured["final_hidden"] = output.detach().to("cpu", torch.bfloat16).clone()

    handles.append(decoder.norm.register_forward_hook(norm_hook))

    # The logits the sampler actually decides on: softcapped by the model, then temperature-scaled by
    # the logits processor. Capturing them at the stopping criterion is the same tap section 9 used,
    # so the entropy printed here is directly comparable to TT's halt telemetry.
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as G

    class CaptureAndStop(G.StableAndConfidentStoppingCriteria):
        def __call__(self, argmax_canvas, logits, **kwargs):
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            captured["entropy"] = entropy.detach().to("cpu").clone()
            captured["argmax"] = argmax_canvas.detach().to("cpu").clone()
            captured["logits_stats"] = {
                "mean_entropy": float(entropy.mean()),
                "logit_std": float(logits.float().std()),
                "logit_max_mean": float(logits.float().max(dim=-1).values.mean()),
                "logit_min_mean": float(logits.float().min(dim=-1).values.mean()),
                "top1_minus_top2_mean": float(
                    (lambda t: (t.values[..., 0] - t.values[..., 1]).mean())(logits.float().topk(2, dim=-1))
                ),
                "distinct_argmax": int(torch.unique(argmax_canvas).numel()),
            }
            raise _StopAfterFirstForward

    G.StableAndConfidentStoppingCriteria = CaptureAndStop

    input_ids = input_ids.to(args.device)
    try:
        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=canvas_length,
                do_sample=True,
                decoder_input_ids=canvas.to(args.device),
            )
    except _StopAfterFirstForward:
        print("captured the first denoise forward", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    missing = [i for i in range(len(decoder.layers)) if i not in captured["layers"]]
    if missing:
        print(f"WARNING: no hidden captured for layers {missing}")

    stats = captured.get("logits_stats", {})
    print("\nfirst-forward decision statistics (reference, bf16):")
    for key, value in stats.items():
        print(f"  {key:>22} = {value}")

    per_layer = {}
    for layer_idx, hidden in sorted(captured["layers"].items()):
        h = hidden.float()
        per_layer[layer_idx] = {
            "rms": float(h.pow(2).mean().sqrt()),
            "mean_abs": float(h.abs().mean()),
            "max_abs": float(h.abs().max()),
        }
    print("\nper-layer hidden RMS (the growth curve TT has to match):")
    for layer_idx, row in per_layer.items():
        print(f"  layer {layer_idx:>2}: rms={row['rms']:.4f} mean|h|={row['mean_abs']:.4f} max|h|={row['max_abs']:.2f}")

    torch.save(
        {
            "prompt_file": args.prompt_file,
            "prompt_ids": input_ids.to("cpu"),
            "canvas_ids": canvas,
            "canvas_seed": args.canvas_seed,
            "layer_hidden": captured["layers"],
            "final_hidden": captured.get("final_hidden"),
            "entropy": captured.get("entropy"),
            "argmax": captured.get("argmax"),
            "logits_stats": stats,
        },
        args.out,
    )
    json.dump(
        {
            "logits_stats": stats,
            "per_layer": per_layer,
            "canvas_seed": args.canvas_seed,
            "canvas_head": canvas[0, :16].tolist(),
            "prompt_tokens": int(input_ids.shape[1]),
        },
        open(args.stats_json, "w"),
        indent=2,
    )
    print(f"\n-> {args.out}\n-> {args.stats_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
