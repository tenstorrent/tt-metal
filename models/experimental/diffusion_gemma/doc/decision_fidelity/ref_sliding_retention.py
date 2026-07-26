#!/usr/bin/env python3
"""Ground-truth HF sliding-layer key retention, against a prompt longer than the window.

TT's retention mask (denoise_forward._sliding_layer_needs_denoise_mask) asserts that HF keeps the
last `sliding_window - 1` = 1023 committed tokens on sliding layers. That number is load-bearing:
it is applied on 25 of DiffusionGemma's 30 layers, so an off-by-one there is an off-by-one 25 times
over. This reads the actual cache HF builds instead of trusting the docstring.

Prints, per layer type, the key length the decoder can attend to after prefilling a prompt well
past the window, so the comparison is unambiguous.
"""
import argparse
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/ttuser/zni/dg_models/diffusiongemma-26B-A4B-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-tokens", type=int, default=1500, help="prompt length, must exceed the window")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion

    tok = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True
    )
    model = model.to(args.device).eval()

    text_config = model.config.get_text_config()
    window = text_config.sliding_window
    layer_types = text_config.layer_types
    print(
        f"config: sliding_window={window}  layers={len(layer_types)}  "
        f"sliding={layer_types.count('sliding_attention')}  full={layer_types.count('full_attention')}"
    )

    # A prompt comfortably past the window, so retention actually binds.
    ids = torch.arange(2000, 2000 + args.prompt_tokens, dtype=torch.long).unsqueeze(0).to(args.device)
    print(f"prompt: {ids.shape[1]} tokens")

    with torch.no_grad():
        enc = model.model.encoder(input_ids=ids, use_cache=True)
    cache = enc.past_key_values
    print(f"cache class: {type(cache).__name__}  is_compileable={getattr(cache, 'is_compileable', None)}")

    canvas_length = model.config.canvas_length
    print(f"\n{'layer':>6} {'type':>18} {'cached keys':>12} {'get_mask_sizes(kv_len, kv_off)':>32}")
    seen = set()
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type in seen and layer_idx > 6:
            continue
        seen.add(layer_type)
        try:
            keys = cache.layers[layer_idx].keys
            cached = None if keys is None else keys.shape[-2]
        except (AttributeError, IndexError, TypeError):
            cached = None
        try:
            sizes = cache.get_mask_sizes(canvas_length, layer_idx)
        except Exception as exc:  # noqa: BLE001 - want the reason printed, not raised
            sizes = f"error: {exc}"
        print(f"{layer_idx:>6} {layer_type:>18} {str(cached):>12} {str(sizes):>32}")

    sliding_idx = layer_types.index("sliding_attention")
    full_idx = layer_types.index("full_attention")
    s_keys = cache.layers[sliding_idx].keys.shape[-2]
    f_keys = cache.layers[full_idx].keys.shape[-2]
    print(f"\nsliding layer {sliding_idx}: {s_keys} cached keys")
    print(f"   full layer {full_idx}: {f_keys} cached keys")
    print(f"\nprompt {ids.shape[1]} tokens, window {window}:")
    print(f"  retained on sliding layers = {s_keys}")
    for label, value in (("sliding_window", window), ("sliding_window - 1", window - 1), ("full prompt", ids.shape[1])):
        print(f"    {'MATCHES' if s_keys == value else '      no'}  {label} = {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
