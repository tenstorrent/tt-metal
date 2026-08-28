# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 2 demo -- text -> prompt embedding, the model's role inside Flux2KleinPipeline.

`model_index.json` of `black-forest-labs/FLUX.2-klein-9B` declares
`text_encoder: Qwen3ForCausalLM`: the diffusion transformer consumes this model's
encoded hidden states, not its logits. This demo produces exactly that tensor
through the chained TTNN pipeline.

    python -m models.demos.flux_2_klein_9b.text_encoder.demo.demo_prompt_encoding \
        --prompt "A photograph of a rusted lighthouse at dawn"

The wiring lives in `tt/pipeline.py` and is imported, not copied.
"""
from __future__ import annotations

import argparse

from models.demos.flux_2_klein_9b.text_encoder.selftest import own_a_mesh
from models.demos.flux_2_klein_9b.text_encoder.tt import model_ref
from models.demos.flux_2_klein_9b.text_encoder.tt.pipeline import build_pipeline


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default=model_ref.DEFAULT_PROMPT, help="the text prompt to encode")
    ap.add_argument("--tp", type=int, default=8, help="tensor-parallel width (the mesh is 1 x tp)")
    ap.add_argument("--layers", type=int, default=None, help="cap the decoder depth (default: all 36)")
    ap.add_argument("--compare-hf", action="store_true", help="also run the HF reference and print the PCC")
    ap.add_argument("--save", default=None, help="optional .pt path to write the prompt embedding to")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    with own_a_mesh(args.tp) as device:
        pipeline = build_pipeline(device, layers=args.layers)
        input_ids = model_ref.encode_prompt(args.prompt)

        print(f"prompt: {args.prompt}")
        print(f"prompt tokens: {int(input_ids.shape[-1])}")

        prompt_embeds = pipeline.run_prompt_encoding(input_ids=input_ids)

        print(f"prompt_embeds: shape={tuple(prompt_embeds.shape)} dtype={prompt_embeds.dtype}")
        print(f"prompt_embeds: norm={prompt_embeds.norm().item():.4f} mean={prompt_embeds.mean().item():.6f}")
        print("this is the tensor Flux2Transformer2DModel consumes as the text conditioning")

        if args.compare_hf:
            golden = model_ref.hf_reference_prompt_encoding(pipeline.hf_model, input_ids)
            achieved_pcc = model_ref.pcc(golden, prompt_embeds)
            print(f"e2e PCC={achieved_pcc}")

        if args.save:
            import torch

            torch.save(prompt_embeds, args.save)
            print(f"saved to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
