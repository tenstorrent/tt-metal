# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 1 demo -- text -> text on FLUX.2-klein-9B's text encoder (Qwen3ForCausalLM).

Real input (the model's own Qwen2TokenizerFast), the real chained TTNN forward
pass over the graduated stubs, real output (decoded text).

    python -m models.demos.flux_2_klein_9b.text_encoder.demo.demo_text_generation \
        --prompt "A photograph of a rusted lighthouse at dawn" --max-new-tokens 40

The wiring itself lives in `tt/pipeline.py` and is imported, not copied -- this
demo and `tests/e2e/test_e2e_pipeline.py` run identical code.
"""
from __future__ import annotations

import argparse

from models.demos.flux_2_klein_9b.text_encoder.selftest import own_a_mesh
from models.demos.flux_2_klein_9b.text_encoder.tt import model_ref
from models.demos.flux_2_klein_9b.text_encoder.tt.pipeline import build_pipeline


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default=model_ref.DEFAULT_PROMPT, help="the text prompt to continue")
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the model's chat template")
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="safety bound on the decode horizon; decoding stops on the model's own eos either way",
    )
    ap.add_argument("--tp", type=int, default=8, help="tensor-parallel width (the mesh is 1 x tp)")
    ap.add_argument("--layers", type=int, default=None, help="cap the decoder depth (default: all 36)")
    ap.add_argument("--compare-hf", action="store_true", help="also run the HF reference and print the PCC")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    with own_a_mesh(args.tp) as device:
        pipeline = build_pipeline(device, layers=args.layers)
        input_ids = model_ref.encode_prompt(args.prompt, chat=args.chat)

        print(f"prompt: {args.prompt}")
        print(f"prompt tokens: {int(input_ids.shape[-1])}")

        result = pipeline.run_text_generation(input_ids=input_ids, max_new_tokens=args.max_new_tokens)

        print(f"generated {len(result['token_ids'])} tokens (horizon {result['horizon']})")
        print(f"token ids: {result['token_ids']}")
        print("--- TT output ---")
        print(result["text"])

        if args.compare_hf:
            golden = model_ref.hf_reference_text_generation(
                pipeline.hf_model, input_ids, max_new_tokens=len(result["token_ids"])
            )
            print("--- HF reference ---")
            print(golden["text"])
            # Scored the way tests/e2e does: the reference on the SAME contexts this
            # run decoded, so the number is this port's arithmetic rather than the
            # greedy tie-break two free-running sequences inherit once they part.
            golden_logits = model_ref.hf_reference_step_logits(pipeline.hf_model, input_ids, result["token_ids"])
            print(f"e2e PCC={model_ref.pcc(golden_logits, result['step_logits'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
