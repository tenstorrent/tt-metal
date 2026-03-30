#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Reference-style DeepSeek-R1 smoke runner.

Usage examples:

  # Quick smoke test with tiny model
  python models/demos/speculative_deepseek_r1_broad/reference/test_run_model.py \
    --model-id sshleifer/tiny-gpt2 \
    --prompt "Speculative decoding is" \
    --max-new-tokens 16 --device cpu

  # Structure-only (no weight loading, no generation)
  python models/demos/speculative_deepseek_r1_broad/reference/test_run_model.py \
    --model-id deepseek-ai/DeepSeek-R1-0528 \
    --prompt "Hello" \
    --structure-only --trust-remote-code

  # Full R1 on GPU
  python models/demos/speculative_deepseek_r1_broad/reference/test_run_model.py \
    --model-id deepseek-ai/DeepSeek-R1-0528 \
    --prompt "Explain speculative decoding in one sentence." \
    --max-new-tokens 32 --device cuda --dtype bfloat16 --trust-remote-code
"""

from __future__ import annotations

import argparse
import sys

import torch

from models.demos.speculative_deepseek_r1_broad.reference.reference_utils import build_reference_bundle, summarize_model_structure


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Reference-style DeepSeek-R1 smoke runner")
    p.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-R1-0528")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--trust-remote-code", action="store_true", default=False)
    p.add_argument(
        "--structure-only",
        action="store_true",
        default=False,
        help="Print model structure summary and exit (no weight loading, no generation).",
    )
    return p


def main() -> None:
    args = create_parser().parse_args()

    if args.structure_only:
        from models.demos.speculative_deepseek_r1_broad.reference.configuration_deepseek_r1 import (
            DeepseekR1Config,
            load_reference_config,
        )
        from models.demos.speculative_deepseek_r1_broad.reference.modeling_deepseek_r1 import DeepseekR1ForCausalLM

        ref_cfg = load_reference_config(args.model_id, trust_remote_code=args.trust_remote_code)
        print(f"Reference config: {ref_cfg}")
        print()

        try:
            from transformers import AutoConfig
            from transformers.modeling_utils import no_init_weights

            hf_cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
            model_type = getattr(hf_cfg, "model_type", "")
            if model_type == "deepseek_v3":
                r1_config = DeepseekR1Config(**{k: v for k, v in hf_cfg.to_dict().items() if k != "model_type"})
                r1_config.model_type = "deepseek_v3"
                with no_init_weights():
                    model = DeepseekR1ForCausalLM(r1_config)
                print("Model structure (DeepseekR1ForCausalLM):")
                print(model)
            else:
                print(f"Model type '{model_type}' is not DeepSeek-V3 architecture; skipping structure print.")
        except Exception as e:
            print(f"Could not instantiate model structure: {e}")
        return

    bundle = build_reference_bundle(
        args.model_id,
        device=args.device,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Loaded: {summarize_model_structure(bundle)}")

    model_inputs = bundle.tokenizer(args.prompt, return_tensors="pt")
    model_inputs = {k: v.to(torch.device(args.device)) for k, v in model_inputs.items()}
    with torch.no_grad():
        out = bundle.model.generate(**model_inputs, max_new_tokens=args.max_new_tokens)
    print(bundle.tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
