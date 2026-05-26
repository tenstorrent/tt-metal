# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch / HuggingFace reference inference for Devstral-2-123B (CPU, bfloat16)."""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config
from transformers.integrations.finegrained_fp8 import Fp8Dequantize


_ORIGINAL_DEQUANTIZE_ONE = Fp8Dequantize._dequantize_one


def _dequantize_one_compat(self, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Handle scalar FP8 scales on dense linears (some checkpoints use shape [])."""
    if scales.ndim == 0:
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
            quantized_fp32 = self._unpack_fp4(quantized)
        else:
            quantized_fp32 = quantized.to(torch.float32)
        out_dtype = scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
        scale = scales.to(torch.float32)
        return (quantized_fp32 * scale).to(out_dtype)
    return _ORIGINAL_DEQUANTIZE_ONE(self, quantized, scales)


Fp8Dequantize._dequantize_one = _dequantize_one_compat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single text inference with Devstral-2-123B.")
    parser.add_argument(
        "--prompt",
        default="Write a Python function to reverse a linked list.",
        help="Instruction for the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--offload-folder",
        default="./hf_offload_devstral2_123b",
        help="Folder used by Transformers/Accelerate for disk offloading.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model_name = "mistralai/Devstral-2-123B-Instruct-2512"

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model for {model_name}...")
    offload_folder = Path(args.offload_folder)
    offload_folder.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_quant_cfg = getattr(config, "quantization_config", {}) or {}
    quantization_config = FineGrainedFP8Config(
        activation_scheme=model_quant_cfg.get("activation_scheme", "static"),
        weight_block_size=model_quant_cfg.get("weight_block_size", None),
        dequantize=False,
        modules_to_not_convert=model_quant_cfg.get("modules_to_not_convert", None),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        offload_folder=str(offload_folder),
        offload_state_dict=True,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device

    messages = [{"role": "user", "content": args.prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
    else:
        encoded = tokenizer(args.prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

    input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
        )

    new_tokens = output_ids[:, input_ids.shape[1] :]
    output_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Response ===")
    print(output_text)


if __name__ == "__main__":
    main()
