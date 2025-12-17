# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from models.demos.deepseek_v3.utils.config_helpers import dequantize
from models.demos.deepseek_v3.utils.hf_model_utils import apply_with_names, load_model_uninitialized, load_model_weights

MODEL_PATH = Path(
    os.getenv(
        "DEEPSEEK_V3_HF_MODEL",
        "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528",
    )
)

REFERENCE_FILE = Path(__file__).with_name("deepseek_v3_teacher_forcing.refpt")

TEST_PROMPT = "What is the capital of France? Please provide a brief answer."


def generate_reference(
    max_new_tokens: int = 32,
    reference_file: Path = REFERENCE_FILE,
) -> Path:
    """Generate reference tokens using HuggingFace model and save to refpt file."""

    print("\n=== Phase 1: Generate reference tokens with HuggingFace model ===")
    print(f"Using model_path={MODEL_PATH}")
    print(f"Prompt: {TEST_PROMPT!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading model with uninitialized weights (memory efficient)...")
    model = load_model_uninitialized(str(MODEL_PATH))
    model.eval()
    print("Model structure created successfully")

    print("Loading weights dictionary from disk...")
    weights_dict = load_model_weights(str(MODEL_PATH))
    print(f"Loaded {len(weights_dict)} weight tensors from disk")

    @torch.no_grad()
    def load_weight_with_dtype(name: str, tensor: torch.Tensor) -> torch.Tensor:
        if name not in weights_dict:
            return tensor
        loaded_weight = weights_dict[name]
        if loaded_weight.dtype == torch.float8_e4m3fn:
            loaded_weight_scale = weights_dict[f"{name}_scale_inv"]
            loaded_weight = dequantize(loaded_weight, loaded_weight_scale, (128, 128))
            del loaded_weight_scale
        target_dtype = torch.bfloat16
        target_device = tensor.device
        loaded_weight = loaded_weight.to(dtype=target_dtype, device=target_device)
        if tensor.dtype != target_dtype:
            tensor.data = tensor.data.to(target_dtype)
        tensor.data = loaded_weight
        del loaded_weight
        return tensor

    def add_dynamic_weight_loading_hooks_with_dtype(
        module, weights_dict, lazy_modules=["DeepseekV3Attention", "DeepseekV3MLP"], model_name=""
    ):
        is_lazy = any(module.__class__.__name__ == lazy_module for lazy_module in lazy_modules)
        if not is_lazy and next(module.children(), None) is not None:
            for child_name, child in module.named_children():
                add_dynamic_weight_loading_hooks_with_dtype(
                    child, weights_dict, model_name=f"{model_name}{child_name}."
                )
            return
        elif not is_lazy:
            apply_with_names(model_name, module, load_weight_with_dtype)
            return
        module.register_forward_pre_hook(
            lambda module, args, kwargs: apply_with_names(model_name, module, load_weight_with_dtype),
            with_kwargs=True,
        )
        from models.demos.deepseek_v3.utils.hf_model_utils import unload_weight_from_weights_dict

        module.register_forward_hook(
            lambda module, args, kwargs, output: apply_with_names(
                model_name, module, unload_weight_from_weights_dict(weights_dict)
            ),
            with_kwargs=True,
        )

    print("Setting up lazy weight loading (weights loaded on-demand during inference)...")
    add_dynamic_weight_loading_hooks_with_dtype(model, weights_dict)
    print("Lazy weight loading configured - weights will be loaded as needed during generation")

    model = model.to(device)
    print(f"Model moved to {device}")

    raw_prompt_tokens = tokenizer.encode(TEST_PROMPT, add_special_tokens=False)
    prompt_tokens_tensor = torch.tensor([raw_prompt_tokens], device=device, dtype=torch.long)

    print(f"Prompt tokens: {len(raw_prompt_tokens)}")

    generated_tokens = []
    all_top5_tokens = []

    with torch.no_grad():
        prompt_len = prompt_tokens_tensor.shape[1]
        prefill_attention_mask = torch.full(
            (1, 1, prompt_len, prompt_len), float("-inf"), device=device, dtype=torch.float32
        )
        mask_cond = torch.arange(prompt_len, device=device)
        prefill_attention_mask.masked_fill_(mask_cond[:, None] >= mask_cond[None, :], 0.0)

        outputs = model(prompt_tokens_tensor, attention_mask=prefill_attention_mask, use_cache=True)
        logits = outputs.logits

        next_token_logits = logits[0, -1, :]

        probs = torch.softmax(next_token_logits, dim=-1)
        _, top5 = torch.topk(probs, k=5, dim=-1)
        all_top5_tokens.append(top5.cpu())

        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        generated_tokens.append(next_token_id)

        past_key_values = outputs.past_key_values
        current_seq_len = prompt_len

        for step in range(max_new_tokens - 1):
            next_token_tensor = torch.tensor([[next_token_id]], device=device, dtype=torch.long)

            current_seq_len += 1
            decode_attention_mask = torch.zeros(1, 1, 1, current_seq_len, device=device, dtype=torch.float32)

            outputs = model(
                next_token_tensor, attention_mask=decode_attention_mask, past_key_values=past_key_values, use_cache=True
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_logits = logits[0, -1, :]

            probs = torch.softmax(next_token_logits, dim=-1)
            _, top5 = torch.topk(probs, k=5, dim=-1)
            all_top5_tokens.append(top5.cpu())

            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            generated_tokens.append(next_token_id)

            if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
                break

    print(f"Phase 1: Generated {len(generated_tokens)} tokens")

    full_sequence = raw_prompt_tokens + generated_tokens
    reference_tokens_tensor = torch.tensor(full_sequence, dtype=torch.long).unsqueeze(0)

    total_length = len(full_sequence)
    top5_tokens_full = torch.zeros(total_length, 5, dtype=torch.long)

    for i, top5 in enumerate(all_top5_tokens):
        pos = len(raw_prompt_tokens) - 1 + i
        if pos < total_length:
            top5_tokens_full[pos] = top5

    reference_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "reference_tokens": reference_tokens_tensor.cpu(),
        "top5_tokens": top5_tokens_full.cpu(),
        "tf_prompt_len": len(raw_prompt_tokens),
        "max_new_tokens": max_new_tokens,
        "prompt": TEST_PROMPT,
    }

    torch.save(payload, reference_file)

    print(
        f"Phase 1: Created reference file with {len(full_sequence)} tokens "
        f"(prompt={len(raw_prompt_tokens)}, generated={len(generated_tokens)})"
    )
    print(f"Top5 tokens shape: {top5_tokens_full.shape}")
    print(f"Reference file saved to: {reference_file}")

    return reference_file


if __name__ == "__main__":
    path = generate_reference()
    print(f"\nDone. Reference saved to: {path}")
