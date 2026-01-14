# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Reproducibility helper for DeepSeek V3 teacher-forcing reference files.

This script regenerates the checked-in `.refpt` artifact so the reference can
be refreshed or reproduced when needed.
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from models.demos.deepseek_v3.utils.config_helpers import dequantize
from models.demos.deepseek_v3.utils.hf_model_utils import (
    apply_with_names,
    load_model_uninitialized,
    load_model_weights,
    unload_weight_from_weights_dict,
)

MODEL_PATH = Path(
    os.getenv(
        "DEEPSEEK_V3_HF_MODEL",
        "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528",
    )
)

REFERENCE_FILE = Path(__file__).with_name("deepseek_v3_teacher_forcing.refpt")

TEST_PROMPT = "What is the correct answer to this question:Racemic 3-methylpent-1-ene is treated with Grubbs catalyst. How many possible products are there (excluding ethene)?\nChoices:\n(A) 8\n(B) 2\n(C) 6\n(D) 4\nPlease reason step by step, and your final answer must be only (A,B,C or D) within \\boxed\nAnswer:"


def generate_reference(
    max_new_tokens: int = 128,
    reference_file: Path = REFERENCE_FILE,
    prompt: str = TEST_PROMPT,
) -> Path:
    """
    Generate reference tokens using HuggingFace model and save to refpt file.

    Key fixes vs the previous version:
      1) Pass an explicit attention_mask (prevents pad/bos collisions breaking generate()).
      2) Use a "safe" pad_token_id (typically eos_token_id, not 0).
      3) Save prompt_tokens and generated_tokens explicitly (downstream teacher forcing is simpler).
      4) Compute top-5 directly from generation logits (single pass).

    top5_tokens alignment convention:
      - reference_tokens[0, i] is the *actual* token at position i
      - top5_tokens[i] is the model's top-5 prediction for token at position i,
        given context tokens [0..i-1]
      - top5_tokens[0] is zeros (no prediction for the first token)
      - For single-pass generation, we populate top5_tokens only for generated
        positions (>= prompt_len) and leave earlier rows as zeros.
    """

    print("\n=== Phase 1: Generate reference tokens with HuggingFace model ===")
    print(f"Using model_path={MODEL_PATH}")
    print(f"Prompt: {prompt!r}")

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

        # DeepSeek checkpoints may store float8 + scale_inv.
        if loaded_weight.dtype == torch.float8_e4m3fn:
            scale_inv = weights_dict.get(f"{name}_scale_inv", None)
            if scale_inv is None:
                raise KeyError(f"Missing scale_inv tensor for float8 weight: {name}_scale_inv")
            loaded_weight = dequantize(loaded_weight, scale_inv, (128, 128))
            del scale_inv

        target_dtype = torch.bfloat16
        target_device = tensor.device

        loaded_weight = loaded_weight.to(dtype=target_dtype, device=target_device)
        if tensor.dtype != target_dtype:
            tensor.data = tensor.data.to(target_dtype)

        tensor.data = loaded_weight
        del loaded_weight
        return tensor

    def add_dynamic_weight_loading_hooks_with_dtype(
        module,
        weights_dict_local,
        lazy_modules=("DeepseekV3Attention", "DeepseekV3MLP"),
        model_name="",
    ):
        is_lazy = any(module.__class__.__name__ == lazy_module for lazy_module in lazy_modules)

        if not is_lazy and next(module.children(), None) is not None:
            for child_name, child in module.named_children():
                add_dynamic_weight_loading_hooks_with_dtype(
                    child,
                    weights_dict_local,
                    lazy_modules=lazy_modules,
                    model_name=f"{model_name}{child_name}.",
                )
            return

        if not is_lazy:
            apply_with_names(model_name, module, load_weight_with_dtype)
            return

        # Lazy modules: load right before forward, unload right after forward.
        module.register_forward_pre_hook(
            lambda m, args, kwargs: apply_with_names(model_name, m, load_weight_with_dtype),
            with_kwargs=True,
        )
        module.register_forward_hook(
            lambda m, args, kwargs, output: apply_with_names(
                model_name, m, unload_weight_from_weights_dict(weights_dict_local)
            ),
            with_kwargs=True,
        )

    print("Setting up lazy weight loading (weights loaded on-demand during inference)...")
    add_dynamic_weight_loading_hooks_with_dtype(model, weights_dict)
    print("Lazy weight loading configured - weights will be loaded as needed during generation")

    model = model.to(device)
    print(f"Model moved to {device}")

    # --- Build prompt tokens (must match TT generator) ---
    raw_prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_len = len(raw_prompt_tokens)
    prompt_tokens_tensor = torch.tensor([raw_prompt_tokens], device=device, dtype=torch.long)

    # IMPORTANT: explicit attention mask prevents generate() from inferring pad positions incorrectly.
    attention_mask = torch.ones_like(prompt_tokens_tensor, dtype=torch.long, device=device)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # Fall back to config if eos is not set (some remote-code tokenizers do this).
    if eos_id is None:
        eos_id = getattr(config, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0]
    if eos_id is None:
        raise RuntimeError("Could not determine eos_token_id (tokenizer.eos_token_id/config.eos_token_id is None).")

    # Use a safe pad id. If pad==bos (often 0), it can break inferred attention masks.
    safe_pad_id = eos_id
    if pad_id is not None and (bos_id is None or pad_id != bos_id):
        safe_pad_id = pad_id

    print(f"Prompt tokens: {prompt_len}")
    print(f"bos_id={bos_id} eos_id={eos_id} pad_id={pad_id} safe_pad_id={safe_pad_id}")

    # --- Generation (greedy, deterministic) ---
    # NOTE: use_cache=False avoids DynamicCache API mismatch with some transformers versions
    print(f"Generating up to {max_new_tokens} new tokens using model.generate()...")

    class TokenProgress(StoppingCriteria):
        def __init__(self, prompt_len: int, pbar) -> None:
            self.prompt_len = prompt_len
            self.pbar = pbar

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            generated = input_ids.shape[-1] - self.prompt_len
            if generated > self.pbar.n:
                self.pbar.update(generated - self.pbar.n)
            return False

    pbar = None
    stopping_criteria = None
    if tqdm is not None and max_new_tokens > 0:
        pbar = tqdm(total=max_new_tokens, desc="Generating tokens", unit="tok", mininterval=1)
        stopping_criteria = StoppingCriteriaList([TokenProgress(prompt_len, pbar)])
    elif tqdm is None and max_new_tokens > 0:
        print("tqdm not available; generation progress bar disabled.")

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=prompt_tokens_tensor,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,  # reuse logits from the generation pass
                pad_token_id=safe_pad_id,
                eos_token_id=eos_id,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )
    finally:
        if pbar is not None:
            pbar.close()

    full_sequence_tensor = outputs.sequences[0]  # [prompt_len + gen_len]
    generated_tokens_tensor = full_sequence_tensor[prompt_len:]
    generated_tokens = generated_tokens_tensor.tolist()

    print(f"Phase 1: Generated {len(generated_tokens)} tokens")
    print("Decoded generation (raw):", repr(tokenizer.decode(generated_tokens, skip_special_tokens=False)))

    # --- Top-5 aligned to generated positions (from generate() logits) ---
    logits_steps = outputs.logits if getattr(outputs, "logits", None) is not None else outputs.scores
    if logits_steps is None:
        raise RuntimeError("Generation did not return logits; expected output_logits=True.")

    gen_len = int(generated_tokens_tensor.numel())
    if gen_len != len(logits_steps):
        raise RuntimeError(
            f"Mismatch between generated tokens ({gen_len}) and logit steps ({len(logits_steps)})."
        )

    top5_generated = torch.empty((gen_len, 5), dtype=torch.long)
    for i, step_logits in enumerate(logits_steps):
        top5_generated[i] = torch.topk(step_logits[0], k=5, dim=-1).indices.to(torch.long).cpu()

    total_length = int(full_sequence_tensor.numel())
    top5_tokens_full = torch.zeros(total_length, 5, dtype=torch.long)  # [L, 5]
    if gen_len:
        start = prompt_len
        end = start + gen_len
        top5_tokens_full[start:end] = top5_generated

    # --- Save payload (explicit prompt/generated + full sequence) ---
    reference_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "reference_tokens": full_sequence_tensor.unsqueeze(0).cpu(),  # [1, L]
        "prompt_tokens": torch.tensor(raw_prompt_tokens, dtype=torch.long).unsqueeze(0),  # [1, prompt_len]
        "generated_tokens": generated_tokens_tensor.unsqueeze(0).cpu(),  # [1, gen_len]
        "top5_tokens": top5_tokens_full,  # [L, 5]
        "tf_prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "prompt": prompt,
        "decoded_generated_text": tokenizer.decode(generated_tokens, skip_special_tokens=False),
        "token_ids_meta": {
            "bos_id": bos_id,
            "eos_id": eos_id,
            "pad_id": pad_id,
            "safe_pad_id_used_for_generate": safe_pad_id,
        },
    }

    torch.save(payload, reference_file)

    print(
        f"Saved reference file with {total_length} tokens " f"(prompt={prompt_len}, generated={len(generated_tokens)})"
    )
    print(f"Top5 tokens shape: {tuple(top5_tokens_full.shape)}")
    print(f"Reference file saved to: {reference_file}")

    return reference_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DeepSeek V3 teacher-forcing reference file.")
    parser.add_argument("--prompt", default=TEST_PROMPT, help="Prompt text to generate from.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REFERENCE_FILE,
        help="Path to output .refpt file.",
    )
    args = parser.parse_args()

    path = generate_reference(
        max_new_tokens=args.max_new_tokens,
        reference_file=args.output,
        prompt=args.prompt,
    )
    print(f"\nDone. Reference saved to: {path}")
