# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Reproducibility helper for DeepSeek V3 teacher-forcing reference files.

This script regenerates the checked-in `.refpt` artifact so the reference can
be refreshed or reproduced when needed.
"""

import argparse
import os
import types
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
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

# Remote-code models may expect DynamicCache.get_max_length; add a shim if missing.
if not hasattr(DynamicCache, "get_max_length"):

    def _get_max_length(self):
        return self.get_seq_length()

    DynamicCache.get_max_length = _get_max_length

REFERENCE_FILE = Path(__file__).with_name("deepseek_v3_teacher_forcing.refpt")

TEST_PROMPT = "What is the correct answer to this question:Racemic 3-methylpent-1-ene is treated with Grubbs catalyst. How many possible products are there (excluding ethene)?\nChoices:\n(A) 8\n(B) 2\n(C) 6\n(D) 4\nPlease reason step by step, and your final answer must be only (A,B,C or D) within \\boxed\nAnswer:"


def generate_reference(
    max_new_tokens: int = 128,
    reference_file: Path = REFERENCE_FILE,
    prompt: str = TEST_PROMPT,
    debug_one_layer: bool = False,
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

    if debug_one_layer:
        print("Debug mode: truncating to 1 decoder layer")
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers = model.model.layers[:1]
        if hasattr(model, "config"):
            model.config.num_hidden_layers = 1

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

    def _get_past_length(past_key_values) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        if hasattr(past_key_values, "get_max_length"):
            return past_key_values.get_max_length()
        try:
            return past_key_values[0][0].shape[-2]
        except Exception:
            return 0

    original_prepare = model.prepare_inputs_for_generation.__func__

    def _patched_prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = original_prepare(self, *args, **kwargs)
        attention_mask = model_inputs.get("attention_mask")
        past_key_values = model_inputs.get("past_key_values", kwargs.get("past_key_values"))
        input_ids = model_inputs.get("input_ids")
        if attention_mask is not None and past_key_values is not None and input_ids is not None:
            expected_len = _get_past_length(past_key_values) + input_ids.shape[-1]
            if attention_mask.shape[-1] != expected_len:
                if attention_mask.shape[-1] < expected_len:
                    pad = torch.ones(
                        attention_mask.shape[0],
                        expected_len - attention_mask.shape[-1],
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([attention_mask, pad], dim=-1)
                else:
                    attention_mask = attention_mask[:, :expected_len]
                model_inputs["attention_mask"] = attention_mask
        return model_inputs

    model.prepare_inputs_for_generation = types.MethodType(_patched_prepare_inputs_for_generation, model)

    original_generate = model.generate.__func__

    def _patched_generate(self, *args, **kwargs):
        kwargs.pop("cache_implementation", None)
        return original_generate(self, *args, **kwargs)

    model.generate = types.MethodType(_patched_generate, model)

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

    reference_file.parent.mkdir(parents=True, exist_ok=True)

    # --- Generation (greedy, deterministic) ---
    # NOTE: use_cache=False avoids DynamicCache API mismatch with some transformers versions
    print(f"Generating up to {max_new_tokens} new tokens using model.generate()...")

    def build_payload(reference_tokens_tensor: torch.Tensor, top5_tokens_full: torch.Tensor) -> dict:
        generated_tokens_tensor = reference_tokens_tensor[0, prompt_len:]
        generated_tokens = generated_tokens_tensor.tolist()
        generated_tokens_tensor = generated_tokens_tensor.unsqueeze(0)
        return {
            "reference_tokens": reference_tokens_tensor,  # [1, L]
            "prompt_tokens": torch.tensor(raw_prompt_tokens, dtype=torch.long).unsqueeze(0),  # [1, prompt_len]
            "generated_tokens": generated_tokens_tensor,  # [1, gen_len]
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

    class ReferenceSnapshotter(StoppingCriteria):
        def __init__(
            self,
            prompt_len: int,
            max_new_tokens: int,
            reference_file: Path,
            tokenizer,
            pbar,
        ) -> None:
            self.prompt_len = prompt_len
            self.reference_file = reference_file
            self.max_total_len = prompt_len + max_new_tokens
            self.top5_tokens_full = torch.zeros(self.max_total_len, 5, dtype=torch.long)
            self.last_len = prompt_len
            self.tokenizer = tokenizer
            self.pbar = pbar

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            current_len = input_ids.shape[-1]
            if current_len > self.last_len:
                if scores is not None:
                    # scores is a list/tuple of per-step logits; use the latest step.
                    step_scores = scores[-1] if isinstance(scores, (list, tuple)) else scores
                    if step_scores is not None:
                        if step_scores.dim() == 2:
                            step_scores = step_scores[0]
                        top5 = torch.topk(step_scores, k=5, dim=-1).indices.to(torch.long).cpu()
                        self.top5_tokens_full[current_len - 1] = top5
                self.last_len = current_len
                reference_tokens_tensor = input_ids.detach().cpu()
                generated_tokens = reference_tokens_tensor[0, self.prompt_len :].tolist()
                decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                if self.pbar is not None:
                    self.pbar.write(f"Decoded so far: {decoded!r}")
                else:
                    print(f"Decoded so far: {decoded!r}")
                payload = build_payload(
                    reference_tokens_tensor,
                    self.top5_tokens_full[:current_len].clone().cpu(),
                )
                torch.save(payload, self.reference_file)
            return False

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
    snapshotter = None
    if tqdm is not None and max_new_tokens > 0:
        pbar = tqdm(total=max_new_tokens, desc="Generating tokens", unit="tok", mininterval=1)
        snapshotter = ReferenceSnapshotter(prompt_len, max_new_tokens, reference_file, tokenizer, pbar)
        stopping_criteria = StoppingCriteriaList([TokenProgress(prompt_len, pbar), snapshotter])
    elif tqdm is None and max_new_tokens > 0:
        print("tqdm not available; generation progress bar disabled.")
        snapshotter = ReferenceSnapshotter(prompt_len, max_new_tokens, reference_file, tokenizer, None)
        stopping_criteria = StoppingCriteriaList([snapshotter])

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=prompt_tokens_tensor,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,  # reuse logits from the generation pass
                output_scores=True,  # needed for incremental top5 in stopping criteria
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

    total_length = int(full_sequence_tensor.numel())
    top5_tokens_full = snapshotter.top5_tokens_full[:total_length].clone().cpu()

    # --- Save payload (explicit prompt/generated + full sequence) ---
    payload = build_payload(full_sequence_tensor.unsqueeze(0).cpu(), top5_tokens_full)
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
    parser.add_argument(
        "--debug-one-layer",
        action="store_true",
        help="Use only the first decoder layer (debugging; not for real references).",
    )
    args = parser.parse_args()

    path = generate_reference(
        max_new_tokens=args.max_new_tokens,
        reference_file=args.output,
        prompt=args.prompt,
        debug_one_layer=args.debug_one_layer,
    )
    print(f"\nDone. Reference saved to: {path}")
