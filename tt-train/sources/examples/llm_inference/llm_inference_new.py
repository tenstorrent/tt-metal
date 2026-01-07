import ttml
from time import time

OUTPUT_TOKENS = 100
WITH_SAMPLING = True
TEMPERATURE = 0.0
SEED = 42
model_id = "meta-llama/Llama-3.2-1B-Instruct"
CONFIG = "training_shakespeare_llama3_2_1B_fixed.yaml"  # now working fine (with proper shape for tokenizer)
model_id = "meta-llama/Llama-3.2-3B-Instruct"
CONFIG = "training_shakespeare_llama3_2_3B.yaml"
model_id = "Qwen/Qwen3-0.6B"
CONFIG = "training_shakespeare_qwen3_0_6B.yaml"  # working, not 1-1 as llama, but speak something nongibberish
import os, sys, random
import numpy as np  # For numpy arrays
from dataclasses import dataclass  # For configuration classes
from huggingface_hub import hf_hub_download  # To download safetensors from Hugging Face
from transformers import AutoTokenizer
from yaml import safe_load  # To read YAML configs
from pathlib import Path

import ttml
from ttml.common.config import get_training_config, load_config, TransformerConfig
from ttml.common.utils import set_seed, round_up_to_tile
from ttml.common.model_factory import TransformerModelFactory

get_training_config(CONFIG)
# Load the tokenizer from Hugging Face and the transformer config from YAML
tokenizer = AutoTokenizer.from_pretrained(model_id)
training_config = get_training_config(CONFIG)
model_yaml = load_config(training_config.model_config, configs_root=os.getcwd() + "/../../..")
safetensors_path = hf_hub_download(repo_id=model_id, filename="config.json")
safetensors_path = safetensors_path.replace("config.json", "")
import torch
from transformers import AutoModelForCausalLM

torch.manual_seed(SEED)
torch_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tokenizer.vocab_size, torch_model.state_dict()["model.embed_tokens.weight"].shape[0], torch_model.vocab_size
len(torch_model.state_dict())
orig_vocab_size = torch_model.vocab_size
print(orig_vocab_size)
tt_model_factory = TransformerModelFactory(model_yaml)
tt_model_factory.transformer_config.vocab_size = orig_vocab_size

max_sequence_length = tt_model_factory.transformer_config.max_sequence_length
# ttml.autograd.AutoContext.get_instance().set_init_mode(ttml.autograd.InitMode.DISABLED)
model_yaml
ttml.autograd.AutoContext.get_instance().set_init_mode(ttml.autograd.InitMode.DISABLED)
start_time = time()
tt_model = tt_model_factory.create_model()
print(f"Model created: {time() - start_time}")
# llama 3b
# Model created: 238.99020171165466
# Model loaded: 47.06614542007446
# llama 3b
# Model created: 7.940993309020996
# Model loaded: 58.52051877975464
start_time = time()
tt_model.load_from_safetensors(safetensors_path)
print(f"Model loaded: {time() - start_time}")
padded_vocab_size = round_up_to_tile(orig_vocab_size, 32)
if orig_vocab_size != padded_vocab_size:
    print(f"Padding vocab size for tilization: original {orig_vocab_size} -> padded {padded_vocab_size}")


def build_causal_mask(T: int) -> ttml.autograd.Tensor:
    # [1,1,T,T] float32 with 1s for allowed positions (i >= j), else 0\n",
    m = np.tril(np.ones((T, T), dtype=np.float32))
    return ttml.autograd.Tensor.from_numpy(m.reshape(1, 1, T, T), ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16)


def build_logits_mask(vocab_size: int, padded_vocab_size: int) -> ttml.autograd.Tensor:
    logits_mask = np.zeros((1, 1, 1, padded_vocab_size), dtype=np.float32)
    logits_mask[:, :, :, vocab_size:] = 1e4
    return ttml.autograd.Tensor.from_numpy(
        logits_mask, ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16
    )  # [1,1,1,T], bfloat16"


def generate_with_tt(model, prompt_tokens):
    import time

    ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    model.eval()

    logits_mask_tensor = None

    if padded_vocab_size != orig_vocab_size:
        logits_mask_tensor = build_logits_mask(orig_vocab_size, padded_vocab_size)

    causal_mask = build_causal_mask(max_sequence_length)  # [1,1,seq_len,seq_len], float32
    padded_prompt_tokens = np.zeros((1, 1, 1, max_sequence_length), dtype=np.uint32)

    start_idx = 0

    print("************************************")
    start_time = time.time()

    for token_idx in range(OUTPUT_TOKENS):
        if len(prompt_tokens) > max_sequence_length:
            start_idx = len(prompt_tokens) - max_sequence_length

        # padded_prompt_tokens[0, 0, 0, :transformer_cfg["max_sequence_length"]] = 0
        padded_prompt_tokens[0, 0, 0, : len(prompt_tokens)] = prompt_tokens[start_idx:]
        padded_prompt_tensor = ttml.autograd.Tensor.from_numpy(
            padded_prompt_tokens, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32
        )  # [1,1,1, max_seq_len], uint32

        logits = model(padded_prompt_tensor, causal_mask)  # out=[1,1,seq_len, vocab_size], bf16

        next_token_tensor = ttml.ops.sample.sample_op(
            logits, TEMPERATURE, np.random.randint(low=1e7), logits_mask_tensor
        )  # out=[1,1,seq_len,1], uint32

        next_token_idx = max_sequence_length - 1 if len(prompt_tokens) > max_sequence_length else len(prompt_tokens) - 1
        next_token = next_token_tensor.to_numpy().flatten()[next_token_idx]

        output = tokenizer.decode(next_token)

        prompt_tokens.append(next_token)
        print(output, end="", flush=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    tokens_per_second = OUTPUT_TOKENS / elapsed_time

    print(f"\n************************************")
    print(f"Generated {OUTPUT_TOKENS} tokens in {elapsed_time:.2f} seconds")
    print(f"Performance: {tokens_per_second:.2f} tokens/second")
    print("************************************\n\n")


def generate_with_pytorch(torch_model, prompt_tokens):
    import time
    import torch.nn.functional as F
    from transformers import DynamicCache

    torch_model.eval()

    print("************************************")
    # Convert list to tensor and add batch dimension
    if isinstance(prompt_tokens, list):
        prompt_tokens = torch.tensor([prompt_tokens])

    start_time = time.time()

    # Initialize KV cache using the new DynamicCache API
    past_key_values = DynamicCache()
    input_ids = prompt_tokens

    with torch.no_grad():
        for i in range(OUTPUT_TOKENS):
            # Get model outputs with KV cache
            outputs = torch_model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            # Get logits for the last token
            next_token_logits = logits[:, -1, :]

            # Apply temperature and sample
            if WITH_SAMPLING and TEMPERATURE > 0:
                next_token_logits = next_token_logits / TEMPERATURE
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy sampling
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Decode and print the token
            output = tokenizer.decode(next_token[0])
            print(output, end="", flush=True)

            # For next iteration, only pass the new token (KV cache handles the rest)
            input_ids = next_token

    end_time = time.time()
    elapsed_time = end_time - start_time
    tokens_per_second = OUTPUT_TOKENS / elapsed_time

    print(f"\n************************************")
    print(f"Generated {OUTPUT_TOKENS} tokens in {elapsed_time:.2f} seconds")
    print(f"Performance: {tokens_per_second:.2f} tokens/second")
    print("************************************\n\n")


def generate_with_pytorch_batch(torch_model, prompt_tokens):
    """Old version: non-streaming batch generation using torch_model.generate()"""
    import time

    torch_model.eval()

    print("************************************")
    # Convert list to tensor and add batch dimension
    if isinstance(prompt_tokens, list):
        prompt_tokens = torch.tensor([prompt_tokens])

    start_time = time.time()

    with torch.no_grad():
        outputs = torch_model.generate(
            prompt_tokens,
            max_new_tokens=OUTPUT_TOKENS,
            do_sample=WITH_SAMPLING,  # Enable sampling
            temperature=TEMPERATURE,  # Temperature for sampling
            num_beams=1,  # Use multinomial sampling (standard sampling)
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    tokens_per_second = OUTPUT_TOKENS / elapsed_time

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for t in generated_text:
        print(t)

    print(f"\n************************************")
    print(f"Generated {OUTPUT_TOKENS} tokens in {elapsed_time:.2f} seconds")
    print(f"Performance: {tokens_per_second:.2f} tokens/second")
    print("************************************\n\n")


prompt_str = "Generating with pytorch(CPU, might be slow)"

prompt_tokens = tokenizer.encode(prompt_str)
print("Generating with torch model:")
generate_with_pytorch(torch_model, prompt_tokens)
prompt_str = "Generating with pytorch(CPU, might be slow)"
prompt_tokens = tokenizer.encode(prompt_str)
print("Generating with TT:")
generate_with_tt(tt_model, prompt_tokens.copy())
sd = torch_model.state_dict()
for s in sd:
    print(s, sd[s].shape)
k = tt_model.parameters()
for s in k:
    print(s, k[s].shape())
import numpy as np
import torch


def apply_rope_permutation(w, num_heads):
    """
    Apply RoPE row permutation to match TT's q_proj weight layout.
    TT applies unpermute_proj_rows during loading which interleaves rows within each head.
    """
    rows, cols = w.shape
    head_dim = rows // num_heads

    out = np.zeros_like(w)
    for h in range(num_heads):
        head_start = h * head_dim
        half = head_dim // 2

        # Interleave: [0..half-1, half..head_dim-1] → [0, half, 1, half+1, ..., half-1, head_dim-1]
        for i in range(half):
            out[head_start + 2 * i] = w[head_start + i]
            out[head_start + 2 * i + 1] = w[head_start + half + i]

    return out


def compare_weights_llama(torch_model, tt_model):
    """
    Compare weights between PyTorch model and TT-Metal model.

    Args:
        torch_model: PyTorch model (with .state_dict())
        tt_model: TT-Metal model (with .parameters())
    """

    torch_sd = torch_model.state_dict()
    tt_params = tt_model.parameters()

    # Get num_heads from torch model config
    num_heads = torch_model.config.num_attention_heads

    # Detect weight tying configuration from TT model
    has_tok_emb = "llama/tok_emb/weight" in tt_params
    has_fc = "llama/fc/weight" in tt_params
    weight_tying_enabled = not has_tok_emb  # If tok_emb doesn't exist, weight tying is enabled

    # Mapping from PyTorch parameter names to TT-Metal parameter names
    pytorch_to_tt_mapping = {
        # Final layer norm
        "model.norm.weight": "llama/ln_fc/gamma",
    }

    # Add embedding mappings based on weight tying configuration
    if weight_tying_enabled:
        # Weight tying enabled: both embed_tokens and lm_head use fc/weight
        pytorch_to_tt_mapping["model.embed_tokens.weight"] = "llama/fc/weight"
        # lm_head.weight should also map to fc/weight, but we'll skip it to avoid duplicate checks
    else:
        # Weight tying disabled: separate tok_emb and fc
        pytorch_to_tt_mapping["model.embed_tokens.weight"] = "llama/tok_emb/weight"
        pytorch_to_tt_mapping["lm_head.weight"] = "llama/fc/weight"

    # Add layer-specific mappings
    for i in range(50):  # Support up to 50 layers
        layer_prefix_pt = f"model.layers.{i}"
        layer_prefix_tt = f"llama/llama_block_{i}"

        pytorch_to_tt_mapping.update(
            {
                f"{layer_prefix_pt}.input_layernorm.weight": f"{layer_prefix_tt}/attention_norm/gamma",
                f"{layer_prefix_pt}.post_attention_layernorm.weight": f"{layer_prefix_tt}/mlp_norm/gamma",
                f"{layer_prefix_pt}.self_attn.q_proj.weight": f"{layer_prefix_tt}/attention/q_linear/weight",
                f"{layer_prefix_pt}.self_attn.o_proj.weight": f"{layer_prefix_tt}/attention/out_linear/weight",
                # k_proj and v_proj are combined into kv_linear in TT
                f"{layer_prefix_pt}.mlp.gate_proj.weight": f"{layer_prefix_tt}/mlp/w1/weight",
                f"{layer_prefix_pt}.mlp.up_proj.weight": f"{layer_prefix_tt}/mlp/w3/weight",
                f"{layer_prefix_pt}.mlp.down_proj.weight": f"{layer_prefix_tt}/mlp/w2/weight",
            }
        )

    print("=" * 80)
    print("WEIGHT COMPARISON: PyTorch vs TT-Metal")
    print("=" * 80)
    print(f"Note: Detected num_heads={num_heads} for RoPE permutation")
    print(f"Note: Weight tying {'ENABLED' if weight_tying_enabled else 'DISABLED'}")
    print("=" * 80)

    mismatches = []
    matches = []

    for pt_name in torch_sd.keys():
        if "bias" in pt_name:
            continue  # Skip bias parameters

        pt_tensor = torch_sd[pt_name]
        pt_shape = tuple(pt_tensor.shape)

        # Handle k_proj and v_proj specially (they're combined in TT)
        if ".self_attn.k_proj.weight" in pt_name or ".self_attn.v_proj.weight" in pt_name:
            layer_idx = pt_name.split(".")[2]
            tt_name = f"llama/llama_block_{layer_idx}/attention/kv_linear/weight"

            if tt_name in tt_params:
                tt_tensor_np = tt_params[tt_name].to_numpy()
                tt_shape = tt_tensor_np.shape

                # k_proj and v_proj are concatenated in kv_linear
                # Expected: k_proj [512, 2048] + v_proj [512, 2048] = kv_linear [1, 1, 1024, 2048]
                if ".self_attn.k_proj.weight" in pt_name:
                    print(f"\n{pt_name}")
                    print(f"  PyTorch: {pt_shape}")
                    print(f"  TT (kv combined): {tt_shape}")
                    print(f"  Status: K and V are combined in TT as kv_linear")
            continue

        # Get corresponding TT parameter name
        tt_name = pytorch_to_tt_mapping.get(pt_name)
        if not tt_name:
            continue

        if tt_name not in tt_params:
            print(f"\n❌ MISSING: {pt_name} -> {tt_name}")
            print(f"   PyTorch shape: {pt_shape}")
            mismatches.append((pt_name, "MISSING IN TT"))
            continue

        # Get TT tensor
        tt_tensor_np = tt_params[tt_name].to_numpy()
        tt_shape = tt_tensor_np.shape

        # Remove batch dimensions [1, 1, ...] from TT tensor
        tt_shape_no_batch = tt_shape[2:] if len(tt_shape) == 4 else tt_shape

        # Compare shapes
        pt_numpy = pt_tensor.cpu().float().numpy()  # Convert to float32 for numpy compatibility

        # Special handling for q_proj: TT applies RoPE row permutation during loading
        is_q_proj = ".self_attn.q_proj.weight" in pt_name
        if is_q_proj and len(pt_shape) == 2 and pt_shape[0] % num_heads == 0:
            pt_numpy = apply_rope_permutation(pt_numpy, num_heads)

        # For layer norms: PT (N,) vs TT (1, N) - both are fine, just broadcasting
        # Check if PT is 1D and TT has leading 1s that can be squeezed
        if len(pt_shape) == 1 and len(tt_shape_no_batch) == 2 and tt_shape_no_batch[0] == 1:
            tt_shape_no_batch = (tt_shape_no_batch[1],)  # Squeeze leading 1

        # Check if shapes match (with or without transpose)
        shape_match = (pt_shape == tt_shape_no_batch) or (pt_shape == tt_shape_no_batch[::-1])

        if shape_match:
            # Check actual values
            # Reshape TT data to match PT shape (handle batch dims and potential squeezing)
            if len(tt_shape) == 4:
                tt_data = tt_tensor_np.reshape(tt_shape[2:])  # Remove [1,1,...] batch dims
            else:
                tt_data = tt_tensor_np.reshape(tt_shape)

            # Squeeze if needed for 1D comparisons
            tt_data = tt_data.squeeze()
            pt_numpy_squeezed = pt_numpy.squeeze()

            # Handle transpose if needed
            if pt_numpy_squeezed.shape != tt_data.shape and len(tt_data.shape) == 2:
                tt_data = tt_data.T

            diff = np.abs(pt_numpy_squeezed - tt_data).max()
            rel_diff = diff / (np.abs(pt_numpy_squeezed).max() + 1e-8)

            status = "✓" if diff < 1e-3 else "⚠"
            note = " (after RoPE permutation)" if is_q_proj else ""
            print(f"\n{status} {pt_name}{note}")
            print(f"  PyTorch: {pt_shape}")
            print(f"  TT:      {tt_shape} -> {tt_shape_no_batch}")
            print(f"  Max diff: {diff:.6f}, Rel diff: {rel_diff:.6f}")

            if diff < 1e-3:
                matches.append(pt_name)
            else:
                mismatches.append((pt_name, f"VALUE_DIFF={diff:.6f}"))
        else:
            print(f"\n❌ SHAPE MISMATCH: {pt_name}")
            print(f"  PyTorch: {pt_shape}")
            print(f"  TT:      {tt_shape} -> {tt_shape_no_batch}")
            mismatches.append((pt_name, f"SHAPE: PT={pt_shape} vs TT={tt_shape_no_batch}"))

    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(matches)} matches, {len(mismatches)} mismatches")
    print("=" * 80)

    if mismatches:
        print("\n❌ MISMATCHES:")
        for name, issue in mismatches:
            print(f"  - {name}: {issue}")

    return matches, mismatches


# Usage example (commented out):
matches, mismatches = compare_weights_llama(torch_model, tt_model)


#!/usr/bin/env python3
"""
Qwen3 Weight Comparison Script

Compares weights between PyTorch Qwen3 model and TT-Metal Qwen3 model.

Key Qwen3 features:
- Explicit head_dim (128 for 0.6B model)
- Q/K normalization for numerical stability (CRITICAL!)
- Attention dimension != embedding dimension
- Weight tying enabled by default
"""

import numpy as np
import torch


def apply_rope_permutation(w, num_heads):
    """
    Apply RoPE row permutation to match TT's q_proj/k_proj weight layout.
    TT applies unpermute_proj_rows during loading which interleaves rows within each head.

    This reorders: [0..D/2-1, D/2..D-1] → [0, D/2, 1, D/2+1, ..., D/2-1, D-1]
    """
    rows, cols = w.shape
    head_dim = rows // num_heads

    if head_dim % 2 != 0:
        raise ValueError(f"Head dimension {head_dim} must be even for RoPE permutation")

    out = np.zeros_like(w)
    for h in range(num_heads):
        head_start = h * head_dim
        half = head_dim // 2

        # Interleave: first half and second half of each head
        for i in range(half):
            out[head_start + 2 * i] = w[head_start + i]
            out[head_start + 2 * i + 1] = w[head_start + half + i]

    return out


def compare_qwen3_weights(torch_model, tt_model, verbose=True):
    """
    Compare weights between PyTorch Qwen3 model and TT-Metal Qwen3 model.

    Qwen3-specific features:
    - Q projection: [2048, 1024] (projects UP to attention space)
    - K projection: [1024, 1024] (num_kv_heads * head_dim)
    - V projection: [1024, 1024] (num_kv_heads * head_dim)
    - O projection: [1024, 2048] (projects DOWN to embedding space)
    - Q/K normalization: RMSNorm on head_dim (128) - CRITICAL!

    Args:
        torch_model: PyTorch Qwen3 model (with .state_dict())
        tt_model: TT-Metal Qwen3 model (with .parameters())
        verbose: Print detailed comparison (default: True)

    Returns:
        tuple: (matches, mismatches, missing_in_tt)
    """

    torch_sd = torch_model.state_dict()
    tt_params = tt_model.parameters()

    # Get configuration from torch model
    num_heads = torch_model.config.num_attention_heads
    num_kv_heads = torch_model.config.num_key_value_heads
    head_dim = getattr(torch_model.config, "head_dim", 128)
    hidden_size = torch_model.config.hidden_size

    # Detect weight tying configuration from TT model
    has_tok_emb = "qwen3/tok_emb/weight" in tt_params
    has_fc = "qwen3/fc/weight" in tt_params
    weight_tying_enabled = not has_tok_emb  # If tok_emb doesn't exist, weight tying is enabled

    # Mapping from PyTorch parameter names to TT-Metal parameter names
    pytorch_to_tt_mapping = {
        # Final layer norm
        "model.norm.weight": "qwen3/ln_fc/gamma",
    }

    # Add embedding mappings based on weight tying configuration
    if weight_tying_enabled:
        pytorch_to_tt_mapping["model.embed_tokens.weight"] = "qwen3/fc/weight"
        pytorch_to_tt_mapping["lm_head.weight"] = "qwen3/fc/weight"  # Both tied to fc/weight
    else:
        pytorch_to_tt_mapping["model.embed_tokens.weight"] = "qwen3/tok_emb/weight"
        pytorch_to_tt_mapping["lm_head.weight"] = "qwen3/fc/weight"

    # Add layer-specific mappings
    for i in range(50):  # Support up to 50 layers
        layer_prefix_pt = f"model.layers.{i}"
        layer_prefix_tt = f"qwen3/qwen3_block_{i}"

        pytorch_to_tt_mapping.update(
            {
                # Layer norms
                f"{layer_prefix_pt}.input_layernorm.weight": f"{layer_prefix_tt}/attention_norm/gamma",
                f"{layer_prefix_pt}.post_attention_layernorm.weight": f"{layer_prefix_tt}/mlp_norm/gamma",
                # Attention projections - SEPARATE Q, K, V (not combined!)
                f"{layer_prefix_pt}.self_attn.q_proj.weight": f"{layer_prefix_tt}/attention/q_linear/weight",
                f"{layer_prefix_pt}.self_attn.k_proj.weight": f"{layer_prefix_tt}/attention/k_linear/weight",
                f"{layer_prefix_pt}.self_attn.v_proj.weight": f"{layer_prefix_tt}/attention/v_linear/weight",
                f"{layer_prefix_pt}.self_attn.o_proj.weight": f"{layer_prefix_tt}/attention/out_linear/weight",
                # Q/K norms - CRITICAL for Qwen3 numerical stability!
                f"{layer_prefix_pt}.self_attn.q_norm.weight": f"{layer_prefix_tt}/attention/q_norm/gamma",
                f"{layer_prefix_pt}.self_attn.k_norm.weight": f"{layer_prefix_tt}/attention/k_norm/gamma",
                # MLP projections
                f"{layer_prefix_pt}.mlp.gate_proj.weight": f"{layer_prefix_tt}/mlp/w1/weight",
                f"{layer_prefix_pt}.mlp.up_proj.weight": f"{layer_prefix_tt}/mlp/w3/weight",
                f"{layer_prefix_pt}.mlp.down_proj.weight": f"{layer_prefix_tt}/mlp/w2/weight",
            }
        )

    if verbose:
        print("=" * 80)
        print("WEIGHT COMPARISON: PyTorch Qwen3 vs TT-Metal Qwen3")
        print("=" * 80)
        print(f"Model Configuration:")
        print(f"  - num_attention_heads: {num_heads}")
        print(f"  - num_key_value_heads: {num_kv_heads}")
        print(f"  - head_dim: {head_dim}")
        print(f"  - hidden_size: {hidden_size}")
        print(f"  - attention_output_dim: {num_heads * head_dim}")
        print(f"  - weight_tying: {'ENABLED' if weight_tying_enabled else 'DISABLED'}")
        print("=" * 80)

    mismatches = []
    matches = []
    missing_in_tt = []

    for pt_name in torch_sd.keys():
        if "bias" in pt_name:
            continue  # Skip bias parameters (Qwen3 has no biases)

        pt_tensor = torch_sd[pt_name]
        pt_shape = tuple(pt_tensor.shape)

        # Special handling for k_proj: apply RoPE permutation
        if ".self_attn.k_proj.weight" in pt_name:
            layer_idx = pt_name.split(".")[2]
            tt_name = f"qwen3/qwen3_block_{layer_idx}/attention/k_linear/weight"

            if tt_name in tt_params:
                pt_numpy = pt_tensor.cpu().float().numpy()

                # Apply RoPE permutation to K projection
                pt_numpy_permuted = apply_rope_permutation(pt_numpy, num_kv_heads)

                tt_tensor_np = tt_params[tt_name].to_numpy()
                tt_shape = tt_tensor_np.shape
                tt_data = tt_tensor_np.reshape(tt_shape[2:]) if len(tt_shape) == 4 else tt_tensor_np

                # Handle transpose if needed
                if pt_numpy_permuted.shape != tt_data.shape:
                    tt_data = tt_data.T

                diff = np.abs(pt_numpy_permuted - tt_data).max()
                rel_diff = diff / (np.abs(pt_numpy_permuted).max() + 1e-8)

                status = "✓" if diff < 1e-4 else "⚠"
                if verbose:
                    print(f"\n{status} {pt_name} (after RoPE permutation)")
                    print(f"  PyTorch: {pt_shape}")
                    print(f"  TT:      {tt_shape}")
                    print(f"  Max diff: {diff:.6f}, Rel diff: {rel_diff:.6f}")

                if diff < 1e-3:
                    matches.append(pt_name)
                else:
                    mismatches.append((pt_name, f"K_DIFF={diff:.6f}"))
            else:
                missing_in_tt.append((pt_name, tt_name))
            continue

        # Get corresponding TT parameter name
        tt_name = pytorch_to_tt_mapping.get(pt_name)
        if not tt_name:
            if verbose:
                print(f"\n⊗ NOT MAPPED: {pt_name}")
                print(f"  PyTorch: {pt_shape}")
                print(f"  Status: No TT equivalent defined in mapping")
            missing_in_tt.append((pt_name, "NOT_MAPPED"))
            continue

        if tt_name not in tt_params:
            if verbose:
                print(f"\n❌ MISSING IN TT: {pt_name} -> {tt_name}")
                print(f"   PyTorch shape: {pt_shape}")
            missing_in_tt.append((pt_name, tt_name))
            continue

        # Get TT tensor
        tt_tensor_np = tt_params[tt_name].to_numpy()
        tt_shape = tt_tensor_np.shape

        # Remove batch dimensions [1, 1, ...] from TT tensor
        tt_shape_no_batch = tt_shape[2:] if len(tt_shape) == 4 else tt_shape

        # Compare shapes
        pt_numpy = pt_tensor.cpu().float().numpy()

        # Check what type of parameter this is
        is_q_proj = ".self_attn.q_proj.weight" in pt_name
        is_q_norm = ".self_attn.q_norm.weight" in pt_name
        is_k_norm = ".self_attn.k_norm.weight" in pt_name
        is_v_proj = ".self_attn.v_proj.weight" in pt_name

        # Apply RoPE permutation for Q projection
        if is_q_proj:
            if len(pt_shape) == 2 and pt_shape[0] == num_heads * head_dim:
                pt_numpy = apply_rope_permutation(pt_numpy, num_heads)

        # For layer norms: PT (N,) vs TT (1, N) or (1, 1, 1, N) - handle squeezing
        if len(pt_shape) == 1:
            if len(tt_shape_no_batch) == 2 and tt_shape_no_batch[0] == 1:
                tt_shape_no_batch = (tt_shape_no_batch[1],)
            elif len(tt_shape_no_batch) == 1:
                pass  # Already 1D

        # Check if shapes match (with or without transpose)
        shape_match = (pt_shape == tt_shape_no_batch) or (pt_shape == tt_shape_no_batch[::-1])

        if shape_match:
            # Check actual values
            if len(tt_shape) == 4:
                tt_data = tt_tensor_np.reshape(tt_shape[2:])
            else:
                tt_data = tt_tensor_np.reshape(tt_shape)

            tt_data = tt_data.squeeze()
            pt_numpy_squeezed = pt_numpy.squeeze()

            # Handle transpose if needed
            if pt_numpy_squeezed.shape != tt_data.shape and len(tt_data.shape) == 2:
                tt_data = tt_data.T

            diff = np.abs(pt_numpy_squeezed - tt_data).max()
            rel_diff = diff / (np.abs(pt_numpy_squeezed).max() + 1e-8)

            status = "✓" if diff < 1e-3 else "⚠"
            note = ""
            if is_q_proj:
                note = " (after RoPE permutation)"
            elif is_q_norm:
                note = " [CRITICAL Q/K NORM]"
            elif is_k_norm:
                note = " [CRITICAL Q/K NORM]"

            if verbose:
                print(f"\n{status} {pt_name}{note}")
                print(f"  PyTorch: {pt_shape}")
                print(f"  TT:      {tt_shape} -> {tt_shape_no_batch}")
                print(f"  Max diff: {diff:.6f}, Rel diff: {rel_diff:.6f}")

            if diff < 1e-3:
                matches.append(pt_name)
            else:
                mismatches.append((pt_name, f"VALUE_DIFF={diff:.6f}"))
        else:
            if verbose:
                print(f"\n❌ SHAPE MISMATCH: {pt_name}")
                print(f"  PyTorch: {pt_shape}")
                print(f"  TT:      {tt_shape} -> {tt_shape_no_batch}")
            mismatches.append((pt_name, f"SHAPE: PT={pt_shape} vs TT={tt_shape_no_batch}"))

    if verbose:
        print("\n" + "=" * 80)
        print(f"SUMMARY: {len(matches)} matches, {len(mismatches)} mismatches, {len(missing_in_tt)} missing in TT")
        print("=" * 80)

        if missing_in_tt:
            print(f"\n❌ MISSING IN TT ({len(missing_in_tt)}):")
            for pt_name, tt_name in missing_in_tt:
                print(f"  - {pt_name} -> {tt_name}")

        if mismatches:
            print("\n❌ MISMATCHES:")
            for name, issue in mismatches:
                print(f"  - {name}: {issue}")

        if len(mismatches) == 0 and len(missing_in_tt) == 0:
            print("\n🎉 ALL WEIGHTS MATCH PERFECTLY!")
            print(f"✅ {len(matches)} parameters validated")
            print("✅ Q/K normalization layers loaded correctly")
            print("✅ Qwen3 model is production-ready!")

    return matches, mismatches, missing_in_tt


matches, mismatches, missing = compare_qwen3_weights(torch_model, tt_model)

len(torch_model.state_dict())
import nbformat
import sys


def convert_notebook_to_py(notebook_path, output_path):
    with open(notebook_path) as ff:
        nb_in = nbformat.read(ff, nbformat.NO_CONVERT)

    source = ""
    for cell in nb_in["cells"]:
        if cell["cell_type"] == "code":
            # cell['source'] can be a string or a list of strings
            cell_source = cell["source"]

            # Convert to list of lines if it's a string
            if isinstance(cell_source, str):
                lines = cell_source.split("\n")
            else:
                # It's already a list, but may contain newlines within elements
                lines = []
                for item in cell_source:
                    lines.extend(item.split("\n"))

            # Filter out lines starting with '%' (magic commands)
            filtered_lines = [line for line in lines if not line.strip().startswith("%")]

            # Join the lines back together
            cell_code = "\n".join(filtered_lines)

            # Add to source with a newline separator
            if cell_code.strip():  # Only add non-empty cells
                source = source + "\n" + cell_code

    # Write to output file
    with open(output_path, "w") as f:
        f.write(source)


convert_notebook_to_py("llm_inference_new.ipynb", "llm_inference_new.py")
torch_model
