# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ipdb",
#     "msgpack-numpy",
#     "numpy",
#     "torch",
#     "transformers",
# ]
# ///

import argparse
from pdb import pm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import msgpack
import msgpack_numpy  # required for serializing numpy arrays
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
import functools
import argparse
import re
import torch
import pdb
import sys
import io
import inspect
import os

hf_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1")
emb = hf_model.get_submodule("model.embed_tokens")

# test token embedding
input = np.load("/home/j/intermediate_results/test_input_tokens.npy")
input = torch.tensor(input, dtype=torch.int)

tok_emb_res = emb(input)
tok_emb_res = tok_emb_res.detach().cpu().numpy()

np.save("/home/j/intermediate_results/test_embedded_tokens_new.npy", tok_emb_res)


# testing first qkv projections
def interleave_halves(x, dim: int = -1) -> torch.Tensor:
    """
    Convert contiguous split-half layout  ➜  interleaved real/imag layout
    by hopping through NumPy (always contiguous).

    [a0 ... an-1 | b0 ... bn-1] → [a0, b0, a1, b1, ...]
    """
    d = x.shape[dim]
    assert d % 2 == 0, "hidden dim must be even"

    np_arr = x
    a, b = np.split(np_arr, 2, axis=-1)
    inter = np.stack((a, b), axis=-1).reshape(np_arr.shape)

    return inter


first_q_proj = hf_model.state_dict()["model.layers.0.self_attn.q_proj.weight"]
first_q_proj = first_q_proj.detach().cpu().numpy()
seq_len = 32
num_heads = 32
head_dim = 64
emb_dim = num_heads * head_dim
q_test_input = np.random.randn(1, 1, seq_len, emb_dim)
np.save("/home/j/intermediate_results/first_q_test_input.npy", q_test_input.astype(np.float32))
q_test_input = q_test_input.reshape(1, seq_len, emb_dim)
expected_q = np.einsum("dc,bsc->bsd", first_q_proj, q_test_input)
expected_q = expected_q.reshape(1, seq_len, num_heads, head_dim)
expected_q = interleave_halves(expected_q)
expected_q = expected_q.reshape(1, 1, seq_len, emb_dim)
np.save("/home/j/intermediate_results/expected_first_q_result.npy", expected_q.astype(np.float32))

# testing first kv projection
num_groups = 4
first_k_proj = hf_model.state_dict()["model.layers.0.self_attn.k_proj.weight"]
first_k_proj = first_k_proj.detach().cpu().numpy()
first_v_proj = hf_model.state_dict()["model.layers.0.self_attn.v_proj.weight"]
first_v_proj = first_v_proj.detach().cpu().numpy()

# Use the same input as q_proj for consistency
kv_test_input = q_test_input  # Shape (1, seq_len, num_heads*head_dim)

expected_k = np.einsum("dc,bsc->bsd", first_k_proj, kv_test_input)
expected_v = np.einsum("dc,bsc->bsd", first_v_proj, kv_test_input)

# Reshape and interleave k
expected_k = expected_k.reshape(1, seq_len, num_groups, head_dim)
expected_k = interleave_halves(expected_k)
expected_k = expected_k.reshape(1, 1, seq_len, num_groups * head_dim)

# Reshape and interleave v
expected_v = expected_v.reshape(1, seq_len, num_groups, head_dim)
expected_v = interleave_halves(expected_v)
expected_v = expected_v.reshape(1, 1, seq_len, num_groups * head_dim)

# Concatenate k and v along the last dimension
expected_kv = np.concatenate((expected_k, expected_v), axis=-1)
np.save("/home/j/intermediate_results/expected_first_kv_result.npy", expected_kv.astype(np.float32))

expected_k_grouped = expected_k.reshape(1, seq_len, num_groups, head_dim)
expected_k_grouped = expected_k_grouped.transpose(0, 2, 1, 3)
expected_v_grouped = expected_v.reshape(1, seq_len, num_groups, head_dim)
expected_v_grouped = expected_v_grouped.transpose(0, 2, 1, 3)
np.save("/home/j/intermediate_results/expected_first_k_result.npy", expected_k_grouped.astype(np.float32))
np.save("/home/j/intermediate_results/expected_first_v_result.npy", expected_v_grouped.astype(np.float32))


# test attention norm
attn_norm = hf_model.get_submodule("model.layers.0.input_layernorm")
attn_norm_input = tok_emb_res
attn_norm_res = attn_norm(torch.tensor(attn_norm_input, dtype=torch.float32))
attn_norm_res = attn_norm_res.detach().cpu().numpy()
np.save("/home/j/intermediate_results/expected_first_attn_norm_result.npy", attn_norm_res)

# test grouped query attention
# gqa = hf_model.get_submodule("model.layers.0.self_attn")
# gqa_input = attn_norm_res
# gqa_input = gqa_input.reshape(1,seq_len,emb_dim)
# gqa_mask = np.ones((1,1,seq_len,seq_len))
# gqa_mask[:,:,:seq_len-1,:] = 0

# np.save("/home/j/intermediate_results/first_gqa_input.npy", gqa_input)
# np.save("/home/j/intermediate_results/first_gqa_mask.npy", gqa_mask)
# gqa_mask = torch.tensor(gqa_mask, dtype=torch.float32)
# gqa_input = torch.tensor(gqa_input, dtype=torch.float32)

# gqa_res = gqa(gqa_input, gqa_mask)
# gqa_res = gqa_res.detach().cpu().numpy()
# np.save("/home/j/intermediate_results/expected_first_gqa_result.npy", gqa_res)

import bdb


class ReturnDebugger(bdb.Bdb):
    def __init__(self, target_func):
        super().__init__()
        self.target_code = target_func.__code__
        self.results = {"breaks": [], "returns": []}

    def user_return(self, frame, return_value):
        if frame.f_code == self.target_code:
            self.results["returns"].append(
                {
                    "return_value": return_value,
                    "function_name": self.target_code.co_name,
                    "locals": frame.f_locals.copy(),
                }
            )

    def user_line(self, frame):
        if frame.f_code == self.target_code:
            break_locals = frame.f_locals.copy()
            self.results["breaks"].append(
                {
                    "function_name": self.target_code.co_name,
                    "file_name": self.target_code.co_filename,
                    "line_number": self.target_code.co_firstlineno,
                    "locals": break_locals,
                }
            )


def run_and_capture(target_func, entry_callable):
    _, start = inspect.getsourcelines(target_func)
    dbg = ReturnDebugger(target_func)
    dbg.run(entry_callable.__code__, globals(), locals())
    return dbg.results


# Full inference pass to capture self_attn state
input_text = "The capital of Canada is "
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
input_ids = tokenizer.encode(input_text, return_tensors="pt")
target_length = 32
current_length = input_ids.shape[1]
pad_len = target_length - current_length
if tokenizer.pad_token_id is None:
    # Common practice for models without a specific pad token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Tokenizer does not have a pad token, using EOS token ID: {tokenizer.pad_token_id}")

if pad_len > 0:
    print(f"Padding input_ids from {current_length} to {target_length}")
    # Assuming padding on the right, which is standard for decoder models
    padding = torch.full(
        (input_ids.shape[0], pad_len), tokenizer.pad_token_id, dtype=input_ids.dtype, device=input_ids.device
    )
    input_ids = torch.cat([input_ids, padding], dim=1)
elif pad_len < 0:
    print(f"Warning: input_ids length ({current_length}) is greater than target length ({target_length}). Truncating.")
    input_ids = input_ids[:, :target_length]

print(f"Shape of input_ids after padding/truncation: {input_ids.shape}")

target_module = hf_model.get_submodule("model.layers.0.self_attn")
target_func = target_module.forward


def inference_call():
    # Run the model forward pass. Ensure inputs are on the correct device if using GPU.
    # Assuming hf_model is already on the desired device (e.g., CPU for this example)
    with torch.no_grad():  # Disable gradient calculation for inference
        _ = hf_model(input_ids=input_ids)


print("Running inference to capture self_attn state...")
captured_data = run_and_capture(target_func, inference_call)
print("Capture complete.")

attn_interface = None  # grabbing attn interface for sdpa test below

if captured_data:
    first_attn_capture = captured_data["returns"][0]["locals"]
    attn_interface = first_attn_capture["attention_interface"]
    attn_output = first_attn_capture["attn_output"]  # Assuming the first element is the hidden states outpu
    attn_output_np = attn_output.detach().cpu().numpy()
    np.save("/home/j/intermediate_results/expected_first_attn_output.npy", attn_output_np)

    attn_input = first_attn_capture["hidden_states"]
    attn_input_np = attn_input.detach().cpu().numpy()
    attn_mask = first_attn_capture["attention_mask"]
    if attn_mask is None:
        # Create the standard causal mask if none is provided
        bsz, tgt_len, _ = attn_input.shape
        device = attn_input.device
        dtype = attn_input.dtype

        # Create a lower triangular boolean mask
        # Equivalent to HF's _make_causal_mask logic for seq_len > 1 and no past_key_values
        mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device, dtype=torch.bool))
        # Expand to 4D: (bsz, 1, tgt_len, tgt_len) as commonly used in HF transformers
        causal_mask_4d = mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, tgt_len, tgt_len)

        # Convert to float mask (0.0 where True/allowed, large negative where False/masked)
        # This matches the format expected by attention mechanisms before softmax
        attn_mask_float = torch.zeros_like(causal_mask_4d, dtype=dtype)
        attn_mask_float.masked_fill_(causal_mask_4d == 0, torch.finfo(dtype).min)

        # Prepare the numpy array for saving
        attn_mask_np = attn_mask_float.detach().cpu().numpy()
    else:
        attn_mask_np = attn_mask.detach().cpu().numpy()
    np.save("/home/j/intermediate_results/expected_first_attn_input.npy", attn_input_np)
    np.save("/home/j/intermediate_results/expected_first_attn_mask.npy", attn_mask_np)
    print("Saved captured self_attn input, mask, and output.")
else:
    raise Exception("Failed to capture data. Did the target function execute?")


## dump sdpa results
assert attn_interface is not None
np.random.seed(42)

batch_size = 1
dtype = np.float32
# Generate random Q, K, V using existing dimensions
# Q: (batch_size, num_heads, seq_len, head_dim)
# K: (batch_size, num_key_value_heads, seq_len, head_dim)
# V: (batch_size, num_key_value_heads, seq_len, head_dim)
rand_q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(dtype)
rand_k = np.random.randn(batch_size, num_groups, seq_len, head_dim).astype(dtype)
rand_v = np.random.randn(batch_size, num_groups, seq_len, head_dim).astype(dtype)

# Generate causal attention mask (4D)
causal_mask_np = np.triu(np.ones((seq_len, seq_len), dtype=dtype) * np.finfo(dtype).min, k=1)
causal_mask_np = np.broadcast_to(causal_mask_np[None, None, :, :], (batch_size, 1, seq_len, seq_len))

# Save random inputs
np.save("/home/j/intermediate_results/random_sdpa_q.npy", rand_q)
np.save("/home/j/intermediate_results/random_sdpa_k.npy", rand_k)
np.save("/home/j/intermediate_results/random_sdpa_v.npy", rand_v)
np.save("/home/j/intermediate_results/random_sdpa_mask.npy", causal_mask_np)
print("Saved random Q, K, V, and Mask.")

# Convert to tensors
rand_q_t = torch.tensor(rand_q)
rand_k_t = torch.tensor(rand_k)
rand_v_t = torch.tensor(rand_v)
causal_mask_t = torch.tensor(causal_mask_np)

# Call the captured interface function directly
# sdpa_interface holds the reference to the attention function (e.g., eager_attention_forward)
rand_attn_output, _ = attn_interface(
    hf_model.get_submodule("model.layers.0.self_attn"),  # The bound 'self' for the method
    rand_q_t,
    rand_k_t,
    rand_v_t,
    causal_mask_t,
    scaling=1.0,  # No scaling applied
    dropout=0.0,
)

# Save the result
rand_attn_output_np = rand_attn_output.detach().cpu().numpy()
np.save("/home/j/intermediate_results/random_sdpa_res.npy", rand_attn_output_np)

print("Called SDPA interface with random inputs and saved the result.")
print("Random SDPA output shape:", rand_attn_output_np.shape)

## dump torch sdpa results
np.random.seed(42)

batch_size = 1
dtype = np.float32
# Generate random Q, K, V using existing dimensions
# Q: (batch_size, num_heads, seq_len, head_dim)
# K: (batch_size, num_key_value_heads, seq_len, head_dim)
# V: (batch_size, num_key_value_heads, seq_len, head_dim)
rand_q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(dtype)
rand_k = np.random.randn(batch_size, num_groups, seq_len, head_dim).astype(dtype)
rand_v = np.random.randn(batch_size, num_groups, seq_len, head_dim).astype(dtype)

# Generate causal attention mask (4D)
causal_mask_np = np.triu(np.ones((seq_len, seq_len), dtype=dtype) * np.finfo(dtype).min, k=1)
causal_mask_np = np.broadcast_to(causal_mask_np[None, None, :, :], (batch_size, 1, seq_len, seq_len))

# Save random inputs
np.save("/home/j/intermediate_results/torch_sdpa_q.npy", rand_q)
np.save("/home/j/intermediate_results/torch_sdpa_k.npy", rand_k)
np.save("/home/j/intermediate_results/torch_sdpa_v.npy", rand_v)
np.save("/home/j/intermediate_results/torch_sdpa_mask.npy", causal_mask_np)
print("Saved torch Q, K, V, and Mask.")

# Convert to tensors
query = torch.tensor(rand_q)
key = torch.tensor(rand_k)
value = torch.tensor(rand_v)
causal_mask_t = torch.tensor(causal_mask_np)

# Call the captured interface function directly
# sdpa_interface holds the reference to the attention function (e.g., eager_attention_forward)
attn_output = torch.nn.functional.scaled_dot_product_attention(query, key, value, causal_mask_t, enable_gqa=True)

# Save the result
attn_output_np = attn_output.detach().cpu().numpy()
np.save("/home/j/intermediate_results/torch_sdpa_res.npy", attn_output_np)

print("Called SDPA interface with torch inputs and saved the result.")
print("Torch SDPA output shape:", attn_output_np.shape)

import torch
import numpy as np
import os  # Added import

## testing group shared matmul

# Assume dtype is defined earlier, but define it here for clarity if needed
dtype = np.float32


def group_shared_matmul(query_tensor, kv_tensor, transpose_a=False, transpose_b=False):
    """
    Python implementation of group_shared_matmul function, handling GQA broadcasting.

    Performs matrix multiplication between query_tensor and kv_tensor, potentially
    transposing the last two dimensions, and broadcasting kv_tensor if needed.

    Args:
        query_tensor: tensor of shape (batch_num, heads, dim1_q, dim2_q)
        kv_tensor: tensor of shape (batch_num, groups, dim1_kv, dim2_kv)
                      or (batch_num, heads, dim1_kv, dim2_kv) if heads == groups.
        transpose_a: whether to transpose the last two dimensions of query_tensor before matmul.
        transpose_b: whether to transpose the last two dimensions of kv_tensor before matmul.

    Returns:
        Result tensor of the matrix multiplication.

    Raises:
        ValueError: If shapes are incompatible for matrix multiplication after transpose
                    and broadcasting, or if heads is not divisible by groups.
    """
    # Get shapes
    batch_num, heads, dim1_q, dim2_q = query_tensor.shape
    batch_num_v, groups, dim1_kv, dim2_kv = kv_tensor.shape

    if batch_num != batch_num_v:
        raise ValueError(
            f"query_tensor and kv_tensor must have the same batch size, got shapes {query_tensor.shape} and {kv_tensor.shape}"
        )
    if heads % groups != 0:
        raise ValueError(f"Number of heads ({heads}) must be divisible by the number of groups ({groups})")

    # Determine shapes *before* matmul based on transpose flags
    # Shape format: (batch, heads_or_groups, M_dim, K_dim)
    q_shape_pre_matmul = (batch_num, heads, dim2_q, dim1_q) if transpose_a else (batch_num, heads, dim1_q, dim2_q)
    kv_shape_pre_matmul = (
        (batch_num, groups, dim2_kv, dim1_kv) if transpose_b else (batch_num, groups, dim1_kv, dim2_kv)
    )

    # K dimension (inner dimension for matmul)
    K_q = q_shape_pre_matmul[-1]
    K_kv = kv_shape_pre_matmul[-2]

    if K_q != K_kv:
        raise ValueError(
            f"Inner dimensions for matrix multiplication must match after transpose. "
            f"Query shape pre-matmul: {q_shape_pre_matmul}, KV shape pre-matmul: {kv_shape_pre_matmul}. "
            f"Inner dimensions: {K_q} vs {K_kv}"
        )

    # Output dimensions M, N
    M = q_shape_pre_matmul[-2]
    N = kv_shape_pre_matmul[-1]

    # --- Prepare tensors for matmul ---
    # Use functional transpose (.transpose returns a view, not in-place)
    q_eff = query_tensor.transpose(-2, -1) if transpose_a else query_tensor
    # q_eff shape: (batch, heads, M, K_q)

    if heads == groups:
        # MHA case: No broadcasting needed
        if kv_tensor.shape[1] != heads:
            raise ValueError(
                f"Expected kv_tensor to have {heads} heads when heads==groups, but got {kv_tensor.shape[1]}"
            )

        kv_eff = kv_tensor.transpose(-2, -1) if transpose_b else kv_tensor
        # kv_eff shape: (batch, heads, K_kv, N)

        # Perform matmul directly: (b, h, M, K_q) @ (b, h, K_kv, N) -> (b, h, M, N)
        result = torch.matmul(q_eff, kv_eff)

    else:  # GQA case: heads > groups
        # Reshape query for grouped computation
        # (b, h, M, K_q) -> (b * g, h/g, M, K_q)
        q_grouped = q_eff.reshape(batch_num * groups, heads // groups, M, K_q)

        # Prepare kv_tensor: potentially transpose, then reshape and repeat
        # kv_tensor original shape: (b, g, dim1_kv, dim2_kv)
        kv_transposed = kv_tensor.transpose(-2, -1) if transpose_b else kv_tensor
        # kv_transposed shape: (b, g, K_kv, N)

        # Reshape and repeat kv_tensor to match query groups
        # (b, g, K_kv, N) -> (b * g, 1, K_kv, N)
        kv_batched = kv_transposed.reshape(batch_num * groups, 1, K_kv, N)
        # (b * g, 1, K_kv, N) -> (b * g, h/g, K_kv, N)
        kv_repeated = kv_batched.repeat(1, heads // groups, 1, 1)
        # kv_repeated shape: (b*g, h/g, K_kv, N)

        # Perform matrix multiplication
        # (b*g, h/g, M, K_q) @ (b*g, h/g, K_kv, N) -> (b*g, h/g, M, N)  (since K_q == K_kv)
        bcasted_mm = torch.matmul(q_grouped, kv_repeated)

        # Reshape back to original batch and heads dimensions
        # (b*g, h/g, M, N) -> (b, h, M, N)
        result = bcasted_mm.reshape(batch_num, heads, M, N)

    # Final check of output shape
    expected_shape = (batch_num, heads, M, N)
    if result.shape != expected_shape:
        # This should ideally not happen if logic is correct, but good safeguard
        raise RuntimeError(f"Internal error: Unexpected output shape. Got {result.shape}, expected {expected_shape}")

    return result


def sum_over_groups(ungrouped_tensor, groups):
    """
    Python implementation of sum_over_groups function.

    Sums the tensor over groups of heads. Assumes the head dimension is dim 1.

    Args:
        ungrouped_tensor: tensor of shape (batch_num, num_heads, ...)
                          Typically (batch_num, num_heads, seq_len, embedding_dim)
        groups: number of groups (must divide num_heads)

    Returns:
        Tensor with values summed over groups, shape (batch_num, groups, ...)
    """
    shape = ungrouped_tensor.shape
    if len(shape) < 2:
        raise ValueError(f"ungrouped_tensor must have at least 2 dimensions, got shape {shape}")

    batch_num = shape[0]
    num_heads = shape[1]
    remaining_dims = shape[2:]

    if num_heads % groups != 0:
        raise ValueError(f"Number of heads ({num_heads}) must be divisible by the number of groups ({groups})")

    if groups == num_heads:
        # Group size is 1, nothing to sum, but need to ensure shape is (b, g, ...)
        # If the input already has the 'groups' dimension, return it.
        # If it has the 'heads' dimension where heads=groups, it's already correct.
        return ungrouped_tensor  # Shape (b, h, ...) == (b, g, ...)

    # Reshape for grouped computation: (b, h, ...) -> (b, g, h/g, ...)
    reshaped_tensor = ungrouped_tensor.reshape(batch_num, groups, num_heads // groups, *remaining_dims)

    # Sum over the heads-within-group dimension (dim=2)
    summed_tensor = reshaped_tensor.sum(dim=2)  # Shape: (b, g, ...)

    # Verify final shape
    expected_shape = (batch_num, groups, *remaining_dims)
    if summed_tensor.shape != expected_shape:
        raise RuntimeError(
            f"Internal error: Unexpected output shape after sum. Got {summed_tensor.shape}, expected {expected_shape}"
        )

    return summed_tensor


# --- Generate test tensors for group_shared_matmul and sum_over_groups ---
np.random.seed(43)
output_dir = "/home/j/intermediate_results"
os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

# --- Test Case 1: MHA (heads == groups) ---
test1_batch = 2
test1_heads = 4
test1_groups = 4
test1_seq_len_q = 16
test1_seq_len_kv = 16  # Can be different, but keep same for simplicity here
test1_dim = 32

print(f"\n--- Generating Test Case 1: MHA (heads={test1_heads}, groups={test1_groups}) ---")
# Q: (b, h, s_q, d)
test1_q_np = np.random.randn(test1_batch, test1_heads, test1_seq_len_q, test1_dim).astype(dtype)
# K: (b, g, s_kv, d) -> g=h
test1_k_np = np.random.randn(test1_batch, test1_groups, test1_seq_len_kv, test1_dim).astype(dtype)
# V: (b, g, s_kv, d_v) -> g=h, d_v=d
test1_v_np = np.random.randn(test1_batch, test1_groups, test1_seq_len_kv, test1_dim).astype(dtype)

test1_q_t = torch.tensor(test1_q_np)
test1_k_t = torch.tensor(test1_k_np)
test1_v_t = torch.tensor(test1_v_np)

# Test 1a: Q @ K^T (transpose_a=False, transpose_b=True)
# Input Q: (b, h, s_q, d), Input K: (b, h, s_kv, d)
# Output Scores: (b, h, s_q, s_kv)
print("Test 1a: Q @ K^T")
test1_scores_t = group_shared_matmul(test1_q_t, test1_k_t, transpose_a=False, transpose_b=True)
print(f"  Q shape: {test1_q_t.shape}, K shape: {test1_k_t.shape}")
print(f"  Output Scores shape: {test1_scores_t.shape}")

# Test 1b: Scores @ V (transpose_a=False, transpose_b=False)
# Input Scores: (b, h, s_q, s_kv), Input V: (b, h, s_kv, d)
# Output Result: (b, h, s_q, d)
print("Test 1b: Scores @ V")
test1_result_t = group_shared_matmul(test1_scores_t, test1_v_t, transpose_a=False, transpose_b=False)
print(f"  Scores shape: {test1_scores_t.shape}, V shape: {test1_v_t.shape}")
print(f"  Output Result shape: {test1_result_t.shape}")

# Test 1c: Sum over groups (should be identity op since groups=heads)
test1_grad_np = np.random.randn(test1_batch, test1_heads, test1_seq_len_q, test1_dim).astype(dtype)
test1_grad_t = torch.tensor(test1_grad_np)
test1_sum_result_t = sum_over_groups(test1_grad_t, test1_groups)
print("Test 1c: Sum over groups")
print(f"  Input Grad shape: {test1_grad_t.shape}")
print(f"  Output Summed Grad shape: {test1_sum_result_t.shape}")
assert torch.equal(test1_grad_t, test1_sum_result_t)  # Verify it's unchanged


# --- Test Case 2: GQA (heads > groups) ---
test2_batch = 2
test2_heads = 8
test2_groups = 4  # heads % groups == 0
test2_seq_len_q = 16
test2_seq_len_kv = 16
test2_dim = 32

print(f"\n--- Generating Test Case 2: GQA (heads={test2_heads}, groups={test2_groups}) ---")
# Q: (b, h, s_q, d)
test2_q_np = np.random.randn(test2_batch, test2_heads, test2_seq_len_q, test2_dim).astype(dtype)
# K: (b, g, s_kv, d)
test2_k_np = np.random.randn(test2_batch, test2_groups, test2_seq_len_kv, test2_dim).astype(dtype)
# V: (b, g, s_kv, d_v) -> d_v=d
test2_v_np = np.random.randn(test2_batch, test2_groups, test2_seq_len_kv, test2_dim).astype(dtype)

test2_q_t = torch.tensor(test2_q_np)
test2_k_t = torch.tensor(test2_k_np)
test2_v_t = torch.tensor(test2_v_np)

# Test 2a: Q @ K^T (transpose_a=False, transpose_b=True)
# Input Q: (b, h, s_q, d), Input K: (b, g, s_kv, d)
# Output Scores: (b, h, s_q, s_kv)
print("Test 2a: Q @ K^T")
test2_scores_t = group_shared_matmul(test2_q_t, test2_k_t, transpose_a=False, transpose_b=True)
print(f"  Q shape: {test2_q_t.shape}, K shape: {test2_k_t.shape}")
print(f"  Output Scores shape: {test2_scores_t.shape}")


# Test 2b: Scores @ V (transpose_a=False, transpose_b=False)
# Input Scores: (b, h, s_q, s_kv), Input V: (b, g, s_kv, d)
# Output Result: (b, h, s_q, d)
print("Test 2b: Scores @ V")
test2_result_t = group_shared_matmul(test2_scores_t, test2_v_t, transpose_a=False, transpose_b=False)
print(f"  Scores shape: {test2_scores_t.shape}, V shape: {test2_v_t.shape}")
print(f"  Output Result shape: {test2_result_t.shape}")

# Test 2c: Sum over groups
# Input Grad: (b, h, s_q, d)
test2_grad_np = np.random.randn(test2_batch, test2_heads, test2_seq_len_q, test2_dim).astype(dtype)
test2_grad_t = torch.tensor(test2_grad_np)
# Output Summed Grad: (b, g, s_q, d)
test2_sum_result_t = sum_over_groups(test2_grad_t, test2_groups)
print("Test 2c: Sum over groups")
print(f"  Input Grad shape: {test2_grad_t.shape}")
print(f"  Output Summed Grad shape: {test2_sum_result_t.shape}")
# Verification check (optional but good)
expected_sum_shape = (test2_batch, test2_groups, test2_seq_len_q, test2_dim)
assert test2_sum_result_t.shape == expected_sum_shape


# --- Save inputs and outputs ---
print(f"\n--- Saving test data to {output_dir} ---")

# Test case 1 (MHA)
np.save(os.path.join(output_dir, "mha_q.npy"), test1_q_np)
np.save(os.path.join(output_dir, "mha_k.npy"), test1_k_np)
np.save(os.path.join(output_dir, "mha_v.npy"), test1_v_np)
np.save(os.path.join(output_dir, "mha_scores_qkt.npy"), test1_scores_t.detach().cpu().numpy())
np.save(os.path.join(output_dir, "mha_result_scoresv.npy"), test1_result_t.detach().cpu().numpy())
np.save(os.path.join(output_dir, "mha_grad_in.npy"), test1_grad_np)
np.save(os.path.join(output_dir, "mha_grad_summed.npy"), test1_sum_result_t.detach().cpu().numpy())

# Test case 2 (GQA)
np.save(os.path.join(output_dir, "gqa_q.npy"), test2_q_np)
np.save(os.path.join(output_dir, "gqa_k.npy"), test2_k_np)
np.save(os.path.join(output_dir, "gqa_v.npy"), test2_v_np)
np.save(os.path.join(output_dir, "gqa_scores_qkt.npy"), test2_scores_t.detach().cpu().numpy())
np.save(os.path.join(output_dir, "gqa_result_scoresv.npy"), test2_result_t.detach().cpu().numpy())
np.save(os.path.join(output_dir, "gqa_grad_in.npy"), test2_grad_np)
np.save(os.path.join(output_dir, "gqa_grad_summed.npy"), test2_sum_result_t.detach().cpu().numpy())

print("Finished saving test data.")


## dump sdpa intermediate results
print("\n--- Generating SDPA Intermediate Test Data ---")
output_dir = "/home/j/intermediate_results"
os.makedirs(output_dir, exist_ok=True)
np.random.seed(44)  # Use a different seed
dtype = np.float32
scale_factor = 1.0  # Default, will be calculated based on head_dim

# --- MHA Case ---
print("--- MHA Case ---")
mha_batch = 1
mha_heads = 4
mha_groups = 4  # heads == groups
mha_seq_len = 16
mha_head_dim = 32
mha_scale = 1.0 / np.sqrt(mha_head_dim)

# Generate inputs
mha_q_np = np.random.randn(mha_batch, mha_heads, mha_seq_len, mha_head_dim).astype(dtype)
mha_k_np = np.random.randn(mha_batch, mha_groups, mha_seq_len, mha_head_dim).astype(dtype)
mha_v_np = np.random.randn(mha_batch, mha_groups, mha_seq_len, mha_head_dim).astype(dtype)
# Additive causal mask (0.0 where allowed, -inf where masked)
mha_mask_np = np.triu(np.ones((mha_seq_len, mha_seq_len), dtype=dtype) * np.finfo(dtype).min, k=1)
mha_mask_np = np.broadcast_to(mha_mask_np[None, None, :, :], (mha_batch, 1, mha_seq_len, mha_seq_len))

mha_q_t = torch.tensor(mha_q_np)
mha_k_t = torch.tensor(mha_k_np)
mha_v_t = torch.tensor(mha_v_np)
mha_mask_t = torch.tensor(mha_mask_np)

# Calculate intermediates
mha_q_scaled_t = mha_q_t * mha_scale
mha_qk_t = group_shared_matmul(mha_q_scaled_t, mha_k_t, transpose_a=False, transpose_b=True)
mha_qk_masked_t = mha_qk_t + mha_mask_t  # Additive mask
mha_attn_weights_t = torch.softmax(mha_qk_masked_t, dim=-1)
mha_attn_qkv_t = group_shared_matmul(mha_attn_weights_t, mha_v_t, transpose_a=False, transpose_b=False)

# Save MHA data
np.save(os.path.join(output_dir, "sdpa_interm_mha_q.npy"), mha_q_np)
np.save(os.path.join(output_dir, "sdpa_interm_mha_k.npy"), mha_k_np)
np.save(os.path.join(output_dir, "sdpa_interm_mha_v.npy"), mha_v_np)
np.save(os.path.join(output_dir, "sdpa_interm_mha_mask.npy"), mha_mask_np)
np.save(os.path.join(output_dir, "sdpa_interm_mha_q_scaled.npy"), mha_q_scaled_t.detach().cpu().numpy())
np.save(os.path.join(output_dir, "sdpa_interm_mha_qk_masked.npy"), mha_qk_masked_t.detach().cpu().numpy())
np.save(os.path.join(output_dir, "sdpa_interm_mha_attn_weights.npy"), mha_attn_weights_t.detach().cpu().numpy())
np.save(os.path.join(output_dir, "sdpa_interm_mha_attn_qkv.npy"), mha_attn_qkv_t.detach().cpu().numpy())
print("Saved MHA intermediate data.")

# --- GQA Case ---
print("--- GQA Case ---")
gqa_batch = 1
gqa_heads = 8
gqa_groups = 2  # heads > groups
gqa_seq_len = 16
gqa_head_dim = 32
gqa_scale = 1.0 / np.sqrt(gqa_head_dim)

# Generate inputs
gqa_q_np = np.random.randn(gqa_batch, gqa_heads, gqa_seq_len, gqa_head_dim).astype(dtype)
gqa_k_np = np.random.randn(gqa_batch, gqa_groups, gqa_seq_len, gqa_head_dim).astype(dtype)
gqa_v_np = np.random.randn(gqa_batch, gqa_groups, gqa_seq_len, gqa_head_dim).astype(dtype)
# Additive causal mask
gqa_mask_np = np.triu(np.ones((gqa_seq_len, gqa_seq_len), dtype=dtype) * np.finfo(dtype).min, k=1)
gqa_mask_np = np.broadcast_to(gqa_mask_np[None, None, :, :], (gqa_batch, 1, gqa_seq_len, gqa_seq_len))

gqa_q_t = torch.tensor(gqa_q_np)
gqa_k_t = torch.tensor(gqa_k_np)
gqa_v_t = torch.tensor(gqa_v_np)
gqa_mask_t = torch.tensor(gqa_mask_np)

# Calculate intermediates
gqa_q_scaled_t = gqa_q_t * gqa_scale
gqa_qk_t = group_shared_matmul(gqa_q_scaled_t, gqa_k_t, transpose_a=False, transpose_b=True)
gqa_qk_masked_t = gqa_qk_t + gqa_mask_t  # Additive mask
gqa_attn_weights_t = torch.softmax(gqa_qk_masked_t, dim=-1)
gqa_attn_qkv_t = group_shared_matmul(gqa_attn_weights_t, gqa_v_t, transpose_a=False, transpose_b=False)

# Save GQA data
np.save(os.path.join(output_dir, "sdpa_interm_gqa_q.npy"), gqa_q_np)
np.save(os.path.join(output_dir, "sdpa_interm_gqa_k.npy"), gqa_k_np)
np.save(os.path.join(output_dir, "sdpa_interm_gqa_v.npy"), gqa_v_np)
np.save(os.path.join(output_dir, "sdpa_interm_gqa_mask.npy"), gqa_mask_np)
np.save(os.path.join(output_dir, "sdpa_interm_gqa_q_scaled.npy"), gqa_q_scaled_t.detach().cpu().numpy())
np.save(os.path.join(output_dir, "sdpa_interm_gqa_qk_masked.npy"), gqa_qk_masked_t.detach().cpu().numpy())
np.save(os.path.join(output_dir, "sdpa_interm_gqa_attn_weights.npy"), gqa_attn_weights_t.detach().cpu().numpy())
np.save(os.path.join(output_dir, "sdpa_interm_gqa_attn_qkv.npy"), gqa_attn_qkv_t.detach().cpu().numpy())
print("Saved GQA intermediate data.")

print("Finished generating SDPA intermediate test data.")


input_text = "The capital of Canada is "
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
input_ids = tokenizer.encode(input_text, return_tensors="pt")
target_length = 32
current_length = input_ids.shape[1]
pad_len = target_length - current_length
if tokenizer.pad_token_id is None:
    # Common practice for models without a specific pad token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Tokenizer does not have a pad token, using EOS token ID: {tokenizer.pad_token_id}")

if pad_len > 0:
    print(f"Padding input_ids from {current_length} to {target_length}")
    # Assuming padding on the right, which is standard for decoder models
    padding = torch.full(
        (input_ids.shape[0], pad_len), tokenizer.pad_token_id, dtype=input_ids.dtype, device=input_ids.device
    )
    input_ids = torch.cat([input_ids, padding], dim=1)
elif pad_len < 0:
    print(f"Warning: input_ids length ({current_length}) is greater than target length ({target_length}). Truncating.")
    input_ids = input_ids[:, :target_length]

print("saving first block input embs")
input_embs = emb(input_ids).detach().cpu().numpy().astype(np.float32)
np.save(os.path.join(output_dir, "first_block_input_embs.npy"), input_embs)

print(f"Shape of input_ids after padding/truncation: {input_ids.shape}")

target_module = hf_model.get_submodule("model.layers.0")
target_func = target_module.forward


def run_fwd():
    # Run the model forward pass. Ensure inputs are on the correct device if using GPU.
    # Assuming hf_model is already on the desired device (e.g., CPU for this example)
    with torch.no_grad():  # Disable gradient calculation for inference
        _ = hf_model(input_ids=input_ids)


print("Running inference to capture first block state...")
captured_data = run_and_capture(target_func, run_fwd)
print("Capture complete.")

first_block_output = captured_data["returns"][0]["locals"]["hidden_states"]
np.save(
    os.path.join(output_dir, "expected_first_block_output.npy"),
    first_block_output.detach().cpu().numpy().astype(np.float32),
)
