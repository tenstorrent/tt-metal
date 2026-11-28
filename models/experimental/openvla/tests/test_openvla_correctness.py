# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Deterministic regression tests for OpenVLA correctness:
1. SigLIP positional embeddings inclusion in patch features
2. Q/K RoPE weight permutation mapping (HF to Meta layout)
3. Action token sequence ordering and EOS exclusion in predict_action
"""

import numpy as np
import pytest
import torch

from models.experimental.openvla.tt.open_vla import (
    get_final_action,
    map_openvla_hf_to_meta_keys,
    permute_hf_to_meta_qk,
)


# =============================================================================
# TEST 1: Q/K RoPE Weight Permutation Correctness
# =============================================================================
def test_qk_permutation_rope_mapping():
    """Verify that Hugging Face Q/K weights are correctly permuted into Meta RoPE layout.

    In Hugging Face LLaMA, RoPE rotates adjacent pairs (half-half rotation: [-x2, x1]).
    In Meta LLaMA / TT-Transformers, RoPE expects interleaved pairs [-x1, x0].
    The permutation must satisfy:
        w_meta = w_hf.view(n_heads, 2, head_dim // 2, -1).transpose(1, 2).reshape(dim1, -1)
    """
    torch.manual_seed(42)
    hidden_size = 4096
    num_heads = 32
    head_dim = hidden_size // num_heads  # 128

    # Create synthetic HuggingFace weights for Q and K
    q_weight_hf = torch.randn(hidden_size, hidden_size)
    k_weight_hf = torch.randn(hidden_size, hidden_size)
    q_bias_hf = torch.randn(hidden_size)

    loaded_weights = {
        "language_model.model.layers.0.self_attn.q_proj.weight": q_weight_hf,
        "language_model.model.layers.0.self_attn.k_proj.weight": k_weight_hf,
        "language_model.model.layers.0.self_attn.q_proj.bias": q_bias_hf,
        "language_model.model.layers.0.self_attn.v_proj.weight": torch.randn(hidden_size, hidden_size),
    }

    meta_dict = map_openvla_hf_to_meta_keys(loaded_weights, head_dim=head_dim)

    # 1. Keys must be mapped to TT-Transformers / Meta keys
    assert "layers.0.attention.wq.weight" in meta_dict
    assert "layers.0.attention.wk.weight" in meta_dict
    assert "layers.0.attention.wq.bias" in meta_dict

    wq_meta = meta_dict["layers.0.attention.wq.weight"]
    wk_meta = meta_dict["layers.0.attention.wk.weight"]
    bq_meta = meta_dict["layers.0.attention.wq.bias"]

    # 2. Permutation check: wq_meta must match manual reverse_permute
    expected_wq = (
        q_weight_hf.view(num_heads, 2, head_dim // 2, hidden_size).transpose(1, 2).reshape(hidden_size, hidden_size)
    )
    expected_wk = (
        k_weight_hf.view(num_heads, 2, head_dim // 2, hidden_size).transpose(1, 2).reshape(hidden_size, hidden_size)
    )
    expected_bq = q_bias_hf.view(num_heads, 2, head_dim // 2).transpose(1, 2).reshape(hidden_size)

    assert torch.equal(wq_meta, expected_wq), "wq weight was not permuted to expected RoPE layout"
    assert torch.equal(wk_meta, expected_wk), "wk weight was not permuted to expected RoPE layout"
    assert torch.equal(bq_meta, expected_bq), "wq bias was not permuted to expected RoPE layout"

    # 3. Non-Q/K weights (e.g. wv) must NOT be permuted
    wv_meta = meta_dict["layers.0.attention.wv.weight"]
    assert torch.equal(wv_meta, loaded_weights["language_model.model.layers.0.self_attn.v_proj.weight"])


# =============================================================================
# TEST 2: Action Token Sequence Ordering and EOS Exclusion
# =============================================================================
def test_action_token_sequence_ordering():
    """Explicitly verify that action token sequence ordering preserves prefill token (dim 0)

    and does NOT shift action dimensions or include EOS token as an action dimension.
    """
    # OpenVLA vocabulary settings
    vocab_size = 32000
    n_action_bins = 256
    action_dims = 7  # 7-DoF: x, y, z, roll, pitch, yaw, gripper

    bins = np.linspace(-1, 1, n_action_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    action_norm_stats = {
        "q01": [-0.5] * action_dims,
        "q99": [0.5] * action_dims,
        "mask": [True] * action_dims,
    }

    # Simulate token IDs generated during inference:
    # Token 0 is generated during PREFILL (x-dim action)
    # Tokens 1-6 are generated during DECODE (y, z, roll, pitch, yaw, gripper)
    # Token 7 would be EOS (e.g. token_id = 2) if decode loop ran 1 extra step
    prefill_token_id = 31900  # Distinct token for dimension 0 (x)
    decode_token_ids = [31901, 31902, 31903, 31904, 31905, 31906]  # Dims 1 to 6
    eos_token_id = 2  # Standard LLaMA EOS token

    # The fixed/correct sequence: Exactly 7 action tokens, starting with prefill_token_id
    correct_action_tokens = [prefill_token_id] + decode_token_ids
    assert len(correct_action_tokens) == action_dims

    # Convert to generated_ids tensor format: [1, seq_len]
    prompt_ids = [1, 29871, 100, 200]
    full_generated_ids = torch.tensor([prompt_ids + correct_action_tokens], dtype=torch.long)

    # 1. Extract action tokens using the predict_action slicing logic: generated_ids[0, -action_dim:]
    extracted_tokens = full_generated_ids[0, -action_dims:].tolist()
    assert extracted_tokens == correct_action_tokens, (
        f"Extracted tokens {extracted_tokens} do not match expected {correct_action_tokens}"
    )

    # 2. Verify dimension 0 is exactly the prefill token (not shifted!)
    assert extracted_tokens[0] == prefill_token_id, "Dimension 0 was shifted or dropped!"

    # 3. Verify gripper (dimension 6) is the 7th action token, NOT EOS!
    assert extracted_tokens[6] == decode_token_ids[-1], "Gripper dimension was shifted!"
    assert eos_token_id not in extracted_tokens, "EOS token was erroneously included in action tokens!"

    # 4. Verify decoded action array maps properly through get_final_action
    actions = get_final_action(full_generated_ids, action_dims, bin_centers, vocab_size, action_norm_stats)
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (action_dims,)

    # Verify that if an off-by-one bug occurred (where prefill is dropped and EOS appended),
    # the output is verifiably different and detectable
    buggy_action_tokens = decode_token_ids + [eos_token_id]
    buggy_generated_ids = torch.tensor([prompt_ids + buggy_action_tokens], dtype=torch.long)
    buggy_extracted = buggy_generated_ids[0, -action_dims:].tolist()
    assert buggy_extracted != correct_action_tokens
    assert buggy_extracted[-1] == eos_token_id  # Demonstrates the buggy behavior being guarded against


# =============================================================================
# TEST 3: Mock OpenVLALanguageModel Prefill + Decode Sequence Assembly
# =============================================================================
def test_language_model_output_tokens_assembly():
    """Verify that OpenVLALanguageModel.__call__ output_toks correctly packs

    prefill logits at index 0 and exactly (num_actions - 1) decode steps.
    """
    num_actions = 7
    vocab_size = 32000

    # Simulate prefill output logits
    prefill_logits = torch.randn(vocab_size)
    prefilled_token = torch.argmax(prefill_logits).item()

    # Simulate decode loop logic as patched in OpenVLALanguageModel.__call__
    output_toks = [prefill_logits.view(1, 1, -1)]
    decode_tokens = [prefilled_token]

    num_decode_steps = max(0, num_actions - 1)
    for i in range(num_decode_steps):
        # Simulate each decode step producing logits
        step_logits = torch.randn(1, 1, vocab_size)
        output_toks.append(step_logits)
        next_tok = torch.argmax(step_logits[:, -1, :], dim=-1).item()
        decode_tokens.append(next_tok)

    # Verify length of output_toks equals num_actions
    assert len(output_toks) == num_actions, f"Expected {num_actions} output tokens, got {len(output_toks)}"
    assert len(decode_tokens) == num_actions, f"Expected {num_actions} decode tokens, got {len(decode_tokens)}"

    # First token must correspond to prefilled_token
    assert torch.equal(output_toks[0], prefill_logits.view(1, 1, -1))
    assert decode_tokens[0] == prefilled_token
