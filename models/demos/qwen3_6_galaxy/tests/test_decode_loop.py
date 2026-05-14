# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""T10: Decode-loop tests for TtQwen36Transformer on BH GLX 8×4 mesh.

Tests the decode-after-prefill path, which was broken by a TTNN quirk:
fast_reduce_nc computes its output shape from padded_shape rather than
logical_shape, causing T=1 → T=32 for the MLP output on each decode step.

Fix: in llama_decoder.py, slice mlp_out back to T_in after TtQwen36MLP.forward().

Hardware required: 32-chip BH GLX 8×4 mesh.

Run all tests:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    mkdir -p /tmp/qwen36_logs
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_decode_loop.py -x -s -v 2>&1 | tee /tmp/qwen36_logs/t10_decode_loop.log
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import pytest
import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SNAPSHOT_DIR = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)


# ---------------------------------------------------------------------------
# Fabric mesh fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_8x4():
    """Open the full 8×4 fabric mesh with FABRIC_1D_RING topology."""
    import ttnn

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# Weight loading helpers (shared with test_full_model.py)
# ---------------------------------------------------------------------------


def _load_layer_weights(layer_idx: int) -> dict:
    """Load all weights for a single decoder layer from safetensors."""
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    pfx = f"model.language_model.layers.{layer_idx}"
    keys_needed = [k for k in weight_map if k.startswith(pfx + ".")]

    files_needed = sorted({weight_map[k] for k in keys_needed})
    raw = {}
    for fn in files_needed:
        shard = load_st(str(_SNAPSHOT_DIR / fn))
        for k in keys_needed:
            if k in shard:
                raw[k] = shard[k].float()

    result = {}
    for k, v in raw.items():
        short = k[len(pfx) + 1 :]
        result[short] = v
    return result


def _load_embedding_and_norm_and_head_weights() -> dict:
    """Load embedding, final norm, and lm_head weights."""
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    needed = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]

    files_needed = sorted({weight_map[k] for k in needed if k in weight_map})
    raw = {}
    for fn in files_needed:
        shard = load_st(str(_SNAPSHOT_DIR / fn))
        for k in needed:
            if k in shard:
                raw[k] = shard[k].float()

    return {
        "tok_embeddings.weight": raw["model.language_model.embed_tokens.weight"],
        "norm.weight": raw["model.language_model.norm.weight"],
        "output.weight": raw["lm_head.weight"],
    }


def _load_config():
    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config

    with open(_SNAPSHOT_DIR / "config.json") as f:
        d = json.load(f)
    return Qwen36Config(d)


def _build_tt_model(mesh_device, args, num_layers: int, global_weights: dict, layers_weights: list):
    """Build TtQwen36Transformer with real weights."""
    from models.demos.qwen3_6_galaxy.tt.llama_model import TtQwen36Transformer

    return TtQwen36Transformer(
        mesh_device=mesh_device,
        args=args,
        global_weights=global_weights,
        layers_weights=layers_weights,
        num_layers=num_layers,
        dtype=None,
    )


# ---------------------------------------------------------------------------
# Test 1: test_decode_one_token_after_prefill
#
# RED before fix: DeltaNet/MLP fast_reduce_nc sets T=32 instead of T=1.
#   Either:
#     - fast_reduce_nc MLP output has T=32 → residual add shape mismatch
#     - DeltaNet assertion "Decode expects T=1, got T=32" fires
# GREEN after fix: mlp_out sliced to T=1 in llama_decoder.py
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_decode_one_token_after_prefill(mesh_8x4):
    """Generate 1 token after prefill via forward_decode path.

    Uses a 4-layer model (cheap, exercises both DeltaNet and full-attention
    layers via the hybrid pattern: layers 0,1,2 are linear_attention, layer 3
    is full_attention in the Qwen3.6-27B pattern).

    Prefills with prompt "The capital of France is" (5 tokens, padded to T=32).
    Calls forward_decode with input_ids=[[next_token_id]] (T=1).

    Asserts:
    - No exception
    - Returned logits shape is [1, 1, vocab_size]
    - No NaN or Inf in logits
    """
    from transformers import AutoTokenizer

    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    # 4-layer model: exercises layers 0 (linear), 1 (linear), 2 (linear), 3 (full).
    # This covers the DeltaNet decode path AND the GatedAttention decode path.
    N_LAYERS = 4
    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)

    print(f"\n[T10-Test1] Loading weights for {N_LAYERS} layers...")
    global_weights = _load_embedding_and_norm_and_head_weights()
    layers_weights = [_load_layer_weights(i) for i in range(N_LAYERS)]

    tokenizer = AutoTokenizer.from_pretrained(str(_SNAPSHOT_DIR), trust_remote_code=True)
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # [1, T_prompt]
    T_prompt = input_ids.shape[-1]
    print(f"[T10-Test1] Prompt: '{prompt}' → {T_prompt} tokens: {input_ids.tolist()}")

    # Pad to multiple of 32 for prefill tile alignment
    T_padded = ((T_prompt + 31) // 32) * 32
    if T_padded > T_prompt:
        pad = torch.zeros(1, T_padded - T_prompt, dtype=input_ids.dtype)
        input_ids_padded = torch.cat([input_ids, pad], dim=1)
    else:
        input_ids_padded = input_ids

    print("[T10-Test1] Building TTNN 4-layer model...")
    tt_model = _build_tt_model(mesh_8x4, args, N_LAYERS, global_weights, layers_weights)

    print("[T10-Test1] Running prefill...")
    t0 = time.time()
    tt_logits, kv_caches, dn_states, conv_states = tt_model.forward_prefill(input_ids_padded, return_caches=True)
    t1 = time.time()
    print(f"[T10-Test1] Prefill done in {t1-t0:.2f}s. Logits shape: {tt_logits.shape}")

    # Get first new token (last real position)
    last_logits = tt_logits[0, T_prompt - 1, : config.vocab_size]
    next_token_id = last_logits.argmax().item()
    decoded_tok = tokenizer.decode([next_token_id])
    print(f"[T10-Test1] Prefill → first new token id={next_token_id}, text='{decoded_tok}'")

    # Decode step: call forward_decode with T=1
    next_id_tensor = torch.tensor([[next_token_id]], dtype=torch.long)  # [1, 1]
    current_pos = T_padded  # next position after padded prefill

    print(
        f"[T10-Test1] Running forward_decode at position {current_pos} with input_ids shape {next_id_tensor.shape}..."
    )
    t0 = time.time()
    step_logits, kv_caches, dn_states, conv_states = tt_model.forward_decode(
        next_id_tensor,
        current_pos=current_pos,
        kv_caches=kv_caches,
        dn_states=dn_states,
        conv_states=conv_states,
    )
    t1 = time.time()
    print(f"[T10-Test1] forward_decode done in {t1-t0:.2f}s. Logits shape: {step_logits.shape}")

    # Assertions
    assert step_logits.shape == (
        1,
        1,
        args.padded_vocab_size,
    ), f"Expected logits shape (1, 1, {args.padded_vocab_size}), got {step_logits.shape}"
    step_last = step_logits[0, 0, : config.vocab_size]
    assert not torch.isnan(step_last).any(), "NaN in decode step 1 logits"
    assert not torch.isinf(step_last).any(), "Inf in decode step 1 logits"

    next_token_2 = step_last.argmax().item()
    decoded_2 = tokenizer.decode([next_token_2])
    print(f"[T10-Test1] Decode step 1 → token id={next_token_2}, text='{decoded_2}'")
    print("[T10-Test1] PASSED — forward_decode completed without error")


# ---------------------------------------------------------------------------
# Test 2: test_greedy_decode_5_tokens_after_prefill
#
# Greedily generate 5 tokens after "The capital of France is" prefill.
# Prints the resulting 10-token string (prompt + 5 new tokens).
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_greedy_decode_5_tokens_after_prefill(mesh_8x4):
    """Greedy decode 5 tokens after 'The capital of France is' prefill.

    Uses the full 64-layer model (same as the Paris generation test) to exercise
    the decode-loop end-to-end.  Each decode step calls forward_decode with T=1
    input and checks that no NaN/Inf appears.

    Prints the 5-10 token continuation and validates the first continuation
    token is ' Paris' (or contains 'paris', case-insensitive).
    """
    from transformers import AutoTokenizer

    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    N_LAYERS = 64
    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)

    print(f"\n[T10-Test2] Loading weights for {N_LAYERS} layers (FULL MODEL + decode)...")
    global_weights = _load_embedding_and_norm_and_head_weights()
    layers_weights = [_load_layer_weights(i) for i in range(N_LAYERS)]

    tokenizer = AutoTokenizer.from_pretrained(str(_SNAPSHOT_DIR), trust_remote_code=True)
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    T_prompt = input_ids.shape[-1]
    print(f"[T10-Test2] Prompt: '{prompt}' → {T_prompt} tokens")

    T_padded = ((T_prompt + 31) // 32) * 32
    if T_padded > T_prompt:
        pad = torch.zeros(1, T_padded - T_prompt, dtype=input_ids.dtype)
        input_ids_padded = torch.cat([input_ids, pad], dim=1)
    else:
        input_ids_padded = input_ids

    print("[T10-Test2] Building TTNN full 64-layer model...")
    tt_model = _build_tt_model(mesh_8x4, args, N_LAYERS, global_weights, layers_weights)

    print("[T10-Test2] Running prefill...")
    t0 = time.time()
    tt_logits, kv_caches, dn_states, conv_states = tt_model.forward_prefill(input_ids_padded, return_caches=True)
    t1 = time.time()
    print(f"[T10-Test2] Prefill done in {t1-t0:.2f}s")

    # First token from prefill
    last_logits = tt_logits[0, T_prompt - 1, : config.vocab_size]
    generated = [last_logits.argmax().item()]
    first_tok_text = tokenizer.decode([generated[-1]])
    print(f"[T10-Test2] Token 1 (from prefill): id={generated[-1]}, text='{first_tok_text}'")

    # Validate first token
    assert "paris" in first_tok_text.lower(), (
        f"Expected first generated token to contain 'paris', got '{first_tok_text}' " f"(token_id={generated[-1]})"
    )

    # Decode loop: 4 more tokens (total 5 new tokens)
    current_pos = T_padded
    for step in range(1, 5):
        next_id = torch.tensor([[generated[-1]]])
        t0 = time.time()
        step_logits, kv_caches, dn_states, conv_states = tt_model.forward_decode(
            next_id,
            current_pos=current_pos,
            kv_caches=kv_caches,
            dn_states=dn_states,
            conv_states=conv_states,
        )
        t1 = time.time()

        # step_logits: [B, 1, vocab_size]
        step_last = step_logits[0, 0, : config.vocab_size]

        # Validate: no NaN/Inf
        assert not torch.isnan(step_last).any(), f"NaN in decode step {step+1} logits"
        assert not torch.isinf(step_last).any(), f"Inf in decode step {step+1} logits"

        next_tok = step_last.argmax().item()
        generated.append(next_tok)
        current_pos += 1
        tok_text = tokenizer.decode([next_tok])
        print(f"[T10-Test2] Token {step+1} (decode step {step}): id={next_tok}, text='{tok_text}' ({t1-t0:.2f}s)")

    full_text = tokenizer.decode(generated)
    print(f"\n[T10-Test2] Generated continuation ({len(generated)} tokens): '{full_text}'")
    print(f"[T10-Test2] Full sequence: '{prompt}{full_text}'")
    print("[T10-Test2] PASSED — 5 decode steps completed without NaN/Inf")


# ---------------------------------------------------------------------------
# Test 3: greedy decode 10 tokens (e2e sanity after T14b.9)
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_greedy_decode_10_tokens_after_prefill(mesh_8x4):
    """Full 64-layer model, greedy decode 10 tokens after the Paris prompt.

    End-to-end eager sanity check after the T14b.9 work. First decode token
    must be ' Paris' (token 11751). Subsequent tokens must produce no
    NaN/Inf. Prints each token + its decoded text + step latency.
    """
    from transformers import AutoTokenizer

    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    N_LAYERS = 64
    N_DECODE_STEPS = 10
    config = _load_config()
    args = TtQwen36ModelArgs(mesh_8x4)

    print(f"\n[T10-decode-10] Loading weights for {N_LAYERS} layers ...")
    global_weights = _load_embedding_and_norm_and_head_weights()
    layers_weights = [_load_layer_weights(i) for i in range(N_LAYERS)]

    tokenizer = AutoTokenizer.from_pretrained(str(_SNAPSHOT_DIR), trust_remote_code=True)
    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    T_prompt = input_ids.shape[-1]
    print(f"[T10-decode-10] Prompt: '{prompt}' -> {T_prompt} tokens")

    T_padded = ((T_prompt + 31) // 32) * 32
    if T_padded > T_prompt:
        pad = torch.zeros(1, T_padded - T_prompt, dtype=input_ids.dtype)
        input_ids_padded = torch.cat([input_ids, pad], dim=1)
    else:
        input_ids_padded = input_ids

    print("[T10-decode-10] Building TTNN full 64-layer model ...")
    tt_model = _build_tt_model(mesh_8x4, args, N_LAYERS, global_weights, layers_weights)

    print("[T10-decode-10] Running prefill ...")
    t0 = time.time()
    tt_logits, kv_caches, dn_states, conv_states = tt_model.forward_prefill(input_ids_padded, return_caches=True)
    t1 = time.time()
    print(f"[T10-decode-10] Prefill done in {t1-t0:.2f}s")

    last_logits = tt_logits[0, T_prompt - 1, : config.vocab_size]
    generated = [last_logits.argmax().item()]
    first_tok_text = tokenizer.decode([generated[-1]])
    print(f"[T10-decode-10] Token 1 (from prefill): id={generated[-1]}, text='{first_tok_text}'")
    assert "paris" in first_tok_text.lower(), (
        f"Expected first generated token to contain 'paris', got '{first_tok_text}' " f"(token_id={generated[-1]})"
    )

    current_pos = T_padded
    for step in range(1, N_DECODE_STEPS):
        next_id = torch.tensor([[generated[-1]]])
        t0 = time.time()
        step_logits, kv_caches, dn_states, conv_states = tt_model.forward_decode(
            next_id,
            current_pos=current_pos,
            kv_caches=kv_caches,
            dn_states=dn_states,
            conv_states=conv_states,
        )
        t1 = time.time()

        step_last = step_logits[0, 0, : config.vocab_size]
        assert not torch.isnan(step_last).any(), f"NaN in decode step {step+1} logits"
        assert not torch.isinf(step_last).any(), f"Inf in decode step {step+1} logits"

        next_tok = step_last.argmax().item()
        generated.append(next_tok)
        current_pos += 1
        tok_text = tokenizer.decode([next_tok])
        print(f"[T10-decode-10] Token {step+1} (decode step {step}): id={next_tok}, text='{tok_text}' ({t1-t0:.2f}s)")

    full_text = tokenizer.decode(generated)
    print(f"\n[T10-decode-10] Generated continuation ({len(generated)} tokens): '{full_text}'")
    print(f"[T10-decode-10] Full sequence: '{prompt}{full_text}'")
    print(f"[T10-decode-10] PASSED — {N_DECODE_STEPS} decode steps without NaN/Inf")
