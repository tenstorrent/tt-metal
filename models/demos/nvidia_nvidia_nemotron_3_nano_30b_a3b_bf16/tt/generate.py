# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Token generation loop for NemotronH-30B on QB TP=4 (4× Blackhole).

Flow
----
1. Tokenize prompt.
2. Prefill: one eager S=1 step per prompt token to build up SSM state and KV
   cache.  (A batched prefill kernel would be faster but S=1 reuse is simpler
   and correct.)
3. Decode: capture a single-token trace on the pre-warmed state, then replay
   it for each generated token.  Between replay executions:
     a. Copy new SSM states back to state inputs via ttnn.assign.
     b. Write the next token ID and incremented position to the persistent
        device tensors via copy_host_to_device_tensor.

Usage
-----
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.generate import generate

    text = generate(
        "Hello, I am a language model",
        mesh_device=mesh_device,
        wc=weight_cache,
        max_new_tokens=50,
    )
    print(text)
"""

from __future__ import annotations

import time

import torch

import ttnn

from .kv_cache import DEFAULT_BLOCK_SIZE, DEFAULT_MAX_SEQ_LEN, allocate_decoder_state
from .model import WeightCache, nemotron_h_forward_stateful
from .tp import _R, _host_rep

SNAP = (
    "/home/ttuser/.cache/huggingface/hub/"
    "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/"
    "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
)


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(SNAP)


def _sample_token(logits: torch.Tensor, temperature: float = 0.0, top_p: float = 0.9) -> int:
    """Greedy (temperature=0) or nucleus sampling."""
    logits = logits[0, 0].float()  # [vocab_size]
    if temperature == 0.0:
        return int(logits.argmax())
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cumprobs = sorted_probs.cumsum(dim=0)
        cutoff = (cumprobs - sorted_probs > top_p).nonzero()
        cutoff_idx = cutoff[0].item() if len(cutoff) else len(probs)
        mask = torch.zeros_like(probs)
        mask[sorted_idx[:cutoff_idx]] = 1
        probs = probs * mask
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1))


def _to_device_token(token_id: int, mesh_device) -> ttnn.Tensor:
    cpu = torch.tensor([[token_id]], dtype=torch.int32)
    return ttnn.from_torch(
        cpu,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )


def _update_ids(ids_tt: ttnn.Tensor, token_id: int):
    cpu = torch.tensor([[token_id]], dtype=torch.int32)
    ids_host = ttnn.from_torch(cpu, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn.copy_host_to_device_tensor(ids_host, ids_tt)


def _update_pos(current_pos_tt: ttnn.Tensor, pos: int):
    cpu = torch.tensor([pos], dtype=torch.int32)
    pos_host = ttnn.from_torch(cpu, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn.copy_host_to_device_tensor(pos_host, current_pos_tt)


def generate(
    prompt: str,
    mesh_device,
    wc: WeightCache | None = None,
    max_new_tokens: int = 100,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    block_size: int = DEFAULT_BLOCK_SIZE,
    temperature: float = 0.0,
    top_p: float = 0.9,
    verbose: bool = True,
    cpu_gate: bool = False,
) -> str:
    """Generate text continuation for `prompt`.

    Args:
        prompt:         Input text.
        mesh_device:    Open TTNN MeshDevice (TP=4 QB).
        wc:             WeightCache; created from default SNAP path if None.
        max_new_tokens: Maximum new tokens to generate.
        max_seq_len:    Maximum total sequence length (prompt + generated).
        block_size:     Paged KV cache block size (tokens per block).
        temperature:    Sampling temperature; 0 = greedy.
        top_p:          Nucleus sampling threshold.
        verbose:        Print progress.
        cpu_gate:       If False (default), compute MoE gate on device in bfloat16 —
                        trace-compatible; decode loop uses ttnn.execute_trace for
                        ~16 tok/s on TP=4 QB.
                        If True, gate runs on CPU in float32 — exact HF routing
                        but not trace-compatible (each decode step calls forward
                        directly, ~7 tok/s).

    Returns:
        The full generated text (prompt + new tokens).
    """
    if wc is None:
        wc = WeightCache()

    tokenizer = _load_tokenizer()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()

    if verbose:
        print(f"Prompt tokens: {len(input_ids)}")

    state = allocate_decoder_state(mesh_device, B=1, max_seq_len=max_seq_len, block_size=block_size)

    # Pre-allocate persistent device token tensor (updated between trace replays).
    ids_tt = _to_device_token(input_ids[0], mesh_device)

    # --- Prefill: process prompt tokens one at a time (S=1 decode steps) ---
    if verbose:
        print("Prefilling...")
    t_prefill = time.perf_counter()
    for pos, tok in enumerate(input_ids):
        _update_ids(ids_tt, tok)
        _update_pos(state.current_pos, pos)
        logits_tt = nemotron_h_forward_stateful(mesh_device, ids_tt, wc, state, cpu_gate=cpu_gate)
        ttnn.synchronize_device(mesh_device)
        state.advance()
    elapsed_prefill = time.perf_counter() - t_prefill
    if verbose:
        print(f"Prefill done: {len(input_ids)} tokens in {elapsed_prefill:.1f}s")

    # Sample first decode token from prefill logits.
    logits_cpu = _host_rep(logits_tt, mesh_device, 1)
    next_token = _sample_token(logits_cpu, temperature, top_p)
    generated = [next_token]
    if verbose:
        tok_str = tokenizer.decode([next_token])
        print(f"First decoded token: {repr(tok_str)}")

    decode_pos = len(input_ids)

    if cpu_gate:
        # cpu_gate=True: D2H inside forward makes trace incompatible — run eager decode loop.
        if verbose:
            print("Decode mode: eager (cpu_gate=True, no trace)")
        t_decode = time.perf_counter()
        for step in range(max_new_tokens - 1):
            decode_pos += 1
            _update_ids(ids_tt, next_token)
            _update_pos(state.current_pos, decode_pos - 1)
            logits_tt = nemotron_h_forward_stateful(mesh_device, ids_tt, wc, state, cpu_gate=True)
            ttnn.synchronize_device(mesh_device)
            state.advance()

            logits_cpu = _host_rep(logits_tt, mesh_device, 1)
            next_token = _sample_token(logits_cpu, temperature, top_p)
            generated.append(next_token)

            if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
                break
            if decode_pos >= max_seq_len:
                break

        elapsed_decode = time.perf_counter() - t_decode
        n_gen = len(generated)
        if verbose and n_gen > 1:
            print(f"Generated {n_gen} tokens in {elapsed_decode:.2f}s ({n_gen / elapsed_decode:.2f} tok/s)")
    else:
        # cpu_gate=False: on-device gate is trace-compatible — use traced decode.
        #
        # Trace decode contract:
        #   1. Trace capture runs token_0 at decode_pos → produces logits_tt for token_1.
        #   2. Read logits_tt immediately, sample token_1, advance state, increment pos.
        #   3. Loop re-executes the trace with token_1 at (decode_pos+1), token_2 at
        #      (decode_pos+2), etc.  Each execute_trace updates logits_tt in-place.
        #
        # BUG that this fixes: old code advanced state + pos BEFORE reading logits_tt,
        # then started the loop with next_token = token_0 at the incremented pos,
        # processing token_0 a second time and discarding the trace-capture logits.
        remaining = max_new_tokens - 1  # token_0 already in generated
        if remaining > 0:
            _update_ids(ids_tt, next_token)
            _update_pos(state.current_pos, decode_pos)

            t_decode = time.perf_counter()
            if verbose:
                print("Capturing decode trace...")
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            logits_tt = nemotron_h_forward_stateful(mesh_device, ids_tt, wc, state, cpu_gate=False)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            if verbose:
                print("Trace captured.")

            # Consume trace-capture output: this IS decode step 1.
            logits_cpu = _host_rep(logits_tt, mesh_device, 1)
            state.advance()
            decode_pos += 1
            next_token = _sample_token(logits_cpu, temperature, top_p)
            generated.append(next_token)
            remaining -= 1

            for step in range(remaining):
                if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
                    break
                if decode_pos >= max_seq_len:
                    break

                ids_host = ttnn.from_torch(
                    torch.tensor([[next_token]], dtype=torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                pos_host = ttnn.from_torch(
                    torch.tensor([decode_pos], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                ttnn.copy_host_to_device_tensor(ids_host, ids_tt)
                ttnn.copy_host_to_device_tensor(pos_host, state.current_pos)

                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

                state.advance()
                decode_pos += 1

                logits_cpu = _host_rep(logits_tt, mesh_device, 1)
                next_token = _sample_token(logits_cpu, temperature, top_p)
                generated.append(next_token)

            elapsed_decode = time.perf_counter() - t_decode
            n_gen = len(generated)
            if verbose and n_gen > 1:
                print(f"Generated {n_gen} tokens in {elapsed_decode:.2f}s ({n_gen / elapsed_decode:.2f} tok/s)")

            ttnn.release_trace(mesh_device, trace_id)

    full_ids = input_ids + generated
    return tokenizer.decode(full_ids, skip_special_tokens=True)
