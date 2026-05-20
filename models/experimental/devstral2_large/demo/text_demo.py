# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Devstral-2-123B (Ministral3) TTNN text-generation demo.

Mirrors the structure of the PyTorch reference at
``reference/devstral2_123b_inference.py``::

    prompt → tokenizer.apply_chat_template → prefill → decode loop → tokenizer.decode

but routes all heavy compute through ``TtMinistral3ForCausalLM`` on a TT mesh device
instead of HuggingFace's FP8 path.

Tracing
-------
Both prefill (single bucket per run) and decode are captured as TTNN traces. The
first prefill is a compile pass, the second is the trace capture (which also
produces the real logits and refills KV idempotently for our prompt). Decode
follows the standard pattern: iteration 0 is an untraced compile pass, iteration
1 captures the trace bound to persistent ``(token, current_pos)`` device buffers,
and every subsequent iteration only does a host→device copy + ``execute_trace``.

Usage (pytest, single Quietbox / 1x4 mesh by default)::

    pytest models/experimental/devstral2_large/demo/text_demo.py

Runtime knobs are environment variables (kept on env vars to avoid project-wide
pytest CLI option churn)::

    DEVSTRAL2_PROMPT="Write a Python function to reverse a linked list."
    DEVSTRAL2_MAX_NEW_TOKENS=100
    DEVSTRAL2_NUM_LAYERS=         # unset/empty = full num_hidden_layers
    MESH_DEVICE=N150|N300|N150x4|P150x4|T3K|TG    # default 1x4 (Quietbox)

The Devstral-2-123B Hub checkpoint is gated, so the first run must have
``HF_TOKEN`` set. The shard-by-shard FP8 → bf16 dequant + tiled TTNN weight upload
is cached on disk (see ``tt/weight_loading.py``).
"""

from __future__ import annotations

import os
import time
from typing import Optional

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.devstral2_large.tests._devstral_weights import (
    model_prefill_weight_keys,
    require_hf_weights,
    require_text_config,
)
from models.experimental.devstral2_large.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_large.tt.tt_ministral3_model import (
    TtMinistral3ForCausalLM,
)
from models.experimental.devstral2_large.tt.weight_loading import DEVSTRAL2_LARGE_REPO_ID
from models.tt_transformers.tt.ccl import TT_CCL


def _mesh_device_param():
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
    }.get(os.environ.get("MESH_DEVICE"), (1, 8))


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _input_ids_to_tt(input_ids: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """Upload token indices ``[batch, seq]`` for ``ttnn.embedding`` on device."""
    return ttnn.from_torch(
        input_ids,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _input_ids_host(input_ids: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """Host-side ``ttnn.Tensor`` of token indices, ready to ``copy_host_to_device_tensor``."""
    return ttnn.from_torch(
        input_ids,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _current_pos_to_tt(positions: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """Upload decode position indices ``[batch]`` as int32 on device."""
    return ttnn.from_torch(
        positions.reshape(-1).to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _current_pos_host(positions: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """Host-side ``ttnn.Tensor`` of int32 positions, ready to ``copy_host_to_device_tensor``."""
    return ttnn.from_torch(
        positions.reshape(-1).to(torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _logits_to_torch(tt_logits: ttnn.Tensor, mesh_device, vocab_size: int) -> torch.Tensor:
    """Concat the column-parallel ``lm_head`` outputs back to a full vocab row."""
    out_last = int(tt_logits.shape[-1])
    if out_last == vocab_size:
        out = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0])
    else:
        out = ttnn.to_torch(
            tt_logits,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
        )
    if out.ndim == 4:
        out = out[0, 0]
    return out


def _build_state_dict(num_layers: int, *, want_lm_head: bool):
    """Download embed + ``num_layers`` decoder blocks + final norm (+ optional lm_head)."""
    base_keys = model_prefill_weight_keys(num_layers)
    if not want_lm_head:
        return require_hf_weights(base_keys)
    try:
        return require_hf_weights(base_keys + ["lm_head.weight"])
    except Exception:
        logger.warning("lm_head.weight not found on the Hub; falling back to tied embeddings.")
        return require_hf_weights(base_keys)


def _generate(
    mesh_device,
    *,
    prompt: str,
    max_new_tokens: int,
    num_layers_override: Optional[int],
) -> str:
    logger.info(f"Loading tokenizer for {DEVSTRAL2_LARGE_REPO_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(DEVSTRAL2_LARGE_REPO_ID, trust_remote_code=True)

    text_cfg = require_text_config()
    num_layers = num_layers_override or int(text_cfg.num_hidden_layers)
    logger.info(f"Building TT model with {num_layers} / {text_cfg.num_hidden_layers} decoder layers.")

    # Tokenize first so we can size the KV cache to (prompt + max_new_tokens).
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"][0].to(torch.long)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(torch.long)
    prompt_len = int(input_ids.shape[0])

    # Tile-align the prefill length: the device prefill kernel expects multiples of 32.
    padded_prompt_len = max(_round_up(prompt_len, 32), 32)
    max_seq_len = max(_round_up(padded_prompt_len + max_new_tokens, 32), 512)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    input_ids_padded = torch.full((padded_prompt_len,), int(pad_id), dtype=torch.long)
    input_ids_padded[:prompt_len] = input_ids
    input_ids_padded = input_ids_padded.unsqueeze(0)  # (1, padded_prompt_len)

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )
    state_dict = _build_state_dict(num_layers, want_lm_head=not args.tie_word_embeddings)

    tt_ccl = TT_CCL(mesh_device)
    t_build = time.time()
    model = TtMinistral3ForCausalLM(args, mesh_device, state_dict, tt_ccl, num_layers=num_layers)
    logger.info(f"TT model built in {time.time() - t_build:.1f}s")

    eos_token_id = tokenizer.eos_token_id
    logger.info(f"Prompt tokens: {prompt_len} (padded to {padded_prompt_len}); max_new_tokens={max_new_tokens}")

    # Persistent device buffers: trace capture binds to these, and replays read from them.
    prefill_tokens_dev = _input_ids_to_tt(input_ids_padded, mesh_device)
    decode_tok_host_init = torch.zeros((1, 1), dtype=torch.long)
    decode_tok_dev = _input_ids_to_tt(decode_tok_host_init, mesh_device)
    decode_pos_dev = _current_pos_to_tt(torch.tensor([prompt_len], dtype=torch.long), mesh_device)

    prefill_trace_id = None
    decode_trace_id = None
    prefill_trace_logits = None
    decode_trace_logits = None
    try:
        # ── Prefill: 1) compile warmup, 2) capture trace ────────────────────
        t_prefill = time.time()
        warm_logits = model(prefill_tokens_dev, mode="prefill", start_pos=0)
        ttnn.synchronize_device(mesh_device)
        compile_prefill_time = time.time() - t_prefill
        warm_logits.deallocate(True)
        logger.info(f"Prefill compile pass: {compile_prefill_time*1000:.0f}ms")

        t_capture = time.time()
        prefill_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        prefill_trace_logits = model(prefill_tokens_dev, mode="prefill", start_pos=0)
        ttnn.end_trace_capture(mesh_device, prefill_trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"Prefill trace captured in {(time.time()-t_capture)*1000:.0f}ms")

        # Sample first new token from the trace's output buffer (capture pass produced valid logits).
        logits_torch = _logits_to_torch(prefill_trace_logits, mesh_device, args.vocab_size)
        next_token = int(logits_torch[prompt_len - 1].argmax().item())
        logger.info(f"First generated token {next_token} = {tokenizer.decode([next_token])!r}")

        # ── Decode: 1) untraced compile, 2) capture trace at iter 1, 3) replay ───
        generated = [next_token]
        current_pos = prompt_len
        compile_decode_time = 0.0
        capture_decode_time = 0.0
        t_decode_start = time.time()
        for iteration in range(max_new_tokens - 1):
            if eos_token_id is not None and next_token == eos_token_id:
                logger.info("EOS reached; stopping.")
                break

            if iteration == 0:
                # Untraced compile pass.
                t_compile = time.time()
                ttnn.copy_host_to_device_tensor(
                    _input_ids_host(torch.tensor([[next_token]], dtype=torch.long), mesh_device),
                    decode_tok_dev,
                )
                ttnn.copy_host_to_device_tensor(
                    _current_pos_host(torch.tensor([current_pos], dtype=torch.long), mesh_device),
                    decode_pos_dev,
                )
                tt_out = model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
                logits_torch = _logits_to_torch(tt_out, mesh_device, args.vocab_size)
                tt_out.deallocate(True)
                next_token = int(logits_torch[0].argmax().item())
                generated.append(next_token)
                current_pos += 1
                compile_decode_time = time.time() - t_compile
                logger.info(f"Decode compile pass: {compile_decode_time*1000:.0f}ms")

                # Capture trace bound to the (decode_tok_dev, decode_pos_dev) buffers.
                t_capture = time.time()
                ttnn.copy_host_to_device_tensor(
                    _input_ids_host(torch.tensor([[next_token]], dtype=torch.long), mesh_device),
                    decode_tok_dev,
                )
                ttnn.copy_host_to_device_tensor(
                    _current_pos_host(torch.tensor([current_pos], dtype=torch.long), mesh_device),
                    decode_pos_dev,
                )
                decode_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                decode_trace_logits = model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
                ttnn.end_trace_capture(mesh_device, decode_trace_id, cq_id=0)
                ttnn.synchronize_device(mesh_device)
                capture_decode_time = time.time() - t_capture
                logger.info(f"Decode trace captured in {capture_decode_time*1000:.0f}ms")

                # The capture pass already produced logits for `current_pos`; use them.
                logits_torch = _logits_to_torch(decode_trace_logits, mesh_device, args.vocab_size)
                next_token = int(logits_torch[0].argmax().item())
                generated.append(next_token)
                current_pos += 1
            else:
                # Replay path: update the persistent input buffers and re-fire the trace.
                ttnn.copy_host_to_device_tensor(
                    _input_ids_host(torch.tensor([[next_token]], dtype=torch.long), mesh_device),
                    decode_tok_dev,
                )
                ttnn.copy_host_to_device_tensor(
                    _current_pos_host(torch.tensor([current_pos], dtype=torch.long), mesh_device),
                    decode_pos_dev,
                )
                ttnn.execute_trace(mesh_device, decode_trace_id, cq_id=0, blocking=False)
                logits_torch = _logits_to_torch(decode_trace_logits, mesh_device, args.vocab_size)
                next_token = int(logits_torch[0].argmax().item())
                generated.append(next_token)
                current_pos += 1
        decode_time = time.time() - t_decode_start
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
    finally:
        if prefill_trace_id is not None:
            ttnn.release_trace(mesh_device, prefill_trace_id)
        if decode_trace_id is not None:
            ttnn.release_trace(mesh_device, decode_trace_id)

    logger.info("=== Prompt ===")
    logger.info(prompt)
    logger.info("=== Response ===")
    logger.info(decoded)

    num_decoded = len(generated)
    if num_decoded > 1 and decode_time > 0:
        logger.info(
            f"Decode: {num_decoded - 1} steps in {decode_time*1000:.0f}ms "
            f"(~{(num_decoded - 1)/decode_time:.2f} tok/s/user)"
        )
    return decoded


@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 100_000_000,
            "num_command_queues": 1,
            "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
def test_devstral2_large_text_demo(mesh_device):
    prompt = os.environ.get("DEVSTRAL2_PROMPT") or "Write a Python function to reverse a linked list."
    max_new_tokens = int(os.environ.get("DEVSTRAL2_MAX_NEW_TOKENS") or "100")
    raw_layers = os.environ.get("DEVSTRAL2_NUM_LAYERS", "")
    num_layers_override = int(raw_layers) if raw_layers else None

    response = _generate(
        mesh_device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        num_layers_override=num_layers_override,
    )
    assert response, "Empty response from TTNN Devstral-2 demo"
