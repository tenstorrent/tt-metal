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
Prefill uses chunked paged attention (``kv_block_size`` tokens per dispatch, same
as ``tt_demo_agent.py``). With ``DEVSTRAL2_TRACE_PREFILL=1`` (default), one
full-model prefill trace is captured and replayed for every chunk:

- ``chunk_start_idx_tensor`` (device int32) drives flexible chunked SDPA inside the trace.
- RoPE cos/sin and the per-chunk page-table slice are copied into persistent device
  buffers before each ``execute_trace`` (same pattern as llama3 galaxy prefix-caching traces).

Without prefill tracing, each chunk is dispatched with legacy scalar ``start_pos``.

Decode follows the standard pattern: iteration 0 is an untraced compile pass,
iteration 1 captures the trace bound to persistent ``(token, current_pos)``
device buffers, and later iterations only do host→device copy + ``execute_trace``.

Set ``DEVSTRAL2_TRACE_PREFILL=0`` to run chunked prefill untraced.

Usage (pytest, single Loudbox / 1x8 mesh by default)::

    pytest models/experimental/devstral2_large/demo/text_demo.py

Runtime knobs are environment variables (kept on env vars to avoid project-wide
pytest CLI option churn)::

    DEVSTRAL2_PROMPT="Write a Python function to reverse a linked list."
    DEVSTRAL2_MAX_NEW_TOKENS=100
    DEVSTRAL2_NUM_LAYERS=         # unset/empty = full num_hidden_layers
    DEVSTRAL2_TRACE_PREFILL=1     # 0 = never capture prefill trace
    DEVSTRAL2_MIN_MAX_SEQ_LEN=1024  # KV floor (1024 => 8 blocks, enables flexible prefill trace)
    MESH_DEVICE=N150|N300|N150x4|P150x4|T3K|TG    # default 1x4 (Quietbox)

The Devstral-2-123B Hub checkpoint is gated, so the first run must have
``HF_TOKEN`` set. The shard-by-shard FP8 → bf16 dequant + tiled TTNN weight upload
is cached on disk (see ``tt/weight_loading.py``).
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence, Tuple

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


def _min_max_seq_len() -> int:
    """Minimum KV/RoPE budget (multiple of ``kv_block_size``).

    Default 1024 (8 logical KV blocks == padded page-table cols) so flexible chunked
    SDPA + multi-chunk prefill trace are safe. Use 512 only if you accept int ``start_pos``
    prefill (trace disabled when cols != logical blocks).
    """
    raw = os.environ.get("DEVSTRAL2_MIN_MAX_SEQ_LEN", "1024").strip()
    floor = int(raw)
    block = Devstral2Args.kv_block_size
    if floor % block != 0:
        raise ValueError(f"DEVSTRAL2_MIN_MAX_SEQ_LEN ({floor}) must be a multiple of {block}")
    return floor


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


def _page_table_host(page_table: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        page_table,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _rope_host_tt(t: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """Host staging tensor for ``copy_host_to_device_tensor`` (no ``device=``)."""
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _rope_dev_tt(t: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """Persistent on-device RoPE slice buffer (trace binds to these addresses)."""
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


# Flexible chunked SDPA requires page-table row width (int32 cols) to be a multiple of 8.
_CHUNK_PAGE_TABLE_COLS = 8


def _init_prefill_trace_buffers(
    mesh_device,
    *,
    kv_block_size: int,
    head_dim: int,
) -> Tuple[ttnn.Tensor, ttnn.Tensor, List[ttnn.Tensor]]:
    """Persistent device buffers updated before each prefill trace replay."""
    chunk_start_dev = _current_pos_to_tt(torch.tensor([0], dtype=torch.long), mesh_device)
    chunk_pt_dev = ttnn.from_torch(
        torch.zeros((1, _CHUNK_PAGE_TABLE_COLS), dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rope_shape = (1, 1, kv_block_size, head_dim)
    zeros = torch.zeros(rope_shape, dtype=torch.bfloat16)
    rope_dev_bufs: List[ttnn.Tensor] = [_rope_dev_tt(zeros, mesh_device) for _ in range(4)]
    return chunk_start_dev, chunk_pt_dev, rope_dev_bufs


def _host_prefill_rope_slice(
    rotary_emb,
    chunk_start: int,
    kv_block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice prefill RoPE from host tables (matches ``get_prefill_tables`` on device)."""
    end = chunk_start + kv_block_size
    head_dim = rotary_emb.args.head_dim
    shape = (1, 1, kv_block_size, head_dim)

    def _slice(table: torch.Tensor) -> torch.Tensor:
        return table[chunk_start:end].to(torch.bfloat16).reshape(shape)

    return (
        _slice(rotary_emb._cos_q_host),
        _slice(rotary_emb._sin_q_host),
        _slice(rotary_emb._cos_host),
        _slice(rotary_emb._sin_host),
    )


def _update_prefill_trace_buffers(
    *,
    rotary_emb,
    mesh_device,
    prefill_chunk_dev: ttnn.Tensor,
    chunk_tokens: torch.Tensor,
    chunk_start: int,
    kv_block_size: int,
    chunk_start_dev: ttnn.Tensor,
    chunk_pt_dev: ttnn.Tensor,
    rope_dev_bufs: Sequence[ttnn.Tensor],
) -> ttnn.Tensor:
    """Host→device copies for one chunked-prefill step (tokens, SDPA offset, RoPE, page table)."""
    ttnn.copy_host_to_device_tensor(
        _input_ids_host(chunk_tokens, mesh_device),
        prefill_chunk_dev,
    )
    ttnn.copy_host_to_device_tensor(
        _current_pos_host(torch.tensor([chunk_start], dtype=torch.long), mesh_device),
        chunk_start_dev,
    )
    # Slice on host: device ``get_prefill_tables`` tensors are mesh-replicated and cannot be
    # ``to_torch``'d without a mesh composer on multi-device meshes (e.g. 1x8 Loudbox).
    for host_slice, dev_buf in zip(
        _host_prefill_rope_slice(rotary_emb, chunk_start, kv_block_size),
        rope_dev_bufs,
    ):
        ttnn.copy_host_to_device_tensor(_rope_host_tt(host_slice, mesh_device), dev_buf)
    # Always update chunk_pt_dev (including chunk 0 with block_idx=0) so trace capture binds
    # paged_fill_cache to chunk_pt_dev; trace replays then pick up the updated block index.
    block_idx = chunk_start // kv_block_size
    chunk_pt_host = torch.zeros((1, _CHUNK_PAGE_TABLE_COLS), dtype=torch.int32)
    chunk_pt_host[0, 0] = block_idx
    ttnn.copy_host_to_device_tensor(
        _page_table_host(chunk_pt_host, mesh_device),
        chunk_pt_dev,
    )
    return chunk_pt_dev


def _prefill_flexible_kwargs(
    *,
    chunk_start: int,
    chunk_start_dev: ttnn.Tensor,
    chunk_page_table: Optional[ttnn.Tensor],
    rope_dev_bufs: Sequence[ttnn.Tensor],
) -> dict:
    return dict(
        start_pos=chunk_start,
        chunk_start_idx_tensor=chunk_start_dev,
        chunk_page_table=chunk_page_table,
        prefill_rope_tables=tuple(rope_dev_bufs),
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

    # Chunked paged prefill: pad to whole ``kv_block_size`` blocks (default 128).
    kv_block_size = Devstral2Args.kv_block_size
    num_prefill_chunks = max(1, (prompt_len + kv_block_size - 1) // kv_block_size)
    padded_prompt_len = num_prefill_chunks * kv_block_size
    min_seq_len = _min_max_seq_len()
    max_seq_len = max(_round_up(padded_prompt_len + max_new_tokens, kv_block_size), min_seq_len)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    input_ids_padded = torch.full((padded_prompt_len,), int(pad_id), dtype=torch.long)
    input_ids_padded[:prompt_len] = input_ids
    input_ids_padded = input_ids_padded.unsqueeze(0)  # (1, padded_prompt_len)

    trace_prefill = os.environ.get("DEVSTRAL2_TRACE_PREFILL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )
    # Flexible chunked SDPA + trace replay requires page_table width == logical block count.
    # With max_seq_len=512 we have 4 logical blocks but 8 padded cols (stick alignment); flexible
    # SDPA would attend too far and corrupt KV. Default min is 1024 (8==8). Override via
    # DEVSTRAL2_MIN_MAX_SEQ_LEN=512 to reproduce the old path.
    if trace_prefill and not args.supports_flexible_chunked_sdpa:
        logger.warning(
            "Disabling prefill trace: flexible chunked SDPA needs page_table blocks "
            f"({args.kv_page_table_blocks_per_user}) == logical KV blocks "
            f"({args.kv_num_blocks_per_user}); using int start_pos prefill instead. "
            "Set DEVSTRAL2_TRACE_PREFILL=0 to silence."
        )
        trace_prefill = False
    state_dict = _build_state_dict(num_layers, want_lm_head=not args.tie_word_embeddings)

    tt_ccl = TT_CCL(mesh_device)
    t_build = time.time()
    model = TtMinistral3ForCausalLM(args, mesh_device, state_dict, tt_ccl, num_layers=num_layers)
    logger.info(f"TT model built in {time.time() - t_build:.1f}s")

    eos_token_id = tokenizer.eos_token_id
    logger.info(
        f"Prompt tokens: {prompt_len} (padded to {padded_prompt_len}, "
        f"{num_prefill_chunks} prefill chunk(s) of {kv_block_size}); "
        f"max_seq_len={max_seq_len} (min floor {min_seq_len}); max_new_tokens={max_new_tokens}"
    )
    if trace_prefill:
        logger.info(
            f"Prefill trace: flexible chunked SDPA "
            f"(logical blocks={args.kv_num_blocks_per_user}, "
            f"page_table cols={args.kv_page_table_blocks_per_user})"
        )
    if trace_prefill and num_prefill_chunks > 1:
        logger.info(
            f"Multi-chunk prefill trace: {num_prefill_chunks} chunks via one capture + "
            f"{num_prefill_chunks - 1} execute_trace replay(s)."
        )

    # Persistent chunk buffer for prefill (one KV block); decode buffers for traced replay.
    prefill_chunk_dev = _input_ids_to_tt(
        torch.full((1, kv_block_size), int(pad_id), dtype=torch.long),
        mesh_device,
    )
    decode_tok_host_init = torch.zeros((1, 1), dtype=torch.long)
    decode_tok_dev = _input_ids_to_tt(decode_tok_host_init, mesh_device)
    decode_pos_dev = _current_pos_to_tt(torch.tensor([prompt_len], dtype=torch.long), mesh_device)

    prefill_trace_id = None
    decode_trace_id = None
    prefill_trace_logits = None
    decode_trace_logits = None
    rotary_emb = model.model.rotary_emb
    try:
        # ── Chunked paged prefill (``paged_fill_cache`` + ``chunked_scaled_dot_product_attention``) ──
        t_prefill = time.time()
        next_token: Optional[int] = None

        if trace_prefill:
            chunk_start_dev, chunk_pt_dev, rope_dev_bufs = _init_prefill_trace_buffers(
                mesh_device,
                kv_block_size=kv_block_size,
                head_dim=args.head_dim,
            )

            # Compile: run every chunk once with flexible SDPA + persistent trace buffers.
            for chunk_idx in range(num_prefill_chunks):
                chunk_start = chunk_idx * kv_block_size
                chunk_tokens = input_ids_padded[:, chunk_start : chunk_start + kv_block_size]
                chunk_page_table = _update_prefill_trace_buffers(
                    rotary_emb=rotary_emb,
                    mesh_device=mesh_device,
                    prefill_chunk_dev=prefill_chunk_dev,
                    chunk_tokens=chunk_tokens,
                    chunk_start=chunk_start,
                    kv_block_size=kv_block_size,
                    chunk_start_dev=chunk_start_dev,
                    chunk_pt_dev=chunk_pt_dev,
                    rope_dev_bufs=rope_dev_bufs,
                )
                flex = _prefill_flexible_kwargs(
                    chunk_start=chunk_start,
                    chunk_start_dev=chunk_start_dev,
                    chunk_page_table=chunk_page_table,
                    rope_dev_bufs=rope_dev_bufs,
                )
                warm_logits = model(prefill_chunk_dev, mode="prefill", **flex)
                warm_logits.deallocate(True)
            ttnn.synchronize_device(mesh_device)
            logger.info(
                f"Prefill compile pass: {(time.time() - t_prefill) * 1000:.0f}ms "
                f"({num_prefill_chunks} chunk(s), flexible trace)"
            )

            # Capture on chunk 0; replay chunks 1..N-1 with updated buffers.
            chunk_start = 0
            chunk_tokens = input_ids_padded[:, :kv_block_size]
            chunk_page_table = _update_prefill_trace_buffers(
                rotary_emb=rotary_emb,
                mesh_device=mesh_device,
                prefill_chunk_dev=prefill_chunk_dev,
                chunk_tokens=chunk_tokens,
                chunk_start=0,
                kv_block_size=kv_block_size,
                chunk_start_dev=chunk_start_dev,
                chunk_pt_dev=chunk_pt_dev,
                rope_dev_bufs=rope_dev_bufs,
            )
            flex = _prefill_flexible_kwargs(
                chunk_start=0,
                chunk_start_dev=chunk_start_dev,
                chunk_page_table=chunk_page_table,
                rope_dev_bufs=rope_dev_bufs,
            )
            t_capture = time.time()
            prefill_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            prefill_trace_logits = model(prefill_chunk_dev, mode="prefill", **flex)
            ttnn.end_trace_capture(mesh_device, prefill_trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            logger.info(f"Prefill trace captured in {(time.time() - t_capture) * 1000:.0f}ms")

            for chunk_idx in range(1, num_prefill_chunks):
                chunk_start = chunk_idx * kv_block_size
                chunk_tokens = input_ids_padded[:, chunk_start : chunk_start + kv_block_size]
                chunk_page_table = _update_prefill_trace_buffers(
                    rotary_emb=rotary_emb,
                    mesh_device=mesh_device,
                    prefill_chunk_dev=prefill_chunk_dev,
                    chunk_tokens=chunk_tokens,
                    chunk_start=chunk_start,
                    kv_block_size=kv_block_size,
                    chunk_start_dev=chunk_start_dev,
                    chunk_pt_dev=chunk_pt_dev,
                    rope_dev_bufs=rope_dev_bufs,
                )
                ttnn.synchronize_device(mesh_device)
                ttnn.execute_trace(mesh_device, prefill_trace_id, cq_id=0, blocking=True)
                ttnn.synchronize_device(mesh_device)

            last_chunk_start = (num_prefill_chunks - 1) * kv_block_size
            ttnn.synchronize_device(mesh_device)
            logits_torch = _logits_to_torch(prefill_trace_logits, mesh_device, args.vocab_size)
            local_pos = (prompt_len - 1) - last_chunk_start
            next_token = int(logits_torch[local_pos].argmax().item())
            logger.info(
                f"Chunked prefill trace ({num_prefill_chunks} chunk(s)): " f"{(time.time() - t_prefill) * 1000:.0f}ms"
            )
        else:
            for chunk_idx in range(num_prefill_chunks):
                chunk_start = chunk_idx * kv_block_size
                chunk_tokens = input_ids_padded[:, chunk_start : chunk_start + kv_block_size]
                ttnn.copy_host_to_device_tensor(
                    _input_ids_host(chunk_tokens, mesh_device),
                    prefill_chunk_dev,
                )
                tt_out = model(prefill_chunk_dev, mode="prefill", start_pos=chunk_start)
                if chunk_idx == num_prefill_chunks - 1:
                    prefill_trace_logits = tt_out
                else:
                    tt_out.deallocate(True)

            last_chunk_start = (num_prefill_chunks - 1) * kv_block_size
            ttnn.synchronize_device(mesh_device)
            logits_torch = _logits_to_torch(prefill_trace_logits, mesh_device, args.vocab_size)
            local_pos = (prompt_len - 1) - last_chunk_start
            next_token = int(logits_torch[local_pos].argmax().item())
            prefill_trace_logits.deallocate(True)
            prefill_trace_logits = None
            logger.info(f"Chunked prefill ({num_prefill_chunks} chunk(s)): {(time.time() - t_prefill) * 1000:.0f}ms")

        assert next_token is not None
        logger.info(f"First generated token {next_token} = {tokenizer.decode([next_token])!r}")

        # Release prefill trace before decode so compile/capture does not allocate under an
        # active trace (Metal warns this can corrupt trace-owned buffers).
        if prefill_trace_id is not None:
            ttnn.release_trace(mesh_device, prefill_trace_id)
            prefill_trace_id = None

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
                # Untraced compile pass (kernel cache only; do not sample — avoids an extra
                # decode step and KV write before trace capture).
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
                tt_out.deallocate(True)
                ttnn.synchronize_device(mesh_device)
                compile_decode_time = time.time() - t_compile
                logger.info(f"Decode compile pass: {compile_decode_time*1000:.0f}ms")

                # Capture trace bound to (decode_tok_dev, decode_pos_dev); first replay logits
                # come from this capture pass.
                t_capture = time.time()
                decode_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                decode_trace_logits = model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
                ttnn.end_trace_capture(mesh_device, decode_trace_id, cq_id=0)
                ttnn.synchronize_device(mesh_device)
                capture_decode_time = time.time() - t_capture
                logger.info(f"Decode trace captured in {capture_decode_time*1000:.0f}ms")

                logits_torch = _logits_to_torch(decode_trace_logits, mesh_device, args.vocab_size)
                next_token = int(logits_torch[0].argmax().item())
                generated.append(next_token)
                current_pos += 1
            else:
                # Replay: refresh inputs, execute trace, then sync before reading logits.
                ttnn.copy_host_to_device_tensor(
                    _input_ids_host(torch.tensor([[next_token]], dtype=torch.long), mesh_device),
                    decode_tok_dev,
                )
                ttnn.copy_host_to_device_tensor(
                    _current_pos_host(torch.tensor([current_pos], dtype=torch.long), mesh_device),
                    decode_pos_dev,
                )
                ttnn.execute_trace(mesh_device, decode_trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
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
