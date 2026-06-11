# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
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
as ``tt_demo_agent.py``). **Prefill trace is on by default**: one full-model
prefill trace is captured and replayed for every chunk:

- ``chunk_start_idx_tensor`` (device int32) drives flexible chunked SDPA inside the trace.
- RoPE cos/sin and the per-chunk page-table slice are copied into persistent device
  buffers before each ``execute_trace`` (same pattern as llama3 galaxy prefix-caching traces).

If ``max_seq_len`` is too small for flexible chunked SDPA, prefill trace is disabled
automatically and each chunk uses legacy scalar ``start_pos``.

Decode uses traced replay; **2CQ** (CQ1=input H2D, CQ0=forward/trace) is on by default
(``DEVSTRAL2_DECODE_TRACE_2CQ=1``). Set ``DEVSTRAL2_DECODE_TRACE_2CQ=0`` for single-CQ.

Set ``DEVSTRAL2_TRACE_PREFILL=0`` to run chunked prefill without trace capture.

Throughput logging (end of run) reports **TT device time only** — prefill trace forward and
decode trace replay tok/s, excluding H2D, logits D2H, sampling, and compile/capture warmup.

Usage (pytest, single Loudbox / 1x8 mesh by default)::

    pytest models/experimental/devstral2_123B_instruct/demo/text_demo.py

Runtime knobs are environment variables (kept on env vars to avoid project-wide
pytest CLI option churn)::

    DEVSTRAL2_PROMPT="Write a Python function to reverse a linked list."
    DEVSTRAL2_MESSAGES_JSON="messages_256k_text (1).json"  # repo-root 256k chat messages
    DEVSTRAL2_ROPE_DRAM=1         # force RoPE prefill tables to DRAM (auto at 256K if unset)
    DEVSTRAL2_MAX_NEW_TOKENS=100
    DEVSTRAL2_NUM_LAYERS=         # unset/empty = full num_hidden_layers
    DEVSTRAL2_TRACE_PREFILL=1     # 0 = never capture prefill trace (default: on)
    DEVSTRAL2_DECODE_TRACE_2CQ=1  # 0 = single CQ decode (default: on)
    DEVSTRAL2_ONDEVICE_SAMPLING=0 # unset = on-device greedy sampling (single-CQ); 0 = host argmax
    DEVSTRAL2_SAMPLE_IN_TRACE=0   # 1 = fold sampling into decode trace (needs L1 headroom)
    DEVSTRAL2_VERBOSE_SAMPLING=0 # 1 = log every on-device argmax/top-k step
    DEVSTRAL2_MIN_MAX_SEQ_LEN=98304  # KV floor (must be >= prompt + max_new; 96K default)
    MESH_DEVICE=N150|N300|N150x4|P150x4|T3K|TG    # default 1x4 (Quietbox)

The Devstral-2-123B Hub checkpoint is gated, so the first run must have
``HF_TOKEN`` set. The shard-by-shard FP8 → bf16 dequant + tiled TTNN weight upload
is cached on disk (see ``tt/weight_loading.py``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.devstral2_123B_instruct.demo.decode_trace_2cq import (
    DecodeTrace2CQ,
    decode_trace_2cq_enabled,
    num_command_queues_for_decode,
    signal_decode_step_done,
    stage_decode_inputs,
)
from models.experimental.devstral2_123B_instruct.demo.on_device_sampling import (
    OnDeviceSampler,
    build_sampler,
    on_device_sampling_enabled,
    quiet_per_token_sampling_logs,
    sample_first_token_from_prefill_logits,
    sample_in_decode_trace,
    supports_on_device_sampling,
)
from models.experimental.devstral2_123B_instruct.tests._devstral_weights import (
    model_prefill_weight_keys,
    require_hf_weights,
    require_text_config,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_123B_instruct.tt.tt_ministral3_model import (
    TtMinistral3ForCausalLM,
)
from models.experimental.devstral2_123B_instruct.tt.weight_loading import DEVSTRAL2_LARGE_REPO_ID
from models.tt_transformers.tt.ccl import TT_CCL


_DEFAULT_MESSAGES_JSON = "messages_256k_text (1).json"


def _repo_root() -> Path:
    return Path(os.environ.get("TT_METAL_HOME", Path(__file__).resolve().parents[4]))


def _resolve_messages_json_path(raw: str) -> Path:
    """Resolve ``DEVSTRAL2_MESSAGES_JSON`` relative to ``TT_METAL_HOME`` when needed."""
    path = Path(raw).expanduser()
    if path.is_file():
        return path
    candidate = _repo_root() / raw
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"DEVSTRAL2_MESSAGES_JSON not found: {raw!r} (also tried {candidate})")


def _load_messages_json(path: Path) -> Tuple[List[dict], dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Expected non-empty 'messages' list in {path}")
    return messages, data.get("metadata") or {}


def _encode_chat(
    tokenizer,
    *,
    prompt: Optional[str],
    messages: Optional[List[dict]],
) -> torch.Tensor:
    if messages is not None:
        chat_messages = messages
    elif getattr(tokenizer, "chat_template", None):
        chat_messages = [{"role": "user", "content": prompt}]
    else:
        return tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(torch.long)

    encoded = tokenizer.apply_chat_template(
        chat_messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return encoded["input_ids"][0].to(torch.long)


def _prompt_log_text(*, prompt: Optional[str], messages: Optional[List[dict]], metadata: dict) -> str:
    if prompt is not None:
        return prompt
    if metadata:
        meta_bits = ", ".join(f"{k}={v}" for k, v in metadata.items() if v is not None)
        if meta_bits:
            return f"<messages json> ({meta_bits})"
    return "<messages json>"


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


def _round_up_max_seq_len(seq_len: int, kv_block_size: int) -> int:
    """Round ``max_seq_len`` so logical KV blocks are a multiple of 8 (flexible chunked SDPA)."""
    return _round_up(seq_len, kv_block_size * 8)


def _min_max_seq_len() -> int:
    """Minimum KV/RoPE budget (multiple of ``kv_block_size``); default 98304 via env."""
    raw = os.environ.get("DEVSTRAL2_MIN_MAX_SEQ_LEN", "262144").strip()
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


# Page-table width must be a multiple of 8 (flexible chunked SDPA stick alignment).
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


def _time_tt_op(t0: float) -> float:
    """Elapsed seconds since ``t0`` (``time.perf_counter``)."""
    return time.perf_counter() - t0


def _log_tt_throughput(stats: dict, *, padded_prompt_len: int) -> None:
    """Log TT-only prefill forward and decode trace-replay throughput."""
    tt_prefill_s = stats["tt_prefill_s"]
    tt_decode_s = stats["tt_decode_s"]
    tt_decode_steps = stats["tt_decode_steps"]
    if tt_prefill_s > 0:
        logger.info(
            f"Prefill TT forward: {tt_prefill_s * 1000:.0f}ms "
            f"({padded_prompt_len / tt_prefill_s:.1f} prompt tok/s, device only)"
        )
    if tt_decode_steps > 0 and tt_decode_s > 0:
        tt_decode_tok_per_s = tt_decode_steps / tt_decode_s
        logger.info(
            f"Decode TT throughput: {tt_decode_steps} trace replay(s) in {tt_decode_s * 1000:.0f}ms "
            f"({tt_decode_tok_per_s:.2f} tok/s/user; excludes H2D, logits D2H, sampling, compile/capture)"
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


def _first_token_from_prefill_logits(
    prefill_logits: ttnn.Tensor,
    mesh_device,
    vocab_size: int,
    local_pos: int,
    sampler: Optional[OnDeviceSampler],
) -> int:
    """First generated token: on-device argmax when ``sampler`` is set, else host ``torch.argmax``."""
    if sampler is not None:
        return sample_first_token_from_prefill_logits(sampler, prefill_logits, local_pos=local_pos)
    logits_torch = _logits_to_torch(prefill_logits, mesh_device, vocab_size)
    return int(logits_torch[local_pos].argmax().item())


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
    prompt: Optional[str],
    messages: Optional[List[dict]],
    messages_metadata: Optional[dict],
    max_new_tokens: int,
    num_layers_override: Optional[int],
) -> str:
    quiet_per_token_sampling_logs()
    logger.info(f"Loading tokenizer for {DEVSTRAL2_LARGE_REPO_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(DEVSTRAL2_LARGE_REPO_ID, trust_remote_code=True)

    text_cfg = require_text_config()
    num_layers = num_layers_override or int(text_cfg.num_hidden_layers)
    logger.info(f"Building TT model with {num_layers} / {text_cfg.num_hidden_layers} decoder layers.")

    # Tokenize first so we can size the KV cache to (prompt + max_new_tokens).
    input_ids = _encode_chat(tokenizer, prompt=prompt, messages=messages)
    prompt_len = int(input_ids.shape[0])
    prompt_log = _prompt_log_text(
        prompt=prompt,
        messages=messages,
        metadata=messages_metadata or {},
    )

    # Chunked paged prefill: pad to whole ``kv_block_size`` blocks (default 128).
    kv_block_size = Devstral2Args.kv_block_size
    num_prefill_chunks = max(1, (prompt_len + kv_block_size - 1) // kv_block_size)
    padded_prompt_len = num_prefill_chunks * kv_block_size
    min_seq_len = _min_max_seq_len()
    if messages_metadata:
        target_tokens = messages_metadata.get("measured_tokens") or messages_metadata.get("target_tokens")
        if target_tokens is not None:
            min_seq_len = max(
                min_seq_len,
                _round_up(int(target_tokens) + max_new_tokens, kv_block_size),
            )
    max_seq_len = _round_up_max_seq_len(
        max(_round_up(padded_prompt_len + max_new_tokens, kv_block_size), min_seq_len),
        kv_block_size,
    )
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
            "Set DEVSTRAL2_TRACE_PREFILL=0 to silence, or raise DEVSTRAL2_MIN_MAX_SEQ_LEN."
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

    # On-device sampling: first token from last prefill row + decode via ``ttnn.argmax`` (force-argmax).
    # Each step transfers one int32 token instead of a full sharded vocab row D2H (host fallback if disabled).
    sampler: Optional[OnDeviceSampler] = None
    if on_device_sampling_enabled():
        if not supports_on_device_sampling(args, mesh_device):
            logger.warning(
                f"On-device sampling requested but unsupported on mesh {tuple(mesh_device.shape)} "
                f"(vocab {args.vocab_size} / {args.num_devices} devices); using host argmax."
            )
        elif decode_trace_2cq_enabled():
            logger.warning(
                "On-device sampling is single-CQ only; ignoring DEVSTRAL2_DECODE_TRACE_2CQ=1 and "
                "running single-CQ decode. Set DEVSTRAL2_DECODE_TRACE_2CQ=0 to silence this."
            )
            sampler = build_sampler(args, mesh_device, tt_ccl)
        else:
            sampler = build_sampler(args, mesh_device, tt_ccl)

    fold_sampling_in_trace = sampler is not None and sample_in_decode_trace(mesh_device)
    if sampler is not None and not fold_sampling_in_trace:
        logger.info(
            "On-device sampling runs after decode trace replay (not inside the trace) to avoid "
            "L1 circular-buffer clashes on Blackhole. Set DEVSTRAL2_SAMPLE_IN_TRACE=1 to override."
        )

    decode_2cq: Optional[DecodeTrace2CQ] = None
    if decode_trace_2cq_enabled() and sampler is None:
        decode_2cq = DecodeTrace2CQ.create(mesh_device, decode_tok_dev, decode_pos_dev)
        logger.info("Decode trace 2CQ enabled (CQ1=input H2D, CQ0=forward/trace).")

    prefill_trace_id = None
    decode_trace_id = None
    prefill_trace_logits = None
    decode_trace_logits = None
    decoded = ""
    stats = {
        "tt_prefill_s": 0.0,
        "tt_decode_s": 0.0,
        "tt_decode_steps": 0,
    }
    rotary_emb = model.model.rotary_emb
    try:
        # ── Chunked paged prefill (``paged_fill_cache`` + ``chunked_scaled_dot_product_attention``) ──
        t_prefill_compile = time.perf_counter()
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
            compile_prefill_s = _time_tt_op(t_prefill_compile)
            logger.info(
                f"Prefill compile pass: {compile_prefill_s * 1000:.0f}ms "
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
            t_tt_prefill = time.perf_counter()
            prefill_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            prefill_trace_logits = model(prefill_chunk_dev, mode="prefill", **flex)
            ttnn.end_trace_capture(mesh_device, prefill_trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)

            for chunk_idx in range(1, num_prefill_chunks):
                chunk_start = chunk_idx * kv_block_size
                chunk_tokens = input_ids_padded[:, chunk_start : chunk_start + kv_block_size]
                _update_prefill_trace_buffers(
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

            stats["tt_prefill_s"] = _time_tt_op(t_tt_prefill)

            last_chunk_start = (num_prefill_chunks - 1) * kv_block_size
            ttnn.synchronize_device(mesh_device)
            local_pos = (prompt_len - 1) - last_chunk_start
            next_token = _first_token_from_prefill_logits(
                prefill_trace_logits, mesh_device, args.vocab_size, local_pos, sampler
            )
            prefill_trace_logits.deallocate(True)
            logger.info(
                f"Chunked prefill trace ({num_prefill_chunks} chunk(s)): "
                f"TT forward={stats['tt_prefill_s'] * 1000:.0f}ms"
            )
        else:
            t_tt_prefill = time.perf_counter()
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
            stats["tt_prefill_s"] = _time_tt_op(t_tt_prefill)
            local_pos = (prompt_len - 1) - last_chunk_start
            next_token = _first_token_from_prefill_logits(
                prefill_trace_logits, mesh_device, args.vocab_size, local_pos, sampler
            )
            prefill_trace_logits.deallocate(True)
            logger.info(
                f"Chunked prefill ({num_prefill_chunks} chunk(s)): " f"TT forward={stats['tt_prefill_s'] * 1000:.0f}ms"
            )

        assert next_token is not None
        logger.info(f"First generated token {next_token} = {tokenizer.decode([next_token])!r}")

        # Release prefill trace before decode so compile/capture does not allocate under an
        # active trace (Metal warns this can corrupt trace-owned buffers).
        if prefill_trace_id is not None:
            ttnn.release_trace(mesh_device, prefill_trace_id)
            prefill_trace_id = None

        # ── Decode: 1) untraced compile, 2) capture trace at iter 1, 3) replay ───
        def _read_next_token() -> int:
            """On-device sampled token (one int32 D2H) when enabled, else host argmax over logits."""
            if sampler is not None:
                return sampler.tt_read_token(0)
            logits_torch = _logits_to_torch(decode_trace_logits, mesh_device, args.vocab_size)
            return int(logits_torch[0].argmax().item())

        generated = [next_token]
        current_pos = prompt_len
        for iteration in range(max_new_tokens - 1):
            if eos_token_id is not None and next_token == eos_token_id:
                logger.info("EOS reached; stopping.")
                break

            if iteration == 0:
                # Untraced compile pass (kernel cache only; do not sample — avoids an extra
                # decode step and KV write before trace capture). When on-device sampling is on,
                # the sampling pipeline is run here too so its kernels are warm before capture.
                t_compile = time.perf_counter()
                if sampler is not None:
                    sampler.advance_seed()
                stage_decode_inputs(decode_2cq, mesh_device, decode_tok_dev, decode_pos_dev, next_token, current_pos)
                tt_out = model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
                if sampler is not None:
                    sampler.sample_into_buffer(tt_out)
                tt_out.deallocate(True)
                ttnn.synchronize_device(mesh_device)
                signal_decode_step_done(decode_2cq)
                compile_ms = (time.perf_counter() - t_compile) * 1000
                logger.info(f"Decode compile pass: {compile_ms:.0f}ms")

                # Capture trace bound to (decode_tok_dev, decode_pos_dev). On BH, sampling runs
                # after execute_trace (see fold_sampling_in_trace) so top-k CBs do not share L1
                # with the 123B decode graph inside one trace.
                t_capture = time.perf_counter()
                if sampler is not None:
                    sampler.advance_seed()
                stage_decode_inputs(decode_2cq, mesh_device, decode_tok_dev, decode_pos_dev, next_token, current_pos)
                decode_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                decode_trace_logits = model(decode_tok_dev, mode="decode", current_pos=decode_pos_dev)
                if fold_sampling_in_trace:
                    sampler.sample_into_buffer(decode_trace_logits)
                ttnn.end_trace_capture(mesh_device, decode_trace_id, cq_id=0)
                ttnn.synchronize_device(mesh_device)
                signal_decode_step_done(decode_2cq)
                capture_decode_s = _time_tt_op(t_capture)
                suffix = " (2CQ)" if decode_2cq is not None else (" (on-device sampling)" if sampler else "")
                logger.info(f"Decode trace captured in {capture_decode_s * 1000:.0f}ms{suffix} (TT forward only)")

                if sampler is not None and not fold_sampling_in_trace:
                    ttnn.execute_trace(mesh_device, decode_trace_id, cq_id=0, blocking=True)
                    ttnn.synchronize_device(mesh_device)
                    sampler.advance_seed()
                    sampler.sample_into_buffer(decode_trace_logits)

                next_token = _read_next_token()
                generated.append(next_token)
                current_pos += 1
            else:
                if sampler is not None:
                    sampler.advance_seed()
                stage_decode_inputs(decode_2cq, mesh_device, decode_tok_dev, decode_pos_dev, next_token, current_pos)
                t_decode = time.perf_counter()
                ttnn.execute_trace(mesh_device, decode_trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                signal_decode_step_done(decode_2cq)
                stats["tt_decode_s"] += _time_tt_op(t_decode)
                stats["tt_decode_steps"] += 1

                if sampler is not None and not fold_sampling_in_trace:
                    sampler.advance_seed()
                    sampler.sample_into_buffer(decode_trace_logits)

                next_token = _read_next_token()
                generated.append(next_token)
                current_pos += 1
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
    finally:
        if prefill_trace_id is not None:
            ttnn.release_trace(mesh_device, prefill_trace_id)
        if decode_trace_id is not None:
            ttnn.release_trace(mesh_device, decode_trace_id)
        if sampler is not None:
            sampler.deallocate()

    logger.info("=== Prompt ===")
    if len(prompt_log) > 500:
        logger.info(f"{prompt_log[:500]}... [{len(prompt_log)} chars total]")
    else:
        logger.info(prompt_log)
    logger.info("=== Response ===")
    logger.info(decoded)

    if stats["tt_prefill_s"] > 0 or stats["tt_decode_steps"] > 0:
        _log_tt_throughput(stats, padded_prompt_len=padded_prompt_len)
    return decoded


def _text_demo_device_params():
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 100_000_000,
        "num_command_queues": num_command_queues_for_decode(),
        "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    }


def _resolve_text_demo_inputs() -> Tuple[Optional[str], Optional[List[dict]], Optional[dict]]:
    """Prompt from ``DEVSTRAL2_MESSAGES_JSON`` (256k bench) or ``DEVSTRAL2_PROMPT``."""
    raw_json = os.environ.get("DEVSTRAL2_MESSAGES_JSON")
    if raw_json is not None:
        raw_json = raw_json.strip()
        if raw_json.lower() not in ("", "0", "false", "no"):
            path = _resolve_messages_json_path(raw_json)
            messages, metadata = _load_messages_json(path)
            logger.info(f"Loaded {len(messages)} chat message(s) from {path}")
            if metadata:
                logger.info(f"Messages metadata: {metadata}")
            return None, messages, metadata

    prompt = os.environ.get("DEVSTRAL2_PROMPT") or "Write a Python function to reverse a linked list."
    return prompt, None, None


@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", [_text_demo_device_params()], indirect=True)
def test_devstral2_123B_instruct_text_demo(mesh_device):
    prompt, messages, messages_metadata = _resolve_text_demo_inputs()
    max_new_tokens = int(os.environ.get("DEVSTRAL2_MAX_NEW_TOKENS") or "100")
    raw_layers = os.environ.get("DEVSTRAL2_NUM_LAYERS", "")
    num_layers_override = int(raw_layers) if raw_layers else None

    response = _generate(
        mesh_device,
        prompt=prompt,
        messages=messages,
        messages_metadata=messages_metadata,
        max_new_tokens=max_new_tokens,
        num_layers_override=num_layers_override,
    )
    assert response, "Empty response from TTNN Devstral-2 demo"
