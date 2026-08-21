# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Persistent traced Gemma 4 prefill runtime used by the HTTP demo service."""

from __future__ import annotations

import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.utils.partial_weights import load_cache_completion_state
from models.tt_transformers.tt.common import PagedAttentionConfig

MODEL_DTYPE = ttnn.bfloat16
PAGE_BLOCK_SIZE = 64
SLIDING_WINDOW_TOKENS = 1024
DEFAULT_CACHE_SLOTS = 8


@dataclass(frozen=True)
class PrefillResult:
    request_id: str
    status: str
    prompt_tokens: int
    padded_tokens: int
    chunks: int
    prefill_time_ms: float
    tokens_per_second: float
    cache_slot: int
    cache_generation: int
    cache_resident: bool
    next_token: None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CacheSlotState:
    slot: int
    request_id: str | None = None
    prompt_tokens: int | None = None
    padded_tokens: int | None = None
    generation: int | None = None
    cache_blocks: tuple[int, ...] = ()

    @property
    def resident(self) -> bool:
        return self.request_id is not None

    def invalidate(self) -> None:
        self.request_id = None
        self.prompt_tokens = None
        self.padded_tokens = None
        self.generation = None

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "padded_tokens": self.padded_tokens,
            "generation": self.generation,
            "cache_blocks": len(self.cache_blocks),
            "resident": self.resident,
        }


def _load_full_weights() -> bool:
    return os.environ.get("GEMMA4_PREFILL_LOAD_FULL_WEIGHTS", "0").lower() in ("1", "true", "yes")


def _cache_root(model_path: str) -> str:
    args = Gemma4ModelArgs()
    args.model_cache_path = Gemma4ModelArgs.resolve_model_cache_path(model_path)
    return str(args.weight_cache_path(MODEL_DTYPE))


def _require_cache(cache_root: str, tp: int, num_layers: int) -> None:
    if _load_full_weights():
        return

    missing = []
    if not os.path.isdir(cache_root):
        missing.append(cache_root)
    else:
        if not os.path.isdir(os.path.join(cache_root, f"layer_{num_layers - 1}")):
            missing.append(f"layer_{num_layers - 1}/")
        if not os.path.isdir(os.path.join(cache_root, "final_norm")):
            missing.append("final_norm/")
        entries = os.listdir(cache_root)
        if not any(entry.startswith(f"embed_tokens.weight_tp{tp}_") for entry in entries):
            missing.append(f"embed_tokens.weight_tp{tp}_*")
        if not any(entry.startswith(f"lm_head.weight_tp{tp}_") for entry in entries):
            missing.append(f"lm_head.weight_tp{tp}_*")

    if missing:
        raise RuntimeError(
            f"Tensor cache at {cache_root} is incomplete for TP={tp} (missing: {', '.join(missing)}). "
            "Populate it with a full-weight Gemma 4 entry point, or set "
            "GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1 to load the checkpoint and write the cache."
        )


def _cache_completion_state(model_path: str):
    if _load_full_weights():
        logger.info("GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1: loading the full host state dict")
        return None
    return load_cache_completion_state(model_path)


def _mesh_config(mesh_device) -> MeshConfig:
    return MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))


def _cp_or_replicate_mapper(mesh_device, mesh_config, seq_dim=-2):
    from models.demos.gemma4.tt.ccl import cp_degree

    if cp_degree(mesh_config) > 1:
        shard_dims = (seq_dim, None) if mesh_config.sp_axis == 0 else (None, seq_dim)
        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=shard_dims)
    return ttnn.ReplicateTensorToMesh(mesh_device)


def _host_tensor(mesh_device, torch_tensor, dtype, layout, mesh_config, seq_dim=-2):
    return ttnn.from_torch(
        torch_tensor,
        device=None,
        dtype=dtype,
        layout=layout,
        mesh_mapper=_cp_or_replicate_mapper(mesh_device, mesh_config, seq_dim=seq_dim),
    )


def _page_table_row(width: int, cache_blocks: tuple[int, ...]) -> torch.Tensor:
    if not cache_blocks:
        raise ValueError("a resident cache slot must own at least one physical block")
    if len(cache_blocks) > width:
        raise ValueError(f"cache slot owns {len(cache_blocks)} blocks, exceeding page-table width {width}")
    blocks = list(cache_blocks)
    blocks.extend([blocks[0]] * (width - len(blocks)))
    return torch.tensor(blocks, dtype=torch.int32).reshape(1, width)


def _chunk_page_table_row(cache_blocks: tuple[int, ...], chunk_idx: int, chunk_page_table_width: int) -> torch.Tensor:
    start = chunk_idx * chunk_page_table_width
    chunk_blocks = cache_blocks[start : start + chunk_page_table_width]
    if len(chunk_blocks) != chunk_page_table_width:
        raise ValueError(
            f"chunk {chunk_idx} needs {chunk_page_table_width} cache blocks, but only {len(chunk_blocks)} remain"
        )
    return torch.tensor(chunk_blocks, dtype=torch.int32).reshape(1, chunk_page_table_width)


def _fixed_cache_slot_blocks(slot: int, blocks_per_slot: int) -> tuple[int, ...]:
    """Return the disjoint CP-local physical block range permanently owned by a slot."""
    start = slot * blocks_per_slot
    return tuple(range(start, start + blocks_per_slot))


def _cp_chunk_valid_lengths(valid_tokens: int, chunk_size: int, cp: int, tp: int) -> tuple[int, ...]:
    """Return one real-token count per device for a CP-sharded chunk."""
    local_chunk = chunk_size // cp
    per_cp_rank = [max(0, min(local_chunk, valid_tokens - rank * local_chunk)) for rank in range(cp)]
    return tuple(length for length in per_cp_rank for _ in range(tp))


def _host_mesh_scalars(mesh_device, values: tuple[int, ...]):
    if len(values) != mesh_device.get_num_devices():
        raise ValueError(f"expected {mesh_device.get_num_devices()} mesh scalars, got {len(values)}")
    shards = [
        ttnn.from_torch(torch.tensor([value], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for value in values
    ]
    return ttnn.from_host_shards(shards, mesh_device.shape)


def _device_page_table(mesh_device, page_table: torch.Tensor):
    return ttnn.from_torch(
        page_table,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _host_page_table(mesh_device, page_table: torch.Tensor):
    return ttnn.from_torch(
        page_table,
        device=None,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@contextmanager
def _lm_head_deferred(model):
    previous = getattr(model, "_prefill_trace_mode", False)
    model._prefill_trace_mode = True
    try:
        yield
    finally:
        model._prefill_trace_mode = previous


class TracedPrefillRuntime:
    """Own one model, one trace, and several device-resident KV cache slots.

    Requests run serially and replace slots in round-robin order. The response's
    ``cache_generation`` identifies which request currently owns the selected slot.
    """

    def __init__(
        self,
        mesh_device,
        *,
        model_path: str,
        chunk_size: int,
        max_context_len: int,
        cache_slots: int = DEFAULT_CACHE_SLOTS,
    ):
        from models.demos.gemma4.tt.ccl import cp_degree

        if chunk_size <= 0 or max_context_len <= 0 or cache_slots <= 0:
            raise ValueError("chunk_size, max_context_len, and cache_slots must be positive")
        if max_context_len % chunk_size:
            raise ValueError("max_context_len must be divisible by chunk_size")
        if max_context_len % PAGE_BLOCK_SIZE:
            raise ValueError(f"max_context_len must be divisible by page block size {PAGE_BLOCK_SIZE}")

        self.mesh_device = mesh_device
        self.model_path = model_path
        self.chunk_size = chunk_size
        self.max_context_len = max_context_len
        self.cache_slots = cache_slots
        self.mesh_config = _mesh_config(mesh_device)
        self.cp = cp_degree(self.mesh_config)
        if self.cp <= 1:
            raise ValueError(f"the traced long-context runtime requires CP>1, got CP={self.cp}")
        if chunk_size < SLIDING_WINDOW_TOKENS * self.cp:
            raise ValueError(
                f"chunk_size={chunk_size} gives {chunk_size // self.cp} tokens per CP rank; "
                f"ring attention requires at least {SLIDING_WINDOW_TOKENS}, so chunk_size must be >= "
                f"{SLIDING_WINDOW_TOKENS * self.cp}"
            )
        if max_context_len % (PAGE_BLOCK_SIZE * self.cp):
            raise ValueError(
                f"max_context_len must be divisible by PAGE_BLOCK_SIZE * CP " f"({PAGE_BLOCK_SIZE} * {self.cp})"
            )

        self._generation = 0
        self._next_cache_slot = 0
        self._prefill_lock = threading.Lock()
        self._trace_id = None
        self._trace_output = None

        self._build_model()
        self._build_tokenizer()
        self._allocate_trace_inputs()
        self._capture_trace()

    def _build_model(self) -> None:
        tp = self.mesh_device.shape[1]
        hf_config = Gemma4ModelArgs.load_hf_config(self.model_path)
        num_layers = Gemma4ModelArgs.from_hf_config(hf_config).num_hidden_layers
        _require_cache(_cache_root(self.model_path), tp, num_layers)
        paged_config = PagedAttentionConfig(
            block_size=PAGE_BLOCK_SIZE,
            # Each logical slot owns a complete, disjoint max-context physical pool.
            max_num_blocks=max(1, self.cache_slots * self.max_context_len // PAGE_BLOCK_SIZE),
        )

        logger.info(
            "Creating Gemma 4 service model: layers={}, TP={}, CP={}, chunk={}, max_context={}, cache_slots={}",
            num_layers,
            tp,
            self.cp,
            self.chunk_size,
            self.max_context_len,
            self.cache_slots,
        )
        started = time.perf_counter()
        self.model_args, self.model, self.kv_cache, _ = create_tt_model(
            mesh_device=self.mesh_device,
            # Ring KV is prefill scratch and requests never interleave, so one user is sufficient.
            max_batch_size=1,
            max_seq_len=self.max_context_len,
            dtype=MODEL_DTYPE,
            state_dict=_cache_completion_state(self.model_path),
            model_path=self.model_path,
            create_kv_cache=True,
            paged_attention_config=paged_config,
            bounded_sliding_kv_cache=True,
            bounded_sliding_cache_slots=self.cache_slots,
            prefill_chunk_size=self.chunk_size,
        )
        self._page_table_width = self.max_context_len // PAGE_BLOCK_SIZE // self.cp
        if self.model_args.sliding_window % PAGE_BLOCK_SIZE:
            raise ValueError(f"sliding window {self.model_args.sliding_window} must be divisible by {PAGE_BLOCK_SIZE}")
        self._sliding_blocks_per_slot = self.model_args.sliding_window // PAGE_BLOCK_SIZE
        self._sliding_layers = tuple(
            layer_type == "sliding_attention" for layer_type in self.model_args.layer_types[:num_layers]
        )
        self._slot_states = [
            CacheSlotState(
                slot=slot,
                cache_blocks=_fixed_cache_slot_blocks(slot, self._page_table_width),
            )
            for slot in range(self.cache_slots)
        ]
        identity_page_table = torch.arange(self._page_table_width, dtype=torch.int32).reshape(1, -1)
        self.page_table = _device_page_table(self.mesh_device, identity_page_table)
        self._chunk_page_table_width = self.chunk_size // PAGE_BLOCK_SIZE // self.cp
        self.chunk_page_table = _device_page_table(
            self.mesh_device, identity_page_table[:, : self._chunk_page_table_width]
        )
        self.model._active_page_tables_per_layer = self._layer_page_tables(cache_slot=0)
        logger.info(
            "Gemma 4 service model ready in {:.1f}s: full_blocks/slot={}, sliding_blocks/slot={}",
            time.perf_counter() - started,
            self._page_table_width,
            self._sliding_blocks_per_slot,
        )

    def _build_tokenizer(self) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id or 0

    def _allocate_trace_inputs(self) -> None:
        zeros = torch.zeros((1, self.chunk_size), dtype=torch.int32)
        self.device_input = ttnn.to_device(
            _host_tensor(
                self.mesh_device,
                zeros,
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_config,
                seq_dim=-1,
            ),
            device=self.mesh_device,
        )
        self.device_positions = ttnn.to_device(
            _host_tensor(
                self.mesh_device,
                torch.arange(self.chunk_size, dtype=torch.int32).unsqueeze(0),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_config,
                seq_dim=-1,
            ),
            device=self.mesh_device,
        )
        self.model.set_prefill_rope_positions(self.device_positions)
        self.model._ring_metadata_external = True

    def _release_cache_slot(self, slot_state: CacheSlotState) -> None:
        # Physical ownership is permanent; only the request metadata is cleared.
        slot_state.invalidate()

    def _reserve_cache_slot(self, padded_tokens: int, request_id: str) -> CacheSlotState:
        if padded_tokens > self.max_context_len:
            raise ValueError(
                f"request needs {padded_tokens} padded tokens, exceeding per-slot capacity {self.max_context_len}"
            )
        matching_slots = [slot for slot in self._slot_states if slot.request_id == request_id]
        if matching_slots:
            # Prompt-fitting and client retries reuse a request_id and its physical slot.
            slot_state = max(matching_slots, key=lambda slot: slot.generation)
            for duplicate in matching_slots:
                if duplicate is not slot_state:
                    self._release_cache_slot(duplicate)
        else:
            cache_slot = self._next_cache_slot
            self._next_cache_slot = (cache_slot + 1) % self.cache_slots
            slot_state = self._slot_states[cache_slot]
        self._release_cache_slot(slot_state)
        slot_state.padded_tokens = padded_tokens
        return slot_state

    def _layer_page_tables(self, cache_slot: int) -> list[torch.Tensor]:
        full_row = _page_table_row(self._page_table_width, self._slot_states[cache_slot].cache_blocks)
        sliding_blocks = _fixed_cache_slot_blocks(cache_slot, self._sliding_blocks_per_slot)
        sliding_row = _page_table_row(self._page_table_width, sliding_blocks)
        return [sliding_row if is_sliding else full_row for is_sliding in self._sliding_layers]

    def _stage_page_table(self, cache_slot: int) -> None:
        layer_page_tables = self._layer_page_tables(cache_slot)
        page_table = _host_page_table(self.mesh_device, layer_page_tables[0])
        ttnn.copy_host_to_device_tensor(page_table, self.page_table)
        self.model.update_persistent_per_layer_page_tables(layer_page_tables)

    def _stage_chunk_page_table(self, cache_slot: int, chunk_idx: int) -> None:
        cache_blocks = self._slot_states[cache_slot].cache_blocks
        row = _chunk_page_table_row(cache_blocks, chunk_idx, self._chunk_page_table_width)
        chunk_page_table = _host_page_table(self.mesh_device, row)
        ttnn.copy_host_to_device_tensor(chunk_page_table, self.chunk_page_table)

    def _stage(self, tokens: torch.Tensor, chunk_idx: int, cache_slot: int, prompt_tokens: int | None = None) -> int:
        chunk_start = chunk_idx * self.chunk_size
        self._stage_chunk_page_table(cache_slot, chunk_idx)
        total_real_tokens = tokens.shape[-1] if prompt_tokens is None else prompt_tokens
        valid_tokens = max(0, min(self.chunk_size, total_real_tokens - chunk_start))
        valid_lengths = _cp_chunk_valid_lengths(valid_tokens, self.chunk_size, self.cp, self.mesh_device.shape[1])
        valid_lengths_host = _host_mesh_scalars(self.mesh_device, valid_lengths)
        ttnn.copy_host_to_device_tensor(valid_lengths_host, self.model.prefill_valid_len_dev)
        chunk_tokens = tokens[:, chunk_start : chunk_start + self.chunk_size].contiguous()
        staged_tokens = _host_tensor(
            self.mesh_device,
            chunk_tokens,
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_config,
            seq_dim=-1,
        )
        ttnn.copy_host_to_device_tensor(staged_tokens, self.device_input)
        # The ring cache is shared scratch; durable slot isolation comes from the paged cache.
        self.model.ccl_manager.set_ring_metadata(slot_idx=0, kv_actual_global=chunk_start)
        for semaphore in self.model.ccl_manager.ring_attention_ccl_semaphore_handles:
            ttnn.reset_global_semaphore_value(semaphore, 0)
        staged_positions = _host_tensor(
            self.mesh_device,
            torch.arange(chunk_start, chunk_start + self.chunk_size, dtype=torch.int32).unsqueeze(0),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self.mesh_config,
            seq_dim=-1,
        )
        ttnn.copy_host_to_device_tensor(staged_positions, self.device_positions)
        return chunk_start

    def _forward(self, chunk_start: int):
        with _lm_head_deferred(self.model):
            embeds, page_table, chunk_page_table, _ = self.model.transform_and_embed_prefill_inputs_device(
                self.device_input, self.page_table, self.chunk_page_table, None
            )
            return self.model.ttnn_prefill_forward(
                x=embeds,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start,
                kv_cache=self.kv_cache,
                get_last_token=-1,
                user_id=0,
            )

    def _capture_trace(self) -> None:
        from models.demos.gemma4.tt.attention import ring_prefill

        capture_tokens = torch.zeros((1, self.chunk_size), dtype=torch.int32)
        started = time.perf_counter()
        warmup_output = self._forward(self._stage(capture_tokens, 0, 0))
        ttnn.synchronize_device(self.mesh_device)
        warmup_output.deallocate(True)
        warmup_s = time.perf_counter() - started

        ring_prefill.reset_ring_attention_calls()
        capture_start = self._stage(capture_tokens, 0, 0)
        started = time.perf_counter()
        self._trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._trace_output = self._forward(capture_start)
        ttnn.end_trace_capture(self.mesh_device, self._trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        capture_s = time.perf_counter() - started

        captured_ring_calls = ring_prefill.ring_attention_calls()
        if captured_ring_calls < len(self.model.layers):
            raise RuntimeError(
                f"captured only {captured_ring_calls} ring calls; expected at least {len(self.model.layers)}"
            )

        self._stage(capture_tokens, 0, 0)
        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)
        logger.info("Prefill trace ready: warmup={:.1f}s capture={:.1f}s", warmup_s, capture_s)

    def _tokenize(self, prompt: str) -> torch.Tensor:
        conversation = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            tokens = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        else:
            tokens = self.tokenizer.encode(prompt, return_tensors="pt")
        if hasattr(tokens, "input_ids"):
            tokens = tokens.input_ids
        return tokens.to(dtype=torch.int32)

    @torch.no_grad()
    def prefill(self, prompt: str, request_id: str) -> dict:
        with self._prefill_lock:
            return self._prefill_serial(prompt, request_id)

    def _prefill_serial(self, prompt: str, request_id: str) -> dict:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        tokens = self._tokenize(prompt)
        prompt_tokens = int(tokens.shape[-1])
        if prompt_tokens > self.max_context_len:
            raise ValueError(f"prompt has {prompt_tokens} tokens, exceeding max_context_len={self.max_context_len}")
        padded_tokens = math.ceil(prompt_tokens / self.chunk_size) * self.chunk_size
        if padded_tokens > prompt_tokens:
            padding = torch.full(
                (1, padded_tokens - prompt_tokens),
                self.pad_token_id,
                dtype=torch.int32,
            )
            tokens = torch.cat((tokens, padding), dim=-1)

        chunks = padded_tokens // self.chunk_size
        slot_state = self._reserve_cache_slot(padded_tokens, request_id)
        cache_slot = slot_state.slot
        started = time.perf_counter()
        try:
            self._stage_page_table(cache_slot)
            for chunk_idx in range(chunks):
                self._stage(tokens, chunk_idx, cache_slot, prompt_tokens)
                ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(self.mesh_device)
        except Exception:
            self._release_cache_slot(slot_state)
            raise
        elapsed_s = time.perf_counter() - started

        self._generation += 1
        slot_state.request_id = request_id
        slot_state.prompt_tokens = prompt_tokens
        slot_state.generation = self._generation
        return PrefillResult(
            request_id=request_id,
            status="prefilled",
            prompt_tokens=prompt_tokens,
            padded_tokens=padded_tokens,
            chunks=chunks,
            prefill_time_ms=round(elapsed_s * 1000, 3),
            tokens_per_second=round(prompt_tokens / elapsed_s, 3),
            cache_slot=cache_slot,
            cache_generation=self._generation,
            cache_resident=True,
        ).to_dict()

    def info(self) -> dict:
        resident_slots = [slot.to_dict() for slot in self._slot_states if slot.resident]
        return {
            "status": "ready",
            "model": self.model_path,
            "mesh": [int(self.mesh_device.shape[0]), int(self.mesh_device.shape[1])],
            "chunk_size": self.chunk_size,
            "max_context_len": self.max_context_len,
            "cache_slots": self.cache_slots,
            "cache_slot_capacity_tokens": self.max_context_len,
            "cache_capacity_tokens": self.cache_slots * self.max_context_len,
            "cache_blocks_per_slot": self._page_table_width,
            "sliding_cache_blocks_per_slot": self._sliding_blocks_per_slot,
            "free_cache_slots": self.cache_slots - len(resident_slots),
            "resident_cache_slots": len(resident_slots),
            "next_cache_slot": self._next_cache_slot,
            "resident_slots": resident_slots,
            "cache_resident": bool(resident_slots),
            "cache_generation": self._generation,
            "next_token_enabled": False,
        }

    def close(self) -> None:
        if self._trace_id is not None:
            ttnn.release_trace(self.mesh_device, self._trace_id)
            self._trace_id = None
