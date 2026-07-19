# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Metal-readiness generator for the optimized TP=4 Llama 3.1 8B model."""

from __future__ import annotations

import json
import math
import os
import secrets
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.model import (
    DEFAULT_NUM_BLOCKS,
    FullModelConfig,
    Llama31FullModel,
)
from models.common.readiness_check.contract import Generator, NextInputFn
from models.tt_transformers.tt.common import copy_host_to_device

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
UINT32_MAX = 2**32 - 1


class SafetensorStateDict(Mapping[str, torch.Tensor]):
    """Lazy, bounded-memory view over a sharded safetensors checkpoint."""

    def __init__(self, snapshot_path: str | Path):
        self.snapshot_path = Path(snapshot_path)
        index_path = self.snapshot_path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        self.weight_map: dict[str, str] = index["weight_map"]

    def __getitem__(self, key: str) -> torch.Tensor:
        filename = self.weight_map[key]
        with safe_open(self.snapshot_path / filename, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.weight_map)

    def __len__(self) -> int:
        return len(self.weight_map)


def _round_up(value: int, multiple: int) -> int:
    return multiple * math.ceil(value / multiple)


def _first_device_to_torch(tensor) -> torch.Tensor:
    device_tensors = ttnn.get_device_tensors(tensor)
    return ttnn.to_torch(device_tensors[0] if device_tensors else tensor)


class Llama31Generator(Generator):
    """Explicit-cache low-level API plus traced device-feedback generation."""

    def __init__(self, model: Llama31FullModel, tokenizer):
        self.model = model
        self.mesh_device = model.mesh_device
        self.tokenizer = tokenizer
        self.batch = model.batch

        self._kv_cache = None
        self._page_table_host: torch.Tensor | None = None
        self._trace_kv_cache = None
        self._trace_model_id: int | None = None
        self._trace_sampling_id: int | None = None
        self._trace_inputs = None
        self._trace_logits = None
        self._trace_page_table_snapshot: torch.Tensor | None = None
        self._trace_active_batch: int | None = None
        self._decode_warm_key: tuple[int, int] | None = None
        self._sampling_params = None
        self._sampling_param_snapshot = None
        self._sampling_stochastic = False
        self._sampling_seed_transition_pending = False
        self._sampling_request_active = False
        self._sampling_request_seed_values: list[int] | None = None
        self._prefill_trace_ids: dict[tuple, Any] = {}
        self.trace_stats = {
            "captures": 0,
            "replays": 0,
            "releases": 0,
            "prefill_captures": 0,
            "prefill_replays": 0,
            "prefill_releases": 0,
            "decode_warmups": 0,
            "token_host_copies": 0,
            "position_host_copies": 0,
            "page_table_host_copies": 0,
            "sampling_param_host_copies": 0,
            "sampling_seed_host_copies": 0,
            "caller_token_readbacks": 0,
            "explicit_synchronizations": 0,
        }
        self._allocate_trace_inputs()

    def _ensure_kv_cache(self):
        if self._kv_cache is None:
            self._kv_cache = self.model.allocate_kv_cache()
        return self._kv_cache

    def _page_table_to_torch(self, page_table) -> torch.Tensor:
        if isinstance(page_table, torch.Tensor):
            result = page_table.detach().cpu().to(torch.int32)
        elif isinstance(page_table, ttnn.Tensor):
            result = _first_device_to_torch(page_table).to(torch.int32)
        else:
            raise TypeError("page_table must be a torch or TTNN tensor")
        if result.ndim != 2:
            raise ValueError(f"page_table must be rank two, got {tuple(result.shape)}")
        return result

    def _normalise_page_table(self, page_table, active_batch: int) -> torch.Tensor:
        host = self._page_table_to_torch(page_table)
        if host.shape[0] > self.batch:
            raise ValueError(f"page table batch {host.shape[0]} exceeds configured batch {self.batch}")
        if host.shape[0] < active_batch:
            raise ValueError("page table has fewer rows than active prompts")
        if host.shape[0] < self.batch:
            host = torch.nn.functional.pad(host, (0, 0, 0, self.batch - host.shape[0]), value=-1)
        if host.shape[1] > self.model.config.num_blocks:
            raise ValueError(
                f"page table has {host.shape[1]} columns, configured cache has {self.model.config.num_blocks} blocks"
            )
        if host.shape[1] < self.model.config.num_blocks:
            host = torch.nn.functional.pad(host, (0, self.model.config.num_blocks - host.shape[1]), value=-1)
        return host.contiguous()

    def _make_page_table(self, lengths: list[int]) -> torch.Tensor:
        if len(lengths) > self.batch:
            raise ValueError(f"{len(lengths)} prompts exceed configured batch {self.batch}")
        table = torch.full((self.batch, self.model.config.num_blocks), -1, dtype=torch.int32)
        next_block = 0
        for user, length in enumerate(lengths):
            blocks = math.ceil(length / 64)
            if next_block + blocks > self.model.config.num_blocks:
                raise ValueError(
                    f"paged cache requires {next_block + blocks} blocks, only "
                    f"{self.model.config.num_blocks} are available"
                )
            table[user, :blocks] = torch.arange(next_block, next_block + blocks, dtype=torch.int32)
            next_block += blocks
        return table

    def _page_table_to_device(self, host: torch.Tensor):
        return ttnn.from_torch(
            host,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _tokens_to_device(self, host: torch.Tensor):
        return ttnn.from_torch(
            host.to(torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _allocate_trace_inputs(self) -> None:
        """Allocate all shape-varying inputs before any trace is captured."""

        chunk_size = self.model.config.prefill_chunk_size
        max_chunks = math.ceil(self.model.config.max_context_len / chunk_size)
        self._prefill_page_table_device = self._page_table_to_device(
            torch.full((self.batch, self.model.config.num_blocks), -1, dtype=torch.int32)
        )
        self._prefill_full_chunk_tokens = [
            self._tokens_to_device(torch.zeros((self.batch, chunk_size), dtype=torch.long)) for _ in range(max_chunks)
        ]
        self._prefill_full_chunk_tables = [
            self._page_table_to_device(torch.full((self.batch, chunk_size // 64), -1, dtype=torch.int32))
            for _ in range(1, max_chunks)
        ]
        tail_lengths = range(128, chunk_size, 128)
        self._prefill_tail_tokens = {
            length: self._tokens_to_device(torch.zeros((self.batch, length), dtype=torch.long))
            for length in tail_lengths
        }
        self._prefill_tail_chunk_tables = {
            length: self._page_table_to_device(torch.full((self.batch, length // 64), -1, dtype=torch.int32))
            for length in tail_lengths
        }
        self._prefill_sampled = ttnn.from_torch(
            torch.zeros((1, 1, 1, 32), dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        self._decode_trace_input_pool = (
            ttnn.from_torch(
                torch.zeros((1, 1, 1, 32), dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.full((self.batch,), -1, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.zeros((1, self.batch), dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            self._page_table_to_device(torch.full((self.batch, self.model.config.num_blocks), -1, dtype=torch.int32)),
        )

    def _copy_prefill_host_to_device(self, host: torch.Tensor, device, *, dtype) -> None:
        host_tensor = ttnn.from_torch(
            host,
            device=None,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host_tensor, device)

    def _stage_prefill_trace_inputs(
        self,
        tokens: torch.Tensor,
        page_host: torch.Tensor,
        *,
        active_batch: int,
        logical_width: int,
        padded_width: int,
    ):
        """Refresh persistent prefill inputs without allocating device buffers."""

        self._copy_prefill_host_to_device(page_host, self._prefill_page_table_device, dtype=ttnn.int32)
        staged = []
        chunk_size = self.model.config.prefill_chunk_size
        for chunk_index, start in enumerate(range(0, padded_width, chunk_size)):
            chunk_len = min(chunk_size, padded_width - start)
            chunk_host = torch.zeros((self.batch, chunk_len), dtype=torch.int32)
            source_end = min(logical_width, start + chunk_len)
            if source_end > start:
                chunk_host[:active_batch, : source_end - start] = tokens[:, start:source_end].to(torch.int32)
            token_device = (
                self._prefill_full_chunk_tokens[chunk_index]
                if chunk_len == chunk_size
                else self._prefill_tail_tokens[chunk_len]
            )
            self._copy_prefill_host_to_device(chunk_host, token_device, dtype=ttnn.uint32)

            chunk_table_device = None
            if start:
                chunk_table_host = self._chunk_page_table(page_host, start, chunk_len)
                chunk_table_device = (
                    self._prefill_full_chunk_tables[chunk_index - 1]
                    if chunk_len == chunk_size
                    else self._prefill_tail_chunk_tables[chunk_len]
                )
                self._copy_prefill_host_to_device(chunk_table_host, chunk_table_device, dtype=ttnn.int32)
            staged.append((start, chunk_len, token_device, chunk_table_device))
        return staged

    def _local_logits_to_torch(self, logits) -> torch.Tensor:
        logits = ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        host = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3))
        return host[..., : self.model.vocab_size].float()

    def _sampled_tokens_to_torch(self, tokens) -> torch.Tensor:
        self.trace_stats["caller_token_readbacks"] += 1
        return _first_device_to_torch(tokens).reshape(-1)[: self.batch].to(torch.long)

    def _synchronize_device(self) -> None:
        ttnn.synchronize_device(self.mesh_device)
        self.trace_stats["explicit_synchronizations"] += 1

    def _run_device_prefill(
        self,
        staged_inputs,
        *,
        kv_cache,
        page_device,
        prompt_lens: list[int],
        padded_prompt_lens: list[int],
    ):
        selections = [None] * len(prompt_lens)
        for start, chunk_len, chunk_tokens, chunk_page_table in staged_inputs:
            hidden = self.model.prefill_forward(
                chunk_tokens,
                page_table=page_device,
                kv_cache=kv_cache,
                prompt_lens=padded_prompt_lens,
                chunk_start_idx=start,
                chunk_page_table=chunk_page_table,
                return_hidden=True,
            )
            finishing_users = [user for user, length in enumerate(prompt_lens) if start < length <= start + chunk_len]
            for user in finishing_users:
                selections[user] = self.model.select_prefill_token_hidden(
                    hidden,
                    user,
                    prompt_lens[user] - start - 1,
                )
        if any(selection is None for selection in selections):
            raise RuntimeError(f"failed to select final hidden rows: {[item is not None for item in selections]}")
        selected_logits = self.model.prefill_selected_hidden_logits(selections, fixed_sampling_rows=True)
        k, p, temp = self._ensure_sampling_params()
        return self._sample_device_split(selected_logits, k=k, p=p, temp=temp, tt_out_tok=self._prefill_sampled)

    def _prefill_device_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_host: torch.Tensor,
        kv_cache,
        prompt_lens: list[int],
        active_batch: int,
        logical_width: int,
        padded_width: int,
        enable_trace: bool,
        seed=None,
    ):
        staged_inputs = self._stage_prefill_trace_inputs(
            tokens,
            page_host,
            active_batch=active_batch,
            logical_width=logical_width,
            padded_width=padded_width,
        )
        padded_prompt_lens = list(prompt_lens) + [0] * (self.batch - active_batch)
        trace_key = (id(kv_cache), active_batch, tuple(padded_prompt_lens), padded_width)
        trace_id = self._prefill_trace_ids.get(trace_key)
        live_traces = self._trace_model_id is not None or bool(self._prefill_trace_ids)
        if self._sampling_stochastic and not self._sampling_request_active:
            self.begin_sampling_request(
                seed=seed,
                active_batch=active_batch,
                defer_device_seed=enable_trace and trace_id is None,
            )

        if trace_id is not None and enable_trace:
            ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
            self.trace_stats["prefill_replays"] += 1
            sampled = self._prefill_sampled
        elif not enable_trace:
            if live_traces:
                self._release_all_traces()
            sampled = self._run_device_prefill(
                staged_inputs,
                kv_cache=kv_cache,
                page_device=self._prefill_page_table_device,
                prompt_lens=prompt_lens,
                padded_prompt_lens=padded_prompt_lens,
            )
        else:
            # The prefill trace key includes every shape/slice/cache identity
            # used by the graph. Warm the exact graph on the exact persistent
            # buffers, then drain all eager/reset/copy commands before capture.
            if live_traces:
                self._release_all_traces()
            self._run_device_prefill(
                staged_inputs,
                kv_cache=kv_cache,
                page_device=self._prefill_page_table_device,
                prompt_lens=prompt_lens,
                padded_prompt_lens=padded_prompt_lens,
            )
            self._synchronize_device()
            self._warm_decode_graphs(
                kv_cache,
                page_host,
                active_batch=active_batch,
                positions=prompt_lens,
            )
            self.model.reset_kv_cache(kv_cache)
            self._stage_prefill_trace_inputs(
                tokens,
                page_host,
                active_batch=active_batch,
                logical_width=logical_width,
                padded_width=padded_width,
            )
            self._synchronize_device()

            self.mesh_device.set_program_cache_misses_allowed(False)
            trace_id = None
            capture_open = False
            try:
                trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                capture_open = True
                sampled = self._run_device_prefill(
                    staged_inputs,
                    kv_cache=kv_cache,
                    page_device=self._prefill_page_table_device,
                    prompt_lens=prompt_lens,
                    padded_prompt_lens=padded_prompt_lens,
                )
                ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                capture_open = False
            except Exception:
                if capture_open:
                    try:
                        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                    except Exception:
                        pass
                if trace_id is not None:
                    try:
                        ttnn.release_trace(self.mesh_device, trace_id)
                    except Exception:
                        pass
                    self.trace_stats["prefill_releases"] += 1
                raise
            finally:
                self.mesh_device.set_program_cache_misses_allowed(True)

            self._prefill_trace_ids[trace_key] = trace_id
            self.trace_stats["prefill_captures"] += 1
            self._synchronize_device()

            self.model.reset_kv_cache(kv_cache)
            self._activate_deferred_sampling_seed()
            self._synchronize_device()
            ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
            self.trace_stats["prefill_replays"] += 1
            sampled = self._prefill_sampled
        self._transition_sampling_seed_to_device_advance()
        return sampled

    def _chunk_page_table(self, full_table: torch.Tensor, start: int, chunk_len: int):
        first_page = start // 64
        page_count = math.ceil(chunk_len / 64)
        return full_table[:, first_page : first_page + page_count].contiguous()

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache: Any,
        prompt_lens: list[int],
        return_all_logits: bool = False,
        sampling_mode: str = "host",
        enable_trace: bool = True,
        seed=None,
        **kwargs: Any,
    ):
        """Prefill logical lengths, returning host logits or fixed-slot device samples."""

        if sampling_mode not in {"host", "device"}:
            raise ValueError("sampling_mode must be 'host' or 'device'")
        if sampling_mode == "device" and return_all_logits:
            raise ValueError("return_all_logits is incompatible with device sampling")
        if sampling_mode == "host" and (self._trace_model_id is not None or self._prefill_trace_ids):
            self._release_all_traces()
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [batch, seq], got {tuple(tokens.shape)}")
        active_batch, logical_width = tokens.shape
        if active_batch > self.batch:
            raise ValueError(f"batch {active_batch} exceeds configured batch {self.batch}")
        if len(prompt_lens) != active_batch or any(length < 1 or length > logical_width for length in prompt_lens):
            raise ValueError("prompt_lens must contain one valid logical length per input row")
        if max(prompt_lens) > self.model.config.max_context_len:
            raise ValueError("prompt exceeds the supported context")

        caches = self._ensure_kv_cache() if kv_cache is None else kv_cache
        page_host = self._normalise_page_table(page_table, active_batch)
        padded_width = max(128, _round_up(max(prompt_lens), 128))
        if sampling_mode == "device":
            self._page_table_host = page_host
            return self._prefill_device_forward(
                tokens,
                page_host=page_host,
                kv_cache=caches,
                prompt_lens=prompt_lens,
                active_batch=active_batch,
                logical_width=logical_width,
                padded_width=padded_width,
                enable_trace=enable_trace,
                seed=seed,
            )
        page_device = self._page_table_to_device(page_host)
        padded_prompt_lens = list(prompt_lens) + [0] * (self.batch - active_batch)
        chunk_size = self.model.config.prefill_chunk_size
        all_logits = []
        selections = [None] * active_batch

        for start in range(0, padded_width, chunk_size):
            chunk_len = min(chunk_size, padded_width - start)
            chunk_host = torch.zeros((self.batch, chunk_len), dtype=torch.long)
            source_end = min(logical_width, start + chunk_len)
            if source_end > start:
                chunk_host[:active_batch, : source_end - start] = tokens[:, start:source_end]
            chunk_tokens = self._tokens_to_device(chunk_host)
            chunk_table = None
            if start:
                chunk_table = self._page_table_to_device(self._chunk_page_table(page_host, start, chunk_len))

            finishing_users = [user for user, length in enumerate(prompt_lens) if start < length <= start + chunk_len]
            output = self.model.prefill_forward(
                chunk_tokens,
                page_table=page_device,
                kv_cache=caches,
                prompt_lens=padded_prompt_lens,
                chunk_start_idx=start,
                chunk_page_table=chunk_table,
                return_hidden=not return_all_logits,
            )
            if return_all_logits:
                chunk_logits = self._local_logits_to_torch(output)[0, : self.batch, :chunk_len, :]
                all_logits.append(chunk_logits[:active_batch])
            else:
                for user in finishing_users:
                    selections[user] = self.model.select_prefill_token_hidden(
                        output,
                        user,
                        prompt_lens[user] - start - 1,
                    )

        self._page_table_host = page_host
        if return_all_logits:
            return torch.cat(all_logits, dim=1)[:, :logical_width]
        if any(selection is None for selection in selections):
            raise RuntimeError(f"failed to select final hidden rows: {[item is not None for item in selections]}")
        selected_logits = self.model.prefill_selected_hidden_logits(selections, fixed_sampling_rows=False)
        host_logits = self._local_logits_to_torch(selected_logits)[0, 0, :active_batch, :]
        return host_logits.unsqueeze(1)

    def _prepare_decode_host_inputs(self, tokens: torch.Tensor, positions: torch.Tensor, page_table: torch.Tensor):
        tokens = tokens.reshape(-1).to(torch.int64)
        positions = positions.reshape(-1).to(torch.int64)
        if tokens.numel() > self.batch or positions.numel() > self.batch:
            raise ValueError("decode batch exceeds configured batch")
        padded_tokens = torch.zeros(32, dtype=torch.int32)
        padded_tokens[: tokens.numel()] = tokens.to(torch.int32)
        padded_positions = torch.full((self.batch,), -1, dtype=torch.int32)
        padded_positions[: positions.numel()] = positions.to(torch.int32)
        rope_positions = torch.clamp(padded_positions, min=0).reshape(1, self.batch)
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        return (
            ttnn.from_torch(
                padded_tokens.reshape(1, 1, 1, 32),
                device=None,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                padded_positions,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                rope_positions,
                device=None,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                page_table,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
        )

    def _validate_decode_page_coverage(
        self,
        page_table: torch.Tensor,
        positions: torch.Tensor,
        *,
        active_batch: int,
    ) -> None:
        """Reject missing physical pages before a low-level decode mutates cache."""

        positions = positions.reshape(-1).to(torch.int64)
        for slot in range(active_batch):
            position = int(positions[slot].item())
            if position < 0:
                continue
            logical_page = position // 64
            if logical_page >= page_table.shape[1] or int(page_table[slot, logical_page].item()) < 0:
                raise ValueError(
                    f"page table slot {slot} does not map decode position {position} " f"(logical page {logical_page})"
                )

    def _warm_decode_graphs(self, kv_cache, page_host: torch.Tensor, *, active_batch: int, positions) -> None:
        """Compile the exact persistent decode/model sampler graphs before traces exist."""

        self._trace_inputs = self._decode_trace_input_pool
        host_inputs = self._prepare_decode_host_inputs(
            torch.zeros(active_batch, dtype=torch.long),
            torch.as_tensor(positions, dtype=torch.long),
            page_host,
        )
        self._restore_trace_inputs(host_inputs, include_page_table=True)
        token, current_pos, rope_pos, page_table = self._trace_inputs
        k, p, temp = self._ensure_sampling_params()
        self.model.sampler.load_device_buffers()
        warm_logits = self.model.decode_forward(
            token,
            current_pos,
            rope_pos,
            page_table=page_table,
            kv_cache=kv_cache,
        )
        self._sample_device_split(warm_logits, k=k, p=p, temp=temp, tt_out_tok=token)
        self._synchronize_device()
        self._restore_trace_inputs(host_inputs, include_page_table=True)
        self._synchronize_device()
        self._decode_warm_key = (id(kv_cache), active_batch)
        self.trace_stats["decode_warmups"] += 1

    def _ensure_sampling_params(self):
        if self._sampling_params is not None:
            return self._sampling_params
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        self._sampling_params = (
            ttnn.from_torch(
                torch.ones(32, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.zeros(32, dtype=torch.bfloat16),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.ones(32, dtype=torch.bfloat16),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
        )
        self._sampling_param_snapshot = (
            tuple([1] * 32),
            tuple([0.0] * 32),
            tuple([1.0] * 32),
        )
        return self._sampling_params

    def _sample_device_split(self, logits, *, k, p, temp, tt_out_tok=None):
        if self._sampling_stochastic:
            return self.model.sample_stochastic_split(logits, k=k, p=p, temp=temp, tt_out_tok=tt_out_tok)
        return self.model.sample_greedy_split(logits, k=k, p=p, temp=temp, tt_out_tok=tt_out_tok)

    @staticmethod
    def _expand_sampling_value(value, *, active_batch: int, inactive_value, name: str):
        if isinstance(value, (int, float)):
            active = [value] * active_batch
        else:
            active = list(value)
            if len(active) != active_batch:
                raise ValueError(f"{name} must be scalar or have {active_batch} entries")
        return active + [inactive_value] * (32 - active_batch)

    @staticmethod
    def _format_device_sampling_params(k_values, p_values, temperature_values):
        """Format public sampler values for ``ttnn.sampling`` without device I/O."""

        k_values = list(k_values)
        p_values = list(p_values)
        device_temp_values = []
        for slot, value in enumerate(temperature_values):
            if value == 0.0:
                k_values[slot] = 1
                p_values[slot] = 0.0
                device_temp_values.append(1.0)
            else:
                device_temp_values.append(1.0 / value)
        return k_values, p_values, device_temp_values

    def set_sampling_params(
        self,
        *,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        active_batch: int | None = None,
    ) -> None:
        """Update trace-stable sampler tensors once at a request boundary."""

        active_batch = self.batch if active_batch is None else int(active_batch)
        if active_batch < 1 or active_batch > self.batch:
            raise ValueError(f"active_batch must be in [1, {self.batch}]")
        k_values = [
            int(value)
            for value in self._expand_sampling_value(top_k, active_batch=active_batch, inactive_value=1, name="top_k")
        ]
        p_values = [
            float(value)
            for value in self._expand_sampling_value(top_p, active_batch=active_batch, inactive_value=0.0, name="top_p")
        ]
        temp_values = [
            float(value)
            for value in self._expand_sampling_value(
                temperature, active_batch=active_batch, inactive_value=0.0, name="temperature"
            )
        ]
        if any(value < 1 or value > self.model.sampler.config.max_top_k for value in k_values[:active_batch]):
            raise ValueError(f"top_k must be in [1, {self.model.sampler.config.max_top_k}]")
        if any(value < 0.0 or value > 1.0 for value in p_values[:active_batch]):
            raise ValueError("top_p must be in [0, 1]")
        if any(value < 0.0 for value in temp_values[:active_batch]):
            raise ValueError("temperature must be non-negative")
        # ``ttnn.sampling`` consumes inverse temperature. Match the common
        # formatter's temperature=0 greedy sentinel on every fixed slot.
        k_values, p_values, device_temp_values = self._format_device_sampling_params(
            k_values,
            p_values,
            temp_values,
        )
        snapshot = (tuple(k_values), tuple(p_values), tuple(device_temp_values))
        sampling_stochastic = any(value > 1 for value in k_values[:active_batch])
        if sampling_stochastic != self._sampling_stochastic and (
            self._trace_model_id is not None or self._prefill_trace_ids
        ):
            # Greedy candidate-argmax and stochastic sampling are different
            # trace graphs. Replace them only at this explicit request boundary.
            self._release_all_traces()
            self._decode_warm_key = None
        self._sampling_stochastic = sampling_stochastic
        device_params = self._ensure_sampling_params()
        if snapshot == self._sampling_param_snapshot:
            return
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        host_params = (
            ttnn.from_torch(
                torch.tensor(k_values, dtype=torch.int32),
                device=None,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.tensor(p_values, dtype=torch.bfloat16),
                device=None,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.tensor(device_temp_values, dtype=torch.bfloat16),
                device=None,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
        )
        copy_host_to_device(host_params, device_params)
        self._sampling_param_snapshot = snapshot
        self.trace_stats["sampling_param_host_copies"] += 3

    def _copy_sampling_seeds(self, values: list[int]) -> None:
        self.model.sampler.load_device_buffers()
        host = ttnn.from_torch(
            torch.tensor(values, dtype=torch.int64).to(torch.uint32),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, self.model.sampler._seeds)
        self.trace_stats["sampling_seed_host_copies"] += 1

    def begin_sampling_request(
        self,
        *,
        seed=None,
        active_batch: int | None = None,
        defer_device_seed: bool = False,
    ) -> None:
        """Initialize a trace-stable device RNG stream at a request boundary."""

        active_batch = self.batch if active_batch is None else int(active_batch)
        if active_batch < 1 or active_batch > self.batch:
            raise ValueError(f"active_batch must be in [1, {self.batch}]")
        self._sampling_request_active = True
        self._sampling_seed_transition_pending = False
        self._sampling_request_seed_values = None
        if not self._sampling_stochastic:
            return

        if seed is None:
            active_seeds = [secrets.randbelow(UINT32_MAX) for _ in range(active_batch)]
        elif isinstance(seed, int):
            base = seed % UINT32_MAX
            active_seeds = [(base + slot * 0x9E3779B9) % UINT32_MAX for slot in range(active_batch)]
        else:
            active_seeds = [int(value) % UINT32_MAX for value in seed]
            if len(active_seeds) != active_batch:
                raise ValueError(f"seed must be scalar or have {active_batch} entries")
        self._sampling_request_seed_values = active_seeds + [UINT32_MAX] * (32 - active_batch)
        if defer_device_seed:
            return
        self._copy_sampling_seeds(self._sampling_request_seed_values)
        self._sampling_seed_transition_pending = True

    def _activate_deferred_sampling_seed(self) -> None:
        if self._sampling_request_seed_values is None or self._sampling_seed_transition_pending:
            return
        self._copy_sampling_seeds(self._sampling_request_seed_values)
        self._sampling_seed_transition_pending = True

    def _transition_sampling_seed_to_device_advance(self) -> None:
        if not self._sampling_seed_transition_pending:
            return
        self._copy_sampling_seeds([UINT32_MAX] * 32)
        self._sampling_seed_transition_pending = False
        self._sampling_request_seed_values = None

    def _restore_trace_inputs(self, host_inputs, *, include_page_table: bool):
        count = 4 if include_page_table else 3
        copy_host_to_device(host_inputs[:count], self._trace_inputs[:count])
        self.trace_stats["token_host_copies"] += 1
        self.trace_stats["position_host_copies"] += 2
        if include_page_table:
            self.trace_stats["page_table_host_copies"] += 1

    def _capture_decode_traces(self, host_inputs, kv_cache, *, active_batch: int):
        self._trace_inputs = self._decode_trace_input_pool
        warm_key = (id(kv_cache), active_batch)
        if self._decode_warm_key != warm_key:
            # Direct low-level decode without a preceding device prefill still
            # gets an exact warm epoch. A live prefill trace must be released
            # first because warmup may allocate cached intermediates.
            self._release_prefill_traces()
            self._restore_trace_inputs(host_inputs, include_page_table=True)
            token, current_pos, rope_pos, page_table = self._trace_inputs
            k, p, temp = self._ensure_sampling_params()
            self.model.sampler.load_device_buffers()
            warm_logits = self.model.decode_forward(
                token,
                current_pos,
                rope_pos,
                page_table=page_table,
                kv_cache=kv_cache,
            )
            self._sample_device_split(warm_logits, k=k, p=p, temp=temp, tt_out_tok=token)
            self._synchronize_device()
            self._decode_warm_key = warm_key
            self.trace_stats["decode_warmups"] += 1
        self._restore_trace_inputs(host_inputs, include_page_table=True)
        self._synchronize_device()
        token, current_pos, rope_pos, page_table = self._trace_inputs
        k, p, temp = self._ensure_sampling_params()
        self.model.sampler.load_device_buffers()

        self.mesh_device.set_program_cache_misses_allowed(False)
        model_trace_id = None
        sampling_trace_id = None
        model_capture_open = False
        sampling_capture_open = False
        try:
            model_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            model_capture_open = True
            logits = self.model.decode_forward(
                token,
                current_pos,
                rope_pos,
                page_table=page_table,
                kv_cache=kv_cache,
            )
            ttnn.end_trace_capture(self.mesh_device, model_trace_id, cq_id=0)
            model_capture_open = False
            self._synchronize_device()

            sampling_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            sampling_capture_open = True
            sampled = self._sample_device_split(logits, k=k, p=p, temp=temp, tt_out_tok=token)
            ttnn.end_trace_capture(self.mesh_device, sampling_trace_id, cq_id=0)
            sampling_capture_open = False
            self._synchronize_device()
        except Exception:
            if sampling_capture_open:
                try:
                    ttnn.end_trace_capture(self.mesh_device, sampling_trace_id, cq_id=0)
                except Exception:
                    pass
            if model_capture_open:
                try:
                    ttnn.end_trace_capture(self.mesh_device, model_trace_id, cq_id=0)
                except Exception:
                    pass
            for partial_trace_id in (sampling_trace_id, model_trace_id):
                if partial_trace_id is not None:
                    try:
                        ttnn.release_trace(self.mesh_device, partial_trace_id)
                    except Exception:
                        pass
            raise
        finally:
            self.mesh_device.set_program_cache_misses_allowed(True)

        self._trace_model_id = model_trace_id
        self._trace_sampling_id = sampling_trace_id
        self._trace_logits = logits
        self._trace_sampled = sampled
        self._trace_kv_cache = kv_cache
        self._trace_page_table_snapshot = self._page_table_to_torch(host_inputs[3]).clone()
        self._trace_active_batch = active_batch
        self.trace_stats["captures"] += 1
        self._restore_trace_inputs(host_inputs, include_page_table=True)
        self._synchronize_device()

    def _refresh_trace_state(self, host_inputs, kv_cache, *, active_batch: int | None = None):
        active_batch = self.batch if active_batch is None else active_batch
        new_page_table = self._page_table_to_torch(host_inputs[3])
        page_shape_changed = (
            self._trace_page_table_snapshot is not None
            and new_page_table.shape != self._trace_page_table_snapshot.shape
        )
        if self._trace_model_id is not None and (kv_cache is not self._trace_kv_cache or page_shape_changed):
            self._release_all_traces()
        if self._trace_model_id is None:
            self._capture_decode_traces(host_inputs, kv_cache, active_batch=active_batch)
            return
        page_changed = not torch.equal(new_page_table, self._trace_page_table_snapshot)
        copy_host_to_device(host_inputs[:3], self._trace_inputs[:3])
        self._trace_active_batch = active_batch
        self.trace_stats["token_host_copies"] += 1
        self.trace_stats["position_host_copies"] += 2
        if page_changed:
            ttnn.copy_host_to_device_tensor(host_inputs[3], self._trace_inputs[3])
            self._trace_page_table_snapshot = new_page_table.clone()
            self.trace_stats["page_table_host_copies"] += 1

    def _refresh_persistent_page_table(self, page_table, kv_cache, *, active_batch: int) -> None:
        """Update a captured page-table buffer only when its host contents change."""

        if self._trace_model_id is None:
            raise RuntimeError("decode trace is not initialized; provide tokens and start_pos first")
        if kv_cache is not self._trace_kv_cache:
            self._release_all_traces()
            raise RuntimeError("KV-cache identity changed; initialize a new decode trace")
        if active_batch != self._trace_active_batch:
            raise RuntimeError("active_batch changed; provide explicit tokens and start_pos to refresh fixed slots")
        if page_table is None:
            return
        new_page_table = self._normalise_page_table(page_table, active_batch)
        if new_page_table.shape != self._trace_page_table_snapshot.shape:
            self._release_all_traces()
            raise RuntimeError("page-table shape changed; initialize a new decode trace")
        if torch.equal(new_page_table, self._trace_page_table_snapshot):
            return
        host_page_table = ttnn.from_torch(
            new_page_table,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host_page_table, self._trace_inputs[3])
        self._trace_page_table_snapshot = new_page_table.clone()
        self.trace_stats["page_table_host_copies"] += 1

    def _replay_split_sampling(self):
        ttnn.execute_trace(self.mesh_device, self._trace_model_id, cq_id=0, blocking=False)
        ttnn.execute_trace(self.mesh_device, self._trace_sampling_id, cq_id=0, blocking=False)
        self.trace_stats["replays"] += 1
        return self._trace_sampled

    def _release_decode_traces(self) -> None:
        released = self._trace_model_id is not None or self._trace_sampling_id is not None
        for trace_id in (self._trace_model_id, self._trace_sampling_id):
            if trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace_id)
        if released:
            self.trace_stats["releases"] += 1
        self._trace_model_id = None
        self._trace_sampling_id = None
        self._trace_inputs = None
        self._trace_logits = None
        self._trace_sampled = None
        self._trace_kv_cache = None
        self._trace_page_table_snapshot = None
        self._trace_active_batch = None

    def _release_prefill_traces(self) -> None:
        for trace_id in self._prefill_trace_ids.values():
            ttnn.release_trace(self.mesh_device, trace_id)
            self.trace_stats["prefill_releases"] += 1
        self._prefill_trace_ids.clear()

    def _release_all_traces(self) -> None:
        self._release_decode_traces()
        self._release_prefill_traces()

    def _copy_forced_tokens(self, tokens: torch.Tensor):
        tokens = tokens.reshape(-1).to(torch.int64)
        if tokens.numel() != self._trace_active_batch:
            raise ValueError(f"expected {self._trace_active_batch} forced tokens, got {tokens.numel()}")
        host_token = torch.zeros(32, dtype=torch.int32)
        host_token[: tokens.numel()] = tokens.to(torch.int32)
        host = ttnn.from_torch(
            host_token.reshape(1, 1, 1, 32),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, self._trace_inputs[0])
        self.trace_stats["token_host_copies"] += 1

    def _copy_forced_token(self, token: int):
        """Compatibility helper for existing teacher-forcing evidence."""

        self._copy_forced_tokens(torch.tensor([token]))

    def decode_forward(
        self,
        tokens: torch.Tensor | None,
        start_pos: torch.Tensor | None,
        *,
        page_table,
        kv_cache: Any,
        sampling_mode: str = "host",
        enable_trace: bool = False,
        active_batch: int | None = None,
        **kwargs: Any,
    ):
        """Decode once or replay persistent device token/position state.

        The first traced call supplies host ``tokens`` and ``start_pos``. Later
        device calls may pass both as ``None``; the captured sampler feeds its
        token directly back and the captured model advances positions in place.
        A non-``None`` token with ``start_pos=None`` is the explicit
        teacher-forcing compatibility boundary.
        """

        if sampling_mode not in {"host", "device"}:
            raise ValueError("sampling_mode must be 'host' or 'device'")
        if sampling_mode == "host" and (self._trace_model_id is not None or self._prefill_trace_ids):
            self._release_all_traces()
        caches = self._ensure_kv_cache() if kv_cache is None else kv_cache

        if tokens is None:
            inferred_batch = self._trace_active_batch
        else:
            inferred_batch = tokens.numel()
        active_batch = inferred_batch if active_batch is None else int(active_batch)
        if active_batch is None or active_batch < 1 or active_batch > self.batch:
            raise ValueError(f"active_batch must be in [1, {self.batch}]")
        if tokens is not None and tokens.numel() != active_batch:
            raise ValueError(f"tokens contain {tokens.numel()} rows, active_batch={active_batch}")
        if start_pos is not None and start_pos.numel() != active_batch:
            raise ValueError(f"start_pos contains {start_pos.numel()} rows, active_batch={active_batch}")
        if sampling_mode == "device" and self._sampling_stochastic and not self._sampling_request_active:
            self.begin_sampling_request(active_batch=active_batch)

        if enable_trace and sampling_mode == "device":
            if start_pos is not None:
                if tokens is None or page_table is None:
                    raise ValueError("initial trace state requires tokens, start_pos, and page_table")
                page_host = self._normalise_page_table(page_table, active_batch)
                self._validate_decode_page_coverage(page_host, start_pos, active_batch=active_batch)
                host_inputs = self._prepare_decode_host_inputs(tokens, start_pos, page_host)
                self._refresh_trace_state(host_inputs, caches, active_batch=active_batch)
            else:
                self._refresh_persistent_page_table(page_table, caches, active_batch=active_batch)
                if tokens is not None:
                    self._copy_forced_tokens(tokens)
            sampled = self._replay_split_sampling()
            self._transition_sampling_seed_to_device_advance()
            return sampled

        if tokens is None or start_pos is None or page_table is None:
            raise ValueError("eager or host decode requires tokens, start_pos, and page_table")
        page_host = self._normalise_page_table(page_table, active_batch)
        self._validate_decode_page_coverage(page_host, start_pos, active_batch=active_batch)
        host_inputs = self._prepare_decode_host_inputs(tokens, start_pos, page_host)

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        logits = self.model.decode_forward(
            device_inputs[0],
            device_inputs[1],
            device_inputs[2],
            page_table=device_inputs[3],
            kv_cache=caches,
        )
        if sampling_mode == "device":
            k, p, temp = self._ensure_sampling_params()
            sampled = self._sample_device_split(logits, k=k, p=p, temp=temp)
            self._transition_sampling_seed_to_device_advance()
            return sampled
        return self._local_logits_to_torch(logits)[0, 0, : self.batch, :]

    def _prefill_single_device(self, prompt: list[int], page_host: torch.Tensor, kv_cache):
        prompt_len = len(prompt)
        padded_width = max(128, _round_up(prompt_len, 128))
        page_device = self._page_table_to_device(page_host)
        prompt_lens = [prompt_len] + [0] * (self.batch - 1)
        final_logits = None
        for start in range(0, padded_width, self.model.config.prefill_chunk_size):
            chunk_len = min(self.model.config.prefill_chunk_size, padded_width - start)
            host = torch.zeros((self.batch, chunk_len), dtype=torch.long)
            source = prompt[start : min(prompt_len, start + chunk_len)]
            if source:
                host[0, : len(source)] = torch.tensor(source, dtype=torch.long)
            chunk_table = None
            if start:
                chunk_table = self._page_table_to_device(self._chunk_page_table(page_host, start, chunk_len))
            is_final = start < prompt_len <= start + chunk_len
            output = self.model.prefill_forward(
                self._tokens_to_device(host),
                page_table=page_device,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                chunk_start_idx=start,
                chunk_page_table=chunk_table,
                return_hidden=True,
            )
            if is_final:
                offset = prompt_len - start - 1
                final_logits = self.model.prefill_last_token_logits(
                    output,
                    offset,
                    fixed_sampling_rows=True,
                )
        if final_logits is None:
            raise RuntimeError("prefill did not produce final logits")
        return final_logits

    def _generate_host_compat(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn],
    ) -> list[int]:
        # Eager host-compatibility calls allocate temporary device buffers.
        # TTNN forbids doing that while captured traces reserve allocator state.
        self._release_all_traces()
        kv_cache = self._ensure_kv_cache()
        processed_position_horizon = len(prompt_token_ids) + max_new_tokens - 1
        padded_prefill = max(128, _round_up(len(prompt_token_ids), 128))
        page_host = self._make_page_table([max(_round_up(processed_position_horizon, 64), padded_prefill)])
        logits = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=page_host[:1],
            kv_cache=kv_cache,
            prompt_lens=[len(prompt_token_ids)],
        )
        predicted = int(logits[0, 0].argmax().item())
        outputs = []
        next_token = predicted
        for step in range(max_new_tokens):
            outputs.append(predicted)
            next_token = next_input(step, predicted) if next_input is not None else predicted
            if step + 1 == max_new_tokens:
                break
            decoded = self.decode_forward(
                torch.tensor([[next_token]]),
                torch.tensor([len(prompt_token_ids) + step]),
                page_table=page_host[:1],
                kv_cache=kv_cache,
                sampling_mode="host",
                enable_trace=False,
            )
            predicted = int(decoded[0].argmax().item())
        return outputs

    def generate(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        enable_trace: bool = True,
        sampling_mode: str = "device",
        stop_on_eos: bool = False,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seed=None,
        **kwargs: Any,
    ) -> list[int]:
        """Generate with canonical model trace + split-sampler trace feedback."""

        if not prompt_token_ids or max_new_tokens < 1:
            return []
        # Prefill produces the first output token without a cache write. Only
        # the remaining decode steps consume new cache positions, so a
        # full-context prompt is valid when exactly one token is requested.
        processed_position_horizon = len(prompt_token_ids) + max_new_tokens - 1
        if processed_position_horizon > self.model.config.max_context_len:
            raise ValueError("prompt plus requested output exceeds the supported context")
        if sampling_mode == "host":
            return self._generate_host_compat(
                prompt_token_ids,
                max_new_tokens,
                next_input=next_input,
            )
        if sampling_mode != "device":
            raise ValueError("sampling_mode must be 'device' or 'host'")
        if not enable_trace and max_new_tokens > 1:
            raise ValueError("the optimized token-out device path requires enable_trace=True")
        self.set_sampling_params(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            active_batch=1,
        )
        kv_cache = self._ensure_kv_cache()
        padded_prefill = max(128, _round_up(len(prompt_token_ids), 128))
        page_host = self._make_page_table([max(_round_up(processed_position_horizon, 64), padded_prefill)])
        sampled = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=page_host,
            kv_cache=kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            sampling_mode="device",
            enable_trace=enable_trace,
            seed=seed,
        )
        predicted = int(self._sampled_tokens_to_torch(sampled)[0].item())
        outputs = []

        for step in range(max_new_tokens):
            outputs.append(predicted)
            forced_or_predicted = next_input(step, predicted) if next_input is not None else predicted
            if step + 1 == max_new_tokens:
                break
            if stop_on_eos and next_input is None and predicted == self.tokenizer.eos_token_id:
                break

            initial_decode = step == 0
            sampled = self.decode_forward(
                torch.tensor([[forced_or_predicted]]) if initial_decode or next_input is not None else None,
                torch.tensor([len(prompt_token_ids)]) if initial_decode else None,
                page_table=page_host if initial_decode else None,
                kv_cache=kv_cache,
                sampling_mode="device",
                enable_trace=True,
                active_batch=1,
            )
            predicted = int(self._sampled_tokens_to_torch(sampled)[0].item())
        self._page_table_host = page_host
        return outputs

    def reset(self) -> None:
        self._sampling_seed_transition_pending = False
        self._sampling_request_active = False
        self._sampling_request_seed_values = None
        if self._kv_cache is not None:
            self.model.reset_kv_cache(self._kv_cache)
            self._synchronize_device()
        self._page_table_host = None

    def teardown(self) -> None:
        """The readiness runner owns the mesh; release only generator traces."""

        self._release_all_traces()


def _resolve_snapshot(model_path: str | Path | None = None) -> Path:
    if model_path is not None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshot_root = hf_home / "models--meta-llama--Llama-3.1-8B-Instruct" / "snapshots"
    if snapshot_root.is_dir():
        complete = [
            candidate
            for candidate in snapshot_root.iterdir()
            if (candidate / "config.json").exists()
            and (candidate / "tokenizer.json").exists()
            and (candidate / "model.safetensors.index.json").exists()
            and all((candidate / f"model-{idx:05d}-of-00004.safetensors").exists() for idx in range(1, 5))
        ]
        if len(complete) == 1:
            return complete[0]
        if len(complete) > 1:
            return max(complete, key=lambda path: path.stat().st_mtime)
    return Path(snapshot_download(MODEL_ID, local_files_only=True))


def _resolve_dtype(value, *, default):
    if value is None:
        return default
    if value in (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.bfloat4_b):
        return value
    names = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16, "bfp4": ttnn.bfloat4_b}
    try:
        return names[str(value).lower()]
    except KeyError as error:
        raise ValueError(f"unknown TT dtype {value!r}") from error


def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> Generator:
    """Readiness discovery factory; builds no vLLM state."""

    snapshot = _resolve_snapshot(kwargs.pop("model_path", os.getenv("LLAMA_31_8B_MODEL_PATH")))
    hf_config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    state_dict = SafetensorStateDict(snapshot)
    config = FullModelConfig(
        max_batch_size=int(kwargs.pop("max_batch_size", 1)),
        max_context_len=int(kwargs.pop("max_context_len", hf_config.max_position_embeddings)),
        num_blocks=int(kwargs.pop("num_blocks", DEFAULT_NUM_BLOCKS)),
        prefill_chunk_size=int(kwargs.pop("prefill_chunk_size", 2048)),
        kv_cache_dtype=_resolve_dtype(kwargs.pop("kv_cache_dtype", None), default=ttnn.bfloat8_b),
        lm_head_weight_dtype=_resolve_dtype(kwargs.pop("lm_head_weight_dtype", None), default=ttnn.bfloat8_b),
        override_num_layers=kwargs.pop("override_num_layers", None),
    )
    if kwargs:
        raise TypeError(f"unsupported build_generator kwargs: {sorted(kwargs)}")
    model = Llama31FullModel.from_state_dict(
        state_dict,
        hf_config=hf_config,
        mesh_device=mesh_device,
        full_model_config=config,
    )
    return Llama31Generator(model, tokenizer)


__all__ = ["Llama31Generator", "SafetensorStateDict", "build_generator"]
