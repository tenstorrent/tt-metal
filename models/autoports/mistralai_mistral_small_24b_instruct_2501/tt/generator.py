# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Metal-readiness generator for Mistral-Small-24B-Instruct-2501."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.model import (
    PAGED_BLOCK_SIZE,
    FullModelConfig,
    MistralSmall24BFullModel,
)
from models.common.readiness_check.contract import Generator, NextInputFn
from models.tt_transformers.tt.common import copy_host_to_device

MODEL_ID = "mistralai/Mistral-Small-24B-Instruct-2501"


class SafetensorStateDict(Mapping[str, torch.Tensor]):
    """Lazy bounded-memory view over a sharded safetensors checkpoint."""

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
    shards = ttnn.get_device_tensors(tensor)
    return ttnn.to_torch(shards[0] if shards else tensor)


class MistralSmall24BGenerator(Generator):
    """Explicit-cache API plus canonical traced model/sampler token feedback."""

    def __init__(self, model: MistralSmall24BFullModel, tokenizer):
        self.model = model
        self.mesh_device = model.mesh_device
        self.tokenizer = tokenizer
        self.batch = model.batch
        self.blocks_per_slot = math.ceil(model.config.max_context_len / PAGED_BLOCK_SIZE)

        self._kv_cache = None
        self._page_table_host: torch.Tensor | None = None
        self._trace_model_id: int | None = None
        self._trace_sampling_id: int | None = None
        self._trace_logits = None
        self._trace_sampled = None
        self._trace_kv_cache = None
        self._trace_page_table_snapshot: torch.Tensor | None = None
        self._trace_active_batch: int | None = None
        self._prefill_weights_released = False
        self.last_generate_stats: dict[str, Any] = {}
        self.trace_stats = {
            "captures": 0,
            "model_replays": 0,
            "sampling_replays": 0,
            "releases": 0,
            "decode_warmups": 0,
            "token_host_copies": 0,
            "position_host_copies": 0,
            "page_table_host_copies": 0,
            "sampling_param_host_copies": 0,
            "caller_token_readbacks": 0,
            "full_logit_readbacks": 0,
            "explicit_synchronizations": 0,
        }
        self._allocate_persistent_inputs()

    def _ensure_kv_cache(self):
        if self._kv_cache is None:
            self._kv_cache = self.model.allocate_kv_cache()
        return self._kv_cache

    def _synchronize(self) -> None:
        ttnn.synchronize_device(self.mesh_device)
        self.trace_stats["explicit_synchronizations"] += 1

    def _host_tt(self, tensor: torch.Tensor, *, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            tensor,
            device=None,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _device_tt(self, tensor: torch.Tensor, *, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            tensor,
            device=self.mesh_device,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _allocate_persistent_inputs(self) -> None:
        self._trace_token = self._device_tt(
            torch.zeros((1, 1, 1, 32), dtype=torch.int32),
            dtype=ttnn.uint32,
        )
        self._trace_current_pos = self._device_tt(
            torch.full((self.batch,), -1, dtype=torch.int32),
            dtype=ttnn.int32,
        )
        self._trace_rotary_pos = self._device_tt(
            torch.zeros((1, self.batch), dtype=torch.int32),
            dtype=ttnn.uint32,
        )
        self._trace_page_table = self._device_tt(
            self._default_page_table(),
            dtype=ttnn.int32,
        )
        self._sampling_k = self._device_tt(torch.ones(32, dtype=torch.int32), dtype=ttnn.uint32)
        self._sampling_p = self._device_tt(torch.zeros(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16)
        self._sampling_temp = self._device_tt(torch.ones(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16)
        self._sampling_snapshot = (tuple([1] * 32), tuple([0.0] * 32), tuple([1.0] * 32))

    def _default_page_table(self) -> torch.Tensor:
        table = torch.empty((self.batch, self.blocks_per_slot), dtype=torch.int32)
        for slot in range(self.batch):
            start = slot * self.blocks_per_slot
            table[slot] = torch.arange(start, start + self.blocks_per_slot, dtype=torch.int32)
        return table

    def _page_table_to_torch(self, page_table) -> torch.Tensor:
        if isinstance(page_table, torch.Tensor):
            result = page_table.detach().cpu().to(torch.int32)
        elif isinstance(page_table, ttnn.Tensor):
            result = _first_device_to_torch(page_table).to(torch.int32)
        else:
            raise TypeError("page_table must be a torch or TTNN tensor")
        if result.ndim != 2:
            raise ValueError(f"page_table must be rank two, got {tuple(result.shape)}")
        return result.contiguous()

    def _normalise_page_table(self, page_table, active_batch: int) -> torch.Tensor:
        source = self._page_table_to_torch(page_table)
        if source.shape[0] < active_batch or source.shape[0] > self.batch:
            raise ValueError("page table row count does not match the configured fixed slots")
        if source.shape[1] > self.blocks_per_slot:
            raise ValueError(f"page table has {source.shape[1]} columns, maximum is {self.blocks_per_slot}")
        result = torch.full((self.batch, self.blocks_per_slot), -1, dtype=torch.int32)
        result[: source.shape[0], : source.shape[1]] = source
        supplied = result[result >= 0].to(torch.int64)
        if torch.any(supplied >= self.model.config.num_blocks):
            raise ValueError("page-table entries must name an allocated physical block")
        if supplied.unique().numel() != supplied.numel():
            raise ValueError("physical cache blocks cannot be aliased between fixed slots")
        occupied = set(supplied.tolist())
        free_blocks = iter(block for block in range(self.model.config.num_blocks) if block not in occupied)
        for row, column in (result < 0).nonzero(as_tuple=False).tolist():
            try:
                result[row, column] = next(free_blocks)
            except StopIteration as error:
                raise ValueError("page table plus fixed inactive slots exceed allocated cache blocks") from error
        return result.contiguous()

    def _make_page_table(self) -> torch.Tensor:
        return self._default_page_table()

    def _page_table_device(self, host: torch.Tensor):
        return self._device_tt(host, dtype=ttnn.int32)

    def _tokens_device(self, host: torch.Tensor):
        return self._device_tt(host.to(torch.int32), dtype=ttnn.uint32)

    def _positions_device(self, positions: torch.Tensor):
        positions = positions.reshape(-1).to(torch.int32)
        if positions.numel() > self.batch:
            raise ValueError("position batch exceeds configured fixed slots")
        padded = torch.full((self.batch,), -1, dtype=torch.int32)
        padded[: positions.numel()] = positions
        rotary = torch.clamp(padded, min=0).reshape(1, self.batch)
        return (
            self._device_tt(padded, dtype=ttnn.int32),
            self._device_tt(rotary, dtype=ttnn.uint32),
        )

    def _decode_token_host(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.reshape(-1).to(torch.int32)
        if flat.numel() > self.batch:
            raise ValueError("decode token batch exceeds configured fixed slots")
        padded = torch.zeros(32, dtype=torch.int32)
        padded[: flat.numel()] = flat
        return padded.reshape(1, 1, 1, 32)

    def _local_logits_to_torch(self, logits) -> torch.Tensor:
        logits = ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        host = ttnn.to_torch(
            logits,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3),
        )
        self.trace_stats["full_logit_readbacks"] += 1
        return host[..., : self.model.vocab_size].float()

    def _sampled_tokens_to_torch(self, tokens) -> torch.Tensor:
        self.trace_stats["caller_token_readbacks"] += 1
        return _first_device_to_torch(tokens).reshape(-1)[: self.batch].to(torch.long)

    def set_sampling_params(
        self,
        *,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        active_batch: int = 1,
    ) -> None:
        """Update trace-stable common-sampler tensors at a request boundary."""

        if active_batch < 1 or active_batch > self.batch:
            raise ValueError(f"active_batch must be in [1, {self.batch}]")

        def expand(value, inactive):
            if isinstance(value, (int, float)):
                values = [value] * active_batch
            else:
                values = list(value)
                if len(values) != active_batch:
                    raise ValueError("per-user sampling parameters must match active_batch")
            return values + [inactive] * (32 - active_batch)

        k_values = [int(value) for value in expand(top_k, 1)]
        p_values = [float(value) for value in expand(top_p, 0.0)]
        temperatures = [float(value) for value in expand(temperature, 0.0)]
        if any(value < 1 or value > self.model.sampler.config.max_top_k for value in k_values[:active_batch]):
            raise ValueError(f"top_k must be in [1, {self.model.sampler.config.max_top_k}]")
        if any(value < 0.0 or value > 1.0 for value in p_values[:active_batch]):
            raise ValueError("top_p must be in [0, 1]")
        if any(value < 0.0 for value in temperatures[:active_batch]):
            raise ValueError("temperature must be non-negative")
        inverse_temperatures = []
        for slot, value in enumerate(temperatures):
            if value == 0.0:
                k_values[slot] = 1
                p_values[slot] = 0.0
                inverse_temperatures.append(1.0)
            else:
                inverse_temperatures.append(1.0 / value)
        snapshot = (tuple(k_values), tuple(p_values), tuple(inverse_temperatures))
        if snapshot == self._sampling_snapshot:
            return
        host = (
            self._host_tt(torch.tensor(k_values, dtype=torch.int32), dtype=ttnn.uint32),
            self._host_tt(torch.tensor(p_values, dtype=torch.bfloat16), dtype=ttnn.bfloat16),
            self._host_tt(torch.tensor(inverse_temperatures, dtype=torch.bfloat16), dtype=ttnn.bfloat16),
        )
        copy_host_to_device(host, (self._sampling_k, self._sampling_p, self._sampling_temp))
        self._sampling_snapshot = snapshot
        self.trace_stats["sampling_param_host_copies"] += 3

    def _validate_prompt_inputs(self, tokens: torch.Tensor, prompt_lens: list[int]):
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [batch, seq], got {tuple(tokens.shape)}")
        active_batch, logical_width = tokens.shape
        if active_batch < 1 or active_batch > self.batch:
            raise ValueError(f"batch {active_batch} exceeds configured batch {self.batch}")
        if len(prompt_lens) != active_batch:
            raise ValueError("prompt_lens must contain one length per input row")
        if any(length < 1 or length > logical_width for length in prompt_lens):
            raise ValueError("prompt_lens contains an invalid logical length")
        if max(prompt_lens) > self.model.config.max_context_len:
            raise ValueError("prompt exceeds the supported context")
        if self._prefill_weights_released:
            raise RuntimeError("prefill weights were explicitly released for decode-only capacity")
        return active_batch, logical_width

    def _run_initial_prefill(
        self,
        tokens: torch.Tensor,
        *,
        page_device,
        kv_cache,
        prompt_lens: list[int],
    ):
        active_batch, logical_width = tokens.shape
        initial_logical = min(max(prompt_lens), self.model.config.prefill_chunk_size)
        initial_padded = _round_up(initial_logical, 32)
        host = torch.zeros((self.batch, initial_padded), dtype=torch.long)
        copy_width = min(logical_width, initial_logical)
        host[:active_batch, :copy_width] = tokens[:, :copy_width]
        hidden = self.model.prefill_forward(
            self._tokens_device(host),
            page_table=page_device,
            kv_cache=kv_cache,
            return_hidden=True,
        )
        return hidden, initial_logical, initial_padded

    def _run_suffix_decode(
        self,
        tokens: torch.Tensor,
        *,
        start: int,
        page_device,
        kv_cache,
        prompt_lens: list[int],
        collect_all_logits: bool,
    ):
        active_batch = tokens.shape[0]
        final_logits = [None] * active_batch
        per_position = []
        last_device_logits = None
        for position in range(start, max(prompt_lens)):
            step_tokens = torch.zeros((self.batch,), dtype=torch.long)
            positions = torch.full((self.batch,), -1, dtype=torch.long)
            for user, length in enumerate(prompt_lens):
                if position < length:
                    step_tokens[user] = tokens[user, position]
                    positions[user] = position
            token_device = self._tokens_device(self._decode_token_host(step_tokens))
            current_pos, rotary_pos = self._positions_device(positions)
            last_device_logits = self.model.decode_forward(
                token_device,
                current_pos,
                rotary_pos,
                page_table=page_device,
                kv_cache=kv_cache,
                advance_positions=False,
            )
            need_host = collect_all_logits or any(length - 1 == position for length in prompt_lens)
            host_logits = None
            if need_host:
                host_logits = self._local_logits_to_torch(last_device_logits)[0, 0, :active_batch]
            if collect_all_logits:
                per_position.append(host_logits)
            for user, length in enumerate(prompt_lens):
                if length - 1 == position:
                    final_logits[user] = host_logits[user]
        return last_device_logits, final_logits, per_position

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache: Any,
        prompt_lens: list[int],
        return_all_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Prefill logical prompts; internal padding and long-prompt streaming are private."""

        active_batch, logical_width = self._validate_prompt_inputs(tokens, prompt_lens)
        caches = self._ensure_kv_cache() if kv_cache is None else kv_cache
        page_host = self._normalise_page_table(page_table, active_batch)
        page_device = self._page_table_device(page_host)
        hidden, initial_logical, _ = self._run_initial_prefill(
            tokens,
            page_device=page_device,
            kv_cache=caches,
            prompt_lens=prompt_lens,
        )

        initial_logits = self.model.prefill_hidden_logits(hidden)
        initial_host = self._local_logits_to_torch(initial_logits)[0, :active_batch, :initial_logical]
        final_logits = [None] * active_batch
        for user, length in enumerate(prompt_lens):
            if length <= initial_logical:
                final_logits[user] = initial_host[user, length - 1]

        suffix_rows = []
        if max(prompt_lens) > initial_logical:
            _, suffix_final, suffix_rows = self._run_suffix_decode(
                tokens,
                start=initial_logical,
                page_device=page_device,
                kv_cache=caches,
                prompt_lens=prompt_lens,
                collect_all_logits=return_all_logits,
            )
            for user, row in enumerate(suffix_final):
                if row is not None:
                    final_logits[user] = row
        self._page_table_host = page_host

        if return_all_logits:
            output = torch.zeros(
                (active_batch, logical_width, self.model.vocab_size),
                dtype=initial_host.dtype,
            )
            output[:, :initial_logical] = initial_host
            for offset, rows in enumerate(suffix_rows):
                output[:, initial_logical + offset] = rows
            return output
        if any(row is None for row in final_logits):
            raise RuntimeError("failed to select every prompt's final logits")
        return torch.stack(final_logits, dim=0).unsqueeze(1)

    def _copy_trace_state(
        self,
        *,
        tokens: torch.Tensor | None,
        positions: torch.Tensor | None,
        page_host: torch.Tensor | None,
    ) -> None:
        if tokens is not None:
            host_token = self._host_tt(self._decode_token_host(tokens), dtype=ttnn.uint32)
            ttnn.copy_host_to_device_tensor(host_token, self._trace_token)
            self.trace_stats["token_host_copies"] += 1
        if positions is not None:
            positions = positions.reshape(-1).to(torch.int32)
            padded = torch.full((self.batch,), -1, dtype=torch.int32)
            padded[: positions.numel()] = positions
            rotary = torch.clamp(padded, min=0).reshape(1, self.batch)
            host_pos = self._host_tt(padded, dtype=ttnn.int32)
            host_rotary = self._host_tt(rotary, dtype=ttnn.uint32)
            copy_host_to_device(
                (host_pos, host_rotary),
                (self._trace_current_pos, self._trace_rotary_pos),
            )
            self.trace_stats["position_host_copies"] += 2
        if page_host is not None and (
            self._trace_page_table_snapshot is None or not torch.equal(page_host, self._trace_page_table_snapshot)
        ):
            host_page = self._host_tt(page_host, dtype=ttnn.int32)
            ttnn.copy_host_to_device_tensor(host_page, self._trace_page_table)
            self._trace_page_table_snapshot = page_host.clone()
            self.trace_stats["page_table_host_copies"] += 1

    def _capture_decode_traces(
        self,
        kv_cache,
        page_host: torch.Tensor,
        *,
        active_batch: int,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        self.model.sampler.load_device_buffers()
        self._copy_trace_state(
            tokens=tokens,
            positions=positions,
            page_host=page_host,
        )
        warm_logits = self.model.decode_forward(
            self._trace_token,
            self._trace_current_pos,
            self._trace_rotary_pos,
            page_table=self._trace_page_table,
            kv_cache=kv_cache,
        )
        self.model.sample_split(
            warm_logits,
            k=self._sampling_k,
            p=self._sampling_p,
            temp=self._sampling_temp,
            tt_out_tok=self._trace_token,
        )
        self._synchronize()
        self.trace_stats["decode_warmups"] += 1
        self._copy_trace_state(
            tokens=tokens,
            positions=positions,
            page_host=page_host,
        )
        self._synchronize()

        model_trace = None
        sampling_trace = None
        model_open = False
        sampling_open = False
        self.mesh_device.set_program_cache_misses_allowed(False)
        try:
            model_trace = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            model_open = True
            logits = self.model.decode_forward(
                self._trace_token,
                self._trace_current_pos,
                self._trace_rotary_pos,
                page_table=self._trace_page_table,
                kv_cache=kv_cache,
            )
            ttnn.end_trace_capture(self.mesh_device, model_trace, cq_id=0)
            model_open = False
            self._synchronize()

            sampling_trace = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            sampling_open = True
            sampled = self.model.sample_split(
                logits,
                k=self._sampling_k,
                p=self._sampling_p,
                temp=self._sampling_temp,
                tt_out_tok=self._trace_token,
            )
            ttnn.end_trace_capture(self.mesh_device, sampling_trace, cq_id=0)
            sampling_open = False
            self._synchronize()
        except Exception:
            if sampling_open:
                try:
                    ttnn.end_trace_capture(self.mesh_device, sampling_trace, cq_id=0)
                except Exception:
                    pass
            if model_open:
                try:
                    ttnn.end_trace_capture(self.mesh_device, model_trace, cq_id=0)
                except Exception:
                    pass
            for trace_id in (sampling_trace, model_trace):
                if trace_id is not None:
                    try:
                        ttnn.release_trace(self.mesh_device, trace_id)
                    except Exception:
                        pass
            raise
        finally:
            self.mesh_device.set_program_cache_misses_allowed(True)

        self._trace_model_id = model_trace
        self._trace_sampling_id = sampling_trace
        self._trace_logits = logits
        self._trace_sampled = sampled
        self._trace_kv_cache = kv_cache
        self._trace_active_batch = active_batch
        self.trace_stats["captures"] += 1
        self._copy_trace_state(
            tokens=tokens,
            positions=positions,
            page_host=page_host,
        )
        self._synchronize()

    def _ensure_decode_traces(
        self,
        kv_cache,
        page_host: torch.Tensor,
        *,
        active_batch: int,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if self._trace_model_id is not None and (
            kv_cache is not self._trace_kv_cache or active_batch != self._trace_active_batch
        ):
            self._release_decode_traces()
        if self._trace_model_id is None:
            self._capture_decode_traces(
                kv_cache,
                page_host,
                active_batch=active_batch,
                tokens=tokens,
                positions=positions,
            )
        else:
            self._copy_trace_state(tokens=None, positions=None, page_host=page_host)

    def _replay_split_sampling(self):
        if self._trace_model_id is None or self._trace_sampling_id is None:
            raise RuntimeError("decode traces are not initialized")
        ttnn.execute_trace(self.mesh_device, self._trace_model_id, cq_id=0, blocking=False)
        self.trace_stats["model_replays"] += 1
        ttnn.execute_trace(self.mesh_device, self._trace_sampling_id, cq_id=0, blocking=False)
        self.trace_stats["sampling_replays"] += 1
        return self._trace_sampled

    def _release_decode_traces(self) -> None:
        released = False
        for trace_id in (self._trace_model_id, self._trace_sampling_id):
            if trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace_id)
                released = True
        if released:
            self.trace_stats["releases"] += 1
        self._trace_model_id = None
        self._trace_sampling_id = None
        self._trace_logits = None
        self._trace_sampled = None
        self._trace_kv_cache = None
        self._trace_active_batch = None

    def _copy_forced_tokens(self, tokens: torch.Tensor) -> None:
        self._copy_trace_state(tokens=tokens, positions=None, page_host=None)

    def _validate_page_coverage(self, page_host: torch.Tensor, positions: torch.Tensor) -> None:
        for slot, value in enumerate(positions.reshape(-1).tolist()):
            if value < 0:
                continue
            page = int(value) // PAGED_BLOCK_SIZE
            if page >= page_host.shape[1] or int(page_host[slot, page]) < 0:
                raise ValueError(f"slot {slot} has no page for decode position {value}")

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table,
        kv_cache: Any,
        sampling_mode: str = "host",
        enable_trace: bool = True,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        **kwargs: Any,
    ):
        """Decode with explicit token/position/page/cache state."""

        if tokens.ndim not in (1, 2) or tokens.reshape(-1).numel() > self.batch:
            raise ValueError("tokens must contain at most one token per configured slot")
        active_batch = tokens.reshape(-1).numel()
        positions = start_pos.reshape(-1)
        if positions.numel() != active_batch:
            raise ValueError("start_pos must contain one entry per token")
        caches = self._ensure_kv_cache() if kv_cache is None else kv_cache
        page_host = self._normalise_page_table(page_table, active_batch)
        self._validate_page_coverage(page_host, positions)

        if sampling_mode == "device":
            if not enable_trace:
                raise ValueError("optimized device sampling requires enable_trace=True")
            self.set_sampling_params(
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                active_batch=active_batch,
            )
            self._ensure_decode_traces(
                caches,
                page_host,
                active_batch=active_batch,
                tokens=tokens,
                positions=positions,
            )
            self._copy_trace_state(tokens=tokens, positions=positions, page_host=page_host)
            return self._replay_split_sampling()
        if sampling_mode != "host":
            raise ValueError("sampling_mode must be 'host' or 'device'")
        if self._trace_model_id is not None:
            self._release_decode_traces()
        token_device = self._tokens_device(self._decode_token_host(tokens))
        current_pos, rotary_pos = self._positions_device(positions)
        logits = self.model.decode_forward(
            token_device,
            current_pos,
            rotary_pos,
            page_table=self._page_table_device(page_host),
            kv_cache=caches,
            advance_positions=False,
        )
        return self._local_logits_to_torch(logits)[0, 0, :active_batch]

    def _prefill_device_sample(
        self,
        prompt_tokens: torch.Tensor,
        *,
        page_host: torch.Tensor,
        kv_cache,
    ):
        prompt_lens = [int(prompt_tokens.shape[1])]
        page_device = self._page_table_device(page_host)
        hidden, initial_logical, _ = self._run_initial_prefill(
            prompt_tokens,
            page_device=page_device,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
        )
        if prompt_lens[0] <= initial_logical:
            selected = self.model.select_prefill_token_hidden(hidden, 0, prompt_lens[0] - 1)
            logits = self.model.prefill_selected_hidden_logits([selected], fixed_sampling_rows=True)
        else:
            logits, _, _ = self._run_suffix_decode(
                prompt_tokens,
                start=initial_logical,
                page_device=page_device,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                collect_all_logits=False,
            )
        return self.model.sample_split(
            logits,
            k=self._sampling_k,
            p=self._sampling_p,
            temp=self._sampling_temp,
            tt_out_tok=self._trace_token,
        )

    def _generate_host_compat(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn],
    ) -> list[int]:
        kv_cache = self._ensure_kv_cache()
        if self._trace_model_id is not None:
            self._release_decode_traces()
        self.model.reset_kv_cache(kv_cache)
        page_host = self._make_page_table()
        logits = self.prefill_forward(
            torch.tensor([prompt_token_ids], dtype=torch.long),
            page_table=page_host[:1],
            kv_cache=kv_cache,
            prompt_lens=[len(prompt_token_ids)],
        )
        predicted = int(logits[0, 0].argmax().item())
        outputs = []
        for step in range(max_new_tokens):
            outputs.append(predicted)
            next_token = next_input(step, predicted) if next_input is not None else predicted
            if step + 1 == max_new_tokens:
                break
            decoded = self.decode_forward(
                torch.tensor([[next_token]], dtype=torch.long),
                torch.tensor([len(prompt_token_ids) + step], dtype=torch.long),
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
        **kwargs: Any,
    ) -> list[int]:
        """Generate with traced device token feedback or explicit host compatibility."""

        if not prompt_token_ids or max_new_tokens < 1:
            return []
        horizon = len(prompt_token_ids) + max_new_tokens - 1
        if horizon > self.model.config.max_context_len:
            raise ValueError("prompt plus requested outputs exceeds the supported context")
        if sampling_mode == "host":
            start_s = time.perf_counter()
            outputs = self._generate_host_compat(
                prompt_token_ids,
                max_new_tokens,
                next_input=next_input,
            )
            elapsed_s = time.perf_counter() - start_s
            self.last_generate_stats = {
                "sampling_mode": "host_compatibility",
                "teacher_forcing": next_input is not None,
                "requested_tokens": max_new_tokens,
                "emitted_tokens": len(outputs),
                "elapsed_s": elapsed_s,
                "e2e_t/s/u": len(outputs) / elapsed_s if elapsed_s > 0.0 else 0.0,
            }
            return outputs
        if sampling_mode != "device":
            raise ValueError("sampling_mode must be 'host' or 'device'")
        if not enable_trace and max_new_tokens > 1:
            raise ValueError("the optimized token-out path requires enable_trace=True")

        request_start_s = time.perf_counter()
        trace_stats_before = dict(self.trace_stats)
        self.set_sampling_params(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            active_batch=1,
        )
        kv_cache = self._ensure_kv_cache()
        page_host = self._make_page_table()
        # Prefill may allocate shape-dependent outputs, which is unsafe while
        # trace-bound addresses are reserved. A new request therefore drops
        # the prior request's traces only here (never in reset), prefills, and
        # captures decode against the already-filled cache.
        if self._trace_model_id is not None:
            self._release_decode_traces()
        self.model.reset_kv_cache(kv_cache)
        sampled = self._prefill_device_sample(
            torch.tensor([prompt_token_ids], dtype=torch.long),
            page_host=page_host,
            kv_cache=kv_cache,
        )
        predicted = int(self._sampled_tokens_to_torch(sampled)[0].item())
        first_token_s = time.perf_counter()
        outputs = [predicted]
        next_token = next_input(0, predicted) if next_input is not None else predicted
        if max_new_tokens == 1 or (stop_on_eos and next_input is None and predicted == self.tokenizer.eos_token_id):
            self._page_table_host = page_host
            self.last_generate_stats = {
                "sampling_mode": "device_split_trace",
                "teacher_forcing": next_input is not None,
                "token_feedback": "host_forced" if next_input is not None else "device",
                "requested_tokens": max_new_tokens,
                "emitted_tokens": len(outputs),
                "ttft_ms": 1000.0 * (first_token_s - request_start_s),
                "trace_setup_ms": 0.0,
                "traced_decode_tokens": 0,
                "traced_decode_elapsed_s": 0.0,
                "traced_decode_t/s/u": 0.0,
            }
            return outputs

        first_decode_token = torch.tensor([next_token], dtype=torch.long)
        first_decode_position = torch.tensor([len(prompt_token_ids)], dtype=torch.long)
        trace_setup_start_s = time.perf_counter()
        self._ensure_decode_traces(
            kv_cache,
            page_host,
            active_batch=1,
            tokens=first_decode_token,
            positions=first_decode_position,
        )
        trace_setup_end_s = time.perf_counter()
        decode_start_s = trace_setup_end_s
        for step in range(1, max_new_tokens):
            sampled = self._replay_split_sampling()
            predicted = int(self._sampled_tokens_to_torch(sampled)[0].item())
            outputs.append(predicted)
            next_token = next_input(step, predicted) if next_input is not None else predicted
            if step + 1 == max_new_tokens:
                break
            if stop_on_eos and next_input is None and predicted == self.tokenizer.eos_token_id:
                break
            if next_input is not None:
                self._copy_forced_tokens(torch.tensor([next_token], dtype=torch.long))
        decode_end_s = time.perf_counter()
        self._page_table_host = page_host
        decode_tokens = max(len(outputs) - 1, 0)
        decode_elapsed_s = max(decode_end_s - decode_start_s, 0.0)
        final_device_positions = (
            _first_device_to_torch(self._trace_current_pos).reshape(-1)[: self.batch].to(torch.int32).tolist()
        )
        sampled_address = ttnn.get_device_tensors(self._trace_sampled)[0].buffer_address()
        token_address = ttnn.get_device_tensors(self._trace_token)[0].buffer_address()
        self.last_generate_stats = {
            "sampling_mode": "device_split_trace",
            "teacher_forcing": next_input is not None,
            "token_feedback": "host_forced" if next_input is not None else "device",
            "requested_tokens": max_new_tokens,
            "emitted_tokens": len(outputs),
            "ttft_ms": 1000.0 * (first_token_s - request_start_s),
            "trace_setup_ms": 1000.0 * (trace_setup_end_s - trace_setup_start_s),
            "traced_decode_tokens": decode_tokens,
            "traced_decode_elapsed_s": decode_elapsed_s,
            "traced_decode_t/s/u": decode_tokens / decode_elapsed_s if decode_elapsed_s > 0.0 else 0.0,
            "final_device_positions": final_device_positions,
            "sampled_token_is_feedback_buffer": sampled_address == token_address,
            "trace_stat_delta": {key: self.trace_stats[key] - trace_stats_before[key] for key in self.trace_stats},
        }
        return outputs

    def release_prefill_weights_for_decode(self) -> None:
        """Enter the irreversible maximum-capacity decode residency phase."""

        self.model.release_prefill_weights()
        self._prefill_weights_released = True

    def reset(self) -> None:
        """Clear prompt/cache state while retaining weights and compiled traces."""

        if self._kv_cache is not None:
            self.model.reset_kv_cache(self._kv_cache)
            self._synchronize()
        self._page_table_host = None

    def teardown(self) -> None:
        self._release_decode_traces()


def _resolve_snapshot(snapshot_path: str | Path | None) -> Path:
    if snapshot_path is not None:
        path = Path(snapshot_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    return Path(snapshot_download(MODEL_ID))


def build_generator(model_dir: str | Path, mesh_device, **kwargs: Any) -> Generator:
    """Build the standard readiness generator; no serving integration side effects."""

    snapshot = _resolve_snapshot(kwargs.pop("snapshot_path", None))
    hf_config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, fix_mistral_regex=True)
    max_batch_size = int(kwargs.pop("max_batch_size", 1))
    max_context_len = int(kwargs.pop("max_seq_len", hf_config.max_position_embeddings))
    num_blocks = int(
        kwargs.pop(
            "num_blocks",
            max_batch_size * math.ceil(max_context_len / PAGED_BLOCK_SIZE),
        )
    )
    full_config = FullModelConfig(
        max_batch_size=max_batch_size,
        max_context_len=max_context_len,
        num_blocks=num_blocks,
        prefill_chunk_size=int(kwargs.pop("prefill_chunk_size", 576)),
        kv_cache_dtype=kwargs.pop("kv_cache_dtype", ttnn.bfloat8_b),
        lm_head_weight_dtype=kwargs.pop("lm_head_weight_dtype", ttnn.bfloat16),
        lm_head_math_fidelity=kwargs.pop("lm_head_math_fidelity", ttnn.MathFidelity.HiFi2),
        override_num_layers=kwargs.pop("override_num_layers", None),
    )
    if kwargs:
        raise TypeError(f"unknown build_generator options: {sorted(kwargs)}")
    model = MistralSmall24BFullModel.from_state_dict(
        SafetensorStateDict(snapshot),
        hf_config=hf_config,
        mesh_device=mesh_device,
        full_model_config=full_config,
    )
    return MistralSmall24BGenerator(model, tokenizer)


__all__ = [
    "MODEL_ID",
    "MistralSmall24BGenerator",
    "SafetensorStateDict",
    "build_generator",
]
