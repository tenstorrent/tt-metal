# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Readiness-check generator for the Llama 3.1 8B Instruct full TTNN model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import ttnn
from transformers import AutoTokenizer

from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.model import (
    MODEL_ID,
    TARGET_MESH_SHAPE,
    Llama31_8B_InstructFullModel,
)
from models.common.modules.rope.rope_1d import prepare_rot_idxs
from models.common.readiness_check.contract import Generator as ReadinessGenerator
from models.common.readiness_check.contract import NextInputFn
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params
from models.common.tensor_utils import TILE_SIZE


@dataclass
class DecodeTraceState:
    trace_id: int | None = None
    tokens: ttnn.Tensor | None = None
    hidden_states: ttnn.Tensor | None = None
    current_pos: ttnn.Tensor | None = None
    rot_idxs: ttnn.Tensor | None = None
    rot_mats: tuple[ttnn.Tensor, ttnn.Tensor] | None = None
    page_table: ttnn.Tensor | None = None
    logits: ttnn.Tensor | None = None
    initialized: bool = False


@dataclass
class RuntimeCounters:
    model_trace_captures: int = 0
    model_trace_replays: int = 0
    model_trace_nonblocking_replays: int = 0
    model_trace_syncs_before_sampling_capture: int = 0
    sampling_trace_invocations: int = 0
    token_input_host_copies: int = 0
    position_host_copies: int = 0
    rope_index_host_copies: int = 0
    page_table_host_copies: int = 0
    sampled_token_readbacks: int = 0
    position_device_increments: int = 0
    rope_index_device_increments: int = 0
    rope_matrix_device_refreshes: int = 0
    hidden_device_refreshes: int = 0
    last_page_table_refresh_changed_only: bool = False
    notes: list[str] = field(default_factory=list)


def _nearest_32(value: int) -> int:
    return TILE_SIZE * math.ceil(value / TILE_SIZE)


def _padded_prefill_len(seq_len: int) -> int:
    if seq_len <= 128:
        return 128
    if seq_len <= 1024:
        return 1024
    return 1 << (seq_len - 1).bit_length()


def _copy_host_to_device(
    host_tensors: tuple[ttnn.Tensor | None, ...],
    *,
    mesh_device: ttnn.MeshDevice,
    device_tensors: tuple[ttnn.Tensor | None, ...] | None = None,
) -> tuple[ttnn.Tensor | None, ...]:
    copied: list[ttnn.Tensor | None] = []
    if device_tensors is None:
        for host_tensor in host_tensors:
            if host_tensor is None:
                copied.append(None)
            else:
                copied.append(ttnn.to_device(host_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        return tuple(copied)

    for host_tensor, device_tensor in zip(host_tensors, device_tensors):
        if host_tensor is None or device_tensor is None:
            copied.append(device_tensor)
            continue
        ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
        copied.append(device_tensor)
    return tuple(copied)


def _normal_token_list(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    if isinstance(token_ids, dict) and "input_ids" in token_ids:
        token_ids = token_ids["input_ids"]
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.reshape(-1).tolist()
    return [int(token_id) for token_id in token_ids]


class Llama31_8B_InstructGenerator(ReadinessGenerator):
    """Standard Metal readiness generator with traced split-sampling decode."""

    def __init__(
        self,
        *,
        model_dir: Path,
        mesh_device: ttnn.MeshDevice,
        model: Llama31_8B_InstructFullModel,
        tokenizer: Any,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.mesh_device = mesh_device
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache = self.model.owned_kv_cache()
        self.page_table = self._make_default_page_table()
        self.sampling = SamplingGenerator(
            args=self.model.make_sampling_args(),
            mesh_device=self.mesh_device,
            tt_ccl=self.model.tt_ccl,
            enable_internal_trace=True,
        )
        self.greedy_sampling_params = format_sampling_params(
            SamplingParams(temperature=1.0, top_k=1, top_p=0.0),
            self.sampling.tt_sampling.max_batch_size,
        )
        self.sampling.reset_sampling_params(self.greedy_sampling_params)
        if self.sampling.tt_sampling.force_argmax_sampling:
            raise RuntimeError(
                "Greedy readiness decode must use canonical split sampling; force-argmax unexpectedly activated."
            )

        self._decode_trace = DecodeTraceState()
        self._prev_page_table: torch.Tensor | None = None
        self._decode_host_position: int | None = None
        self._decode_sampling_precompiled = False
        self._counters = RuntimeCounters()

    # ------------------------------------------------------------------
    # Tensor preparation and host conversion
    # ------------------------------------------------------------------

    def _replicate_mapper(self):
        return ttnn.ReplicateTensorToMesh(self.mesh_device)

    def _make_default_page_table(self) -> torch.Tensor:
        max_blocks = int(self.model.full_model_config.max_num_blocks or 1)
        return torch.arange(max_blocks, dtype=torch.int32).reshape(1, max_blocks)

    def _coerce_page_table_torch(self, page_table: torch.Tensor | ttnn.Tensor | None) -> torch.Tensor:
        if page_table is None:
            return self.page_table
        if isinstance(page_table, torch.Tensor):
            if page_table.dim() == 1:
                page_table = page_table.unsqueeze(0)
            return page_table.to(dtype=torch.int32).contiguous()
        if isinstance(page_table, ttnn.Tensor):
            torch_page_table = ttnn.to_torch(ttnn.get_device_tensors(page_table)[0])
            if torch_page_table.dim() == 1:
                torch_page_table = torch_page_table.unsqueeze(0)
            return torch_page_table.to(dtype=torch.int32).contiguous()
        raise TypeError(f"Unsupported page_table type: {type(page_table)!r}")

    def _page_table_to_device(self, page_table: torch.Tensor | ttnn.Tensor | None) -> ttnn.Tensor:
        if isinstance(page_table, ttnn.Tensor):
            return page_table
        page_table_torch = self._coerce_page_table_torch(page_table)
        return ttnn.from_torch(
            page_table_torch,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._replicate_mapper(),
        )

    def _tokens_to_prefill_device(self, tokens: torch.Tensor) -> tuple[ttnn.Tensor, int, int]:
        if tokens.dim() != 2 or tokens.shape[0] != 1:
            raise ValueError(f"Only batch-1 readiness is supported, got tokens shape {tuple(tokens.shape)}")
        real_len = int(tokens.shape[1])
        padded_len = _padded_prefill_len(real_len)
        if padded_len > self.model.full_model_config.max_seq_len:
            raise ValueError(
                f"Prompt length {real_len} pads to {padded_len}, beyond configured max_seq_len "
                f"{self.model.full_model_config.max_seq_len}"
            )
        padded = torch.zeros((1, padded_len), dtype=torch.int64)
        padded[:, :real_len] = tokens.to(torch.int64)
        token_4d = padded.reshape(1, 1, 1, padded_len)
        tokens_tt = ttnn.from_torch(
            token_4d,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._replicate_mapper(),
        )
        return tokens_tt, real_len, padded_len

    def _prepare_decode_inputs_host(
        self,
        token_id: int,
        current_pos: int,
        page_table: torch.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        padded_tokens = torch.zeros(_nearest_32(1), dtype=torch.int64)
        padded_tokens[0] = int(token_id)
        tokens = ttnn.from_torch(
            padded_tokens,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._replicate_mapper(),
        )
        tokens = ttnn.unsqueeze_to_4D(tokens)

        position = torch.tensor([int(current_pos)], dtype=torch.int32)
        current_pos_tt = ttnn.from_torch(
            position,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._replicate_mapper(),
        )
        rot_idxs = prepare_rot_idxs(self.model.rope_setup.config, position.to(torch.long), on_host=True)
        page_table_tt = ttnn.from_torch(
            page_table,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._replicate_mapper(),
        )
        return tokens, current_pos_tt, rot_idxs, page_table_tt

    def _decode_inputs_tuple(self) -> tuple[ttnn.Tensor | None, ...]:
        return (
            self._decode_trace.tokens,
            self._decode_trace.current_pos,
            self._decode_trace.rot_idxs,
            self._decode_trace.page_table,
        )

    def _refresh_hidden_from_token(self) -> None:
        new_hidden = self.model.embed_decode(self._decode_trace.tokens)
        if self._decode_trace.hidden_states is None:
            self._decode_trace.hidden_states = new_hidden
        else:
            ttnn.copy(new_hidden, self._decode_trace.hidden_states)
            new_hidden.deallocate(True)
        self._counters.hidden_device_refreshes += 1

    def _logits_to_torch(self, tt_logits: ttnn.Tensor) -> torch.Tensor:
        logits = ttnn.to_torch(
            tt_logits,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1),
        )
        return logits[..., : self.model.vocab_size].float().cpu()

    def _sampled_token_to_host(self, tt_sample_output: ttnn.Tensor | tuple[ttnn.Tensor, Any]) -> int:
        tt_tokens = tt_sample_output[0] if isinstance(tt_sample_output, tuple) else tt_sample_output
        token_tensor = ttnn.get_device_tensors(tt_tokens)[0]
        token_host = ttnn.to_torch(token_tensor).reshape(-1)
        self._counters.sampled_token_readbacks += 1
        return int(token_host[0].item())

    # ------------------------------------------------------------------
    # Contract: prefill and decode
    # ------------------------------------------------------------------

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor | ttnn.Tensor,
        kv_cache: Any,
        prompt_lens: list[int],
        return_all_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        tt_logits, real_len, last_token_idx = self.prefill_forward_device(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            return_all_logits=return_all_logits,
        )
        ttnn.synchronize_device(self.mesh_device)

        if return_all_logits:
            assert isinstance(tt_logits, list)
            logits = torch.empty(1, real_len, self.model.vocab_size, dtype=torch.float32)
            for tile_idx, tt_tile in enumerate(tt_logits):
                tile_logits = self._logits_to_torch(tt_tile)
                start = tile_idx * TILE_SIZE
                end = min(start + TILE_SIZE, real_len)
                if start >= real_len:
                    break
                logits[:, start:end, :] = tile_logits[:, 0, : end - start, :]
            return logits

        assert isinstance(tt_logits, ttnn.Tensor)
        tile_logits = self._logits_to_torch(tt_logits)
        row = last_token_idx % TILE_SIZE
        return tile_logits[:, 0, row : row + 1, :]

    def prefill_forward_device(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor | ttnn.Tensor,
        kv_cache: Any,
        prompt_lens: list[int],
        return_all_logits: bool = False,
    ) -> tuple[ttnn.Tensor | list[ttnn.Tensor], int, int]:
        tokens_tt, real_len, padded_len = self._tokens_to_prefill_device(tokens)
        page_table_tt = self._page_table_to_device(page_table)
        rot_mats = tuple(self.model.rope_setup.prefill_forward(start_pos=0, seq_len=padded_len))
        last_token_idx = int(prompt_lens[0]) - 1 if prompt_lens else real_len - 1

        tt_logits = self.model.prefill_forward(
            tokens_tt,
            rot_mats=rot_mats,
            page_table=page_table_tt,
            kv_cache=kv_cache,
            return_all_logits=return_all_logits,
            last_token_idx=last_token_idx,
        )
        return tt_logits, real_len, last_token_idx

    def _decode_no_trace_logits(
        self,
        token_id: int,
        current_pos: int,
        page_table: torch.Tensor,
        *,
        increment_positions: bool,
        kv_cache: Any,
    ) -> ttnn.Tensor:
        host_inputs = self._prepare_decode_inputs_host(token_id, current_pos, page_table)
        device_inputs = _copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        logits, _ = self.model.decode_forward_from_ttnn_inputs(
            device_inputs[0],
            device_inputs[1],
            device_inputs[2],
            device_inputs[3],
            increment_positions=increment_positions,
            kv_cache=kv_cache,
        )
        return logits

    def _capture_decode_trace(self, token_id: int, current_pos: int, page_table: torch.Tensor, *, kv_cache: Any) -> None:
        host_inputs = self._prepare_decode_inputs_host(token_id, current_pos, page_table)
        device_inputs = _copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        self._decode_trace.tokens = device_inputs[0]
        self._decode_trace.current_pos = device_inputs[1]
        self._decode_trace.rot_idxs = device_inputs[2]
        self._decode_trace.page_table = device_inputs[3]

        for _ in range(2):
            warm_logits, _ = self.model.decode_forward_from_ttnn_inputs(
                self._decode_trace.tokens,
                current_pos=self._decode_trace.current_pos,
                rot_idxs=self._decode_trace.rot_idxs,
                page_table=self._decode_trace.page_table,
                increment_positions=True,
                kv_cache=kv_cache,
            )
            warm_logits.deallocate(True)
            ttnn.synchronize_device(self.mesh_device)
            _copy_host_to_device(host_inputs, mesh_device=self.mesh_device, device_tensors=self._decode_inputs_tuple())
            ttnn.synchronize_device(self.mesh_device)

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        logits, _ = self.model.decode_forward_from_ttnn_inputs(
            self._decode_trace.tokens,
            current_pos=self._decode_trace.current_pos,
            rot_idxs=self._decode_trace.rot_idxs,
            page_table=self._decode_trace.page_table,
            increment_positions=True,
            kv_cache=kv_cache,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)

        self._decode_trace.trace_id = trace_id
        self._decode_trace.logits = logits
        self._decode_trace.initialized = True
        self._decode_host_position = int(current_pos) + 1
        self._counters.model_trace_captures += 1

    def _refresh_decode_inputs(
        self,
        *,
        token_id: int,
        current_pos: int,
        page_table: torch.Tensor,
        reset_all: bool,
        token_only: bool,
        page_table_changed: bool,
    ) -> None:
        host_inputs = self._prepare_decode_inputs_host(token_id, current_pos, page_table)
        if reset_all:
            _copy_host_to_device(host_inputs, mesh_device=self.mesh_device, device_tensors=self._decode_inputs_tuple())
            self._counters.token_input_host_copies += 1
            self._counters.position_host_copies += 1
            self._counters.rope_index_host_copies += 1
            self._counters.page_table_host_copies += 1
            self._counters.last_page_table_refresh_changed_only = False
            self._decode_host_position = int(current_pos)
            return

        if token_only:
            ttnn.copy_host_to_device_tensor(host_inputs[0], self._decode_trace.tokens)
            self._counters.token_input_host_copies += 1

        if page_table_changed:
            ttnn.copy_host_to_device_tensor(host_inputs[3], self._decode_trace.page_table)
            self._counters.page_table_host_copies += 1
            self._counters.last_page_table_refresh_changed_only = not reset_all

    def _decode_trace_sample(
        self,
        token_id: int,
        current_pos: int,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        enable_trace: bool,
        token_from_host: bool,
        refresh_sampled_hidden: bool,
        readback: bool = True,
    ) -> int | ttnn.Tensor | tuple[ttnn.Tensor, Any] | None:
        if not enable_trace:
            logits = self._decode_no_trace_logits(
                token_id,
                current_pos,
                page_table,
                increment_positions=False,
                kv_cache=kv_cache,
            )
            tt_out = self.sampling.sample(logits, enable_trace=False)
            logits.deallocate(True)
            return self._sampled_token_to_host(tt_out) if readback else tt_out

        if self._decode_trace.trace_id is None:
            self._capture_decode_trace(token_id, current_pos, page_table, kv_cache=kv_cache)
            self._refresh_decode_inputs(
                token_id=token_id,
                current_pos=current_pos,
                page_table=page_table,
                reset_all=True,
                token_only=False,
                page_table_changed=False,
            )
            self._prev_page_table = page_table.clone()
        else:
            page_table_changed = self._prev_page_table is None or not torch.equal(self._prev_page_table, page_table)
            position_mismatch = self._decode_host_position is None or int(current_pos) != self._decode_host_position
            if position_mismatch:
                self._refresh_decode_inputs(
                    token_id=token_id,
                    current_pos=current_pos,
                    page_table=page_table,
                    reset_all=True,
                    token_only=False,
                    page_table_changed=page_table_changed,
                )
            elif token_from_host or page_table_changed:
                self._refresh_decode_inputs(
                    token_id=token_id,
                    current_pos=current_pos,
                    page_table=page_table,
                    reset_all=False,
                    token_only=token_from_host,
                    page_table_changed=page_table_changed,
                )
            if page_table_changed:
                self._prev_page_table = page_table.clone()

        sampling_trace_ready = self.sampling.has_trace_for_current_state(enable_trace=True)
        ttnn.execute_trace(self.mesh_device, self._decode_trace.trace_id, cq_id=0, blocking=False)
        if not sampling_trace_ready:
            ttnn.synchronize_device(self.mesh_device)
            self._counters.model_trace_syncs_before_sampling_capture += 1
        self._counters.model_trace_replays += 1
        self._counters.model_trace_nonblocking_replays += 1
        self._counters.position_device_increments += 1
        self._counters.rope_index_device_increments += 1
        self._decode_host_position = int(current_pos) + 1

        tt_out = self.sampling.sample(
            self._decode_trace.logits,
            enable_trace=True,
            tt_out_tok=self._decode_trace.tokens,
            skip_precompile=self._decode_sampling_precompiled,
        )
        self._counters.sampling_trace_invocations += 1
        if not readback:
            del refresh_sampled_hidden
            return tt_out
        sampled = self._sampled_token_to_host(tt_out)
        del refresh_sampled_hidden
        return sampled

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        enable_trace = bool(kwargs.pop("enable_trace", True))
        return_logits = bool(kwargs.pop("return_logits", False))
        if kwargs:
            self._counters.notes.append(f"decode_forward ignored kwargs={sorted(kwargs)}")

        token_id = int(tokens.reshape(-1)[0].item())
        current_pos = int(start_pos.reshape(-1)[0].item())
        page_table_torch = self._coerce_page_table_torch(page_table)
        if return_logits:
            logits = self._decode_no_trace_logits(
                token_id,
                current_pos,
                page_table_torch,
                increment_positions=False,
                kv_cache=kv_cache,
            )
            return self._logits_to_torch(logits).reshape(1, -1)

        sampled = self._decode_trace_sample(
            token_id,
            current_pos,
            page_table=page_table_torch,
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            token_from_host=True,
            refresh_sampled_hidden=True,
        )
        return torch.tensor([sampled], dtype=torch.long)

    # ------------------------------------------------------------------
    # Contract: high-level generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        enable_trace: bool = True,
        **kwargs: Any,
    ) -> list[int]:
        if max_new_tokens <= 0:
            return []

        stop_on_eos = bool(kwargs.pop("stop_on_eos", next_input is None))
        if kwargs:
            self._counters.notes.append(f"generate ignored kwargs={sorted(kwargs)}")

        self.reset()
        prompt_token_ids = _normal_token_list(prompt_token_ids)
        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        prompt_len = len(prompt_token_ids)
        logits = self.prefill_forward(
            prompt,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[prompt_len],
            return_all_logits=False,
        )
        first_pred = int(torch.argmax(logits.reshape(-1)).item())
        predictions = [first_pred]

        feed_token = next_input(0, first_pred) if next_input is not None else first_pred
        if stop_on_eos and next_input is None and first_pred == self.tokenizer.eos_token_id:
            return predictions

        current_pos = prompt_len
        token_from_host = True
        for step in range(1, max_new_tokens):
            pred = self._decode_trace_sample(
                int(feed_token),
                current_pos,
                page_table=self.page_table,
                kv_cache=self.kv_cache,
                enable_trace=enable_trace,
                token_from_host=token_from_host,
                refresh_sampled_hidden=next_input is None,
            )
            predictions.append(pred)
            current_pos += 1

            if next_input is None:
                feed_token = pred
                token_from_host = False
                if stop_on_eos and pred == self.tokenizer.eos_token_id:
                    break
            else:
                feed_token = next_input(step, pred)
                token_from_host = True

        return predictions

    def _release_decode_trace(self) -> None:
        trace_id = self._decode_trace.trace_id
        self._decode_trace.trace_id = None
        for name in ("tokens", "hidden_states", "current_pos", "rot_idxs", "page_table", "logits"):
            tensor = getattr(self._decode_trace, name)
            if tensor is not None:
                tensor.deallocate(True)
                setattr(self._decode_trace, name, None)
        if self._decode_trace.rot_mats is not None:
            for tensor in self._decode_trace.rot_mats:
                tensor.deallocate(True)
            self._decode_trace.rot_mats = None
        self._decode_trace.logits = None
        self._decode_trace.initialized = False
        if trace_id is not None:
            ttnn.synchronize_device(self.mesh_device)
            ttnn.release_trace(self.mesh_device, trace_id)

    def reset(self, *, keep_decode_trace: bool = False) -> None:
        if not keep_decode_trace:
            self.sampling.reset_trace()
            self._release_decode_trace()
            self.model.release_decode_persistent_buffers()
        self.page_table = self._make_default_page_table()
        self._prev_page_table = None
        self._decode_host_position = None
        self._counters = RuntimeCounters()
        self.sampling.reset_sampling_params(self.greedy_sampling_params)
        self.sampling.reset_output_state()

    def trace_counters(self) -> dict[str, Any]:
        result = self._counters.__dict__.copy()
        result["decode_trace_captured"] = self._decode_trace.trace_id is not None
        result["sampling_force_argmax"] = bool(self.sampling.tt_sampling.force_argmax_sampling)
        result["sampling_max_top_k"] = int(self.sampling.tt_sampling.max_top_k)
        result["padded_vocab_size"] = int(self.model.padded_vocab_size)
        result["vocab_size"] = int(self.model.vocab_size)
        return result

    def teardown(self) -> None:
        self.sampling.reset_trace()
        self._release_decode_trace()
        self.model.release_decode_persistent_buffers()


def build_generator(model_dir: str | Path, mesh_device, **kwargs: Any) -> Llama31_8B_InstructGenerator:
    model_dir = Path(model_dir)
    shape = tuple(mesh_device.shape)
    if shape != TARGET_MESH_SHAPE:
        raise ValueError(f"Llama 3.1 8B full model requires T3K mesh shape {TARGET_MESH_SHAPE}, got {shape}")

    model_id = kwargs.pop("model_id", MODEL_ID)
    max_batch_size = int(kwargs.pop("max_batch_size", 1))
    max_seq_len = kwargs.pop("max_seq_len", None)
    page_block_size = int(kwargs.pop("page_block_size", 64))
    max_num_blocks = kwargs.pop("max_num_blocks", None)
    override_num_layers = kwargs.pop("override_num_layers", None)
    cache_dir = kwargs.pop("cache_dir", model_dir / "tt_cache" / "full_model")
    precision_config_path = kwargs.pop("precision_config_path", None)
    if kwargs:
        raise TypeError(f"Unexpected build_generator kwargs: {sorted(kwargs)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = Llama31_8B_InstructFullModel.from_pretrained(
        mesh_device=mesh_device,
        model_id=model_id,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        page_block_size=page_block_size,
        max_num_blocks=max_num_blocks,
        num_layers=override_num_layers,
        cache_dir=cache_dir,
        precision_config_path=precision_config_path,
    )
    return Llama31_8B_InstructGenerator(
        model_dir=model_dir,
        mesh_device=mesh_device,
        model=model,
        tokenizer=tokenizer,
    )


__all__ = ["Llama31_8B_InstructGenerator", "build_generator"]
