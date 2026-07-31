# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Readiness-check generator for the Gemma 4 12B autoport full model."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Iterable, List, Optional

import torch
import ttnn
from transformers import AutoTokenizer

from models.common.readiness_check.contract import Generator, NextInputFn


def _load_model_module():
    path = Path(__file__).with_name("model.py")
    spec = importlib.util.spec_from_file_location("gemma4_12b_full_model", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"cannot load full model from {path}")
    spec.loader.exec_module(module)
    return module


_model_mod = _load_model_module()

SUPPORTED_HF_MODEL_ID = _model_mod.SUPPORTED_HF_MODEL_ID
DEFAULT_BLOCK_SIZE = _model_mod.DEFAULT_BLOCK_SIZE
DEFAULT_MAX_SEQ_LEN = _model_mod.DEFAULT_MAX_SEQ_LEN


class Gemma412BGenerator(Generator):
    """High-level and low-level generator contract for Gemma 4 12B."""

    def __init__(
        self,
        *,
        model_dir: Path,
        mesh_device,
        hf_model_id: str = SUPPORTED_HF_MODEL_ID,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        max_batch_size: int = 1,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_num_blocks: int | None = None,
        num_layers: int | None = None,
        tensor_cache_path: str | Path | None = None,
        use_on_device_sampling: bool = False,
        suppress_special_tokens: bool = True,
        allocate_standalone_cache: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.mesh_device = mesh_device
        self.hf_model_id = hf_model_id
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks or math.ceil(max_seq_len / block_size)
        self.use_on_device_sampling = use_on_device_sampling
        self.suppress_special_tokens = suppress_special_tokens

        model_path = _model_mod.resolve_model_path(hf_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.suppress_token_ids = self._generation_suppress_token_ids()

        self.model = _model_mod.build_model(
            mesh_device=mesh_device,
            hf_model_id=hf_model_id,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            num_layers=num_layers,
            tensor_cache_path=tensor_cache_path,
            block_size=block_size,
            enable_sampling=use_on_device_sampling,
        )
        if allocate_standalone_cache:
            self.kv_cache = self.model.create_paged_kv_cache(
                max_num_blocks=self.max_num_blocks,
                block_size=self.block_size,
            )
            self.page_table = self.model.create_page_table(self.max_num_blocks)
            self.page_table_tt = self.model.page_table_to_device(self.page_table)
        else:
            self.kv_cache = None
            self.page_table = None
            self.page_table_tt = None
        self._last_prompt_len = 0
        self._decode_traces: dict[tuple[Any, ...], dict[str, Any]] = {}

    @staticmethod
    def _token_ids(token_ids: int | Iterable[int] | None) -> set[int]:
        if token_ids is None:
            return set()
        if isinstance(token_ids, int):
            return {int(token_ids)}
        return {int(token_id) for token_id in token_ids if token_id is not None}

    def _generation_suppress_token_ids(self) -> List[int]:
        if not self.suppress_special_tokens:
            return []
        eos_ids = self._token_ids(self.tokenizer.eos_token_id)
        suppressed = set(int(token_id) for token_id in getattr(self.tokenizer, "all_special_ids", []))
        return sorted(token_id for token_id in suppressed - eos_ids if token_id >= 0)

    def _mask_generation_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.suppress_token_ids:
            return logits
        masked = logits.clone()
        valid_ids = [token_id for token_id in self.suppress_token_ids if token_id < masked.shape[-1]]
        if valid_ids:
            masked[..., valid_ids] = -float("inf")
        return masked

    def _internal_page_table(self, page_table):
        if page_table is None:
            if self.page_table_tt is None:
                raise RuntimeError("page_table must be provided when standalone cache allocation is disabled")
            return self.page_table_tt
        if isinstance(page_table, (list, tuple)):
            return [self._internal_page_table(layer_page_table) for layer_page_table in page_table]
        if isinstance(page_table, torch.Tensor):
            return self.model.page_table_to_device(page_table)
        return page_table

    def _host_page_table_tensor(self, page_table: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            page_table.to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=_model_mod._replicate_mapper(self.mesh_device),
        )

    def _trace_page_table_to_device(self, page_table):
        if page_table is None:
            if self.page_table_tt is None:
                raise RuntimeError("page_table must be provided when standalone cache allocation is disabled")
            return self.page_table_tt
        if isinstance(page_table, (list, tuple)):
            return [self._trace_page_table_to_device(layer_page_table) for layer_page_table in page_table]
        if isinstance(page_table, torch.Tensor):
            return self.model.page_table_to_device(page_table)
        return page_table

    def _update_trace_page_table(self, device_page_table, page_table) -> None:
        if page_table is None or isinstance(page_table, ttnn.Tensor):
            return
        if isinstance(page_table, (list, tuple)):
            if not isinstance(device_page_table, list) or len(device_page_table) != len(page_table):
                raise RuntimeError("decode trace page-table structure changed; rebuild the generator trace")
            for device_layer_page_table, host_layer_page_table in zip(device_page_table, page_table):
                self._update_trace_page_table(device_layer_page_table, host_layer_page_table)
            return
        if isinstance(page_table, torch.Tensor):
            if tuple(device_page_table.shape) != tuple(page_table.shape):
                raise RuntimeError(
                    "decode trace page-table shape changed from "
                    f"{tuple(device_page_table.shape)} to {tuple(page_table.shape)}"
                )
            ttnn.copy_host_to_device_tensor(self._host_page_table_tensor(page_table), device_page_table)

    def _internal_kv_cache(self, kv_cache):
        if kv_cache is None:
            if self.kv_cache is None:
                raise RuntimeError("kv_cache must be provided when standalone cache allocation is disabled")
            return self.kv_cache
        return kv_cache

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache,
        prompt_lens: List[int],
        return_all_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        page_table_tt = self._internal_page_table(page_table)
        kv_cache = self._internal_kv_cache(kv_cache)
        self._last_prompt_len = int(prompt_lens[0])
        return self.model.prefill_forward(
            tokens.to(torch.long),
            page_table=page_table_tt,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            return_all_logits=return_all_logits,
            return_ttnn=bool(kwargs.get("return_ttnn", False)),
            gather_logits=bool(kwargs.get("gather_logits", True)),
        )

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table,
        kv_cache,
        **kwargs: Any,
    ) -> torch.Tensor:
        kv_cache = self._internal_kv_cache(kv_cache)
        if kwargs.get("enable_trace", False):
            return self.decode_forward_traced(
                tokens,
                start_pos,
                page_table=page_table if page_table is not None else self.page_table_tt,
                kv_cache=kv_cache,
                sample_on_device=bool(kwargs.get("sample_on_device", False)),
                return_ttnn=bool(kwargs.get("return_ttnn", False)),
                async_decode=bool(kwargs.get("async_decode", False)),
            )
        page_table_tt = self._internal_page_table(page_table)
        return self.model.decode_forward(
            tokens.to(torch.long),
            start_pos,
            page_table=page_table_tt,
            kv_cache=kv_cache,
            sample_on_device=bool(kwargs.get("sample_on_device", False)),
            return_ttnn=bool(kwargs.get("return_ttnn", False)),
        )

    def _decode_position_value(self, start_pos: torch.Tensor | int) -> int:
        if isinstance(start_pos, torch.Tensor):
            return int(start_pos.reshape(-1)[0].item())
        return int(start_pos)

    def _decode_trace_key(
        self,
        *,
        sample_on_device: bool,
        batch: int,
        pos_value: int,
    ) -> tuple[Any, ...]:
        # The captured decode graph encloses the sampling graph when
        # sample_on_device=True, so the outer trace key must be at least as
        # specific as SamplingGenerator's internal trace key.
        sampling_key: tuple[Any, ...] = ()
        if sample_on_device:
            sampling = self.model.sampling
            sampling_key = (
                bool(getattr(sampling.tt_sampling, "force_argmax_sampling", False)),
                bool(getattr(sampling, "_penalties_active", False)),
                bool(getattr(sampling, "_log_probs_active", False)),
            )
        sliding_window = int(getattr(self.model.model_args, "sliding_window", 0) or 0)
        long_sliding_decode = bool(sliding_window and pos_value >= sliding_window)
        return (sample_on_device, batch, long_sliding_decode, *sampling_key)

    def _format_trace_output(self, output, *, sample_on_device: bool, return_ttnn: bool, batch: int):
        if return_ttnn:
            return output
        if sample_on_device:
            sampled_tokens = output[0] if isinstance(output, tuple) else output
            return _model_mod._first_device_torch(sampled_tokens, self.mesh_device).reshape(-1)[:batch].to(torch.long)
        logits = _model_mod._first_device_torch(output, self.mesh_device).float()
        return logits[:, :, :batch, : self.model.vocab_size].reshape(batch, self.model.vocab_size)

    def decode_forward_traced(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor | int,
        *,
        page_table,
        kv_cache,
        sample_on_device: bool = True,
        return_ttnn: bool = True,
        async_decode: bool = False,
    ):
        if sample_on_device and self.model.sampling is None:
            raise RuntimeError("traced on-device decode requested but sampling is not initialized")

        tokens = tokens.to(torch.long).reshape(tokens.shape[0], 1)
        batch = int(tokens.shape[0])
        pos_value = self._decode_position_value(start_pos)
        trace_key = self._decode_trace_key(
            sample_on_device=sample_on_device,
            batch=batch,
            pos_value=pos_value,
        )
        state = self._decode_traces.get(trace_key)

        if state is None:
            page_table_tt = self._trace_page_table_to_device(page_table)
            compile_inputs = self.model.prepare_decode_device_inputs(tokens, start_pos)
            self.model.decode_forward_device_inputs(
                *compile_inputs,
                page_table=page_table_tt,
                kv_cache=kv_cache,
                sample_on_device=sample_on_device,
                return_ttnn=True,
                token_index=pos_value,
            )
            ttnn.synchronize_device(self.mesh_device)

            trace_inputs = self.model.prepare_decode_device_inputs(tokens, start_pos)
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            output = self.model.decode_forward_device_inputs(
                *trace_inputs,
                page_table=page_table_tt,
                kv_cache=kv_cache,
                sample_on_device=sample_on_device,
                return_ttnn=True,
                token_index=pos_value,
            )
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.mesh_device)
            self._decode_traces[trace_key] = {
                "id": trace_id,
                "inputs": trace_inputs,
                "output": output,
                "token_index": pos_value,
                "page_table": page_table_tt,
            }
            ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(self.mesh_device)
            return self._format_trace_output(
                output,
                sample_on_device=sample_on_device,
                return_ttnn=return_ttnn,
                batch=batch,
            )

        self._update_trace_page_table(state["page_table"], page_table)
        self.model.update_decode_device_inputs(state["inputs"], tokens, start_pos)
        ttnn.execute_trace(self.mesh_device, state["id"], cq_id=0, blocking=not async_decode)
        if not async_decode:
            ttnn.synchronize_device(self.mesh_device)
        return self._format_trace_output(
            state["output"],
            sample_on_device=sample_on_device,
            return_ttnn=return_ttnn,
            batch=batch,
        )

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        **kwargs: Any,
    ) -> List[int]:
        if max_new_tokens <= 0:
            return []
        self.reset()
        enable_trace = bool(kwargs.get("enable_trace", False))

        prompt = torch.tensor([list(prompt_token_ids)], dtype=torch.long)
        prompt_len = len(prompt_token_ids)
        prefill_logits = self.prefill_forward(
            prompt,
            page_table=self.page_table_tt,
            kv_cache=self.kv_cache,
            prompt_lens=[prompt_len],
            return_all_logits=False,
        )
        pred = int(torch.argmax(self._mask_generation_logits(prefill_logits)[0, 0, :], dim=-1).item())
        predictions = [pred]
        feed_token = next_input(0, pred) if next_input is not None else pred

        stop_on_eos = bool(kwargs.get("stop_on_eos", False))
        eos_ids = self.tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = {eos_ids}
        else:
            eos_ids = set(eos_ids or [])
        if stop_on_eos and pred in eos_ids:
            return predictions

        for step in range(1, max_new_tokens):
            token = torch.tensor([[int(feed_token)]], dtype=torch.long)
            start_pos = torch.tensor([prompt_len + step - 1], dtype=torch.long)
            out = self.decode_forward(
                token,
                start_pos,
                page_table=self.page_table_tt,
                kv_cache=self.kv_cache,
                sample_on_device=self.use_on_device_sampling,
                enable_trace=enable_trace,
            )
            if self.use_on_device_sampling:
                pred = int(out.reshape(-1)[0].item())
            else:
                pred = int(torch.argmax(self._mask_generation_logits(out)[0, :], dim=-1).item())
            predictions.append(pred)
            if stop_on_eos and pred in eos_ids:
                break
            feed_token = next_input(step, pred) if next_input is not None else pred

        return predictions

    def reset(self) -> None:
        if self.kv_cache is None:
            return
        self.model.reset_kv_cache(self.kv_cache)
        self._last_prompt_len = 0

    def teardown(self) -> None:
        for state in self._decode_traces.values():
            try:
                ttnn.release_trace(self.mesh_device, state["id"])
            except Exception:
                pass
        self._decode_traces.clear()
        sampling = getattr(self.model, "sampling", None)
        if sampling is not None:
            sampling.reset_trace()


def build_generator(
    model_dir: str | Path,
    mesh_device,
    *,
    hf_model_id: str = SUPPORTED_HF_MODEL_ID,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_batch_size: int = 1,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_num_blocks: int | None = None,
    num_layers: int | None = None,
    tensor_cache_path: str | Path | None = None,
    use_on_device_sampling: bool = False,
    suppress_special_tokens: bool = True,
    allocate_standalone_cache: bool = True,
    **kwargs: Any,
) -> Gemma412BGenerator:
    if kwargs:
        raise TypeError(f"unsupported Gemma412BGenerator build kwargs: {sorted(kwargs)}")
    return Gemma412BGenerator(
        model_dir=Path(model_dir),
        mesh_device=mesh_device,
        hf_model_id=hf_model_id,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        num_layers=num_layers,
        tensor_cache_path=tensor_cache_path,
        use_on_device_sampling=use_on_device_sampling,
        suppress_special_tokens=suppress_special_tokens,
        allocate_standalone_cache=allocate_standalone_cache,
    )


__all__ = ["Gemma412BGenerator", "SUPPORTED_HF_MODEL_ID", "build_generator"]
