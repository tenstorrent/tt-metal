# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Llama3Generator -- thin vLLM adapter wrapping an executor.

Zero trace state, zero warmup state, zero execution logic.
Just signature adaptation for TTModelRunner.
"""

from typing import Any

import torch

import ttnn
from models.common.models.executor import _process_output_decode, _process_output_decode_tokens
from models.common.models.llama3_8b.executor import EagerLlamaExecutor, TracedLlamaExecutor
from models.common.models.llama3_8b.hf_adaptor import from_pretrained
from models.common.models.llama3_8b.model import Llama31_8BPagedAttentionConfig

_VLLM_BLOCK_SIZE = 32
_IGNORED_VLLM_KWARGS = {
    "page_tables_per_layer",
    "prompt_tokens",
    "output_tokens",
    "slot_remap",
    "rope_deltas_all_users",
}


class Llama3Generator:
    """vLLM-compatible adapter. Wraps any executor (typically traced).

    Usage:
        generator = Llama3Generator.initialize_vllm_model(hf_config, mesh_device, ...)
        kv_cache = generator.allocate_kv_cache(shape, dtype, num_layers)
        logits = generator.prefill_forward(tokens, page_table=..., kv_cache=kv_cache, ...)
        output = generator.decode_forward(tokens, start_pos, page_table=..., kv_cache=kv_cache, ...)
    """

    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "required_block_size": _VLLM_BLOCK_SIZE,
    }
    requires_prefill_trace_warmup = True

    def __init__(self, executor: EagerLlamaExecutor | TracedLlamaExecutor):
        self.executor = executor
        self.model = executor.model
        self.model_args = executor.model_args
        self.mesh_device = executor.mesh_device
        self.already_warmed_up_prefill = False

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        optimizations="performance",
    ):
        """Build Llama3Transformer1D from HF config and wrap in traced executor.

        This is the entry point called by vLLM's TTModelRunner.
        """
        hf_model_name = hf_config._name_or_path
        instruct = "Instruct" in hf_model_name
        max_num_blocks = (int(max_seq_len) + _VLLM_BLOCK_SIZE - 1) // _VLLM_BLOCK_SIZE + int(max_batch_size)
        paged_attention_config = Llama31_8BPagedAttentionConfig(
            block_size=_VLLM_BLOCK_SIZE,
            max_num_blocks=max_num_blocks,
        )

        llm = from_pretrained(
            mesh_device=mesh_device,
            hf_model=hf_model_name,
            instruct=instruct,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
            paged_attention_config=paged_attention_config,
        )

        executor = TracedLlamaExecutor(
            llm.model,
            mesh_device,
            model_args=llm.runtime_config,
        )
        return cls(executor)

    def prefill_forward(self, *args, **kwargs):
        self._drop_unsupported_vllm_kwargs(kwargs)
        self._normalize_tensor_kwarg(kwargs, "tokens", torch.long)
        self._normalize_tensor_kwarg(kwargs, "page_table", torch.int32)
        self._normalize_tensor_kwarg(kwargs, "prompt_lens", torch.long)
        self._normalize_tensor_kwarg(kwargs, "start_pos", torch.long)
        return self.executor.prefill_forward(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        self._drop_unsupported_vllm_kwargs(kwargs)
        self._normalize_tensor_kwarg(kwargs, "tokens", torch.long)
        self._normalize_tensor_kwarg(kwargs, "start_pos", torch.long)
        self._normalize_tensor_kwarg(kwargs, "page_table", torch.int32)
        tokens = kwargs.get("tokens")
        if isinstance(tokens, torch.Tensor) and tokens.dim() == 2 and tokens.shape[-1] == 1:
            kwargs["tokens"] = tokens.view(-1)
        return self.executor.decode_forward(*args, **kwargs)

    def process_decode_output_host(self, tt_out, is_tokens=False):
        """Convert decode output returned with read_from_device=False to torch."""
        out, log_probs = tt_out if isinstance(tt_out, tuple) else (tt_out, None)
        batch_size = self.model.config.max_batch_size
        cluster_shape = list(self.model.config.mesh_device.shape)

        if is_tokens:
            out = self._ensure_host_tensor(out)
            tokens = _process_output_decode_tokens(
                out,
                batch_size,
                cluster_shape,
            )
            return tokens.to(torch.int32), log_probs

        host_logits = self._ensure_host_tensor(out)
        logits = _process_output_decode(
            host_logits,
            batch_size,
            self.model.vocab_size,
            self.model.num_devices,
            cluster_shape,
        )
        return logits, log_probs

    def read_decode_output(self, tt_out, async_read=False):
        """Start decode output readback for vLLM async scheduling.

        `decode_forward(..., read_from_device=False)` returns TT tensors. vLLM
        calls this method with `async_read=True` to submit non-blocking host
        copies, then later synchronizes the returned events before calling
        `process_decode_output_host()`.
        """
        out, log_probs = tt_out if isinstance(tt_out, tuple) else (tt_out, None)

        if not async_read:
            return (self._read_to_host(out), self._read_to_host(log_probs))

        host_out = self._read_to_host(out, blocking=False)
        host_log_probs = self._read_to_host(log_probs, blocking=False)
        read_events = [ttnn.record_event(self.mesh_device, 0)]
        return (host_out, host_log_probs), read_events

    def allocate_kv_cache(self, *args, **kwargs):
        return self.executor.allocate_kv_cache(*args, **kwargs)

    @staticmethod
    def _drop_unsupported_vllm_kwargs(kwargs):
        for key in _IGNORED_VLLM_KWARGS:
            kwargs.pop(key, None)

    @staticmethod
    def _normalize_tensor_kwarg(kwargs, key, dtype):
        if kwargs.get(key) is not None and not isinstance(kwargs[key], torch.Tensor):
            kwargs[key] = torch.as_tensor(kwargs[key], dtype=dtype)

    @staticmethod
    def _read_to_host(value, blocking=True):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.cpu()
        if isinstance(value, ttnn.Tensor):
            if value.storage_type() == ttnn.StorageType.HOST:
                return value
            return value.cpu(blocking=blocking)
        if hasattr(value, "cpu"):
            try:
                return value.cpu(blocking=blocking)
            except TypeError:
                return value.cpu()
        return value

    @classmethod
    def _ensure_host_tensor(cls, value):
        host_value = cls._read_to_host(value)
        if isinstance(host_value, ttnn.Tensor):
            assert host_value.storage_type() == ttnn.StorageType.HOST, "Expected host tensor"
        return host_value

    def _warmup_executor(self, enable_trace):
        if enable_trace:
            return self.executor
        engine = getattr(self.executor, "_engine", None)
        return getattr(engine, "_eager", self.executor)

    def warmup_model_prefill(self, *args, **kwargs):
        kv_cache = kwargs.get("kv_cache")
        enable_trace = kwargs.get("enable_trace", True)
        can_sample_on_device = kwargs.get("can_sample_on_device", False)
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True
        executor = self._warmup_executor(enable_trace)

        seq_lens = getattr(self.model_args, "trace_prefill_supported_seq_lens", (128,))
        max_batch_size = int(self.model.config.max_batch_size)
        batch_sizes = [1]
        if max_batch_size > 1 and 128 in seq_lens:
            batch_sizes = [batch_size for batch_size in (1, 2, 4, 8, 16, 32) if batch_size <= max_batch_size]

        for seq_len in seq_lens:
            for batch_size in batch_sizes:
                if batch_size > 1 and seq_len != 128:
                    continue
                tokens = torch.zeros((batch_size, seq_len), dtype=torch.long)
                prompt_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
                num_blocks = (seq_len + _VLLM_BLOCK_SIZE - 1) // _VLLM_BLOCK_SIZE
                page_table = torch.zeros((batch_size, num_blocks), dtype=torch.int32)
                sampling_params = None
                if can_sample_on_device:
                    from models.common.sampling.sampling_params import SamplingParams

                    sampling_params = SamplingParams(
                        temperature=[0.0] * batch_size,
                        top_k=[1] * batch_size,
                        top_p=[1.0] * batch_size,
                    )
                executor.compile_prefill(
                    tokens=tokens,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    prompt_lens=prompt_lens,
                    sampling_params=sampling_params,
                )

    def warmup_model_decode(
        self,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        max_batch_size: int,  # ↓ Coverage dimensions
        num_blocks: int,
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
    ) -> None:
        tokens = torch.zeros(max_batch_size, dtype=torch.int64)
        start_pos = torch.zeros(max_batch_size, dtype=torch.int64)
        page_table = torch.zeros(max_batch_size, num_blocks, dtype=torch.int32)
        sampling_params = [None]

        if can_sample_on_device:
            from models.common.sampling.sampling_params import SamplingParams

            sampling_params.insert(
                0,
                SamplingParams(
                    temperature=[0.0] * max_batch_size,
                    top_k=[1] * max_batch_size,
                    top_p=[1.0] * max_batch_size,
                ),
            )

        executor = self._warmup_executor(enable_trace)
        for param in sampling_params:
            executor.compile_decode(
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                sampling_params=param,
            )

    @property
    def cache_path(self):
        if self.model_args:
            return self.model_args.model_cache_path
        return None
