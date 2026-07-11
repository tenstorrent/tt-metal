# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Llama3Generator -- thin vLLM adapter wrapping an executor.

Zero trace state, zero warmup state, zero execution logic.
Just signature adaptation for TTModelRunner.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields

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
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int = 0,
        max_num_seqs: int = 1,
        **kwargs,
    ) -> int:
        """Return the unpadded per-submesh KV token budget for vLLM sizing."""
        return int(max_model_len)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations="performance",
    ):
        """Build Llama3Transformer1D from HF config and wrap in traced executor.

        This is the entry point called by vLLM's TTModelRunner.
        """
        if tt_data_parallel != 1:
            return _DPLlama3Generator.initialize_vllm_model(
                hf_config=hf_config,
                mesh_device=mesh_device,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                tt_data_parallel=tt_data_parallel,
                optimizations=optimizations,
            )

        return cls._initialize_single_lane(
            hf_config=hf_config,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            optimizations=optimizations,
        )

    @classmethod
    def _initialize_single_lane(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        optimizations="performance",
    ):
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
        kv_cache,
        enable_trace,
        max_batch_size,
        num_blocks,
        can_sample_on_device,
        **kwargs,
    ):
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


class _DPLlama3Generator:
    """Gathered-DP adapter that owns one single-lane generator per submesh."""

    requires_prefill_trace_warmup = True

    def __init__(self, lanes: list[Llama3Generator], mesh_device, tt_data_parallel: int):
        assert lanes, "DP generator requires at least one lane"
        self.lanes = lanes
        self.executors = [lane.executor for lane in lanes]
        self.model = [lane.model for lane in lanes]
        self.model_args = [lane.model_args for lane in lanes]
        self.mesh_device = mesh_device
        self.mesh_devices = [lane.mesh_device for lane in lanes]
        self.tt_data_parallel = tt_data_parallel
        self.per_lane_max_batch_size = lanes[0].model.config.max_batch_size
        self.output_pool = ThreadPoolExecutor(
            max_workers=tt_data_parallel,
            thread_name_prefix="tttv2-dp-output",
        )

    @property
    def already_warmed_up_prefill(self):
        return all(lane.already_warmed_up_prefill for lane in self.lanes)

    @already_warmed_up_prefill.setter
    def already_warmed_up_prefill(self, value):
        for lane in self.lanes:
            lane.already_warmed_up_prefill = value

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations="performance",
    ):
        if int(max_batch_size) % int(tt_data_parallel) != 0:
            raise ValueError(
                f"max_batch_size={max_batch_size} must be divisible by " f"tt_data_parallel={tt_data_parallel}"
            )

        from models.tt_transformers.tt.generator import create_submeshes

        per_lane_max_batch_size = int(max_batch_size) // int(tt_data_parallel)
        submeshes = create_submeshes(mesh_device, int(tt_data_parallel))
        if len(submeshes) != int(tt_data_parallel):
            raise ValueError(f"Expected {tt_data_parallel} submeshes, got {len(submeshes)}")

        lanes = [
            Llama3Generator._initialize_single_lane(
                hf_config=hf_config,
                mesh_device=submesh,
                max_batch_size=per_lane_max_batch_size,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                optimizations=optimizations,
            )
            for submesh in submeshes
        ]
        return cls(lanes, mesh_device, int(tt_data_parallel))

    def allocate_kv_cache(self, *args, **kwargs):
        return [lane.allocate_kv_cache(*args, **kwargs) for lane in self.lanes]

    def prefill_forward(self, *args, **kwargs):
        if args:
            raise TypeError("TTTv2 DP prefill_forward expects keyword arguments")
        Llama3Generator._drop_unsupported_vllm_kwargs(kwargs)
        self._normalize_common_kwargs(kwargs, decode=False)

        tokens = kwargs["tokens"]
        page_table = kwargs["page_table"]
        prompt_lens = kwargs.get("prompt_lens")
        start_pos = kwargs.get("start_pos")
        kv_cache = kwargs.get("kv_cache")
        empty_slots = kwargs.get("empty_slots")
        sampling_params = kwargs.get("sampling_params")
        batch_size = int(tokens.shape[0])
        if empty_slots is None:
            empty_slots = list(range(batch_size))
        if len(empty_slots) != batch_size:
            raise ValueError(f"empty_slots length {len(empty_slots)} must match prefill batch {batch_size}")

        output = None
        output_tokens = None
        row_to_result: dict[int, torch.Tensor] = {}
        for lane_idx, rows, local_slots in self._prefill_lane_groups(empty_slots):
            lane_kwargs = dict(kwargs)
            lane_kwargs["tokens"] = tokens[rows]
            lane_kwargs["page_table"] = page_table[rows]
            lane_kwargs["empty_slots"] = local_slots
            lane_kwargs["sampling_params"] = self._slice_sampling_params(
                sampling_params,
                rows,
            )
            if prompt_lens is not None:
                lane_kwargs["prompt_lens"] = prompt_lens[rows]
            if start_pos is not None:
                lane_kwargs["start_pos"] = start_pos[rows]
            if kv_cache is not None:
                lane_kwargs["kv_cache"] = kv_cache[lane_idx]

            lane_out = self.lanes[lane_idx].prefill_forward(**lane_kwargs)
            if isinstance(lane_out, tuple):
                lane_out, log_probs = lane_out
                if log_probs is not None:
                    raise NotImplementedError("TTTv2 DP prefill logprobs are not implemented")
                if output_tokens is None:
                    output_tokens = torch.empty(
                        (batch_size,),
                        dtype=lane_out.dtype,
                        device=lane_out.device,
                    )
                for local_idx, row in enumerate(rows.tolist()):
                    output_tokens[row] = lane_out[local_idx]
                continue
            if output is None:
                output = torch.empty(
                    (batch_size, *lane_out.shape[1:]),
                    dtype=lane_out.dtype,
                    device=lane_out.device,
                )
            for local_idx, row in enumerate(rows.tolist()):
                row_to_result[row] = lane_out[local_idx]

        if output_tokens is not None:
            return output_tokens, None
        if output is None:
            return torch.empty((0, 1, self.model[0].vocab_size))
        for row, row_output in row_to_result.items():
            output[row] = row_output
        return output

    def decode_forward(self, *args, **kwargs):
        if args:
            raise TypeError("TTTv2 DP decode_forward expects keyword arguments")
        Llama3Generator._drop_unsupported_vllm_kwargs(kwargs)
        self._normalize_common_kwargs(kwargs, decode=True)

        tokens = kwargs["tokens"]
        if tokens.dim() == 2 and tokens.shape[-1] == 1:
            tokens = tokens.view(-1)
            kwargs["tokens"] = tokens
        start_pos = kwargs["start_pos"]
        page_table = kwargs["page_table"]
        kv_cache = kwargs.get("kv_cache")
        read_from_device = kwargs.get("read_from_device", True)

        expected_batch = self.per_lane_max_batch_size * self.tt_data_parallel
        if int(tokens.shape[0]) != expected_batch:
            raise ValueError(f"TTTv2 DP decode expects merged batch {expected_batch}, got {tokens.shape[0]}")

        lane_outputs = []
        for lane_idx in range(self.tt_data_parallel):
            start = lane_idx * self.per_lane_max_batch_size
            end = start + self.per_lane_max_batch_size
            lane_kwargs = dict(kwargs)
            lane_kwargs["tokens"] = tokens[start:end]
            lane_kwargs["start_pos"] = start_pos[start:end]
            lane_kwargs["page_table"] = page_table[start:end]
            lane_kwargs["sampling_params"] = self._slice_sampling_params(
                kwargs.get("sampling_params"),
                list(range(start, end)),
            )
            if kv_cache is not None:
                lane_kwargs["kv_cache"] = kv_cache[lane_idx]
            lane_outputs.append(self.lanes[lane_idx].decode_forward(**lane_kwargs))

        if not read_from_device:
            return lane_outputs

        outputs, log_probs = zip(*(out if isinstance(out, tuple) else (out, None) for out in lane_outputs))
        if any(lp is not None for lp in log_probs):
            raise NotImplementedError("TTTv2 DP decode logprobs are not implemented")
        return torch.cat(outputs, dim=0), None

    def read_decode_output(self, tt_out, async_read=False):
        def _read_lane(args):
            lane_idx, lane_out = args
            output = self._read_lane_decode_output(
                lane_idx,
                lane_out,
                blocking=not async_read,
            )
            event = None
            if async_read:
                event = ttnn.record_event(self.mesh_devices[lane_idx], 0)
            return lane_idx, output, event

        lane_results = list(self.output_pool.map(_read_lane, enumerate(tt_out)))
        lane_results.sort(key=lambda item: item[0])
        lane_host_outputs = [item[1] for item in lane_results]
        read_events = [item[2] for item in lane_results if item[2] is not None]

        if async_read:
            return lane_host_outputs, read_events
        return lane_host_outputs

    def process_decode_output_host(self, tt_out, is_tokens=False):
        def _process_lane(args):
            lane_idx, lane_out = args
            output = self._process_lane_decode_output(lane_idx, lane_out, is_tokens)
            return lane_idx, output

        processed = list(self.output_pool.map(_process_lane, enumerate(tt_out)))
        processed.sort(key=lambda item: item[0])
        lane_results = [item[1] for item in processed]

        outputs, log_probs = zip(*(out if isinstance(out, tuple) else (out, None) for out in lane_results))
        if any(lp is not None for lp in log_probs):
            raise NotImplementedError("TTTv2 DP decode logprobs are not implemented")
        if is_tokens:
            global_tokens = torch.empty(
                (self.tt_data_parallel * self.per_lane_max_batch_size,),
                dtype=torch.int32,
                device=outputs[0].device,
            )
            for lane_idx, tokens in enumerate(outputs):
                start = lane_idx * self.per_lane_max_batch_size
                end = start + self.per_lane_max_batch_size
                global_tokens[start:end] = tokens.reshape(self.per_lane_max_batch_size).to(torch.int32)
            return global_tokens, None
        return torch.cat(outputs, dim=0), None

    def _read_lane_decode_output(self, lane_idx, lane_out, blocking):
        out, log_probs = lane_out if isinstance(lane_out, tuple) else (lane_out, None)
        if log_probs is not None:
            raise NotImplementedError("TTTv2 DP decode logprobs are not implemented")
        return (Llama3Generator._read_to_host(out, blocking=blocking), None)

    def _process_lane_decode_output(self, lane_idx, lane_out, is_tokens):
        out, log_probs = lane_out if isinstance(lane_out, tuple) else (lane_out, None)
        if log_probs is not None:
            raise NotImplementedError("TTTv2 DP decode logprobs are not implemented")

        lane_model = self.model[lane_idx]
        batch_size = lane_model.config.max_batch_size
        cluster_shape = list(lane_model.config.mesh_device.shape)
        if is_tokens:
            return _process_output_decode_tokens(out, batch_size, cluster_shape), None
        return (
            _process_output_decode(
                out,
                batch_size,
                lane_model.vocab_size,
                lane_model.num_devices,
                cluster_shape,
            ),
            None,
        )

    def warmup_model_prefill(self, *args, **kwargs):
        kv_cache = kwargs.get("kv_cache")
        for lane_idx, lane in enumerate(self.lanes):
            lane_kwargs = dict(kwargs)
            if kv_cache is not None:
                lane_kwargs["kv_cache"] = kv_cache[lane_idx]
            lane.warmup_model_prefill(*args, **lane_kwargs)

    def warmup_model_decode(
        self,
        kv_cache,
        enable_trace,
        max_batch_size,
        num_blocks,
        can_sample_on_device,
        **kwargs,
    ):
        for lane_idx, lane in enumerate(self.lanes):
            lane.warmup_model_decode(
                kv_cache=kv_cache[lane_idx],
                enable_trace=enable_trace,
                max_batch_size=self.per_lane_max_batch_size,
                num_blocks=num_blocks,
                can_sample_on_device=can_sample_on_device,
                **kwargs,
            )

    @property
    def cache_path(self):
        return self.lanes[0].cache_path

    @staticmethod
    def _normalize_common_kwargs(kwargs, decode):
        Llama3Generator._normalize_tensor_kwarg(kwargs, "tokens", torch.long)
        Llama3Generator._normalize_tensor_kwarg(kwargs, "page_table", torch.int32)
        Llama3Generator._normalize_tensor_kwarg(kwargs, "start_pos", torch.long)
        if not decode:
            Llama3Generator._normalize_tensor_kwarg(kwargs, "prompt_lens", torch.long)

    @staticmethod
    def _slice_sampling_params(sampling_params, rows):
        if sampling_params is None:
            return None
        row_list = rows.tolist() if isinstance(rows, torch.Tensor) else list(rows)
        row_tensor = torch.as_tensor(row_list, dtype=torch.long)

        def _slice_value(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value[row_tensor]
            if isinstance(value, list):
                return [value[row] for row in row_list]
            return value

        return sampling_params.__class__(
            **{field.name: _slice_value(getattr(sampling_params, field.name)) for field in fields(sampling_params)}
        )

    def _prefill_lane_groups(self, empty_slots):
        groups: list[list[tuple[int, int]]] = [[] for _ in range(self.tt_data_parallel)]
        for row, slot in enumerate(empty_slots):
            lane_idx = int(slot) // self.per_lane_max_batch_size
            if lane_idx < 0 or lane_idx >= self.tt_data_parallel:
                raise ValueError(f"empty slot {slot} maps to invalid DP lane {lane_idx}")
            local_slot = int(slot) % self.per_lane_max_batch_size
            groups[lane_idx].append((row, local_slot))

        for lane_idx, row_slots in enumerate(groups):
            if not row_slots:
                continue
            rows = torch.tensor([row for row, _ in row_slots], dtype=torch.long)
            local_slots = [slot for _, slot in row_slots]
            yield lane_idx, rows, local_slots
