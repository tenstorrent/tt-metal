# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Thin vLLM bridge for the datatype-selected Gemma 4 31B TP4 model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

import ttnn
from models.autoports.google_gemma_4_31b.tt.functional_decoder import HF_MODEL_ID
from models.autoports.google_gemma_4_31b.tt.generator import Gemma4Generator, build_generator
from models.vllm_test_utils.generative_base import GenerativeTestModelBase
from vllm.logger import init_logger

DEFAULT_MODEL_DIR = Path("models/autoports/google_gemma_4_31b")
MODEL_DIR_ENV = "GEMMA4_31B_AUTOPORT_DIR"
HOST_SAMPLING_COMPAT_ENV = "GEMMA4_31B_VLLM_HOST_SAMPLING_COMPAT"
REDUCED_LAYERS_ENV = "GEMMA4_31B_VLLM_LAYER_INDICES"
PAGE_BLOCK_SIZE = 64
logger = init_logger(__name__)


class Gemma4ForCausalLM(GenerativeTestModelBase):
    """vLLM interface translation over :class:`Gemma4Generator`.

    Greedy serving reuses the generator's canonical split model/sampler traces.
    The sampler writes directly to the persistent decode-token input, so steady
    decode deliberately ignores stale host token and position values.  The
    optional host compatibility mode exists only for shared tests that request
    penalties, logprobs, or non-greedy sampling.
    """

    model_capabilities = {
        "supports_async_decode": True,
        "supports_async_decode_overlap": True,
        "tt_async_decode_allows_overlap": True,
        "supports_prefix_caching": False,
        "supports_sample_on_device": True,
        "sample_on_device_policy": "greedy_only",
        "supports_device_sampling_penalties": False,
    }
    sample_on_device_policy = "greedy_only"
    _HYBRID_KV_CACHE_GROUPS_ENABLED = True

    # vLLM inspects every registered class against its generic text-generation
    # protocol before the TT worker is created.  The TT runner never calls
    # these methods; execution enters through prefill_forward/decode_forward.
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError("Gemma 4 embeddings execute through the TT serving adapter")

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("Gemma 4 execution enters through prefill_forward/decode_forward")

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("Gemma 4 logits are produced by the TT serving adapter")

    def __init__(
        self,
        generator: Gemma4Generator,
        *,
        requested_max_batch_size: int,
        **_: Any,
    ) -> None:
        self.generator = generator
        self.model = generator.model
        self.mesh_device = generator.mesh_device
        self.max_batch_size = generator.max_batch_size
        self.requested_max_batch_size = int(requested_max_batch_size)
        self.vocab_size = generator.model.vocab_size
        self.host_sampling_compat = os.environ.get(HOST_SAMPLING_COMPAT_ENV, "0") == "1"
        self._page_table_states: list[dict[str, Any] | None] = [None] * len(self.model.layers)
        self._decode_cache_identity: tuple[tuple[int, ...], ...] | None = None
        self._decode_sampling_key: tuple[Any, ...] | None = None
        self._decode_active_batch_size = 0

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size: int,
        max_seq_len: int = 262_144,
        n_layers: int | None = None,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
        **_: Any,
    ) -> "Gemma4ForCausalLM":
        if optimizations is not None:
            raise ValueError("Gemma 4 31B autoport always loads doc/datatype_sweep/selected_precision_config.json")
        if int(tt_data_parallel) != 1:
            raise ValueError(f"Gemma 4 31B autoport supports tt_data_parallel=1, got {tt_data_parallel}")
        if not 1 <= int(max_batch_size) <= 32:
            raise ValueError(f"Gemma 4 31B supports max_batch_size in [1, 32], got {max_batch_size}")

        model_dir = Path(os.environ.get(MODEL_DIR_ENV, DEFAULT_MODEL_DIR)).resolve()
        context_contract = _read_context_contract(model_dir)
        supported_context = int(
            context_contract.get("vllm_supported_context", context_contract["current_supported_context"])
        )
        if int(max_seq_len) != supported_context:
            raise ValueError(
                f"Gemma 4 31B serving must use context_contract max_model_len={supported_context}, "
                f"got {max_seq_len}"
            )

        model_mesh = _select_tp4_mesh(mesh_device)
        text_config = getattr(hf_config, "text_config", hf_config)
        hf_model = os.environ.get("HF_MODEL") or getattr(hf_config, "_name_or_path", None) or HF_MODEL_ID
        layer_indices = _requested_layer_indices(text_config, n_layers=n_layers)
        generator = build_generator(
            model_dir=model_dir,
            mesh_device=model_mesh,
            model_id_or_path=hf_model,
            max_batch_size=int(max_batch_size),
            max_seq_len=supported_context,
            layer_indices=layer_indices,
            allocate_standalone_cache=False,
        )
        if generator.model.config.precision_config_id != "lm_head_bfp8_hifi2":
            raise RuntimeError(
                "vLLM constructed the wrong precision policy: " f"{generator.model.config.precision_config_id!r}"
            )
        return cls(generator, requested_max_batch_size=max_batch_size)

    @classmethod
    def get_max_tokens_all_users(
        cls,
        *,
        model_name: str,
        num_devices: int,
        tt_data_parallel: int,
        max_model_len: int,
        max_num_seqs: int,
    ) -> int:
        del model_name, num_devices, tt_data_parallel
        # vLLM disables chunked prefill on TT.  A newly-admitted request must
        # therefore reserve its whole prompt in each hybrid cache group before
        # the first model call.  Gemma 4 has five 10-layer sliding groups and
        # one 10-layer global group. Page-size unification keeps sliding blocks
        # at 64 tokens and widens the global view to 128 tokens.
        sliding_groups = 5
        sliding_blocks = _ceil_div(int(max_model_len), PAGE_BLOCK_SIZE) + 1
        global_blocks = _ceil_div(int(max_model_len), 2 * PAGE_BLOCK_SIZE)
        required_pool_blocks = sliding_groups * sliding_blocks + global_blocks

        # get_num_available_blocks_tt adds these two generic terms after this
        # hook returns. Subtract them here so its final override is the exact
        # full-context hybrid pool requirement, not a second approximation.
        generic_batch_padding_blocks = int(max_num_seqs)
        generic_sliding_headroom_blocks = _ceil_div(1024 * int(max_num_seqs) * 8, PAGE_BLOCK_SIZE)
        model_blocks = required_pool_blocks - generic_batch_padding_blocks - generic_sliding_headroom_blocks
        if model_blocks <= 0:
            raise ValueError("Gemma 4 hybrid KV budget resolved to a non-positive model allocation")
        return model_blocks * PAGE_BLOCK_SIZE

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        text_config = getattr(model_config.hf_config, "text_config", model_config.hf_config)
        layer_types = list(text_config.layer_types)
        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )
        block_size = int(cache_config.block_size)
        if block_size != PAGE_BLOCK_SIZE:
            raise ValueError(f"Gemma 4 31B requires vLLM --block-size {PAGE_BLOCK_SIZE}, got {block_size}")

        specs = {}
        for layer_idx, layer_kind in enumerate(layer_types):
            name = f"model.layers.{layer_idx}.self_attn"
            if layer_kind == "sliding_attention":
                specs[name] = SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=int(text_config.num_key_value_heads),
                    head_size=int(text_config.head_dim),
                    dtype=dtype,
                    sliding_window=int(text_config.sliding_window),
                )
            elif layer_kind == "full_attention":
                specs[name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=int(text_config.num_global_key_value_heads),
                    head_size=int(text_config.global_head_dim),
                    dtype=dtype,
                )
            else:
                raise ValueError(f"unsupported Gemma 4 layer type {layer_kind!r} at layer {layer_idx}")
        return specs

    @property
    def cache_path(self) -> Path:
        path = self.generator.model_dir / "tt_cache" / "vllm_kv"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        selected = [per_layer_specs[index] for index in self.model.layer_indices]
        unique_buffers: dict[int, tuple[int, ttnn.DataType, list[ttnn.Tensor]]] = {}
        kv_cache: list[list[ttnn.Tensor]] = []
        for local_idx, (shape, source_dtype, tensor_idx) in enumerate(selected):
            shape = tuple(int(value) for value in shape)
            layer = self.model.layers[local_idx]
            cache_dtype = layer.policy.kv_cache_dtype
            existing = unique_buffers.get(int(tensor_idx))
            if existing is not None:
                existing_numel, existing_dtype, buffers = existing
                numel = int(torch.tensor(shape).prod().item())
                if existing_numel != numel or existing_dtype != cache_dtype:
                    raise ValueError(
                        "vLLM attempted to share one KV tensor across incompatible Gemma 4 specs: "
                        f"tensor_idx={tensor_idx}, first={(existing_numel, existing_dtype)}, "
                        f"current={(numel, cache_dtype)}"
                    )
                kv_cache.append(buffers)
                continue

            source = torch.zeros(shape, dtype=source_dtype)
            buffers = [
                ttnn.as_tensor(
                    source,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=cache_dtype,
                    cache_file_name=self.cache_path
                    / f"layer_{self.model.layer_indices[local_idx]}_tensor_{tensor_idx}_{kind}_{shape}_{cache_dtype}",
                )
                for kind in ("k", "v")
            ]
            unique_buffers[int(tensor_idx)] = (int(source.numel()), cache_dtype, buffers)
            kv_cache.append(buffers)
        return kv_cache

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self.allocate_kv_cache_per_layer([(kv_cache_shape, dtype, index) for index in range(num_layers)])

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens: Iterable[int] | torch.Tensor | None = None,
        sampling_params=None,
        page_tables_per_layer=None,
        **_: Any,
    ):
        # A new prefill starts a new serving sequence. Release any prior
        # model/sampler traces before page-table conversion can allocate or
        # update device buffers; TT allocations are unsafe while a trace owns
        # allocator addresses.
        self._release_decode_state()
        kv_cache = self._require_kv_cache(kv_cache)
        prompt_lens_list = _to_int_list(prompt_lens, default=int(tokens.shape[1]))
        page_tables, _ = self._page_tables_to_tt(page_tables_per_layer, page_table)
        if sampling_params is None:
            self._require_host_sampling_compat()
            return self.generator.prefill_forward(
                tokens.to(torch.long),
                page_table=page_tables,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens_list,
                return_all_logits=False,
            )

        _require_semantic_greedy(sampling_params)
        logits = self.generator.prefill_forward(
            tokens.to(torch.long),
            page_table=page_tables,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens_list,
            return_device_logits=True,
        )
        output_buffer = self.generator._new_token_buffer(len(prompt_lens_list))
        sampled, _ = self.generator._sample_eager(
            logits,
            tt_out_tok=output_buffer,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
        )
        tokens_host = _tokens_to_host(sampled, active_batch=len(prompt_lens_list))
        logits.deallocate(True)
        output_buffer.deallocate(True)
        return tokens_host.reshape(len(prompt_lens_list), 1)

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table=None,
        kv_cache=None,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params=None,
        reset_batch: bool = False,
        page_tables_per_layer=None,
        perform_device_sampling: bool | None = None,
        prompt_tokens=None,
        output_tokens=None,
        slot_remap=None,
        **_: Any,
    ):
        del prompt_tokens, output_tokens, slot_remap, perform_device_sampling
        if sampling_params is None:
            # The runner may switch an in-flight request from device-greedy to
            # the optional vLLM host sampler as the active batch changes. An
            # eager decode must never run with the prior token-out trace live.
            self._release_decode_state()
        kv_cache = self._require_kv_cache(kv_cache)

        if sampling_params is None:
            self._require_host_sampling_compat()
            page_tables, _ = self._page_tables_to_tt(page_tables_per_layer, page_table)
            output = self.generator.decode_forward(
                tokens.to(torch.long),
                start_pos.reshape(-1).to(torch.int32),
                page_table=page_tables,
                kv_cache=kv_cache,
                enable_trace=False,
                return_device_logits=not read_from_device,
            )
            return output

        if not enable_trace:
            raise ValueError("Gemma 4 31B on-device serving sampling requires decode tracing")
        _require_semantic_greedy(sampling_params)
        positions = start_pos.reshape(-1).to(torch.int32)
        active = positions >= 0
        active_batch_size = int(active.sum().item())
        if active_batch_size < 1:
            return torch.empty((0, 1), dtype=torch.int32)
        if not bool(active[:active_batch_size].all()) or bool(active[active_batch_size:].any()):
            raise ValueError("Gemma 4 31B dynamic decode requires active requests in a contiguous prefix")
        trace_tokens = tokens.reshape(tokens.shape[0], -1)[:active_batch_size, 0].to(torch.int32)
        trace_positions = positions[:active_batch_size]
        cache_identity = _kv_cache_identity(kv_cache)
        sampling_key = ("greedy", active_batch_size)
        cache_changed = self._decode_cache_identity != cache_identity
        must_prepare = (
            bool(reset_batch)
            or self.model.trace_state.trace_id is None
            or cache_changed
            or self._decode_sampling_key != sampling_key
        )
        if must_prepare:
            # Dynamic-B page-table buffers can change shape.  Release and
            # synchronize both nonblocking traces before their conversion is
            # allowed to allocate a replacement buffer.
            self._release_decode_state()
        page_tables, generations = self._page_tables_to_tt(page_tables_per_layer, page_table, rows=active_batch_size)
        if must_prepare:
            logger.info(
                "Gemma 4 decode trace prepare: active_batch=%d cache_changed=%s reset_batch=%s",
                active_batch_size,
                cache_changed,
                bool(reset_batch),
            )
            output = self.generator.prepare_token_out_decode(
                first_input_tokens=trace_tokens,
                start_positions=trace_positions,
                page_table=page_tables,
                kv_cache=kv_cache,
                page_table_generations=generations,
                prompt_lengths=[int(value) for value in trace_positions.tolist()],
                active_batch_size=active_batch_size,
                pad_to_max_batch=False,
                top_k=1,
                top_p=0.0,
                temperature=1.0,
            )
            logger.info(
                "Gemma 4 decode traces ready: active_batch=%d model_trace_id=%s sampler_trace_id=%s",
                active_batch_size,
                self.model.trace_state.trace_id,
                self.generator._sampling_trace_id,
            )
            self._decode_cache_identity = cache_identity
            self._decode_sampling_key = sampling_key
        else:
            # With async scheduling the host token and position can be one step
            # stale here.  The split sampler already wrote the emitted token to
            # trace_state.token_input and the model trace advanced both position
            # tensors, so only scheduler-owned page-table changes are refreshed.
            output = self.generator.decode_next_token_traced(
                page_table=page_tables,
                kv_cache=kv_cache,
                page_table_generations=generations,
            )
        self._decode_active_batch_size = active_batch_size
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(output), is_tokens=True)
        return output

    def read_decode_output(self, tt_out: Any, async_read: bool = False):
        host_out, events = _read_tt_output(tt_out, self.mesh_device, async_read=async_read)
        return (host_out, events) if async_read else host_out

    def process_decode_output_host(self, tt_out: Any, *, is_tokens: bool = True):
        if is_tokens:
            return _tokens_to_host(tt_out, active_batch=None).reshape(-1, 1)
        return _logits_to_host(tt_out, self.mesh_device, self.vocab_size)

    def warmup_model_prefill(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.reset_warmup_state()

    def warmup_model_decode(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.reset_warmup_state()

    def reset_warmup_state(self) -> None:
        self._release_decode_state()

    def _release_decode_state(self) -> None:
        had_live_trace = self.generator._sampling_trace_id is not None or self.model.trace_state.trace_id is not None
        self.generator._release_all_decode_traces()
        if had_live_trace:
            counters = self.generator.trace_lifecycle_counters
            logger.info(
                "Gemma 4 decode traces released after CQ synchronization: releases=%d synchronizations=%d",
                counters["release_calls"],
                counters["release_synchronizations"],
            )
        self._decode_cache_identity = None
        self._decode_sampling_key = None
        self._decode_active_batch_size = 0

    def _require_kv_cache(self, kv_cache):
        if kv_cache is None:
            raise ValueError("Gemma 4 31B vLLM path requires the vLLM-owned kv_cache")
        return kv_cache

    def _require_host_sampling_compat(self) -> None:
        if not self.host_sampling_compat:
            raise ValueError(
                f"host sampling compatibility is disabled; set {HOST_SAMPLING_COMPAT_ENV}=1 only for shared "
                "sampling/logprob compatibility tests"
            )

    def _page_tables_to_tt(self, page_tables_per_layer, page_table, *, rows: int | None = None):
        raw = page_tables_per_layer
        if raw is None:
            if page_table is None:
                raise ValueError("vLLM must provide a page table")
            raw = [page_table] * (max(self.model.layer_indices) + 1)
        raw = list(raw)
        if max(self.model.layer_indices) >= len(raw):
            raise ValueError("vLLM page_tables_per_layer does not cover every selected model layer")
        selected = [_page_table_prefix(raw[index], rows=rows) for index in self.model.layer_indices]

        groups: dict[tuple[Any, ...], list[int]] = {}
        for local_idx, table in enumerate(selected):
            groups.setdefault(_page_table_source_key(table), []).append(local_idx)

        for members in groups.values():
            first = members[0]
            source = selected[first]
            previous = self._page_table_states[first]
            state = _update_page_table_state(previous, source, self.mesh_device)
            for local_idx in members:
                self._page_table_states[local_idx] = state

        if any(state is None for state in self._page_table_states):
            raise RuntimeError("page-table conversion did not initialize every selected layer")
        states = [state for state in self._page_table_states if state is not None]
        return [state["device"] for state in states], [int(state["generation"]) for state in states]


def _read_context_contract(model_dir: Path) -> dict[str, Any]:
    import json

    path = model_dir / "doc/context_contract.json"
    with path.open() as handle:
        return json.load(handle)


def _requested_layer_indices(text_config, *, n_layers: int | None) -> tuple[int, ...] | None:
    raw = os.environ.get(REDUCED_LAYERS_ENV)
    if raw:
        indices = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
        if not indices:
            raise ValueError(f"{REDUCED_LAYERS_ENV} must name at least one layer")
        return indices
    if n_layers is not None:
        return tuple(range(int(n_layers)))
    return None


def _select_tp4_mesh(mesh_device):
    if not isinstance(mesh_device, ttnn.MeshDevice):
        return mesh_device
    if tuple(mesh_device.shape) == (1, 4) and mesh_device.get_num_devices() == 4:
        return mesh_device
    if mesh_device.get_num_devices() >= 4 and tuple(mesh_device.shape)[0] == 1:
        return mesh_device.create_submeshes(ttnn.MeshShape(1, 4))[0]
    raise ValueError(f"Gemma 4 31B requires a 1x4 TP4 mesh, got shape={tuple(mesh_device.shape)}")


def _to_int_list(values, *, default: int) -> list[int]:
    if values is None:
        return [int(default)]
    if isinstance(values, torch.Tensor):
        return [int(value) for value in values.reshape(-1).tolist()]
    return [int(value) for value in values]


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _sampling_values(sampling_params, name: str) -> list[Any]:
    value = getattr(sampling_params, name)
    if isinstance(value, torch.Tensor):
        return value.reshape(-1).tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _require_semantic_greedy(sampling_params) -> None:
    temperatures = _sampling_values(sampling_params, "temperature")
    top_ks = _sampling_values(sampling_params, "top_k")
    if len(temperatures) != len(top_ks):
        raise ValueError("temperature and top_k must contain one value per fixed slot")
    if not all(float(temp) <= 0.0 or int(top_k) == 1 for temp, top_k in zip(temperatures, top_ks)):
        raise ValueError("on-device Gemma 4 31B serving sampling is greedy-only")


def _kv_cache_identity(kv_cache) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(id(tensor) for tensor in pair) for pair in kv_cache)


def _page_table_source_key(table) -> tuple[Any, ...]:
    if isinstance(table, torch.Tensor):
        return (
            "torch",
            int(table.untyped_storage().data_ptr()),
            int(table.storage_offset()),
            tuple(table.shape),
            tuple(table.stride()),
        )
    if isinstance(table, ttnn.Tensor):
        return ("ttnn", id(table))
    host = torch.as_tensor(table)
    return ("object", id(table), tuple(host.shape))


def _page_table_prefix(table, *, rows: int | None):
    if rows is None:
        return table
    if int(rows) < 1:
        raise ValueError("page-table row count must be positive")
    if isinstance(table, ttnn.Tensor):
        if int(table.shape[0]) != int(rows):
            raise ValueError("dynamic decode requires TT page tables to already match the active batch")
        return table
    host = torch.as_tensor(table)
    if host.ndim != 2 or int(host.shape[0]) < int(rows):
        raise ValueError("page table must be rank-2 with at least one row per active request")
    return host[: int(rows)]


def _update_page_table_state(previous, source, mesh_device):
    if isinstance(source, ttnn.Tensor):
        if previous is not None and previous["source_identity"] == id(source):
            return previous
        return {
            "host": None,
            "device": source,
            "generation": 0 if previous is None else int(previous["generation"]) + 1,
            "source_identity": id(source),
            "owned": False,
        }

    host = torch.as_tensor(source, dtype=torch.int32).contiguous()
    if previous is not None and previous["host"] is not None and torch.equal(host, previous["host"]):
        return previous
    if previous is None or tuple(previous["device"].shape) != tuple(host.shape):
        if previous is not None and previous.get("owned", False) and previous["device"].is_allocated():
            previous["device"].deallocate(True)
        device = ttnn.from_torch(
            host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        generation = 0 if previous is None else int(previous["generation"]) + 1
    else:
        host_tt = ttnn.from_torch(
            host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host_tt, previous["device"])
        device = previous["device"]
        generation = int(previous["generation"]) + 1
    return {
        "host": host.clone(),
        "device": device,
        "generation": generation,
        "source_identity": id(source),
        "owned": True,
    }


def _read_tt_output(obj: Any, mesh_device, *, async_read: bool) -> tuple[Any, list[Any]]:
    if obj is None or isinstance(obj, torch.Tensor):
        return obj, []
    if isinstance(obj, tuple):
        values = []
        events = []
        for item in obj:
            value, item_events = _read_tt_output(item, mesh_device, async_read=async_read)
            values.append(value)
            events.extend(item_events)
        return tuple(values), events
    if isinstance(obj, ttnn.Tensor):
        if async_read:
            host = ttnn.from_device(obj, blocking=False, cq_id=0)
            return host, [ttnn.record_event(mesh_device, 0)]
        return ttnn.from_device(obj, blocking=True, cq_id=0), []
    raise TypeError(f"unsupported TT decode output type {type(obj).__name__}")


def _tokens_to_host(obj: Any, *, active_batch: int | None) -> torch.Tensor:
    if isinstance(obj, tuple):
        return _tokens_to_host(obj[0], active_batch=active_batch)
    if isinstance(obj, ttnn.Tensor):
        obj = ttnn.to_torch(ttnn.get_device_tensors(obj)[0])
    elif not isinstance(obj, torch.Tensor):
        obj = torch.as_tensor(obj)
    tokens = obj.reshape(-1).to(torch.int32)
    if active_batch is not None:
        tokens = tokens[: int(active_batch)]
    return tokens.contiguous()


def _logits_to_host(obj: Any, mesh_device, vocab_size: int) -> torch.Tensor:
    if isinstance(obj, tuple):
        obj = obj[0]
    if isinstance(obj, ttnn.Tensor):
        obj = ttnn.to_torch(obj, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    elif not isinstance(obj, torch.Tensor):
        obj = torch.as_tensor(obj)
    if obj.dim() == 2:
        obj = obj.unsqueeze(1)
    elif obj.dim() == 4 and obj.shape[0] == 1 and obj.shape[1] == 1:
        obj = obj.reshape(obj.shape[2], 1, obj.shape[3])
    elif obj.dim() > 3:
        obj = obj.reshape(-1, obj.shape[-2], obj.shape[-1])
    return obj.to(torch.float32)[..., :vocab_size].contiguous()


def allocate_vllm_kv_cache(*args: Any, **kwargs: Any):
    del args, kwargs
    raise NotImplementedError("Use Gemma4ForCausalLM.allocate_kv_cache after initialize_vllm_model")


__all__ = ["Gemma4ForCausalLM", "allocate_vllm_kv_cache"]
