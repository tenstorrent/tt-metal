# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Thin vLLM adapter for the GPT-OSS 120B autoport.

The adapter translates vLLM scheduler state into :mod:`tt.generator` calls.
It does not implement a sampler: device sampling is the full-model generator's
canonical split model/sampling trace, with ``tt_out_tok`` feeding the next
decode replay.  Host logits exist only as an explicit compatibility path for
requests the shared plugin cannot sample on device.
"""

from __future__ import annotations

import atexit
import gc
import json
import math
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

_LOG_PROGRAM_CACHE = os.environ.get("GPT_OSS_120B_LOG_PROGRAM_CACHE") == "1"
_LOG_SLOTS = os.environ.get("GPT_OSS_120B_LOG_SLOTS") == "1"
_TABLES_ALWAYS = os.environ.get("GPT_OSS_120B_TABLES_ALWAYS") == "1"
_TABLES_SYNC = os.environ.get("GPT_OSS_120B_TABLES_SYNC") == "1"
_TABLES_VERIFY = os.environ.get("GPT_OSS_120B_TABLES_VERIFY") == "1"

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss_120b_qb2.tt.generator import GREEDY, Generator
from models.demos.gpt_oss_120b_qb2.tt.model import (
    HF_CONTEXT_LENGTH,
    MODEL_LAYERS,
    FullModelCapacityError,
    Model,
    decode_trace_buckets,
)
from models.demos.gpt_oss_120b_qb2.tt.precision import dtype_name
from models.demos.gpt_oss_120b_qb2.tt.sliding_ring import (
    PAGE_SIZE,
    PREFILL_CHUNK_ALIGN,
    SLIDING_RING_BLOCKS,
    SLIDING_RING_TOKENS,
    sliding_ring_enabled,
)
from models.tt_transformers.tt.common import get_padded_prefill_len

MAX_CONCURRENT_SEQS = 32
FULL_CONTEXT_BLOCKS = math.ceil(HF_CONTEXT_LENGTH / PAGE_SIZE)
KV_POOL_BLOCKS_DEFAULT = 4640
ENV_KV_POOL_BLOCKS = "GPT_OSS_120B_KV_POOL_BLOCKS"
ENV_CHUNK_WARMUP = "GPT_OSS_120B_CHUNK_WARMUP"
ENV_PREFIX_CACHING = "GPT_OSS_120B_PREFIX_CACHING"
COLD_RING_REPREFILL_TOKENS = 2560
CHUNK_WARMUP_DEFAULT = "8192"
SLIDING_RING_ENABLED = sliding_ring_enabled()
PREFIX_CACHING_ENABLED = SLIDING_RING_ENABLED and os.environ.get(ENV_PREFIX_CACHING, "1") != "0"
LEGACY_SCHEDULER_KV_GROUPS = 2
ENV_REDUCED_LAYERS = "GPT_OSS_120B_VLLM_NUM_LAYERS"
_NOT_THE_TT_PATH = (
    "vLLM uses initialize_vllm_model(), prefill_forward(), and decode_forward() "
    "for TT execution; this method exists only for vLLM model introspection"
)


def _freeze(value):
    """Return a comparable sampling-state key without retaining tensor views."""

    if isinstance(value, torch.Tensor):
        return (
            str(value.dtype),
            tuple(value.shape),
            tuple(value.detach().cpu().reshape(-1).tolist()),
        )
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _sampling_key(sampling_params) -> tuple | None:
    if sampling_params is None:
        return None
    if is_dataclass(sampling_params):
        return tuple((field.name, _freeze(getattr(sampling_params, field.name))) for field in fields(sampling_params))
    return _freeze(vars(sampling_params))


def _sampling_params_to_host_values(sampling_params):
    """Materialize tensor-backed vLLM parameters only when device state changes."""

    if sampling_params is None or not is_dataclass(sampling_params):
        return sampling_params
    values = {}
    for field in fields(sampling_params):
        value = getattr(sampling_params, field.name)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        if field.name == "seed":
            if isinstance(value, list):
                value = [None if seed == -1 else seed for seed in value]
            elif value == -1:
                value = None
        values[field.name] = value
    return type(sampling_params)(**values)


def _all_none(value) -> bool:
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return value.numel() == 0 or bool(torch.all(value == -1))
    if isinstance(value, (list, tuple)):
        return all(item is None for item in value)
    return False


def _all_false(value) -> bool:
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return not bool(value.any())
    if isinstance(value, (list, tuple)):
        return not any(bool(item) for item in value)
    return not bool(value)


def _all_equal(value, expected) -> bool:
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return bool(torch.all(value == expected))
    if isinstance(value, (list, tuple)):
        return all(item == expected for item in value)
    return value == expected


def kv_pool_blocks(max_num_seqs: int) -> int:
    """Blocks of the vLLM-managed pool that backs the 18 full-attention layers."""

    raw = os.environ.get(ENV_KV_POOL_BLOCKS, "").strip()
    blocks = int(raw) if raw else KV_POOL_BLOCKS_DEFAULT
    minimum = FULL_CONTEXT_BLOCKS + int(max_num_seqs)
    maximum = int(max_num_seqs) * (FULL_CONTEXT_BLOCKS + 1)
    if not minimum <= blocks <= maximum:
        raise ValueError(
            f"{ENV_KV_POOL_BLOCKS}={blocks} is outside [{minimum}, {maximum}]: the pool must hold one "
            f"{HF_CONTEXT_LENGTH}-token request plus one output block per sequence, and more than "
            f"{max_num_seqs} full-context requests can never be admitted"
        )
    return blocks


def sliding_layer_indices(hf_config, n_layers: int) -> list[int]:
    text_config = getattr(hf_config, "text_config", hf_config)
    layer_types = list(getattr(text_config, "layer_types", None) or [])
    if not layer_types:
        return [layer_idx for layer_idx in range(int(n_layers)) if layer_idx % 2 == 0]
    return [layer_idx for layer_idx, kind in enumerate(layer_types[: int(n_layers)]) if kind == "sliding_attention"]


def ring_page_table(rings, columns: int, valid_blocks=None, base: int = 0) -> torch.Tensor:
    """Cyclic page table over each row's ring: column c maps to block base + ring*W + c mod W.

    Written through the ordinary paged-cache kernels, position p lands in ring
    slot p mod SLIDING_RING_TOKENS, and decode attention reads the 128-token
    window through the same mapping. Columns at or beyond ``valid_blocks`` are
    -1, which the fill kernel skips, so a padded prefill tail never lands in
    the ring. ``base`` is the first ring block: the rings sit above the
    vLLM-managed pool inside the paired full-attention layer's buffer, so the
    table keeps its full column count (the paged kernels require fewer columns
    than the buffer has blocks) while vLLM never hands out a ring block.
    """

    rows = len(rings)
    ring_base = torch.tensor([int(ring) for ring in rings], dtype=torch.int32).reshape(rows, 1) * SLIDING_RING_BLOCKS
    offsets = (torch.arange(int(columns), dtype=torch.int32) % SLIDING_RING_BLOCKS).reshape(1, int(columns))
    table = int(base) + ring_base + offsets
    if valid_blocks is not None:
        limit = torch.tensor([int(blocks) for blocks in valid_blocks], dtype=torch.int32).reshape(rows, 1)
        keep = torch.arange(int(columns), dtype=torch.int32).reshape(1, int(columns)) < limit
        table = torch.where(keep, table, torch.full_like(table, -1))
    return table


class TTGptOssForCausalLM:
    """vLLM's TT model surface backed by the completed full-model generator."""

    decode_input_update_contract = 1
    _HYBRID_KV_CACHE_GROUPS_ENABLED = not SLIDING_RING_ENABLED
    # Both widths are prepared during warmup. Advertising them lets the shared
    # runner keep singleton host-sampling requests at B1 instead of padding an
    # exact full-vocabulary fallback to the serving-width B32 eager graph.
    tt_supported_decode_batch_sizes = decode_trace_buckets(MAX_CONCURRENT_SEQS)
    model_capabilities = {
        "supports_prefix_caching": PREFIX_CACHING_ENABLED,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
        "supports_batched_prefill": True,
        "supports_chunked_prefill": True,
        "fabric_config": {"config": ttnn.FabricConfig.FABRIC_1D_RING},
    }

    def __init__(
        self,
        model: Model,
        model_args,
        *,
        mesh_device,
        max_batch_size: int,
        max_model_len: int,
        hf_config,
        vllm_config=None,
    ):
        del vllm_config
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.max_batch_size = int(max_batch_size)
        # Advertise only the buckets this instance warms and captures.
        self.tt_supported_decode_batch_sizes = decode_trace_buckets(self.max_batch_size)
        self.max_model_len = int(max_model_len)
        self.hf_config = hf_config
        self.generator: Generator | None = None
        self.page_table_blocks = math.ceil(self.max_model_len / PAGE_SIZE)
        self._last_page_tables: list[torch.Tensor | Any] | None = None
        self._last_page_table_state_id: Any | None = None
        self._last_sampling_key: tuple | None = None
        self._last_decode_was_device_sampled = True
        self._active_decode_bucket: int | None = None
        self._decode_bucket_compile_key: tuple | None = None
        self._prepared_decode_buckets: dict[int, Any] = {}
        self._bucket_page_table_store: dict[int, list[Any] | None] = {}
        self._host_sampling_active = False
        self._device_trace_recapture_requires_reset = False
        self._cache_tensor_indices: list[int] = []
        self._cache_shapes: list[tuple[int, ...]] = []
        self._sliding_layers: list[int] = (
            sliding_layer_indices(hf_config, self.model.n_layers) if SLIDING_RING_ENABLED else []
        )
        self._ring_of_slot: list[int] = list(range(self.max_batch_size))
        self._ring_block_base = 0
        self._ring_tables_dirty = False
        self._slot_prefill_end: list[int | None] = [None] * self.max_batch_size
        self._prefill_shapes_warm = False
        self.serving_counters = {
            "prefill_calls": 0,
            "decode_calls": 0,
            "device_sampled_decodes": 0,
            "host_sampled_decodes": 0,
            "sampling_state_pushes": 0,
            "sampling_state_reuses": 0,
            "host_trace_releases": 0,
            "device_trace_recaptures": 0,
            "page_table_refreshes": 0,
            "page_table_reuses": 0,
            "async_reads": 0,
        }
        atexit.register(self._write_serving_capability)

    # vLLM checks this surface structurally; TT execution never calls it.
    def embed_input_ids(self, input_ids):  # pragma: no cover
        raise NotImplementedError(_NOT_THE_TT_PATH)

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover
        raise NotImplementedError(_NOT_THE_TT_PATH)

    def compute_logits(self, hidden_states):  # pragma: no cover
        raise NotImplementedError(_NOT_THE_TT_PATH)

    @classmethod
    def get_max_tokens_all_users(
        cls,
        *,
        model_name,
        num_devices,
        tt_data_parallel,
        max_model_len,
        max_num_seqs,
    ) -> int:
        del model_name
        if int(tt_data_parallel) != 1:
            raise ValueError("GPT-OSS 120B uses one TP mesh; TT data parallelism is not supported")
        if int(num_devices) != 4:
            raise FullModelCapacityError(
                "GPT-OSS 120B needs P150x4: its fixed resident state exceeds 32 GiB/device on P150/P150x2"
            )
        if int(max_model_len) != HF_CONTEXT_LENGTH:
            raise ValueError(
                f"served max_model_len must match context_contract.json ({HF_CONTEXT_LENGTH}), got {max_model_len}"
            )
        if not 1 <= int(max_num_seqs) <= MAX_CONCURRENT_SEQS:
            raise ValueError(
                f"GPT-OSS 120B vLLM serving supports 1..{MAX_CONCURRENT_SEQS} concurrent sequences; "
                f"got max_num_seqs={max_num_seqs}"
            )
        if not SLIDING_RING_ENABLED:
            return LEGACY_SCHEDULER_KV_GROUPS * HF_CONTEXT_LENGTH
        return (kv_pool_blocks(max_num_seqs) - int(max_num_seqs)) * PAGE_SIZE

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Describe the full-attention layers to vLLM; sliding layers use device rings."""

        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if int(cache_config.block_size) != PAGE_SIZE:
            raise ValueError(f"GPT-OSS serving requires --block-size {PAGE_SIZE}")
        hf_config = getattr(model_config.hf_config, "text_config", model_config.hf_config)
        layer_types = list(hf_config.layer_types)
        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )
        common = {
            "block_size": PAGE_SIZE,
            "num_kv_heads": model_config.get_num_kv_heads(vllm_config.parallel_config),
            "head_size": model_config.get_head_size(),
            "dtype": dtype,
        }
        result = {}
        for layer_idx, layer_type in enumerate(layer_types):
            name = f"model.layers.{layer_idx}.self_attn"
            if layer_type == "sliding_attention":
                if not SLIDING_RING_ENABLED:
                    result[name] = SlidingWindowSpec(sliding_window=int(hf_config.sliding_window), **common)
                continue
            if layer_type == "full_attention":
                result[name] = FullAttentionSpec(**common)
            else:
                raise ValueError(f"unsupported GPT-OSS layer type {layer_type!r} at layer {layer_idx}")
        if not result:
            raise ValueError("GPT-OSS layer_types name no full_attention layer")
        return result

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ) -> "TTGptOssForCausalLM":
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unknown GPT-OSS vLLM initialization arguments: {unknown}")
        if int(tt_data_parallel) != 1:
            raise ValueError("GPT-OSS 120B vLLM integration supports tensor parallelism on one TT mesh only")
        if not 1 <= int(max_batch_size) <= MAX_CONCURRENT_SEQS:
            raise ValueError(
                f"GPT-OSS 120B vLLM serving supports 1..{MAX_CONCURRENT_SEQS} concurrent sequences; "
                f"got max_batch_size={max_batch_size}"
            )
        if optimizations is not None:
            raise ValueError("GPT-OSS serving always loads the selected datatype-sweep policy")
        if int(max_seq_len) != HF_CONTEXT_LENGTH:
            raise ValueError(
                f"served max_model_len must match context_contract.json ({HF_CONTEXT_LENGTH}), got {max_seq_len}"
            )

        env_layers = os.environ.get(ENV_REDUCED_LAYERS)
        num_layers = int(n_layers or env_layers or MODEL_LAYERS)
        if not 1 <= num_layers <= MODEL_LAYERS:
            raise ValueError(f"{ENV_REDUCED_LAYERS} must be within [1, {MODEL_LAYERS}]")
        snapshot = os.environ.get("GPT_OSS_120B_SNAPSHOT")
        model, model_args = Model.from_checkpoint(
            mesh_device,
            snapshot_path=snapshot,
            max_batch_size=int(max_batch_size),
            max_context_length=int(max_seq_len),
            num_layers=num_layers,
            allow_reduced_model=num_layers != MODEL_LAYERS,
            create_kv_cache=False,
            salt_duplicate_seeds=False,
        )
        return cls(
            model,
            model_args,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_model_len=max_seq_len,
            hf_config=hf_config,
        )

    @property
    def cache_path(self) -> Path:
        return Path(self.model_args.tensor_cache_path) / "vllm_kv_cache"

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Allocate and bind exactly the shared buffers vLLM sized."""

        if self.generator is not None:
            raise RuntimeError("vLLM KV cache was already allocated")
        if len(per_layer_specs) < self.model.n_layers:
            raise ValueError(
                f"vLLM supplied {len(per_layer_specs)} cache specs for {self.model.n_layers} resident layers"
            )
        self.cache_path.mkdir(parents=True, exist_ok=True)
        selected_dtype = self.model.precision_config.decoder_policy_for_layer(0).kv_cache_dtype
        unique_buffers: dict[int, list] = {}
        layer_cache = []
        tensor_indices = []
        cache_shapes = []
        sliding = set(self._sliding_layers)
        ring_blocks = SLIDING_RING_BLOCKS * self.max_batch_size
        full_layers = [idx for idx in range(min(len(per_layer_specs), self.model.n_layers)) if idx not in sliding]
        if sliding and not full_layers:
            raise ValueError("sliding-window rings need at least one resident full-attention layer to share a buffer")
        ring_host: dict[int, int] = {}
        for layer_idx in sorted(sliding):
            later = [idx for idx in full_layers if idx > layer_idx]
            ring_host[layer_idx] = later[0] if later else full_layers[-1]
        pool_blocks = None

        def allocate(shape, name):
            host = torch.zeros(shape, dtype=torch.bfloat16)
            return [
                ttnn.as_tensor(
                    host,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=selected_dtype,
                    cache_file_name=get_cache_file_name(self.cache_path, f"empty_{kind}_{name}"),
                )
                for kind in ("k", "v")
            ]

        # A reduced-layer hardware smoke still carries the checkpoint's full
        # HF config through vLLM. Bind only its explicitly resident prefix;
        # production consumes every spec because all 36 layers are resident.
        for layer_idx, (shape, _torch_dtype, tensor_idx) in enumerate(per_layer_specs[: self.model.n_layers]):
            shape = tuple(int(dim) for dim in shape)
            if shape[1:] != (
                8 // int(self.mesh_device.shape[1]),
                PAGE_SIZE,
                int(self.hf_config.head_dim),
            ):
                raise ValueError(f"unexpected vLLM KV shape for layer {layer_idx}: {shape}")
            tensor_idx = int(tensor_idx)
            if pool_blocks is None:
                pool_blocks = shape[0]
            elif shape[0] != pool_blocks:
                raise ValueError(f"vLLM KV specs disagree on the block count: {pool_blocks} vs {shape[0]}")
            if layer_idx in sliding:
                layer_cache.append(None)
                tensor_indices.append(None)
                cache_shapes.append(None)
                continue
            cache = unique_buffers.get(tensor_idx)
            if cache is None:
                shape = (shape[0] + ring_blocks,) + shape[1:] if sliding else shape
                cache = allocate(shape, f"cache_{shape}_tensor_{tensor_idx}")
                unique_buffers[tensor_idx] = cache
            layer_cache.append(cache)
            tensor_indices.append(tensor_idx)
            cache_shapes.append(tuple(int(dim) for dim in cache[0].shape))
        for layer_idx, host in ring_host.items():
            if layer_idx < len(layer_cache):
                layer_cache[layer_idx] = layer_cache[host]
                tensor_indices[layer_idx] = tensor_indices[host]
                cache_shapes[layer_idx] = cache_shapes[host]
        self._ring_block_base = int(pool_blocks or 0)

        for layer, cache in zip(self.model.layers, layer_cache):
            layer.decoder.self_attn.kv_cache = cache
            layer.decoder.self_attn.layer_past = cache
        self.model.kv_cache = layer_cache
        self.model.kv_cache_owner = "vllm"
        self.model._validate_precision_runtime()
        self.generator = Generator(self.model, self.model_args, kv_cache=layer_cache, cache_owner="vllm")
        self._cache_tensor_indices = tensor_indices
        self._cache_shapes = cache_shapes
        self._write_serving_capability()
        return [layer_cache]

    def allocate_kv_cache(self, shape, dtype, num_layers):
        """Legacy uniform-cache entry point retained for plugin compatibility."""

        return self.allocate_kv_cache_per_layer([(shape, dtype, index) for index in range(num_layers)])

    def _require_generator(self) -> Generator:
        if self.generator is None:
            raise RuntimeError("vLLM must allocate its KV cache before model execution")
        return self.generator

    def _activate_decode_bucket(self, width: int) -> None:
        """Select persistent model, sampling, and hybrid-page-table trace state."""

        generator = self._require_generator()
        inner = generator._inner
        previous = self._active_decode_bucket
        if previous is None:
            initial = getattr(self.model, "_persistent_per_layer_page_tables", None)
            if initial is not None:
                self._bucket_page_table_store.setdefault(self.max_batch_size, initial)
        if previous is not None:
            self._bucket_page_table_store[previous] = getattr(self.model, "_persistent_per_layer_page_tables", None)

        store = getattr(inner, "_bucket_trace_store", None)
        if store is None:
            store = inner._bucket_trace_store = {}
        if width not in store:
            store[width] = (
                defaultdict(lambda: None),
                defaultdict(lambda: None),
                defaultdict(lambda: None),
            )
        inner.trace_ids_decode, inner.trace_inputs_decode, inner.trace_output_decode = store[width]
        activate_decode_batch_size = getattr(self.model, "activate_decode_batch_size", None)
        if activate_decode_batch_size is not None:
            activate_decode_batch_size(width)
        for model in inner.model:
            sampling = getattr(model, "sampling", None)
            if sampling is not None and hasattr(sampling, "set_trace_bucket"):
                sampling.set_trace_bucket(width)

        persistent = self._bucket_page_table_store.get(width)
        if persistent is None:
            if hasattr(self.model, "_persistent_per_layer_page_tables"):
                del self.model._persistent_per_layer_page_tables
        else:
            self.model._persistent_per_layer_page_tables = persistent
        self._active_decode_bucket = width

    @staticmethod
    def _slice_page_tables(page_tables, width: int):
        if page_tables is None:
            return None
        sliced = {}
        result = []
        for table in page_tables:
            key = id(table)
            if key not in sliced:
                sliced[key] = table[:width] if isinstance(table, torch.Tensor) else table
            result.append(sliced[key])
        return result

    def _decode_bucket(self, *, start_pos, slot_remap, device_sampling: bool, enable_trace: bool) -> int:
        width = int(torch.as_tensor(start_pos).numel())
        if not device_sampling or not enable_trace or width <= 1 or slot_remap is not None:
            return width
        positions = torch.as_tensor(start_pos).reshape(-1)
        active_rows = (positions >= 0).nonzero().reshape(-1).tolist()
        return 1 if active_rows == [0] else width

    @property
    def already_warmed_up_prefill(self) -> bool:
        """Let the plugin reset the canonical two-phase warmup latch."""

        return False if self.generator is None else self.generator.already_warmed_up_prefill

    @already_warmed_up_prefill.setter
    def already_warmed_up_prefill(self, value: bool) -> None:
        if self.generator is not None:
            self.generator.already_warmed_up_prefill = bool(value)

    def _normalise_page_tables(self, page_tables_per_layer, page_table, *, ring_rows=None, ring_valid_blocks=None):
        tables = [page_table] * self.model.n_layers if page_tables_per_layer is None else list(page_tables_per_layer)
        if len(tables) < self.model.n_layers:
            raise ValueError(f"page_tables_per_layer has {len(tables)} entries for {self.model.n_layers} layers")
        tables = tables[: self.model.n_layers]
        if not self._sliding_layers:
            return tables
        reference = next(
            (table for idx, table in enumerate(tables) if idx not in self._sliding_layers and table is not None),
            page_table,
        )
        if not isinstance(reference, torch.Tensor):
            reference = torch.as_tensor(reference)
        rows, columns = int(reference.shape[0]), int(reference.shape[-1])
        if ring_rows is None:
            ring_rows = self._ring_of_slot[:rows]
        if len(ring_rows) != rows:
            raise ValueError(f"ring rows {len(ring_rows)} do not match {rows} page-table rows")
        ring = ring_page_table(ring_rows, columns, ring_valid_blocks, base=self._ring_block_base)
        for layer_idx in self._sliding_layers:
            tables[layer_idx] = ring
        return tables

    def _apply_ring_slot_remap(self, slot_remap) -> None:
        """Row i of the coming decode reads slot ``slot_remap[i]``; move the ring ownership with it."""

        if slot_remap is None:
            return
        remap = [int(value) for value in torch.as_tensor(slot_remap).reshape(-1).tolist()]
        width = min(len(remap), self.max_batch_size)
        moved = [self._ring_of_slot[old] for old in remap[:width]]
        if sorted(moved) != sorted(self._ring_of_slot[:width]):
            raise ValueError(f"slot_remap {remap[:width]} is not a permutation of the device slots")
        self._ring_of_slot[:width] = moved
        self._slot_prefill_end[:width] = [self._slot_prefill_end[old] for old in remap[:width]]
        self._ring_tables_dirty = True
        if _LOG_SLOTS:
            logger.info("slots: remap {} -> ring_of_slot {}", remap[:width], self._ring_of_slot[:width])

    def release_request(self, slot) -> None:
        """The plugin's request-finished hook: the slot's rings no longer hold a live prompt."""

        slot = int(slot)
        if _LOG_SLOTS:
            logger.info(
                "slots: release slot {} (marker was {})",
                slot,
                self._slot_prefill_end[slot] if 0 <= slot < self.max_batch_size else None,
            )
        if 0 <= slot < self.max_batch_size:
            self._slot_prefill_end[slot] = None

    def _resume_plan(
        self, cached: int, slot: int, chunk_end: int | None = None, new_request: bool | None = None
    ) -> tuple[int, int, bool]:
        """Where a prompt with ``cached`` computed tokens resumes on ``slot``.

        Returns (attention start, first position written to the full-attention
        cache, ring cold). The attention restarts on a 512 boundary, the largest
        attention chunk and a whole number of ring blocks. A warm ring (the slot's
        last prefill ended exactly at ``cached``: a chunked-prefill continuation)
        holds the window before that boundary. A cold ring (a cached prefix
        computed by another request or slot) starts COLD_RING_REPREFILL_TOKENS
        further back, so the error of the missing window has washed out of every
        sliding layer, and the ring holds an exact window, by the first new token.
        ``new_request`` (from the plugin) settles warm or cold outright; without it
        the slot's prefill-end marker decides.
        """

        cached = int(cached)
        if cached <= 0:
            return 0, 0, False
        fill_start = cached - cached % PAGE_SIZE
        aligned = cached - cached % PREFILL_CHUNK_ALIGN
        cold = bool(self._sliding_layers) and (
            bool(new_request) if new_request is not None else self._slot_prefill_end[slot] != cached
        )
        start = max(0, aligned - COLD_RING_REPREFILL_TOKENS) if cold else aligned
        if chunk_end is not None:
            start = self._fit_padded_chunk(start, int(chunk_end))
        return start, fill_start, cold and start > 0

    def _fit_padded_chunk(self, start: int, chunk_end: int) -> int:
        """Move ``start`` earlier until the chunk's padded length ends inside the context."""

        limit = int(self.max_model_len)
        while start > 0 and start + get_padded_prefill_len(chunk_end - start) > limit:
            padded = get_padded_prefill_len(chunk_end - start)
            candidate = limit - padded
            candidate -= candidate % PREFILL_CHUNK_ALIGN
            start = max(0, min(start - PREFILL_CHUNK_ALIGN, candidate))
        return start

    @staticmethod
    def _page_table_equal(left, right) -> bool:
        if left is right:
            return True
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return left.shape == right.shape and torch.equal(left, right)
        return False

    @contextmanager
    def _route_page_tables(
        self,
        page_tables_per_layer,
        page_table,
        *,
        update_persistent=True,
        page_table_state_id=None,
        ring_rows=None,
        ring_valid_blocks=None,
    ):
        tables = self._normalise_page_tables(
            page_tables_per_layer,
            page_table,
            ring_rows=ring_rows,
            ring_valid_blocks=ring_valid_blocks,
        )
        if update_persistent:
            changed = (
                _TABLES_ALWAYS
                or self._last_page_tables is None
                or self._ring_tables_dirty
                or (page_table_state_id is not None and page_table_state_id != self._last_page_table_state_id)
            )
            if not changed and page_table_state_id is None:
                comparisons = {}
                for old, new in zip(self._last_page_tables or [], tables):
                    pair = (id(old), id(new))
                    if pair not in comparisons:
                        comparisons[pair] = self._page_table_equal(old, new)
                    if not comparisons[pair]:
                        changed = True
                        break
            if changed:
                if _LOG_SLOTS:
                    persistent = getattr(self.model, "_persistent_per_layer_page_tables", None)
                    logger.info(
                        "tables: bucket {} host rows {} cols {} device rows {} cols {}",
                        self._active_decode_bucket,
                        [int(t.shape[0]) for t in tables[:2] if isinstance(t, torch.Tensor)],
                        [int(t.shape[-1]) for t in tables[:2] if isinstance(t, torch.Tensor)],
                        None
                        if not persistent
                        else [tuple(int(v) for v in p.shape) for p in persistent[:2] if p is not None],
                        None,
                    )
                self.model.update_persistent_per_layer_page_tables(tables)
                if _TABLES_SYNC:
                    ttnn.synchronize_device(self.mesh_device)
                if _TABLES_VERIFY:
                    self._verify_persistent_tables(tables)
                clones = {}
                self._last_page_tables = []
                for table in tables:
                    key = id(table)
                    if key not in clones:
                        clones[key] = table.clone() if isinstance(table, torch.Tensor) else table
                    self._last_page_tables.append(clones[key])
                self._last_page_table_state_id = page_table_state_id
                self._ring_tables_dirty = False
                self.serving_counters["page_table_refreshes"] += 1
            else:
                self.serving_counters["page_table_reuses"] += 1
        self.model._active_page_tables_per_layer = tables
        try:
            yield
        finally:
            del self.model._active_page_tables_per_layer

    def _verify_persistent_tables(self, tables) -> None:
        persistent = getattr(self.model, "_persistent_per_layer_page_tables", None)
        if not persistent:
            logger.warning("tables: no persistent page tables to verify")
            return
        for layer_idx, (host, device) in enumerate(zip(tables, persistent)):
            if device is None or not isinstance(host, torch.Tensor):
                continue
            shards = ttnn.get_device_tensors(device)
            readback = ttnn.to_torch(shards[0]).to(torch.int32)
            rows = min(int(host.shape[0]), int(readback.shape[0]))
            columns = min(int(host.shape[1]), int(readback.shape[1]))
            if not torch.equal(readback[:rows, :columns], host[:rows, :columns].to(torch.int32)):
                diff = (readback[:rows, :columns] != host[:rows, :columns]).nonzero()
                logger.error(
                    "tables: layer {} device table differs from host at {} entries, first {} (host {} device {})",
                    layer_idx,
                    int(diff.shape[0]),
                    diff[0].tolist(),
                    int(host[diff[0][0], diff[0][1]]),
                    int(readback[diff[0][0], diff[0][1]]),
                )
                return
        if _LOG_SLOTS:
            logger.info("tables: {} layers verified", len(persistent))

    @staticmethod
    def _sampling_state_reusable(sampling_params) -> bool:
        if sampling_params is None:
            return False
        if not _all_none(getattr(sampling_params, "seed", None)):
            return False
        for name in ("presence_penalty", "frequency_penalty"):
            if not _all_equal(getattr(sampling_params, name, None), 0.0):
                return False
        if not _all_equal(getattr(sampling_params, "repetition_penalty", None), 1.0):
            return False
        return _all_false(getattr(sampling_params, "enable_log_probs", None))

    def prefill_forward(
        self,
        *,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        page_tables_per_layer=None,
        sampling_params=None,
        empty_slots=None,
        enable_trace=False,
        start_pos=None,
        new_request_rows=None,
        **kwargs,
    ):
        del enable_trace, kwargs
        generator = self._require_generator()
        prefill_release_epoch = generator._lifetime_decode_trace_releases_for_prefill_compile
        slots = list(range(len(prompt_lens))) if empty_slots is None else [int(slot) for slot in empty_slots]
        if _LOG_SLOTS:
            logger.info(
                "slots: prefill slots {} rings {} prompt_lens {} start_pos {}",
                slots,
                [self._ring_of_slot[slot] for slot in slots],
                [int(length) for length in prompt_lens],
                None if start_pos is None else torch.as_tensor(start_pos).reshape(-1).tolist(),
            )
        starts = fills = colds = None
        if start_pos is not None:
            raw = [int(value) for value in torch.as_tensor(start_pos).reshape(-1).tolist()]
            if len(raw) != len(prompt_lens):
                raise ValueError(f"start_pos has {len(raw)} entries for {len(prompt_lens)} prompts")
            fresh = [None] * len(prompt_lens) if new_request_rows is None else [bool(v) for v in new_request_rows]
            if len(fresh) != len(prompt_lens):
                raise ValueError(f"new_request_rows has {len(fresh)} entries for {len(prompt_lens)} prompts")
            plan = [
                self._resume_plan(cached, slot, int(length), new)
                for cached, slot, length, new in zip(raw, slots, prompt_lens, fresh)
            ]
            if _LOG_SLOTS:
                logger.info(
                    "slots: resume plan {} (markers {})",
                    plan,
                    [self._slot_prefill_end[slot] for slot in slots],
                )
            if any(start or fill for start, fill, _ in plan):
                starts = [start for start, _, _ in plan]
                fills = [fill for _, fill, _ in plan]
                colds = [cold for _, _, cold in plan]
        lifecycle_changed = self._transition_sampling_lifecycle(
            generator,
            device_sampling=sampling_params is not None,
            kv_cache=kv_cache,
        )
        ring_rows = [self._ring_of_slot[slot] for slot in slots]
        ring_valid_blocks = [math.ceil(int(length) / PAGE_SIZE) for length in prompt_lens]
        tables = self._normalise_page_tables(
            page_tables_per_layer,
            page_table,
            ring_rows=ring_rows,
            ring_valid_blocks=ring_valid_blocks,
        )
        with self._route_page_tables(
            tables,
            page_table,
            update_persistent=False,
            ring_rows=ring_rows,
            ring_valid_blocks=ring_valid_blocks,
        ):
            result = generator.prefill_forward(
                torch.as_tensor(tokens),
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=[int(length) for length in prompt_lens],
                sampling_params=sampling_params,
                empty_slots=empty_slots,
                page_tables_per_layer=tables,
                start_pos=starts,
                fill_start_pos=fills,
                ring_cold=colds,
                # vLLM prefills have their active batch row count while decode
                # is padded to max_batch_size.  Capturing/hoisting decode from
                # the prefill page table would bind a wrong-sized decode trace.
                # Keep prefill untraced and let the first real padded decode
                # capture the canonical model/sampling traces.
                enable_trace=False,
            )
        if (
            sampling_params is not None
            and generator._lifetime_decode_trace_releases_for_prefill_compile != prefill_release_epoch
        ):
            # A first-time prefill program variant releases every bucket before
            # it compiles. Rebuild both prepared model/sampling traces now,
            # while no trace is live, instead of lazily capturing one bucket
            # and later trying to allocate the other beside it.
            self._restore_device_decode_traces(kv_cache)
            lifecycle_changed = True
        for slot, length in zip(slots, prompt_lens):
            if 0 <= slot < self.max_batch_size:
                self._slot_prefill_end[slot] = int(length)
        self.serving_counters["prefill_calls"] += 1
        if _LOG_PROGRAM_CACHE:
            # Opt-in evidence that the untraced prefill adds no programs once its
            # shapes are warm (every cached program pins a DRAM kernel buffer).
            logger.info(
                "prefill call {}: prompt_lens={} program_cache_entries={}",
                self.serving_counters["prefill_calls"],
                [int(length) for length in prompt_lens],
                self.mesh_device.num_program_cache_entries(),
            )
        self._last_sampling_key = None
        # EngineCore multiprocessing may bypass Python atexit. Persist the
        # rare trace-lifecycle transitions here so capability evidence cannot
        # remain an initial zero-counter snapshot. This is never on steady
        # decode and is not part of the measured token-out path.
        if lifecycle_changed:
            self._write_serving_capability()
        return result

    def decode_forward(
        self,
        *,
        tokens,
        page_table,
        kv_cache,
        start_pos,
        page_tables_per_layer=None,
        sampling_params=None,
        prompt_tokens=None,
        output_tokens=None,
        slot_remap=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        enable_trace=True,
        read_from_device=True,
        **kwargs,
    ):
        if "reset_batch" in kwargs:
            raise TypeError("decode_input_update_contract=1 requires explicit reload flags")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("Resetting sampling state requires current tokens and positions")
        reset_batch = bool(reload_inputs)
        generator = self._require_generator()
        device_sampling = sampling_params is not None
        # Re-establish the trace invariant at the decode boundary itself. A
        # prefill can be routed differently from the requests already waiting
        # to decode (for example during admission), so decode must not assume
        # prefill already released or restored the resident trace buckets.
        self._transition_sampling_lifecycle(
            generator,
            device_sampling=device_sampling,
            kv_cache=kv_cache,
        )
        recapture_reset = device_sampling and self._device_trace_recapture_requires_reset
        if recapture_reset:
            # Model capture is KV-inactive, but sampling capture necessarily
            # writes a meaningless token into its persistent feedback buffer.
            # The first real device decode must restage every authoritative
            # scheduler input before either model or sampler trace can replay.
            reset_batch = True
            reset_sampling_state = True
        # Host sampling is an explicit compatibility path for shared tests.
        # Its eager full-logits decode runs only after prefill released every
        # resident trace; normal serving remains the device token-out trace.
        if not device_sampling:
            enable_trace = False
        requested_bucket = self._decode_bucket(
            start_pos=start_pos,
            slot_remap=slot_remap,
            device_sampling=device_sampling,
            enable_trace=enable_trace,
        )
        host_width = int(torch.as_tensor(start_pos).numel())
        bucket_changed = False
        force_host_tokens = bool(reload_inputs) or recapture_reset
        if device_sampling and enable_trace:
            if self._active_decode_bucket is None:
                bucket = requested_bucket
                self._activate_decode_bucket(bucket)
            elif requested_bucket != self._active_decode_bucket and reset_batch:
                # The plugin drains pending async output before asserting a
                # reset boundary.  Only then is it safe to switch to another
                # trace namespace and restage the caller-authoritative token,
                # position, and page table.  Between resets the active trace
                # stays sticky even if admission or condensation changes the
                # apparent active width.
                bucket = requested_bucket
                bucket_changed = True
                force_host_tokens = True
                self._activate_decode_bucket(bucket)
            else:
                bucket = self._active_decode_bucket
        else:
            # Host sampling is eager and always sees serving-width inputs. Its
            # lifecycle transition released traces before any eager allocation.
            bucket = int(torch.as_tensor(start_pos).numel())
            if bucket != self._active_decode_bucket:
                # The preceding device request may have left the model's
                # persistent page-table/norm configuration at B1. Host decode
                # receives the full serving-width table, so activate that
                # matching eager configuration before copying the table.
                self._activate_decode_bucket(bucket)
        if bucket < int(torch.as_tensor(start_pos).numel()):
            tokens = torch.as_tensor(tokens)[:bucket]
            start_pos = torch.as_tensor(start_pos)[:bucket]
            page_table = page_table[:bucket] if isinstance(page_table, torch.Tensor) else page_table
        # The hybrid runner deliberately keeps per-layer tables padded to the
        # warmup width even when its primary table, tokens, and positions use a
        # declared B1 bucket.  Route only the selected bucket's rows into the
        # matching persistent buffers; B32 inputs are unchanged by the slice.
        page_tables_per_layer = self._slice_page_tables(page_tables_per_layer, bucket)
        self._apply_ring_slot_remap(slot_remap)
        if _LOG_SLOTS:
            logger.info(
                "slots: decode bucket {} reset {} host_width {} positions {} rings {}",
                bucket,
                reset_batch,
                host_width,
                torch.as_tensor(start_pos).reshape(-1)[:bucket].tolist(),
                self._ring_of_slot[:bucket],
            )
        can_reuse_sampling = (
            device_sampling
            and enable_trace
            and not reset_batch
            and not reload_sampling_params
            and not reset_sampling_state
            and not bucket_changed
            and self._sampling_state_reusable(sampling_params)
            and self._last_sampling_key is not None
        )
        sampling_params_for_device = _sampling_params_to_host_values(sampling_params)
        key = _sampling_key(sampling_params_for_device)
        reuse_sampling = can_reuse_sampling and self._last_sampling_key == key
        if reuse_sampling:
            sampling_params_for_device = None
        reload_page_table = reload_page_table or self._ring_tables_dirty
        with self._route_page_tables(
            page_tables_per_layer,
            page_table,
            update_persistent=reload_inputs or reload_page_table or bucket_changed,
        ):
            result = generator.decode_forward(
                torch.as_tensor(tokens),
                torch.as_tensor(start_pos),
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                sampling_mode="device" if device_sampling else "host",
                sampling_params=sampling_params_for_device,
                reload_inputs=force_host_tokens or not enable_trace,
                reload_page_table=reload_page_table,
                reload_sampling_params=reload_sampling_params or bucket_changed or recapture_reset,
                reset_sampling_state=reset_sampling_state or bucket_changed or recapture_reset,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                slot_remap=slot_remap,
                read_from_device=read_from_device,
                reuse_sampling_state=reuse_sampling,
                **kwargs,
            )
        if recapture_reset:
            # Clear only after the authoritative reset was submitted
            # successfully; an exception must leave the guard armed.
            self._device_trace_recapture_requires_reset = False
        self.serving_counters["decode_calls"] += 1
        counter = "device_sampled_decodes" if device_sampling else "host_sampled_decodes"
        self.serving_counters[counter] += 1
        self.serving_counters["sampling_state_reuses" if reuse_sampling else "sampling_state_pushes"] += int(
            device_sampling
        )
        self._last_sampling_key = key if device_sampling else None
        self._last_decode_was_device_sampled = device_sampling
        return result

    def read_decode_output(self, device_output, async_read=False):
        if async_read:
            self.serving_counters["async_reads"] += 1
        return self._require_generator().read_decode_output(
            device_output,
            async_read=async_read,
            is_tokens=self._last_decode_was_device_sampled,
        )

    def process_decode_output_host(self, host_output, is_tokens=False):
        return self._require_generator().process_decode_output_host(host_output, is_tokens=is_tokens)

    def release_persistent_capture(self):
        """Persist final serving evidence and release traces before mesh close."""

        generator = self.generator
        if generator is None:
            return
        try:
            self._write_serving_capability()
        finally:
            generator.teardown()
            self.generator = None

    def warmup_model_prefill(self, *, kv_cache, enable_trace, can_sample_on_device, **kwargs):
        del enable_trace, kwargs
        table = torch.zeros(self.max_batch_size, self.page_table_blocks, dtype=torch.int32)
        with self._route_page_tables(None, table, update_persistent=False):
            generator = self._require_generator()
            result = generator.warmup_model_prefill(
                kv_cache=kv_cache,
                # Serving deliberately traces decode only; see prefill_forward.
                enable_trace=False,
                can_sample_on_device=can_sample_on_device,
            )
        if can_sample_on_device:
            # The generic warmup routes persistent max-width page tables through
            # the model. Real vLLM sequential prefill instead supplies explicit
            # per-request row slices. Compile that exact paged-fill signature
            # before any decode trace is captured so its program-cache buffers
            # cannot be allocated beside a live B1/B32 trace.
            seq_len = 128
            num_blocks = math.ceil(seq_len / PAGE_SIZE)
            request_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
            generator.prefill_forward(
                torch.zeros(1, seq_len, dtype=torch.int64),
                page_table=request_table,
                kv_cache=kv_cache,
                prompt_lens=[seq_len],
                sampling_params=GREEDY,
                empty_slots=[0],
                page_tables_per_layer=[request_table] * self.model.n_layers,
                enable_trace=False,
            )
            if not self._prefill_shapes_warm:
                self._warmup_batched_prefill_shapes(generator, kv_cache)
                self._warmup_chunk_continuations(generator, kv_cache)
                self._prefill_shapes_warm = True
        # The indexed expert prefill picks matmul shapes from the routing of
        # each prompt; compile the whole (height x group size) set once now so
        # the first long prompts do not pay tens of seconds of JIT.
        layers = getattr(self.model, "layers", None) or ()
        warmup = getattr(getattr(layers[0], "decoder", None), "mlp", None) if len(layers) else None
        warmup = getattr(warmup, "warmup_indexed_prefill_shapes", None)
        if warmup is not None:
            compiled = warmup()
            if compiled:
                logger.info(f"indexed prefill: compiled {compiled} expert-group shapes")
        return result

    def _warmup_batched_prefill_shapes(self, generator, kv_cache) -> None:
        """Compile the batched-prefill programs for the common (batch, length) pairs.

        A batched prefill has its own program shapes per (device batch, padded
        length); the first request of each pair otherwise pays 5 to 20 s of
        compilation in serving.  Lengths come from
        ``GPT_OSS_120B_BATCHED_PREFILL_WARMUP`` (default ``128,1024,2048``; empty
        disables), batches are every supported size up to the serving width.
        Each row gets its own scratch blocks so the fills do not race.
        """
        model_args = getattr(self.model, "args", None) or getattr(generator, "model_args", None)
        if getattr(model_args, "disable_batched_prefill", True):
            return
        if getattr(self, "_batched_prefill_shapes_warm", False):
            # The plugin runs the prefill warmup twice per boot; the programs
            # are cached after the first pass (30 s cold, 11 s to re-run).
            return
        self._batched_prefill_shapes_warm = True
        spec = os.environ.get("GPT_OSS_120B_BATCHED_PREFILL_WARMUP", "128,1024,2048").strip()
        if not spec:
            return
        from models.tt_transformers.tt.generator import MAX_BATCHED_PREFILL_SEQ_LEN, SUPPORTED_PREFILL_BATCH_SIZES

        cap = getattr(model_args, "batched_prefill_max_tokens_per_user", None)
        lengths = [int(v) for v in spec.split(",") if v.strip()]
        started = time.perf_counter()
        compiled = 0
        for seq_len in lengths:
            if cap is not None and seq_len > int(cap):
                continue
            num_blocks = math.ceil(seq_len / PAGE_SIZE)
            for batch in SUPPORTED_PREFILL_BATCH_SIZES:
                if batch < 2 or batch > self.max_batch_size or batch * seq_len >= MAX_BATCHED_PREFILL_SEQ_LEN:
                    continue
                if batch * num_blocks > self.page_table_blocks * self.max_batch_size:
                    continue
                table = torch.arange(batch * num_blocks, dtype=torch.int32).reshape(batch, num_blocks)
                generator.prefill_forward(
                    torch.zeros(batch, seq_len, dtype=torch.int64),
                    page_table=table,
                    kv_cache=kv_cache,
                    prompt_lens=[seq_len] * batch,
                    sampling_params=GREEDY,
                    empty_slots=list(range(batch)),
                    page_tables_per_layer=[table] * self.model.n_layers,
                    enable_trace=False,
                )
                compiled += 1
        if compiled:
            logger.info(
                f"batched prefill: compiled {compiled} (batch, length) shapes in {time.perf_counter() - started:.1f} s"
            )

    def _warmup_chunk_continuations(self, generator, kv_cache) -> None:
        """Compile the chunk-continuation programs (one budget-sized chunk after each prefix)."""

        spec = os.environ.get(ENV_CHUNK_WARMUP, CHUNK_WARMUP_DEFAULT).strip()
        if not spec:
            return
        chunk_text, _, limit_text = spec.partition(":")
        chunk = int(chunk_text)
        limit = min(int(limit_text), self.max_model_len) if limit_text else self.max_model_len
        if chunk <= 0 or chunk % PREFILL_CHUNK_ALIGN:
            raise ValueError(f"{ENV_CHUNK_WARMUP} chunk {chunk} is not a positive multiple of {PREFILL_CHUNK_ALIGN}")
        started = time.perf_counter()
        compiled = 0
        for start in range(chunk, limit, chunk):
            total = start + chunk
            if total > limit:
                break
            num_blocks = math.ceil(total / PAGE_SIZE)
            table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
            tables = self._normalise_page_tables(
                None,
                table,
                ring_rows=[self._ring_of_slot[0]] if self._sliding_layers else None,
                ring_valid_blocks=[num_blocks],
            )
            generator.prefill_forward(
                torch.zeros(1, total, dtype=torch.int64),
                page_table=table,
                kv_cache=kv_cache,
                prompt_lens=[total],
                sampling_params=GREEDY,
                empty_slots=[0],
                page_tables_per_layer=tables,
                enable_trace=False,
                start_pos=[start],
            )
            compiled += 1
        if compiled:
            logger.info(
                f"chunked prefill: compiled {compiled} continuation shapes of {chunk} tokens "
                f"in {time.perf_counter() - started:.1f} s"
            )

    def warmup_model_decode(
        self,
        *,
        kv_cache,
        enable_trace,
        max_batch_size,
        num_blocks,
        can_sample_on_device,
        **kwargs,
    ):
        del kwargs
        widths = list(decode_trace_buckets(int(max_batch_size)))
        compile_key = (tuple(widths), int(num_blocks), bool(can_sample_on_device))
        result = None
        if self._decode_bucket_compile_key != compile_key:
            for width in widths:
                self._activate_decode_bucket(width)
                table = torch.zeros(width, num_blocks, dtype=torch.int32)
                with self._route_page_tables(None, table):
                    result = self._require_generator().warmup_model_decode(
                        kv_cache=kv_cache,
                        enable_trace=False,
                        max_batch_size=width,
                        num_blocks=num_blocks,
                        can_sample_on_device=can_sample_on_device,
                    )
                    if can_sample_on_device:
                        self._prepared_decode_buckets[width] = self._require_generator().prepare_model_decode_trace(
                            kv_cache=kv_cache,
                            max_batch_size=width,
                            num_blocks=num_blocks,
                        )
            self._decode_bucket_compile_key = compile_key

        if enable_trace:
            ttnn.synchronize_device(self.mesh_device)
            gc.collect()
            for width in widths:
                self._activate_decode_bucket(width)
                table = torch.zeros(width, num_blocks, dtype=torch.int32)
                with self._route_page_tables(None, table):
                    prepared = self._prepared_decode_buckets.get(width)
                    if prepared is not None:
                        self._require_generator().capture_prepared_model_decode_trace(prepared)
                    result = self._require_generator().warmup_model_decode(
                        kv_cache=kv_cache,
                        enable_trace=True,
                        max_batch_size=width,
                        num_blocks=num_blocks,
                        can_sample_on_device=can_sample_on_device,
                        skip_trace_precompile=True,
                    )
            if can_sample_on_device:
                self._device_trace_recapture_requires_reset = True
        self._activate_decode_bucket(int(max_batch_size))
        self._last_sampling_key = None
        return result

    def _restore_device_decode_traces(self, kv_cache) -> None:
        """Recapture prepared B1/B32 signatures before device-sampled execution."""

        if not self._prepared_decode_buckets:
            raise RuntimeError("device trace signatures were not prepared during warmup")
        ttnn.synchronize_device(self.mesh_device)
        gc.collect()
        for width, prepared in sorted(self._prepared_decode_buckets.items()):
            self._activate_decode_bucket(width)
            table = torch.zeros(width, self.page_table_blocks, dtype=torch.int32)
            with self._route_page_tables(None, table):
                self._require_generator().capture_prepared_model_decode_trace(prepared)
        self._activate_decode_bucket(self.max_batch_size)
        self._last_sampling_key = None
        self._device_trace_recapture_requires_reset = True
        self.serving_counters["device_trace_recaptures"] += 1

    def _transition_sampling_lifecycle(self, generator, *, device_sampling: bool, kv_cache) -> bool:
        """Make resident decode traces match the sampling route at this boundary."""

        if device_sampling:
            if not self._host_sampling_active:
                return False
            self._restore_device_decode_traces(kv_cache)
            self._host_sampling_active = False
            return True

        # The generator's release is idempotent and checks actual live trace
        # state. Do this even when the adapter flag says host mode: a preceding
        # device-prefill or first-time program compile may have changed trace
        # lifetime independently of the last decode route.
        released = bool(generator.release_decode_traces_for_host_sampling())
        self.serving_counters["host_trace_releases"] += int(released)
        self._host_sampling_active = True
        return released

    def serving_capability(self) -> dict[str, Any]:
        generator = self.generator
        return {
            "model": "openai/gpt-oss-120b",
            "architecture": type(self).__name__,
            "mesh_shape": [int(value) for value in self.mesh_device.shape],
            "resident_layers": self.model.n_layers,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_batch_size,
            "page_size": PAGE_SIZE,
            "hybrid_kv_cache_groups": not SLIDING_RING_ENABLED,
            "kv_pool_blocks": kv_pool_blocks(self.max_batch_size) if SLIDING_RING_ENABLED else None,
            "sliding_ring_tokens": SLIDING_RING_TOKENS if SLIDING_RING_ENABLED else None,
            "prefix_caching": PREFIX_CACHING_ENABLED,
            "cold_ring_reprefill_tokens": COLD_RING_REPREFILL_TOKENS if PREFIX_CACHING_ENABLED else None,
            "sliding_ring_layers": list(self._sliding_layers),
            "sliding_ring_block_base": self._ring_block_base,
            "prefill_trace_enabled": False,
            "decode_trace_enabled": True,
            "decode_trace_buckets": list(decode_trace_buckets(self.max_batch_size)),
            "kv_cache_owner": self.model.kv_cache_owner,
            "kv_cache_dtype": dtype_name(self.model.precision_config.decoder_policy_for_layer(0).kv_cache_dtype),
            "kv_cache_unique_tensors": len(set(self._cache_tensor_indices)),
            "kv_cache_layer_tensor_indices": self._cache_tensor_indices,
            "kv_cache_layer_shapes": self._cache_shapes,
            "precision": self.model.precision_runtime_evidence(),
            "model_capabilities": {
                **self.model_capabilities,
                "fabric_config": {"config": str(self.model_capabilities["fabric_config"]["config"])},
            },
            "serving_counters": dict(self.serving_counters),
            "generator": None if generator is None else generator.capability_report(),
        }

    def _write_serving_capability(self):
        if self.generator is None or self.model.n_layers != MODEL_LAYERS:
            return
        root = Path(os.environ.get("GPT_OSS_120B_RESULTS", "generated/test_reports/gpt_oss_120b_qb2"))
        path = root / "vllm_serving_capability.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.serving_capability(), indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            logger.warning("Could not write GPT-OSS serving evidence to {}: {}", path, error)


__all__ = ["TTGptOssForCausalLM"]
