# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, fields, replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import regex as re
import torch
import ttnn

from vllm.config import VllmConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm_tt_plugin.async_decode import (
    AsyncTTModelRunnerOutput,
    CompletedDecodeStep,
    DeferredDecodeOutput,
    TTAsyncDecodeController,
)
from vllm_tt_plugin.config import (
    get_tt_data_parallel_size,
    get_tt_max_batch_size,
    get_tt_per_lane_max_num_seqs,
)
from vllm_tt_plugin.input_batch import (
    LOGPROBS_NONE_SENTINEL,
    SEED_NONE_SENTINEL,
    CachedRequestState,
    InputBatch,
    TTLaneInputBatch,
    apply_cached_req_state_update,
    build_cached_request_state,
    clone_torch_generator,
)
from vllm_tt_plugin.lane_scheduler import get_tt_step_plan
from vllm_tt_plugin.loader import TTModelLoader
from vllm_tt_plugin.logger import init_tt_logger
from vllm_tt_plugin.logprobs import build_device_logprobs
from vllm_tt_plugin.model_input import (
    TTModelInput,
    TTSamplingParams,
    slice_tt_sampling_params,
)
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.structured_output import (
    has_structured_outputs,
    reorder_grammar_bitmask_for_tt_batch,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

import numpy as np

logger = init_tt_logger(__name__)

# Matches the upstream attention-layer naming convention used by registered
# vLLM models (e.g. "model.language_model.layers.5.self_attn") as well as
# bare "layers.5" forms used in TT spec hooks. The first capture group is
# the integer layer index.
_LAYER_NAME_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _parse_layer_index(layer_name: str) -> int:
    """Extract the integer layer index from a layer name.

    Used to map ``KVCacheGroupSpec.layer_names`` back to model-side layer
    indices when distributing per-group KV cache shapes across the
    layer-indexed allocator. The hook author is expected to follow the
    ``...layers.<idx>...`` convention.
    """
    match = _LAYER_NAME_RE.search(layer_name)
    if match is None:
        raise ValueError(
            f"Could not parse a layer index from layer name '{layer_name}'. "
            "TT spec hooks must use the '...layers.<idx>...' naming convention."
        )
    return int(match.group(1))


@dataclass(frozen=True)
class _SyncForward:
    """Materialized non-DP forward result awaiting host/device sampling.

    Carries everything ``_get_output_tokens`` needs so the sampling tail can
    run in a later ``sample_tokens`` call rather than inside ``execute_model``.
    """

    tt_out: Any
    tt_log_probs: Any
    sampling_params: TTSamplingParams
    model_input: TTModelInput
    batch_size_per_dp: list[int]
    perform_device_sampling: bool
    is_decode: bool


class TTModelRunner:
    def __init__(
        self,
        vllm_config: VllmConfig,
        mesh_device: ttnn.MeshDevice,
        trace_mode: str,
        enable_model_warmup: bool,
        num_devices: int,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        if self.model_config.is_encoder_decoder:
            raise ValueError("Encoder-decoder models aren't yet supported for TT")

        # Detect if the model has "mrope" rope_scaling type.
        # mrope requires keeping "rope_deltas" between prefill/decode phases.
        self.request_specific_rope = bool(self.model_config.uses_mrope)
        if self.request_specific_rope:
            self.previous_req_ids: set[str] = set()

        self.mesh_device = mesh_device
        self.trace_mode = trace_mode
        self.enable_model_warmup = enable_model_warmup
        # Runtime-discovered physical device count, supplied by the worker.
        self.num_devices = num_devices
        # Whether to sample on device
        self.sample_on_device_mode = getattr(TTPlatform, "sample_on_device_mode", None)
        assert self.sample_on_device_mode in (None, "all", "decode_only")
        # Whether the model supports top-K logprobs on device.
        # Detected from model_type (available to all DP ranks without
        # requiring the model to be loaded). Models like gpt-oss-120b
        # set use_topk_logprobs=True and return top-32 logprobs from device.
        # TODO: Update this check as more models add top-K logprobs support.
        # https://github.com/tenstorrent/tt-metal/issues/40810
        self.supports_topk_logprobs = (
            self.model_config.hf_config.model_type == "gpt_oss"
        )

        logger.info(
            "TTModelRunner: trace_mode=%s, "
            "sample_on_device_mode=%s, enable_model_warmup=%s",
            self.trace_mode,
            self.sample_on_device_mode,
            self.enable_model_warmup,
        )

        # mm_hash -> encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

        # Cached request states. Request states are tracked in the runner so
        # they don't need to be re-sent every scheduling step. For requests
        # that have been scheduled before, only the diff is received from
        # the scheduler output.
        self.requests: dict[str, CachedRequestState] = {}

        # Cache the arange needed for unpacking structured output bitmask
        self.structured_output_arange = torch.arange(0, 32)
        self.vocab_size = self.model_config.get_vocab_size()
        self.bitmask_size = cdiv(self.vocab_size, 32)

        # For on-device decode sampling, we must signal if the padded decode
        # batch layout changed since the *previous decode step*. Layout can
        # change during prefill steps (e.g. new requests added), so we keep a
        # sticky flag and clear it only after a decode input consumes it.
        self._decode_layout_changed_since_last_decode: bool = True

        # Forward/sampling split: ``execute_model`` runs the device forward,
        # enqueues a deferred sampler here, and returns ``None``; the engine
        # then calls ``sample_tokens`` which pops and runs it. FIFO matches the
        # batch-queue's oldest-first finalization order, so a single queue stays
        # correct even with one forward in flight.
        self._pending_samples: deque[Any] = deque()

        # Non-DP async scheduling: overlap CPU scheduling with device execution.
        # Only supported for DP=1 (DP>1 uses a different execution path).
        self.non_dp_async_scheduling = (
            self.scheduler_config.async_scheduling
            and self.parallel_config.data_parallel_size == 1
        )
        self._steady_decode_lock = threading.Lock()
        self._pending_async_steps: deque[DeferredDecodeOutput] = deque()
        self._pending_async_overlap_ok: deque[bool] = deque()
        self._completed_decode_steps: deque[CompletedDecodeStep] = deque()
        self.async_decode = TTAsyncDecodeController(self)
        self.tt_data_parallel_size = get_tt_data_parallel_size(vllm_config)
        self.tt_max_batch_size = get_tt_max_batch_size(vllm_config)
        self.tt_per_lane_max_num_seqs = get_tt_per_lane_max_num_seqs(vllm_config)

        # req_id -> device slot holding its per-slot state (GDN recurrent/conv, seed
        # RNG, decode trace buffers). Needed because evict/re-add and condense move a
        # request's ROW, not its state.
        self._req_state_slot: dict[str, int] = {}

        # Sampler for sampling on host when device sampling is not supported.
        # Only used by device ranks (local dp rank 0).
        if self.parallel_config.data_parallel_rank_local == 0:
            self.host_sampler = Sampler()

        # Host-side logits processors (min_p, logit_bias, min_tokens, plus any
        # custom logits processors). Used by the host sampler when device
        # sampling isn't supported for a given batch.
        self._host_logitsprocs: LogitsProcessors = build_logitsprocs(
            vllm_config=vllm_config,
            device=torch.device("cpu"),
            is_pin_memory=False,
            is_pooling_model=False,
            custom_logitsprocs=(self.model_config.logits_processors or ()),
        )

    @property
    def lane_batch(self) -> TTLaneInputBatch:
        """The persistent batch as its lane-DP type.

        Valid only in lane mode, where ``initialize_kv_cache`` builds a
        ``TTLaneInputBatch``. The lane step orchestration (``_execute_lane_step``)
        reads this instead of re-casting ``input_batch`` at every use site.
        """
        return cast(TTLaneInputBatch, self.input_batch)

    @property
    def _is_lane_mode(self) -> bool:
        """Single-process multi-lane (lane-DP) mode: one engine drives
        ``tt_data_parallel_size`` in-process DP replicas. In this mode the
        persistent batch is a ``TTLaneInputBatch`` that owns the lane layout
        (stable per-lane device slots, merged sampling) and the runner only
        orchestrates; gathered multi-process DP and non-DP keep the plain
        ``InputBatch`` and the per-rank gather paths.

        Derived (rather than cached in ``__init__``) so it depends only on
        already-set runner state -- this mirrors ``uses_tt_lane_coordinator``:
        vLLM sees a single engine (``data_parallel_size == 1``) while the TT
        backend runs more than one lane.
        """
        return (
            self.parallel_config.data_parallel_size == 1
            and self.tt_data_parallel_size > 1
        )

    def load_model(self) -> None:
        loader = TTModelLoader(self.load_config)
        self.model = loader.load_model(
            vllm_config=self.vllm_config, model_config=self.model_config
        )

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        # TT backend currently supports text generation only.
        # (No transcription support yet.)
        return ["generate"]

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        # TT backend does not support pooling/embedding tasks yet.
        return []

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on ``kv_cache_config``.

        Args:
            kv_cache_config: Configuration for the KV cache. May contain one
                group (uniform attention) or multiple groups (hybrid models
                like Gemma3/4 / GPT-OSS that mix sliding-window and full
                attention layers).
        """
        kv_cache_groups = kv_cache_config.kv_cache_groups
        self._validate_kv_cache_groups(kv_cache_groups)

        # Stash on the runner for downstream phases that need to walk the
        # group structure during input prep / forward.
        self.kv_cache_config = kv_cache_config

        # Upstream's hybrid kv cache manager equalises *page size*
        # (block_size × num_kv_heads × head_size × dtype_bytes) across
        # groups, not block_size itself: when groups have different
        # ``num_kv_heads × head_size`` (e.g. Gemma4's full layers use
        # head_dim=512 vs sliding head_dim=256), upstream's
        # ``unify_kv_cache_spec_page_size`` adjusts ``block_size`` per
        # spec instead. Use each group's own ``block_size`` here; the
        # input batch / MultiGroupBlockTable already takes a per-group
        # list. ``self.cache_config.block_size`` (the user-specified
        # value) is still used elsewhere for per-request bounds — that's
        # the smaller of the unified sizes, which conservatively
        # overestimates for the larger-block groups (extra block-table
        # rows allocated, never indexed).
        per_group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]

        max_num_reqs = self.tt_max_batch_size
        max_model_len = self.model_config.max_model_len
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        if self._is_lane_mode:
            # Lane-chunked persistent batch: rows are partitioned into
            # ``tt_data_parallel_size`` lanes of ``tt_per_lane_max_num_seqs``
            # slots, so a request's persistent row is its stable device slot.
            # ``num_lanes * per_lane == tt_max_batch_size`` (the global
            # max_num_seqs in lane mode), so the slot count matches the device
            # batch the merged step executes.
            self.input_batch: InputBatch = TTLaneInputBatch(
                num_lanes=self.tt_data_parallel_size,
                per_lane=self.tt_per_lane_max_num_seqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                vocab_size=self.vocab_size,
                block_sizes=per_group_block_sizes,
                kernel_block_sizes=per_group_block_sizes,
                logitsprocs=self._host_logitsprocs,
            )
        else:
            self.input_batch = InputBatch(
                max_num_reqs=max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                vocab_size=self.vocab_size,
                block_sizes=per_group_block_sizes,
                kernel_block_sizes=per_group_block_sizes,
                logitsprocs=self._host_logitsprocs,
            )

        # The block tables in the persistent input batch have
        # max_num_blocks_per_req = cdiv(max_model_len, block_size) but this
        # does not take into account num blocks in KV cache. Actual max is
        # min of these two. Used to slice block tables during input prep.
        self.max_num_blocks_per_req = min(
            cdiv(max_model_len, self.cache_config.block_size),
            kv_cache_config.num_blocks,
        )

        # Number of kv_cache_groups; needed by DP gather/merge to pack
        # per-group block tables into the gather payload.
        self._num_kv_cache_groups = len(kv_cache_groups)
        # Cache layer→group index mapping for hybrid models so submit_*
        # can expand ``block_tables_per_group`` into ``block_tables_per_layer``
        # without re-deriving vLLM's group construction order. Non-hybrid
        # configurations (single group) skip the expansion entirely. The
        # check is on ``len(kv_cache_groups)`` rather than the model class
        # so it works on every DP rank — only ``data_parallel_rank_local
        # == 0`` actually loads ``self.model``.
        self._layer_to_group_idx: list[int] | None = None
        if len(kv_cache_groups) > 1:
            num_layers = self.model_config.get_num_layers_by_block_type(
                self.parallel_config, "attention"
            )
            mapping: list[int | None] = [None] * num_layers
            for g_idx, group in enumerate(kv_cache_groups):
                for layer_name in group.layer_names:
                    idx = _parse_layer_index(layer_name)
                    mapping[idx] = g_idx
            missing = [i for i, g in enumerate(mapping) if g is None]
            if missing:
                raise ValueError(
                    f"No KVCacheGroupSpec covers layer indices {missing} "
                    f"on hybrid model; every attention layer must appear "
                    "in some group's layer_names."
                )
            self._layer_to_group_idx = mapping  # type: ignore[assignment]

        # Only DP rank 0 allocates KV cache.
        if self.parallel_config.data_parallel_rank_local != 0:
            return

        self.kv_caches = self._allocate_kv_caches(kv_cache_config)

    @staticmethod
    def _validate_kv_cache_groups(kv_cache_groups: list) -> None:
        if not kv_cache_groups:
            raise ValueError("kv_cache_config has no groups")
        for group in kv_cache_groups:
            # ``UniformTypeKVCacheSpecs`` wraps several same-type (e.g. all
            # FullAttentionSpec) layer specs that differ only in shape. vLLM
            # emits it when a hybrid model disables hybrid kv-cache groups and
            # exposes heterogeneous per-layer KV shapes (e.g. Gemma4 with
            # ``_HYBRID_KV_CACHE_GROUPS_ENABLED = False``: sliding layers 8x256,
            # full layers 1x512). We unwrap it to per-layer specs in
            # ``_build_per_layer_specs``, so accept it here too.
            if not isinstance(
                group.kv_cache_spec, (AttentionSpec, UniformTypeKVCacheSpecs)
            ):
                raise TypeError(
                    f"Expected AttentionSpec/UniformTypeKVCacheSpecs for group "
                    f"{group.layer_names}, got {type(group.kv_cache_spec).__name__}"
                )

    def _kv_cache_shape(
        self, spec: AttentionSpec, num_blocks: int
    ) -> tuple[int, int, int, int]:
        """Per-buffer shape ``(num_blocks, num_kv_heads, block_size,
        head_size)`` from a group's attention spec.

        Legacy specs count heads before TT mesh sharding. Models emitting
        device-local specs opt in with ``kv_cache_specs_use_local_heads``;
        dividing those heads again would underallocate the cache.
        """
        num_devices = self.num_devices // self.tt_data_parallel_size
        num_kv_heads = spec.num_kv_heads
        if not getattr(self.model, "kv_cache_specs_use_local_heads", False):
            num_kv_heads //= min(num_devices, num_kv_heads)
        return (num_blocks, num_kv_heads, spec.block_size, spec.head_size)

    def _allocate_kv_caches(self, kv_cache_config: KVCacheConfig) -> Any:
        """Allocate KV cache tensors, falling back to legacy uniform API.

        Builds a ``per_layer_specs`` list of ``(shape, dtype)`` tuples — one
        entry per attention layer in model layer-index order. Hybrid models
        opt in to per-layer allocation by exposing
        ``allocate_kv_cache_per_layer(per_layer_specs)``; legacy models keep
        the older ``allocate_kv_cache(shape, dtype, num_layers)`` signature
        and we adapt to it here, asserting the per-layer specs are uniform.
        """
        num_layers = self.model_config.get_num_layers_by_block_type(
            self.parallel_config, "attention"
        )
        per_layer_specs = self._build_per_layer_specs(kv_cache_config, num_layers)

        if hasattr(self.model, "allocate_kv_cache_per_layer"):
            return self.model.allocate_kv_cache_per_layer(per_layer_specs)

        # Legacy ``allocate_kv_cache(shape, dtype, num_layers)`` API: every
        # layer must have the same shape/dtype. The third tuple element is
        # the tensor index, which is irrelevant for the legacy uniform
        # path (each layer gets its own buffer there).
        shape, dtype, _ = per_layer_specs[0]
        for entry_shape, entry_dtype, _ in per_layer_specs[1:]:
            if (entry_shape, entry_dtype) != (shape, dtype):
                raise NotImplementedError(
                    f"{type(self.model).__name__} only implements legacy "
                    "allocate_kv_cache; hybrid attention models must "
                    "override allocate_kv_cache_per_layer."
                )
        return self.model.allocate_kv_cache(shape, dtype, len(per_layer_specs))

    def _block_tables_per_layer(
        self, block_tables_per_group: list[torch.Tensor]
    ) -> list[torch.Tensor] | None:
        """Expand per-group block tables to per-layer using the cached mapping.

        Returns None for non-hybrid models (the mapping is only populated
        when the model class exposes ``get_kv_cache_spec``). The output is
        a list of ``num_layers`` tensors, where entry ``i`` is the block
        table for layer ``i``'s containing kv_cache_group — what hybrid
        bridges hand to the underlying TT model so attention layer ``i``
        can index its own paged pool (full vs. sliding-window) without
        knowing how vLLM ordered the groups.
        """
        if self._layer_to_group_idx is None:
            return None
        result = [block_tables_per_group[g] for g in self._layer_to_group_idx]
        # Pad to the warmup shape ``[max_batch, max_num_blocks_per_req]``.
        # The model side (``Transformer._page_tables_to_ttnn``) allocates
        # *persistent* ttnn device tensors at this shape during warmup so
        # captured traces can be replayed against stable device addresses;
        # ``ttnn.copy_host_to_device_tensor`` then asserts
        # ``host_shape == device_shape`` when the runtime per-layer block
        # tables push their content into those buffers. Padding rows with
        # zeros is harmless — the kernel only reads up to each layer's
        # active block count.
        max_batch = self.tt_max_batch_size
        target_shape = (max_batch, self.max_num_blocks_per_req)
        padded = []
        for bt in result:
            if bt is None:
                padded.append(None)
                continue
            if bt.shape == target_shape:
                padded.append(bt)
                continue
            full = torch.zeros(target_shape, dtype=bt.dtype)
            rows = min(bt.shape[0], target_shape[0])
            cols = min(bt.shape[1], target_shape[1])
            full[:rows, :cols] = bt[:rows, :cols]
            padded.append(full)
        return padded

    def _build_per_layer_specs(
        self, kv_cache_config: KVCacheConfig, num_layers: int
    ) -> list[tuple[tuple[int, int, int, int], Any, int]]:
        """Resolve ``KVCacheConfig`` → list of ``(shape, dtype, tensor_idx)``
        per layer in model layer-index order.

        ``tensor_idx`` identifies the unique DRAM buffer that each layer's
        KV cache lives in. Multiple layers from different
        ``KVCacheGroupSpec``\\ s can carry the same ``tensor_idx`` — this
        is upstream's tensor-sharing model: with a 5:1 sliding/full split,
        a full-attention layer and several sliding-window layers all share
        one buffer, and they index disjoint slots within it via per-group
        block tables (vLLM's ``BlockPool`` allocates disjoint block IDs
        across groups, so the shared tensor is sized for the worst-case
        full-attention demand and the sliding-window layers fit within
        their window's worth of slots).

        Single-group (uniform-attention) models keep the previous behavior:
        every layer gets a unique buffer and ``tensor_idx == layer_idx``.
        """
        kv_cache_groups = kv_cache_config.kv_cache_groups

        if len(kv_cache_groups) == 1:
            spec = kv_cache_groups[0].kv_cache_spec
            if isinstance(spec, AttentionSpec):
                shape = self._kv_cache_shape(spec, kv_cache_config.num_blocks)
                return [(shape, spec.dtype, i) for i in range(num_layers)]
            # ``UniformTypeKVCacheSpecs``: one group / one block table, but the
            # wrapped per-layer specs have heterogeneous shapes (e.g. Gemma4
            # with hybrid groups disabled: sliding 8x256 vs full 1x512). Every
            # layer indexes the same ``num_blocks`` block table but gets its own
            # DRAM buffer sized to its own spec, so ``tensor_idx == layer_idx``
            # (matching the uniform single-group convention above).
            assert isinstance(spec, UniformTypeKVCacheSpecs)
            per_layer: list[tuple[tuple[int, int, int, int], Any, int] | None] = [
                None
            ] * num_layers
            for layer_name, layer_spec in spec.kv_cache_specs.items():
                assert isinstance(layer_spec, AttentionSpec)
                idx = _parse_layer_index(layer_name)
                if not 0 <= idx < num_layers:
                    raise ValueError(
                        f"Layer index {idx} parsed from '{layer_name}' is out "
                        f"of range for {num_layers} attention layers"
                    )
                shape = self._kv_cache_shape(layer_spec, kv_cache_config.num_blocks)
                per_layer[idx] = (shape, layer_spec.dtype, idx)
            missing = [i for i, e in enumerate(per_layer) if e is None]
            if missing:
                raise ValueError(
                    f"UniformTypeKVCacheSpecs missing per-layer specs for "
                    f"layer indices {missing}"
                )
            return per_layer  # type: ignore[return-value]

        # Multi-group: walk ``kv_cache_tensors`` (one entry per unique DRAM
        # buffer) and assign every layer in ``shared_by`` the same
        # ``tensor_idx``. Shape/dtype come from the layer's own group spec.
        spec_by_layer_name: dict[str, AttentionSpec] = {}
        for group in kv_cache_groups:
            assert isinstance(group.kv_cache_spec, AttentionSpec)
            for layer_name in group.layer_names:
                spec_by_layer_name[layer_name] = group.kv_cache_spec

        per_layer: list[tuple[tuple[int, int, int, int], Any, int] | None] = [
            None
        ] * num_layers
        for tensor_idx, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            for layer_name in kv_cache_tensor.shared_by:
                spec = spec_by_layer_name.get(layer_name)
                if spec is None:
                    raise ValueError(
                        f"KVCacheTensor.shared_by names layer '{layer_name}' "
                        "but it doesn't appear in any kv_cache_group"
                    )
                idx = _parse_layer_index(layer_name)
                if not 0 <= idx < num_layers:
                    raise ValueError(
                        f"Layer index {idx} parsed from '{layer_name}' is "
                        f"out of range for {num_layers} attention layers"
                    )
                if per_layer[idx] is not None:
                    raise ValueError(
                        f"Layer index {idx} (from '{layer_name}') is named "
                        "by more than one KVCacheTensor.shared_by; each "
                        "layer must map to exactly one DRAM buffer"
                    )
                shape = self._kv_cache_shape(spec, kv_cache_config.num_blocks)
                per_layer[idx] = (shape, spec.dtype, tensor_idx)

        missing = [i for i, e in enumerate(per_layer) if e is None]
        if missing:
            raise ValueError(
                f"No KVCacheTensor covers layer indices {missing}; "
                "every attention layer must appear in some "
                "kv_cache_tensors[i].shared_by"
            )
        return per_layer  # type: ignore[return-value]

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        """Update the cached states and the persistent batch with the
        scheduler output.
        The updated states are used in `_prepare_model_inputs` to create the
        input tensors for the model.
        Based on _update_states for GPU/TPU backends.
        """
        persistent_batch_layout_changed = False

        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        self._release_dead_state_slots(scheduler_output)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)
                persistent_batch_layout_changed = True

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)
            persistent_batch_layout_changed = True

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            self.requests[req_id] = build_cached_request_state(new_req_data)
            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids

            # Update the cached states.
            apply_cached_req_state_update(
                req_state,
                num_computed_tokens,
                new_block_ids,
                resumed_from_preemption,
                num_output_tokens=req_data.num_output_tokens[i],
                all_token_ids=req_data.all_token_ids.get(req_id),
            )

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            if resumed_from_preemption:
                # Resume can arrive before this row was evicted. Replace its
                # token prefix, logical lengths and block table from the
                # reconciled state instead of appending to the freed KV mapping.
                self.input_batch.add_request(req_state, req_index)
                persistent_batch_layout_changed = True
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)
                # Reset also refreshes persistent device page tables. The
                # build_model_input drain applies pending tokens before any
                # reset reloads token/position inputs from host state.
                if any(new_block_ids):
                    persistent_batch_layout_changed = True

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            # Fill the empty index, or append to the end.
            req_index = removed_req_indices.pop() if removed_req_indices else None
            self.input_batch.add_request(req_state, req_index)
            persistent_batch_layout_changed = True

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)
            persistent_batch_layout_changed = True
        # Mark decode layout changed if persistent batch changed. This is
        # sticky across steps and will be consumed by the next decode batch.
        if persistent_batch_layout_changed:
            self._decode_layout_changed_since_last_decode = True

        # Refresh logits processors with batch state changes
        self.input_batch.refresh_logitsprocs()

    def _validate_mm_feature(self, mm_feature: MultiModalFeatureSpec) -> None:
        """Validate the multimodal feature is an image."""
        if mm_feature.modality != "image":
            raise NotImplementedError("Only images are supported for now")

    def _gather_multi_modal_inputs(
        self, req_indices: list[int] | None = None
    ) -> dict[str, Any]:
        """
        Gather and batch multi-modal inputs for the current persistent batch.

        Currently only supports image inputs in the "pixel_values" and
        "image_grid_thw" fields.

        Returns a dict with keys "pixel_values" and "image_grid_thw".
        Each value is a list aligned with the persistent batch order
        (`self.input_batch.req_ids[:num_reqs]`).

        For request i:
        - If it has no `mm_features`, the entry is None.
        - Otherwise the entry is a list aligned with that request's
          `mm_features` (currently only images), where each element is a
          tensor (or None if that feature has no data).

        Example (3 scheduled requests: text-only, 1 image, 2 images):
        {
          "pixel_values": [
            None,
            [pv_req1_img0],
            [pv_req2_img0, pv_req2_img1],
          ],
          "image_grid_thw": [
            None,
            [ig_req1_img0],
            [ig_req2_img0, ig_req2_img1],
          ],
        }
        """

        multi_modal_kwargs: dict[str, Any] = {
            "pixel_values": [],
            "image_grid_thw": [],
        }

        if req_indices is None:
            req_indices = list(range(self.input_batch.num_reqs))
        # The model input tensors are built in persistent batch order, so
        # multi-modal inputs must follow the same order (not just new reqs).
        for batch_index in req_indices:
            req_id = self.input_batch.req_ids[batch_index]
            req_state = self.requests[req_id]

            if not req_state.mm_features:
                multi_modal_kwargs["pixel_values"].append(None)
                multi_modal_kwargs["image_grid_thw"].append(None)
                continue

            pv_array: list[torch.Tensor | None] = []
            image_grid_thw_array: list[torch.Tensor | None] = []
            for mm_feature in req_state.mm_features:
                self._validate_mm_feature(mm_feature)
                item = mm_feature.data
                if item is None:
                    pv_array.append(None)
                    image_grid_thw_array.append(None)
                    continue
                pv_array.append(item["pixel_values"].data)
                image_grid_thw_array.append(
                    item["image_grid_thw"].data if "image_grid_thw" in item else None
                )

            multi_modal_kwargs["pixel_values"].append(pv_array)
            multi_modal_kwargs["image_grid_thw"].append(image_grid_thw_array)

        return multi_modal_kwargs

    def _sampling_params_for_padded_decode(
        self,
        sample_params,
        req_indices: list[int],
        pad_to: int,
    ) -> TTSamplingParams:
        """Build per-row sampling tensors padded to ``pad_to`` for DP gather."""
        defaults = sample_params.create_default_tensors()

        def take(name: str) -> torch.Tensor:
            local = getattr(sample_params, name)[req_indices]
            if local.shape[0] >= pad_to:
                return local[:pad_to]
            pad_count = pad_to - local.shape[0]
            return torch.cat([local, defaults[name][:pad_count]])

        num_logprobs = take("num_logprobs")
        return TTSamplingParams(
            temperature=take("temperature"),
            top_k=take("top_k"),
            top_p=take("top_p"),
            presence_penalty=take("presence_penalty"),
            frequency_penalty=take("frequency_penalty"),
            repetition_penalty=take("repetition_penalty"),
            seed=take("seed"),
            num_logprobs=num_logprobs,
            enable_log_probs=num_logprobs >= 0,
        )

    # --- Per-request device state slots (see ``self._req_state_slot``) ---

    def _release_dead_state_slots(self, scheduler_output: SchedulerOutput) -> None:
        """Drop the slot claims of requests whose device state is no longer
        authoritative.

        The predicate is not "did the step schedule it": a RUNNING request the step
        left out (what every prefill step does to the whole decode batch) still owns
        live state and must keep its slot. Only two states release:

        - FINISHED: the request is gone.
        - PREEMPTED: ``Scheduler._preempt_request`` freed the KV blocks and reset
          ``num_computed_tokens``, so a resume re-prefills the prompt plus every
          generated token and writes the slot's final state itself. Nothing reads the
          old contents, and holding the slot only inflates ``held`` in
          ``_alloc_prefill_state_slots``, where a shortfall is a hard assert. The
          per-slot seed RNG does not need the hold either: the device seed is derived
          from the absolute decode position, not from slot residency.

        ``preempted_req_ids`` is typed optional, hence the ``or ()``.
        """
        for req_id in scheduler_output.finished_req_ids:
            self._req_state_slot.pop(req_id, None)
        for req_id in scheduler_output.preempted_req_ids or ():
            self._req_state_slot.pop(req_id, None)

    def _alloc_prefill_state_slots(self, row_req_ids: list[str]) -> list[int]:
        """Pick each prefilling request's state slot, skipping slots that live
        off-batch requests own. Prefers its own row (where it decodes), so the
        steady state moves nothing.

        Exhaustion is unreachable: holders and prefills are disjoint and both count
        against ``max_num_seqs``, which is ``n_slots``. Reaching an assert means the
        map has stopped describing the device, and nothing on the host can recover
        it, so fail rather than guess and return plausible, wrong text.
        """
        n_slots = self.tt_per_lane_max_num_seqs
        assert len(row_req_ids) <= n_slots, (
            f"{len(row_req_ids)} prefill(s) exceed the {n_slots} device state "
            "slots; admission is the scheduler's job, not this function's"
        )
        prefilling = set(row_req_ids)
        held = {
            slot
            for req_id, slot in self._req_state_slot.items()
            if req_id not in prefilling and req_id in self.requests
        }
        slots: list[int] = []
        for row, req_id in enumerate(row_req_ids):
            if row not in held:
                slot = row
            else:
                free = [s for s in range(n_slots) if s not in held]
                assert free, (
                    f"no free device state slot for {len(row_req_ids)} prefill(s): "
                    f"held={sorted(held)}, capacity={n_slots}, "
                    f"map={self._req_state_slot}"
                )
                slot = free[0]
            held.add(slot)
            self._req_state_slot[req_id] = slot
            slots.append(slot)
        return slots

    def _merge_dp_prefill_slots(
        self, inputs: list[TTModelInput | None]
    ) -> list[int] | None:
        """Lift each gathered-DP rank's prefill slots from its local ``[0, stride)``
        space into the global one decode gathers from (``rank * stride``, matching
        the offsets applied to ``slot_remap``). Rank order, so the slots line up
        with the merged rows. None when no rank supplied any, leaving
        ``submit_prefill``'s scheduling-order fallback in place for stateless
        models."""
        stride = self.tt_per_lane_max_num_seqs
        active = [mi for mi in inputs if mi is not None]
        if not any(mi.prefill_empty_slots for mi in active):
            return None
        merged: list[int] = []
        for rank, mi in enumerate(inputs):
            if mi is None:
                continue
            slots = mi.prefill_empty_slots
            # The row concat below takes every ``mi is not None``, so the slots must
            # match it one for one, in order. Nothing else enforces that.
            assert slots is not None and len(slots) == mi.input_tokens.shape[0], (
                f"gathered-DP prefill: rank {rank} contributes "
                f"{mi.input_tokens.shape[0]} row(s) but {slots} state slot(s)"
            )
            merged.extend(rank * stride + slot for slot in slots)
        return merged

    def _decode_state_slot_remap(self, row_req_ids: list[str]) -> torch.Tensor | None:
        """Gather permutation taking each request's state to its decode row: row
        ``i`` reads slot ``remap[i]``. Always full slot width (no OOB gather); None
        means identity, so skip it. Commits the move to ``self._req_state_slot`` for
        every request the permutation touches, off-batch holders included."""
        n_slots = self.tt_per_lane_max_num_seqs
        # More decode rows than slots means the batch cannot be described at all, so
        # truncating would just drop a request's state silently.
        assert len(row_req_ids) <= n_slots, (
            f"{len(row_req_ids)} decode row(s) exceed the {n_slots} device state slots"
        )
        want: list[int] = []
        for req_id in row_req_ids:
            slot = self._req_state_slot.get(req_id)
            # Every decoding request got a slot at prefill. Inventing one here is
            # how a second request ends up recorded at an owned slot.
            assert slot is not None, (
                f"decoding request {req_id!r} has no device state slot: "
                f"map={self._req_state_slot}"
            )
            want.append(slot)
        if len(set(want)) != len(want) or any(not 0 <= s < n_slots for s in want):
            # Refusing the remap is not the safe option: no gather goes out for the
            # WHOLE batch, so every off-row request reads another's state.
            duplicates = sorted({s for s in want if want.count(s) > 1})
            raise AssertionError(
                f"TT decode state slots are not a permutation: want={want}, "
                f"duplicated={duplicates}, capacity={n_slots}, "
                f"map={self._req_state_slot}"
            )
        taken = set(want)
        remap = want + [s for s in range(n_slots) if s not in taken]
        # The gather moves every slot, not just the batch rows: ownership must follow.
        # Build the new map whole and swap it in, so no read sees it half-written.
        by_slot: dict[int, list[str]] = {}
        for req_id, slot in self._req_state_slot.items():
            by_slot.setdefault(slot, []).append(req_id)
        moved = dict(self._req_state_slot)
        for row, slot in enumerate(remap):
            for req_id in by_slot.get(slot, ()):
                moved[req_id] = row
        self._req_state_slot = moved
        if all(remap[i] == i for i in range(n_slots)):
            return None
        return torch.tensor(remap, dtype=torch.int32)

    @staticmethod
    def _build_host_generators(
        input_batch: InputBatch,
        req_indices: list[int],
        intermediate_prefill_mask: torch.Tensor | None,
    ) -> dict[int, torch.Generator]:
        intermediate_rows = (
            set(intermediate_prefill_mask.nonzero().view(-1).tolist())
            if intermediate_prefill_mask is not None
            else set()
        )
        generators: dict[int, torch.Generator] = {}
        rows_to_advance: list[int] = []
        for local_row, batch_row in enumerate(req_indices):
            generator = input_batch.sampling.generators.get(batch_row)
            if generator is None:
                continue
            if local_row in intermediate_rows:
                generators[local_row] = clone_torch_generator(generator)
            else:
                generators[local_row] = generator
                rows_to_advance.append(batch_row)
        input_batch.advance_generators(rows_to_advance)
        return generators

    def _prepare_model_inputs(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput | None,
    ) -> TTModelInput:
        """Build a ``TTModelInput`` for one prefill or decode step.

        Reads the current persistent ``self.input_batch`` and assembles the
        padded, fixed-shape tensors a TT model needs (constant shapes are
        required for ttnn tracing). This is the input builder shared by the
        non-DP and gathered multi-process DP paths; each operates on its whole
        local ``input_batch`` (gathered DP later gathers the per-rank results to
        rank 0). Single-process lane mode does not use this builder -- it builds
        its merged device input directly from ``TTLaneInputBatch``.

        Args:
            scheduler_output: Scheduler decisions for this step. Used to detect
                prefill vs. decode and (for grammar) which requests carry
                structured-output bitmasks.
            grammar_output: Structured-output bitmasks for this step, or
                ``None`` when no request uses guided decoding.

        Returns:
            A ``TTModelInput`` with tokens, positions, block tables, sampling
            params and host-only metadata, padded to the appropriate wire
            capacity. ``unpadded_batch_size`` records the real (pre-pad)
            request count.
        """
        assert scheduler_output.total_num_scheduled_tokens > 0
        input_batch = self.input_batch
        batch_num_reqs = input_batch.num_reqs
        assert batch_num_reqs > 0

        # The whole local batch.
        req_indices = list(range(batch_num_reqs))
        num_reqs = len(req_indices)

        # Pad decode to the per-rank/per-lane wire capacity by default. The
        # DP-decode gather packs and unpacks each rank at
        # ``tt_per_lane_max_num_seqs``; padding to ``input_batch.max_num_reqs``
        # (the *global* gathered capacity ``max_num_seqs * dp_size`` for
        # gathered multi-process DP) would carry too many rows in the
        # never-trimmed tokens/positions/block_tables fields and desync the
        # packed gather layout. For non-DP these two capacities are equal.
        #
        # Models that declare ``tt_supported_decode_batch_sizes`` (e.g. Gemma4)
        # may pad only to the nearest supported size >= num_reqs, so B=1 is not
        # forced through a B=max graph.
        #
        # A model must declare only the buckets it has actually captured a decode
        # trace for on this instance: padding to a bucket with no captured trace
        # leaves the device in an undefined state. There is deliberately no
        # separate "warmed" list to fall back from -- a bucket that is not warmed
        # is not supported.
        decode_pad_to = self.tt_per_lane_max_num_seqs
        decode_buckets = getattr(self.model, "tt_supported_decode_batch_sizes", None)
        if decode_buckets:
            bucket = next(
                (
                    int(b)
                    for b in sorted(int(x) for x in decode_buckets)
                    if int(b) >= num_reqs and int(b) <= self.tt_per_lane_max_num_seqs
                ),
                self.tt_per_lane_max_num_seqs,
            )
            decode_pad_to = bucket

        # Second dim of each block table is (ceil(max_model_len / block_size)).
        # Slice/pad to ``self.max_num_blocks_per_req``: slicing handles
        # over-wide tables when the total KV-cache limit is tighter than
        # ``ceil(max_model_len / block_size)``, and padding handles
        # under-wide ones from hybrid kv-cache groups whose native
        # block-table widths differ after upstream page-size unification.
        target_width = self.max_num_blocks_per_req
        block_tables_per_group = input_batch.block_tables_for_rows(
            req_indices, target_width
        )

        # DP optimization: don't send padding blocks if possible to reduce
        # overhead from gathering inputs to rank 0 and rely on DP concat
        # function to pad to global max blocks.
        if self.tt_data_parallel_size > 1:
            max_tokens_in_batch = max(input_batch.num_tokens[i] for i in req_indices)
            max_blocks_in_batch = cdiv(
                max_tokens_in_batch, self.cache_config.block_size
            )
            block_tables_per_group = [
                bt[:, :max_blocks_in_batch] for bt in block_tables_per_group
            ]

        # Group-0 view kept on TTModelInput.block_tables for back-compat with
        # the existing single-tensor consumers (DP pack/gather, decode_forward
        # page_table arg). Hybrid models additionally consume
        # ``block_tables_per_group`` via the ``page_tables_per_group`` kwarg
        # in submit_prefill / submit_decode; the legacy generator_vllm
        # wrappers strip it on the way through and raise loudly if the list
        # has more than one entry.
        block_tables = block_tables_per_group[0]

        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        num_sched = scheduler_output.num_scheduled_tokens
        # A "prefill" step can contain:
        # - brand new requests (scheduled_new_reqs), and/or
        # - resumed-from-preemption requests (scheduled_cached_reqs with
        #   resumed_req_ids set) that need to replay tokens to rebuild KV,
        #   and/or
        # - chunked-prefill continuations (cached requests that haven't
        #   finished computing all their prompt tokens yet).
        has_chunked_continuation = any(
            input_batch.num_computed_tokens_cpu[input_batch.req_id_to_index[req_id]]
            < input_batch.num_prompt_tokens[input_batch.req_id_to_index[req_id]]
            for req_id in cached_reqs.req_ids
            if req_id not in cached_reqs.resumed_req_ids
        )
        is_prompt = (
            len(scheduler_output.scheduled_new_reqs) > 0
            or bool(cached_reqs.resumed_req_ids)
            or has_chunked_continuation
        )
        sample_params = input_batch.sampling
        intermediate_prefill_mask: torch.Tensor | None = None
        if is_prompt:
            # NOTE: In SchedulerOutput, "cached" means "request data already
            # cached on the worker", not necessarily "decode". During a prefill
            # step we can legitimately see cached requests if they are resumed
            # from preemption (still prefill work) or are chunked-prefill
            # continuations (`num_computed < num_prompt_tokens` - still prefilling).
            if cached_reqs.num_reqs > 0:
                any_decode_in_prefill = any(
                    req_id not in cached_reqs.resumed_req_ids
                    and input_batch.num_computed_tokens_cpu[
                        input_batch.req_id_to_index[req_id]
                    ]
                    >= input_batch.num_prompt_tokens[
                        input_batch.req_id_to_index[req_id]
                    ]
                    for req_id in cached_reqs.req_ids
                )
                assert not any_decode_in_prefill, (
                    "Prefill batch should not include decode cached requests "
                    "(cached req_id that has finished its prompt)."
                )

            # num_computed_tokens for each request is the input position
            # (=computed previously and cached)
            input_positions = input_batch.num_computed_tokens_cpu[req_indices]
            # Chunk-aware `prompt_lens`: for each request, the "prompt length"
            # the generator sees is `start_pos` + `chunk_len`, i.e., the position
            # up to which tokens should be processed.  The generator slices
            # `tokens[start_pos : prompt_lens]` and processes that chunk.
            # - Full prefill (no chunking): `start_pos=0`, `chunk_len=prompt_len`
            #   => `prompt_lens = prompt_len` (same as before).
            # - APC hit: `start_pos=cached`, `chunk_len=prompt_len-cached`
            #   => `prompt_lens = prompt_len` (same as before).
            # - Chunked continuation: `start_pos=computed`, `chunk_len=budget`
            #   => `prompt_lens = computed + budget` (the chunk end position).
            chunk_lens = np.array(
                [num_sched[input_batch.req_ids[i]] for i in req_indices],
                dtype=np.int64,
            )
            prompt_lens = input_positions + chunk_lens
            intermediate_prefill_mask = torch.from_numpy(
                prompt_lens < input_batch.num_tokens[req_indices]
            )
            max_prefill_tokens = int(max(prompt_lens))
            input_tokens = input_batch.token_ids_cpu_tensor[
                req_indices, :max_prefill_tokens
            ]
            reset_batch = False
        else:
            positions_np = input_batch.num_tokens[req_indices] - 1
            input_positions = torch.from_numpy(positions_np)
            input_tokens = input_batch.token_ids_cpu_tensor[
                req_indices, positions_np
            ].view(-1, 1)
            prompt_lens = None
            # For on-device decode sampling, tell the backend if the padded
            # decode batch layout changed since the previous step.
            reset_batch = self._decode_layout_changed_since_last_decode
            self._decode_layout_changed_since_last_decode = False

            # TODO: Remove once TT models can support arbitrary batch sizes.
            # Pad decode to the lane/rank wire capacity.
            if input_tokens.shape[0] < decode_pad_to:
                batch_pad = decode_pad_to - input_tokens.shape[0]
                input_tokens = torch.cat(
                    [input_tokens, torch.zeros(batch_pad, 1, dtype=torch.int32)]
                )
                # Pad positions with -1 to indicate no position
                input_positions = torch.cat(
                    [input_positions, torch.ones(batch_pad, dtype=torch.int32) * -1]
                )
                # Pad each per-group block table to max_num_reqs so DP
                # gather produces a fixed-shape payload regardless of how
                # many users are active on this rank. Keep ``block_tables``
                # aliased to the (now padded) group-0 view, matching the
                # alias set up where ``block_tables_per_group`` is built.
                block_tables_per_group = [
                    torch.cat(
                        [bt, torch.zeros(batch_pad, bt.shape[1], dtype=bt.dtype)],
                        dim=0,
                    )
                    for bt in block_tables_per_group
                ]
                block_tables = block_tables_per_group[0]
                # Sampling parameters are intentionally NOT padded here. The
                # per-row wire tensors are built below by
                # ``_sampling_params_for_padded_decode``, which gathers this
                # build's ``req_indices`` rows and right-pads with fresh
                # defaults. The persistent ``input_batch.sampling`` tail is
                # never read, so there is nothing to default in place.

        tt_sampling_params = slice_tt_sampling_params(sample_params, req_indices)
        if not is_prompt and input_tokens.shape[0] > len(req_indices):
            # Decode inputs are padded to the rank batch size; pad the sampling
            # params to match, right-filling padding rows with neutral defaults.
            tt_sampling_params = self._sampling_params_for_padded_decode(
                sample_params, req_indices, input_tokens.shape[0]
            )

        if self.model_config.is_multimodal_model and is_prompt:
            multi_modal_kwargs = self._gather_multi_modal_inputs(
                req_indices=req_indices
            )
        else:
            multi_modal_kwargs = {}

        # If we're not using structured outputs, grammar_bitmask is None.
        bitmask = grammar_output.grammar_bitmask if grammar_output is not None else None
        has_structured = has_structured_outputs(
            self.requests, scheduler_output, bitmask
        )
        if bitmask is not None:
            # Using torch tensor instead of numpy array for consistency
            # because we need it as tensor for gather.
            bitmask = torch.from_numpy(bitmask)
            # unpadded for prefill, padded for decode
            batch_length = input_tokens.shape[0]
            # `structured_output_request_ids` comes from GrammarOutput as a list
            # of request IDs (bitmask rows are in this order). TT does not support
            # speculative decoding in this path, so we assume a single bitmask row
            # per request.
            structured_output_request_ids = (
                grammar_output.structured_output_request_ids
                if grammar_output is not None
                else []
            )
            bitmask = reorder_grammar_bitmask_for_tt_batch(
                bitmask=bitmask,
                structured_output_request_ids=structured_output_request_ids,
                req_id_to_index=input_batch.req_id_to_index,
                req_indices=req_indices,
                batch_length=batch_length,
            )

        perform_device_sampling = self.check_perform_device_sampling(
            is_decode=not is_prompt,
            has_structured_outputs=has_structured,
        )
        if is_prompt and (
            intermediate_prefill_mask.any()
            or (
                self.tt_data_parallel_size > 1
                and self.scheduler_config.enable_chunked_prefill
            )
        ):
            # A gathered prefill can switch to host sampling when another rank
            # has an intermediate chunk, so every rank needs host generator
            # state available before inputs are gathered.
            perform_device_sampling = False

        # Populate prompt_tokens and output_tokens if penalties are needed
        # (decode only).
        prompt_tokens = None
        output_tokens = None
        output_token_counts = None
        if self.tt_data_parallel_size == 1 and getattr(
            self.model, "needs_sampling_output_counts", False
        ):
            output_token_counts = torch.zeros(input_tokens.shape[0], dtype=torch.int32)
            output_token_counts[:num_reqs] = torch.from_numpy(
                input_batch.num_tokens[req_indices]
                - input_batch.num_prompt_tokens[req_indices]
            )
        if (not input_batch.no_penalties) and not is_prompt:
            # Restrict to this build's requests. For lane builds ``req_indices``
            # selects one lane's rows out of the merged batch; passing them
            # keeps penalty history attributed to the right requests instead of
            # the merged batch's leading rows. Non-DP / gathered-DP pass
            # ``range(num_reqs)``, so this is a no-op there.
            prompt_tokens = input_batch.make_prompt_token_ids_tensor(req_indices)
            output_tokens = input_batch.make_output_token_ids_tensor(req_indices)

            # Pad batch to max_num_reqs for non-DP case (don't send padding for
            # DP to reduce overhead from gathering inputs to rank 0).
            if (
                self.tt_data_parallel_size == 1
                and prompt_tokens.shape[0] < input_batch.max_num_reqs
            ):
                batch_pad = input_batch.max_num_reqs - prompt_tokens.shape[0]
                prompt_tokens = torch.cat(
                    [
                        prompt_tokens,
                        torch.full(
                            (batch_pad, prompt_tokens.shape[1]), -1, dtype=torch.int32
                        ),
                    ]
                )
                output_tokens = torch.cat(
                    [
                        output_tokens,
                        torch.full(
                            (batch_pad, output_tokens.shape[1]), -1, dtype=torch.int32
                        ),
                    ]
                )

        # Build host-only sampling params from input_batch. The host sampler
        # interprets every per-request key/index as a row in the batch it is
        # handed, so each of these must be reindexed to this build's rows
        # (lane-local 0..num_reqs-1). ``req_indices`` is ``range(num_reqs)`` for
        # non-DP / gathered-DP, so the remaps below are the identity there and
        # only do real work for lane builds.
        allowed_token_ids_mask = None
        if (
            not input_batch.no_allowed_token_ids
            and input_batch.sampling.allowed_token_ids_mask is not None
        ):
            # Gather already reindexes to lane-local rows.
            allowed_token_ids_mask = input_batch.sampling.allowed_token_ids_mask[
                req_indices
            ].clone()

        # Re-key the bad-words dict (req_index -> token id lists) to lane-local
        # rows. ``condense`` keeps the source keys aligned with active rows.
        src_bad_words = input_batch.sampling.bad_words_token_ids
        bad_words_token_ids = {
            local: src_bad_words[g]
            for local, g in enumerate(req_indices)
            if g in src_bad_words
        }

        # Builtin/custom logits processors hold per-row state over the whole
        # local batch, so the host sampler reuses them as-is.
        logitsprocs = input_batch.sampling.logitsprocs

        generators: dict[int, torch.Generator] = {}
        if not perform_device_sampling:
            generators = self._build_host_generators(
                input_batch, req_indices, intermediate_prefill_mask
            )
            # NOTE: Our sampling paths are different between host and device.
            # Whether a request is sampled on device or host
            # depends also on other requests in the batch.
            # This means sampling is not perfectly deterministic
            # whenever device sampling is enabled.

        block_tables_per_group = [bt.contiguous() for bt in block_tables_per_group]
        block_tables = block_tables_per_group[0]
        # State follows the request, not the row (``self._req_state_slot``), which
        # subsumes the batch's condense-move remap.
        input_batch.reset_slot_remap()
        row_req_ids = [input_batch.req_ids[i] for i in req_indices]
        if is_prompt:
            prefill_empty_slots = self._alloc_prefill_state_slots(row_req_ids)
            slot_remap = None
        else:
            prefill_empty_slots = None
            slot_remap = self._decode_state_slot_remap(row_req_ids)

        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            block_tables=block_tables,
            block_tables_per_group=block_tables_per_group,
            block_tables_per_layer=self._block_tables_per_layer(block_tables_per_group),
            unpadded_batch_size=num_reqs,
            tt_sampling_params=tt_sampling_params,
            multi_modal_kwargs=multi_modal_kwargs,
            perform_device_sampling=perform_device_sampling,
            grammar_bitmask=[bitmask],  # wrap to match DP case
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            output_token_counts=output_token_counts,
            reset_batch=reset_batch,
            slot_remap=slot_remap,
            # Host-only sampling params - wrapped in lists for DP compatibility
            allowed_token_ids_mask_list=[allowed_token_ids_mask],
            bad_words_token_ids_list=[bad_words_token_ids],
            max_num_logprobs=[input_batch.max_num_logprobs],
            logitsprocs_list=[logitsprocs],
            generators_list=[generators],
            # Destination state slot per row. Without it a stateful model falls
            # back to ``range(N)`` and a prefill overwrites a decoding request's
            # state. Stateless models ignore it.
            prefill_empty_slots=prefill_empty_slots,
            intermediate_prefill_mask=intermediate_prefill_mask,
        )

    def build_model_input(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput | None,
    ) -> TTModelInput | None:
        """
        Update internal state with the scheduler output and build
        TTModelInput without executing the model.
        Returns None if there is no scheduled work in this step.

        For data parallel, this function is called by each DP rank to build
        TTModelInput from it's own scheduler output.
        """
        # Pending readbacks must be applied before resumed history is trimmed
        # to the scheduler's retained prefix, or a late output reappears in it.
        if scheduler_output.scheduled_cached_reqs.resumed_req_ids:
            self.async_decode.wait_for_all_pending_async_steps()

        # Update cached state
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            return None

        # ``_update_states`` may have just discovered a layout change that the
        # scheduler-output prediction could not see: it predicts the resets caused by
        # new or resumed requests, but removals, unscheduled requests and batch
        # condensation only surface here, after the drain decision was already made.
        # ``_decode_layout_changed_since_last_decode = True`` implies
        # ``reset_batch=True``: ``_prepare_model_inputs`` below reloads inputs from host
        # state, which a pending async decode step has not been applied to yet. This
        # step is therefore not steady-decode eligible, so drain pending decodes to
        # ensure updated host inputs. No-op when the flag was already set before the
        # step (the caller's drain decision covered it) or when nothing is pending.
        if self._decode_layout_changed_since_last_decode:
            self.async_decode.wait_for_all_pending_async_steps()

        # Prepare model inputs only
        model_input = self._prepare_model_inputs(scheduler_output, grammar_output)
        return model_input

    def can_attempt_steady_decode_from_scheduler(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput | None,
    ) -> bool:
        """Return whether a scheduled non-DP step can overlap steady decode."""
        return self.async_decode.can_attempt_steady_decode_from_scheduler(
            scheduler_output, grammar_output
        )

    def can_attempt_steady_dp_decode_from_scheduler(
        self,
        scheduler_output: SchedulerOutput | None,
        grammar_output: GrammarOutput | None,
    ) -> bool:
        """Check whether one DP rank can participate in steady gathered decode.

        Call this in the gathered-DP path after local scheduling has happened.
        Unlike the non-DP variant, `scheduler_output=None` or a zero-token step
        is treated as steady-eligible because a rank may have no local decode
        work while the global gathered step still overlaps safely.
        """
        return self.async_decode.can_attempt_steady_dp_decode_from_scheduler(
            scheduler_output, grammar_output
        )

    @staticmethod
    def _pad_or_truncate_1d(
        tensor: torch.Tensor,
        max_batch: int,
        pad_value: float | int | bool = 0,
    ) -> torch.Tensor:
        """Flatten ``tensor`` and pad or truncate it to ``max_batch`` elements.

        Used to fit a per-row decode-gather field to the fixed per-rank wire
        size. Padding elements are filled with the caller-supplied ``pad_value``
        (default ``0``). Callers pass the sampling parameter's neutral default
        (see ``SamplingInputBatch.DEFAULTS``) where a bare ``0`` would be an
        invalid value (e.g. ``top_k``), so a padded row can never inject an
        invalid sampling parameter if it is ever sampled.
        """
        flat = tensor.contiguous().view(-1)
        n = int(flat.numel())
        if n == max_batch:
            return flat
        if n > max_batch:
            return flat[:max_batch]
        pad = max_batch - n
        return torch.cat(
            [flat, torch.full((pad,), pad_value, dtype=flat.dtype, device=flat.device)]
        )

    def _decode_gather_slot_remap(
        self, model_input: TTModelInput | None, max_batch: int
    ) -> torch.Tensor:
        if model_input is None or model_input.slot_remap is None:
            return torch.arange(max_batch, dtype=torch.int32)
        flat = model_input.slot_remap.contiguous().view(-1)
        n = int(flat.numel())
        if n == max_batch:
            return flat
        if n > max_batch:
            return flat[:max_batch]
        tail = torch.arange(n, max_batch, dtype=torch.int32, device=flat.device)
        return torch.cat([flat, tail])

    def build_dp_decode_gather_input(
        self,
        model_input: TTModelInput | None,
        max_blocks_decode_batch: int,
        any_structured_inputs: bool,
        any_penalties_inputs: bool,
    ) -> dict[str, Any]:
        """
        Called by each DP rank to build tensorized gather input for decode.
        max_blocks_decode_batch: max blocks in the global DP batch.
        any_structured_inputs: whether the global batch has structured inputs.
        any_penalties_inputs: whether the global batch has penalties.
        Returns dict[str, Any] with keys:
          - "int_inputs": flattened int tensor of constant size.
          - "float_inputs": flattened float tensor of constant size.
          - "sampling_tokens_inputs": Optional[dict[str, torch.Tensor]] with
            keys "prompt_tokens" and "output_tokens", or None if not needed.
        """

        max_batch = self.tt_per_lane_max_num_seqs
        num_groups = self._num_kv_cache_groups
        if model_input is None:
            tokens = torch.zeros((max_batch, 1), dtype=torch.int32)
            positions = torch.full((max_batch,), -1, dtype=torch.int32)
            # One zero-filled block table per kv_cache_group so the gather
            # payload always carries G * B * W block_table ints regardless
            # of whether this rank has local work.
            block_tables_per_group = [
                torch.zeros((max_batch, max_blocks_decode_batch), dtype=torch.int32)
                for _ in range(num_groups)
            ]
            unpadded_batch_size = torch.tensor([0], dtype=torch.int32)
            # Default sampling tensors must match the lane/rank wire capacity,
            # not the merged input batch capacity.
            sampling_defaults = self.input_batch.sampling.create_default_tensors()
            sampling_default_tensors = {
                name: tensor[:max_batch] for name, tensor in sampling_defaults.items()
            }
            temperature = sampling_default_tensors["temperature"]
            top_k = sampling_default_tensors["top_k"]
            top_p = sampling_default_tensors["top_p"]
            presence_penalty = sampling_default_tensors["presence_penalty"]
            frequency_penalty = sampling_default_tensors["frequency_penalty"]
            repetition_penalty = sampling_default_tensors["repetition_penalty"]
            seed = sampling_default_tensors["seed"]
            num_logprobs = sampling_default_tensors["num_logprobs"]
            # enable_log_probs: convert num_logprobs >= 0
            enable_log_probs = sampling_default_tensors["num_logprobs"] >= 0
            max_num_logprobs_val = LOGPROBS_NONE_SENTINEL
        else:
            tokens = model_input.input_tokens
            positions = model_input.input_positions
            if tokens.shape[0] < max_batch:
                batch_pad = max_batch - tokens.shape[0]
                tokens = torch.cat(
                    [
                        tokens,
                        torch.zeros(batch_pad, tokens.shape[1], dtype=tokens.dtype),
                    ]
                )
                positions = torch.cat(
                    [
                        positions,
                        torch.full((batch_pad,), -1, dtype=positions.dtype),
                    ]
                )
            # Pad each group's block_table out to ``max_blocks_decode_batch``
            # so the gather payload has a fixed shape regardless of which
            # group carries the most blocks for this rank.
            block_tables_per_group = []
            for bt in model_input.block_tables_per_group:
                if bt.shape[0] < max_batch:
                    batch_pad = max_batch - bt.shape[0]
                    bt = torch.cat(
                        [bt, torch.zeros(batch_pad, bt.shape[1], dtype=bt.dtype)],
                        dim=0,
                    )
                if bt.shape[1] > max_blocks_decode_batch:
                    bt = bt[:, :max_blocks_decode_batch]
                elif bt.shape[1] < max_blocks_decode_batch:
                    pad_w = max_blocks_decode_batch - bt.shape[1]
                    bt = torch.cat(
                        [
                            bt,
                            torch.zeros((bt.shape[0], pad_w), dtype=bt.dtype),
                        ],
                        dim=1,
                    )
                block_tables_per_group.append(bt)
            assert len(block_tables_per_group) == num_groups, (
                f"build_dp_decode_gather_input: expected {num_groups} "
                f"per-group block tables, got {len(block_tables_per_group)}"
            )
            unpadded_batch_size = torch.tensor(
                [cast(int, model_input.unpadded_batch_size)], dtype=torch.int32
            )
            sampling_params: TTSamplingParams = model_input.tt_sampling_params
            defaults = self.input_batch.sampling.DEFAULTS
            temperature = self._pad_or_truncate_1d(
                sampling_params.temperature, max_batch, defaults["temperature"]
            )
            top_k = self._pad_or_truncate_1d(
                sampling_params.top_k, max_batch, defaults["top_k"]
            )
            top_p = self._pad_or_truncate_1d(
                sampling_params.top_p, max_batch, defaults["top_p"]
            )
            presence_penalty = self._pad_or_truncate_1d(
                sampling_params.presence_penalty,
                max_batch,
                defaults["presence_penalty"],
            )
            frequency_penalty = self._pad_or_truncate_1d(
                sampling_params.frequency_penalty,
                max_batch,
                defaults["frequency_penalty"],
            )
            repetition_penalty = self._pad_or_truncate_1d(
                sampling_params.repetition_penalty,
                max_batch,
                defaults["repetition_penalty"],
            )
            seed = self._pad_or_truncate_1d(
                sampling_params.seed, max_batch, defaults["seed"]
            )
            num_logprobs = self._pad_or_truncate_1d(
                sampling_params.num_logprobs, max_batch, defaults["num_logprobs"]
            )
            # enable_log_probs has no DEFAULTS entry; its neutral default is
            # "logprobs disabled" (num_logprobs default < 0 -> False).
            enable_log_probs = self._pad_or_truncate_1d(
                sampling_params.enable_log_probs, max_batch, False
            )
            max_num_logprobs_val = (
                model_input.max_num_logprobs[0]
                if model_input.max_num_logprobs[0] is not None
                else LOGPROBS_NONE_SENTINEL
            )
        # State slot remap: it moves per-slot GDN state and, under device sampling,
        # reindexes the seed manager. The merged input batch uses global
        # max_num_seqs, but the decode gather wire format is per-lane/per-rank.
        slot_remap = self._decode_gather_slot_remap(model_input, max_batch)
        # Pack into flattened tensors to reduce number of collectives.
        # B = max batch size, W = max_num_blocks_per_req, G = num kv_cache_groups.
        # Layout includes one block_table block per group (G*B*W ints) so
        # hybrid models can carry per-group routing through DP gather; for
        # the legacy single-group case G == 1 and the layout is byte-
        # identical to the pre-hybrid format.
        block_tables_packed = torch.cat(
            [
                bt[:, :max_blocks_decode_batch].contiguous().view(-1)
                for bt in block_tables_per_group
            ],
            dim=0,
        )
        int_inputs = torch.cat(
            [
                tokens.contiguous().view(-1),  # B
                positions.contiguous().view(-1),  # B
                block_tables_packed,  # G*B*W
                unpadded_batch_size.contiguous().view(-1),  # 1
                top_k.contiguous().view(-1),  # B
                seed.contiguous().view(-1),  # B
                num_logprobs.contiguous().view(-1),  # B
                enable_log_probs.contiguous()
                .view(-1)
                .to(torch.int32),  # B (bool->int32)
                torch.tensor([max_num_logprobs_val], dtype=torch.int32),  # 1
                slot_remap.contiguous().view(-1),  # B
            ],
            dim=0,
        ).contiguous()

        if any_structured_inputs:
            if model_input is None or model_input.grammar_bitmask[0] is None:
                has_structured_inputs = torch.tensor([0], dtype=torch.int32)
                bitmasks = torch.zeros(
                    (max_batch, self.bitmask_size), dtype=torch.int32
                )
            else:
                has_structured_inputs = torch.tensor([1], dtype=torch.int32)
                bitmasks = model_input.grammar_bitmask[0]
            if bitmasks.shape[0] > max_batch:
                bitmasks = bitmasks[:max_batch]
            elif bitmasks.shape[0] < max_batch:
                # Padding rows have no active request; fill with all-ones
                # (all tokens allowed) to match the sentinel used everywhere
                # else in the bitmask pipeline.
                pad_rows = max_batch - bitmasks.shape[0]
                bitmasks = torch.cat(
                    [
                        bitmasks,
                        torch.full(
                            (pad_rows, bitmasks.shape[1]), -1, dtype=bitmasks.dtype
                        ),
                    ],
                    dim=0,
                )
            bitmasks = bitmasks.contiguous().view(-1)  # B * bitmask_size
            int_inputs = torch.cat(
                [int_inputs, has_structured_inputs, bitmasks], dim=0
            ).contiguous()

        float_inputs = torch.cat(
            [
                temperature.contiguous().view(-1),  # B
                top_p.contiguous().view(-1),  # B
                presence_penalty.contiguous().view(-1),  # B
                frequency_penalty.contiguous().view(-1),  # B
                repetition_penalty.contiguous().view(-1),  # B
            ],
            dim=0,
        ).contiguous()

        sampling_tokens_inputs = None
        if any_penalties_inputs and model_input is not None:
            sampling_tokens_inputs = {
                "prompt_tokens": model_input.prompt_tokens,
                "output_tokens": model_input.output_tokens,
            }

        # Host-only sampling params for host sampling
        host_only_sample_params = None
        if model_input is not None:
            host_only_sample_params = {
                "allowed_token_ids_mask": model_input.allowed_token_ids_mask_list[0],
                "bad_words_token_ids": model_input.bad_words_token_ids_list[0],
                "logitsprocs": model_input.logitsprocs_list[0],
                "generators": model_input.generators_list[0],
            }

        result = {
            "int_inputs": int_inputs,
            "float_inputs": float_inputs,
            "sampling_tokens_inputs": sampling_tokens_inputs,
            "host_only_sample_params": host_only_sample_params,
        }

        return result

    def concat_dp_model_inputs(
        self,
        inputs,
        is_decode: bool,
        max_blocks_decode_batch: int | None,
        any_structured_inputs: bool,
    ) -> TTModelInput:
        """
        Concatenate a DP-sized set of inputs into a single TTModelInput.
        inputs can be either:
        - For prefill: list[Optional[TTModelInput]]
        - For decode (optimized gather): dict[str, torch.Tensor] with keys:
          - "int_inputs": stacked int32 tensor of shape [world, -1]
          - "float_inputs": stacked float32 tensor of shape [world, -1]
          - "sampling_tokens_inputs":
            Optional[list[dict[str, torch.Tensor]]]
            Only provided when there are requests with penalties.
            One dict per DP rank, each with keys "prompt_tokens" and
            "output_tokens" (tensors padded with -1).
          - "reset_batch": bool for if the batch layout changed
            since the previous step.
          - "all_sample_device": bool for if all ranks can sample on device.
        """

        # Need to pad block tables to global max num blocks for constant shape.
        def pad_block_tables(block_tables):
            max_bt_width = self.max_num_blocks_per_req
            if block_tables.shape[1] < max_bt_width:
                pad_w = max_bt_width - block_tables.shape[1]
                block_tables = torch.cat(
                    [
                        block_tables,
                        torch.zeros(
                            (block_tables.shape[0], pad_w), dtype=block_tables.dtype
                        ),
                    ],
                    dim=1,
                )
            return block_tables

        allowed_token_ids_mask_list: list[torch.Tensor | None] = []
        bad_words_token_ids_list: list[dict[int, list[list[int]]]] = []
        logitsprocs_list: list[LogitsProcessors | None] = []
        max_num_logprobs: list[int | None] = []
        generators_list: list[dict[int, torch.Generator]] = []
        slot_remap = None
        intermediate_prefill_mask = None
        prefill_empty_slots: list[int] | None = None

        if is_decode and isinstance(inputs, dict):
            # For decode, given gathered flattened tensors from all DP ranks.
            # Ints: [toks(B), positions(B), block_tables(B*W),
            #        bs(1), top_k(B), seed(B), num_logprobs(B),
            #        enable_log_probs(B)]
            #   - If any_structured_inputs, also has at the end of the list:
            #     [has_structured_inputs(1), bitmasks(B*bitmask_size)]
            # Floats: [temperature(B), top_p(B), presence_penalty(B),
            #          frequency_penalty(B), repetition_penalty(B)]
            assert max_blocks_decode_batch is not None, (
                "max_blocks_decode_batch must be provided for decode"
            )
            B = self.tt_per_lane_max_num_seqs
            W = max_blocks_decode_batch
            reset_batch = inputs["reset_batch"]
            perform_device_sampling = inputs["all_sample_device"]
            stacked_int: torch.Tensor = inputs["int_inputs"]
            stacked_float: torch.Tensor = inputs["float_inputs"]
            assert isinstance(stacked_int, torch.Tensor) and stacked_int.dim() == 2, (
                "decode expects stacked int_inputs of shape [world, -1]"
            )
            assert (
                isinstance(stacked_float, torch.Tensor) and stacked_float.dim() == 2
            ), "decode expects stacked float_inputs of shape [world, -1]"
            world = int(stacked_int.shape[0])
            total_B = world * B

            # Slice views out of the stacked gather buffers (no per-rank
            # Python lists, no torch.cat). Layout is constant for fixed B.
            off = 0
            input_tokens = stacked_int[:, off : off + B].reshape(total_B, 1)
            off += B
            input_positions = stacked_int[:, off : off + B].reshape(total_B)
            off += B

            max_bt_width = self.max_num_blocks_per_req
            if max_bt_width < W:
                raise ValueError(
                    f"max_blocks_decode_batch={W} exceeds "
                    f"max_num_blocks_per_req={max_bt_width}"
                )
            num_groups = self._num_kv_cache_groups
            # Layout: ``[world, G, B, W]`` because every rank packs its
            # block tables in kv_cache_group order and the gather stacks
            # ranks. Reshape per-group and reassemble each group's table
            # as ``[total_B, W]`` then pad to the kernel-expected width.
            block_tables_raw_per_group = stacked_int[
                :, off : off + num_groups * B * W
            ].reshape(world, num_groups, B, W)
            off += num_groups * B * W
            block_tables_per_group: list[torch.Tensor] = []
            for g in range(num_groups):
                bt_g = block_tables_raw_per_group[:, g, :, :].reshape(total_B, W)
                if max_bt_width != W:
                    padded = bt_g.new_zeros((total_B, max_bt_width))
                    padded[:, :W] = bt_g
                    bt_g = padded
                block_tables_per_group.append(bt_g)
            block_tables = block_tables_per_group[0]

            bs_tensor = stacked_int[:, off]
            off += 1
            batch_size_per_dp = bs_tensor.tolist()

            top_k = stacked_int[:, off : off + B].reshape(total_B)
            off += B
            seed = stacked_int[:, off : off + B].reshape(total_B)
            off += B
            num_logprobs = stacked_int[:, off : off + B].reshape(total_B)
            off += B
            enable_log_probs_int = stacked_int[:, off : off + B].reshape(total_B)
            off += B
            # Convert back to bool tensor
            enable_log_probs = enable_log_probs_int > 0

            # max_num_logprobs: one int per DP rank, always available
            # (packed in int_inputs so it survives even when
            # host_only_sample_params gather is skipped)
            raw_max_num_logprobs = stacked_int[:, off].tolist()
            max_num_logprobs = [
                None if val == LOGPROBS_NONE_SENTINEL else val
                for val in raw_max_num_logprobs
            ]
            off += 1

            # State slot remap: per-rank values are in [0,B), but the merged batch
            # (and the row-sharded SeedManager) uses global indices [0, total_B).
            # Offset each rank's remap values by rank * B.
            raw_remap = stacked_int[:, off : off + B]  # [world, B]
            offsets = torch.arange(world, dtype=torch.int32).unsqueeze(1) * B
            slot_remap = (raw_remap + offsets).reshape(total_B)
            off += B

            # Optional structured inputs: keep as list[Optional[tensor]]
            # per DP rank to match prefill behavior.
            grammar_bitmask_list = []
            if any_structured_inputs:
                has_structured = stacked_int[:, off]
                off += 1
                bitmasks = stacked_int[:, off : off + (B * self.bitmask_size)].reshape(
                    world, B, self.bitmask_size
                )
                off += B * self.bitmask_size
                for r in range(world):
                    if int(has_structured[r].item()) > 0:
                        grammar_bitmask_list.append(bitmasks[r])
                    else:
                        grammar_bitmask_list.append(None)
            else:
                grammar_bitmask_list = [None] * world

            # Extract host-only sampling params
            # from gathered inputs (per-rank lists)
            host_only_sample_params_list = inputs.get("host_only_sample_params")
            if host_only_sample_params_list:
                for rank_params in host_only_sample_params_list:
                    if rank_params is not None:
                        allowed_token_ids_mask_list.append(
                            rank_params.get("allowed_token_ids_mask")
                        )
                        bad_words_token_ids_list.append(
                            rank_params.get("bad_words_token_ids")
                        )
                        logitsprocs_list.append(rank_params.get("logitsprocs"))
                        generators_list.append(rank_params.get("generators", {}))
                    else:
                        allowed_token_ids_mask_list.append(None)
                        bad_words_token_ids_list.append({})
                        logitsprocs_list.append(None)
                        generators_list.append({})
            else:
                # No host-only sampling params - create empty lists
                # Happens when host_only_sample_params gather is skipped
                allowed_token_ids_mask_list = [None] * world
                bad_words_token_ids_list = [{}] * world
                logitsprocs_list = [None] * world
                generators_list = [{}] * world

            off_f = 0
            temperature = stacked_float[:, off_f : off_f + B].reshape(total_B)
            off_f += B
            top_p = stacked_float[:, off_f : off_f + B].reshape(total_B)
            off_f += B
            presence_penalty = stacked_float[:, off_f : off_f + B].reshape(total_B)
            off_f += B
            frequency_penalty = stacked_float[:, off_f : off_f + B].reshape(total_B)
            off_f += B
            repetition_penalty = stacked_float[:, off_f : off_f + B].reshape(total_B)
            off_f += B

            prompt_lens = None
        elif not is_decode:
            input_tokens_list: list[torch.Tensor] = []
            block_tables_list: list[torch.Tensor] = []
            # ``block_tables_per_group_list[g]`` holds one entry per active
            # rank (the rank's group-``g`` block table padded to a common
            # width). Concatenated across ranks at the end so hybrid
            # prefill carries per-group routing through the merged input.
            block_tables_per_group_list: list[list[torch.Tensor]] = [
                [] for _ in range(self._num_kv_cache_groups)
            ]
            input_positions_list: list[
                torch.Tensor
            ] = []  # (prefix cache positions for prefill)
            prompt_lens_list: list[np.ndarray] = []
            batch_size_per_dp = []
            grammar_bitmask_list = []
            # Sampling parameters
            temperature_list: list[torch.Tensor] = []
            top_k_list: list[torch.Tensor] = []
            top_p_list: list[torch.Tensor] = []
            presence_penalty_list: list[torch.Tensor] = []
            frequency_penalty_list: list[torch.Tensor] = []
            repetition_penalty_list: list[torch.Tensor] = []
            seed_list: list[torch.Tensor] = []
            num_logprobs_list: list[torch.Tensor] = []
            enable_log_probs_list: list[torch.Tensor] = []
            intermediate_prefill_masks: list[torch.Tensor] = []
            reset_batch = False

            active_inputs: list[TTModelInput] = [mi for mi in inputs if mi]
            if not active_inputs:
                raise ValueError("All inputs are None; nothing to concatenate")

            # Check if all ranks can sample on device.
            perform_device_sampling = all(
                mi.perform_device_sampling for mi in active_inputs
            )

            prefill_empty_slots = self._merge_dp_prefill_slots(inputs)

            # Determine max token width across slots.
            max_tok_width = 0
            for mi in active_inputs:
                assert mi.input_tokens.dim() == 2, "Input tokens must be 2D"
                max_tok_width = max(max_tok_width, mi.input_tokens.shape[1])
            assert max_tok_width > 0, "At least one input must have tokens"

            # Iterate over DP inputs and build segments for concatenation.
            for mi in inputs:
                # Skip None slots entirely. Decode path reconstructs full
                # inputs, so None should not occur there anymore.
                if mi is not None:
                    # Right-pad tokens and block tables to max widths
                    toks = mi.input_tokens
                    if not is_decode and toks.shape[1] < max_tok_width:
                        pad_w = max_tok_width - toks.shape[1]
                        toks = torch.cat(
                            [
                                toks,
                                torch.zeros((toks.shape[0], pad_w), dtype=toks.dtype),
                            ],
                            dim=1,
                        )
                    input_tokens_list.append(toks)
                    assert mi.prompt_lens is not None
                    prompt_lens_list.append(mi.prompt_lens)
                    block_tables_list.append(pad_block_tables(mi.block_tables))
                    assert (
                        len(mi.block_tables_per_group) == self._num_kv_cache_groups
                    ), (
                        f"DP merge: rank input has "
                        f"{len(mi.block_tables_per_group)} block_tables_per_group "
                        f"entries, expected {self._num_kv_cache_groups}"
                    )
                    for g, bt_g in enumerate(mi.block_tables_per_group):
                        block_tables_per_group_list[g].append(pad_block_tables(bt_g))
                    input_positions_list.append(mi.input_positions)

                    # Extract sampling parameter tensors from TTSamplingParams
                    sp = mi.tt_sampling_params
                    temperature_list.append(sp.temperature)
                    top_k_list.append(sp.top_k)
                    top_p_list.append(sp.top_p)
                    presence_penalty_list.append(sp.presence_penalty)
                    frequency_penalty_list.append(sp.frequency_penalty)
                    repetition_penalty_list.append(sp.repetition_penalty)
                    seed_list.append(sp.seed)
                    num_logprobs_list.append(sp.num_logprobs)
                    enable_log_probs_list.append(sp.enable_log_probs)
                    if mi.intermediate_prefill_mask is None:
                        intermediate_prefill_masks.append(
                            torch.zeros(toks.shape[0], dtype=torch.bool)
                        )
                    else:
                        assert mi.intermediate_prefill_mask.shape[0] == toks.shape[0]
                        intermediate_prefill_masks.append(mi.intermediate_prefill_mask)

                # We know it's not a list here before concatenation
                unpadded_batch_size: int = (
                    cast(int, mi.unpadded_batch_size) if mi else 0
                )
                batch_size_per_dp.append(unpadded_batch_size)
                grammar_bitmask_list.append(mi.grammar_bitmask[0] if mi else None)

                # Collect host-only sampling params per rank
                if mi is not None:
                    allowed_token_ids_mask_list.append(
                        mi.allowed_token_ids_mask_list[0]
                    )
                    bad_words_token_ids_list.append(mi.bad_words_token_ids_list[0])
                    logitsprocs_list.append(mi.logitsprocs_list[0])
                    # TODO: Move up from host-only since it's working on device now
                    max_num_logprobs.append(mi.max_num_logprobs[0])
                    generators_list.append(mi.generators_list[0])
                else:
                    allowed_token_ids_mask_list.append(None)
                    bad_words_token_ids_list.append({})
                    logitsprocs_list.append(None)
                    max_num_logprobs.append(None)
                    generators_list.append({})

            input_tokens = torch.cat(input_tokens_list, dim=0)
            input_positions = np.concatenate(input_positions_list, axis=0)
            prompt_lens = np.concatenate(prompt_lens_list, axis=0)
            block_tables = torch.cat(block_tables_list, dim=0)
            # Build the per-group merged view here so the prefill branch
            # exits with the same shape contract as the decode branch.
            block_tables_per_group = [
                torch.cat(per_rank, dim=0) for per_rank in block_tables_per_group_list
            ]

            # Concatenate sampling parameter tensors across DP ranks
            temperature = torch.cat(temperature_list, dim=0)
            top_k = torch.cat(top_k_list, dim=0)
            top_p = torch.cat(top_p_list, dim=0)
            presence_penalty = torch.cat(presence_penalty_list, dim=0)
            frequency_penalty = torch.cat(frequency_penalty_list, dim=0)
            repetition_penalty = torch.cat(repetition_penalty_list, dim=0)
            seed = torch.cat(seed_list, dim=0)
            num_logprobs = torch.cat(num_logprobs_list, dim=0)
            enable_log_probs = torch.cat(enable_log_probs_list, dim=0)
            intermediate_prefill_mask = torch.cat(intermediate_prefill_masks, dim=0)

        else:
            # Gathered-DP decode passes a dict (handled above) and prefill a
            # list of per-rank inputs (the elif). Single-process lane mode no
            # longer routes through this concat (the runner builds the merged
            # device input directly from ``TTLaneInputBatch``), so a non-dict
            # decode input here is unexpected.
            raise AssertionError(
                "concat_dp_model_inputs received a non-dict decode input; "
                "lane mode builds its decode input via TTLaneInputBatch."
            )

        tt_sampling_params = TTSamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            seed=seed,
            num_logprobs=num_logprobs,
            enable_log_probs=enable_log_probs,
        )

        if self.model_config.is_multimodal_model and not is_decode:
            # Gather multi-modal inputs from all DP ranks
            multi_modal_kwargs: dict[str, Any] = {
                "pixel_values": [],
                "image_grid_thw": [],
            }
            pixel_values = []
            image_grid_thw = []
            for mi in inputs:
                if mi is not None:
                    for pv in mi.multi_modal_kwargs["pixel_values"]:
                        pixel_values.append(pv)
                    for ig in mi.multi_modal_kwargs["image_grid_thw"]:
                        image_grid_thw.append(ig)
            multi_modal_kwargs["pixel_values"] = pixel_values
            multi_modal_kwargs["image_grid_thw"] = image_grid_thw
        else:
            multi_modal_kwargs = {}

        # Extract prompt and output tokens for decode with sampling penalties
        prompt_tokens = None
        output_tokens = None
        sampling_tokens_inputs = (
            inputs.get("sampling_tokens_inputs") if is_decode else None
        )
        if sampling_tokens_inputs:
            # Find max shapes across all ranks
            max_prompt_len = 0
            max_output_len = 0
            for rank_tokens_dict in sampling_tokens_inputs:
                if rank_tokens_dict is not None:
                    rank_prompt_tokens = rank_tokens_dict.get("prompt_tokens")
                    rank_output_tokens = rank_tokens_dict.get("output_tokens")
                    if rank_prompt_tokens is not None:
                        assert rank_output_tokens is not None
                        max_prompt_len = max(
                            max_prompt_len, rank_prompt_tokens.shape[1]
                        )
                        max_output_len = max(
                            max_output_len, rank_output_tokens.shape[1]
                        )

            # Create tensors with shape (per-rank batch * DP_size, max_len)
            max_num_reqs = self.tt_per_lane_max_num_seqs
            total_batch_size = max_num_reqs * len(sampling_tokens_inputs)

            # Create prompt and output tokens tensors
            prompt_tokens = torch.full(
                (total_batch_size, max_prompt_len), -1, dtype=torch.int32
            )
            output_tokens = torch.full(
                (total_batch_size, max_output_len), -1, dtype=torch.int32
            )
            for rank_idx, rank_tokens_dict in enumerate(sampling_tokens_inputs):
                if rank_tokens_dict is not None:
                    start_idx = rank_idx * max_num_reqs
                    rank_prompt_tokens = rank_tokens_dict.get("prompt_tokens")
                    rank_output_tokens = rank_tokens_dict.get("output_tokens")
                    if rank_prompt_tokens is not None:
                        assert rank_output_tokens is not None
                        end_idx = start_idx + rank_prompt_tokens.shape[0]
                        prompt_padded_len = rank_prompt_tokens.shape[1]
                        output_padded_len = rank_output_tokens.shape[1]
                        prompt_tokens[start_idx:end_idx, :prompt_padded_len] = (
                            rank_prompt_tokens
                        )
                        output_tokens[start_idx:end_idx, :output_padded_len] = (
                            rank_output_tokens
                        )

        merged = TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            block_tables=block_tables,
            # ``block_tables_per_group`` carries each kv_cache_group's
            # block table merged across DP ranks (decode unpacks them
            # from ``int_inputs``; prefill concatenates rank-by-rank).
            # ``_block_tables_per_layer`` then expands the per-group view
            # into the per-layer list the hybrid bridge consumes.
            block_tables_per_group=block_tables_per_group,
            block_tables_per_layer=self._block_tables_per_layer(block_tables_per_group),
            unpadded_batch_size=batch_size_per_dp,
            tt_sampling_params=tt_sampling_params,
            multi_modal_kwargs=multi_modal_kwargs,
            perform_device_sampling=perform_device_sampling,
            grammar_bitmask=grammar_bitmask_list,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reset_batch=reset_batch,
            slot_remap=slot_remap,
            # Host-only sampling params (per-rank lists)
            allowed_token_ids_mask_list=allowed_token_ids_mask_list,
            bad_words_token_ids_list=bad_words_token_ids_list,
            max_num_logprobs=max_num_logprobs,
            logitsprocs_list=logitsprocs_list,
            generators_list=generators_list,
            prefill_empty_slots=prefill_empty_slots,
            intermediate_prefill_mask=intermediate_prefill_mask,
        )
        return merged

    # ------------------------------------------------------------------
    # Single-process lane-DP step orchestration
    # ------------------------------------------------------------------
    #
    # All lane-specific input/output shaping lives in ``TTLaneInputBatch``
    # (``apply_step_plan`` / ``build_model_input`` / ``extract_output``) and the
    # merge/redistribute in ``TTLaneCoordinator``; this orchestration only wires
    # the device submission, async decode, and deferred state application -- the
    # genuinely runner/model-owned parts.

    @torch.no_grad()
    def _execute_lane_step(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | None:
        """Run the device forward for one merged lane-DP step.

        Invoked by ``execute_model`` when the lane scheduler attached a step
        plan. Returns ``None`` after enqueuing the lane sampler (the grammar
        bitmask is applied later in ``sample_tokens``); returns
        ``EMPTY_MODEL_RUNNER_OUTPUT`` only when there is nothing scheduled.
        """
        plan = get_tt_step_plan(scheduler_output)
        assert plan is not None, "lane step requires a scheduler-attached plan"

        # Lane-DP lays requests out at sparse stable slots. Request-specific
        # RoPE (mrope/vision models) instead assumes front-packed request rows,
        # so the two are incompatible until the RoPE delta mapping is made
        # slot-aware. No lane-DP model needs it.
        if self.request_specific_rope:
            raise NotImplementedError(
                "lane-DP does not support request-specific RoPE "
                "(mrope/vision models) yet"
            )

        lane_batch = self.lane_batch
        self.async_decode.apply_ready_completed_decode_steps()
        steady_decode_candidate = (
            self.async_decode.can_attempt_steady_dp_decode_from_scheduler(
                scheduler_output, None
            )
        )
        if (
            scheduler_output.scheduled_cached_reqs.resumed_req_ids
            or self.async_decode.must_drain_pending_async_steps(steady_decode_candidate)
        ):
            self.async_decode.wait_for_all_pending_async_steps()

        layout_changed = lane_batch.apply_step_plan(
            scheduler_output, plan, self.requests, self.encoder_cache
        )
        if layout_changed:
            self._decode_layout_changed_since_last_decode = True
            # ``_decode_layout_changed_since_last_decode = True`` implies
            # ``reset_batch=True``: the model will reload inputs. This step is not
            # steady-decode eligible, so drain pending decodes to ensure updated
            # host inputs.
            self.async_decode.wait_for_all_pending_async_steps()

        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT
        scheduled_rows = list(plan.scheduled_rows)
        if not scheduled_rows:
            # Tokens were scheduled but no slot mapped to them; the engine still
            # calls sample_tokens, so enqueue an empty result to keep the
            # forward/sample calls balanced.
            self._pending_samples.append(lambda _grammar: EMPTY_MODEL_RUNNER_OUTPUT)
            return None

        # Grammar is applied at sample time, so the forward builds without it.
        model_input = lane_batch.build_model_input(self, scheduler_output, None, plan)
        if plan.is_decode:
            # ``slot_grammar_bitmask`` reorders against the full decode capacity.
            lane_total = plan.capacity
            if self.non_dp_async_scheduling:
                req_ids = [self.lane_batch.req_ids[row] for row in scheduled_rows]
                context = self.async_decode.capture_submitted_step_context(req_ids)
                wrapper = self.async_decode.submit_async_lane_decode(
                    model_input, context, scheduled_rows
                )
                self._pending_samples.append(
                    partial(
                        self._finish_async_decode,
                        wrapper=wrapper,
                        model_input=model_input,
                        lane_total=lane_total,
                    )
                )
                return None
            submission = self.async_decode.submit_decode(
                model_input, read_from_device=True, async_read=False
            )
            finalized = self.async_decode.finalize_decode(submission)
            assert finalized is not None
            tt_out = finalized.tt_out
            tt_log_probs = finalized.tt_log_probs
            is_decode = True
        else:
            # Prefill reorders against the per-lane request capacity.
            lane_total = lane_batch.max_num_reqs
            tt_out = self.submit_prefill(model_input, model_input.unpadded_batch_size)
            tt_log_probs = None
            assert isinstance(
                model_input.tt_sampling_params.enable_log_probs, torch.Tensor
            )
            if (
                model_input.perform_device_sampling
                and model_input.tt_sampling_params.enable_log_probs.any()
            ):
                assert isinstance(tt_out, tuple) and len(tt_out) == 2
                tt_out, tt_log_probs = tt_out
            elif isinstance(tt_out, tuple):
                tt_out, _ = tt_out
            is_decode = False

        # Forward done; defer the lane sampling tail to ``sample_tokens``. The
        # step runs synchronously, so the lane batch layout is stable between
        # this enqueue and the deferred sample.
        self._pending_samples.append(
            partial(
                self._finish_lane_sync,
                tt_out=tt_out,
                tt_log_probs=tt_log_probs,
                model_input=model_input,
                scheduled_rows=scheduled_rows,
                is_decode=is_decode,
                lane_total=lane_total,
            )
        )
        return None

    def _finish_lane_sync(
        self,
        grammar_output: GrammarOutput | None,
        *,
        tt_out: Any,
        tt_log_probs: Any,
        model_input: TTModelInput,
        scheduled_rows: list[int],
        is_decode: bool,
        lane_total: int,
    ) -> ModelRunnerOutput:
        """Sample and build the runner output for a deferred lane forward."""
        model_input = self._apply_grammar_to_input(
            model_input, grammar_output, lane_total=lane_total
        )
        sampled, logprobs = self.lane_batch.extract_output(
            self, tt_out, tt_log_probs, model_input, scheduled_rows, is_decode=is_decode
        )
        # ``scheduled_rows`` are persistent slots; their req_ids in row order are
        # the canonical merged output order.
        req_ids = [self.lane_batch.req_ids[row] for row in scheduled_rows]

        if not is_decode and model_input.prompt_lens is not None:
            lane_batch = self.lane_batch
            prompt_lens = np.asarray(model_input.prompt_lens)
            num_tokens = np.array(
                [lane_batch.num_tokens[row] for row in scheduled_rows],
                dtype=np.int64,
            )
            intermediate_mask = prompt_lens < num_tokens

            if intermediate_mask.any():
                return self._build_chunked_prefill_output(
                    req_ids=req_ids,
                    sampled_token_ids=sampled,
                    logprobs=logprobs,
                    intermediate_mask=intermediate_mask,
                )

        return self.apply_and_build_runner_output(sampled, logprobs, req_ids=req_ids)

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | None:
        """Run the device forward for one non-DP or lane-DP step.

        Returns ``None`` after enqueuing a pending sampler; the engine computes
        the grammar bitmask while the forward runs and then calls
        ``sample_tokens``. Returns ``EMPTY_MODEL_RUNNER_OUTPUT`` only when
        nothing was scheduled (no sampler is enqueued). Gathered multi-process
        DP does not use this entrypoint; the worker calls its dedicated facade.
        """
        # Single-process lane-DP: the lane scheduler attaches a per-step plan to
        # the scheduler output. When present, the step runs over the merged lane
        # batch (``TTLaneInputBatch`` owns all the lane-specific input/output
        # shaping); the generic path below stays lane-agnostic.
        if get_tt_step_plan(scheduler_output) is not None:
            return self._execute_lane_step(scheduler_output)

        # Apply any decode steps that have already completed on the async
        # thread. In steady decode mode we intentionally allow one step of
        # lag between host application and device submission, but we never let
        # completed work pile up unbounded.
        self.async_decode.apply_ready_completed_decode_steps()
        steady_decode_candidate = (
            self.async_decode.can_attempt_steady_decode_from_scheduler(
                scheduler_output, None
            )
        )
        if self.async_decode.must_drain_pending_async_steps(steady_decode_candidate):
            self.async_decode.wait_for_all_pending_async_steps()

        # Grammar is applied at sample time, so the forward builds without it.
        model_input = self.build_model_input(scheduler_output, None)
        if model_input is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        is_decode = model_input.prompt_lens is None
        if self.non_dp_async_scheduling and is_decode:
            steady_decode_fast_path = self.async_decode.can_use_steady_decode_fast_path(
                model_input
            )
            wrapper = self.async_decode.submit_async_non_dp_decode(
                model_input,
                steady_decode_fast_path=steady_decode_fast_path,
            )
            self._pending_samples.append(
                partial(
                    self._finish_async_decode,
                    wrapper=wrapper,
                    model_input=model_input,
                    lane_total=None,
                )
            )
            return None

        # Synchronous path (prefill, or decode without async scheduling): run
        # the forward now and defer sampling to ``sample_tokens``.
        fwd = self._forward_with_model_input(model_input)
        self._pending_samples.append(partial(self._finish_nondp_sync, fwd=fwd))
        return None

    def _reorder_grammar_bitmask(
        self,
        grammar_output: GrammarOutput | None,
        model_input: TTModelInput,
        *,
        lane_total: int | None,
    ) -> torch.Tensor | None:
        """Reorder a scheduler grammar bitmask into TT batch layout, or ``None``.

        Runs at sample time. ``sample_tokens`` executes before the next
        schedule, so the live batch/lane layout still matches the forward this
        bitmask belongs to. ``lane_total`` selects the lane reorder (full slot
        capacity) over the non-DP front-packed reorder.
        """
        if grammar_output is None or grammar_output.grammar_bitmask is None:
            return None
        if lane_total is not None:
            return self.lane_batch.slot_grammar_bitmask(grammar_output, lane_total)
        bitmask = torch.from_numpy(grammar_output.grammar_bitmask)
        return reorder_grammar_bitmask_for_tt_batch(
            bitmask=bitmask,
            structured_output_request_ids=grammar_output.structured_output_request_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            req_indices=list(range(self.input_batch.num_reqs)),
            batch_length=model_input.input_tokens.shape[0],
        )

    def _apply_grammar_to_input(
        self,
        model_input: TTModelInput,
        grammar_output: GrammarOutput | None,
        *,
        lane_total: int | None,
    ) -> TTModelInput:
        """Return ``model_input`` with the sample-time grammar bitmask attached."""
        bitmask = self._reorder_grammar_bitmask(
            grammar_output, model_input, lane_total=lane_total
        )
        if bitmask is None:
            return model_input
        return replace(model_input, grammar_bitmask=[bitmask])

    def _finish_async_decode(
        self,
        grammar_output: GrammarOutput | None,
        *,
        wrapper: AsyncTTModelRunnerOutput,
        model_input: TTModelInput,
        lane_total: int | None,
    ) -> AsyncTTModelRunnerOutput:
        """Attach the sample-time grammar bitmask to a deferred async decode.

        The reorder happens here on the engine thread (layout still matches the
        forward); the wrapper applies the bitmask when its read completes.
        """
        bitmask = self._reorder_grammar_bitmask(
            grammar_output, model_input, lane_total=lane_total
        )
        if bitmask is not None:
            wrapper.set_grammar_bitmask(bitmask)
        return wrapper

    def _finish_nondp_sync(
        self, grammar_output: GrammarOutput | None, *, fwd: _SyncForward | None
    ) -> ModelRunnerOutput:
        """Sample and build the runner output for a deferred non-DP forward."""
        if fwd is None:
            return self.apply_and_build_runner_output(
                torch.tensor([], dtype=torch.int32), None
            )
        fwd = replace(
            fwd,
            model_input=self._apply_grammar_to_input(
                fwd.model_input, grammar_output, lane_total=None
            ),
        )
        sampled_token_ids_per_dp, logprobs_per_dp = self._sample_sync_forward(fwd)
        sampled_token_ids = sampled_token_ids_per_dp[0]
        logprobs_tensors = logprobs_per_dp[0] if logprobs_per_dp else None
        logprobs = logprobs_tensors.tolists() if logprobs_tensors else None

        if not fwd.is_decode and fwd.model_input.prompt_lens is not None:
            num_reqs = self.input_batch.num_reqs
            prompt_lens = np.asarray(fwd.model_input.prompt_lens)
            num_tokens = self.input_batch.num_tokens[:num_reqs]
            intermediate_mask = prompt_lens < num_tokens

            if intermediate_mask.any():
                output_req_ids = list(self.input_batch.req_ids[:num_reqs])
                return self._build_chunked_prefill_output(
                    req_ids=output_req_ids,
                    sampled_token_ids=sampled_token_ids,
                    logprobs=logprobs,
                    intermediate_mask=intermediate_mask,
                )

        return self.apply_and_build_runner_output(sampled_token_ids, logprobs)

    def _build_chunked_prefill_output(
        self,
        req_ids: list[str],
        sampled_token_ids: torch.Tensor,
        logprobs: list | None,
        intermediate_mask: np.ndarray,
        req_id_to_index: dict[str, int] | None = None,
    ) -> ModelRunnerOutput:
        """Builds output with ``[]`` for intermediate, ``[tokens]`` for final chunks."""
        final_idx_np = np.where(~intermediate_mask)[0]
        if final_idx_np.shape[0] > 0:
            final_idx_tensor = torch.from_numpy(final_idx_np.astype(np.int64))
            final_tokens = sampled_token_ids[final_idx_tensor]
            final_req_ids = [req_ids[int(i)] for i in final_idx_np]
            self._apply_sampled_tokens_to_state(final_tokens, req_ids=final_req_ids)

        num_reqs = len(req_ids)
        sampled_token_ids_np = sampled_token_ids.view(num_reqs).numpy()
        if sampled_token_ids_np.dtype != np.int32:
            sampled_token_ids_np = sampled_token_ids_np.astype(np.int32, copy=False)
        sampled_token_id_lists = [
            [] if intermediate_mask[i] else [int(sampled_token_ids_np[i])]
            for i in range(num_reqs)
        ]

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=(
                dict(req_id_to_index)
                if req_id_to_index is not None
                else {r_id: idx for idx, r_id in enumerate(req_ids)}
            ),
            sampled_token_ids=sampled_token_id_lists,
            logprobs=logprobs,
            prompt_logprobs_dict=dict.fromkeys(req_ids, None),
            pooler_output=[],
        )

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncTTModelRunnerOutput | None:
        """Sample the forward deferred by a preceding ``execute_model``.

        Pops the oldest pending forward (FIFO, matching the engine's
        oldest-first finalization), applies the now-ready grammar bitmask, and
        produces its output (or an async wrapper for overlapped decode). The
        engine calls this exactly once per ``execute_model`` that returned
        ``None``.

        If the deque is empty, ``execute_model`` must have raised an exception
        that was captured by the executor; return ``None`` so the engine can
        surface the original error from the execute future.
        """
        if not self._pending_samples:
            return None

        finish = self._pending_samples.popleft()
        return finish(grammar_output)

    def pack_dp_results(
        self,
        sampled_token_ids_per_dp: list[torch.Tensor],
        logprobs_per_dp: list,
    ) -> tuple[torch.Tensor, list]:
        """Pack per-DP results into the gathered-DP wire format.

        Converts per-DP sampled tokens and logprobs into the stacked tensor/list
        payload consumed by gathered-DP finalization.
        """
        logprobs_lists_per_dp = [
            lp.tolists() if lp is not None else None for lp in logprobs_per_dp
        ]
        world = self.tt_data_parallel_size
        B = self.tt_per_lane_max_num_seqs
        for dp_rank in range(world):
            token_ids = sampled_token_ids_per_dp[dp_rank].to(torch.int32)
            if token_ids.numel() == 0:
                token_ids = torch.zeros((B, 1), dtype=torch.int32)
            else:
                assert token_ids.dim() == 2 and token_ids.shape[1] == 1, (
                    "Currently only supporting 1 output token per request"
                )
                pad_rows = B - token_ids.shape[0]
                if pad_rows > 0:
                    token_ids = torch.cat(
                        [
                            token_ids,
                            torch.zeros(
                                (pad_rows, token_ids.shape[1]),
                                dtype=torch.int32,
                            ),
                        ],
                        dim=0,
                    )
            sampled_token_ids_per_dp[dp_rank] = token_ids
        return torch.stack(sampled_token_ids_per_dp), logprobs_lists_per_dp

    def check_perform_device_sampling(
        self, is_decode: bool, has_structured_outputs: bool
    ) -> bool:
        want_device_sampling = self.sample_on_device_mode == "all" or (
            self.sample_on_device_mode == "decode_only" and is_decode
        )
        if not want_device_sampling:
            return False

        # Calculate number of devices per DP rank
        num_devices = self.num_devices // self.tt_data_parallel_size

        # Always host-only sampling params: min_p, bad_words, logit_bias,
        # allowed_token_ids, min_tokens require host sampling.
        input_batch = self.input_batch
        has_always_host_only_sampling_params = (
            not input_batch.no_allowed_token_ids  # allowed_token_ids set
            or input_batch.sampling.bad_words_token_ids  # bad_words set
            or input_batch.sampling.has_active_logitsprocs()  # min_p, logit_bias,
            # min_tokens
            or bool(self.model_config.logits_processors)  # custom logitsprocs
        )
        if has_always_host_only_sampling_params:
            return False

        # Structured outputs are not supported on device yet
        # https://github.com/tenstorrent/vllm/issues/277
        if has_structured_outputs:
            return False

        # Logprobs on device require multi-device setups (num_devices in {8,32}).
        # On single device, all logprobs require host sampling.
        # https://github.com/tenstorrent/tt-metal/issues/34077
        #
        # Top-K logprobs (max_lp > 0) are only supported on device by models
        # that set use_topk_logprobs=True (e.g. gpt-oss-120b), which return
        # top-32 logprobs as a (logprobs, indices) tuple. Other models only
        # return the sampled token's logprob, so max_lp > 0 falls back to
        # host sampling to compute full top-N from logits.
        max_lp = input_batch.max_num_logprobs
        if max_lp is not None:
            if num_devices not in (8, 32):
                return False
            if max_lp > 0 and not self.supports_topk_logprobs:
                return False

        return True

    def submit_prefill(
        self,
        model_input: TTModelInput,
        batch_size_per_dp: list[int],
    ) -> Any:
        """Submit a prefill step and return the raw TT output.

        Launches TT prefill and returns the raw TT output used by the
        synchronous extraction path.
        """
        kwargs = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": self.kv_caches,
            "enable_trace": self.trace_mode in ["all"],
            "prompt_lens": model_input.prompt_lens,
            "start_pos": model_input.input_positions,
        }
        # Hybrid attention models route per-layer block tables; the
        # runner already expanded ``block_tables_per_group`` into a
        # per-layer list at submission time when the kv_cache_config has
        # multiple groups. Legacy/uniform models leave it as ``None``
        # and never see the kwarg.
        if model_input.block_tables_per_layer is not None:
            kwargs["page_tables_per_layer"] = model_input.block_tables_per_layer
        kwargs.update(model_input.multi_modal_kwargs)
        if model_input.perform_device_sampling:
            sampling_params = model_input.tt_sampling_params
            sampling_param_dict = {
                field.name: (
                    getattr(sampling_params, field.name).tolist()
                    if getattr(sampling_params, field.name) is not None
                    else None
                )
                for field in fields(sampling_params)
            }
            sampling_param_dict["seed"] = [
                None if s == SEED_NONE_SENTINEL else s
                for s in sampling_param_dict["seed"]
            ]
            kwargs["sampling_params"] = TTSamplingParams(**sampling_param_dict)
            if model_input.output_token_counts is not None and getattr(
                self.model, "needs_sampling_output_counts", False
            ):
                kwargs["output_token_counts"] = model_input.output_token_counts
        # Send the slots whenever the build supplied them. Multi-rank-only before,
        # so a DP=1 prefill sent nothing and the model's ``range(N)`` default
        # clobbered live state.
        empty_slots = model_input.prefill_empty_slots
        if empty_slots is None and len(batch_size_per_dp) > 1:
            # TODO: the model should only require DP ranks, but passing
            # "global" user ids instead for backwards compatibility.
            stride = self.tt_per_lane_max_num_seqs
            empty_slots = []
            for dp_rank, sz in enumerate(batch_size_per_dp):
                for i in range(int(sz)):
                    empty_slots.append(dp_rank * stride + i)
        if empty_slots is not None:
            kwargs["empty_slots"] = list(empty_slots)

        if self.request_specific_rope:
            tt_out, rope_deltas = self.model.prefill_forward(**kwargs)
            # Store rope_deltas for each prefilled request
            for i, req_id in enumerate(self.input_batch.req_ids):
                self.requests[req_id].mrope_position_delta = rope_deltas[i].item()
            return tt_out
        return self.model.prefill_forward(**kwargs)

    def _forward_with_model_input(
        self,
        model_input: TTModelInput,
    ) -> _SyncForward | None:
        """Run the synchronous device forward for a prebuilt model input.

        Submits prefill or decode and finalizes the device read, but does not
        sample. Returns the materialized forward state, or ``None`` when the
        batch is empty (no rows to sample).
        """
        is_decode = model_input.prompt_lens is None

        batch_size_per_dp = model_input.unpadded_batch_size
        if not isinstance(batch_size_per_dp, list):
            batch_size_per_dp = [batch_size_per_dp]
        if not any(bs > 0 for bs in batch_size_per_dp):
            return None

        sampling_params = model_input.tt_sampling_params
        perform_device_sampling = model_input.perform_device_sampling
        tt_log_probs = None

        # Execute model
        if not is_decode:
            tt_out = self.submit_prefill(model_input, batch_size_per_dp)
            # Prefill returns the raw model output: an optional
            # ``(tokens/logits, logprobs)`` tuple when device sampling ran.
            # Unpack it here. The decode branch below is already unpacked by
            # ``finalize_decode``, so re-running this would double-unpack and
            # assert on an already-extracted tensor.
            assert isinstance(sampling_params.enable_log_probs, torch.Tensor)
            if perform_device_sampling and sampling_params.enable_log_probs.any():
                assert isinstance(tt_out, tuple) and len(tt_out) == 2
                tt_out, tt_log_probs = tt_out
            elif isinstance(tt_out, tuple):
                tt_out, _ = tt_out
        else:
            submission = self.async_decode.submit_decode(
                model_input, read_from_device=False, async_read=False
            )
            finalized = self.async_decode.finalize_decode(submission)
            assert finalized is not None
            # ``finalize_decode`` already unpacked ``(tt_out, tt_log_probs)``.
            tt_out = finalized.tt_out
            tt_log_probs = finalized.tt_log_probs
            batch_size_per_dp = submission.batch_size_per_dp
            sampling_params = submission.sampling_params
            perform_device_sampling = submission.perform_device_sampling

        return _SyncForward(
            tt_out=tt_out,
            tt_log_probs=tt_log_probs,
            sampling_params=sampling_params,
            model_input=model_input,
            batch_size_per_dp=batch_size_per_dp,
            perform_device_sampling=perform_device_sampling,
            is_decode=is_decode,
        )

    def _sample_sync_forward(
        self, fwd: _SyncForward
    ) -> tuple[list[torch.Tensor], list[LogprobsTensors | None]]:
        """Sample a forward produced by ``_forward_with_model_input``."""
        return self._get_output_tokens(
            tt_out=fwd.tt_out,
            tt_log_probs=fwd.tt_log_probs,
            sampling_params=fwd.sampling_params,
            model_input=fwd.model_input,
            batch_size_per_dp=fwd.batch_size_per_dp,
            perform_device_sampling=fwd.perform_device_sampling,
            is_decode=fwd.is_decode,
        )

    def execute_sync_with_model_input(
        self,
        model_input: TTModelInput,
    ) -> tuple[list[torch.Tensor], list[LogprobsTensors | None]]:
        """Run a fully synchronous TT execution for a prebuilt model input.

        Executes a prebuilt TT input to completion, including prefill or decode
        submission, decode finalization when needed, and per-DP token/logprob
        extraction.

        Returns:
            Tuple of (sampled_token_ids_per_dp, logprobs_per_dp).
            Each element in logprobs_per_dp is None if logprobs were not
            requested for that DP rank.
        """
        fwd = self._forward_with_model_input(model_input)
        if fwd is None:
            num_dp = (
                len(model_input.unpadded_batch_size)
                if isinstance(model_input.unpadded_batch_size, list)
                else 1
            )
            return ([torch.tensor([], dtype=torch.int32)] * num_dp, [None] * num_dp)
        return self._sample_sync_forward(fwd)

    def prepare_dp_model_input(
        self,
        scheduler_output: SchedulerOutput | None,
        grammar_output: GrammarOutput | None,
    ) -> tuple[
        TTModelInput | None,
        int,
        int,
        int,
        int,
        int,
        int,
        torch.Tensor | None,
        list[str],
        dict[str, int],
    ]:
        """Build the per-rank DP payload consumed by gather orchestration.

        Returns the local TT model input plus the per-rank metadata needed by
        gathered-DP negotiation and input gathering.
        """
        model_input = None
        has_penalties = 0
        reset_batch = 0
        can_sample_device = 1
        needs_logprobs = 0
        intermediate_prefill_mask = None
        req_ids: list[str] = []
        req_id_to_index: dict[str, int] = {}
        if scheduler_output is not None:
            model_input = self.build_model_input(scheduler_output, grammar_output)
            if model_input is not None:
                has_penalties = int(not self.input_batch.no_penalties)
                reset_batch = int(model_input.reset_batch)
                can_sample_device = int(model_input.perform_device_sampling)
                max_num_logprobs = model_input.max_num_logprobs[0]
                # max_num_logprobs=0 still requests the sampled token's logprob.
                needs_logprobs = int(max_num_logprobs is not None)
                intermediate_prefill_mask = model_input.intermediate_prefill_mask
                num_reqs = self.input_batch.num_reqs
                req_ids = list(self.input_batch.req_ids[:num_reqs])
                req_id_to_index = dict(self.input_batch.req_id_to_index)
        max_blocks = model_input.block_tables.shape[1] if model_input else 0
        has_structured_input = (
            int(model_input.grammar_bitmask[0] is not None) if model_input else 0
        )
        return (
            model_input,
            max_blocks,
            has_structured_input,
            has_penalties,
            reset_batch,
            can_sample_device,
            needs_logprobs,
            intermediate_prefill_mask,
            req_ids,
            req_id_to_index,
        )

    def submit_dp_execution(
        self,
        inputs: list[TTModelInput | None] | dict[str, Any],
        is_decode: bool,
        max_blocks_decode_batch: int | None,
        any_structured_inputs: bool,
        non_block: bool = False,
        allow_decode_overlap: bool = True,
    ) -> Any:
        """Execute one merged DP batch and return the DP-facing result shape.

        Merges gathered DP inputs, selects sync or async decode execution, and
        returns the packed DP-facing result expected by the worker facade.
        """
        merged = self.concat_dp_model_inputs(
            inputs,
            is_decode,
            max_blocks_decode_batch,
            any_structured_inputs,
        )

        if non_block and is_decode:
            return self.async_decode.submit_async_dp_decode(
                merged, allow_decode_overlap=allow_decode_overlap
            )

        sampled_token_ids_per_dp, logprobs_per_dp = self.execute_sync_with_model_input(
            merged
        )
        return self.pack_dp_results(sampled_token_ids_per_dp, logprobs_per_dp)

    def apply_dp_execution_result(
        self,
        sampled_token_ids: torch.Tensor,
        logprobs_lists: LogprobsLists | None = None,
        req_ids: list[str] | None = None,
        req_id_to_index: dict[str, int] | None = None,
        intermediate_prefill_mask: torch.Tensor | None = None,
    ) -> ModelRunnerOutput:
        """Apply the local DP rank result to runner state and build output.

        Converts the local gathered-DP result into the same state update and
        `ModelRunnerOutput` used by non-DP execution.
        """
        num_reqs = len(req_ids) if req_ids is not None else self.input_batch.num_reqs
        sampled_token_ids = sampled_token_ids[:num_reqs]
        if intermediate_prefill_mask is not None:
            intermediate_mask = intermediate_prefill_mask[:num_reqs]
            if intermediate_mask.any():
                output_req_ids = (
                    list(req_ids)
                    if req_ids is not None
                    else list(self.input_batch.req_ids[:num_reqs])
                )
                return self._build_chunked_prefill_output(
                    req_ids=output_req_ids,
                    sampled_token_ids=sampled_token_ids,
                    logprobs=logprobs_lists,
                    intermediate_mask=intermediate_mask.cpu().numpy(),
                    req_id_to_index=req_id_to_index,
                )
        return self.apply_and_build_runner_output(
            sampled_token_ids,
            logprobs_lists,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )

    def _get_output_tokens(
        self,
        tt_out: torch.Tensor,
        tt_log_probs: torch.Tensor | None,
        sampling_params: TTSamplingParams,
        model_input: TTModelInput,
        batch_size_per_dp: list[int],
        perform_device_sampling: bool,
        is_decode: bool,
    ) -> tuple[list[torch.Tensor], list[LogprobsTensors | None]]:
        """Return sampled tokens per DP rank using concatenated model
        outputs, plus optional logprobs per DP rank.

        If perform_device_sampling is True, tokens are already sampled on
        device. Otherwise, sample on host using host_sampler.

        Args:
            tt_out: Model output (logits or tokens depending on sampling mode)
            tt_log_probs: Optional logprobs from device sampling

        Returns:
            Tuple of (sampled_token_ids_per_dp, logprobs_per_dp).
            Each element in logprobs_per_dp is None if logprobs were not
            requested for that DP rank.
        """
        sampled_token_ids_per_dp: list[torch.Tensor] = []
        logprobs_per_dp: list[LogprobsTensors | None] = []

        start = 0
        for dp_rank, sz in enumerate(batch_size_per_dp):
            if sz <= 0:
                sampled_token_ids_per_dp.append(torch.tensor([], dtype=torch.int32))
                logprobs_per_dp.append(None)
                if is_decode:
                    # Fixed stride segments per DP rank for decode
                    start += self.tt_per_lane_max_num_seqs
                continue

            # Gathered-DP / non-DP pack active requests at the front of each
            # rank's segment, so this rank's rows are the contiguous
            # ``[start, start + sz)`` range.
            rows = torch.arange(start, start + sz, dtype=torch.long)

            def _take(tensor: torch.Tensor, _rows: torch.Tensor = rows) -> torch.Tensor:
                return tensor[_rows]

            if (
                not is_decode
                and model_input.intermediate_prefill_mask is not None
                and bool(model_input.intermediate_prefill_mask[rows].all())
            ):
                next_token_ids = torch.zeros(sz, dtype=torch.int32)
                logprobs_per_dp.append(None)
            elif not perform_device_sampling:
                logits = tt_out[rows, -1, :]

                grammar_bitmask = model_input.grammar_bitmask[dp_rank]

                if grammar_bitmask is not None:
                    # match shape of logits, which are now unpadded on batch dim
                    grammar_bitmask = grammar_bitmask[:sz, :]
                    self.apply_grammar_bitmask(logits, grammar_bitmask)

                # Extract sampling params for this DP rank from concatenated
                # tensors.
                assert isinstance(sampling_params.temperature, torch.Tensor)
                assert isinstance(sampling_params.top_k, torch.Tensor)
                assert isinstance(sampling_params.top_p, torch.Tensor)
                assert isinstance(sampling_params.presence_penalty, torch.Tensor)
                assert isinstance(sampling_params.frequency_penalty, torch.Tensor)
                assert isinstance(sampling_params.repetition_penalty, torch.Tensor)
                assert isinstance(sampling_params.seed, torch.Tensor)
                temperature = _take(sampling_params.temperature)
                top_k = _take(sampling_params.top_k)
                top_p = _take(sampling_params.top_p)
                presence_penalty = _take(sampling_params.presence_penalty)
                frequency_penalty = _take(sampling_params.frequency_penalty)
                repetition_penalty = _take(sampling_params.repetition_penalty)

                # Determine if all greedy (temperature == 0.0) or all random
                all_greedy = (temperature == 0.0).all().item()
                all_random = (temperature != 0.0).all().item()

                generators = model_input.generators_list[dp_rank]

                # Determine if penalties are needed
                no_penalties = (
                    (presence_penalty == 0.0).all().item()
                    and (frequency_penalty == 0.0).all().item()
                    and (repetition_penalty == 1.0).all().item()
                )

                # Output history as list[list[int]] (filter TT -1 padding).
                output_token_ids: list[list[int]] = []
                if is_decode and model_input.output_tokens is not None:
                    output_tokens = _take(model_input.output_tokens)
                    for i in range(sz):
                        output_tokens_i = output_tokens[i].tolist()
                        output_token_ids.append(
                            [tok for tok in output_tokens_i if tok != -1]
                        )
                else:
                    output_token_ids = [[] for _ in range(sz)]

                # Prompt tokens for penalties: must be int64 and padded with a
                # valid index (vocab_size), not TT's -1 sentinel.
                prompt_token_ids: torch.Tensor | None = None
                if not no_penalties:
                    if is_decode and model_input.prompt_tokens is not None:
                        prompt_token_ids = _take(model_input.prompt_tokens).to(
                            torch.int64
                        )
                        prompt_token_ids = prompt_token_ids.masked_fill(
                            prompt_token_ids == -1, self.vocab_size
                        )
                    elif not is_decode:
                        prompt_token_ids = model_input.input_tokens[
                            start : start + sz
                        ].to(torch.int64)
                        assert model_input.prompt_lens is not None
                        prompt_lens_t = torch.as_tensor(
                            model_input.prompt_lens[start : start + sz],
                            dtype=torch.int64,
                        )
                        positions = torch.arange(
                            prompt_token_ids.shape[1],
                        ).unsqueeze(0)
                        pad_mask = positions >= prompt_lens_t.unsqueeze(1)
                        prompt_token_ids = prompt_token_ids.masked_fill(
                            pad_mask, self.vocab_size
                        )

                # Get host-only sampling params from model_input
                # (per-rank lists).
                # These are populated for both DP and non-DP cases.
                rank_max_num_logprobs = model_input.max_num_logprobs[dp_rank]
                allowed_token_ids_mask = model_input.allowed_token_ids_mask_list[  # noqa: E501
                    dp_rank
                ]
                if allowed_token_ids_mask is not None:
                    # Slice to actual batch size for this rank
                    allowed_token_ids_mask = allowed_token_ids_mask[:sz]

                bad_words_token_ids = model_input.bad_words_token_ids_list[dp_rank]

                logitsprocs = model_input.logitsprocs_list[dp_rank]
                if logitsprocs is None:
                    logitsprocs = LogitsProcessors()

                # Create SamplingMetadata for this DP rank
                sampling_metadata = SamplingMetadata(
                    temperature=temperature if not all_greedy else None,
                    all_greedy=all_greedy,
                    all_random=all_random,
                    top_p=top_p,
                    top_k=top_k,
                    generators=generators,
                    max_num_logprobs=rank_max_num_logprobs,
                    no_penalties=no_penalties,
                    prompt_token_ids=prompt_token_ids,
                    frequency_penalties=frequency_penalty,
                    presence_penalties=presence_penalty,
                    repetition_penalties=repetition_penalty,
                    output_token_ids=output_token_ids,
                    allowed_token_ids_mask=allowed_token_ids_mask,
                    bad_words_token_ids=bad_words_token_ids,
                    logitsprocs=logitsprocs,
                )

                sampler_output = self.host_sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
                next_token_ids = sampler_output.sampled_token_ids
                # Capture logprobs for this DP rank
                logprobs_per_dp.append(sampler_output.logprobs_tensors)
            else:  # sample on device
                # Normalize TT sampled tokens to 1D [sz]. Prefill can return [sz]
                # while decode may return [sz, 1]; downstream logprobs packing
                # expects a flat vector here.
                next_token_ids = _take(tt_out).reshape(sz)
                rank_max_num_logprobs = model_input.max_num_logprobs[dp_rank]
                # Extract logprobs if available from device sampling
                # Always tensors - turned into lists only when passing to model
                assert isinstance(sampling_params.enable_log_probs, torch.Tensor)
                rank_enable_lp = _take(sampling_params.enable_log_probs)
                if rank_enable_lp.any():
                    # Sanity check for if we correctly detect
                    # when logprobs are supported.
                    assert tt_log_probs is not None, (
                        "model should return logprobs when requested"
                    )
                    logprobs_per_dp.append(
                        build_device_logprobs(
                            tt_log_probs=tt_log_probs,
                            sampled_token_ids=next_token_ids,
                            rows=rows,
                            max_num_logprobs=rank_max_num_logprobs or 0,
                        )
                    )
                else:
                    logprobs_per_dp.append(None)

            sampled_token_ids_per_dp.append(next_token_ids.view(sz, 1))

            if is_decode:
                # Fixed stride segments per DP rank for decode
                start += self.tt_per_lane_max_num_seqs
            else:
                # Prefill packed contiguously
                start += sz

        return sampled_token_ids_per_dp, logprobs_per_dp

    def apply_grammar_bitmask(
        self, logits: torch.Tensor, grammar_bitmask: torch.Tensor
    ) -> None:
        """Apply structured output grammar constraints to logits in-place"""
        # The grammar bitmask is compressed as packed int32 values
        # where each bit represents one token. We need to unpack it
        # like the TPU model runner does.
        # Ones in the compressed bitmask represent tokens that are allowed.

        # TODO this is likely a quite inefficient way of doing it on host.

        # grammar_bitmask: (batch_size, bitmask_size)
        # logits: (batch_size, vocab_size)
        unpacked_bitmask = (
            torch.bitwise_right_shift(
                grammar_bitmask[:, :, None],
                self.structured_output_arange[None, None, :],
            )
            & 1
        ) == 0
        unpacked_bitmask = unpacked_bitmask.reshape(grammar_bitmask.shape[0], -1)[
            :, : logits.shape[-1]
        ]
        logits.masked_fill_(unpacked_bitmask, -float("inf"))

    def _build_runner_output(
        self,
        sampled_token_ids: torch.Tensor,
        logprobs: LogprobsLists | None = None,
        req_ids: list[str] | None = None,
        req_id_to_index: dict[str, int] | None = None,
    ) -> ModelRunnerOutput:
        num_reqs = len(req_ids) if req_ids is not None else self.input_batch.num_reqs
        output_req_ids = (
            list(req_ids)
            if req_ids is not None
            else list(self.input_batch.req_ids[:num_reqs])
        )
        output_req_id_to_index = (
            dict(req_id_to_index)
            if req_id_to_index is not None
            else {req_id: idx for idx, req_id in enumerate(output_req_ids)}
        )
        assert sampled_token_ids.shape[0] == num_reqs, (
            f"Number of request outputs {sampled_token_ids.shape[0]} != "
            f"number of requests in input batch {num_reqs}"
        )

        sampled_token_ids_np = sampled_token_ids.view(num_reqs).numpy()
        if sampled_token_ids_np.dtype != np.int32:
            sampled_token_ids_np = sampled_token_ids_np.astype(np.int32, copy=False)

        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = dict.fromkeys(
            (output_req_ids[i] for i in range(num_reqs)), None
        )
        sampled_token_id_lists = [
            [int(token_id)] for token_id in sampled_token_ids_np.tolist()
        ]

        return ModelRunnerOutput(
            req_ids=output_req_ids,
            req_id_to_index=output_req_id_to_index,
            sampled_token_ids=sampled_token_id_lists,
            logprobs=logprobs,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
        )

    def _apply_sampled_tokens_to_state(
        self,
        sampled_token_ids: torch.Tensor,
        req_ids: list[str] | None = None,
        request_states: tuple[CachedRequestState, ...] | None = None,
    ) -> None:
        # When applying a deferred async step, the write row is resolved live
        # from ``req_id_to_index`` (below), not from the row captured at submit
        # time: lane mode pins each request to a stable slot for its lifetime
        # and the ``request_states`` identity check guards slot reuse, so the
        # live row equals the captured one. ``req_id_to_index`` is therefore the
        # single source of truth for the target row.
        use_captured_req_ids = req_ids is not None
        num_reqs = len(req_ids) if req_ids is not None else self.input_batch.num_reqs
        assert sampled_token_ids.shape[0] == num_reqs, (
            f"Number of request outputs {sampled_token_ids.shape[0]} != "
            f"number of requests in input batch {num_reqs}"
        )
        num_out_tokens = sampled_token_ids.shape[1]
        assert num_out_tokens == 1, "Currently only supporting 1 output token"

        sampled_token_ids_np = sampled_token_ids.view(num_reqs).numpy()
        if sampled_token_ids_np.dtype != np.int32:
            sampled_token_ids_np = sampled_token_ids_np.astype(np.int32, copy=False)

        if not use_captured_req_ids:
            rows = np.arange(num_reqs)
            start_idxs = self.input_batch.num_tokens[rows]
            end_idxs = start_idxs + 1
            max_end = int(end_idxs.max()) if num_reqs > 0 else 0
            assert max_end <= self.model_config.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {max_end} > max_model_len: "
                f"{self.model_config.max_model_len}"
            )

            self.input_batch.token_ids_cpu[rows, start_idxs] = sampled_token_ids_np
            self.input_batch.num_tokens[rows] = end_idxs

            for req_idx in range(num_reqs):
                output_token_ids = self.input_batch.req_output_token_ids[req_idx]
                assert output_token_ids is not None
                output_token_ids.append(int(sampled_token_ids_np[req_idx]))
            return

        assert req_ids is not None
        captured_req_ids = req_ids
        for req_idx, req_id in enumerate(captured_req_ids):
            req_state = self.requests.get(req_id)
            if req_state is None:
                continue
            if request_states is not None and req_state is not request_states[req_idx]:
                continue

            current_row = self.input_batch.req_id_to_index.get(req_id)
            if current_row is not None:
                start_idx = int(self.input_batch.num_tokens[current_row])
                end_idx = start_idx + 1
                assert end_idx <= self.model_config.max_model_len, (
                    "Sampled token IDs exceed the max model length. "
                    f"Total number of tokens: {end_idx} > max_model_len: "
                    f"{self.model_config.max_model_len}"
                )
                self.input_batch.token_ids_cpu[current_row, start_idx] = (
                    sampled_token_ids_np[req_idx]
                )
                self.input_batch.num_tokens[current_row] = end_idx

            req_state.output_token_ids.append(int(sampled_token_ids_np[req_idx]))

    def apply_and_build_runner_output(
        self,
        sampled_token_ids: torch.Tensor,
        logprobs: LogprobsLists | None = None,
        req_ids: list[str] | None = None,
        req_id_to_index: dict[str, int] | None = None,
    ):
        """Apply sampled tokens to runner state and build `ModelRunnerOutput`.

        Updates persistent runner state from sampled tokens and returns the
        `ModelRunnerOutput` consumed by the rest of vLLM.
        """
        self._apply_sampled_tokens_to_state(
            sampled_token_ids=sampled_token_ids,
            req_ids=req_ids,
        )
        return self._build_runner_output(
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )

    def warmup_model(self) -> None:
        # Two-phase warmup: compile first, then capture traces.
        #
        # Phase 1 compiles all op variants (prefill + decode) into the
        # program cache WITHOUT capturing any traces.  Phase 2 then
        # captures traces with every op already compiled, so no new
        # kernel-cache allocations occur that could corrupt trace memory.
        #
        # Assumptions / limitations:
        #   1. Traced and non-traced code paths must use the same ops.
        #      If a model uses different operators when enable_trace=False
        #      vs True, Phase 1 will not compile the ops that Phase 2
        #      traces, and new compilations during trace capture will
        #      allocate corruptible buffers.
        #   2. Prefill warmup must cover all supported sequence lengths.
        #      If a new sequence length appears during inference, its
        #      first compilation will allocate new kernel cache entries
        #      (including reshape caches) that can corrupt active traces.
        #
        # See: https://github.com/tenstorrent/tt-metal/commit/5043de3df5
        trace_prefill_mode = self.trace_mode in ["all"]
        trace_decode_mode = self.trace_mode in ["all", "decode_only"]
        sample_on_device_mode = getattr(TTPlatform, "sample_on_device_mode", None)
        assert sample_on_device_mode in (None, "all", "decode_only")
        prefill_kwargs = dict(
            kv_cache=self.kv_caches,
            can_sample_on_device=sample_on_device_mode == "all",
        )
        decode_kwargs = dict(
            kv_cache=self.kv_caches,
            max_batch_size=self.tt_max_batch_size,
            num_blocks=self.max_num_blocks_per_req,
            can_sample_on_device=sample_on_device_mode in ("all", "decode_only"),
        )

        # Phase 1: compile all code paths (no trace capture)
        self.model.warmup_model_prefill(enable_trace=False, **prefill_kwargs)
        self.model.warmup_model_decode(enable_trace=False, **decode_kwargs)

        # Reset prefill warmup flag so Phase 2 re-runs with tracing
        if hasattr(self.model, "already_warmed_up_prefill"):
            self.model.already_warmed_up_prefill = False

        # Phase 2: capture traces (all ops already compiled)
        if trace_prefill_mode:
            self.model.warmup_model_prefill(enable_trace=True, **prefill_kwargs)
        if trace_decode_mode:
            self.model.warmup_model_decode(enable_trace=True, **decode_kwargs)
