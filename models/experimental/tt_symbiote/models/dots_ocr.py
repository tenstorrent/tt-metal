# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standalone TTNN inference pipeline for dots.ocr.

Replaces HF model.generate() entirely, combining scatter-merge on device,
argmax on device, and a custom generation loop.  No monkey-patching, no
HF generate dependency.

Logits are cast to float32 before greedy argmax, and the LM head uses FP32 destination
accumulation, for stable token choices across mesh layouts (same cost as the prior fast
path in practice for this pipeline).
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import ttnn


from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import TracedRun, trace_enabled
from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
    dp_batch_shard_tensor_mapper,
)
from models.experimental.tt_symbiote.modules.dots_ocr_decoder_layer import (
    TTNNDotsOCRDecoderLayer,
    TTNNDotsOCRLayerStack,
)
from models.experimental.tt_symbiote.modules.dots_ocr_vision import TTNNDotsOCRVisionTower
from models.experimental.tt_symbiote.modules.embedding import TTNNEmbedding
from models.experimental.tt_symbiote.modules.linear import (
    TTNNDotsOCRDRAMShardedLMHead,
)
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.utils.device_management import timed_call


def _argmax_token_on_device(logits: ttnn.Tensor) -> ttnn.Tensor:
    """Greedy token from logits.

    Note on the reduce-axis-sharding rule: literally that rule says never
    split the reduce axis across cores. For ``argmax(dim=-1)`` on
    ``[1, 1, vocab=152064]`` the non-reduce extent is 1, so the rule
    points at single-core. We tried ``use_multicore=False`` and it
    regressed wall-clock by ~3 ms/iter (decode 3.62 s -> 4.17 s at DP=8).
    The single-core kernel becomes a sequential 152064-element scan with
    a small read pipeline, while the multi-core kernel parallelizes the
    scan and pays a small NoC fold at the end -- for vocab this large the
    fold cost is dwarfed by the parallel speedup. Stay on
    ``use_multicore=True`` here; the rule still applies as written for
    "normal" reduction shapes (small reduce dim, larger non-reduce
    extent), just not for argmax over a giant vocab with M=1.
    """
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    token = ttnn.argmax(
        logits_rm,
        dim=-1,
        keepdim=True,
        use_multicore=True,
    )
    b = int(token.shape[0])
    return ttnn.reshape(token, (b, 1))


def _unwrap_ttnn_tensor(tt_hidden):
    if hasattr(tt_hidden, "ttnn_tensor") and tt_hidden.ttnn_tensor is not None:
        return tt_hidden.ttnn_tensor
    return tt_hidden


def _sync_profile_enabled() -> bool:
    return os.environ.get("DOTS_OCR_PROFILE_SYNC", "").lower() in {"1", "true", "yes", "on"}


def _defer_decode_readback_enabled() -> bool:
    return os.environ.get("DOTS_OCR_DEFER_DECODE_READBACK", "1").lower() not in {"0", "false", "no", "off"}


def _deep_sync_profile_enabled() -> bool:
    return os.environ.get("DOTS_OCR_PROFILE_DECODE_GRAPH", "").lower() in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def _profile_stage(device, name: str):
    if not _sync_profile_enabled():
        yield
        return
    ttnn.synchronize_device(device)
    start = time.time()
    try:
        yield
    finally:
        ttnn.synchronize_device(device)
        elapsed_ms = (time.time() - start) * 1000.0
        print(f"[DOTS_OCR_PROFILE_SYNC] {name}: {elapsed_ms:.3f} ms")


@contextlib.contextmanager
def _profile_graph_stage(device, name: str):
    if not _deep_sync_profile_enabled():
        yield
        return
    ttnn.synchronize_device(device)
    start = time.time()
    try:
        yield
    finally:
        ttnn.synchronize_device(device)
        elapsed_ms = (time.time() - start) * 1000.0
        print(f"[DOTS_OCR_PROFILE_SYNC] {name}: {elapsed_ms:.3f} ms")


def _dp_repack_batch_sharded_hidden_for_device(device, batch_input_mapper, tt_hidden: ttnn.Tensor) -> ttnn.Tensor:
    """Re-materialize a DP batch-sharded hidden tensor with the pipeline batch mapper.

    Some TTNN ops preserve a global logical batch shape even though DP owns one
    row per device. A host compose + re-upload makes the local shard layout
    match the token-id/embedding mapper before traced decoder execution.
    """
    if batch_input_mapper is None:
        return tt_hidden

    t = _unwrap_ttnn_tensor(tt_hidden)
    ttnn.synchronize_device(device)
    owned = ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mesh_shape = tuple(int(x) for x in device.shape)
    full = ttnn.to_torch(
        owned,
        mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape, (0, -1)),
    )
    ttnn.deallocate(owned)
    return ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=batch_input_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@trace_enabled
class TTNNDotsOCRPrefillGraph(TTNNModule):
    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def __init__(self, decoder_stack, final_norm, lm_head):
        super().__init__()
        self._p_stack = decoder_stack
        self._p_norm = final_norm
        self._p_lm = lm_head

    def forward(self, hidden_states, cache_position, past_key_value):
        h = self._p_stack.forward(hidden_states, past_key_value=past_key_value, cache_position=cache_position)
        h = self._p_norm.forward(h)
        sl = int(h.shape[-2])
        if sl > 1:
            b = int(h.shape[0])
            hd = int(h.shape[-1])
            h = ttnn.slice(h, [0, sl - 1, 0], [b, sl, hd])
        logits = self._p_lm.forward(h)
        return _argmax_token_on_device(logits)

    def post_trace_execute(self, func_args, func_kwargs, result):
        past_key_value = func_kwargs.get("past_key_value")
        if past_key_value is None or not hasattr(past_key_value, "update_seq_length"):
            return
        hid = func_args[0]
        seq_len = int(hid.shape[-2])
        for layer in self._p_stack.layers:
            past_key_value.update_seq_length(layer_idx=layer.self_attn.layer_idx, seq_len=seq_len)


@trace_enabled
class TTNNDotsOCRDecodeGraph(TTNNModule):
    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def __init__(self, decoder_stack, final_norm, lm_head, embedding=None):
        super().__init__()
        self._d_stack = decoder_stack
        self._d_norm = final_norm
        self._d_lm = lm_head
        self._d_embedding = embedding

    def forward(self, decode_input, cache_position, past_key_value):
        with _profile_graph_stage(self.device, "decode.graph.embedding"):
            hidden_states = self._d_embedding.forward(decode_input) if self._d_embedding is not None else decode_input
        with _profile_graph_stage(self.device, "decode.graph.layer_stack"):
            h = self._d_stack.forward(hidden_states, past_key_value=past_key_value, cache_position=cache_position)
        with _profile_graph_stage(self.device, "decode.graph.final_norm"):
            h = self._d_norm.forward(h)
        with _profile_graph_stage(self.device, "decode.graph.lm_head"):
            logits = self._d_lm.forward(h)
        with _profile_graph_stage(self.device, "decode.graph.argmax"):
            return _argmax_token_on_device(logits)

    def post_trace_execute(self, func_args, func_kwargs, result):
        past_key_value = func_kwargs.get("past_key_value")
        if past_key_value is None or not hasattr(past_key_value, "update_seq_length"):
            return
        seq_len = 1
        for layer in self._d_stack.layers:
            past_key_value.update_seq_length(layer_idx=layer.self_attn.layer_idx, seq_len=seq_len)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for TTNNDotsOCRPipeline."""

    vocab_size: int = 151936
    hidden_size: int = 1536
    image_token_id: int = 151665
    eos_token_ids: List[int] = field(default_factory=lambda: [151643, 151673])
    max_new_tokens: int = 512
    num_devices: int = 2  # N300
    batch_size: int = 1


# ---------------------------------------------------------------------------
# Helper: create paged KV cache (mirrors test helper)
# ---------------------------------------------------------------------------


def _create_paged_kv_cache(model_config, device, batch_size: int = 1):
    """Create a paged attention KV cache for dots.ocr.

    Cache dtype is left at default ``bfloat16``. ``bfloat8_b`` was tried
    twice in isolation -- both runs produced visibly corrupted text
    (``EX丝滿º衔task放...``). The K/V values for dots.ocr decode SDPA
    are sensitive to per-element quantization in a way that is not
    captured by simple per-tile statistics; the BFP8 shared exponent
    appears to be too coarse for the long-horizon attention scores in
    this model. Do not flip back to ``bfloat8_b`` without a per-layer
    Q/K/V max-error sweep.
    """
    head_dim = getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // model_config.num_attention_heads,
    )
    # Keep at least 64 pages per DP stream. The vision prompt is ~2.8K tokens;
    # a fixed 256 global block pool gives only 2K tokens/stream at batch 8.
    blocks_per_sequence = 64
    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=max(256, batch_size * blocks_per_sequence),
        batch_size=batch_size,
    )
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
        config=config,
        device=None,
    ).to_device(device)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TTNNDotsOCRPipeline(TTNNModule):
    """Standalone TTNN inference pipeline for dots.ocr.

    Orchestrates embedding, vision tower, decoder stack, final norm, and
    lm_head entirely through TTNN -- no HF ``model.generate()`` dependency.
    Scatter-merge and argmax run on device.

    Inherits from TTNNModule so that ``set_device()`` can recursively
    initialize all child modules (device, device_state, bypass flags,
    weight preprocessing).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        embedding: TTNNEmbedding,
        vision_tower: TTNNDotsOCRVisionTower,
        decoder_stack: TTNNDotsOCRLayerStack,
        final_norm: TTNNDistributedRMSNorm,
        lm_head: TTNNModule,
        paged_cache: TTNNPagedAttentionKVCache,
        graph_prefill: TTNNDotsOCRPrefillGraph,
        graph_decode: TTNNDotsOCRDecodeGraph,
        device: "ttnn.MeshDevice",
        config: PipelineConfig,
    ):
        super().__init__()
        self._bypass_tensor_wrapping = True

        self.embedding = embedding
        self.vision_tower = vision_tower
        self.decoder_stack = decoder_stack
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.graph_prefill = graph_prefill
        self.graph_decode = graph_decode
        self.paged_cache = paged_cache
        self._device = device
        self.config = config
        self._batch_input_mapper = dp_batch_shard_tensor_mapper(device, config.batch_size)

        # Decode-loop buffer (allocated on first decode call, reused thereafter)
        self._decode_cache_position: Optional[ttnn.Tensor] = None
        self._decode_token_buffer: Optional[ttnn.Tensor] = None
        self._decode_token_buffer_has_next: bool = False
        self._decode_token_host: Optional[torch.Tensor] = None
        self._decode_cache_pos_host: Optional[torch.Tensor] = None
        self._decode_seq_counter: int = 0
        self._scatter_cache_input_ids: Optional[torch.Tensor] = None
        self._scatter_cache_key: Optional[tuple] = None
        self._scatter_cache_idx: Optional[ttnn.Tensor] = None
        self._scatter_cache_mask: Optional[ttnn.Tensor] = None
        self._scatter_zero_row_key: Optional[tuple] = None
        self._scatter_zero_row: Optional[ttnn.Tensor] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_hf_model(
        cls,
        model_path: str,
        device: "ttnn.MeshDevice",
        batch_size: int = 1,
    ) -> "TTNNDotsOCRPipeline":
        """Build pipeline from a HuggingFace model path.

        Loads the HF model, extracts components, creates TTNN modules,
        and assembles the pipeline.
        """
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # --- Create TTNN modules from HF components ---
        # _bypass_tensor_wrapping is set later by set_device() which sees
        # that the parent (the pipeline) is a TTNNModule and propagates
        # bypass=True to all children.

        embedding = TTNNEmbedding.from_torch(hf_model.model.embed_tokens)
        embedding._unique_name = "model.embed_tokens"

        vision_tower = TTNNDotsOCRVisionTower.from_torch(hf_model.vision_tower)
        vision_tower._unique_name = "vision_tower"
        vision_tower.override_children_module_names()

        decoder_layers = []
        for i, hf_layer in enumerate(hf_model.model.layers):
            layer = TTNNDotsOCRDecoderLayer.from_torch(hf_layer)
            layer._unique_name = f"model.layers.{i}"
            layer.override_children_module_names()
            decoder_layers.append(layer)
        decoder_stack = TTNNDotsOCRLayerStack(decoder_layers)
        decoder_stack._unique_name = "model.layer_stack"

        final_norm = TTNNDistributedRMSNorm.from_torch(hf_model.model.norm)
        final_norm._unique_name = "model.norm"

        # ``TTNNDotsOCRDRAMShardedLMHead`` lays the lm_head weight out
        # WIDTH_SHARDED across all 12 DRAM banks per chip and runs the matmul
        # with ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``. The
        # decode lm_head is bandwidth-bound (~60-65% DRAM in Tracy), so the
        # 12-bank parallel read pulls per-chunk device time from
        # ~78-100 µs (auto-config, 1-bank serial) toward ~40-50 µs.
        #
        # The previous correctness regression (garbled output when this class
        # was wired in) was a weight-mapper bug: ``shard_tensor_to_mesh_mapper(
        # dim=-1)`` always sharded along the first mesh axis, so on T3K
        # mesh=(8,1) (DP=8, TP=1) every chip got 1/8 of the vocab columns,
        # ``_tp_requires_ccl`` was False so no final all_gather ran, and
        # argmax picked from a 1/8 garbage slice. The class now uses
        # ``_tp_mesh_mapper`` which replicates across the DP axis and shards
        # only across the TP axis -- so DP-only meshes get a fully-replicated
        # vocab on every chip and TP meshes still get the N-parallel split
        # plus all_gather restore.
        lm_head = TTNNDotsOCRDRAMShardedLMHead.from_torch(hf_model.lm_head)
        lm_head._unique_name = "lm_head"

        # Paged KV cache
        paged_cache = _create_paged_kv_cache(hf_model.config, device, batch_size)

        # Pipeline config (before graphs)
        eos_token_ids = [151643, 151673]
        if hasattr(hf_model, "generation_config") and hf_model.generation_config is not None:
            gc_eos = getattr(hf_model.generation_config, "eos_token_id", None)
            if gc_eos is not None:
                eos_token_ids = gc_eos if isinstance(gc_eos, list) else [gc_eos]

        config = PipelineConfig(
            vocab_size=hf_model.config.vocab_size,
            hidden_size=hf_model.config.hidden_size,
            image_token_id=getattr(hf_model.config, "image_token_id", 151665),
            eos_token_ids=eos_token_ids,
            num_devices=device.get_num_devices(),
            batch_size=batch_size,
        )

        graph_prefill = TTNNDotsOCRPrefillGraph(decoder_stack, final_norm, lm_head)
        graph_prefill._unique_name = "dots_ocr_graph_prefill"
        # Bisect step 2: embedding inside the decode trace for **both**
        # batch_size==1 and DP batch-sharded modes. ``TTNNEmbedding`` is
        # ``@trace_enabled``, but the trace decorator at
        # ``run_config.py::TracedRun.__call__`` (line ~1197) explicitly
        # falls back to inline ``forward()`` when ``_TRACE_RUNNING`` is
        # already set by the parent ``graph_decode`` capture (so embedding
        # does NOT create a nested per-instance trace cache here).
        # ``prefill``'s separate ``self.embedding(tt_input_ids)`` call
        # captures embedding's own trace cache for the prefill input shape;
        # that is a different cache key from the decode embedding ops
        # baked into ``graph_decode``'s trace, so they coexist safely.
        #
        # Per chip the decode embedding input is ``[1, 1] uint32 ROW_MAJOR``
        # regardless of DP, identical to the proven batch_size==1 path. This
        # also kills the per-iter ``_dp_repack_batch_sharded_hidden`` host
        # roundtrip (sync + clone + to_torch via ConcatMesh2dToTensor +
        # from_torch via _batch_input_mapper) since the trace was captured
        # with already-properly-DP-sharded ``_decode_token_buffer`` and
        # therefore preserves mesh-sharding metadata on its outputs.
        graph_decode = TTNNDotsOCRDecodeGraph(decoder_stack, final_norm, lm_head, embedding=embedding)
        graph_decode._unique_name = "dots_ocr_graph_decode"

        pipeline = cls(
            embedding=embedding,
            vision_tower=vision_tower,
            decoder_stack=decoder_stack,
            final_norm=final_norm,
            lm_head=lm_head,
            paged_cache=paged_cache,
            graph_prefill=graph_prefill,
            graph_decode=graph_decode,
            device=device,
            config=config,
        )
        pipeline._unique_name = "dots_ocr_pipeline"

        # Set device and preprocess weights
        pipeline._set_device_and_preprocess(device)
        pipeline.prefill = timed_call(pipeline.prefill, "prefill", "TTNNDotsOCRPipelinePrefill")
        pipeline.decode_step = timed_call(pipeline.decode_step, "decode", "TTNNDotsOCRPipelineDecode")
        return pipeline

    # ------------------------------------------------------------------
    # Device / weight setup
    # ------------------------------------------------------------------

    def _set_device_and_preprocess(self, device: "ttnn.MeshDevice") -> None:
        """Recursively set device/device_state on all children, then preprocess weights."""
        from models.experimental.tt_symbiote.utils.device_management import set_device

        set_device(self, device, register_forward_hook=False, dump_visualization=False)

        # Preprocess and move weights for every leaf TTNNModule.
        for module in self._collect_ttnn_modules():
            module.preprocess_weights()
            module.move_weights_to_device()

        # LM head: HiFi2 + packer_l1_acc + FP32 dest accum (matches stable argmax path).
        self.lm_head.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _collect_ttnn_modules(self) -> List[TTNNModule]:
        """Recursively collect all TTNNModule instances from pipeline components."""
        found: List[TTNNModule] = []
        visited: set = set()

        def _recurse(obj):
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)
            if isinstance(obj, TTNNModule):
                found.append(obj)
            for attr_name in dir(obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    value = getattr(obj, attr_name)
                except Exception:
                    continue
                if isinstance(value, TTNNModule):
                    _recurse(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, TTNNModule):
                            _recurse(item)
                elif isinstance(value, dict):
                    for item in value.values():
                        if isinstance(item, TTNNModule):
                            _recurse(item)

        for component in [
            self.embedding,
            self.vision_tower,
            self.decoder_stack,
            self.final_norm,
            self.lm_head,
            self.graph_prefill,
            self.graph_decode,
        ]:
            _recurse(component)
        return found

    def _mesh_dp_dual_stream(self) -> bool:
        """True when batch is sharded one row per device (DP batch parallel)."""
        return self._batch_input_mapper is not None

    def _dp_repack_batch_sharded_hidden(self, tt_hidden: ttnn.Tensor) -> ttnn.Tensor:
        return _dp_repack_batch_sharded_hidden_for_device(self.device, self._batch_input_mapper, tt_hidden)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Union[int, List[int]]:
        """Run prefill (first forward pass) and return the first generated token(s).

        Args:
            input_ids: ``[1, S]`` or ``[B, S]`` int64/int32 token IDs on host.
                With DP mesh batch sharding (``B == num_devices``), use one row
                per independent stream; each device runs local batch 1.
            pixel_values: Optional vision input for multimodal prefill.
            image_grid_thw: Optional grid info for vision input.

        Returns:
            First predicted token ID (int), or a list of ``B`` IDs when
            ``_mesh_dp_dual_stream()`` is active.
        """
        dual = self._mesh_dp_dual_stream()
        if dual and int(input_ids.shape[0]) != int(self.config.batch_size):
            raise ValueError(
                f"DP batch-parallel prefill expects input_ids batch {self.config.batch_size}, "
                f"got {input_ids.shape[0]}"
            )
        seq_len = input_ids.shape[-1]

        # --- Embedding ---
        # All children have _bypass_tensor_wrapping=True (pipeline is a
        # TTNNModule parent), so we convert input_ids to ttnn ourselves.
        id_mapper = (
            self._batch_input_mapper
            if self._batch_input_mapper is not None
            else ttnn.ReplicateTensorToMesh(self.device)
        )
        with _profile_stage(self.device, "prefill.input_ids_h2d"):
            tt_input_ids = ttnn.from_torch(
                input_ids.to(torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=id_mapper,
            )
        with _profile_stage(self.device, "prefill.text_embedding"):
            text_embeds = self.embedding(tt_input_ids)

        # --- Vision scatter-merge (if applicable) ---
        if pixel_values is not None:
            with _profile_stage(self.device, "prefill.vision_scatter_merge"):
                hidden_states = self._scatter_merge_on_device(text_embeds, pixel_values, image_grid_thw, input_ids)
        else:
            hidden_states = text_embeds
        if dual and self._batch_input_mapper is not None and int(hidden_states.shape[0]) > 1:
            with _profile_stage(self.device, "prefill.dp_repack_hidden"):
                hidden_states = self._dp_repack_batch_sharded_hidden(hidden_states)

        # --- Cache position for prefill ---
        with _profile_stage(self.device, "prefill.cache_position_h2d"):
            cache_position = torch.arange(0, seq_len, dtype=torch.int32)
            tt_cache_position = ttnn.from_torch(
                cache_position,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Traced prefill graph includes decoder stack, final norm, lm_head, and argmax.
        with _profile_stage(self.device, "prefill.graph_prefill_sync"):
            token_id_tt = self.graph_prefill(hidden_states, tt_cache_position, past_key_value=self.paged_cache)

        # --- Read to host ---
        with _profile_stage(self.device, "prefill.first_token_readback"):
            token_id_torch = ttnn.to_torch(
                token_id_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
            )
        flat_ids = [int(x) for x in token_id_torch.reshape(-1).tolist()]
        if dual:
            if len(flat_ids) != int(self.config.batch_size):
                raise RuntimeError(
                    f"Expected {self.config.batch_size} first tokens from mesh concat, got {len(flat_ids)}"
                )
            first_out: Union[int, List[int]] = flat_ids
        else:
            first_out = flat_ids[0]

        # Clean up prefill-only tensors
        ttnn.deallocate(tt_cache_position)

        return first_out

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def _init_decode_buffers(self, prev_token_id: Union[int, List[int]]):
        """Allocate reusable device buffers for decode loop on first call."""
        if isinstance(prev_token_id, int):
            self._decode_token_host = torch.tensor([[prev_token_id]], dtype=torch.int32)
        else:
            self._decode_token_host = torch.tensor(prev_token_id, dtype=torch.int32).reshape(len(prev_token_id), 1)
        token_mapper = (
            self._batch_input_mapper
            if self._batch_input_mapper is not None
            else ttnn.ReplicateTensorToMesh(self.device)
        )
        self._decode_token_buffer = ttnn.from_torch(
            self._decode_token_host,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=token_mapper,
        )
        self._decode_token_buffer_has_next = True
        self._decode_seq_counter = self.paged_cache.get_seq_length(layer_idx=0)
        self._decode_cache_pos_host = torch.tensor([self._decode_seq_counter], dtype=torch.int32)
        # Scalar cache position: always replicate (same global index on every
        # device). Do not use ``_batch_input_mapper`` here: ND shard expects one
        # host chunk per mesh device, which a length-1 tensor does not satisfy.
        self._decode_cache_position = ttnn.from_torch(
            self._decode_cache_pos_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def decode_step(
        self, prev_token_id: Union[int, List[int]], read_from_device: bool = True
    ) -> Union[int, List[int], ttnn.Tensor]:
        """Execute one decode step entirely on device.

        Args:
            prev_token_id: Previous token ID(s) on host: ``int`` for batch 1, or
                a list of length ``B`` for DP batch-parallel streams.

        Returns:
            Next predicted token ID, or a list of ``B`` IDs when dual-stream
            DP batch mode is active.
        """
        if self._decode_token_buffer is None:
            with _profile_stage(self.device, "decode.init_buffers"):
                self._init_decode_buffers(prev_token_id)
        else:
            # ``has_next`` is True whenever the previous iter's argmax already
            # wrote the next-token directly into ``_decode_token_buffer`` via
            # ``ttnn.copy(token_id_tt, _decode_token_buffer)`` below. In that
            # case the host upload is redundant -- the device already has the
            # token. This holds for **both** non-DP single-stream and DP
            # batch-sharded modes: per chip, ``token_id_tt`` and
            # ``_decode_token_buffer`` are both [1,1] uint32 ROW_MAJOR, so
            # the on-device copy is just a per-chip element copy and the
            # mesh-sharding metadata stays consistent.
            #
            # NOTE: this flag is the **only** DP host-overhead reduction
            # being landed in this pass (step 1 of the bisect after the
            # malloc-corrupted multi-change attempt). Embedding-in-trace
            # and pipelined readback stay disabled for DP until step 1 is
            # confirmed clean across two ``generate()`` calls.
            upload_prev_token = not self._decode_token_buffer_has_next
            if upload_prev_token:
                if isinstance(prev_token_id, int):
                    self._decode_token_host[0][0] = prev_token_id
                else:
                    for row, tid in enumerate(prev_token_id):
                        self._decode_token_host[row][0] = tid
            if upload_prev_token and self._batch_input_mapper is not None:
                # Mesh-sharded device buffer does not match host logical shape [B,1];
                # upload with the same mapper, then device-to-device copy into the
                # preallocated decode buffer (for trace-stable tensor identity).
                with _profile_stage(self.device, "decode.prev_token_h2d_dp"):
                    token_upload = ttnn.from_torch(
                        self._decode_token_host,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.device,
                        mesh_mapper=self._batch_input_mapper,
                    )
                    ttnn.copy(token_upload, self._decode_token_buffer)
                    ttnn.deallocate(token_upload)
            elif upload_prev_token:
                with _profile_stage(self.device, "decode.prev_token_h2d"):
                    token_host_tt = ttnn.from_torch(
                        self._decode_token_host,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                    ttnn.copy_host_to_device_tensor(token_host_tt, self._decode_token_buffer)

            self._decode_seq_counter += 1
            # Device-side cache_position increment for both DP and non-DP.
            # ``_decode_cache_position`` is REPLICATED across the mesh (a
            # single global counter, see ``_init_decode_buffers``) so the
            # ``ttnn.add`` runs identically on every chip and stays in sync
            # without any host-side h2d. Used to be DP-gated to a host h2d
            # path (``_batch_input_mapper is not None``); removing the gate
            # cuts another per-iter ``ttnn.from_torch + copy_host_to_device``
            # (~3-5 ms wall-clock at DP=8 on T3K).
            with _profile_stage(self.device, "decode.cache_position_device_inc"):
                # NOTE: tried ``ttnn.add(..., output_tensor=cache_position)``
                # to eliminate the explicit ``ttnn.copy`` + ``ttnn.deallocate``,
                # but the in-place output_tensor path raises
                # ``Optional output tensor with Row Major input is not
                # supported right now for Elementwise operations``
                # (ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp:362)
                # because ``_decode_cache_position`` is INT32 / ROW_MAJOR. Stay
                # on the temp-buffer path until in-place row-major elementwise
                # is supported.
                cache_next = ttnn.add(self._decode_cache_position, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.copy(cache_next, self._decode_cache_position)
                ttnn.deallocate(cache_next)

        trace_decode_tokens = getattr(self.graph_decode, "_d_embedding", None) is not None
        if trace_decode_tokens:
            decode_input = self._decode_token_buffer
        else:
            with _profile_stage(self.device, "decode.embedding"):
                decode_input = self.embedding(self._decode_token_buffer)
            if self._mesh_dp_dual_stream() and self._batch_input_mapper is not None:
                with _profile_stage(self.device, "decode.dp_repack_hidden"):
                    decode_input = self._dp_repack_batch_sharded_hidden(decode_input)

        with _profile_stage(self.device, "decode.graph_decode_sync"):
            token_id_tt = self.graph_decode(decode_input, self._decode_cache_position, past_key_value=self.paged_cache)
        # On-device token feedback. Skip the per-iter ``ttnn.from_torch +
        # ttnn.copy_host_to_device_tensor`` host upload by writing the
        # next-iter input directly on device.
        #
        # Per chip, both ``token_id_tt`` (argmax output reshaped to ``[b,1]``
        # in ``_argmax_token_on_device``) and ``_decode_token_buffer`` are
        # ``uint32`` ``ROW_MAJOR`` with the same physical ``[1,1]`` shape per
        # chip (DP shards along batch dim, replicated metadata otherwise),
        # so the copy is a per-chip element copy and mesh-sharding metadata
        # is preserved.
        ttnn.copy(token_id_tt, self._decode_token_buffer)
        self._decode_token_buffer_has_next = True
        token_id_snapshot = None
        if self._batch_input_mapper is None and not read_from_device:
            # Pipelined async d2h via ``.cpu(blocking=False)`` is enabled
            # **only** for non-DP single-stream / replicated-mesh decode.
            # Tried for DP (step 3 of the bisect) but reproducibly crashed
            # with ``malloc(): unaligned tcache chunk detected`` on the
            # warmup pass 2 prefill ``from_torch(input_ids)`` -- the
            # multi-device mesh-sharded snapshot lifecycle is broken in
            # this path. Stay gated until a safer DP-aware mechanism
            # (ttnn events / pre-allocated host buffer pool) is wired up.
            token_id_snapshot = token_id_tt.cpu(blocking=False)

        if not read_from_device:
            if token_id_snapshot is None:
                raise RuntimeError("Deferred decode readback did not produce a token snapshot")
            return token_id_snapshot

        # --- Read to host ---
        with _profile_stage(self.device, "decode.token_readback"):
            token_id_torch = ttnn.to_torch(
                token_id_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
            )
        flat_ids = [int(x) for x in token_id_torch.reshape(-1).tolist()]
        if self._mesh_dp_dual_stream():
            return flat_ids
        return flat_ids[0]

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
    ) -> Union[List[int], List[List[int]]]:
        """Full generation: prefill + decode loop.

        Args:
            input_ids: ``[1, S]`` or ``[B, S]`` token IDs on host (``B`` streams
                for DP mesh batch sharding).
            pixel_values: Optional vision input.
            image_grid_thw: Optional grid info for vision.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            List of generated token IDs per stream (prompt excluded). Single
            stream: ``List[int]``. DP batch-parallel: ``List[List[int]]`` with
            one inner list per stream (same decode depth; each stream stops
            appending on EOS but the step keeps KV aligned for active rows).
        """
        # Reset cache for fresh generation
        self.paged_cache.reset()
        self._decode_cache_position = None
        self._decode_token_buffer = None
        self._decode_token_buffer_has_next = False
        self._decode_seq_counter = 0

        # Prefill
        first_out = self.prefill(input_ids, pixel_values, image_grid_thw)

        if self._mesh_dp_dual_stream():
            if not isinstance(first_out, list):
                raise RuntimeError("prefill must return a list of first tokens in DP dual-stream mode")
            currents: List[int] = list(first_out)
            generated: List[List[int]] = [[t] for t in currents]
            active = [True] * len(currents)
            num_streams = len(currents)
            # Bisect-step-3 host-overhead state:
            # - cache_position: device-side ``ttnn.add`` increment -- on.
            # - token feedback: on-device ``ttnn.copy`` -- on (step 1).
            # - embedding: inside the decode trace -- on (step 2).
            # - readback: 1-deep pipelined via ``token_id_tt.cpu(blocking=False)``
            #   in ``decode_step`` (step 3, this pass). The d2h overlaps
            #   with the NEXT iter's trace replay; resolve via
            #   ``ttnn.to_torch + ConcatMeshToTensor`` lagged by 1 iter.
            #
            # Correctness:
            # - The next iter's trace replay reads ``_decode_token_buffer``,
            #   which the previous iter wrote on-device via
            #   ``ttnn.copy(token_id_tt, _decode_token_buffer)`` BEFORE the
            #   ``.cpu(blocking=False)`` was scheduled. So the next trace
            #   input is fully on-device and independent of the pending
            #   d2h.
            # - tt-metal serializes a queued d2h on ``token_id_tt`` before
            #   any subsequent write to the same buffer (the next iter's
            #   argmax inside the decode trace), so the host snapshot
            #   captures the correct per-stream tokens even though the
            #   device buffer is reused.
            # - EOS lag is at most 1 iter; the trailing snapshot is
            #   resolved after the loop.
            # Synchronous DP loop. Step 3 (1-deep pipelined readback via
            # ``token_id_tt.cpu(blocking=False)``) crashed with
            # ``malloc(): unaligned tcache chunk detected`` even on its own
            # (with steps 1 & 2 already verified stable in isolation): the
            # ``.cpu(blocking=False)`` snapshot lifecycle on a
            # multi-device mesh-sharded tensor is broken when combined
            # with ``ttnn.deallocate(snapshot)``. Skipping the deallocate
            # alone was insufficient (still corrupted the host heap on
            # warmup pass 2's prefill ``from_torch(input_ids)``), so the
            # failure is in ``.cpu(blocking=False)`` itself for DP, not
            # just the explicit deallocate.
            #
            # Stay on the synchronous loop for DP. Steps 1 (on-device token
            # feedback) and 2 (embedding-in-trace) remain enabled, so the
            # loop is much cheaper than baseline -- only the per-iter
            # ~10 ms ``ConcatMeshToTensor`` d2h on the host critical path
            # remains as the dominant DP host overhead. Future work:
            # explicit ttnn events, or a pre-allocated host buffer pool
            # populated via a non-blocking ``ttnn.to_torch`` variant.
            for _ in range(max_new_tokens - 1):
                if not any(active):
                    break
                next_toks = self.decode_step(currents)
                if not isinstance(next_toks, list):
                    raise RuntimeError("decode_step must return a list in DP dual-stream mode")
                for i in range(num_streams):
                    if not active[i]:
                        continue
                    generated[i].append(next_toks[i])
                    if next_toks[i] in self.config.eos_token_ids:
                        active[i] = False
                currents = list(next_toks)
            return generated

        if not isinstance(first_out, int):
            raise RuntimeError("prefill must return a single int in non-DP mode")
        first_token_id = first_out
        generated_single: List[int] = [first_token_id]
        current_token = first_token_id

        if _defer_decode_readback_enabled() and max_new_tokens > 1:
            if self._batch_input_mapper is not None:
                raise RuntimeError("DOTS_OCR_DEFER_DECODE_READBACK is only supported for non-DP single-stream decode")
            deferred_tokens: List[ttnn.Tensor] = []
            for _ in range(max_new_tokens - 1):
                token_tt = self.decode_step(current_token, read_from_device=False)
                if not isinstance(token_tt, ttnn.Tensor):
                    raise RuntimeError("Deferred decode step must return a TTNN token tensor")
                deferred_tokens.append(token_tt)
            with _profile_stage(self.device, "decode.deferred_token_readback"):
                ttnn.synchronize_device(self.device)
                for token_tt in deferred_tokens:
                    token_torch = ttnn.to_torch(
                        token_tt,
                        mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                    )
                    generated_single.append(int(token_torch.reshape(-1)[0].item()))
                    ttnn.deallocate(token_tt)
            for idx, token in enumerate(generated_single):
                if token in self.config.eos_token_ids:
                    return generated_single[: idx + 1]
            return generated_single

        # 1-deep pipelined readback: issue iter N+1's trace replay before
        # resolving iter N's token. Removes the per-token sync wall caused
        # by ``ttnn.to_torch`` blocking until the d2h copy of the previous
        # token completes. Safe because:
        #   * ``decode_step`` already does an on-device
        #     ``ttnn.copy(token_id_tt, _decode_token_buffer)`` before
        #     issuing the d2h, so the next trace replay's input is set
        #     without needing a host roundtrip.
        #   * In steady-state non-DP single-stream decode the
        #     ``prev_token_id`` arg to ``decode_step`` is unused (gated on
        #     ``not _decode_token_buffer_has_next or _batch_input_mapper``).
        #   * tt-metal serializes a queued d2h copy of ``token_id_tt``
        #     before the next trace replay's write to that same buffer, so
        #     the host snapshot captures the correct token even though
        #     the device tensor is reused.
        # EOS still early-exits; worst case we executed exactly 1 extra
        # decode iteration past the EOS token before discarding it.
        if max_new_tokens > 1 and self._batch_input_mapper is None:
            prev_snapshot: Optional[ttnn.Tensor] = None
            for _ in range(max_new_tokens - 1):
                cur_snapshot = self.decode_step(current_token, read_from_device=False)
                if not isinstance(cur_snapshot, ttnn.Tensor):
                    raise RuntimeError("decode_step must return a TTNN snapshot in pipelined readback")
                if prev_snapshot is not None:
                    prev_torch = ttnn.to_torch(
                        prev_snapshot,
                        mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                    )
                    next_token = int(prev_torch.reshape(-1)[0].item())
                    ttnn.deallocate(prev_snapshot)
                    generated_single.append(next_token)
                    if next_token in self.config.eos_token_ids:
                        ttnn.deallocate(cur_snapshot)
                        return generated_single
                prev_snapshot = cur_snapshot
            if prev_snapshot is not None:
                prev_torch = ttnn.to_torch(
                    prev_snapshot,
                    mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                )
                next_token = int(prev_torch.reshape(-1)[0].item())
                ttnn.deallocate(prev_snapshot)
                generated_single.append(next_token)
            return generated_single

        for _ in range(max_new_tokens - 1):
            next_tok = self.decode_step(current_token)
            if isinstance(next_tok, list):
                raise RuntimeError("decode_step returned a list in single-stream mode")
            next_token = next_tok
            generated_single.append(next_token)

            if next_token in self.config.eos_token_ids:
                break

            current_token = next_token

        return generated_single

    # ------------------------------------------------------------------
    # Scatter-merge (vision + text on device)
    # ------------------------------------------------------------------

    def _scatter_merge_on_device(
        self,
        text_embeds: ttnn.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> ttnn.Tensor:
        """Merge vision embeddings into text embeddings at image token positions.

        Args:
            text_embeds: ``[1, S, H_per_device]`` per mesh device (col-sharded) when
                DP batch-sharded, else ``[1, S, H_per_device]`` replicated batch 1.
            pixel_values: Vision input (torch on host).
            image_grid_thw: Grid info (torch on host).
            input_ids: ``[B, S]`` token IDs (torch on host, for gather/mask). With
                DP batch sharding, ``B`` matches the mesh stream count.

        Returns:
            fused_embeds: same shape as text_embeds, bf16 on device.
        """
        S = text_embeds.shape[-2]
        H = self.config.hidden_size

        num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1

        # =====================================================================
        # Step A: Vision tower produces col-sharded output
        # =====================================================================
        vision_tt = self.vision_tower.forward(pixel_values, image_grid_thw)

        # =====================================================================
        # Step B: Build vision table ON DEVICE (no host round-trip).
        # Convert TILE -> ROW_MAJOR, reshape to 2D, prepend a tiny
        # zero row, so the embedding table is [N_vision+1, H/num_devices]
        # per device, col-sharded.
        # =====================================================================
        N_vision = int(vision_tt.shape[2])
        H_per_device = int(vision_tt.shape[3])

        # B.1: TILE -> ROW_MAJOR on device (needed by ttnn.embedding)
        vision_rm = ttnn.to_layout(vision_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(vision_tt)

        # B.2: Reshape [1, 1, N_vision, H_per_device] -> [N_vision, H_per_device]
        vision_2d = ttnn.reshape(vision_rm, (N_vision, H_per_device))

        # B.3: Create zero row (tiny: ~3 KB upload, not a bottleneck)
        if num_devices > 1:
            zero_row_mapper = ttnn.ShardTensor2dMesh(
                self.device,
                dims=(None, -1),
                mesh_shape=list(self.device.shape),
            )
        else:
            zero_row_mapper = ttnn.ReplicateTensorToMesh(self.device)

        zero_row_key = (int(H), int(num_devices), self._batch_input_mapper is not None)
        if self._scatter_zero_row is None or self._scatter_zero_row_key != zero_row_key:
            if self._scatter_zero_row is not None:
                ttnn.deallocate(self._scatter_zero_row)
            self._scatter_zero_row = ttnn.from_torch(
                torch.zeros(1, H, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=zero_row_mapper,
            )
            self._scatter_zero_row_key = zero_row_key

        # B.4: Concat zero row + vision embeddings -> vision table
        vision_table = ttnn.concat([self._scatter_zero_row, vision_2d], dim=0)
        ttnn.deallocate(vision_2d)

        # =====================================================================
        # Step C: Build gather index on host (tiny: ~19 KB) and upload
        # gather_idx[0, pos] = 0 for non-image tokens (gathers zero row)
        #                    = 1..N_vision for image tokens
        # =====================================================================
        idx_mapper = (
            self._batch_input_mapper
            if self._batch_input_mapper is not None
            else ttnn.ReplicateTensorToMesh(self.device)
        )
        B = int(input_ids.shape[0])
        cache_key = (B, int(S), int(N_vision), int(H_per_device), self._batch_input_mapper is not None)
        cache_hit = (
            self._scatter_cache_key == cache_key
            and self._scatter_cache_input_ids is not None
            and torch.equal(self._scatter_cache_input_ids, input_ids)
            and self._scatter_cache_idx is not None
            and self._scatter_cache_mask is not None
        )
        if cache_hit:
            tt_idx = self._scatter_cache_idx
            tt_mask = self._scatter_cache_mask
        else:
            gather_idx = torch.zeros(B, S, dtype=torch.int32)
            for b in range(B):
                img_mask_b = input_ids[b] == self.config.image_token_id
                img_positions = img_mask_b.nonzero(as_tuple=True)[0]
                n_img = min(len(img_positions), N_vision)
                gather_idx[b, img_positions[:n_img]] = torch.arange(1, n_img + 1, dtype=torch.int32)

            tt_idx = ttnn.from_torch(
                gather_idx,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=idx_mapper,
            )

            img_mask = input_ids == self.config.image_token_id
            mask_float = img_mask.float().unsqueeze(-1)  # [B, S, 1]
            tt_mask = ttnn.from_torch(
                mask_float,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=idx_mapper,
            )
            if self._scatter_cache_idx is not None:
                ttnn.deallocate(self._scatter_cache_idx)
            if self._scatter_cache_mask is not None:
                ttnn.deallocate(self._scatter_cache_mask)
            self._scatter_cache_key = cache_key
            self._scatter_cache_input_ids = input_ids.detach().clone()
            self._scatter_cache_idx = tt_idx
            self._scatter_cache_mask = tt_mask

        # =====================================================================
        # Step D: Device-side gather via ttnn.embedding
        # With col-sharded table [N_vision+1, H/num_devices] and replicated
        # index, each device independently gathers its own H/num_devices
        # columns -> [1, S, H/num_devices] col-sharded output.
        # No identity matmul needed!
        # =====================================================================
        full_vision_col_sharded = ttnn.embedding(tt_idx, vision_table, layout=ttnn.TILE_LAYOUT)
        # full_vision_col_sharded: [1, S, H/num_devices] col-sharded, TILE_LAYOUT
        ttnn.deallocate(vision_table)

        # =====================================================================
        # Step E: Build mask and merge
        # =====================================================================
        fused = ttnn.where(tt_mask, full_vision_col_sharded, text_embeds)

        return fused

    # ------------------------------------------------------------------
    # Argmax on device
    # ------------------------------------------------------------------

    def _argmax_on_device(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        """Run argmax on device.

        Args:
            logits: ``[1, 1, vocab_size]`` bf16 TILE_LAYOUT on device (rank 3).

        Returns:
            token_id tensor on device.
        """
        return _argmax_token_on_device(logits)

    # ------------------------------------------------------------------
    # Warmup / trace management
    # ------------------------------------------------------------------

    def warmup(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> None:
        """Run warmup passes to prime JIT and capture traces.

        Call sequence:
          1. Warmup run (no trace): primes TTNN JIT, CCL, allocators.
          2. TracedRun.release_all(): release any traces from warmup.
          3. Trace capture run: captures decode trace.
          4. Reset: clear KV cache, reset positions.
        """
        # Pass 1: Warmup (TracedRun phase 1 -- no trace capture)
        self.generate(input_ids, pixel_values, image_grid_thw, max_new_tokens=2)
        self.paged_cache.reset()
        self._decode_cache_position = None

        # Release all traces so run 2 starts clean
        TracedRun.release_all()

        # Pass 2: Trace capture (TracedRun phase 2)
        self.generate(input_ids, pixel_values, image_grid_thw, max_new_tokens=4)
        self.paged_cache.reset()
        self._decode_cache_position = None

    def release(self) -> None:
        """Release all traced runs and deallocate pre-allocated buffers."""
        TracedRun.release_all()
        self._decode_cache_position = None
        if self._scatter_cache_idx is not None:
            ttnn.deallocate(self._scatter_cache_idx)
            self._scatter_cache_idx = None
        if self._scatter_cache_mask is not None:
            ttnn.deallocate(self._scatter_cache_mask)
            self._scatter_cache_mask = None
        if self._scatter_zero_row is not None:
            ttnn.deallocate(self._scatter_zero_row)
            self._scatter_zero_row = None
            self._scatter_zero_row_key = None
        self._scatter_cache_input_ids = None
        self._scatter_cache_key = None
