# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer


@dataclass
class TtPrefillRuntimeConfig:
    num_layers: int  # layers built by THIS runtime (the rank's slice; == model total for single-rank)
    max_seq_len: int  # per-user KV-cache length (tokens), e.g. 60 * 1024
    mesh_shape: tuple = (32, 4)
    # Chunked prefill streams tokens in chunks of `chunk_size`, with `num_users` independent cache
    # slots (user-major batch). The full cache holds num_users * num_layers slots of max_seq_len each.
    chunk_size: int = 5 * 1024
    num_users: int = 2
    sp_axis: int = 0
    tp_axis: int = 1
    num_links: int = 1
    topology: ttnn.Topology = ttnn.Topology.Linear
    capacity_factor: int = 2
    gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL
    routed_expert_activations_dtype: ttnn.DataType = ttnn.bfloat8_b
    routed_expert_weights_dtype: ttnn.DataType = ttnn.bfloat4_b
    shared_expert_activations_dtype: ttnn.DataType = ttnn.bfloat16
    shared_expert_weights_dtype: ttnn.DataType = ttnn.bfloat8_b
    weight_cache_path: Optional[Path] = None
    # Route the MoE routing all-gather's global semaphores to L1_SMALL (instead of pinning the main-L1
    # floor and clashing with the next layer's MLA static CBs). Requires the mesh opened with
    # l1_small_size > 0. Enable for Kimi (single expert group, device gate). See TtMoERoutingSetup.
    routing_use_l1_small_for_semaphores: bool = False
    # Static model-dimension constants for the model being built
    # (DeepSeekV3Config | KimiK26Config). Drives expert counts, dense-layer
    # count, route groups, etc. in the TT layer code. Supplied by the model
    # adapter — no default, so the runtime never bakes in a specific model.
    model_cfg: Optional[type] = None
    # When True, the last transformer layer runs kv-only: it fills the KV cache
    # (which migration needs) and skips its Q/SDPA/output projection, FFN/MoE,
    # the final RMSNorm, and the LM head. `prefill()` then returns None. The pipeline
    # sets this on the last rank so the final stage is headless.
    kv_only_last_layer: bool = False
    # Pipeline-parallel rank slicing. first_layer_idx is the global index of this
    # rank's first layer; is_first_rank gates the embedding, is_last_rank marks the
    # final stage (non-last ranks forward the hidden state instead of running a tail).
    # Defaults make a single-rank runtime own the whole model.
    first_layer_idx: int = 0
    is_first_rank: bool = True
    is_last_rank: bool = True

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]


class TtPrefillRuntime:
    """Single-rank prefill execution lifecycle: build model -> allocate KV cache ->
    compile -> prefill(chunk). Owns the KVPE cache and the per-layer LayerAck wiring.

    A runtime owns one rank's layer slice. For single-rank prefill the slice is the
    whole model (the config defaults). For pipeline-parallel prefill, a driver builds
    one runtime per rank with first_layer_idx / is_first_rank / is_last_rank set, and
    the non-boundary ranks consume/produce hidden-state activations instead of token
    IDs / sampled tokens.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        hf_config: PretrainedConfig,
        state_dict: dict,
        config: TtPrefillRuntimeConfig,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config
        assert config.model_cfg is not None, "TtPrefillRuntimeConfig.model_cfg must be set by the model adapter"
        # Per-layer LayerAck callback, built once in set_layer_ack_channel() after compile.
        self._on_layer_complete = None
        # Per-layer completion sink (pipelined mode), set by set_layer_completion_sink().
        # Signature: sink(layer_idx, request_id). prefill() binds the current request_id into
        # a fresh per-call closure, so there is no shared mutable chunk-index for the callback
        # to race on (immune even if the threading model changes).
        self._layer_completion_sink = None

        assert (
            config.max_seq_len % config.chunk_size == 0
        ), f"max_seq_len ({config.max_seq_len}) must be a multiple of chunk_size ({config.chunk_size})"

        self.model_built = False
        self.compiled = False

        self._build_model(state_dict)

    def _build_model(self, state_dict: dict) -> None:
        logger.info(
            f"Building TtPrefillRuntime model: "
            f"num_layers={self.config.num_layers}, first_layer_idx={self.config.first_layer_idx}, "
            f"is_first_rank={self.config.is_first_rank}, is_last_rank={self.config.is_last_rank}, "
            f"max_seq_len={self.config.max_seq_len}, mesh_shape={self.config.mesh_shape}, "
            f"chunk_size={self.config.chunk_size}, num_users={self.config.num_users}"
        )
        model_cfg = self.config.model_cfg
        if self.config.weight_cache_path:
            num_devices = self.config.mesh_shape[0] * self.config.mesh_shape[1]
            experts_per_chip = model_cfg.NUM_ROUTED_EXPERTS // num_devices
            if TtPrefillTransformer.check_cache_complete(
                self.config.weight_cache_path,
                self.config.num_layers,
                experts_per_chip,
                first_k_dense=model_cfg.NUM_DENSE_LAYERS,
                first_layer_idx=self.config.first_layer_idx,
                is_first_rank=self.config.is_first_rank,
                is_last_rank=self.config.is_last_rank,
                kv_only_last_layer=self.config.kv_only_last_layer,
            ):
                logger.info(f"TTNN weight cache complete at {self.config.weight_cache_path}; loading from disk")
            else:
                logger.warning(
                    f"TTNN weight cache not complete at {self.config.weight_cache_path}; "
                    f"build will fail without a populated cache. "
                    f"Run the pretrained smoke test once to populate it."
                )
        self.model = TtPrefillTransformer(
            mesh_device=self.mesh_device,
            config=self.hf_config,
            model_cfg=model_cfg,
            state_dict=state_dict,
            num_layers=self.config.num_layers,
            seq_len=self.config.chunk_size,  # per-chunk size -> MoE/FFN dispatch buffers
            max_seq_len=self.config.max_seq_len,  # KV ring buffer = full per-user cache
            num_links=self.config.num_links,
            topology=self.config.topology,
            sp_axis=self.config.sp_axis,
            tp_axis=self.config.tp_axis,
            is_balanced=False,  # chunked prefill is block-cyclic (non-balanced)
            dispatch_buffer_capacity_factor=self.config.capacity_factor,
            gate_fallback_mode=self.config.gate_fallback_mode,
            routed_expert_activations_dtype=self.config.routed_expert_activations_dtype,
            routed_expert_weights_dtype=self.config.routed_expert_weights_dtype,
            shared_expert_activations_dtype=self.config.shared_expert_activations_dtype,
            shared_expert_weights_dtype=self.config.shared_expert_weights_dtype,
            weight_cache_path=self.config.weight_cache_path,
            lm_head_is_column_parallel=True,
            is_chunked=True,
            slot_num=self.config.num_users,
            kv_only_last_layer=self.config.kv_only_last_layer,
            routing_use_l1_small_for_semaphores=self.config.routing_use_l1_small_for_semaphores,
            first_layer_idx=self.config.first_layer_idx,
            is_first_rank=self.config.is_first_rank,
            is_last_rank=self.config.is_last_rank,
        )
        self.model_built = True

    def make_placeholder_activation(self) -> ttnn.Tensor:
        """Allocate a zero hidden-state activation matching the embedding output:
        [1, 1, chunk_per_chip, emb_dim/tp], TILE_LAYOUT, DRAM, replicated.

        Stand-in input for a non-first rank until the upstream D2D-socket sync op
        delivers the real activation. The first block's attn_norm reads from this
        tensor; once the sync op lands, the wait-op overwrites it in place.
        """
        chunk_per_chip = self.config.chunk_size // self.config.sp_factor
        emb_per_tp = self.hf_config.hidden_size // self.config.tp_factor
        zeros = torch.zeros(1, 1, chunk_per_chip, emb_per_tp, dtype=torch.bfloat16)
        return ttnn.from_torch(
            zeros,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def make_chunk_input(self, token_ids: list[int]) -> ttnn.Tensor:
        """Build one chunk's device input for `prefill_chunk`. First-rank input is
        SP-sharded token IDs; a non-first pipeline rank instead gets a placeholder
        hidden-state activation (it does not embed — it receives the real activation
        over the D2D socket)."""
        if self.config.is_first_rank:
            return prepare_prefill_input_tensor(
                token_ids,
                self.mesh_device,
                self.config.sp_factor,
                False,  # chunked prefill is block-cyclic (non-balanced)
                self.config.mesh_shape,
                self.config.sp_axis,
            )
        return self.make_placeholder_activation()

    def compile(self, kv_caches: KvCaches) -> None:
        """Warm up one chunk so the per-chunk loop hits no first-run cost. The engine passes the
        `KvCaches` tuple it owns; the warm-up writes into them (slot 0) and is harmless. The runtime
        holds NO cache state — the same `kv_caches` is passed back into every prefill_chunk."""
        assert self.model_built
        chunk = self.config.chunk_size
        logger.info(f"TtPrefillRuntime.compile() — warming up one {chunk}-token chunk")
        t0 = time.perf_counter()
        tt_input = self.make_chunk_input([0] * chunk)
        self.prefill_chunk(tt_input, kv_caches, slot_id=0, actual_start=0, actual_end=chunk)
        ttnn.synchronize_device(self.mesh_device)
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"[prefill timing] task_id=WARMUP num_tokens={chunk} runtime.prefill_chunk(chunk) = {warmup_ms:.2f} ms"
        )
        self.compiled = True

    def prefill_chunk(
        self,
        input_tensor: ttnn.Tensor,
        kv_caches: KvCaches,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        request_id: int = 0,
        d2h_service=None,
        record_dev: Optional[ttnn.Tensor] = None,
    ) -> Optional[ttnn.Tensor]:
        """Prefill ONE chunk into user `slot_id`'s slice of the engine-owned `kv_caches`.

        On the last rank (and single-rank) this returns None — the populated cache is
        the output (read by the decode stage / migration consumer). On a non-last
        pipeline rank it returns this rank's output hidden-state activation, which the
        driver hands to the next rank (today via a placeholder; via a D2D-socket
        publish op once that lands).

        [actual_start, actual_end) is the absolute KV-position range of this chunk's real (non-pad)
        tokens: actual_start is the cache write offset (cumulative valid KV before this chunk) and
        actual_end - actual_start is the real-token count in the chunk (the tail of the last chunk
        may be pad, so actual_end < actual_start + chunk_size). actual_end is the migration pad-zero
        boundary, passed straight through to MLA. The caller drives chunked prefill by
        calling this once per chunk, in order; a chunk's KV must be populated before the next reads
        it. If d2h_service + record_dev are passed, the model sends one per-layer ack completion signal back
        to host (via the outbound_socket_service_sync device op) once each layer's KV cache is populated.

        Always returns None: no token is sampled. (When `kv_only_last_layer` is set on the config the
        last layer's compute is stripped down to the KV cache fill, which migration consumes, and the
        final RMSNorm / LM head / sample are skipped entirely.)

        Args:
            input_tensor: on the first rank, one chunk's tokens as an SP-sharded uint32 ROW_MAJOR DRAM
                tensor (prepare_prefill_input_tensor, block-cyclic, chip-major); on a non-first rank,
                the upstream hidden-state activation. Deallocated here.
            kv_caches: the engine-owned KV cache(s) (from the adapter's allocate_kv_cache), an ordered
                tuple — index 0 the primary KV cache this chunk's KV is written into, index 1+ any
                secondary caches (a sparse/DSA model's index cache at 1). The same tuple is passed on
                every call; the runtime holds none of it.
            slot_id: cache user slot to fill, in [0, num_users).
            actual_start: absolute KV pos of the chunk's first real token (the cache write offset).
            actual_end: absolute KV pos past the chunk's last real token.
            d2h_service: optional service used to send a layer-ack completion signal back to host once
                each layer's KV cache has been populated on device. When set, each block zeros the cache
                pad window and enqueues the ack via the outbound_socket_service_sync device op on the same
                CQ (no host sync). When None, no ack or zeroing.
            record_dev: the chunk's PrefillMetadata device tensor sent as each ack record; required when
                d2h_service is set.
        """
        # Not gated on self.compiled: compile() warms up by calling prefill_chunk() once before
        # marking the runtime compiled. The model must exist, though.
        assert self.model_built, "build the model before prefill_chunk()"
        assert 0 <= slot_id < self.config.num_users, f"slot_id {slot_id} out of range [0, {self.config.num_users})"
        assert (
            actual_start + self.config.chunk_size <= self.config.max_seq_len
        ), f"chunk at actual_start={actual_start} exceeds per-user cache {self.config.max_seq_len}"
        assert (
            actual_start <= actual_end <= actual_start + self.config.chunk_size
        ), f"[actual_start={actual_start}, actual_end={actual_end}) not within one chunk of {self.config.chunk_size}"

        # Bind this chunk's request_id into a fresh per-call callback. The pipelined sink needs it to
        # build a globally-dense key (seq = request_id*num_layers + layer_idx); capturing by value per
        # call means there is no shared mutable chunk-index for the synchronously-fired callback to race
        # on. Single-host layer-ack mode ignores request_id.
        if self._layer_completion_sink is not None:
            sink = self._layer_completion_sink

            def on_layer_complete(layer_idx: int) -> None:
                sink(layer_idx, request_id)

        else:
            on_layer_complete = self._on_layer_complete

        out = self.model.forward(
            input_tensor,
            kv_caches[0],
            actual_isl=actual_end - actual_start,
            on_layer_complete=on_layer_complete,
            d2h_service=d2h_service,
            record_dev=record_dev,
            actual_start=actual_start,
            actual_end=actual_end,
            cache_user_id=slot_id,
            index_kv_cache=kv_caches[1] if len(kv_caches) > 1 else None,
        )
        ttnn.deallocate(input_tensor)
        # Non-last rank: forward returns the hidden-state activation to forward downstream.
        # Last/single rank: forward returns the (token, prob, intermediates) tuple, which this
        # KV-output path ignores.
        return out if not self.config.is_last_rank else None

    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        """Register the per-layer-ack channel (docs/scheduler/prefill.md §3.11).

        `layer_ack_channel` is a `ttnn.InterProcessCounterChannel` on
        `/tt_prefill_layer_acks_<service_id>`. The runner bumps it once per
        layer (`inject(1)`); the scheduler reads the delta and drives the
        migration worker. The ack carries no payload — the scheduler correlates
        acks with the chunk it pushed (its InFlightChunkFIFO).

        Per-layer cadence means NUM_LAYERS acks per chunk, so the scheduler must
        be configured with layers_per_chunk == NUM_LAYERS.
        """
        assert self.compiled, "Call compile() before set_layer_ack_channel()"

        def on_layer_complete(layer_idx: int) -> None:
            layer_ack_channel.inject(1)

        self._on_layer_complete = on_layer_complete

    def build_kv_chunk_table(
        self,
        kv_caches: KvCaches,
        path: str,
        *,
        first_layer_idx: int = 0,
        num_my_layers: Optional[int] = None,
        stage_layout=None,
    ) -> str:
        """Build + serialize the KV-chunk address table for the engine-owned `KvCaches` to
        `path` and return it. Model-specific: DeepSeek/Kimi's MLA block-cyclic cache layout.

        The table maps each natural KV position to its true block-cyclic storage chip + offset
        (the MLA chunked-prefill cache layout), so the migration worker copies the right chunks.
        The runner publishes the serialized table to the worker — this method only describes the
        cache layout; it issues no migration comms. The generic config-setup + protobuf serialize
        lives in ``serialize_kv_chunk_table`` (model-agnostic); this method supplies the model's
        block-cyclic builder + chunk constants.

        Multi-rank (pipeline-parallel): this rank owns layers [first_layer_idx, first_layer_idx +
        num_my_layers). The runner runs the all-ranks all-gather and passes the merged `stage_layout`
        so ONLY rank 0 builds the table spanning every stage; the single-rank default (stage_layout
        None) covers config.num_layers == the full model.

        Describes the primary KVPE cache (`kv_caches[0]`) only; the single-config builder does not
        emit a sparse model's secondary index-key cache (config 1).

        ``config.chunk_size`` is the block-cyclic period; the kimi builder hardcodes it as
        PREFILL_CHUNK_OUTPUT_TOKENS, so a non-default period is rejected here rather than mismapped."""
        from models.demos.common.prefill.runners.migration import serialize_kv_chunk_table
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
            NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
            PREFILL_CHUNK_OUTPUT_TOKENS,
            _dram_chunk_size_bytes,
            create_kv_chunk_address_table_kimi,
        )

        # Per-DRAM-bank chunk size = one NdShardSpec shard ([32, head_dim]) in the cache's real
        # dtype/layout (matches main's _dram_chunk_size_bytes). Used as BOTH the per-bank read size and
        # the per-round DRAM stride in create_kv_chunk_address_table_kimi, so it MUST equal the physical
        # bank size: dense MLA kvpe is bf8_b/TILE (19584 B); sparse-DSA (GLM) kvpe is bf16/ROW_MAJOR
        # (36864 B). Deriving from kv_caches[0] fixes the sparse cache — a hardcoded dense size truncated
        # reads and mis-strided every bank (-> garbage / out-of-bounds).
        chunk_size_bytes = _dram_chunk_size_bytes(kv_caches[0])
        assert self.config.chunk_size == PREFILL_CHUNK_OUTPUT_TOKENS, (
            f"create_kv_chunk_address_table_kimi assumes a block-cyclic period of "
            f"PREFILL_CHUNK_OUTPUT_TOKENS={PREFILL_CHUNK_OUTPUT_TOKENS}, but config.chunk_size="
            f"{self.config.chunk_size}. A different period would mismap every position; re-introduce a "
            f"parametrized builder if needed."
        )

        def _builder(*, config, chunk_size_bytes, num_users):
            return create_kv_chunk_address_table_kimi(
                config=config,
                mesh_device=self.mesh_device,
                mesh_shape=self.config.mesh_shape,
                seq_len=self.config.max_seq_len,
                sp_axis=self.config.sp_axis,
                tt_kvpe_cache=kv_caches[0],
                chunk_size_bytes=chunk_size_bytes,
                num_users=num_users,
                first_layer_idx=first_layer_idx,
                num_my_layers=num_my_layers,
                stage_layout=stage_layout,
            )

        return serialize_kv_chunk_table(
            table_builder=_builder,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len,
            num_users=self.config.num_users,
            chunk_n_tokens=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
            chunk_size_bytes=chunk_size_bytes,
            path=path,
        )

    def kv_cache_pcc_check(
        self, kv_caches: KvCaches, *, slot_id: int, n_chunks: int, trace_dir=None, first_layer_idx: int = 0
    ) -> float:
        """Optional bring-up hook (not part of the core runtime contract; never called in production
        serving). PCC the populated engine-owned primary KV cache (`kv_caches[0]`) for `slot_id`
        against the golden trace; returns the min per-layer PCC and asserts on failure (unless
        PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1). Thin forwarder into the model's validation module."""
        from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import kv_cache_pcc_check

        return kv_cache_pcc_check(
            self, kv_caches[0], slot_id=slot_id, n_chunks=n_chunks, trace_dir=trace_dir, first_layer_idx=first_layer_idx
        )

    def set_layer_completion_sink(self, sink) -> None:
        """Register a per-layer completion sink for pipelined prefill.

        `sink` is called once per layer as `sink(layer_idx, request_id)` — the
        global layer index plus the current request/chunk id, which prefill()
        binds per call (so the sink need not read any mutable runtime state). It
        replaces the direct counter-channel inject used in single-host mode:
        instead of bumping a counter, the runner pushes a full completion
        {seq, source_rank, layer_idx, request_id} into the host-local
        LayerCompletionQueue, and the LayerCompletionRouter routes it to the
        master host and re-emits it (in seq order) into the scheduler-facing
        counter channel (see runners/pipelined_prefill/).
        """
        assert self.compiled, "Call compile() before set_layer_completion_sink()"
        self._layer_completion_sink = sink
