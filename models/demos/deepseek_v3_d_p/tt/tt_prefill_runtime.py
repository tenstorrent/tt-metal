# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import DFlashDrafterConfig
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.tt_dflash_drafter import TtDFlashDrafter
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.utils import load_drafter_state_dict
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import MlaKvCaches
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, allocate_dflash_kv_cache


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
    # Scalar applies to both mesh axes; a (sp_axis_0, tp_axis_1) tuple configures each independently.
    # Derived from the opened fabric via tt_ccl.per_axis_topology() in the runner.
    topology: Union[ttnn.Topology, Tuple[ttnn.Topology, ttnn.Topology]] = ttnn.Topology.Linear
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
    # Build the DFlash drafter context-KV cache during this prefill (opt-in). Every rank builds its owned fc
    # slices from $DFLASH_HF_MODEL; only the last rank builds the KV tail + cache.
    dflash_enabled: bool = False
    # Pipeline-parallel rank slicing. first_layer_idx is the global index of this
    # rank's first layer; is_first_rank gates the embedding, is_last_rank marks the
    # final stage (non-last ranks forward the hidden state instead of running a tail).
    # Defaults make a single-rank runtime own the whole model.
    first_layer_idx: int = 0
    is_first_rank: bool = True
    is_last_rank: bool = True
    sparse_kv_cache_format: MlaKvCacheFormat = MlaKvCacheFormat.BF16_RM

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
        # DFlash drafter, built in _build_model when config.dflash_enabled (else all None and prefill_chunk's
        # dflash branches are inert). Its tap closure + last-rank K/V caches are set there.
        self._on_layer_hidden = None
        self.drafter = None
        self._dflash_k_cache = None
        self._dflash_v_cache = None

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
            sparse_kv_cache_format=self.config.sparse_kv_cache_format,
        )
        self.model_built = True

        if self.config.dflash_enabled:
            self._build_dflash_drafter()

    def _build_dflash_drafter(self) -> None:
        """Build this rank's DFlash speculative-drafter (issue #49586) when ``config.dflash_enabled``.

        Each rank taps only the target layers it owns; the last rank also builds the KV tail and allocates
        the caller-owned context K/V caches. Checkpoint (config + weights) comes from ``$DFLASH_HF_MODEL``."""
        path = os.environ.get("DFLASH_HF_MODEL")
        assert path, (
            "PREFILL_DFLASH=1 requires DFLASH_HF_MODEL=/path/to/Kimi-K2.x-DFlash "
            "(a dir with config.json + model.safetensors)"
        )
        dcfg = DFlashDrafterConfig.from_pretrained(path)

        first = self.config.first_layer_idx
        last_excl = first + self.config.num_layers
        owned = tuple(t for t in dcfg.target_layer_ids if first <= t < last_excl)
        # A kv-only last layer returns before its post-FFN tap fires, so a target layer at that index would
        # be silently dropped. Kimi is safe (targets <= 58); assert so a future layout can't regress it.
        if self.config.kv_only_last_layer and self.config.is_last_rank and self.config.num_layers > 0:
            kv_only_idx = last_excl - 1
            assert kv_only_idx not in dcfg.target_layer_ids, (
                f"drafter target layer {kv_only_idx} coincides with the kv-only last layer; its post-FFN tap "
                f"never fires. Move the tap off the last layer or disable PREFILL_KV_ONLY_LAST_LAYER."
            )

        logger.info(
            f"Building DFlash drafter: owned_target_layers={owned} of {tuple(dcfg.target_layer_ids)}, "
            f"build_kv_tail={self.config.is_last_rank}, "
            f"checkpoint={path}"
        )
        state_dict = load_drafter_state_dict(path, build_kv_tail=self.config.is_last_rank)
        # The drafter's context cache and rope table span the FULL per-user sequence, like the verifier's
        # kvpe cache: chunked prefill writes chunk c at global offset actual_start, and multi-turn resumes a
        # slot mid-sequence, so neither can be expressed by a chunk_size-deep cache (issue #50725).
        dflash_seq = self.config.max_seq_len
        self.drafter = TtDFlashDrafter(
            self.mesh_device,
            dcfg,
            state_dict=state_dict,
            sp_axis=self.config.sp_axis,
            tp_axis=self.config.tp_axis,
            max_seq_len=dflash_seq,
            num_links=self.config.num_links,
            topology=self.config.topology,
            owned_target_layer_ids=owned,
            build_kv_tail=self.config.is_last_rank,
        )

        # on_layer_hidden fires post-FFN with the sharded residual h; tap only owned target layers. No clone:
        # tap only READS h into the FC matmul (must not free/mutate it) and does not retain it.
        owned_set = set(owned)

        def on_layer_hidden(global_idx: int, h: ttnn.Tensor) -> None:
            if global_idx not in owned_set:
                return
            self.drafter.tap(h, global_idx)

        self._on_layer_hidden = on_layer_hidden

        # Only the last rank finalizes the drafter KV → allocate the caller-owned context caches it fills.
        if self.config.is_last_rank:
            self._dflash_k_cache, self._dflash_v_cache = allocate_dflash_kv_cache(
                self.mesh_device,
                dcfg,
                dflash_seq,  # max_seq_len — MUST match the drafter's cache_seq/rope (see note above)
                sp_axis=self.config.sp_axis,
                tp_axis=self.config.tp_axis,
                num_users=self.config.num_users,  # user-major slots, like the verifier's kvpe cache
            )

    def _pack_activation(self, hidden: ttnn.Tensor, partial: ttnn.Tensor) -> ttnn.Tensor:
        """Fuse this rank's output hidden and finalized drafter FC partial into ONE activation for the D2D
        handoff — per-chip ``[1,1,chunk/sp,H/tp]`` → ``[1,1,chunk/sp,2H/tp]`` (the runner widens the D2D
        activation spec to 2H when dflash is on). Consumes both inputs; only a non-last rank packs."""
        packed = ttnn.concat([hidden, partial], dim=3)
        ttnn.deallocate(hidden)
        ttnn.deallocate(partial)
        return packed

    def _unpack_activation(self, packed: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Inverse of :meth:`_pack_activation`: split the received per-chip ``[1,1,chunk/sp,2H/tp]`` into the
        hidden fed to the verifier and the drafter partial imported into the drafter (each
        ``[1,1,chunk/sp,H/tp]``). Only a non-first rank unpacks."""
        half = packed.shape[-1] // 2
        s0, s1, s2, s3 = packed.shape
        hidden = ttnn.slice(packed, [0, 0, 0, 0], [s0, s1, s2, half])
        partial = ttnn.slice(packed, [0, 0, 0, half], [s0, s1, s2, s3])
        return hidden, partial

    def make_placeholder_activation(self) -> ttnn.Tensor:
        """Allocate a zero hidden-state activation matching what the D2D socket delivers:
        [1, 1, chunk_per_chip, emb_dim/tp] — or 2·emb_dim/tp under DFlash, which packs the drafter
        partial alongside the hidden — TILE_LAYOUT, DRAM, replicated.

        Stand-in input for a non-first rank until the upstream D2D-socket sync op
        delivers the real activation. The first block's attn_norm reads from this
        tensor; once the sync op lands, the wait-op overwrites it in place. Under DFlash the
        delivered tensor is the packed [hidden ‖ partial]; prefill_chunk unpacks it before the model runs.
        """
        chunk_per_chip = self.config.chunk_size // self.config.sp_factor
        # DFlash packs [hidden ‖ drafter-partial] into the D2D activation, so a non-first rank receives a
        # 2H-wide tensor and this receive buffer must match. Non-dflash keeps H, byte-identical.
        feature_size = self.hf_config.hidden_size * (2 if self.config.dflash_enabled else 1)
        emb_per_tp = feature_size // self.config.tp_factor
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

    def compile(self, kv_caches: MlaKvCaches) -> None:
        """Warm up one chunk so the per-chunk loop hits no first-run cost. The engine passes the
        `MlaKvCaches` it owns; the warm-up writes into it (slot 0) and is harmless. The runtime holds NO
        cache state — the same `kv_caches` is passed back into every prefill_chunk."""
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
        kv_caches: MlaKvCaches,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        request_id: int = 0,
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
        it. If a LayerAck channel is registered (set_layer_ack_channel), the model bumps it per layer.

        Always returns None: no token is sampled. (When `kv_only_last_layer` is set on the config the
        last layer's compute is stripped down to the KV cache fill, which migration consumes, and the
        final RMSNorm / LM head / sample are skipped entirely.)

        Args:
            input_tensor: on the first rank, one chunk's tokens as an SP-sharded uint32 ROW_MAJOR DRAM
                tensor (prepare_prefill_input_tensor, block-cyclic, chip-major); on a non-first rank,
                the upstream hidden-state activation. Deallocated here.
            kv_caches: the engine-owned KV cache (from the adapter's allocate_kv_cache): ``.kvpe`` is the
                primary cache this chunk's KV is written into; ``.index`` is the sparse/DSA indexer cache
                (None for dense). The same object is passed on every call; the runtime holds none of it.
            slot_id: cache user slot to fill, in [0, num_users).
            actual_start: absolute KV pos of the chunk's first real token (the cache write offset).
            actual_end: absolute KV pos past the chunk's last real token.
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

        # DFlash: reset the per-chunk FC accumulator; a non-first rank also unpacks the drafter partial the
        # previous rank packed with the hidden and imports it. First/single rank's input (token IDs) is unchanged.
        model_input = input_tensor
        if self.config.dflash_enabled:
            self.drafter.reset()
            if not self.config.is_first_rank:
                model_input, partial = self._unpack_activation(input_tensor)
                ttnn.deallocate(input_tensor)
                self.drafter.import_partial(partial)

        out = self.model.forward(
            model_input,
            kv_caches.kvpe,
            actual_isl=actual_end - actual_start,
            on_layer_complete=on_layer_complete,
            on_layer_hidden=self._on_layer_hidden,
            actual_start=actual_start,
            actual_end=actual_end,
            cache_user_id=slot_id,
            index_kv_cache=kv_caches.index,
        )
        ttnn.deallocate(model_input)

        if self.config.dflash_enabled:
            if self.config.is_last_rank:
                # Finalize the drafter context-KV into the runtime-owned caches (read back for PCC /
                # migration). Same (offset, slot) the verifier's kvpe write uses, so the drafter cache stays
                # positionally aligned with it across chunks and users.
                self.drafter.write_kv_cache(
                    self._dflash_k_cache,
                    self._dflash_v_cache,
                    actual_start,
                    slot_idx=slot_id,
                )
                return None
            # Non-last rank: pack this rank's finalized FC partial alongside the hidden for the next rank.
            return self._pack_activation(out, self.drafter.export_partial())

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

    def kv_migration_base_address(self, kv_caches: MlaKvCaches) -> int:
        """This stage's KV base DRAM address — the engine's per-rank anchor for the migration
        all-gather (it holds the cache but must not introspect its layout). The pipeline-parallel
        path migrates the primary KVPE cache, so that is the base; `.kvpe` is an MlaKvCache wrapper
        rather than a bare tensor, hence `.storage`."""
        return int(kv_caches.kvpe.storage.buffer_address())

    def build_kv_chunk_table(
        self,
        kv_caches: MlaKvCaches,
        path: str,
        *,
        first_layer_idx: int = 0,
        num_my_layers: Optional[int] = None,
        stage_layout=None,
    ) -> str:
        """Build + serialize the KV-chunk address table for the engine-owned `MlaKvCaches` to
        `path` and return it.

        The table maps each natural KV position to its true block-cyclic storage chip + offset
        (the MLA chunked-prefill cache layout), so the migration worker copies the right chunks.
        The runner publishes the serialized table to the worker — this method only describes the
        cache layout; it issues no migration comms.

        Multi-rank (pipeline-parallel): this rank owns layers [first_layer_idx, first_layer_idx +
        num_my_layers). The runner runs the all-ranks all-gather and passes the merged `stage_layout`
        so ONLY rank 0 builds the table spanning every stage; the single-rank default (stage_layout
        None) covers config.num_layers == the full model.

        For a sparse/DSA model (``.index`` present) the result is a single MERGED table describing BOTH
        caches — config 0 = the KVPE cache, config 1 = the index-key cache. A dense model (``.index`` None)
        → the usual single-config table over the KVPE cache alone. The index-cache merge is single-rank
        only; the pipeline-parallel path (stage_layout given) migrates the KVPE cache alone."""
        from models.demos.deepseek_v3_d_p.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        # PP path migrates the primary (KVPE) cache only; the sparse/DSA index cache isn't wired
        # through the cross-stage merge yet (port it into _build_and_serialize_merged_kv_chunk_table
        # to add it) — fail loudly rather than silently drop it.
        if stage_layout is not None:
            assert kv_caches.index is None, (
                "build_kv_chunk_table: index-cache (sparse/DSA) migration is not supported on the "
                "pipeline-parallel path yet."
            )

        return build_and_serialize_kv_chunk_table(
            mesh_device=self.mesh_device,
            kvpe_cache=kv_caches.kvpe,
            seq_len=self.config.max_seq_len,
            num_layers=self.config.num_layers,
            mesh_shape=self.config.mesh_shape,
            sp_axis=self.config.sp_axis,
            num_users=self.config.num_users,
            chunk_size_global=self.config.chunk_size,  # block-cyclic period (prefill chunk size)
            path=path,
            index_kv_cache=kv_caches.index,
            first_layer_idx=first_layer_idx,
            num_my_layers=num_my_layers,
            stage_layout=stage_layout,
        )

    def read_slot_kv(self, kv_caches: MlaKvCaches, slot: int):
        """Read one slot's KV cache from device to host: the `.kvpe` block as a single host tensor
        ``[num_layers, 1, seq_cache, kvpe]`` (one TP replica), in the raw on-device (block-cyclic) layout —
        not un-rotated to natural token order. DRAM_MEMORY_CONFIG on the slice is REQUIRED — the cache is
        ND-sharded ROUND_ROBIN_1D, and slicing into another ND-shard miscomputes the DRAM core on host
        read-back."""
        mesh_device = self.mesh_device
        num_layers = self.config.num_layers
        # `.kvpe` is an MlaKvCache wrapper, NOT a bare tensor: physical ops use `.storage`, and physical
        # rows may be packed (SCALED_FP8), so decode them with `unpack_host` to logical [latent || RoPE] —
        # the same path kv_cache_pcc_check takes. (Using `kvpe` directly here raised
        # 'MlaKvCache' object has no attribute 'shape'.)
        kvpe = kv_caches.kvpe
        storage = kvpe.storage
        s = list(storage.shape)
        sl = ttnn.slice(
            storage,
            [slot * num_layers, 0, 0, 0],
            [(slot + 1) * num_layers, s[1], s[2], s[3]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        physical = ttnn.to_torch(
            sl, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
        )[
            :, :1
        ]  # one TP replica: [num_layers, 1, seq_cache, packed_row]
        ttnn.deallocate(sl)
        block = kvpe.unpack_host(physical).to(torch.float32)  # [num_layers, 1, seq_cache, kvpe_logical]
        return [block]

    def kv_cache_pcc_check(
        self, kv_caches: MlaKvCaches, *, slot_id: int, n_chunks: int, trace_dir=None, first_layer_idx: int = 0
    ) -> float:
        """Optional bring-up hook (not part of the core runtime contract; never called in production
        serving). PCC the populated engine-owned primary KV cache (`.kvpe`) for `slot_id` against the
        golden trace; returns the min per-layer PCC and asserts on failure (unless
        PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1). Thin forwarder into the model's validation module."""
        from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import kv_cache_pcc_check

        return kv_cache_pcc_check(
            self,
            kv_caches.kvpe,
            slot_id=slot_id,
            n_chunks=n_chunks,
            trace_dir=trace_dir,
            first_layer_idx=first_layer_idx,
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
        counter channel (see ttnn._experimental.layer_completion).
        """
        assert self.compiled, "Call compile() before set_layer_completion_sink()"
        self._layer_completion_sink = sink
