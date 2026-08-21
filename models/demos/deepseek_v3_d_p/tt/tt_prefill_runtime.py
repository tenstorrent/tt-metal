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
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_OUTPUT_TOKENS
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, allocate_dflash_kv_cache
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController


@dataclass
class TtPrefillRuntimeConfig:
    num_layers: int  # layers built by THIS runtime (the rank's slice; == model total for single-rank)
    max_seq_len: int  # per-user KV-cache length (tokens), e.g. 60 * 1024
    mesh_shape: tuple = (32, 4)
    # Chunked prefill streams tokens in chunks of `chunk_size`, with `num_users` independent cache
    # slots (user-major batch). The full cache holds num_users * num_layers slots of max_seq_len each.
    chunk_size: int = PREFILL_CHUNK_OUTPUT_TOKENS
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
    # Trace-safe metadata prefill: capture the per-chunk forward ONCE as a (segmented) ttnn trace during
    # compile(), then replay it every chunk — advancing the per-chunk scalars (slot_id, actual_start,
    # actual_end) on-device via an in-place host update of a persistent per-element metadata tensor, so the
    # captured command stream carries no host transfers. Collapses the per-op host-dispatch (op2op) gaps.
    # Requires the mesh opened with trace_region_size > 0. Off by default (eager per-op dispatch).
    use_trace: bool = False
    # MoE shared-expert ∥ dispatch overlap. Keeps the optimization ON by default, but it loads/clears a
    # 2-sub-device manager around each MoE layer — which forces the segmented trace to split there, adding
    # ~2*(MoE layers) host load/clear round-trips per replay. Set False (PREFILL_OVERLAP_SHARED_EXPERT=0) to
    # capture the forward as ONE trace segment (no per-chunk swaps -> faster replay); costs the overlap.
    overlap_shared_expert_with_dispatch: bool = True

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

        # Trace-safe metadata prefill state (config.use_trace). compile() allocates the buffers and
        # warm-compiles the metadata programs; the capture itself is recorded later by capture_trace(),
        # after the driver has built any D2D endpoints and registered the per-layer ack/completion
        # callback. Everything happens before the first request; there is no lazy or repeat capture.
        #   _controller       — SubDeviceTraceController driving the segmented capture/replay
        #   _trace_input      — persistent per-chunk input buffer (captured address; updated in place)
        #   _trace_metadata   — 3 persistent 1-element uint32 tensors (slot_id, actual_start, actual_end)
        #   _trace_output     — persistent output activation (non-last rank only; read by the D2D send)
        #   _trace_captured   — flips True once capture_trace() records the segmented capture (single-shot)
        #   _kv_cache         — the engine cache handle from compile(), used by capture_trace()
        #   _trace_request_id — the request/chunk id of the replay in flight. The controller's per-layer
        #                       callback is fixed at capture time, so a traced run cannot bind request_id
        #                       into a fresh closure per call (the eager path does); prefill_chunk()
        #                       publishes it here and set_layer_completion_sink()'s callback reads it.
        self._controller = None
        self._trace_input = None
        self._trace_metadata = None
        self._trace_output = None
        self._trace_captured = False
        self._kv_cache = None
        self._trace_request_id = 0

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
                # Required for a LatentMoE model (Kimi-K3): without it the per-block check cannot know
                # to look for the latent-projection cache files and would call an incomplete cache
                # complete. model_cfg is already in hand two lines up.
                model_cfg=model_cfg,
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
            overlap_shared_expert_with_dispatch=self.config.overlap_shared_expert_with_dispatch,
        )
        self.model_built = True

        if self.config.dflash_enabled:
            self._build_dflash_drafter()

    def _build_dflash_drafter(self) -> None:
        """Build this rank's DFlash speculative-drafter when ``config.dflash_enabled``.

        Each rank taps only the target layers it owns; the last rank also builds the KV tail and allocates
        the caller-owned context K/V caches. Checkpoint (config + weights) comes from ``$DFLASH_HF_MODEL``."""
        path = os.environ.get("DFLASH_HF_MODEL")
        assert path, (
            "DFlash drafter build requires DFLASH_HF_MODEL=/path/to/Kimi-K2.x-DFlash "
            "(a dir with config.json + model.safetensors)"
        )
        dcfg = DFlashDrafterConfig.from_pretrained(path)
        # The adapter gates which MODEL may run DFlash (ADAPTER.supports_dflash, checked in the runner); this
        # gates which DRAFTER checkpoint may attach to it. Sibling drafters exist for other parents, and a
        # mismatched one is dimensionally plausible enough to build and produce meaningless KV, so check the
        # drafter's own declaration of its target against the verifier actually loaded here.
        assert dcfg.num_target_layers, (
            f"drafter checkpoint at {path} declares no `num_target_layers` in its config.json, so which "
            "verifier it was trained against cannot be verified (the Kimi and DeepSeek-V3 families are "
            "dimensionally identical, 61 x 7168, so hidden_size alone proves nothing). Add the key, or "
            "point DFLASH_HF_MODEL at a checkpoint that declares it."
        )
        assert (dcfg.num_target_layers, dcfg.hidden_size) == (
            self.hf_config.num_hidden_layers,
            self.hf_config.hidden_size,
        ), (
            f"drafter checkpoint at {path} targets {dcfg.num_target_layers} layers x hidden "
            f"{dcfg.hidden_size}, but this verifier is {self.hf_config.num_hidden_layers} x "
            f"{self.hf_config.hidden_size}. Wrong DFLASH_HF_MODEL for this model."
        )

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
            chunk_size=self.config.chunk_size,
            num_links=self.config.num_links,
            topology=self.config.topology,
            owned_target_layer_ids=owned,
            build_kv_tail=self.config.is_last_rank,
        )

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
        # 2H-wide tensor and this receive buffer must match. Non-dflash keeps H.
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
        cache state — the same `kv_caches` is passed back into every prefill_chunk.

        use_trace: set up the persistent buffers + controller and warm-compile the metadata programs here,
        but do NOT record the capture yet — the driver calls capture_trace() AFTER any pipeline D2D
        endpoints are built and after set_layer_ack_channel(); prefill_chunk() then only replays."""
        assert self.model_built
        chunk = self.config.chunk_size
        t0 = time.perf_counter()
        if self.config.use_trace:
            # Cache-level companion to the config-level guard in TtPrefillTransformer.set_trace_controller
            # (which rejects a sparse/DSA model outright). Checked separately because it catches the other
            # direction: a sparse INDEX cache handed to a model that resolved as dense. _forward_traced
            # never threads index_kv_cache, so such a run would replay without the indexer cache and
            # produce wrong KV silently instead of failing.
            assert kv_caches.index is None, (
                "use_trace=True with a sparse/DSA index KV cache is not supported: the captured forward "
                "does not thread index_kv_cache, so the indexer would be skipped silently. Supported "
                "traced models are the dense-MLA ones (deepseek_v3, kimi_k2_6, kimi_k2_7); GLM and other "
                "sparse variants need their indexer ops ported to the metadata form first — run them "
                "with use_trace=False (PREFILL_USE_TRACE=0) until then."
            )
            logger.info(
                f"TtPrefillRuntime.compile() — warming traced {chunk}-token chunk (metadata path); capture deferred to capture_trace()"
            )
            self._kv_cache = kv_caches  # kept so capture_trace() can record after the ack is registered
            self._prepare_trace(kv_caches)
        else:
            logger.info(f"TtPrefillRuntime.compile() — warming up one {chunk}-token chunk")
            tt_input = self.make_chunk_input([0] * chunk)
            self.prefill_chunk(tt_input, kv_caches, slot_id=0, actual_start=0, actual_end=chunk)
            ttnn.synchronize_device(self.mesh_device)
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"[prefill timing] task_id={'PREPARE' if self.config.use_trace else 'WARMUP'} num_tokens={chunk} "
            f"runtime.compile() = {warmup_ms:.2f} ms"
        )
        self.compiled = True

    def capture_trace(self, kv_cache: ttnn.Tensor) -> None:
        """Record the segmented trace, ONCE, before the chunk loop opens.

        The driver must call this AFTER building any D2D pipeline endpoints (their receiver-socket L1 must
        be allocated first, or it lands on the captured trace buffers on the last rank and corrupts replay)
        and AFTER set_layer_ack_channel() / set_layer_completion_sink() (the ack callback has to be known
        here so the capture splits at each ack point — a host shm bump cannot live inside a trace).
        No-op if not use_trace or already captured. See compile().

        The SubDeviceTraceController chops the capture at the MoE sub-device swaps (and, with an ack
        callback registered, at each migration ack). On a non-last rank the captured forward's output
        activation stays at its captured address in _trace_output so the driver can read it (and forward it
        over D2D) after every replay."""
        if not self.config.use_trace or self._trace_captured:
            return
        controller = self._controller
        assert controller is not None, "capture_trace(): compile() must run first (prepares the trace)"

        # Ack registered after compile: the block runs the metadata zero_padded_kv_cache + ack routing
        # ONLY when on_layer_complete is set (tt_prefill_block.forward), so those programs were NOT
        # compiled by _prepare_trace's warm pass (which ran with on_layer_complete=None). Warm them now —
        # with a NO-OP ack so this warm pass fires no real migration acks — then register the real ack so
        # the capture splits at each ack point. No ack (standalone) => nothing extra to warm.
        if self._on_layer_complete is not None:
            controller.set_layer_ack_callback(lambda _layer_idx: None)
            self._forward_traced(kv_cache)  # compile zero_padded_kv_cache + ack path (no real ack fires)
            ttnn.synchronize_device(self.mesh_device)
            controller.set_layer_ack_callback(self._on_layer_complete)

        controller.begin_capture()
        out = self._forward_traced(kv_cache)
        controller.end_capture()
        ttnn.synchronize_device(self.mesh_device)
        # Non-last rank: the persistent output activation the replay refreshes each chunk.
        self._trace_output = out if not self.config.is_last_rank else None
        self._trace_captured = True
        logger.info(
            f"[trace] captured {self.config.num_layers}-layer chunk forward = {controller.num_segments} segments, "
            f"{controller.trace_bytes() / (1024 * 1024):.2f} MB"
        )

    def _meta1_dev(self, val: int) -> ttnn.Tensor:
        """One persistent 1-element uint32 replicated-DRAM metadata scalar (captured address)."""
        return ttnn.from_torch(
            torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _meta1_host(self, val: int) -> ttnn.Tensor:
        """Host-side 1-element uint32 tensor for the cheap in-place metadata update (copy_host_to_device)."""
        return ttnn.from_torch(
            torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _forward_traced(self, kv_cache: ttnn.Tensor):
        """The captured/warmed metadata forward: per-chunk scalars come from the persistent metadata
        tensor on-device (actual_start/actual_end = None host-side). Writes user slot metadata[0].
        Returns the forward output — a hidden-state activation on a non-last rank (forwarded downstream
        over D2D), or the last/single rank's ignored KV-only tuple."""
        return self.model.forward(
            self._trace_input,
            kv_cache.kvpe,  # unwrap the engine-owned container to the primary MLA cache (mirrors prefill_chunk)
            # FULL chunk on purpose: downstream (TtMoe.forward) uses actual_isl only as the
            # padding-config GUARD on this path — a static capture-time "padding awareness is ON" —
            # while the real per-chunk bound is read on-device from `metadata` (actual_start/
            # actual_end). Passing the true partial ISL here would bake one chunk's value into the
            # capture; passing None would capture a program with no padding-aware path at all.
            actual_isl=self.config.chunk_size,
            on_layer_complete=self._on_layer_complete,
            actual_start=None,
            actual_end=None,
            cache_user_id=0,
            metadata=self._trace_metadata,
        )

    def _prepare_trace(self, kv_cache: ttnn.Tensor) -> None:
        """Set up the persistent input + per-element metadata buffers and the controller, then warm-compile
        the metadata-variant programs (a full forward). Does NOT begin/end the capture — the driver calls
        capture_trace() later, once any ack/completion callback is registered. Called once from compile()."""
        chunk = self.config.chunk_size
        # Persistent input at a stable (captured) address; seeded with zeros, overwritten per chunk. On a
        # non-first rank make_chunk_input yields a placeholder hidden-state activation (the D2D-received one).
        self._trace_input = self.make_chunk_input([0] * chunk)
        # Per-element metadata: (slot_id, actual_start, actual_end), seeded for chunk 0.
        self._trace_metadata = (self._meta1_dev(0), self._meta1_dev(0), self._meta1_dev(chunk))

        controller = SubDeviceTraceController(self.mesh_device)
        self.model.set_trace_controller(controller)
        self._controller = controller

        self._forward_traced(kv_cache)  # warm/compile the metadata-variant programs
        ttnn.synchronize_device(self.mesh_device)

    def prefill_chunk(
        self,
        input_tensor: ttnn.Tensor,
        kv_caches: MlaKvCaches,
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
        Alternatively, if a host-side per-layer callback is registered (set_layer_ack_channel /
        set_layer_completion_sink), the model fires that once per layer instead.

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
            request_id: this chunk's request/chunk id. Only the pipelined layer-completion sink uses it
                (to build the globally-dense ordering key); the single-host layer-ack paths ignore it.
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

        if self.config.use_trace:
            # Traced path: update the persistent input + per-element metadata IN PLACE, then replay the
            # captured segmented forward. The metadata (slot_id, actual_start, actual_end) drives every
            # per-chunk scalar on-device, so the single capture fills the correct KV for this chunk.
            assert self._controller is not None, "use_trace: compile() must prepare the trace before prefill_chunk"
            # Capture must ALREADY have happened (capture_trace(), before the loop opens). This used to
            # fall back to capturing here, which silently moved a full warm pass + capture into the FIRST
            # request — a ~20s stall that is near-impossible to attribute from a latency graph. Everything
            # trace-related is required to be done before request #1, so make that a precondition.
            assert self._trace_captured, (
                "use_trace: capture_trace() must run before the first prefill_chunk(); capturing here "
                "would stall the first request by a warm pass + capture"
            )
            # The D2H layer-ack is not on the traced path: record_dev is the per-chunk socket metadata
            # tensor, whose address changes every chunk, so the capture would bake in a stale address.
            # Fail loudly rather than silently replaying a trace that emits no acks. (Wiring it up needs
            # record_dev to become a persistent, in-place-updated buffer like _trace_metadata.)
            assert d2h_service is None, (
                "use_trace does not support the D2H layer-ack path yet (record_dev is a per-chunk socket "
                "tensor, so its address cannot be captured); run with PREFILL_USE_TRACE=0 or use the "
                "host-callback ack (set_layer_ack_channel / set_layer_completion_sink)"
            )
            # Per-layer completion callbacks live on the CONTROLLER, registered once at capture time (a
            # host-side callback cannot execute inside a trace, so the capture splits at each ack point).
            # That means the pipelined sink's request_id cannot be re-bound per call the way the eager
            # path does below — publish this chunk's id instead; the captured callback built by
            # set_layer_completion_sink() reads it at replay time.
            self._trace_request_id = request_id
            ttnn.copy(input_tensor, self._trace_input)
            for dst, val in zip(self._trace_metadata, (slot_id, actual_start, actual_end)):
                ttnn.copy_host_to_device_tensor(self._meta1_host(val), dst)
            self._controller.replay()
            ttnn.deallocate(input_tensor)
            # Non-last rank: return the persistent output activation (replay just refreshed it) for the
            # driver to forward downstream over D2D. Last/single rank: the populated KV cache is the output.
            return None if self.config.is_last_rank else self._trace_output

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
            d2h_service=d2h_service,
            record_dev=record_dev,
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
                self.drafter.forward(
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

    def release_trace(self) -> None:
        """Free the captured trace segments and the sub-device managers that own them, BEFORE the
        driver closes the mesh device. Idempotent; safe to call when use_trace is off.

        Required, not hygiene: the captured MeshTraceBuffers live inside the MoE-overlap
        SubDeviceManagers, so closing the mesh with both still registered tears them down in the wrong
        order and segfaults —

            MeshDevice::close -> SubDeviceManagerTracker::~SubDeviceManagerTracker
              -> SubDeviceManager::~SubDeviceManager
                -> hashtable<MeshTraceId, MeshTraceBuffer>::~_Hashtable
                  -> MeshTraceBuffer::~MeshTraceBuffer -> MeshBuffer::deallocate
                    -> BankManager::deallocate_buffer   [SIGSEGV]

        The model tests never hit this because their harness calls
        TtPrefillTransformer.release_sub_device_managers() itself; the runner had no equivalent, so a
        traced runner run always segfaulted at shutdown (after a fully successful chunk loop).
        """
        if self._controller is not None:
            self._controller.release()
            self._trace_captured = False
        # Drop the overlap sub-device managers too: they are what the trace buffers hang off, and
        # leaving them registered at close has been observed to segfault teardown on its own.
        release = getattr(self.model, "release_sub_device_managers", None)
        if release is not None:
            release()

    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        """Register the per-layer-ack channel (docs/scheduler/prefill.md §3.11).

        `layer_ack_channel` is a `ttnn.InterProcessCounterChannel` on
        `/tt_prefill_layer_acks_<service_id>`. The runner bumps it once per
        layer (`inject(1)`); the scheduler reads the delta and drives the
        migration worker. The ack carries no payload — the scheduler correlates
        acks with the chunk it pushed (its InFlightChunkFIFO).

        Per-layer cadence means NUM_LAYERS acks per chunk, so the scheduler must
        be configured with layers_per_chunk == NUM_LAYERS.

        use_trace: the capture splits the trace at each ack point (a host shm bump cannot live inside a
        trace), so the ack callback must be known at CAPTURE time. Call this BEFORE capture_trace() — it
        only registers the callback on the controller and asserts nothing has been captured yet.
        """
        assert self.compiled or self.config.use_trace, "Call compile() before set_layer_ack_channel()"

        def on_layer_complete(layer_idx: int) -> None:
            layer_ack_channel.inject(1)

        self._on_layer_complete = on_layer_complete
        if self.config.use_trace and self._controller is not None:
            # Register on the controller so the (later) capture splits at each ack point. Ordering is a
            # precondition, not something to recover from: re-capturing here would throw away the
            # recorded segments and record a second time, and the caller can simply register first.
            assert not self._trace_captured, (
                "use_trace: set_layer_ack_channel() must run BEFORE capture_trace() — the ack callback has "
                "to be known at capture time so the segments split at each ack point"
            )
            self._controller.set_layer_ack_callback(on_layer_complete)

    def kv_migration_base_address(self, kv_caches: MlaKvCaches) -> int:
        """This stage's primary KV base DRAM address — the engine's single-cache hook for the
        migration all-gather (it holds the cache but must not introspect its layout). `.kvpe` is an
        MlaKvCache wrapper rather than a bare tensor, hence `.storage`. A sparse/DSA model migrates a
        second cache too: see `kv_migration_stages`, which the engine prefers."""
        return int(kv_caches.kvpe.storage.buffer_address())

    def kv_migration_stages(self, kv_caches: MlaKvCaches, first_layer_idx=None, num_my_layers=None):
        """One `KvCacheStage` per merged-table config: KVPE first, then the sparse/DSA index-key cache
        when present. The engine all-gathers one layout per entry (on ALL ranks) and hands them to
        `build_kv_chunk_table`.

        The two caches do NOT share a layer-index space, which is why a stage is per-cache: KVPE holds
        this stage's layers under the model's global numbering, while the index cache is numbered in
        COMPACTED full-indexer space (`full_indexer_rank`) because only `full` layers own an indexer and
        write a slot. Both hold THIS stage only, so each stage's layers sit at its own slot 0.
        """
        from models.demos.common.prefill.runners.migration import KvCacheStage
        from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank

        first_layer_idx = self.config.first_layer_idx if first_layer_idx is None else int(first_layer_idx)
        num_my_layers = self.config.num_layers if num_my_layers is None else int(num_my_layers)
        stages = [KvCacheStage(self.kv_migration_base_address(kv_caches), first_layer_idx, num_my_layers)]

        index_cache = kv_caches.index
        if index_cache is not None:
            first_full = full_indexer_rank(self.hf_config, first_layer_idx)
            count_full = full_indexer_rank(self.hf_config, first_layer_idx + num_my_layers) - first_full
            slots_per_user = index_cache.shape[0] // self.config.num_users
            if slots_per_user != count_full:
                raise RuntimeError(
                    f"index cache holds {slots_per_user} layers per slot but this stage owns "
                    f"{count_full} full-indexer layers; the table cannot place its layers unless the "
                    "cache is sized to the stage (see the GLM-5.2 adapter's allocate_kv_cache)."
                )
            stages.append(KvCacheStage(int(index_cache.buffer_address()), first_full, count_full))
        return stages

    def build_kv_chunk_table(
        self,
        kv_caches: MlaKvCaches,
        path: str,
        *,
        first_layer_idx: int = 0,
        num_my_layers: Optional[int] = None,
        stage_layouts=None,
    ) -> str:
        """Build + serialize the KV-chunk address table for the engine-owned `MlaKvCaches` to
        `path` and return it.

        The table maps each natural KV position to its true block-cyclic storage chip + offset
        (the MLA chunked-prefill cache layout), so the migration worker copies the right chunks.
        The runner publishes the serialized table to the worker — this method only describes the
        cache layout; it issues no migration comms.

        Multi-rank (pipeline-parallel): this rank owns layers [first_layer_idx, first_layer_idx +
        num_my_layers). The runner runs the all-ranks all-gathers and passes `stage_layouts` — one
        layout per cache, in config order, from `kv_migration_stages` — so ONLY rank 0 builds
        the table spanning every stage. The single-rank default (None) gathers inline.

        For a sparse/DSA model (``.index`` present) the result is a single MERGED table describing BOTH
        caches — config 0 = the KVPE cache, config 1 = the index-key cache. A dense model (``.index`` None)
        → the usual single-config table over the KVPE cache alone.

        Under DFlash, this rank's drafter context caches join the same merged table as
        ``2 * num_kv_heads`` further named configs (see the gate below)."""
        from models.demos.deepseek_v3_d_p.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        # DFlash: register the drafter's context K/V as further configs of the same merged table, so a
        # device-less consumer (prefill_producer) can read them back per (layer, head) and PCC them
        # against the golden trace exactly like the verifier's caches. Only when this rank actually owns
        # them (allocated under dflash_enabled AND is_last_rank), and only on the single-stage path:
        #
        #   * the cross-stage (pipeline-parallel) merge does not cover the drafter, and
        #   * real migration must not COPY drafter KV yet. All num_layers layer-acks fire inside
        #     model.forward, while the drafter write happens after forward returns (see prefill_chunk
        #     above), so a worker acting on the last ack would migrate drafter chunks the current chunk
        #     has not written yet. Registering it for the mock path (which only reads) is safe; wiring it
        #     into live migration needs that ordering fixed first.
        dflash_caches = None
        if self._dflash_k_cache is not None:
            # The runner all-gathers whenever migration is enabled, so a layout carrying one stage still
            # means single-rank; only a genuine cross-stage merge has to drop the drafter.
            if stage_layouts is None or all(len(layout) == 1 for layout in stage_layouts):
                dflash_caches = (self._dflash_k_cache, self._dflash_v_cache)
            else:
                logger.warning(
                    "[migration] DFlash drafter caches are NOT in the KV chunk table: the cross-stage "
                    "(pipeline-parallel) merge does not describe them, and the drafter write trails the "
                    "layer-acks within a chunk. Drafter KV will not be migrated or PCC-checked."
                )

        # Dense row -> global layer map, so config 1 is published on the layer axis (see the builder).
        index_layer_ids = None
        if kv_caches.index is not None and getattr(self.hf_config, "indexer_types", None):
            from models.demos.deepseek_v3_d_p.tt.mla.indexer import indexer_layer_is_reused
            from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import merged_num_layers

            # The merged table spans EVERY stage, so the map covers the model's layers, not this rank's slice.
            total_layers = merged_num_layers(stage_layouts[0]) if stage_layouts else self.config.num_layers
            index_layer_ids = [
                layer for layer in range(total_layers) if not indexer_layer_is_reused(self.hf_config, layer)
            ]

        return build_and_serialize_kv_chunk_table(
            mesh_device=self.mesh_device,
            kvpe_cache=kv_caches.kvpe,
            seq_len=self.config.max_seq_len,
            num_layers=self.config.num_layers,
            mesh_shape=self.config.mesh_shape,
            sp_axis=self.config.sp_axis,
            tp_axis=self.config.tp_axis,
            num_users=self.config.num_users,
            chunk_size_global=self.config.chunk_size,  # block-cyclic period (prefill chunk size)
            path=path,
            index_kv_cache=kv_caches.index,
            dflash_caches=dflash_caches,
            first_layer_idx=first_layer_idx,
            num_my_layers=num_my_layers,
            stage_layouts=stage_layouts,
            index_layer_ids=index_layer_ids,
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
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)

        def _slot_block(tensor, rows_per_slot: int):
            """This slot's rows of a user-major cache, gathered to host as one TP replica:
            [rows_per_slot, 1, seq_cache, row]. `rows_per_slot` differs per cache — see below."""
            s = list(tensor.shape)
            sl = ttnn.slice(
                tensor,
                [slot * rows_per_slot, 0, 0, 0],
                [(slot + 1) * rows_per_slot, s[1], s[2], s[3]],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            host = ttnn.to_torch(sl, mesh_composer=composer)[:, :1]
            ttnn.deallocate(sl)
            return host

        physical = _slot_block(kvpe.storage, num_layers)
        blocks = [kvpe.unpack_host(physical).to(torch.float32)]  # [num_layers, 1, seq_cache, kvpe_logical]
        if kv_caches.index is not None:
            # Sparse/DSA second cache (the lightning-indexer keys). Returned so a slot-vs-slot
            # comparison (validate_migrations_pairwise, which is length-agnostic over the returned
            # list) covers BOTH caches — without this a migrated sparse model reports "PASSED"
            # having checked only the KVPE half.
            #
            # Two ways it differs from `.kvpe`: its per-slot stride is its OWN layer count, NOT
            # config.num_layers (GLM-5.2 sizes it to this stage's `full` indexer layers only);
            # and it is a plain ttnn.Tensor, so there is no `.storage` / `unpack_host` (bfp8_b TILE
            # dequantizes on to_torch).
            index = kv_caches.index
            rows_per_slot = index.shape[0] // self.config.num_users
            blocks.append(_slot_block(index, rows_per_slot).to(torch.float32))
        return blocks

    def kv_cache_pcc_check(
        self,
        kv_caches: MlaKvCaches,
        *,
        slot_id: int,
        n_chunks: int,
        trace_dir=None,
        first_layer_idx: int = 0,
        real_len=None,
        pt_path_override=None,
    ) -> float:
        """Optional bring-up hook (not part of the core runtime contract; never called in production
        serving). PCC the populated engine-owned primary KV cache (`.kvpe`) for `slot_id` against the
        golden trace; returns the min per-layer PCC and asserts on failure (unless
        PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1). Thin forwarder into the model's validation module.
        `real_len` caps the compared extent to the real (non-pad) tokens — a partial last chunk makes
        n_chunks * chunk_size overshoot the prompt; `pt_path_override` selects a per-slot .pt golden
        (both are for out-of-tree callers; nothing in-tree passes either)."""
        from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import kv_cache_pcc_check

        return kv_cache_pcc_check(
            self,
            kv_caches.kvpe,
            slot_id=slot_id,
            n_chunks=n_chunks,
            trace_dir=trace_dir,
            first_layer_idx=first_layer_idx,
            real_len=real_len,
            pt_path_override=pt_path_override,
        )

    def dflash_kv_cache_pcc_check(self, kv_caches: MlaKvCaches, *, slot_id: int, out_len: int, golden_dir=None):
        """Optional bring-up hook (never called in production serving): PCC the DFlash drafter's context
        K/V for `slot_id` against the golden trace over the first `out_len` positions. Separate from
        `kv_cache_pcc_check`, which covers only the verifier's `kvpe` — with DFlash on, the drafter cache
        is a second populated cache and the only one whose contents depend on the D2D-transported FC
        partial. Thin forwarder into the drafter's validation module.

        Only the last rank builds the drafter KV tail and owns the caches, so every other rank has
        nothing to check and returns 1.0 unmeasured."""
        if self._dflash_k_cache is None:
            logger.info(
                f"[dflash-pcc] rank owns no drafter KV cache (is_last_rank={self.config.is_last_rank}); "
                "nothing to check"
            )
            return 1.0
        from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_kv_validation import dflash_kv_cache_pcc_check

        dcfg = self.drafter.config
        return dflash_kv_cache_pcc_check(
            self.mesh_device,
            self._dflash_k_cache,
            self._dflash_v_cache,
            sp=self.config.sp_factor,
            chunk_size_global=self.config.chunk_size,
            num_layers=dcfg.num_hidden_layers,
            num_kv_heads=dcfg.num_key_value_heads,
            head_dim=dcfg.head_dim,
            slot_id=slot_id,
            out_len=out_len,
            golden_dir=golden_dir,
            record_only=os.environ.get("PREFILL_STANDALONE_CHUNKED_RECORD_ONLY", "0") == "1",
            rope_convention=dcfg.rope_convention,
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

        use_trace: same constraint as set_layer_ack_channel — the callback must be known at CAPTURE
        time (a host push cannot live inside a trace), so it is registered on the controller and the
        eager capture is re-recorded to split at each ack point. The per-call request_id closure the
        eager path uses is not available there, so the captured callback reads _trace_request_id,
        which prefill_chunk() publishes before each replay.
        """
        assert self.compiled or self.config.use_trace, "Call compile() before set_layer_completion_sink()"
        self._layer_completion_sink = sink
        if not self.config.use_trace:
            return

        def on_layer_complete(layer_idx: int) -> None:
            sink(layer_idx, self._trace_request_id)

        # Route the traced path through the same controller hook the LayerAck channel uses, so the
        # capture is segmented at every completion point.
        self._on_layer_complete = on_layer_complete
        # No silent no-op here: use_trace means compile() ran _prepare_trace and built the controller.
        # Returning quietly if it is missing would register the sink and then never fire it, so a
        # pipelined traced run would emit ZERO completions and the scheduler would simply hang.
        assert self._controller is not None, (
            "use_trace: compile() must run (building the trace controller) before "
            "set_layer_completion_sink(); without it the sink would never fire under trace replay."
        )
        # Same ordering precondition as set_layer_ack_channel: register before capture_trace().
        assert not self._trace_captured, (
            "use_trace: set_layer_completion_sink() must run BEFORE capture_trace() — the completion "
            "callback has to be known at capture time so the segments split at each completion point"
        )
        self._controller.set_layer_ack_callback(on_layer_complete)
