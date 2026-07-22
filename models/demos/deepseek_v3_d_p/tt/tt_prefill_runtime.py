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
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController


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

        assert (
            config.max_seq_len % config.chunk_size == 0
        ), f"max_seq_len ({config.max_seq_len}) must be a multiple of chunk_size ({config.chunk_size})"

        self.model_built = False
        self.compiled = False

        # Trace-safe metadata prefill state (config.use_trace). compile() warm-compiles the programs AND
        # records the segmented capture EAGERLY (before the chunk loop), so the capture cost is out of the
        # timed loop. If a per-layer LayerAck is registered after compile() (request loop), the eager trace
        # is RE-captured with the ack (set_layer_ack_channel -> _capture_now, using _kv_cache):
        #   _controller       — SubDeviceTraceController driving the segmented capture/replay
        #   _trace_input      — persistent per-chunk input buffer (captured address; updated in place)
        #   _trace_metadata   — 3 persistent 1-element uint32 tensors (slot_id, actual_start, actual_end)
        #   _trace_output     — persistent output activation (non-last rank only; read by the D2D send)
        #   _trace_captured   — flips True once compile() records the segmented capture
        #   _kv_cache         — the engine cache handle from compile(), kept for a post-compile re-capture
        self._controller = None
        self._trace_input = None
        self._trace_metadata = None
        self._trace_output = None
        self._trace_captured = False
        self._kv_cache = None

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
            overlap_shared_expert_with_dispatch=self.config.overlap_shared_expert_with_dispatch,
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

    def compile(self, kv_cache: ttnn.Tensor) -> None:
        """Warm up one chunk so the per-chunk loop hits no first-run cost. The engine
        passes the cache it owns; the warm-up writes into it (slot 0) and is harmless.

        use_trace: set up the persistent buffers + controller and warm-compile the metadata programs here,
        but do NOT record the capture yet — the driver must call capture_trace() AFTER any pipeline D2D
        endpoints are built (build_d2d_pipeline_endpoints allocates L1 for the receiver socket, which would
        otherwise land on top of the captured trace buffers on the last rank and corrupt replay) and after
        set_layer_ack_channel(). capture_trace() runs before the chunk loop, so the one-time capture is still
        out of the timed loop; prefill_chunk() then only replays (with a lazy-capture safety net)."""
        assert self.model_built
        chunk = self.config.chunk_size
        t0 = time.perf_counter()
        if self.config.use_trace:
            logger.info(
                f"TtPrefillRuntime.compile() — warming traced {chunk}-token chunk (metadata path); capture deferred to capture_trace()"
            )
            self._kv_cache = kv_cache  # kept so capture_trace()/set_layer_ack_channel() can (re)capture
            self._prepare_trace(kv_cache)
        else:
            logger.info(f"TtPrefillRuntime.compile() — warming up one {chunk}-token chunk")
            tt_input = self.make_chunk_input([0] * chunk)
            self.prefill_chunk(tt_input, kv_cache, slot_id=0, actual_start=0, actual_end=chunk)
            ttnn.synchronize_device(self.mesh_device)
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"[prefill timing] task_id={'PREPARE' if self.config.use_trace else 'WARMUP'} num_tokens={chunk} "
            f"runtime.compile() = {warmup_ms:.2f} ms"
        )
        self.compiled = True

    def capture_trace(self, kv_cache: ttnn.Tensor) -> None:
        """Record the segmented trace, ONCE, before the chunk loop. The driver must call this AFTER
        building any D2D pipeline endpoints (their receiver-socket L1 must be allocated first, or it lands
        on the captured trace buffers on the last rank and corrupts replay) and AFTER set_layer_ack_channel()
        (so the ack segments the capture). No-op if not use_trace or already captured. See compile()."""
        if not self.config.use_trace or self._trace_captured:
            return
        assert self._controller is not None, "capture_trace(): compile() must run first (prepares the trace)"
        self._capture_now(kv_cache)

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
            kv_cache,
            actual_isl=self.config.chunk_size,
            on_layer_complete=self._on_layer_complete,
            actual_start=None,
            actual_end=None,
            cache_user_id=0,
            metadata=self._trace_metadata,
        )

    def _prepare_trace(self, kv_cache: ttnn.Tensor) -> None:
        """Set up the persistent input + per-element metadata buffers and the controller, then warm-compile
        the metadata-variant programs (a full forward). Does NOT begin/end the capture — compile() calls
        _capture_now() right after. Called once from compile()."""
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

    def _capture_now(self, kv_cache: ttnn.Tensor) -> None:
        """Record the segmented forward. Called eagerly from compile() (before the chunk loop), and again
        from set_layer_ack_channel() to re-capture with a newly-registered ack. The SubDeviceTraceController
        chops the capture at the MoE sub-device swaps (and, if a per-layer ack callback is registered, at
        each migration ack). On a non-last rank the captured forward's output activation is kept at its
        captured address in _trace_output so the driver can read it (and forward it over D2D) after every
        replay."""
        controller = self._controller
        assert controller is not None, "_prepare_trace() must run (via compile()) before _capture_now()"
        if self._trace_captured:
            # Re-capture (ack registered after the initial eager capture): drop the old segments first so
            # their trace buffers are freed before recording the new (ack-segmented) capture.
            controller.release()
            self._trace_captured = False
        # Ack registered after compile (request loop, gap #3): the block runs the metadata
        # zero_padded_kv_cache + ack routing ONLY when on_layer_complete is set (tt_prefill_block.forward),
        # so those programs were NOT compiled by _prepare_trace's warm pass (which ran with
        # on_layer_complete=None). Warm them now — with a NO-OP ack so this warm pass fires no real
        # migration acks — then register the real ack so the capture splits the trace at each ack point (a
        # host shm bump cannot live inside a trace). No ack (standalone) => nothing extra to warm.
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

    def prefill_chunk(
        self,
        input_tensor: ttnn.Tensor,
        kv_cache: ttnn.Tensor,
        slot_id: int,
        actual_start: int,
        actual_end: int,
    ) -> Optional[ttnn.Tensor]:
        """Prefill ONE chunk into user `slot_id`'s slice of the engine-owned `kv_cache`.

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
            kv_cache: the engine-owned KV cache (from the adapter's allocate_kv_cache); this chunk's
                KV is written into it. The same tensor is passed on every call.
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

        if self.config.use_trace:
            # Traced path: update the persistent input + per-element metadata IN PLACE, then replay the
            # captured segmented forward. The metadata (slot_id, actual_start, actual_end) drives every
            # per-chunk scalar on-device, so the single capture fills the correct KV for this chunk.
            assert self._controller is not None, "use_trace: compile() must prepare the trace before prefill_chunk"
            # compile() already captured (eagerly). Safety net only: if somehow not captured, capture now.
            if not self._trace_captured:
                self._capture_now(kv_cache)
            ttnn.copy(input_tensor, self._trace_input)
            for dst, val in zip(self._trace_metadata, (slot_id, actual_start, actual_end)):
                ttnn.copy_host_to_device_tensor(self._meta1_host(val), dst)
            self._controller.replay()
            ttnn.deallocate(input_tensor)
            # Non-last rank: return the persistent output activation (replay just refreshed it) for the
            # driver to forward downstream over D2D. Last/single rank: the populated KV cache is the output.
            return None if self.config.is_last_rank else self._trace_output

        out = self.model.forward(
            input_tensor,
            kv_cache,
            actual_isl=actual_end - actual_start,
            on_layer_complete=self._on_layer_complete,
            actual_start=actual_start,
            actual_end=actual_end,
            cache_user_id=slot_id,
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

        use_trace: the capture splits the trace at each ack point (a host shm bump cannot live inside a
        trace), so the ack callback must be known at CAPTURE time. compile() captures eagerly (without the
        ack), so a post-compile call here RE-captures with the ack — a one-time cost before the chunk loop.
        Call this before the first prefill_chunk() (the request loop does).
        """
        assert self.compiled or self.config.use_trace, "Call compile() before set_layer_ack_channel()"

        def on_layer_complete(layer_idx: int) -> None:
            layer_ack_channel.inject(1)

        self._on_layer_complete = on_layer_complete
        if self.config.use_trace and self._controller is not None:
            if self._trace_captured:
                # compile()'s eager capture recorded the trace WITHOUT ack boundaries; re-capture now that
                # the ack is known so the segments split at each per-layer migration ack. One-time, before
                # the loop. _capture_now registers the ack on the controller (with a no-op warm pass first).
                assert self._kv_cache is not None, "use_trace: compile() must run before set_layer_ack_channel()"
                self._capture_now(self._kv_cache)
            else:
                self._controller.set_layer_ack_callback(on_layer_complete)

    def build_kv_chunk_table(self, kv_cache: ttnn.Tensor, path: str) -> str:
        """Build + serialize the KV-chunk address table for the engine-owned `kv_cache` to
        `path` and return it.

        The table maps each natural KV position to its true block-cyclic storage chip + offset
        (the MLA chunked-prefill cache layout), so the migration worker copies the right chunks.
        The runner publishes the serialized table to the worker — this method only describes the
        cache layout; it issues no migration comms. Single-rank only (config.num_layers == the
        full model)."""
        from models.demos.deepseek_v3_d_p.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        return build_and_serialize_kv_chunk_table(
            mesh_device=self.mesh_device,
            kvpe_cache=kv_cache,
            seq_len=self.config.max_seq_len,
            num_layers=self.config.num_layers,
            mesh_shape=self.config.mesh_shape,
            sp_axis=self.config.sp_axis,
            num_users=self.config.num_users,
            chunk_size_global=self.config.chunk_size,  # block-cyclic period (prefill chunk size)
            path=path,
        )

    def kv_cache_pcc_check(
        self, kv_cache: ttnn.Tensor, *, slot_id: int, n_chunks: int, trace_dir=None, first_layer_idx: int = 0
    ) -> float:
        """Optional bring-up hook (not part of the core runtime contract; never called in production
        serving). PCC the populated engine-owned `kv_cache` for `slot_id` against the golden trace;
        returns the min per-layer PCC and asserts on failure (unless PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1).
        Thin forwarder into the model's validation module so the PCC logic lives in one place."""
        from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import kv_cache_pcc_check

        return kv_cache_pcc_check(
            self, kv_cache, slot_id=slot_id, n_chunks=n_chunks, trace_dir=trace_dir, first_layer_idx=first_layer_idx
        )
