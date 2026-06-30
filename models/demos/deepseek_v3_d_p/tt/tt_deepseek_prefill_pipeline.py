# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController


@dataclass
class TtPrefillPipelineConfig:
    num_layers: int
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
    # Static model-dimension constants for the variant being built
    # (DeepSeekV3Config | KimiK26Config). Drives expert counts, dense-layer
    # count, route groups, etc. in the TT layer code.
    model_cfg: type = DeepSeekV3Config
    # When True, the last transformer layer runs kv-only: it fills the KV cache
    # (which migration needs) and skips its Q/SDPA/output projection, FFN/MoE,
    # the final RMSNorm, and the LM head. `prefill()` then returns None.
    kv_only_last_layer: bool = False
    # When True, capture the chunk forward ONCE as a ttnn trace at compile() and replay it on every
    # prefill() via ttnn.execute_trace — collapses the per-op host-dispatch (op2op) gaps. Requires a
    # device-only forward, so kv_only_last_layer must also be True. Pinned to chunk 0 for now: every
    # replay re-runs the captured chunk-0 inputs (multi-chunk tracing is deferred). Needs the mesh
    # opened with trace_region_size > 0.
    use_trace: bool = False
    # When True, capture the chunk forward ONCE as a METADATA-driven ttnn trace and replay it for every
    # chunk (and every user/slot). Unlike `use_trace` (chunk-0-pinned scalar), the per-chunk scalars
    # (slot_id / actual_start / actual_end) are NOT baked: the trace-safe MLA ops read them on-device from
    # a persistent metadata DRAM tensor that prefill() updates in-place per chunk, so one capture serves
    # the whole request stream. The per-layer migration ack (if a LayerAck channel is registered) is
    # chopped out of the trace by the controller and fired between segments at replay. Requires
    # kv_only_last_layer + trace_region_size > 0. Gated by PREFILL_USE_TRACE in the runner.
    use_metadata_trace: bool = False

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]


class TtDeepSeekPrefillPipeline:
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        hf_config: PretrainedConfig,
        state_dict: dict,
        config: TtPrefillPipelineConfig,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config
        # Per-layer LayerAck callback, built once in set_layer_ack_channel() after compile.
        self._on_layer_complete = None
        # Segmented-trace controller + its persistent (chunk-0) input, populated in compile() when use_trace.
        self._trace_controller = None
        self._trace_input = None
        # Metadata-trace path (use_metadata_trace): persistent metadata DRAM tensor + lazy-capture flag.
        # Capture happens on the FIRST prefill() (so a LayerAck channel registered after compile() is
        # already wired into the controller before capture). Both persistent tensors are then reused
        # (updated in-place per chunk) across every replay.
        self._trace_metadata = None
        self._metadata_trace_captured = False

        assert (
            config.max_seq_len % config.chunk_size == 0
        ), f"max_seq_len ({config.max_seq_len}) must be a multiple of chunk_size ({config.chunk_size})"
        if config.use_trace or config.use_metadata_trace:
            assert config.kv_only_last_layer, (
                "use_trace/use_metadata_trace requires kv_only_last_layer=True: trace capture needs a "
                "device-only forward, and kv_only_last_layer strips the host-side LM head + sampling tail."
            )
        assert not (config.use_trace and config.use_metadata_trace), "pick one trace path"

        self.model_built = False
        self.kv_cache_allocated = False
        self.compiled = False

        self._build_model(state_dict)
        self._allocate_kv_cache()

    def _build_model(self, state_dict: dict) -> None:
        logger.info(
            f"Building TtDeepSeekPrefillPipeline model: "
            f"num_layers={self.config.num_layers}, max_seq_len={self.config.max_seq_len}, "
            f"mesh_shape={self.config.mesh_shape}, chunk_size={self.config.chunk_size}, num_users={self.config.num_users}"
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
            ):
                logger.info(f"TTNN weight cache complete at {self.config.weight_cache_path}; loading from disk")
            else:
                logger.warning(
                    f"TTNN weight cache not complete at {self.config.weight_cache_path}; "
                    f"pipeline build will fail without a populated cache. "
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
            # Keep the shared-expert/dispatch overlap on; under use_trace the SubDeviceTraceController
            # splits the capture at the overlap's sub-device load/clear (see utils/sub_device_trace.py).
            overlap_shared_expert_with_dispatch=True,
            routing_use_l1_small_for_semaphores=self.config.routing_use_l1_small_for_semaphores,
        )
        self.model_built = True

    def _allocate_kv_cache(self) -> None:
        kvpe_head_dim = self.hf_config.qk_rope_head_dim + self.hf_config.kv_lora_rank
        # ONE shared cache holding num_users * num_layers slots (user-major batch); each user fills its
        # own layers via cache_user_id + cache_layer_idx during chunked prefill.
        self.kvpe_cache = init_kvpe_cache(
            kvpe_cache_head_dim=kvpe_head_dim,
            mesh_device=self.mesh_device,
            seq_len=self.config.max_seq_len,
            mesh_shape=list(self.config.mesh_shape),
            sp_axis=self.config.sp_axis,
            num_kvpe_cache_layers=self.config.num_layers,
            num_users=self.config.num_users,
        )
        self.kv_cache_allocated = True

    def compile(self) -> None:
        assert self.model_built and self.kv_cache_allocated
        chunk = self.config.chunk_size
        logger.info(f"TtDeepSeekPrefillPipeline.compile() — warming up one {chunk}-token chunk")
        t0 = time.perf_counter()
        tt_tokens = prepare_prefill_input_tensor(
            [0] * chunk,
            self.mesh_device,
            self.config.sp_factor,
            False,  # chunked prefill is block-cyclic (non-balanced)
            self.config.mesh_shape,
            self.config.sp_axis,
        )
        self.model.forward(
            tt_tokens,
            self.kvpe_cache,
            number_of_non_padded_tokens=chunk,
            actual_start=0,
            actual_end=chunk,
            cache_user_id=0,
        )
        ttnn.deallocate(tt_tokens)
        ttnn.synchronize_device(self.mesh_device)
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[prefill timing] task_id=WARMUP num_tokens={chunk} pipeline.prefill(chunk) = {warmup_ms:.2f} ms")

        if self.config.use_trace:
            self._capture_trace(chunk)

        self.compiled = True

    def _capture_trace(self, chunk: int) -> None:
        """Capture the chunk-0 forward as a (segmented) ttnn trace, replayed by prefill() when use_trace.
        The warmup above already populated the program cache, so capture records only dispatch. The input
        is allocated once and kept resident (`self._trace_input`) so the captured buffer addresses stay
        valid across every replay; the forward runs with on_layer_complete=None (migration acks do a host
        sync that can't be captured). The MoE keeps its shared-expert/dispatch overlap on, so
        SubDeviceTraceController splits the capture at each sub-device load/clear (see
        utils/sub_device_trace.py)."""
        # Persistent chunk-0 input — created once, NEVER deallocated until release().
        self._trace_input = prepare_prefill_input_tensor(
            [0] * chunk,
            self.mesh_device,
            self.config.sp_factor,
            False,  # chunked prefill is block-cyclic (non-balanced)
            self.config.mesh_shape,
            self.config.sp_axis,
        )
        logger.info(f"TtDeepSeekPrefillPipeline: capturing forward trace ({self.config.num_layers} layers, chunk 0)")
        self._trace_controller = SubDeviceTraceController(self.mesh_device)
        self.model.set_trace_controller(self._trace_controller)
        self._trace_controller.begin_capture()
        self.model.forward(
            self._trace_input,
            self.kvpe_cache,
            number_of_non_padded_tokens=chunk,
            on_layer_complete=None,
            actual_start=0,
            actual_end=chunk,
            cache_user_id=0,
        )
        self._trace_controller.end_capture()
        ttnn.synchronize_device(self.mesh_device)

        trace_bytes = self._trace_controller.trace_bytes()
        logger.info(
            f"[trace] {self.config.num_layers}-layer forward = {self._trace_controller.num_segments} "
            f"trace segments, {trace_bytes / (1024 * 1024):.2f} MB ({trace_bytes:,} bytes)"
        )

    def capture_trace(self) -> None:
        """Capture the metadata-driven forward ONCE as a (segmented) ttnn trace. Call this explicitly
        AFTER compile() — and, for the request loop, AFTER set_layer_ack_channel() so the per-layer
        migration ack is wired into the controller before capture. Subsequent prefill() calls just
        replay the trace (updating the persistent token + metadata buffers in place per chunk).

        Allocates the persistent token + metadata buffers, compiles the metadata op variants (a metadata
        warmup — distinct program hash from the scalar path), then captures the forward with the
        controller chopping at MoE sub-device swaps AND (if migration is on) at each per-layer ack. The
        per-chunk scalars are NOT baked: replay reads them from the persistent metadata tensor. No-op if
        not config.use_metadata_trace, or if already captured. See utils/sub_device_trace.py."""
        assert self.compiled, "call compile() before capture_trace()"
        if not self.config.use_metadata_trace or self._metadata_trace_captured:
            return
        import torch

        chunk = self.config.chunk_size
        # Persistent buffers — created ONCE, reused (updated in-place) across every replay, so the
        # addresses the capture recorded stay valid.
        self._trace_input = prepare_prefill_input_tensor(
            [0] * chunk, self.mesh_device, self.config.sp_factor, False, self.config.mesh_shape, self.config.sp_axis
        )
        self._trace_metadata = ttnn.from_torch(
            torch.tensor([0, 0, chunk, 0], dtype=torch.int64).reshape(1, 1, 1, 4),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self._trace_controller = SubDeviceTraceController(self.mesh_device)

        migration = self._on_layer_complete is not None

        def _fwd(on_layer_complete):
            # actual_start/end=None: every per-chunk scalar comes from `metadata` on-device.
            self.model.forward(
                self._trace_input,
                self.kvpe_cache,
                number_of_non_padded_tokens=chunk,
                on_layer_complete=on_layer_complete,
                actual_start=None,
                actual_end=None,
                cache_user_id=0,
                metadata=self._trace_metadata,
            )

        # WARMUP (controller idle, NO ack callback yet so has_layer_ack() is False -> the ack site uses
        # the on_layer_complete branch). For migration we pass a NO-OP callback so zero_padded_kv_cache
        # still compiles (it is gated on on_layer_complete is not None) without bumping the real ack
        # counter during warmup. Then we register the REAL ack callback for capture.
        self.model.set_trace_controller(self._trace_controller)
        warmup_olc = (lambda _idx: None) if migration else None
        _fwd(warmup_olc)
        ttnn.synchronize_device(self.mesh_device)

        if migration:
            self._trace_controller.set_layer_ack_callback(self._on_layer_complete)

        logger.info(f"[trace] capturing {self.config.num_layers}-layer metadata forward (migration={migration})...")
        self._trace_controller.begin_capture()
        _fwd(self._on_layer_complete if migration else None)
        self._trace_controller.end_capture()
        ttnn.synchronize_device(self.mesh_device)
        self._metadata_trace_captured = True

        trace_bytes = self._trace_controller.trace_bytes()
        logger.info(
            f"[trace] metadata forward = {self._trace_controller.num_segments} segments, "
            f"{trace_bytes / (1024 * 1024):.2f} MB"
        )

    def prefill(
        self,
        input_tensor: ttnn.Tensor,
        slot_id: int,
        actual_start: int,
        actual_end: int,
    ) -> None:
        """Prefill ONE chunk into user `slot_id`'s KV cache. Does NOT sample — the populated cache is
        the output (read by the decode stage / migration consumer).

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
            input_tensor: one chunk's tokens, SP-sharded uint32 ROW_MAJOR DRAM tensor as produced by
                prepare_prefill_input_tensor (block-cyclic, chip-major). Deallocated here.
            slot_id: cache user slot to fill, in [0, num_users).
            actual_start: absolute KV pos of the chunk's first real token (the cache write offset).
            actual_end: absolute KV pos past the chunk's last real token.
        """
        assert self.compiled, "Call compile() before prefill()"
        assert 0 <= slot_id < self.config.num_users, f"slot_id {slot_id} out of range [0, {self.config.num_users})"
        assert (
            actual_start + self.config.chunk_size <= self.config.max_seq_len
        ), f"chunk at actual_start={actual_start} exceeds per-user cache {self.config.max_seq_len}"
        assert (
            actual_start <= actual_end <= actual_start + self.config.chunk_size
        ), f"[actual_start={actual_start}, actual_end={actual_end}) not within one chunk of {self.config.chunk_size}"

        if self.config.use_metadata_trace:
            # Metadata-driven trace replay: one capture serves every chunk AND every user/slot. Update the
            # two persistent buffers in-place — this chunk's tokens into _trace_input, this chunk's
            # [slot_id, actual_start, actual_end, 0] into _trace_metadata — then replay. The trace-safe
            # MLA ops read the per-chunk scalars from _trace_metadata on-device; the controller fires the
            # per-layer migration ack between segments (if registered). NEVER realloc these buffers (a
            # fresh allocation would land at a different address and the replay would read freed memory).
            import torch

            assert self._metadata_trace_captured, "call pipeline.capture_trace() after compile() before prefill()"
            ttnn.copy(input_tensor, self._trace_input)  # device->device into the persistent buffer
            ttnn.deallocate(input_tensor)
            meta_host = ttnn.from_torch(
                torch.tensor([slot_id, actual_start, actual_end, 0], dtype=torch.int64).reshape(1, 1, 1, 4),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            ttnn.copy_host_to_device_tensor(meta_host, self._trace_metadata)
            self._trace_controller.replay()
            return

        if self.config.use_trace:
            # Replay the captured (segmented) forward. PINNED TO CHUNK 0: it re-runs the chunk-0 inputs
            # it captured, so `input_tensor`/`slot_id`/`actual_start` are ignored for now.
            # TODO(multi-chunk): copy this chunk's tokens into self._trace_input via
            # ttnn.copy_host_to_device_tensor and capture one trace per distinct actual_start; the
            # LayerAck on_layer_complete migration callback also needs reworking (its host sync can't
            # be captured) before trace can drive real disaggregated prefill.
            self._trace_controller.replay()
            ttnn.deallocate(input_tensor)
            return

        self.model.forward(
            input_tensor,
            self.kvpe_cache,
            number_of_non_padded_tokens=actual_end - actual_start,
            on_layer_complete=self._on_layer_complete,
            actual_start=actual_start,
            actual_end=actual_end,
            cache_user_id=slot_id,
        )
        ttnn.deallocate(input_tensor)

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

    def release(self) -> None:
        """Free the captured trace segments + resident input + MoE sub-device managers. Safe to call
        repeatedly. Removing the managers before mesh close avoids a teardown segfault in close_mesh_device."""
        if self._trace_controller is not None:
            self._trace_controller.release()
            self.model.set_trace_controller(None)
            self._trace_controller = None
        if self._trace_input is not None:
            ttnn.deallocate(self._trace_input)
            self._trace_input = None
        if self._trace_metadata is not None:
            ttnn.deallocate(self._trace_metadata)
            self._trace_metadata = None
        self.model.release_sub_device_managers()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
