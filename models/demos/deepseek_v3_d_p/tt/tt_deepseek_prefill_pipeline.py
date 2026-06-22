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

        assert (
            config.max_seq_len % config.chunk_size == 0
        ), f"max_seq_len ({config.max_seq_len}) must be a multiple of chunk_size ({config.chunk_size})"

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
        self.compiled = True

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
