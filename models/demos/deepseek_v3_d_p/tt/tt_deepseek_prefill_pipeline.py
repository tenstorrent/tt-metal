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
    # Chunked prefill is non-balanced (block-cyclic): the indexed RoPE path asserts is_balanced=False.
    is_balanced: bool = False
    # Chunked prefill: tokens are streamed in chunks of `chunk_size` and `num_users` independent
    # cache slots are allocated (user-major batch). max_seq_len is the per-user cache length, so the
    # full cache holds num_users * num_layers slots of max_seq_len each.
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
    # Static model-dimension constants for the variant being built
    # (DeepSeekV3Config | KimiK26Config). Drives expert counts, dense-layer
    # count, route groups, etc. in the TT layer code.
    model_cfg: type = DeepSeekV3Config

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

        assert not config.is_balanced, "Chunked prefill requires is_balanced=False (block-cyclic + indexed RoPE)"
        assert (
            config.max_seq_len % config.chunk_size == 0
        ), f"max_seq_len ({config.max_seq_len}) must be a multiple of chunk_size ({config.chunk_size})"

        self.model_built = False
        self.kv_cache_allocated = False
        self.compiled = False

        self._build_model(state_dict)
        self._allocate_kv_cache()
        # Whole-cache cos/sin/trans for the KV-pad-aware indexed rotated path, built once and reused
        # across every chunk (only the runtime kv_actual offset varies). Requires is_balanced=False.
        self.indexed_rope = self.model.rope_setup.get_rope_tensors_indexed(
            cache_seq_len_global=config.max_seq_len,
            chunk_size_global=config.chunk_size,
        )

    def _build_model(self, state_dict: dict) -> None:
        logger.info(
            f"Building TtDeepSeekPrefillPipeline model: "
            f"num_layers={self.config.num_layers}, max_seq_len={self.config.max_seq_len}, "
            f"mesh_shape={self.config.mesh_shape}, is_balanced={self.config.is_balanced}"
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
            mla_seq_len=self.config.max_seq_len,  # KV ring buffer = full per-user cache
            num_links=self.config.num_links,
            topology=self.config.topology,
            sp_axis=self.config.sp_axis,
            tp_axis=self.config.tp_axis,
            is_balanced=self.config.is_balanced,
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
            self.config.is_balanced,
            self.config.mesh_shape,
            self.config.sp_axis,
        )
        h, _ = self.model.forward_chunk(
            tt_tokens,
            self.kvpe_cache,
            self.indexed_rope,
            kv_actual_isl=0,
            cache_user_id=0,
        )
        ttnn.deallocate(tt_tokens)
        ttnn.deallocate(h)
        ttnn.synchronize_device(self.mesh_device)
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[prefill timing] task_id=WARMUP num_tokens={chunk} pipeline.prefill(chunk) = {warmup_ms:.2f} ms")
        self.compiled = True

    def prefill(
        self,
        input_tensor: ttnn.Tensor,
        slot_id: int,
        kv_actual_isl: int,
    ) -> None:
        """Prefill ONE chunk into user `slot_id`'s KV cache at offset `kv_actual_isl`. Does NOT sample
        — the populated cache is the output (read by the decode stage / migration consumer).

        The caller drives chunked prefill by calling this once per chunk, in order, managing the
        cumulative `kv_actual_isl` externally (advanced by the real token count per chunk). Each layer
        attends to the [0, kv_actual_isl + chunk_size) prefix and writes this chunk at the
        kv_actual_isl offset, so a chunk's KV must be populated before the next chunk reads it.

        If a LayerAck channel is registered (set_layer_ack_channel), the model bumps it once per layer
        (after each layer's chunk KV-cache write) so the scheduler can drive the migration worker.

        Args:
            input_tensor: one chunk's tokens, pre-sharded as a uint32 ROW_MAJOR DRAM tensor
                [sp_factor, 1, chunk_local] — as produced by runner_utils.prepare_prefill_input_tensor
                or streamed over the H2D socket. Assumed already in chip-major block-cyclic order (the
                caller owns reformatting — see test_prefill_transformer_chunked's rotated_chip_positions
                gather). Consumed (deallocated) here.
            slot_id: cache user slot to fill, in [0, num_users).
            kv_actual_isl: cumulative valid-KV count before this chunk (the cache write offset).
        """
        assert self.compiled, "Call compile() before prefill()"
        assert 0 <= slot_id < self.config.num_users, f"slot_id {slot_id} out of range [0, {self.config.num_users})"
        assert (
            kv_actual_isl + self.config.chunk_size <= self.config.max_seq_len
        ), f"chunk at kv_actual_isl={kv_actual_isl} exceeds per-user cache {self.config.max_seq_len}"

        h, _ = self.model.forward_chunk(
            input_tensor,
            self.kvpe_cache,
            self.indexed_rope,
            kv_actual_isl=kv_actual_isl,
            cache_user_id=slot_id,
            on_layer_complete=self._on_layer_complete,
        )
        ttnn.deallocate(input_tensor)
        ttnn.deallocate(h)

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
