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
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache


@dataclass
class TtPrefillPipelineConfig:
    num_layers: int
    max_seq_len: int
    mesh_shape: tuple = (32, 4)
    is_balanced: bool = True
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
        # Per-layer LayerAck channel (set via set_layer_ack_channel after compile).
        self._layer_ack_channel = None

        self.model_built = False
        self.kv_cache_allocated = False
        self.compiled = False

        self._build_model(state_dict)
        self._allocate_kv_cache()

    def _build_model(self, state_dict: dict) -> None:
        logger.info(
            f"Building TtDeepSeekPrefillPipeline model: "
            f"num_layers={self.config.num_layers}, max_seq_len={self.config.max_seq_len}, "
            f"mesh_shape={self.config.mesh_shape}, is_balanced={self.config.is_balanced}"
        )
        if self.config.weight_cache_path:
            num_devices = self.config.mesh_shape[0] * self.config.mesh_shape[1]
            experts_per_chip = 256 // num_devices
            if TtPrefillTransformer.check_cache_complete(
                self.config.weight_cache_path, self.config.num_layers, experts_per_chip
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
            state_dict=state_dict,
            num_layers=self.config.num_layers,
            seq_len=self.config.max_seq_len,
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
        )
        self.model_built = True

    def _allocate_kv_cache(self) -> None:
        kvpe_head_dim = self.hf_config.qk_rope_head_dim + self.hf_config.kv_lora_rank
        self.kvpe_cache = init_kvpe_cache(
            kvpe_cache_head_dim=kvpe_head_dim,
            mesh_device=self.mesh_device,
            seq_len=self.config.max_seq_len,
            mesh_shape=list(self.config.mesh_shape),
            sp_axis=self.config.sp_axis,
            num_kvpe_cache_layers=self.config.num_layers,
        )
        self.kv_cache_allocated = True

    def compile(self) -> None:
        assert self.model_built and self.kv_cache_allocated
        max_seq_len = self.config.max_seq_len
        logger.warning(
            "TtDeepSeekPrefillPipeline: temperature is hardcoded to 0.0 (greedy argmax). "
            "Sampling is not yet supported — every prefill() returns the argmax token."
        )
        logger.info(f"TtDeepSeekPrefillPipeline.compile() — warming up with {max_seq_len} tokens")
        t0 = time.perf_counter()
        tt_token_ids = self._prepare_input_tensor([0] * max_seq_len)
        self.model.forward(
            tt_token_ids,
            self.kvpe_cache,
            number_of_non_padded_tokens=max_seq_len,
            temperature=0.0,
        )
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[prefill timing] task_id=WARMUP num_tokens={max_seq_len} pipeline.prefill() = {warmup_ms:.2f} ms")
        self.compiled = True

    def prefill(
        self,
        token_ids: Optional[list[int]] = None,
        slot_id: int = 0,
        actual_isl: Optional[int] = None,
        dst_slot: Optional[int] = None,
        *,
        input_tensor: Optional[ttnn.Tensor] = None,
        actual_start: int = 0,
    ) -> int:
        """Run one prefill pass and return the first generated token.

        Two input paths:
          * Legacy: pass `token_ids` (a list of ints). The pipeline shards and
            uploads the tokens via `ttnn.from_torch` on every call.
          * Pre-uploaded: pass `input_tensor` (an already SP-sharded uint32
            ROW_MAJOR tensor whose per-shard spec matches `_prepare_input_tensor`'s
            output). Useful when the caller drives a persistent H2D streaming
            service (e.g. `ttnn.H2DStreamService`) that owns the backing tensor.

        `actual_isl` is required when `input_tensor` is supplied — the pipeline
        can't infer it from a device tensor that's padded to `max_seq_len`.
        """
        assert self.compiled, "Call compile() before prefill()"
        if input_tensor is None:
            assert token_ids is not None, "Provide token_ids or input_tensor"
            if actual_isl is None:
                actual_isl = len(token_ids)
            tt_token_ids = self._prepare_input_tensor(token_ids)
        else:
            assert actual_isl is not None, "actual_isl required when input_tensor is provided"
            tt_token_ids = input_tensor
        if dst_slot is None:
            dst_slot = slot_id

        on_layer_complete = self._build_layer_ack_callback()

        first_token_id, _first_token_prob, _ = self.model.forward(
            tt_token_ids,
            self.kvpe_cache,
            number_of_non_padded_tokens=actual_isl,
            on_layer_complete=on_layer_complete,
            temperature=0.0,
            actual_start=actual_start,
        )
        return int(first_token_id)

    def _prepare_input_tensor(self, token_ids: list[int]) -> ttnn.Tensor:
        sp_factor = self.config.sp_factor
        isl_per_chip = len(token_ids) // sp_factor

        if self.config.is_balanced:
            chunk_order = create_balanced_chunk_order(sp_factor)
            t = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            t = reorder_tensor_chunks(t, chunk_order, seq_dim=2)
            token_ids_sharded = t.squeeze(0).squeeze(-1).reshape(sp_factor, 1, isl_per_chip)
        else:
            token_ids_sharded = torch.tensor(token_ids, dtype=torch.int64).reshape(sp_factor, 1, isl_per_chip)

        return ttnn.from_torch(
            token_ids_sharded,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                mesh_shape=self.config.mesh_shape,
                dims=(self.config.sp_axis, None),
            ),
        )

    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        """Register the per-layer-ack channel (docs/scheduler/prefill.md §3.11).

        `layer_ack_channel` is the scheduler-facing counter — a
        `ttnn.InterProcessCounterChannel` on `/tt_prefill_layer_acks_<service_id>`
        (the same segment the scheduler connects to). The runner's ONLY per-layer
        migration duty is to bump this counter once per layer; the scheduler reads
        the delta and drives the migration worker. No IPC with the worker.

        Per-layer cadence means NUM_LAYERS acks per chunk, so the scheduler must
        be configured with layers_per_chunk == NUM_LAYERS.
        """
        assert self.compiled, "Call compile() before set_layer_ack_channel()"
        self._layer_ack_channel = layer_ack_channel

    def _build_layer_ack_callback(self):
        """Per-layer hook → bump the scheduler-facing counter once per layer.

        The ack carries NO payload (no slot/range/layer): the scheduler
        correlates acks with the chunk it pushed (its InFlightChunkFIFO) and
        issues the MigrationCmds. Returns None when no channel is set.
        """
        if self._layer_ack_channel is None:
            return None
        channel = self._layer_ack_channel

        def on_layer_complete(layer_idx: int) -> None:
            channel.inject(1)

        return on_layer_complete
