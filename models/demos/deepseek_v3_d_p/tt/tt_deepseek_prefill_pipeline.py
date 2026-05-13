# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math
import os
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
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache


class BoundMigrationEndpoint:
    def __init__(self, endpoint, remote_endpoint_id: int):
        self._endpoint = endpoint
        self._remote_id = remote_endpoint_id

    def migrate_layer(self, layer: int, pos_start: int, pos_end: int, src_slot: int, dst_slot: int):
        return self._endpoint.migrate_layer(self._remote_id, layer, pos_start, pos_end, src_slot, dst_slot)

    def wait(self, uuid) -> None:
        """Block until the migration with the given uuid is fully sent + acked."""
        self._endpoint.wait_migration_send_completion(uuid)


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
        migration_layer=None,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config
        self.migration_layer = migration_layer

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
        token_ids: list[int],
        slot_id: int,
        actual_isl: Optional[int] = None,
        dst_slot: Optional[int] = None,
    ) -> int:
        assert self.compiled, "Call compile() before prefill()"
        if actual_isl is None:
            actual_isl = len(token_ids)
        if dst_slot is None:
            dst_slot = slot_id

        tt_token_ids = self._prepare_input_tensor(token_ids)
        on_layer_complete = self._build_migration_callback(slot_id, actual_isl, dst_slot)

        first_token_id, _first_token_prob, _ = self.model.forward(
            tt_token_ids,
            self.kvpe_cache,
            number_of_non_padded_tokens=actual_isl,
            on_layer_complete=on_layer_complete,
            temperature=0.0,
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

    def setup_migration(self, endpoint, remote_endpoint_id: int) -> None:
        assert self.compiled, "Call compile() before setup_migration()"
        self.migration_layer = BoundMigrationEndpoint(endpoint, remote_endpoint_id)

    def _build_migration_callback(self, slot_id: int, actual_isl: int, dst_slot: int):
        from models.demos.deepseek_v3_d_p.tt.runners.migration_setup import INVALID_SLOT_ID

        if self.migration_layer is None or dst_slot == INVALID_SLOT_ID:
            return None

        mesh_device = self.mesh_device
        migration_layer = self.migration_layer
        last_layer_idx = self.config.num_layers - 1

        kvpe_cache = self.kvpe_cache

        def on_layer_complete(layer_idx: int) -> None:
            ttnn.synchronize_device(mesh_device)
            end_pos = math.ceil(actual_isl / 128) * 128
            logger.info(
                f"[migration][prefill] on_layer_complete, migrating layer {layer_idx} from src_slot={slot_id} to dst_slot={dst_slot}. Start_pos={0}, End_pos={end_pos}"
            )
            if layer_idx == 0 and os.environ.get("PREFILL_DEBUG", "0") == "1":
                # Metal-side: dump KV cache bytes via the ttnn shard-spec path.
                from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import dump_kv_cache_shard_readback

                dump_kv_cache_shard_readback(layer_idx, kvpe_cache)
                # Blaze-side: dump the migration table for the same layer. Optional —
                # only fires when blaze's prefill_runner_util is on PYTHONPATH.
                try:
                    from prefill_runner_util import dump_migration_table_at_layer

                    dump_migration_table_at_layer(mesh_device, migration_layer, tag="3-pre-migrate")
                except ImportError:
                    pass
            uuid = migration_layer.migrate_layer(layer_idx, 0, end_pos, slot_id, dst_slot)
            ## Wait for each one for initial bringup
            # if layer_idx == last_layer_idx:
            logger.info(f"[migration][prefill] wait for migrate layer completion")
            migration_layer.wait(uuid)
            logger.info(f"[migration][prefill] done migrate layer")

        return on_layer_complete
