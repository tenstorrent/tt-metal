# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TtDeepSeekPrefillPipeline — high-level wrapper around TtPrefillTransformer + KV cache
for disaggregated prefill/decode inference.

Designed to be easy to use from the tt-inference-server's prefill runner:

    # Server startup (all MPI ranks):
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(32, 4))
    config = TtPrefillPipelineConfig(num_layers=61, max_seq_len=102400, ...)
    pipeline = TtDeepSeekPrefillPipeline(mesh_device, hf_config, state_dict, config,
                                         migration_layer=migration_layer)
    pipeline.compile()  # warmup once

    # Per request (rank 0 reads SHM and drives the call; other ranks participate
    # in the collective via their own pipeline.prefill() call):
    first_token = pipeline.prefill(token_ids=..., slot_id=...)

Follows the pattern established by TtSDXLPipeline:
  - Caller opens the mesh and loads the torch model; pipeline just uses them
  - Static params live in a dataclass config
  - State flags track initialization progress; asserts enforce ordering
  - Explicit method chain (compile → prefill) rather than an opaque __call__
  - __del__ releases device resources
"""

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


@dataclass
class TtPrefillPipelineConfig:
    """Static configuration for the prefill pipeline.

    Set once at startup — values that change per request (token_ids, slot_id,
    actual_isl) are passed to prefill() instead.
    """

    num_layers: int
    max_seq_len: int  # maximum sequence length the KV cache is sized for
    mesh_shape: tuple = (32, 4)  # global (SP, TP) mesh
    is_balanced: bool = True  # use zigzag / balanced ring attention
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
    """Owns the prefill model, KV cache, and optional migration layer.

    One instance per prefill process. Call compile() once before the first
    prefill() call. The class is stateful but per-request state (tokens,
    slot_id) is ephemeral — only the model, cache, and migration handles
    persist across requests.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        hf_config: PretrainedConfig,
        state_dict: dict,
        config: TtPrefillPipelineConfig,
        migration_layer=None,
    ):
        """
        Args:
            mesh_device: TTNN mesh device, pre-opened by the caller
                (e.g. ttnn.open_mesh_device(MeshShape(32, 4))).
            hf_config: HuggingFace PretrainedConfig for the model.
            state_dict: DeepSeek weights in TtPrefillTransformer format.
                May be empty {} if loading from weight_cache_path.
            config: Static pipeline parameters (see TtPrefillPipelineConfig).
            migration_layer: Optional MigrationLayer for KV cache transfer to decode.
                When None, prefill runs without migration (useful for testing).
        """
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config
        self.migration_layer = migration_layer

        # State flags — enforced by asserts in prefill()
        self.model_built = False
        self.kv_cache_allocated = False
        self.compiled = False

        self._build_model(state_dict)
        self._allocate_kv_cache()

    # ----------------------------------------------------------------
    # Setup (called in __init__)
    # ----------------------------------------------------------------

    def _build_model(self, state_dict: dict) -> None:
        logger.info(
            f"Building TtDeepSeekPrefillPipeline model: "
            f"num_layers={self.config.num_layers}, max_seq_len={self.config.max_seq_len}, "
            f"mesh_shape={self.config.mesh_shape}, is_balanced={self.config.is_balanced}"
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
            capacity_factor=self.config.capacity_factor,
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
        # ttnn.empty — no host→device zero transfer. MLA zeros padding pages
        # before fill_cache_for_user_() when migration is active.
        self.kvpe_cache = init_kvpe_cache(
            kvpe_cache_head_dim=kvpe_head_dim,
            mesh_device=self.mesh_device,
            seq_len=self.config.max_seq_len,
            mesh_shape=list(self.config.mesh_shape),
            sp_axis=self.config.sp_axis,
            num_kvpe_cache_layers=self.config.num_layers,
        )
        self.kv_cache_allocated = True

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def compile(self) -> None:
        """Warmup pass to JIT-compile kernels. Call once before the first prefill().

        TODO: currently a no-op placeholder. Could run a dummy forward with a
        small sequence to trigger JIT compilation.
        """
        assert self.model_built and self.kv_cache_allocated
        logger.info("TtDeepSeekPrefillPipeline.compile() — no-op (TODO: dummy forward)")
        self.compiled = True

    # ----------------------------------------------------------------
    # Per-request entry point
    # ----------------------------------------------------------------

    def prefill(self, token_ids: list[int], slot_id: int) -> int:
        """Run prefill for one request.

        All MPI ranks must call prefill() collectively with the same arguments;
        TTNN's distributed runtime handles cross-host CCL internally.

        Args:
            token_ids: Full input sequence in the user's original token order.
                This function does the zigzag reorder internally.
            slot_id: KV cache slot assigned by the inference server.

        Returns:
            First generated token ID.
        """
        assert self.compiled, "Call compile() before prefill()"
        actual_isl = len(token_ids)

        tt_token_ids = self._prepare_input_tensor(token_ids, actual_isl)
        on_layer_complete = self._build_migration_callback(slot_id, actual_isl)

        _ = self.model.forward(
            tt_token_ids,
            self.kvpe_cache,
            on_layer_complete=on_layer_complete,
            actual_isl=actual_isl,
        )

        # TODO: LM head + sampling not yet implemented upstream.
        first_token = 128798  # placeholder: <think> token
        return first_token

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _prepare_input_tensor(self, token_ids: list[int], actual_isl: int) -> ttnn.Tensor:
        """Zigzag-reorder tokens (if is_balanced) and shard to the global mesh."""
        sp_factor = self.config.sp_factor
        isl_per_chip = actual_isl // sp_factor

        if self.config.is_balanced:
            # Reorder into zigzag chunk order so each SP device gets one chunk
            # from the front and one from the back of the sequence.
            # reorder_tensor_chunks requires a 4D tensor with seq_dim=2.
            chunk_order = create_balanced_chunk_order(sp_factor)
            t = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            # [1, 1, seq_len, 1]
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

    def _build_migration_callback(self, slot_id: int, actual_isl: int):
        """Build the per-layer migration callback passed to MLA via forward().

        MLA invokes this after fill_cache_for_user_(). MLA also handles zeroing
        padding pages before fill (gated on this callback being set).

        Returns None when no migration_layer is configured — MLA then skips both
        the zero-out and the post-fill callback.
        """
        if self.migration_layer is None:
            return None

        mesh_device = self.mesh_device
        migration_layer = self.migration_layer

        def on_layer_complete(layer_idx, kvpe_cache):
            ttnn.synchronize_device(mesh_device)
            migration_layer.migrate_slot(
                layer=layer_idx,
                pos_start=0,
                pos_end=actual_isl,
                src_slot=slot_id,
                dst_slot=slot_id,
            )

        return on_layer_complete

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------

    def __del__(self):
        # Device tensors are freed when their refs drop. ttnn handles the
        # underlying buffer lifecycle; we just clear our references here.
        try:
            self.kvpe_cache = None
            self.model = None
        except Exception:
            pass
