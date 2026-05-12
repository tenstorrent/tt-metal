# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TT implementation of language model head (final norm + vocab projection) for Mistral Small 4.

Pipeline:
    hidden_states → RMSNorm → linear(W_lm_head) → logits

The final RMSNorm uses the same ``DistributedRMSNorm`` as per-layer norms.
The linear projection maps ``hidden_size → vocab_size`` with column-parallel sharding
across the mesh column axis (each device holds ``vocab_size // mesh_cols`` columns).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.mistral_small_4_119B.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.mistral_small_4_119B.tt_utils.abstract_module import AbstractModule
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.config_dataclass import (
    AllGatherAsyncConfig,
    FromWeightConfig,
    LinearConfig,
    MeshDeviceStub,
)
from models.demos.mistral_small_4_119B.tt_utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    all_gather_mesh_extent_on_cluster_axis,
    shard_and_save,
)
from models.demos.mistral_small_4_119B.tt_utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class Mistral4LMHead(AbstractModule):
    """TT implementation of language model head for Mistral Small 4.

    Combines:
    - Final RMSNorm (``model.norm``)
    - Linear projection to vocab (``lm_head``)
    """

    # ─── Weight conversion ───────────────────────────────────────────────

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor], ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        """Convert HF weights for final norm + lm_head projection.

        Expected keys in state_dict:
        - ``norm.weight`` — final RMSNorm gamma
        - ``lm_head.weight`` — vocab projection [vocab_size, hidden_size]
        """
        norm_weight_cfg = DistributedRMSNorm.convert_weights(
            hf_config,
            tuple({k.replace("norm.", ""): v for k, v in sd.items() if k.startswith("norm.")} for sd in state_dicts),
            output_path / "final_norm",
            mesh_device,
        )

        # lm_head weight: [vocab_size, hidden_size] → [1, 1, vocab_size, hidden_size]
        assert len(state_dicts) == 1, "Only one state dict expected for LM head (single-row shard)"
        lm_head_weight = state_dicts[0]["lm_head.weight"].to(torch.bfloat16)
        vocab_size, hidden_dim = lm_head_weight.shape
        lm_head_weight = lm_head_weight.unsqueeze(0).unsqueeze(0).contiguous()

        lm_head_weight_cfg = {
            "input_tensor_b": shard_and_save(
                output_path / "lm_head.input_tensor_b",
                lm_head_weight,
                shard_dims=(None, -2),
                mesh_device=mesh_device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        }

        return {
            "final_norm": norm_weight_cfg,
            "lm_head": lm_head_weight_cfg,
        }

    # ─── Model config ────────────────────────────────────────────────────

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        batch_size_per_row: int,
    ) -> ModelDecodeConfig:
        """Generate decode-mode config for final norm + lm_head."""
        hidden_dim = hf_config.hidden_size
        vocab_size = hf_config.vocab_size
        tile_size = 32
        mesh_cols = mesh_device.shape[1]

        # Final norm config (same as per-layer norms)
        norm_config = DistributedRMSNorm.decode_model_config(
            hf_config, mesh_device, batch_size_per_row=batch_size_per_row
        )

        # LM head linear config
        n_per_device = vocab_size // mesh_cols if mesh_cols > 0 else vocab_size
        # Pad vocab per device to tile boundary
        n_per_device_padded = math.ceil(n_per_device / tile_size) * tile_size

        # MatmulMultiCoreReuseMultiCast1DProgramConfig with a full 119B-class vocab on few cores
        # oversubscribes L1 circular buffers (TT_THROW in program.cpp: static CB region > max L1).
        # Use DRAM matmul (no fixed 1D reuse grid) like prefill when the per-device output width is large.
        _DRAM_LM_HEAD_DECODE_VOCAB_PAD_THRESHOLD = 4096
        use_dram_lm_decode = n_per_device_padded > _DRAM_LM_HEAD_DECODE_VOCAB_PAD_THRESHOLD

        if use_dram_lm_decode:
            lm_head_config = LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                transpose_b=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            )
            logits_mem = ttnn.DRAM_MEMORY_CONFIG
            gather_mem = ttnn.DRAM_MEMORY_CONFIG
        else:
            K_tiles = hidden_dim // tile_size
            N_tiles = n_per_device_padded // tile_size

            grid_size = mesh_device.compute_with_storage_grid_size()
            num_cores = grid_size.x * grid_size.y
            per_core_N = math.ceil(N_tiles / num_cores)

            in0_block_w = 32
            while K_tiles % in0_block_w != 0 and in0_block_w > 1:
                in0_block_w //= 2

            out_subblock_w = min(per_core_N, 4)
            while out_subblock_w > 1 and per_core_N % out_subblock_w != 0:
                out_subblock_w -= 1

            program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid_size.x, grid_size.y),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                per_core_M=1,
                per_core_N=per_core_N,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

            lm_head_config = LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                transpose_b=True,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
                program_config=program_config,
            )
            logits_mem = ttnn.L1_MEMORY_CONFIG
            gather_mem = ttnn.L1_MEMORY_CONFIG

        return {
            "final_norm": norm_config,
            "lm_head": lm_head_config,
            "all_gather": AllGatherAsyncConfig(
                mesh_device=MeshDeviceStub(mesh_device.shape),
                cluster_axis=1,
                dim=-1,
                memory_config=gather_mem,
            ),
            "input_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "output_memory_config": logits_mem,
        }

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelPrefillConfig:
        """Generate prefill-mode config for final norm + lm_head."""
        norm_config = DistributedRMSNorm.prefill_model_config(hf_config, mesh_device)

        lm_head_config = LinearConfig(
            input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
        )

        return {
            "final_norm": norm_config,
            "lm_head": lm_head_config,
            "all_gather": AllGatherAsyncConfig(
                mesh_device=MeshDeviceStub(mesh_device.shape),
                cluster_axis=1,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "input_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "output_memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }

    # ─── State ───────────────────────────────────────────────────────────

    @classmethod
    def create_state(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice, ccl: CCL) -> ModelState:
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "final_norm": DistributedRMSNorm.create_state(hf_config, mesh_device, ccl),
            "ccl": ccl,
        }

    # ─── Forward ─────────────────────────────────────────────────────────

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        """Decode forward: final_norm → lm_head linear → (optional all_gather)."""
        # Final RMSNorm
        x = DistributedRMSNorm.forward_decode(x, cfg["final_norm"])

        # LM head projection
        logits = ttnn.linear(x, **cfg["lm_head"])
        ttnn.deallocate(x)

        # All-gather across column axis if multi-device
        ag_cfg = cfg.get("all_gather")
        if ag_cfg is not None and all_gather_mesh_extent_on_cluster_axis(ag_cfg) > 1:
            ccl = cfg["ccl"]
            logits = ttnn.experimental.all_gather_async(logits, **ccl.populate_all_gather_runtime_args(ag_cfg))

        return logits

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        """Prefill forward: final_norm → lm_head linear → (optional all_gather)."""
        # Final RMSNorm
        x = DistributedRMSNorm.forward_prefill(x, cfg["final_norm"])

        # LM head projection
        logits = ttnn.linear(x, **cfg["lm_head"])
        ttnn.deallocate(x)

        # All-gather across column axis if multi-device
        ag_cfg = cfg.get("all_gather")
        if ag_cfg is not None and all_gather_mesh_extent_on_cluster_axis(ag_cfg) > 1:
            ccl = cfg["ccl"]
            logits = ttnn.experimental.all_gather_async(logits, **ccl.populate_all_gather_runtime_args(ag_cfg))

        return logits
