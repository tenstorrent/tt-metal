# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Optional

from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants, extract_mesh_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_ffn import TtFfn


class TtPrefillBlock(LightweightModule):
    """
    Single transformer block for DeepSeek V3 prefill.

    Constructs and composes: attn_norm → MLA → residual → ffn_norm → FFN/MoE → residual

    Canonical tensor format: [1, 1, seq_len_local, emb_dim_per_tp], TILE_LAYOUT, TP-sharded on last dim.

    State dict keys:
        attn_norm_weight:       torch.Tensor [emb_dim]
        mla_weights:            dict for ttMLA (q_a_proj.weight, q_b_proj.weight, ...)
        ffn_norm_weight:        torch.Tensor [emb_dim]
        For MoE layers (layer_idx >= n_dense_layers):
            gate_weights:           dict with "weight" and "e_score_correction_bias"
            routed_expert_weights:  list[dict] with "gate_proj", "up_proj", "down_proj" per expert
            shared_expert_weights:  dict with "gate_proj", "up_proj", "down_proj"
        For dense layers (layer_idx < n_dense_layers):
            ffn_weights:            dict with "gate_proj", "up_proj", "down_proj"
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: PretrainedConfig,
        state_dict: dict,
        layer_idx: int,
        seq_len: int,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        sp_axis: int = 0,
        tp_axis: int = 1,
        is_balanced: bool = False,
        capacity_factor: int = 2,
        gate_fallback_mode: GateComputeMode = GateComputeMode.HOST_ALL,
        activations_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        weight_cache_path: Optional[Path] = None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        self.is_moe = layer_idx >= DeepSeekV3Config.NUM_DENSE_LAYERS

        emb_dim = config.hidden_size

        logger.info(f"Building TtPrefillBlock layer_idx={layer_idx} ({'MoE' if self.is_moe else 'dense'})")

        # --- Attention norm ---
        self.attn_norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=emb_dim,
            torch_weight=state_dict["attn_norm_weight"],
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.attn_norm",
        )

        # --- MLA ---
        self.mla = ttMLA(
            config,
            state_dict["mla_weights"],
            mesh_device,
            layer_idx=layer_idx,
            seq_len=seq_len,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            is_balanced=is_balanced,
            weight_cache_path=weight_cache_path,
        )

        # --- FFN norm ---
        self.ffn_norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=emb_dim,
            torch_weight=state_dict["ffn_norm_weight"],
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=topology,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.ffn_norm",
        )

        # --- FFN (MoE or dense) ---
        if self.is_moe:
            self.ffn = self._build_moe(
                mesh_device=mesh_device,
                state_dict=state_dict,
                seq_len=seq_len,
                sp_axis=sp_axis,
                emb_dim=emb_dim,
                num_links=num_links,
                topology=topology,
                capacity_factor=capacity_factor,
                gate_fallback_mode=gate_fallback_mode,
                activations_dtype=activations_dtype,
                weights_dtype=weights_dtype,
                weight_cache_path=weight_cache_path,
                layer_idx=layer_idx,
            )
        else:
            self.ffn = TtFfn(
                mesh_device=mesh_device,
                torch_weights=state_dict["ffn_weights"],
                num_links=num_links,
                topology=topology,
                weight_cache_path=weight_cache_path,
                cache_name_prefix=f"layer_{layer_idx}.ffn",
            )

    @staticmethod
    def _build_moe(
        mesh_device,
        state_dict,
        seq_len,
        sp_axis,
        emb_dim,
        num_links,
        topology,
        capacity_factor,
        gate_fallback_mode,
        activations_dtype,
        weights_dtype,
        weight_cache_path=None,
        layer_idx=0,
    ):
        mesh_config = extract_mesh_config(mesh_device)
        sp_factor = mesh_device.shape[sp_axis]
        seq_len_per_chip = seq_len // sp_factor

        experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
            seq_len_per_chip=seq_len_per_chip,
            num_routed_experts=DeepSeekV3Config.NUM_ROUTED_EXPERTS,
            num_experts_per_tok=DeepSeekV3Config.NUM_EXPERTS_PER_TOKEN,
            num_devices=mesh_device.get_num_devices(),
            dispatch_group_size=mesh_config.dispatch_group_size,
            capacity_factor=capacity_factor,
        )

        return TtMoe(
            mesh_device=mesh_device,
            dispatch_group_size=mesh_config.dispatch_group_size,
            num_dispatch_groups=mesh_config.num_dispatch_groups,
            experts_per_chip=experts_per_chip,
            num_routed_experts=DeepSeekV3Config.NUM_ROUTED_EXPERTS,
            num_experts_per_tok=DeepSeekV3Config.NUM_EXPERTS_PER_TOKEN,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            hidden_dim=DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
            num_links=num_links,
            topology=topology,
            routed_expert_weights=state_dict["routed_expert_weights"],
            shared_expert_weights=state_dict["shared_expert_weights"],
            activations_dtype=activations_dtype,
            weights_dtype=weights_dtype,
            gate_weights=state_dict["gate_weights"],
            gate_fallback_mode=gate_fallback_mode,
            weight_cache_path=weight_cache_path,
            layer_idx=layer_idx,
        )

    def forward(self, x: ttnn.Tensor, rope_tensors: dict) -> ttnn.Tensor:
        """
        Args:
            x: [1, 1, seq_len_local, emb_dim_per_tp] TILE_LAYOUT, TP-sharded
            rope_tensors: dict with keys "cos_matrix", "sin_matrix", "trans_matrix"

        Returns:
            [1, 1, seq_len_local, emb_dim_per_tp] TILE_LAYOUT, TP-sharded
        """
        # --- Attention ---
        attn_norm_out = self.attn_norm(x)
        mla_out = self.mla.forward(attn_norm_out, rope_tensors)
        ttnn.deallocate(attn_norm_out)
        x = ttnn.add(x, mla_out)
        ttnn.deallocate(mla_out)

        # --- FFN ---
        ffn_norm_out = self.ffn_norm(x)

        if self.is_moe:
            ffn_out = self._moe_path(ffn_norm_out)
        else:
            ffn_out = self._dense_ffn_path(ffn_norm_out)

        ttnn.deallocate(ffn_norm_out)
        x = ttnn.add(x, ffn_out)
        ttnn.deallocate(ffn_out)

        return x

    def _moe_path(self, ffn_norm_out: ttnn.Tensor) -> ttnn.Tensor:
        """MoE FFN path: 4D TILE → 3D ROW_MAJOR → MoE → 3D TILE → 4D TILE."""
        moe_input = ttnn.squeeze(ffn_norm_out, dim=0)
        moe_input = ttnn.to_layout(moe_input, ttnn.ROW_MAJOR_LAYOUT)

        moe_out, _ = self.ffn(moe_input)

        moe_out = ttnn.to_layout(moe_out, ttnn.TILE_LAYOUT)
        moe_out = ttnn.unsqueeze(moe_out, dim=0)
        return moe_out

    def _dense_ffn_path(self, ffn_norm_out: ttnn.Tensor) -> ttnn.Tensor:
        """Dense FFN path: all_gather to get full emb_dim → TtFfn (does reduce_scatter internally)."""
        gathered = ttnn.all_gather(
            ffn_norm_out,
            dim=-1,
            cluster_axis=1,
            num_links=self.num_links,
            topology=self.topology,
        )
        ffn_out = self.ffn(gathered)
        ttnn.deallocate(gathered)
        return ffn_out
