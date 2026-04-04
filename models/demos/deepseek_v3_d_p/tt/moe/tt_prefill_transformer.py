# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TtPrefillTransformer — multi-layer prefill model for DeepSeek V3.

Composes: embed -> [block x N] -> norm

Equivalent to the reference Transformer class (models/demos/deepseek_v3/reference/deepseek/model.py:419)
but targeting the TT prefill path with SP+TP parallelism.

No LM head — returns hidden states after final norm.
"""

from loguru import logger
from tracy import signpost
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding


class TtPrefillTransformer(LightweightModule):
    """
    Multi-layer prefill transformer for DeepSeek V3.

    Architecture: embed -> [TtPrefillBlock x num_layers] -> norm

    State dict keys:
        embed_weight:   torch.Tensor [vocab_size, emb_dim]
        norm_weight:    torch.Tensor [emb_dim]
        layers:         list[dict] — per-layer state dicts for TtPrefillBlock

    Note: LM head (ColumnParallelLinear output projection) is not implemented yet.
    TODO: Add LM head after https://github.com/tenstorrent/tt-metal/pull/41275 lands.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: PretrainedConfig,
        state_dict: dict,
        num_layers: int,
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
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.seq_len = seq_len

        logger.info(f"Building TtPrefillTransformer with {num_layers} layers, seq_len={seq_len}")

        # --- Embedding ---
        self.embed = TtParallelEmbedding(
            mesh_device=mesh_device,
            vocab_size=config.vocab_size,
            emb_dim=config.hidden_size,
            torch_weight=state_dict["embed_weight"],
            sp_axis=sp_axis,
            tp_axis=tp_axis,
        )

        # --- Transformer layers ---
        self.layers = []
        for i in range(num_layers):
            logger.info(f"Building layer {i}/{num_layers}...")
            layer = TtPrefillBlock(
                mesh_device=mesh_device,
                config=config,
                state_dict=state_dict["layers"][i],
                layer_idx=i,
                seq_len=seq_len,
                num_links=num_links,
                topology=topology,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                is_balanced=is_balanced,
                capacity_factor=capacity_factor,
                gate_fallback_mode=gate_fallback_mode,
                activations_dtype=activations_dtype,
                weights_dtype=weights_dtype,
            )
            self.layers.append(layer)

        # --- Final norm ---
        self.norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=config.hidden_size,
            torch_weight=state_dict["norm_weight"],
            cluster_axis=tp_axis,
            num_links=num_links,
            topology=topology,
        )

        # --- RoPE (computed once, reused across all layers) ---
        self.rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)

        logger.info(f"TtPrefillTransformer construction complete ({num_layers} layers)")

    def forward(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass: embed -> [block x N] -> norm.

        Args:
            token_ids: [1, 1, seq_len_per_chip] uint32, SP-sharded

        Returns:
            [1, 1, seq_len_per_chip, emb_dim_per_tp] TILE_LAYOUT
        """
        rope_tensors = self.rope_setup.get_rope_tensors(self.seq_len)

        h = self.embed(token_ids)  # [1, seq_per_chip, emb_dim/tp]
        h = ttnn.unsqueeze_to_4D(h)  # [1, 1, seq_per_chip, emb_dim/tp]

        for i, layer in enumerate(self.layers):
            signpost(f"forward_layer_{i}_start")
            h = layer(h, rope_tensors)
            signpost(f"forward_layer_{i}_end")
        h = self.norm(h)
        return h
