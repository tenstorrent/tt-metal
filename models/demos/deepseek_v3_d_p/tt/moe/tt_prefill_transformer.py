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

from pathlib import Path
from typing import Optional

import torch
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
        routed_expert_activations_dtype=ttnn.bfloat16,
        routed_expert_weights_dtype=ttnn.bfloat16,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat16,
        weight_cache_path: Optional[Path] = None,
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
            weight_cache_path=weight_cache_path,
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
                routed_expert_activations_dtype=routed_expert_activations_dtype,
                routed_expert_weights_dtype=routed_expert_weights_dtype,
                shared_expert_activations_dtype=shared_expert_activations_dtype,
                shared_expert_weights_dtype=shared_expert_weights_dtype,
                weight_cache_path=weight_cache_path,
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
            weight_cache_path=weight_cache_path,
            cache_name_prefix="norm",
        )

        # --- RoPE (computed once, reused across all layers) ---
        self.rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)

        logger.info(f"TtPrefillTransformer construction complete ({num_layers} layers)")

    def _to_host(self, tt_tensor):
        """Bring SP+TP sharded tensor to host as [1, seq, emb] bfloat16."""
        ndim = len(tt_tensor.shape)
        dims = (2, 3) if ndim == 4 else (1, 2)
        host = ttnn.to_torch(
            tt_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=dims, mesh_shape=self.mesh_device.shape),
        ).to(torch.bfloat16)
        if ndim == 4:
            host = host.squeeze(0)
        return host

    def forward(self, token_ids: ttnn.Tensor, return_intermediates: bool = False, return_kv_cache: bool = False):
        """
        Forward pass: embed -> [block x N] -> norm.

        Args:
            token_ids: [1, 1, seq_len_per_chip] uint32, SP-sharded
            return_intermediates: if True, sync + snapshot to host after each stage
            return_kv_cache: if True, collect per-layer KVPE caches

        Returns:
            If return_intermediates=False and return_kv_cache=False:
                tt_output: [1, 1, seq_per_chip, emb_dim/tp] TILE_LAYOUT
            If return_intermediates=True and return_kv_cache=False:
                (tt_output, intermediates)
            If return_intermediates=True and return_kv_cache=True:
                (tt_output, intermediates, kv_caches)
            If return_intermediates=False and return_kv_cache=True:
                (tt_output, kv_caches)
        """
        rope_tensors = self.rope_setup.get_rope_tensors(self.seq_len)
        intermediates = [] if return_intermediates else None
        kv_caches = [] if return_kv_cache else None

        h = self.embed(token_ids)  # [1, seq_per_chip, emb_dim/tp]
        h = ttnn.unsqueeze_to_4D(h)  # [1, 1, seq_per_chip, emb_dim/tp]

        if return_intermediates:
            ttnn.synchronize_device(self.mesh_device)
            intermediates.append(("embed", self._to_host(h)))

        for i, layer in enumerate(self.layers):
            signpost(f"forward_layer_{i}_start")
            h, layer_kv_cache = layer(h, rope_tensors, return_kv_cache=return_kv_cache)
            signpost(f"forward_layer_{i}_end")
            if return_intermediates or return_kv_cache:
                ttnn.synchronize_device(self.mesh_device)
            if return_intermediates:
                intermediates.append((f"layer_{i}", self._to_host(h)))
            if return_kv_cache:
                kv_caches.append((f"layer_{i}_kvpe", layer_kv_cache))

        h = self.norm(h)

        if return_intermediates:
            ttnn.synchronize_device(self.mesh_device)
            intermediates.append(("norm", self._to_host(h)))
            if return_kv_cache:
                return h, intermediates, kv_caches
            return h, intermediates
        if return_kv_cache:
            return h, kv_caches
        return h
