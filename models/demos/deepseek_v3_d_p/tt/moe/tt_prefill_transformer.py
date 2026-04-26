# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TtPrefillTransformer — multi-layer prefill model for DeepSeek V3.

Composes: embed -> [block x N] -> norm

Equivalent to the reference Transformer class (models/demos/deepseek_v3/reference/deepseek/model.py:419)
but targeting the TT prefill path with SP+TP parallelism.

No LM head — returns hidden states after final norm.
"""

import os
from pathlib import Path
from typing import Callable, Optional

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
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker


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

    @staticmethod
    def check_cache_complete(
        cache_path: Path | None,
        num_layers: int,
        experts_per_chip: int = 8,
        first_k_dense: int = 3,
    ) -> bool:
        """
        Top-level cache completeness check for the full transformer.

        Checks embedding, all blocks (norms + MLA + FFN/MoE), and final norm.
        Replaces the monolithic check_ttnn_cache_complete from cache_utils.py.

        Args:
            cache_path: Path to TTNN weight cache directory
            num_layers: Number of transformer layers
            experts_per_chip: Number of routed experts per chip (default: 8)
            first_k_dense: Number of initial dense (non-MoE) layers (default: 3)

        Returns:
            True if all expected cache files exist, False otherwise
        """
        if not cache_path or not cache_path.exists():
            logger.debug(f"TTNN cache path does not exist: {cache_path}")
            return False

        # Initialize fast cache checker for this directory
        init_checker(cache_path)

        # Embedding
        if not TtParallelEmbedding.check_cache_complete(cache_path):
            return False

        # Per-layer blocks
        for layer_idx in range(num_layers):
            is_dense = layer_idx < first_k_dense
            if not TtPrefillBlock.check_cache_complete(cache_path, layer_idx, is_dense, experts_per_chip):
                return False

        # Final norm
        if not TtDistributedRmsNorm.check_cache_complete(cache_path, "norm"):
            return False

        logger.info(f"TTNN cache complete at {cache_path} ({num_layers} layers)")
        return True

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
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        weight_cache_path: Optional[Path] = None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.seq_len = seq_len

        # Log environment variables that define reference output cache and TTNN weights cache.
        # This is to prevent accidental cache creation at unusal places and fill disk space.
        TT_DS_PREFILL_TTNN_CACHE = os.getenv("TT_DS_PREFILL_TTNN_CACHE", None)
        TT_DS_PREFILL_HOST_REF_CACHE = os.getenv("TT_DS_PREFILL_HOST_REF_CACHE", None)

        logger.debug(f"{TT_DS_PREFILL_TTNN_CACHE=}")
        logger.debug(f"{TT_DS_PREFILL_HOST_REF_CACHE=}")
        if TT_DS_PREFILL_TTNN_CACHE is None:
            logger.error(f"TT_DS_PREFILL_TTNN_CACHE environment variable is not set; export TT_DS_PREFILL_TTNN_CACHE=")
            import pytest

            pytest.fail(f"{TT_DS_PREFILL_TTNN_CACHE=}")
        if TT_DS_PREFILL_HOST_REF_CACHE is None:
            logger.error(
                f"TT_DS_PREFILL_HOST_REF_CACHE environment variable is not set; export TT_DS_PREFILL_HOST_REF_CACHE="
            )
            import pytest

            pytest.fail(f"{TT_DS_PREFILL_HOST_REF_CACHE=}")

        logger.info(f"Building TtPrefillTransformer with {num_layers} layers, seq_len={seq_len}")

        # --- Embedding ---
        self.embed = TtParallelEmbedding(
            mesh_device=mesh_device,
            vocab_size=config.vocab_size,
            emb_dim=config.hidden_size,
            torch_weight=state_dict.get("embed_weight"),  # None if cache exists
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            weight_cache_path=weight_cache_path,
        )

        # --- Transformer layers ---
        self.layers = []
        for i in range(num_layers):
            logger.info(f"Building layer {i}/{num_layers}...")
            # Get layer weights or empty dict if loading from cache
            layer_state = state_dict["layers"][i] if state_dict.get("layers") else {}
            layer = TtPrefillBlock(
                mesh_device=mesh_device,
                config=config,
                state_dict=layer_state,
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
            torch_weight=state_dict.get("norm_weight"),  # None if cache exists
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
        host = ttnn.to_torch(
            tt_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(-2, -1), mesh_shape=self.mesh_device.shape),
        ).to(torch.bfloat16)
        if len(tt_tensor.shape) == 4:
            host = host.squeeze(0)
        return host

    def forward(
        self,
        token_ids: ttnn.Tensor,
        kvpe_cache: ttnn.Tensor,
        return_intermediates: bool = False,
        read_profiler: bool = False,
        on_layer_complete: Optional[Callable[[int, ttnn.Tensor], None]] = None,
        actual_isl: Optional[int] = None,
    ):
        """
        Forward pass: embed -> [block x N] -> norm.

        Args:
            token_ids: [1, 1, seq_len_per_chip] uint32, SP-sharded
            kvpe_cache: externally created KVPE cache [num_layers, 1, seq_len_local, head_dim];
                        each layer writes to its own slot via cache_layer_idx
            return_intermediates: if True, sync + snapshot to host after each stage
            read_profiler: if True, read TTNN profiler after each layer to avoid profiler buffer overflows
            on_layer_complete: optional callback invoked by MLA after fill_cache_for_user_().
                Called as on_layer_complete(layer_idx, kvpe_cache). Used for KV cache
                migration in disaggregated prefill/decode. When set, MLA also zeros
                the padding region of the cache before fill so migration sees valid KV
                + zero padding. When None, no migration or zeroing.
            actual_isl: actual (unpadded) input sequence length. Required when
                on_layer_complete is set (MLA uses it to compute padding region).

        Returns:
            If return_intermediates=False:
                tt_output: [1, 1, seq_per_chip, emb_dim/tp] TILE_LAYOUT
            If return_intermediates=True:
                (tt_output, intermediates)
        """
        rope_tensors = self.rope_setup.get_rope_tensors(self.seq_len)
        intermediates = [] if return_intermediates else None

        h = self.embed(token_ids)  # [1, seq_per_chip, emb_dim/tp]
        h = ttnn.unsqueeze_to_4D(h)  # [1, 1, seq_per_chip, emb_dim/tp]

        if return_intermediates:
            ttnn.synchronize_device(self.mesh_device)
            intermediates.append(("embed", self._to_host(h)))

        for i, layer in enumerate(self.layers):
            signpost(f"forward_layer_{i}_start")
            h, _ = layer(
                h,
                rope_tensors,
                kvpe_cache,
                cache_layer_idx=i,
                on_layer_complete=on_layer_complete,
                actual_isl=actual_isl,
            )
            signpost(f"forward_layer_{i}_end")
            if return_intermediates:
                ttnn.synchronize_device(self.mesh_device)
                intermediates.append((f"layer_{i}", self._to_host(h)))
            if read_profiler:
                ttnn.ReadDeviceProfiler(self.mesh_device)

        h = self.norm(h)

        if return_intermediates:
            ttnn.synchronize_device(self.mesh_device)
            intermediates.append(("norm", self._to_host(h)))
            return h, intermediates
        return h
