# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-27B Decoder Layer for TTNN.

Replaces Qwen3_5DecoderLayer to perform residual adds on-device using ttnn.add,
eliminating host round-trips that force device synchronization.

Handles both layer types in the hybrid architecture:
- linear_attention: GDN (Gated DeltaNet) via TTNNQwen35GatedDeltaNet
- full_attention: GQA via TTNNQwen35FullAttention
"""

import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import trace_enabled, DistributedTensorConfig
from models.experimental.tt_symbiote.core.utils import tree_map
from models.experimental.tt_symbiote.modules.qwen35_attention import TTNNQwen35FullAttention
from models.experimental.tt_symbiote.modules.qwen35_gated_deltanet import TTNNQwen35GatedDeltaNet
from models.experimental.tt_symbiote.modules.qwen35_mlp import TTNNQwen35MLP
from models.experimental.tt_symbiote.modules.qwen35_normalization import TTNNQwen35RMSNorm


@trace_enabled
class TTNNQwen35DecoderLayer(TTNNModule):
    """Replaces Qwen3_5DecoderLayer to keep residual adds on-device.

    Eliminates 2 host round-trips per layer (one for attention residual,
    one for MLP residual) by using ttnn.add instead of aten::add.

    Handles both linear_attention (GDN) and full_attention (GQA) layer types.
    """

    def __init__(self):
        super().__init__()
        self.layer_type = None  # "linear_attention" or "full_attention"
        self.hidden_size = None
        self.rotary_dim = None  # Expected full rotary dim for detecting sharded cos/sin
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.attention = None
        self.mlp = None

    @property
    def _is_distributed(self):
        """Check if running in distributed mode with CCL manager."""
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        """All-gather tensor across mesh devices if actually sharded."""
        if not self._is_distributed:
            return tensor
        if self.hidden_size is not None and tensor.shape[-1] >= self.hidden_size:
            return tensor
        gathered = ttnn.all_gather(
            tensor,
            dim=-1,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
        )
        ttnn.synchronize_device(self.device)
        return gathered

    def set_output_tensors_config_impl(self, output_tensors):
        """Set col-sharded output config for tensor-parallel decoder output.

        Attention and MLP outputs are col-sharded (each device has hidden_size/N).
        ConcatMeshToTensor(dim=-1) reconstructs the full hidden dimension.
        """
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if self._is_distributed and self.device is not None:
                    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
                    mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=-1)

                    num_devices = self.device.get_num_devices()

                    def logical_shape_for_col_sharded(shape):
                        shape_list = list(shape)
                        shape_list[-1] = shape_list[-1] * num_devices
                        return tuple(shape_list)

                    config = DistributedTensorConfig(
                        mesh_mapper=mesh_mapper,
                        mesh_composer=mesh_composer,
                        logical_shape_fn=logical_shape_for_col_sharded,
                    )
                    e.set_distributed_tensor_config(config)
            return e

        if not self._is_distributed:
            return super().set_output_tensors_config_impl(output_tensors)

        return tree_map(set_col_sharded_config, output_tensors)

    @classmethod
    def from_torch(cls, torch_layer):
        """Create from Qwen3_5DecoderLayer.

        Args:
            torch_layer: HuggingFace Qwen3_5DecoderLayer instance.
        """
        new_layer = cls()
        new_layer._fallback_torch_layer = torch_layer
        new_layer.layer_type = torch_layer.layer_type
        new_layer.hidden_size = torch_layer.mlp.gate_proj.in_features

        # Replace sub-modules with TTNN equivalents
        new_layer.input_layernorm = TTNNQwen35RMSNorm.from_torch(torch_layer.input_layernorm)
        new_layer.post_attention_layernorm = TTNNQwen35RMSNorm.from_torch(torch_layer.post_attention_layernorm)
        new_layer.mlp = TTNNQwen35MLP.from_torch(torch_layer.mlp)

        if torch_layer.layer_type == "linear_attention":
            new_layer.attention = TTNNQwen35GatedDeltaNet.from_torch(torch_layer.linear_attn)
        else:
            new_layer.attention = TTNNQwen35FullAttention.from_torch(torch_layer.self_attn)
            # Store expected rotary dim to detect inadvertent sharding of cos/sin
            config = torch_layer.self_attn.config
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            new_layer.rotary_dim = int(head_dim * partial_rotary_factor)

        return new_layer

    def _all_gather_if_sharded(self, tensor):
        """All-gather a tensor on dim=-1 if it was inadvertently sharded.

        The framework's default ShardTensor2dMesh config shards ALL input tensors
        on dim=-1. For T3K (8 devices), this splits the last dimension by 8.
        Tensors that should be replicated (cos/sin, attention_mask) need to be
        reconstructed via all-gather.

        Only gathers if the tensor's last dim is smaller than hidden_size (indicating
        it was sharded rather than replicated).
        """
        if tensor is None or not self._is_distributed:
            return tensor
        if self.hidden_size is not None and tensor.shape[-1] >= self.hidden_size:
            return tensor
        gathered = ttnn.all_gather(
            tensor,
            dim=-1,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
        )
        ttnn.synchronize_device(self.device)
        return gathered

    def _all_gather_position_embeddings(self, position_embeddings):
        """All-gather position embeddings that were inadvertently sharded.

        The framework's default ShardTensor2dMesh config shards ALL input tensors
        on dim=-1, including cos/sin position embeddings which must be replicated.
        For T3K (8 devices), cos/sin with rotary_dim=64 get sharded to 8 per device,
        causing RoPE to see the wrong rotary dimension.
        """
        if position_embeddings is None or not self._is_distributed:
            return position_embeddings
        cos, sin = position_embeddings
        # Only gather if cos/sin appear sharded (last dim < expected rotary_dim)
        if self.rotary_dim is not None and cos.shape[-1] >= self.rotary_dim:
            return position_embeddings
        cos = ttnn.all_gather(
            cos,
            dim=-1,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
        )
        sin = ttnn.all_gather(
            sin,
            dim=-1,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
        )
        ttnn.synchronize_device(self.device)
        return (cos, sin)

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        """Forward pass with col-sharded tensor-parallel residual adds.

        Uses standard TP pattern: input arrives col-sharded, all-gather for
        layernorm, attention/MLP return col-sharded output, residual adds
        operate on col-sharded tensors.

        Args:
            hidden_states: Input tensor, col-sharded [batch, seq, hidden/N] per device.
            position_embeddings: Tuple of (cos, sin) for RoPE (full attention only).
            attention_mask: Attention mask tensor.
            position_ids: Position IDs (unused by GDN, used by full attention).
            past_key_values: Paged KV cache (full attention) or cache_params (GDN).
            cache_position: Cache position tensor for paged decode.

        Returns:
            hidden_states tensor, col-sharded [batch, seq, hidden/N] per device.
        """
        hs = hidden_states

        # Ensure TILE layout for TTNN ops
        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if hs.dtype != ttnn.bfloat16:
            hs = ttnn.typecast(hs, ttnn.bfloat16)

        # Save col-sharded residual BEFORE all-gather (matches attention/MLP output shape)
        residual = hs

        # All-gather for layernorm (needs full hidden dim for correct RMS)
        hs = self._maybe_all_gather(hs)

        # All-gather position embeddings (cos/sin) which get inadvertently sharded
        # by the framework's default ShardTensor2dMesh on dim=-1
        position_embeddings = self._all_gather_position_embeddings(position_embeddings)

        # All-gather attention_mask if sharded (same default ShardTensor2dMesh issue)
        attention_mask = self._all_gather_if_sharded(attention_mask)

        # Input layernorm (on full tensor)
        hs = self.input_layernorm(hs)

        # Attention (dispatch based on layer type)
        # Output is col-sharded [batch, seq, hidden/N] — no all-gather in attention
        if self.layer_type == "linear_attention":
            attn_out = self.attention(
                hidden_states=hs,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
        else:
            attn_out, _ = self.attention(
                hidden_states=hs,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )

        # Residual add ON DEVICE (both col-sharded)
        hs = ttnn.add(residual, attn_out)
        ttnn.deallocate(attn_out)
        # NOTE: Do NOT deallocate residual here — it is the pre-allocated trace
        # input buffer.

        # Save col-sharded residual
        residual = hs

        # All-gather for post-attention layernorm (needs full hidden dim)
        hs = self._maybe_all_gather(hs)

        # Post-attention layernorm (on full tensor)
        hs = self.post_attention_layernorm(hs)

        # MLP — output is col-sharded [batch, seq, hidden/N]
        mlp_out = self.mlp(hs)

        # Residual add ON DEVICE (both col-sharded)
        hs = ttnn.add(residual, mlp_out)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(residual)

        return hs
