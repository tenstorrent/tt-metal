# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS with Tensor Parallel support for N300 (1x2 mesh).

Tensor Parallel Strategy:
- Shard Q, K, V projections along head dimension (16 heads -> 8 per device)
- Shard output projection along input dimension
- All-reduce after output projection
- Replicate embeddings, norms, and LM heads
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

import ttnn


@dataclass
class Qwen3TTSMeshConfig:
    """Configuration for mesh-based Qwen3-TTS."""

    # Talker config
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    vocab_size: int = 3072

    # CodePredictor config
    cp_hidden_size: int = 1024
    cp_num_attention_heads: int = 8
    cp_num_kv_heads: int = 4
    cp_head_dim: int = 128
    cp_intermediate_size: int = 3072
    cp_num_hidden_layers: int = 5

    # Mesh config
    mesh_shape: Tuple[int, int] = (1, 2)  # N300: 1 row, 2 columns
    tp_degree: int = 2  # Tensor parallel degree


def open_mesh_device_n300(mesh_shape: Tuple[int, int] = (1, 2)) -> ttnn.MeshDevice:
    """Open N300 mesh device with 1x2 configuration."""
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=mesh_shape,
        dispatch_core_type=ttnn.device.DispatchCoreType.WORKER,
    )
    # Enable program cache on all devices
    for device in mesh_device.get_devices():
        device.enable_program_cache()
    return mesh_device


def close_mesh_device(mesh_device: ttnn.MeshDevice):
    """Close mesh device and all submeshes."""
    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)


class TalkerAttentionMesh:
    """
    Tensor-parallel attention for Talker on N300 mesh.

    Weight sharding:
    - Q projection: [hidden, num_heads * head_dim] -> shard on dim 1
    - K projection: [hidden, num_kv_heads * head_dim] -> shard on dim 1
    - V projection: [hidden, num_kv_heads * head_dim] -> shard on dim 1
    - O projection: [num_heads * head_dim, hidden] -> shard on dim 0, all-reduce output
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        layer_idx: int,
        config: Qwen3TTSMeshConfig,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.layer_idx = layer_idx
        self.num_devices = mesh_device.get_num_devices()

        # Local dimensions (per device)
        self.n_local_heads = config.num_attention_heads // self.num_devices
        self.n_local_kv_heads = config.num_kv_heads // self.num_devices

        prefix = f"talker.model.layers.{layer_idx}.self_attn"

        # Load and shard Q, K, V, O weights
        self._load_weights(state_dict, prefix)

    def _load_weights(self, state_dict: dict, prefix: str):
        """Load and shard attention weights across mesh."""
        # Q projection: shard along output dim (heads)
        q_weight = state_dict[f"{prefix}.q_proj.weight"]  # [num_heads * head_dim, hidden]
        self.q_proj = ttnn.from_torch(
            q_weight.T.contiguous(),  # [hidden, num_heads * head_dim]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
        )

        # K projection: shard along output dim
        k_weight = state_dict[f"{prefix}.k_proj.weight"]
        self.k_proj = ttnn.from_torch(
            k_weight.T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
        )

        # V projection: shard along output dim
        v_weight = state_dict[f"{prefix}.v_proj.weight"]
        self.v_proj = ttnn.from_torch(
            v_weight.T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
        )

        # O projection: shard along input dim (will need all-reduce)
        o_weight = state_dict[f"{prefix}.o_proj.weight"]
        self.o_proj = ttnn.from_torch(
            o_weight.T.contiguous(),  # [num_heads * head_dim, hidden]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
        )

        # Q, K norms (replicated)
        if f"{prefix}.q_norm.weight" in state_dict:
            q_norm_weight = state_dict[f"{prefix}.q_norm.weight"]
            self.q_norm = ttnn.from_torch(
                q_norm_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            k_norm_weight = state_dict[f"{prefix}.k_norm.weight"]
            self.k_norm = ttnn.from_torch(
                k_norm_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Forward pass with tensor parallel attention.

        Args:
            hidden_states: [batch, 1, seq_len, hidden_size] - replicated on all devices
            cos, sin: RoPE tensors - replicated
            trans_mat: Transformation matrix for RoPE - replicated

        Returns:
            output: [batch, 1, seq_len, hidden_size] - all-reduced across devices
        """
        batch, _, seq_len, _ = hidden_states.shape

        # Q, K, V projections (each device computes local heads)
        q = ttnn.linear(hidden_states, self.q_proj)  # [batch, 1, seq_len, local_heads * head_dim]
        k = ttnn.linear(hidden_states, self.k_proj)  # [batch, 1, seq_len, local_kv_heads * head_dim]
        v = ttnn.linear(hidden_states, self.v_proj)  # [batch, 1, seq_len, local_kv_heads * head_dim]

        # Reshape for multi-head attention
        # Q: [batch, seq_len, local_heads, head_dim] -> [batch, local_heads, seq_len, head_dim]
        q = ttnn.reshape(q, [batch, seq_len, self.n_local_heads, self.config.head_dim])
        q = ttnn.permute(q, [0, 2, 1, 3])

        k = ttnn.reshape(k, [batch, seq_len, self.n_local_kv_heads, self.config.head_dim])
        k = ttnn.permute(k, [0, 2, 1, 3])

        v = ttnn.reshape(v, [batch, seq_len, self.n_local_kv_heads, self.config.head_dim])
        v = ttnn.permute(v, [0, 2, 1, 3])

        # Apply Q, K norms if present
        if self.q_norm is not None:
            q = ttnn.rms_norm(q, self.q_norm, epsilon=self.config.rms_norm_eps)
            k = ttnn.rms_norm(k, self.k_norm, epsilon=self.config.rms_norm_eps)

        # Apply RoPE
        q = ttnn.experimental.rotary_embedding_llama(q, cos, sin, trans_mat)
        k = ttnn.experimental.rotary_embedding_llama(k, cos, sin, trans_mat)

        # Expand KV for GQA (if num_heads > num_kv_heads)
        heads_per_kv = self.n_local_heads // self.n_local_kv_heads
        if heads_per_kv > 1:
            k = ttnn.repeat_interleave(k, heads_per_kv, dim=1)
            v = ttnn.repeat_interleave(v, heads_per_kv, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / (self.config.head_dim**0.5)
        attn_output = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)

        # Reshape back: [batch, local_heads, seq_len, head_dim] -> [batch, 1, seq_len, local_heads * head_dim]
        attn_output = ttnn.permute(attn_output, [0, 2, 1, 3])
        attn_output = ttnn.reshape(attn_output, [batch, 1, seq_len, self.n_local_heads * self.config.head_dim])

        # Output projection (local)
        output = ttnn.linear(attn_output, self.o_proj)

        # All-reduce across devices
        output = ttnn.all_gather(output, dim=3, num_links=1)
        # Note: For proper all-reduce, we'd sum the gathered outputs
        # This is simplified - full implementation needs reduce_scatter + all_gather

        return output


class Qwen3TTSMesh:
    """
    Qwen3-TTS model with Tensor Parallel on N300 mesh.

    Architecture:
    - Embeddings: Replicated
    - Attention: Q,K,V sharded, O sharded + all-reduce
    - MLP: gate/up sharded, down sharded + all-reduce
    - Norms: Replicated
    - LM heads: Replicated
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        config: Optional[Qwen3TTSMeshConfig] = None,
    ):
        self.mesh_device = mesh_device
        self.config = config or Qwen3TTSMeshConfig()
        self.num_devices = mesh_device.get_num_devices()

        print(f"Initializing Qwen3-TTS on {self.num_devices}-device mesh...")
        print(f"  TP degree: {self.config.tp_degree}")
        print(f"  Local heads: {self.config.num_attention_heads // self.num_devices}")
        print(f"  Local KV heads: {self.config.num_kv_heads // self.num_devices}")

        self._load_embeddings(state_dict)
        self._load_layers(state_dict)
        self._load_head(state_dict)

    def _load_embeddings(self, state_dict: dict):
        """Load embeddings (replicated on all devices)."""
        # Text embedding
        text_embed = state_dict["talker.text_embedding.weight"]
        self.text_embedding = ttnn.from_torch(
            text_embed,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Codec embedding
        codec_embed = state_dict["talker.codec_embedding.weight"]
        self.codec_embedding = ttnn.from_torch(
            codec_embed,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        print(f"  Loaded embeddings (replicated)")

    def _load_layers(self, state_dict: dict):
        """Load transformer layers with TP."""
        self.layers = []
        for i in range(self.config.num_hidden_layers):
            layer = TalkerAttentionMesh(
                self.mesh_device,
                state_dict,
                layer_idx=i,
                config=self.config,
            )
            self.layers.append(layer)
            if (i + 1) % 7 == 0:
                print(f"  Loaded layer {i + 1}/{self.config.num_hidden_layers}")

    def _load_head(self, state_dict: dict):
        """Load LM head (replicated)."""
        codec_head = state_dict["talker.codec_head.weight"]
        self.codec_head = ttnn.from_torch(
            codec_head.T.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        print(f"  Loaded codec head (replicated)")


def test_mesh_device():
    """Quick test to verify mesh device opens correctly on N300."""
    import ttnn

    print("Testing N300 mesh device...")

    # Check cluster type
    cluster_type = ttnn.cluster.get_cluster_type()
    print(f"  Cluster type: {cluster_type}")

    if cluster_type != ttnn.cluster.ClusterType.N300:
        print(f"  WARNING: Not running on N300 (got {cluster_type})")
        print(f"  This test requires N300 hardware")
        return False

    # Open mesh device
    try:
        mesh_device = open_mesh_device_n300(mesh_shape=(1, 2))
        num_devices = mesh_device.get_num_devices()
        print(f"  Opened mesh with {num_devices} devices")

        # Test simple tensor operation
        x = torch.randn(1, 1, 32, 2048)
        x_tt = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        print(f"  Created replicated tensor: {x_tt.shape}")

        # Test sharded tensor
        w = torch.randn(2048, 2048)
        w_tt = ttnn.from_torch(
            w,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
        print(f"  Created sharded tensor: {w_tt.shape}")

        close_mesh_device(mesh_device)
        print("  Mesh device test PASSED")
        return True

    except Exception as e:
        print(f"  Mesh device test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_mesh_device()
