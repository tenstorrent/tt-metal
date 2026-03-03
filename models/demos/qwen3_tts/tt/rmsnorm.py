# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
RMSNorm implementation for Qwen3-TTS.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32


class RMSNorm(LightweightModule):
    """
    RMSNorm for Qwen3-TTS.

    This is a simplified RMSNorm implementation for single device (N150/N300).
    """

    def __init__(
        self,
        device,
        dim: int,
        state_dict: dict,
        weight_key: str,
        eps: float = 1e-6,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.dim = dim
        self.eps = eps

        # Load weight from state dict
        # Reshape to [1, 1, dim // TILE, TILE] for TTNN rms_norm
        # The last dimension must equal TILE_WIDTH=32
        torch_weight = state_dict[weight_key].unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // TILE, TILE])

        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        cache_name = None
        if weight_cache_path is not None:
            cache_name = weight_cache_path / weight_key.replace(".", "_")

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape [batch, 1, seq_len, hidden_size]

        Returns:
            Normalized tensor of same shape
        """
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            compute_kernel_config=self.compute_kernel_config,
        )


class QKNorm(LightweightModule):
    """
    QK-Norm for attention heads.

    Applies RMSNorm to query and key tensors per-head.
    This is used in Qwen3-TTS attention for better training stability.
    """

    def __init__(
        self,
        device,
        head_dim: int,
        state_dict: dict,
        q_norm_key: str,
        k_norm_key: str,
        eps: float = 1e-6,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.head_dim = head_dim
        self.eps = eps

        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        # Q norm weight - reshape to [1, 1, head_dim // TILE, TILE] for ROW_MAJOR_LAYOUT
        # The last dimension must equal TILE_WIDTH=32
        q_weight = state_dict[q_norm_key].unsqueeze(0).view(1, 1, head_dim).reshape([1, 1, head_dim // TILE, TILE])

        q_cache_name = None
        if weight_cache_path is not None:
            q_cache_name = weight_cache_path / q_norm_key.replace(".", "_")

        self.q_norm_weight = ttnn.as_tensor(
            q_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=q_cache_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        # K norm weight - reshape to [1, 1, head_dim // TILE, TILE] for ROW_MAJOR_LAYOUT
        k_weight = state_dict[k_norm_key].unsqueeze(0).view(1, 1, head_dim).reshape([1, 1, head_dim // TILE, TILE])

        k_cache_name = None
        if weight_cache_path is not None:
            k_cache_name = weight_cache_path / k_norm_key.replace(".", "_")

        self.k_norm_weight = ttnn.as_tensor(
            k_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=k_cache_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, q: ttnn.Tensor, k: ttnn.Tensor) -> tuple:
        """
        Apply QK-norm to query and key tensors.

        Args:
            q: Query tensor of shape [batch, num_heads, seq_len, head_dim]
            k: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]

        Returns:
            Tuple of (normalized_q, normalized_k)
        """
        q_normed = ttnn.rms_norm(
            q,
            epsilon=self.eps,
            weight=self.q_norm_weight,
            compute_kernel_config=self.compute_kernel_config,
        )

        k_normed = ttnn.rms_norm(
            k,
            epsilon=self.eps,
            weight=self.k_norm_weight,
            compute_kernel_config=self.compute_kernel_config,
        )

        return q_normed, k_normed
