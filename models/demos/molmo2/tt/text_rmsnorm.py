# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
RMSNorm for Molmo2 Text Model.

Implements Root Mean Square Layer Normalization:
    output = x * rsqrt(mean(x^2) + eps) * weight

No bias term, following Llama-style RMSNorm.
"""


import ttnn
from models.common.lightweightmodule import LightweightModule


class TextRMSNorm(LightweightModule):
    """
    RMSNorm layer for Molmo2 text model.

    Unlike LayerNorm, RMSNorm:
    - Does not center (no mean subtraction)
    - Uses root mean square for scaling
    - Has no bias term
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        hidden_dim: int = 4096,
        eps: float = 1e-5,
        weight_cache_path=None,
        state_dict_prefix: str = "",
        weight_key: str = "weight",
        dtype=ttnn.bfloat16,
    ):
        """
        Initialize TextRMSNorm.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing weights
            hidden_dim: Hidden dimension (4096 for Molmo2)
            eps: Epsilon for numerical stability
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            weight_key: Key suffix for weight in state dict
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.hidden_dim = hidden_dim
        self.eps = eps

        # Cache file naming
        if weight_cache_path is None:
            cache_name = None
        else:
            cache_name = weight_cache_path / f"{state_dict_prefix}.{weight_key}"

        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

        # Load weight
        weight_key_full = f"{state_dict_prefix}.{weight_key}" if state_dict_prefix else weight_key
        weight = state_dict[weight_key_full]

        self.weight = ttnn.as_tensor(
            weight.reshape(1, 1, 1, -1),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through RMSNorm.

        Args:
            x: Input tensor of shape [1, 1, seq_len, hidden_dim]

        Returns:
            Normalized tensor of shape [1, 1, seq_len, hidden_dim]
        """
        # Use TTNN's rms_norm operation
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)
