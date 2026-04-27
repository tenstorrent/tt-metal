# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
RMSNorm for Molmo2 Text Model.

Implements Root Mean Square Layer Normalization (HuggingFace ``Molmo2RMSNorm`` numerics):
    output = weight * (x * rsqrt(mean(x^2) + eps))

The forward pass uses **PyTorch** on host (float32 reduction) and uploads the result with
``ttnn.from_torch`` to match reference behavior; ``self.weight`` remains on device for
compatibility with weight-loading and any code that inspects parameters.
"""


import torch

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
        eps: float = 1e-6,
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
        w_view = weight.reshape(1, 1, 1, -1).contiguous()
        # CPU / torch copy for forward() HF-aligned RMS (same math as ``Molmo2RMSNorm`` in HF).
        self._weight_torch = w_view.clone()

        self.weight = ttnn.as_tensor(
            w_view,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name,
        )

    @staticmethod
    def _molmo2_rms_norm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Port of HuggingFace ``Molmo2RMSNorm.forward``: variance in float32, then back to
        activations' dtype, then ``weight * x``.
        """
        og_dtype = x.dtype
        x_f = x.to(torch.float32)
        variance = x_f.pow(2).mean(-1, keepdim=True)
        x_f = x_f * torch.rsqrt(variance + eps)
        x_f = x_f.to(og_dtype)
        w = weight.to(device=x.device, dtype=og_dtype)
        if w.dim() == 1:
            w = w.view(1, 1, 1, -1)
        return w * x_f

    @staticmethod
    def _ttnn_to_torch_replicated(x: ttnn.Tensor, mesh_device) -> torch.Tensor:
        is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
        if is_mesh:
            mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
            t = ttnn.to_torch(x, mesh_composer=mesh_composer)
            n = mesh_device.get_num_devices()
            if t.shape[0] == n:
                t = t[0]
        else:
            t = ttnn.to_torch(x)
        if t.dim() == 4 and t.shape[1] == 1:
            return t
        if t.dim() == 3:
            return t.unsqueeze(1)
        return t

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through RMSNorm.

        Args:
            x: Input tensor of shape [1, 1, seq_len, hidden_dim]

        Returns:
            Normalized tensor of shape [1, 1, seq_len, hidden_dim]
        """
        is_mesh = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None
        x_torch = self._ttnn_to_torch_replicated(x, self.mesh_device)
        out = self._molmo2_rms_norm_torch(x_torch, self._weight_torch, self.eps)
        return ttnn.from_torch(
            out,
            device=self.mesh_device,
            dtype=x.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
