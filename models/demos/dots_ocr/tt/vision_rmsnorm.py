# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Simple RMSNorm for Dots OCR vision stack.

This is a lightweight version that works in both CPU test environments
and full TTNN device environments (no distributed norm sharding).
"""

from __future__ import annotations

import torch

from models.common.lightweightmodule import LightweightModule
from models.demos.dots_ocr.tt._ttnn_import import get_ttnn


class RMSNorm(LightweightModule):
    """
    Simple RMSNorm for Dots OCR PatchMerger.

    Uses ttnn.rms_norm when available, falls back gracefully for testing.
    """

    def __init__(
        self,
        device,
        dim: int,
        state_dict,
        state_dict_prefix: str = "",
        weight_key: str = "ln_q",
        weight_dtype=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.device = device
        self.dim = dim

        ttnn = get_ttnn()
        _HAS_TTNN = ttnn is not None
        # Default dtype when ttnn is not available
        if weight_dtype is None:
            weight_dtype = ttnn.bfloat16 if _HAS_TTNN and hasattr(ttnn, "bfloat16") else torch.bfloat16

        # Get weight/bias from state_dict. Some HF vision towers use LayerNorm (weight+bias)
        # even when configs call it "norm".
        if state_dict_prefix and not state_dict_prefix.endswith("."):
            state_dict_prefix = state_dict_prefix + "."

        def _candidates(k: str) -> list[str]:
            # HF checkpoints sometimes name norms as `ln1/ln2` instead of `norm1/norm2`.
            c = [k]
            if k == "norm1":
                c.append("ln1")
            elif k == "norm2":
                c.append("ln2")
            elif k == "norm":
                c.append("ln")
            return c

        weight_names = [f"{state_dict_prefix}{k}.weight" for k in _candidates(weight_key)]
        bias_names = [f"{state_dict_prefix}{k}.bias" for k in _candidates(weight_key)]
        weight_name = next((n for n in weight_names if n in state_dict), weight_names[0])
        bias_name = next((n for n in bias_names if n in state_dict), bias_names[0])

        if weight_name in state_dict:
            self.weight = state_dict[weight_name].clone()
        else:
            # Fallback for tests
            self.weight = torch.ones(dim, dtype=torch.bfloat16)

        self.bias = state_dict[bias_name].clone() if bias_name in state_dict else None
        self._use_layer_norm = self.bias is not None

        ttnn = get_ttnn()
        self._weight_ttnn = None
        self._bias_ttnn = None
        if ttnn is not None and self.device is not None and weight_name in state_dict:
            w = self.weight.to(torch.bfloat16)
            if hasattr(ttnn, "as_tensor") and hasattr(ttnn, "ReplicateTensorToMesh"):
                mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
                # Match tt_transformers' RMSNorm weight layout: [1,1,dim//32,32] in ROW_MAJOR.
                tile = 32
                assert w.numel() == dim
                w = w.view(1, 1, dim).reshape(1, 1, dim // tile, tile)
                self._weight_ttnn = ttnn.as_tensor(
                    w,
                    dtype=ttnn.bfloat16,
                    device=self.device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=mem,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                )
                if self.bias is not None:
                    b = self.bias.to(torch.bfloat16)
                    assert b.numel() == dim
                    b = b.view(1, 1, dim).reshape(1, 1, dim // tile, tile)
                    self._bias_ttnn = ttnn.as_tensor(
                        b,
                        dtype=ttnn.bfloat16,
                        device=self.device,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=mem,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                    )

    def forward(self, x: torch.Tensor | "ttnn.Tensor") -> torch.Tensor | "ttnn.Tensor":
        """Norm: LayerNorm when bias exists, else RMSNorm."""
        ttnn = get_ttnn()
        if ttnn is not None and isinstance(x, ttnn.Tensor) and self._weight_ttnn is not None:
            if self._use_layer_norm and self._bias_ttnn is not None and hasattr(ttnn, "layer_norm"):
                return ttnn.layer_norm(x, weight=self._weight_ttnn, bias=self._bias_ttnn, epsilon=self.eps)
            if hasattr(ttnn, "rms_norm"):
                return ttnn.rms_norm(x, weight=self._weight_ttnn, epsilon=self.eps)
        if ttnn is not None and isinstance(x, ttnn.Tensor) and self._weight_ttnn is None:
            return self._rms_norm_ttnn_via_host_layout(x, ttnn)
        return self._rms_norm_torch(x)

    def _rms_norm_ttnn_via_host_layout(self, x, ttnn) -> ttnn.Tensor:
        """Fallback when weight is not on device: round-trip through torch (dev smoke only)."""
        # Mesh tensors require a mesh composer for to_torch; slice off one replica if weights are replicated.
        composer = None
        dev = x.device() if hasattr(x, "device") and callable(x.device) else getattr(x, "device", None)
        if dev is not None and hasattr(ttnn, "ConcatMeshToTensor"):
            try:
                composer = ttnn.ConcatMeshToTensor(dev, dim=0)
            except Exception:
                composer = None
        x_t = ttnn.to_torch(x, mesh_composer=composer) if composer is not None else ttnn.to_torch(x)
        try:
            num_devices = dev.get_num_devices() if dev is not None and hasattr(dev, "get_num_devices") else 1
            if num_devices > 1 and x_t.shape[0] % num_devices == 0:
                per = x_t.shape[0] // num_devices
                x_t = x_t[:per]
        except Exception:
            pass
        h = self._rms_norm_torch(x_t)
        layout = getattr(x, "layout", None) or ttnn.TILE_LAYOUT
        return ttnn.from_torch(h, device=dev, layout=layout, dtype=ttnn.bfloat16)

    def _rms_norm_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch norm implementation (LayerNorm when bias exists, else RMSNorm)."""
        if self._use_layer_norm and self.bias is not None:
            # Standard LayerNorm over last dim.
            w = self.weight.to(dtype=x.dtype, device=x.device)
            b = self.bias.to(dtype=x.dtype, device=x.device)
            return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=w, bias=b, eps=self.eps)
        # x: [..., dim]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        # Apply weight (broadcast)
        if self.weight.dim() == 1:
            weight = self.weight.view(1, 1, 1, -1) if x.dim() == 4 else self.weight
            return x * weight
        return x * self.weight


# Export for backward compatibility
__all__ = ["RMSNorm"]
