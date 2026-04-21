# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from models.demos.dots_ocr.tt._ttnn_import import get_ttnn

ttnn = get_ttnn()
_HAS_TTNN = ttnn is not None

from models.common.lightweightmodule import LightweightModule
from models.demos.dots_ocr.tt.vision_rmsnorm import RMSNorm


class PatchMerger(LightweightModule):
    """
    TTNN PatchMerger (ported from qwen25_vl) with DotsOCR-compatible dimensions.

    Input:  x [B, 1, S_patch, H_v]
    Output: y [B, 1, S_img, H_out], where S_img = S_patch / (spatial_merge_size^2)
    """

    def __init__(
        self,
        mesh_device,
        *,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int,
        state_dict,
        state_dict_prefix: str,
        weight_cache_path,
        dtype,
    ):
        super().__init__()
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.hidden_size = hidden_size
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size

        self.mlp_size = hidden_size * (spatial_merge_size**2)

        # HF `PatchMerger` defaults to `nn.LayerNorm` on `context_dim` (`ln_q`), not RMSNorm.
        # Using RMSNorm here tanked PCC vs HF. Prefer LayerNorm when checkpoint has bias.
        weight_dtype = ttnn.bfloat16 if _HAS_TTNN and hasattr(ttnn, "bfloat16") else torch.bfloat16
        ln_w = f"{state_dict_prefix}.ln_q.weight"
        ln_b = f"{state_dict_prefix}.ln_q.bias"
        if ln_w in state_dict and ln_b in state_dict:
            self._merge_pre_norm = "layernorm"
            # LightweightModule is not nn.Module; keep norms as plain tensors.
            self._ln_q_weight = state_dict[ln_w].clone().float()
            self._ln_q_bias = state_dict[ln_b].clone().float()
            self.norm = None
        else:
            self._merge_pre_norm = "rmsnorm"
            self.norm = RMSNorm(
                device=mesh_device,
                dim=hidden_size,
                state_dict=state_dict,
                state_dict_prefix=state_dict_prefix + ".",
                weight_key="ln_q",
                weight_dtype=weight_dtype,
                eps=1e-6,
            )

        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)

        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        def as_weight_tensor(name: str, weight_dtype):
            # Use DRAM_MEMORY_CONFIG if available, otherwise fallback for test environments
            memory_config = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
            mesh_mapper = (
                ttnn.ReplicateTensorToMesh(self.mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
            )

            return ttnn.as_tensor(
                torch_weight(name),
                dtype=weight_dtype,
                device=self.mesh_device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                cache_file_name=cache_name(name),
            )

        # feed_forward.0: mlp_size -> mlp_size
        # feed_forward.2: mlp_size -> out_hidden_size
        self.w1 = as_weight_tensor("feed_forward.0", dtype)
        self.w2 = as_weight_tensor("feed_forward.2", dtype)

        # Torch path uses HF-shaped ``nn.Linear`` weights [out, in] + bias (ttnn path transposes for tilize).
        p0w = f"{state_dict_prefix}.feed_forward.0.weight"
        p0b = f"{state_dict_prefix}.feed_forward.0.bias"
        p2w = f"{state_dict_prefix}.feed_forward.2.weight"
        p2b = f"{state_dict_prefix}.feed_forward.2.bias"
        self._ff0_w_torch = state_dict[p0w].clone().float() if p0w in state_dict else None
        self._ff0_b_torch = state_dict[p0b].clone().float() if p0b in state_dict else None
        self._ff2_w_torch = state_dict[p2w].clone().float() if p2w in state_dict else None
        self._ff2_b_torch = state_dict[p2b].clone().float() if p2b in state_dict else None

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Support both TT tensors and torch tensors (hybrid vision path).
        if isinstance(x, torch.Tensor) or not _HAS_TTNN:
            if getattr(self, "_merge_pre_norm", "rmsnorm") == "layernorm":
                w = self._ln_q_weight.to(dtype=x.dtype, device=x.device)
                b = self._ln_q_bias.to(dtype=x.dtype, device=x.device)
                x = torch.nn.functional.layer_norm(x, (self.hidden_size,), w, b, eps=1e-6)
            else:
                x = self.norm(x)
            if x.dim() == 2:
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 3:
                x = x.unsqueeze(1)

            # [B, 1, S_patch, H] -> [B, 1, S_img, H*m^2]
            x = x.reshape(x.shape[0], x.shape[1], -1, self.mlp_size)

            # Prefer checkpoint-faithful ``nn.Linear`` weights/biases (same as HF ``PatchMerger.mlp``).
            if self._ff0_w_torch is not None and self._ff2_w_torch is not None:
                w1_t = self._ff0_w_torch.to(device=x.device)
                w2_t = self._ff2_w_torch.to(device=x.device)
                b1 = self._ff0_b_torch.to(device=x.device) if self._ff0_b_torch is not None else None
                b2 = self._ff2_b_torch.to(device=x.device) if self._ff2_b_torch is not None else None
            else:
                w1 = self.w1
                w2 = self.w2
                if _HAS_TTNN and not isinstance(w1, torch.Tensor):
                    w1 = ttnn.to_torch(w1)
                if _HAS_TTNN and not isinstance(w2, torch.Tensor):
                    w2 = ttnn.to_torch(w2)
                w1_t = w1
                w2_t = w2
                if w2_t.dim() == 2 and w2_t.shape[0] == self.mlp_size and w2_t.shape[1] == self.out_hidden_size:
                    w2_t = w2_t.t()
                b1 = b2 = None

            x = torch.nn.functional.linear(x.float(), w1_t.float(), b1).to(torch.bfloat16)
            x = torch.nn.functional.gelu(x, approximate="none")
            x = torch.nn.functional.linear(x.float(), w2_t.float(), b2).to(torch.bfloat16)
            return x

        if getattr(self, "_merge_pre_norm", "rmsnorm") == "layernorm":
            x_torch = ttnn.to_torch(x).float()
            w = self._ln_q_weight.to(device=x_torch.device)
            b = self._ln_q_bias.to(device=x_torch.device)
            x_torch = torch.nn.functional.layer_norm(x_torch, (self.hidden_size,), w, b, eps=1e-6).to(torch.bfloat16)
            x = ttnn.from_torch(
                x_torch,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
                if hasattr(ttnn, "ReplicateTensorToMesh")
                else None,
            )
        else:
            x = self.norm(x)
        # Merge spatial dimensions into channel dim: [B, 1, S_patch, H] -> [B, 1, S_img, H*m^2]
        # Workaround for reshape tilized hangs: convert to row-major first.
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (x.shape[0], x.shape[1], -1, self.mlp_size))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Use DRAM_MEMORY_CONFIG if available, otherwise fallback for test environments
        memory_config = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)

        x = ttnn.linear(x, self.w1, compute_kernel_config=None, memory_config=memory_config)
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.w2, compute_kernel_config=None, memory_config=memory_config)
        return x
