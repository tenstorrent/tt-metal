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

        # Dots patch merger uses RMSNorm with eps ~1e-6 in similar repos; keep configurable if needed.
        # Use proper dtype or fallback for environments without full TTNN
        weight_dtype = ttnn.bfloat16 if _HAS_TTNN and hasattr(ttnn, "bfloat16") else torch.bfloat16
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

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Support both TT tensors and torch tensors (hybrid vision path).
        if isinstance(x, torch.Tensor) or not _HAS_TTNN:
            x = self.norm(x)
            if x.dim() == 2:
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 3:
                x = x.unsqueeze(1)

            # [B, 1, S_patch, H] -> [B, 1, S_img, H*m^2]
            x = x.reshape(x.shape[0], x.shape[1], -1, self.mlp_size)

            # Bring weights back to torch if they are TT tensors
            w1 = self.w1
            w2 = self.w2
            if _HAS_TTNN and not isinstance(w1, torch.Tensor):
                w1 = ttnn.to_torch(w1)
            if _HAS_TTNN and not isinstance(w2, torch.Tensor):
                w2 = ttnn.to_torch(w2)

            # `self.w1`/`self.w2` were built from `torch_weight(...)` which transposes the checkpoint weights.
            # `torch.nn.functional.linear` expects weights shaped [out_features, in_features], so transpose back
            # when needed.
            w1_t = w1
            w2_t = w2
            if w1_t.dim() == 2 and w1_t.shape[0] == self.mlp_size and w1_t.shape[1] == self.mlp_size:
                # square: either orientation works
                pass
            if w2_t.dim() == 2 and w2_t.shape[0] == self.mlp_size and w2_t.shape[1] == self.out_hidden_size:
                # stored as [in, out] due to transpose during load → convert to [out, in]
                w2_t = w2_t.t()

            x = torch.nn.functional.linear(x.float(), w1_t.float()).to(torch.bfloat16)
            x = torch.nn.functional.gelu(x)
            x = torch.nn.functional.linear(x.float(), w2_t.float()).to(torch.bfloat16)
            return x

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
