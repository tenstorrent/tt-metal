from __future__ import annotations

import torch
import ttnn

from models.experimental.tt_symbiote.core.module import Module

from .context_parallel_conv3d import ContextParallelConv3d


class TTNNVisionPatchEmbed(Module):
    """
    TT implementation of Qwen3OmniMoeVisionPatchEmbed

    Converts input video/image into patch embeddings using Conv3D:
    - kernel_size = (T, P, P)
    - stride = (T, P, P)
    - no padding
    - non-causal
    """

    def __init__(
        self,
        config,
        *,
        mesh_device: ttnn.MeshDevice,
        parallel_config,
        ccl_manager,
    ):
        super().__init__()

        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )

        stride = kernel_size  # IMPORTANT: non-overlapping patches

        # 🔴 Use Conv3D but disable causal + padding behavior
        self.proj = ContextParallelConv3d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
            causal=False,  # 🔴 REQUIRED FIX
            context_parallel=False,  # usually not needed here
            groups=1,
            padding_mode="zeros",
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

        # 🔴 Override padding to ZERO (critical for patch extraction)
        self.proj.padding = (0, 0, 0)

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Input:
            hidden_states: (N, T, H, W, C)  [TT format]

        Output:
            (num_patches, embed_dim)
        """

        # Conv3D projection → patch embeddings
        x = self.proj(hidden_states)

        # Output shape: (N, T_out, H_out, W_out, embed_dim)

        # Flatten patches → (num_patches, embed_dim)
        x = ttnn.reshape(x, (-1, self.embed_dim))

        return x

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """
        Convert PyTorch Conv3D weights → TT format
        """

        weight = state.get("proj.weight")
        if weight is not None:
            # PyTorch: (out_channels, in_channels, kd, kh, kw)
            # TT expects: handled inside ContextParallelConv3d
            state["proj.weight"] = weight

        if "proj.bias" in state:
            state["proj.bias"] = state["proj.bias"]
