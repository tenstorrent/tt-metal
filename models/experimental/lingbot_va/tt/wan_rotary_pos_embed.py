# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.tt_dit.layers.module import Module
from models.tt_transformers.tt.common import get_rot_transformation_mat


class WanRotaryPosEmbed(Module):
    """RoPE for Wan models.

    Frequency computation is done on CPU in float64 (matching the reference
    ``WanRotaryPosEmbed``) and the resulting cos/sin tables are uploaded to
    the device as float32.  Previous versions computed frequencies on-device
    in bfloat16 which introduced significant RoPE drift that accumulated
    across 30 transformer blocks and degraded PCC.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        self.f_freqs_base_cpu, self.h_freqs_base_cpu, self.w_freqs_base_cpu = self._precompute_freqs_base_cpu()

        # Precompute transformation matrix (BFLOAT16 required by wan_fused_rmsnorm_post_allgather)
        self.transformation_mat = ttnn.from_torch(
            get_rot_transformation_mat(), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

    def _precompute_freqs_base_cpu(self):
        """Precompute frequency bases on CPU in float64 (matching reference)."""
        f_freqs = 1.0 / (self.theta ** (torch.arange(0, self.f_dim, 2)[: (self.f_dim // 2)].double() / self.f_dim))
        h_freqs = 1.0 / (self.theta ** (torch.arange(0, self.h_dim, 2)[: (self.h_dim // 2)].double() / self.h_dim))
        w_freqs = 1.0 / (self.theta ** (torch.arange(0, self.w_dim, 2)[: (self.w_dim // 2)].double() / self.w_dim))
        return f_freqs, h_freqs, w_freqs

    def forward(self, grid_ids: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Compute RoPE cos/sin on CPU in float64, upload as float32.

        This mirrors the reference which uses ``torch.polar`` on float64
        frequencies then converts to the model dtype.  Computing on-device
        in bfloat16 loses too much precision.
        """
        grid_ids_cpu = ttnn.to_torch(ttnn.get_device_tensors(grid_ids)[0]).detach().cpu()

        f_ids = grid_ids_cpu[:, 0, :].unsqueeze(-1).double()
        h_ids = grid_ids_cpu[:, 1, :].unsqueeze(-1).double()
        w_ids = grid_ids_cpu[:, 2, :].unsqueeze(-1).double()

        f_freqs = f_ids * self.f_freqs_base_cpu.to(f_ids.device)
        h_freqs = h_ids * self.h_freqs_base_cpu.to(h_ids.device)
        w_freqs = w_ids * self.w_freqs_base_cpu.to(w_ids.device)

        freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()

        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)

        rope_cos = torch.cat([cos_freqs, cos_freqs], dim=-1)
        rope_sin = torch.cat([sin_freqs, sin_freqs], dim=-1)

        rope_cos_tt = ttnn.from_torch(rope_cos, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        rope_sin_tt = ttnn.from_torch(rope_sin, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        return rope_cos_tt, rope_sin_tt
