# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 (denoise) suffix MLP slice for Option C.

Identical shape contract to `option_b.suffix_slice.Pi0_5SubmeshSuffixSlice`,
but every upload goes through `option_c.vlm_slice._upload_l1_replicated` so
the weights live in L1 (Option C never runs all_reduce, so the L1/CB clash
that drove Option B to DRAM doesn't apply).

Suffix weights are tiny (~2 MB total): action_in_proj, time_mlp_in/out,
action_out_proj. We replicate them on every denoise chip so each Euler step
runs locally — no transport between chips during the 10-step loop.
"""

from __future__ import annotations

from typing import Dict

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.reference.torch_suffix import (
    create_sinusoidal_pos_embedding as torch_sincos,
)
from models.experimental.pi0_5.tt.ttnn_common import tensor_1d_to_2d_ttnn

from .vlm_slice import _upload_l1_replicated


class Pi0_5OptionCSuffixSlice:
    """Replicated-weight suffix MLP slice on the denoise submesh (L1-resident).

    Args:
        config:    full PaliGemma config (uses .expert_config.width).
        weights:   the full categorized weights dict. Suffix lives under
                   weights["pi0_projections"] with keys
                     action_in_proj.{weight,bias},
                     action_out_proj.{weight,bias},
                     time_mlp_in.{weight,bias},
                     time_mlp_out.{weight,bias}.
        submesh:   the 6-chip denoise MeshDevice.
        action_dim:      defaults to 32 (pi0.5 LIBERO).
        action_horizon:  defaults to 50; LIBERO upstream uses 10 (padded 32).
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        submesh,
        action_dim: int = 32,
        action_horizon: int = 50,
    ) -> None:
        self.config = config
        self.submesh = submesh
        self.W = config.expert_config.width
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        suffix = weights["pi0_projections"]

        def upload_w(name: str, dtype=ttnn.bfloat8_b):
            return _upload_l1_replicated(
                suffix[f"{name}.weight"].T.contiguous(),
                submesh,
                dtype=dtype,
            )

        def upload_b(name: str):
            key = f"{name}.bias"
            if key not in suffix:
                return None
            # Keep biases at bf16. bf8_b on the suffix MLP biases compounds
            # PCC drift across 10 denoise steps × 18 expert-layer feedback —
            # cost is ~8 KB total so the L1 saving is negligible.
            return tensor_1d_to_2d_ttnn(suffix[key], submesh, dtype=ttnn.bfloat16)

        self.action_in_w = upload_w("action_in_proj")
        self.action_in_b = upload_b("action_in_proj")
        self.action_out_w = upload_w("action_out_proj")
        self.action_out_b = upload_b("action_out_proj")
        self.time_mlp_in_w = upload_w("time_mlp_in")
        self.time_mlp_in_b = upload_b("time_mlp_in")
        self.time_mlp_out_w = upload_w("time_mlp_out")
        self.time_mlp_out_b = upload_b("time_mlp_out")

        # Force a 1-row grid for the small time-MLP matmuls (m_tiles=1 path).
        # Without this the default kernel selection picks per_core_M=0 and
        # rejects the call — same issue as option_b.suffix_slice.
        gsz = submesh.compute_with_storage_grid_size()
        self._small_core_grid = ttnn.CoreGrid(y=1, x=min(gsz.x, 8))

    def embed_adarms_cond(self, timestep_value: float, batch_size: int = 1) -> "ttnn.Tensor":
        """Build adarms_cond [B, W] for a scalar timestep value.

        Sincos computed on HOST (torch) to sidestep the on-device sincos
        tiny-matmul program-config issue on MeshDevice. Two MLP+silu stages
        run on device with the small_core_grid.
        """
        t_host = torch.tensor([timestep_value] * batch_size, dtype=torch.float32)
        sincos_host = torch_sincos(t_host, self.W, min_period=4e-3, max_period=4.0)
        sincos = _upload_l1_replicated(
            sincos_host.contiguous(),
            self.submesh,
            dtype=ttnn.bfloat16,
        )
        x = ttnn.linear(
            sincos,
            self.time_mlp_in_w,
            bias=self.time_mlp_in_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self._small_core_grid,
        )
        ttnn.deallocate(sincos)
        x = ttnn.silu(x)
        x2 = ttnn.linear(
            x,
            self.time_mlp_out_w,
            bias=self.time_mlp_out_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self._small_core_grid,
        )
        ttnn.deallocate(x)
        out = ttnn.silu(x2)
        ttnn.deallocate(x2)
        return out

    def embed_actions(self, noisy_actions: "ttnn.Tensor") -> "ttnn.Tensor":
        """[B, S, action_dim] → [B, S, W]."""
        return ttnn.linear(
            noisy_actions,
            self.action_in_w,
            bias=self.action_in_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self._small_core_grid,
        )

    def project_output(self, expert_hidden: "ttnn.Tensor") -> "ttnn.Tensor":
        """[B, S, W] → [B, S, action_dim] — velocity prediction."""
        return ttnn.linear(
            expert_hidden,
            self.action_out_w,
            bias=self.action_out_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self._small_core_grid,
        )
