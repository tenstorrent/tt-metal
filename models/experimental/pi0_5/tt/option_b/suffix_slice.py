# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-3 suffix MLP slice for Option B (pi0.5 action expert).

Wraps the 4 small linear layers used to embed/de-embed actions and to build
the adaRMS time conditioning vector:

  - action_in_proj   [action_dim → expert_width]  — embed noisy actions
  - time_mlp_in      [expert_width → expert_width] — adaRMS cond stage 1
  - time_mlp_out     [expert_width → expert_width] — adaRMS cond stage 2
  - action_out_proj  [expert_width → action_dim]  — project velocity back

All four weights are tiny (≤1 MB at bf8 each) so we replicate them across
the 4×2 submesh. The expert backbone (separate slice) uses these on every
denoise step.

Reuses `ttnn_common.create_sinusoidal_pos_embedding_ttnn` for the time
sincos embedding.
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

from .vlm_slice import _upload_replicated


class Pi0_5SubmeshSuffixSlice:
    """Replicated-weight suffix slice on stage 3's submesh.

    Args:
        config:    full PaliGemma config (uses .expert_config.width).
        weights:   the full categorized weights dict. Suffix lives under
                   weights["pi0_projections"] with keys
                   action_in_proj.{weight,bias},
                   action_out_proj.{weight,bias},
                   time_mlp_in.{weight,bias},
                   time_mlp_out.{weight,bias}.
        submesh:   stage 3's 4×2 MeshDevice.
        action_dim:      defaults to 32 (pi0.5 LIBERO).
        action_horizon:  defaults to 50 (pi0.5).
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
            # Suffix weights are tiny (~2 MB total). DRAM for now to match
            # the rest of the option_b path; revisit when L1 placement
            # plan lands.
            return _upload_replicated(
                suffix[f"{name}.weight"].T.contiguous(),
                submesh,
                dtype=dtype,
            )

        def upload_b(name: str):
            key = f"{name}.bias"
            if key not in suffix:
                return None
            return tensor_1d_to_2d_ttnn(suffix[key], submesh, dtype=ttnn.bfloat16)

        self.action_in_w = upload_w("action_in_proj")
        self.action_in_b = upload_b("action_in_proj")
        self.action_out_w = upload_w("action_out_proj")
        self.action_out_b = upload_b("action_out_proj")
        self.time_mlp_in_w = upload_w("time_mlp_in")
        self.time_mlp_in_b = upload_b("time_mlp_in")
        self.time_mlp_out_w = upload_w("time_mlp_out")
        self.time_mlp_out_b = upload_b("time_mlp_out")

        # Core grid for the small suffix matmuls. ttnn.linear without an
        # explicit core_grid can pick a width-sharded path with
        # per_core_M = m_tiles // grid_y; for time-MLP (m_tiles=1) on a
        # 12×10 grid that yields per_core_M=0 and the kernel rejects it.
        # Force a 1-row grid so per_core_M = m_tiles = 1 always.
        gsz = submesh.compute_with_storage_grid_size()
        self._small_core_grid = ttnn.CoreGrid(y=1, x=min(gsz.x, 8))

    # ------------------------------------------------------------------ #
    # Time / adaRMS conditioning                                          #
    # ------------------------------------------------------------------ #

    def embed_adarms_cond(self, timestep_value: float, batch_size: int = 1) -> "ttnn.Tensor":
        """Build adarms_cond [B, W] for a scalar timestep value.

        sincos is computed on HOST (torch) — the on-device sincos involves a
        tiny matmul whose default kernel selection picks an invalid
        program-config on MeshDevice (per_core_M=0). Uploading a precomputed
        sincos sidesteps that. The two MLP linears + silus then run on
        device with the small_core_grid (1-row grid) to avoid the same
        per_core_M=0 trap.
        """
        # Host sincos.
        t_host = torch.tensor([timestep_value] * batch_size, dtype=torch.float32)
        sincos_host = torch_sincos(t_host, self.W, min_period=4e-3, max_period=4.0)
        sincos = _upload_replicated(
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

    # ------------------------------------------------------------------ #
    # Action embed / unembed                                              #
    # ------------------------------------------------------------------ #

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
