# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage 3 — Action expert (Gemma-300M, 18 layers) + Suffix MLP + denoise loop."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig

from .stages import StageSpec
from .kv_migration import KVMigration
from .expert_slice import Pi0_5SubmeshExpertSlice
from .suffix_slice import Pi0_5SubmeshSuffixSlice


class Stage3Expert:
    """Expert + suffix + Euler denoise on a 4x2 submesh.

    Construction is two-phase: `__init__` records args, `initialize()`
    uploads weights for both the expert backbone and the suffix MLP.

    `denoise()` runs the 10-step Euler integrator end-to-end:
      x_0 = noisy_actions
      for i in 0..num_steps-1:
          t = 1.0 - i / num_steps
          dt = -1.0 / num_steps              (timesteps go 1.0 → 0.0)
          suffix_h = embed_actions(x_t)
          adarms_cond = embed_adarms_cond(t)
          velocity_hidden = expert.forward(suffix_h, adarms_cond, prefix_kv)
          v_t = project_output(velocity_hidden)
          x_{i+1} = x_t + dt * v_t
    """

    def __init__(
        self,
        spec: StageSpec,
        submesh,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        denoise_steps: int = 10,
        tp_shard: bool = False,
        action_dim: int = 32,
        action_horizon: int = 50,
    ) -> None:
        assert spec.stage_idx == 3, "Stage3Expert must be stage 3"
        assert spec.runs_denoise_loop, "Stage 3 must run the denoise loop"
        assert spec.receives_kv_migration, "Stage 3 must receive KV migration"
        self.spec = spec
        self.submesh = submesh
        self.config = config
        self.weights = weights
        self.denoise_steps = denoise_steps
        self.tp_shard = tp_shard
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.slice: Optional[Pi0_5SubmeshExpertSlice] = None
        self.suffix: Optional[Pi0_5SubmeshSuffixSlice] = None
        self._kv_migrator: Optional[KVMigration] = None

    def initialize(self, kv_migrator: Optional[KVMigration] = None) -> None:
        """Upload expert backbone + suffix MLP weights onto the submesh."""
        if self.slice is not None:
            return
        self.slice = Pi0_5SubmeshExpertSlice(
            config=self.config,
            weights=self.weights,
            submesh=self.submesh,
            expert_layer_range=self.spec.expert_layer_range,
            tp_shard=self.tp_shard,
        )
        # Suffix MLP needs pi0_projections in weights; only init if present.
        if "pi0_projections" in self.weights and self.weights["pi0_projections"]:
            self.suffix = Pi0_5SubmeshSuffixSlice(
                config=self.config,
                weights=self.weights,
                submesh=self.submesh,
                action_dim=self.action_dim,
                action_horizon=self.action_horizon,
            )
        self._kv_migrator = kv_migrator

    def forward_expert_step(
        self,
        hidden_states: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        prefix_kv_cache: Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]] = None,
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        cos_override: Optional["ttnn.Tensor"] = None,
        sin_override: Optional["ttnn.Tensor"] = None,
    ) -> "ttnn.Tensor":
        if self.slice is None:
            raise RuntimeError("Stage3Expert.forward_expert_step called before initialize()")
        return self.slice.forward(
            hidden_states,
            adarms_cond,
            prefix_kv_cache=prefix_kv_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos_override=cos_override,
            sin_override=sin_override,
        )

    # ------------------------------------------------------------------ #
    # Denoise loop                                                        #
    # ------------------------------------------------------------------ #

    def denoise(
        self,
        noisy_actions: "ttnn.Tensor",
        prefix_kv_cache: List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]],
        attention_mask: "ttnn.Tensor",
    ) -> "ttnn.Tensor":
        """Run the full N-step Euler integrator.

        Args:
            noisy_actions: [B, action_horizon, action_dim] bf16 initial noise
                on this submesh (replicated). action_horizon should be
                tile-aligned (pad to multiple of 32 — pi0.5 horizon=50 →
                pad to 64).
            prefix_kv_cache: full list of (K, V) per VLM layer, on this
                submesh, from KV migration.
            attention_mask: joint mask [B, 1, S_suffix_padded,
                S_prefix + S_suffix_padded] (zero = unmasked).

        Returns: predicted clean actions [B, action_horizon_padded,
            action_dim] on this submesh.
        """
        if self.slice is None or self.suffix is None:
            raise RuntimeError(
                "Stage3Expert.denoise requires both expert slice and suffix slice; "
                "ensure pi0_projections weights are present and initialize() ran."
            )

        num_steps = self.denoise_steps
        # Euler timesteps: 1.0, 1-1/N, ..., 1/N. dt = -1/N (constant).
        dt = -1.0 / num_steps

        x_t = noisy_actions
        x_t_owned = False  # caller owns `noisy_actions`

        for i in range(num_steps):
            t = 1.0 - i / num_steps
            B = x_t.shape[0]
            adarms_cond = self.suffix.embed_adarms_cond(t, batch_size=B)
            suffix_h = self.suffix.embed_actions(x_t)

            velocity_hidden = self.slice.forward(
                suffix_h,
                adarms_cond,
                prefix_kv_cache=prefix_kv_cache,
                attention_mask=attention_mask,
            )
            ttnn.deallocate(suffix_h)
            ttnn.deallocate(adarms_cond)

            v_t = self.suffix.project_output(velocity_hidden)
            ttnn.deallocate(velocity_hidden)

            # x_{i+1} = x_t + dt * v_t. dt is constant -1/N.
            dx = ttnn.multiply(v_t, dt)
            ttnn.deallocate(v_t)
            x_t_new = ttnn.add(x_t, dx)
            ttnn.deallocate(dx)
            if x_t_owned:
                ttnn.deallocate(x_t)
            x_t = x_t_new
            x_t_owned = True

        return x_t  # caller owns
