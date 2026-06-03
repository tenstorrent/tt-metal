# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage 2 — denoise orchestrator for Option C.

Owns the 6-chip denoise submesh. Holds:
  - the action expert (Gemma-300M, 18 layers split 3-per-chip in the
    target placement; replicated in the current scaffolding pass)
  - the suffix MLP (replicated on every denoise chip so each Euler step
    runs locally)
  - the layer-paired migrated VLM KV (consumed via the existing
    `past_key_value` kwarg in AdaRMSGemmaBlockTTNN)

`denoise()` runs the full N-step Euler integrator on the denoise submesh
without leaving the submesh — no per-step transport. Stages 0/1 sit idle
for this window (see deployment plan §3.4); the 4 spare chips can host a
second denoise pipeline if throughput becomes the bottleneck.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig

from .expert_slice import Pi0_5OptionCExpertSlice, Pi0_5OptionCExpertSlicePaired
from .kv_migration import KVMigration
from .mesh_setup import create_per_chip_submeshes
from .stages import EXPERT_LAYERS_PER_DENOISE_CHIP, NUM_DENOISE_CHIPS, StageSpec
from .suffix_slice import Pi0_5OptionCSuffixSlice
from .transport import send_activation_via_host


class StageDenoise:
    """Stage 2: expert backbone + suffix MLP + N-step Euler denoise.

    Two-phase construction. `__init__` records args; `initialize(kv_migrator)`
    uploads expert + suffix weights and attaches the migrator (whose
    `migrate_layer_paired` is called by the pipeline driver at end of
    prefill).

    The Euler loop matches `option_b/stage_3_expert.py`:

        x_0 = noisy_actions
        for i in 0..N-1:
            t = 1.0 - i / N
            dt = -1.0 / N
            adarms_cond = suffix.embed_adarms_cond(t)
            suffix_h    = suffix.embed_actions(x_t)
            v_hidden    = expert.forward(suffix_h, adarms_cond, prefix_kv)
            v_t         = suffix.project_output(v_hidden)
            x_{i+1}     = x_t + dt * v_t
    """

    def __init__(
        self,
        spec: StageSpec,
        submesh,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        denoise_steps: int = 10,
        action_dim: int = 32,
        action_horizon: int = 50,
        layer_paired_l1: bool = False,
        layers_per_chip: int = EXPERT_LAYERS_PER_DENOISE_CHIP,
    ) -> None:
        if spec.stage_idx != 2:
            raise AssertionError(f"StageDenoise must be stage 2, got {spec.stage_idx}")
        if not spec.runs_denoise_loop:
            raise AssertionError("Stage 2 must run the denoise loop")
        if not spec.receives_kv_migration:
            raise AssertionError("Stage 2 must receive KV migration")

        self.spec = spec
        self.submesh = submesh
        self.config = config
        self.weights = weights
        self.denoise_steps = denoise_steps
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.layer_paired_l1 = layer_paired_l1
        self.layers_per_chip = layers_per_chip
        # Populated in initialize() when layer_paired_l1=True.
        self.micro_submeshes: Optional[List] = None

        self.slice = None  # Pi0_5OptionCExpertSlice or Pi0_5OptionCExpertSlicePaired
        self.suffix: Optional[Pi0_5OptionCSuffixSlice] = None
        self._kv_migrator: Optional[KVMigration] = None

    # ------------------------------------------------------------------ #
    # Build                                                              #
    # ------------------------------------------------------------------ #

    def initialize(self, kv_migrator: Optional[KVMigration] = None) -> None:
        """Upload expert backbone + suffix MLP weights onto the denoise submesh.

        When `layer_paired_l1=True`, the denoise submesh is carved into N
        single-chip micro-submeshes (one per `layers_per_chip`-sized group)
        and each layer's weights live in L1 on exactly its owning chip — the
        target Option C placement (deployment plan §3.1).

        Args:
            kv_migrator: a `KVMigration` instance bound to this submesh. The
                pipeline driver calls `kv_migrator.migrate_layer_paired(...)`
                at end of prefill to populate the per-layer migrated KV.
                Pass None for stage smoke tests that don't need migration.
        """
        if self.slice is not None:
            return

        if self.layer_paired_l1:
            lo, hi = self.spec.expert_layer_range
            num_layers = hi - lo
            num_chips = (num_layers + self.layers_per_chip - 1) // self.layers_per_chip
            if num_chips > NUM_DENOISE_CHIPS:
                raise ValueError(
                    f"layer-paired denoise needs {num_chips} chips for "
                    f"expert_layer_range {self.spec.expert_layer_range} with "
                    f"layers_per_chip={self.layers_per_chip}; submesh only has "
                    f"{NUM_DENOISE_CHIPS} chips"
                )
            self.micro_submeshes = create_per_chip_submeshes(self.submesh, num_chips)
            self.slice = Pi0_5OptionCExpertSlicePaired(
                config=self.config,
                weights=self.weights,
                micro_submeshes=self.micro_submeshes,
                expert_layer_range=self.spec.expert_layer_range,
                layers_per_chip=self.layers_per_chip,
            )
            if "pi0_projections" in self.weights and self.weights["pi0_projections"]:
                # Suffix MLP is small (~2 MB); upload one replicated copy on
                # the FIRST denoise micro-submesh (where each Euler step
                # starts). The driver moves x_t back to that chip between
                # steps so embed/project can run locally.
                self.suffix = Pi0_5OptionCSuffixSlice(
                    config=self.config,
                    weights=self.weights,
                    submesh=self.micro_submeshes[0],
                    action_dim=self.action_dim,
                    action_horizon=self.action_horizon,
                )
        else:
            self.slice = Pi0_5OptionCExpertSlice(
                config=self.config,
                weights=self.weights,
                submesh=self.submesh,
                expert_layer_range=self.spec.expert_layer_range,
            )
            if "pi0_projections" in self.weights and self.weights["pi0_projections"]:
                self.suffix = Pi0_5OptionCSuffixSlice(
                    config=self.config,
                    weights=self.weights,
                    submesh=self.submesh,
                    action_dim=self.action_dim,
                    action_horizon=self.action_horizon,
                )
        self._kv_migrator = kv_migrator

    @property
    def first_chip_submesh(self):
        """Where the suffix MLP / x_t lives between Euler steps."""
        if self.layer_paired_l1:
            assert self.micro_submeshes is not None
            return self.micro_submeshes[0]
        return self.submesh

    @property
    def last_chip_submesh(self):
        """Where the post-final-norm hidden lives after one expert step."""
        if self.layer_paired_l1:
            assert self.micro_submeshes is not None
            return self.micro_submeshes[-1]
        return self.submesh

    # ------------------------------------------------------------------ #
    # Single expert step (used by smoke tests + the denoise loop)        #
    # ------------------------------------------------------------------ #

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
            raise RuntimeError("StageDenoise.forward_expert_step called before initialize()")
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
    # Full N-step Euler denoise loop                                     #
    # ------------------------------------------------------------------ #

    def denoise(
        self,
        noisy_actions: "ttnn.Tensor",
        prefix_kv_cache: List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]],
        attention_mask: "ttnn.Tensor",
    ) -> "ttnn.Tensor":
        """Run the full `self.denoise_steps`-step Euler integrator.

        Args:
            noisy_actions: [B, action_horizon_padded, action_dim] bf16
                initial noise on the denoise submesh (replicated).
                action_horizon_padded must be tile-aligned (pad 50 → 64).
            prefix_kv_cache: depth-indexed list of (K, V) per VLM layer on
                this submesh, from `KVMigration.as_list(depth)`.
            attention_mask: joint mask [B, 1, S_suffix_padded,
                S_prefix + S_suffix_padded] (0 = unmasked).

        Returns: predicted clean actions [B, action_horizon_padded,
            action_dim] on this submesh. Caller owns the returned tensor.
        """
        if self.slice is None or self.suffix is None:
            raise RuntimeError(
                "StageDenoise.denoise requires both expert + suffix slices; "
                "ensure pi0_projections weights are present and initialize() ran."
            )

        num_steps = self.denoise_steps
        dt = -1.0 / num_steps  # timesteps walk 1.0 → 0.0

        x_t = noisy_actions
        x_t_owned = False  # caller owns the initial noisy_actions

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

            # In layer-paired mode the expert output lives on the LAST chip;
            # transport back to the first chip so suffix.project_output (and
            # the next step's embed_actions) can run locally there.
            if self.layer_paired_l1 and len(self.micro_submeshes) > 1:
                velocity_hidden_first = send_activation_via_host(velocity_hidden, self.micro_submeshes[0])
                ttnn.deallocate(velocity_hidden)
                velocity_hidden = velocity_hidden_first

            v_t = self.suffix.project_output(velocity_hidden)
            ttnn.deallocate(velocity_hidden)

            dx = ttnn.multiply(v_t, dt)
            ttnn.deallocate(v_t)
            x_t_new = ttnn.add(x_t, dx)
            ttnn.deallocate(dx)

            if x_t_owned:
                ttnn.deallocate(x_t)
            x_t = x_t_new
            x_t_owned = True

        return x_t  # caller owns

    # ------------------------------------------------------------------ #
    # KV migration access                                                #
    # ------------------------------------------------------------------ #

    @property
    def kv_migrator(self) -> Optional[KVMigration]:
        return self._kv_migrator
