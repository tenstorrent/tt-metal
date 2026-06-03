# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 — VLM prefill orchestrator for Option C.

Owns the 18-chip prefill submesh, target placement: 1 VLM transformer
layer per chip + final RMSNorm on chip 17. For the scaffolding pass the
slice uploads weights replicated across the submesh — see
`vlm_slice.py` for the L1-resident upload helpers. Layer-paired
sharding (1 layer placed on its 1 owning chip, no replication) is the
follow-up; the orchestrator interface doesn't change when that lands.

The KV cache produced here is consumed by the denoise stage via the
one-shot layer-paired KV migration (`kv_migration.py`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig

from .mesh_setup import create_per_chip_submeshes, create_tp_submeshes_2x1
from .stages import StageSpec
from .vlm_slice import (
    Pi0_5OptionCVLMSlice,
    Pi0_5OptionCVLMSlicePaired,
    Pi0_5OptionCVLMSliceTP,
)


class StagePrefill:
    """Stage 1: full VLM transformer prefill on the prefill submesh.

    Two-phase construction: `__init__` records arguments, `initialize`
    uploads weights and builds the block list. This lets the pipeline
    orchestrator open every submesh first and then initialize stages in
    parallel (or in a controlled order if init memory pressure matters).
    """

    def __init__(
        self,
        spec: StageSpec,
        submesh,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        layer_paired_l1: bool = False,
        prefill_tp_size: int = 1,
    ) -> None:
        if spec.stage_idx != 1:
            raise AssertionError(f"StagePrefill must be stage 1, got {spec.stage_idx}")
        if spec.vlm_layer_range[1] <= spec.vlm_layer_range[0]:
            raise ValueError(f"StagePrefill requires a non-empty vlm_layer_range; got {spec.vlm_layer_range}")
        if prefill_tp_size not in (1, 2):
            raise ValueError(
                f"prefill_tp_size must be 1 or 2 (Option C currently only supports TP=2 "
                f"via (2,1) col-pairs on the (6,3) prefill submesh); got {prefill_tp_size}"
            )
        if prefill_tp_size > 1 and layer_paired_l1:
            raise ValueError(
                "prefill_tp_size>1 and layer_paired_l1 are mutually exclusive — "
                "TP slice already runs N layers per sub-mesh, paired places 1 layer per chip"
            )

        self.spec = spec
        self.submesh = submesh
        self.config = config
        self.weights = weights
        self.layer_lo, self.layer_hi = spec.vlm_layer_range
        self.num_layers = self.layer_hi - self.layer_lo
        self.layer_paired_l1 = layer_paired_l1
        self.prefill_tp_size = prefill_tp_size
        # Populated in initialize() when layer_paired_l1=True or TP > 1.
        self.micro_submeshes: Optional[List] = None
        self.tp_submeshes: Optional[List] = None
        self.slice = None  # one of Pi0_5OptionCVLMSlice, *Paired, *TP
        self.last_kv_cache: Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]] = None

    def initialize(self) -> None:
        """Upload weights for this stage's layer range onto the prefill submesh.

        When `layer_paired_l1=True`, the prefill submesh is carved into N
        single-chip micro-submeshes (one per layer in the range) and each
        layer's weights live in L1 on exactly its owning chip — the target
        Option C placement (deployment plan §3.1).
        """
        if self.slice is not None:
            return  # idempotent

        if self.prefill_tp_size > 1:
            # Carve the (6,3) submesh into N (tp,1) col-pair sub-meshes.
            # For TP=2 we get 9 sub-meshes; layers per sub-mesh = num_layers/9.
            self.tp_submeshes = create_tp_submeshes_2x1(self.submesh)
            n_subs = len(self.tp_submeshes)
            if self.num_layers % n_subs != 0:
                raise ValueError(
                    f"num_layers={self.num_layers} is not divisible by "
                    f"len(tp_submeshes)={n_subs} — uneven layer placement not supported"
                )
            layers_per_sub = self.num_layers // n_subs
            self.slice = Pi0_5OptionCVLMSliceTP(
                config=self.config,
                weights=self.weights,
                tp_submeshes=self.tp_submeshes,
                layer_range=(self.layer_lo, self.layer_hi),
                layers_per_submesh=layers_per_sub,
                tp_size=self.prefill_tp_size,
                holds_embed_tokens=self.spec.holds_embed_tokens,
                holds_vlm_final_norm=self.spec.holds_vlm_final_norm,
            )
        elif self.layer_paired_l1:
            self.micro_submeshes = create_per_chip_submeshes(self.submesh, self.num_layers)
            self.slice = Pi0_5OptionCVLMSlicePaired(
                config=self.config,
                weights=self.weights,
                micro_submeshes=self.micro_submeshes,
                layer_range=(self.layer_lo, self.layer_hi),
                holds_embed_tokens=self.spec.holds_embed_tokens,
                holds_vlm_final_norm=self.spec.holds_vlm_final_norm,
            )
        else:
            self.slice = Pi0_5OptionCVLMSlice(
                config=self.config,
                weights=self.weights,
                submesh=self.submesh,
                layer_range=(self.layer_lo, self.layer_hi),
                holds_embed_tokens=self.spec.holds_embed_tokens,
                holds_vlm_final_norm=self.spec.holds_vlm_final_norm,
            )

    @property
    def first_chip_submesh(self):
        """Submesh that callers must upload the input activation onto.

        - Replicated mode: the whole prefill submesh.
        - Layer-paired mode: the single-chip submesh owning the first layer.
        - TP mode: the first (tp_size,1) sub-mesh (which owns layers 0..N).
        """
        if self.prefill_tp_size > 1:
            assert self.tp_submeshes is not None, "first_chip_submesh accessed before initialize()"
            return self.tp_submeshes[0]
        if self.layer_paired_l1:
            assert self.micro_submeshes is not None, "first_chip_submesh accessed before initialize()"
            return self.micro_submeshes[0]
        return self.submesh

    @property
    def last_chip_submesh(self):
        """Submesh where the stage's final activation lives after `forward`.

        - Replicated mode: the whole prefill submesh.
        - Layer-paired mode: the single-chip submesh owning the last layer.
        - TP mode: the last (tp_size,1) sub-mesh.
        """
        if self.prefill_tp_size > 1:
            assert self.tp_submeshes is not None, "last_chip_submesh accessed before initialize()"
            return self.tp_submeshes[-1]
        if self.layer_paired_l1:
            assert self.micro_submeshes is not None, "last_chip_submesh accessed before initialize()"
            return self.micro_submeshes[-1]
        return self.submesh

    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        past_key_values: Optional[List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]] = None,
        use_cache: bool = True,
        cos_override: Optional["ttnn.Tensor"] = None,
        sin_override: Optional["ttnn.Tensor"] = None,
    ) -> "ttnn.Tensor":
        """Run the VLM layer range. KV cache is stashed for the migration
        emitter (see `get_kv_cache`)."""
        if self.slice is None:
            raise RuntimeError("StagePrefill.forward called before initialize()")

        hidden_states, new_cache = self.slice.forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cos_override=cos_override,
            sin_override=sin_override,
        )
        self.last_kv_cache = new_cache
        return hidden_states

    # ------------------------------------------------------------------ #
    # KV migration emitter                                               #
    # ------------------------------------------------------------------ #

    def get_kv_cache(self) -> List[Tuple[int, Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        """Return (global_layer_idx, (K, V)) tuples for this stage's layers.

        The pipeline driver feeds these to `KVMigration.migrate_layer_paired`
        which ships layer i's (K, V) to denoise chip `i // EXPERT_LAYERS_PER_DENOISE_CHIP`.
        """
        if self.slice is None or self.last_kv_cache is None:
            return []
        return self.slice.get_kv_cache_for_slice(self.last_kv_cache)
