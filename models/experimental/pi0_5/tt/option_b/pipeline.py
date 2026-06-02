# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pi0_5PipelineB — Option B orchestrator (vision → vlm/2 → vlm/2 → expert).

For the bring-up dry run this drives:

    stage 0 (host SigLIP + embed) → submesh 0
    transport → submesh 1
    stage 1 (VLM layers 0..K)
    transport → submesh 2
    stage 2 (VLM layers K..D)  (KV cache captured)
    KV migration → submesh 3
    stage 3.forward_expert_step (single step; full denoise loop is a follow-up)

The full Euler denoise loop (10 steps + suffix MLP + Δt schedule) is not yet
wired — see Stage3Expert.denoise NotImplementedError. The single-step path is
enough to validate that the full 4-stage submesh+transport+KV-migrate path
runs end-to-end without compilation errors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig

from .stages import StageLayout
from .stage_0_vision import Stage0Vision
from .stage_vlm import StageVLM
from .stage_3_expert import Stage3Expert
from .kv_migration import KVMigration
from .transport import send_activation_via_host


@dataclass
class StageTimings:
    stage_0_vision_ms: float = 0.0
    transport_0_to_1_ms: float = 0.0
    stage_1_vlm_first_half_ms: float = 0.0
    transport_1_to_2_ms: float = 0.0
    stage_2_vlm_second_half_ms: float = 0.0
    kv_migration_ms: float = 0.0
    stage_3_expert_step_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class Pi0_5PipelineB:
    """Orchestrates a single pi0.5 forward across 4 submeshes."""

    layout: StageLayout
    submeshes: List
    config: PaliGemmaConfig
    weights: Dict[str, Dict[str, torch.Tensor]]
    stage_0: Optional[Stage0Vision] = None
    stage_1: Optional[StageVLM] = None
    stage_2: Optional[StageVLM] = None
    stage_3: Optional[Stage3Expert] = None
    kv_migrator: Optional[KVMigration] = None

    def initialize(self) -> None:
        s = self.layout.stages
        self.stage_0 = Stage0Vision(s[0], self.submeshes[0], self.config, self.weights)
        self.stage_1 = StageVLM(s[1], self.submeshes[1], self.config, self.weights)
        self.stage_2 = StageVLM(s[2], self.submeshes[2], self.config, self.weights)
        self.stage_3 = Stage3Expert(s[3], self.submeshes[3], self.config, self.weights)
        self.kv_migrator = KVMigration(expert_submesh=self.submeshes[3])

        self.stage_0.initialize()
        self.stage_1.initialize()
        self.stage_2.initialize()
        self.stage_3.initialize(self.kv_migrator)

    def run_one_step(
        self,
        pixel_values: torch.Tensor,
        language_token_ids: torch.Tensor,
        suffix_hidden: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        attention_mask_prefix: Optional["ttnn.Tensor"] = None,
        attention_mask_suffix: Optional["ttnn.Tensor"] = None,
    ) -> Tuple["ttnn.Tensor", StageTimings]:
        """Run vision → VLM/2 → VLM/2 → KV-migrate → one expert step.

        suffix_hidden + adarms_cond are precomputed by the caller (eventually
        the suffix MLP on stage 3). They live on stage 3's submesh.

        Returns (expert_output_hidden, timings).
        """
        if self.stage_0 is None:
            raise RuntimeError("Pipeline.run_one_step called before initialize()")

        t = StageTimings()
        start_total = time.perf_counter()

        # Stage 0 — vision + embed → prefix hidden on submesh 0
        t0 = time.perf_counter()
        prefix_hidden_s0 = self.stage_0.forward(pixel_values, language_token_ids)
        t.stage_0_vision_ms = (time.perf_counter() - t0) * 1000

        # Transport 0 → 1
        t0 = time.perf_counter()
        h_on_1 = send_activation_via_host(prefix_hidden_s0, self.submeshes[1])
        t.transport_0_to_1_ms = (time.perf_counter() - t0) * 1000

        # Stage 1 — VLM first half
        t0 = time.perf_counter()
        h_after_1 = self.stage_1.forward(h_on_1, attention_mask=attention_mask_prefix, use_cache=True)
        t.stage_1_vlm_first_half_ms = (time.perf_counter() - t0) * 1000

        # Transport 1 → 2
        t0 = time.perf_counter()
        h_on_2 = send_activation_via_host(h_after_1, self.submeshes[2])
        t.transport_1_to_2_ms = (time.perf_counter() - t0) * 1000

        # Stage 2 — VLM second half (captures KV; emits)
        t0 = time.perf_counter()
        h_after_2 = self.stage_2.forward(h_on_2, attention_mask=attention_mask_prefix, use_cache=True)
        t.stage_2_vlm_second_half_ms = (time.perf_counter() - t0) * 1000

        # KV migration: stage 1 + stage 2 KV → stage 3 submesh
        t0 = time.perf_counter()
        prefix_kv_on_3 = self._migrate_kv()
        t.kv_migration_ms = (time.perf_counter() - t0) * 1000

        # Stage 3 — one expert step (full denoise loop is a follow-up)
        t0 = time.perf_counter()
        expert_out = self.stage_3.forward_expert_step(
            suffix_hidden,
            adarms_cond,
            prefix_kv_cache=prefix_kv_on_3,
            attention_mask=attention_mask_suffix,
        )
        t.stage_3_expert_step_ms = (time.perf_counter() - t0) * 1000

        t.total_ms = (time.perf_counter() - start_total) * 1000
        return expert_out, t

    def _migrate_kv(self) -> List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        """Collect KV from stages 1 and 2, ship to stage 3's submesh.

        Returns a depth-indexed list (length = vlm_config.depth) of (K, V)
        tensors on submesh 3, or None for layers not present.
        """
        depth = self.config.vlm_config.depth
        migrated: List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]] = [None] * depth
        for stage in (self.stage_1, self.stage_2):
            for global_idx, kv in stage.get_kv_cache():
                k_on_3 = send_activation_via_host(kv[0], self.submeshes[3])
                v_on_3 = send_activation_via_host(kv[1], self.submeshes[3])
                migrated[global_idx] = (k_on_3, v_on_3)
        return migrated
