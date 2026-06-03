# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pi0_5PipelineC — Option C orchestrator (vision → prefill → denoise).

Drives one pi0.5 inference across 3 heterogeneous submeshes:

    stage 0 (vision, 4 chips):
        host SigLIP + mm_projector + host embed lookup →
        upload prefix hidden to vision submesh
        ↓ transport via host bounce
    stage 1 (prefill, 18 chips):
        VLM transformer prefill (1 layer per chip in the target
        placement; replicated in the scaffolding pass) →
        capture per-layer (K, V)
        ↓ layer-paired KV migration via host bounce
    stage 2 (denoise, 6 chips):
        suffix MLP + expert backbone +
        10-step Euler integrator → final action tensor
        ↓ host bounce back
    HOST: clean actions [B, action_horizon, action_dim]

Transport is host-bounce in this first cut (matches Option B); the
tt-blaze direct D2D socket path is the follow-up. Per-step timings are
recorded in `StageTimingsC` so end-to-end perf can be reported
side-by-side with Option B's `StageTimings`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig

from .kv_migration import KVMigration
from .stage_denoise import StageDenoise
from .stage_prefill import StagePrefill
from .stage_vision import StageVision
from .stages import StageLayout
from .transport import send_activation_via_host


@dataclass
class StageTimingsC:
    """Wall-clock breakdown of a single end-to-end Pi0_5PipelineC run."""

    stage_0_vision_ms: float = 0.0
    transport_0_to_1_ms: float = 0.0
    stage_1_prefill_ms: float = 0.0
    kv_migration_ms: float = 0.0
    stage_2_denoise_ms: float = 0.0
    transport_2_to_host_ms: float = 0.0
    total_ms: float = 0.0

    # Per-Euler-step timings (length == denoise_steps), so we can see
    # cold-step vs warm-step costs.
    denoise_step_ms: List[float] = field(default_factory=list)


@dataclass
class Pi0_5PipelineC:
    """End-to-end Option C driver.

    Construction:
        layout    — `StageLayout` from `stages.build_default_layout()`.
        submeshes — list of 3 MeshDevices from `open_galaxy_mesh(layout)`.
        config    — full PaliGemma config.
        weights   — categorized weights dict (vlm_vision, vlm_projector,
                    vlm_language, action_expert, pi0_projections).

    Usage:
        with open_galaxy_mesh(layout) as (parent, submeshes):
            pipe = Pi0_5PipelineC(layout, submeshes, cfg, weights)
            pipe.initialize()
            actions, timings = pipe.run_inference(
                pixel_values=images_torch,
                language_token_ids=tokens_torch,
                noisy_actions=noise_torch,
                attention_mask_prefix=mask_prefix_torch,
                attention_mask_joint=mask_joint_torch,
            )
    """

    layout: StageLayout
    submeshes: List
    config: PaliGemmaConfig
    weights: Dict[str, Dict[str, torch.Tensor]]
    denoise_steps: int = 10
    action_dim: int = 32
    action_horizon: int = 50
    embed_on_host: bool = True
    # Layer-paired L1 placement: 1 VLM layer / prefill chip, `expert_layers_per_chip`
    # expert layers / denoise chip. Each layer's weights live in L1 on exactly its
    # owning chip — the target Option C placement (deployment plan §3.1). When False
    # (default), all layers are replicated across the full submesh in DRAM, which
    # is the smoke / bench path. Stages already accept this flag; the pipeline
    # just plumbs it through and routes transports via each stage's
    # first_chip_submesh / last_chip_submesh accessor.
    layer_paired_l1: bool = False
    # Number of expert layers per denoise chip when `layer_paired_l1=True`.
    # Default 3 matches stage_denoise.EXPERT_LAYERS_PER_DENOISE_CHIP — 18 layers
    # over 6 chips. Ignored when layer_paired_l1 is False.
    expert_layers_per_chip: int = 3
    # Device-side SigLIP: SigLIP-27 runs across 3 vision chips + 1 projector chip
    # (`Pi0_5OptionCVisionSliceSplit`) instead of on the host CPU. When False
    # (default), `embed_on_host` controls the host path. The two flags are
    # mutually exclusive at the slice level — passing device_siglip=True
    # overrides embed_on_host.
    device_siglip: bool = False
    # When True (and `device_siglip=True`), every SigLIP weight tensor +
    # the mm_projector weight/bias are migrated to L1 after construction —
    # the deployment plan §3.1 placement (~140-160 MB L1 / vision chip).
    # Ignored when device_siglip is False (host path has no on-chip weights).
    vision_weights_l1: bool = False
    # Tensor-parallel factor INSIDE stage 1 (prefill). When > 1, the (6,3)
    # prefill submesh is carved into (prefill_tp_size, 1) col-pair sub-meshes
    # and each sub-mesh runs N VLM layers with TP=prefill_tp_size sharding.
    # Default 1 = no TP (replicated or layer-paired path, depending on
    # `layer_paired_l1`). Currently supports 1 or 2 (the col-pair carving).
    # See OPTION_C_TP_WITHIN_STAGE_PLAN.md for the per-bank arithmetic.
    prefill_tp_size: int = 1
    # When True (and `prefill_tp_size > 1`), walk every constructed TP block
    # post-init and migrate matmul weights to L1 via to_memory_config + dealloc.
    # The L1 path is only viable in TP mode because TP=2 shrinks the per-chip
    # matmul shape, which in turn shrinks the kernel's static CB region —
    # leaving enough L1 headroom above it for the weights to land cleanly.
    # See OPTION_B_L1_ASSESSMENT.md for the validated arithmetic (Option B
    # TP=8 path; Option C TP=2 has even more headroom: 0.46 vs 1.03 MB/bank).
    prefill_weights_l1: bool = False
    # When True (and `prefill_weights_l1=True`), only migrate the MLP weights
    # (gate/up/down) to L1; Q/K/V/O stay in DRAM. Needed when per-chip weight
    # load otherwise overflows the L1 headroom above the matmul kernel's
    # static CB region (e.g. vlm_depth=18 with 2 layers per (2,1) at TP=2).
    prefill_weights_l1_mlp_only: bool = False

    stage_0: Optional[StageVision] = None
    stage_1: Optional[StagePrefill] = None
    stage_2: Optional[StageDenoise] = None
    kv_migrator: Optional[KVMigration] = None

    # ------------------------------------------------------------------ #
    # Build                                                              #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Build the three stage orchestrators and upload all weights.

        Order matters slightly: we initialize denoise last so the migrator
        bound to its submesh is ready when prefill emits KV.
        """
        if len(self.submeshes) < 3:
            raise ValueError(
                f"Pi0_5PipelineC needs ≥3 submeshes (vision, prefill, denoise); " f"got {len(self.submeshes)}"
            )

        s = self.layout.stages
        self.stage_0 = StageVision(
            s[0],
            self.submeshes[0],
            self.config,
            self.weights,
            embed_on_host=self.embed_on_host,
            device_siglip=self.device_siglip,
            vision_weights_l1=self.vision_weights_l1,
        )
        self.stage_1 = StagePrefill(
            s[1],
            self.submeshes[1],
            self.config,
            self.weights,
            layer_paired_l1=self.layer_paired_l1,
            prefill_tp_size=self.prefill_tp_size,
        )
        self.stage_2 = StageDenoise(
            s[2],
            self.submeshes[2],
            self.config,
            self.weights,
            denoise_steps=self.denoise_steps,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            layer_paired_l1=self.layer_paired_l1,
            layers_per_chip=self.expert_layers_per_chip,
        )
        self.kv_migrator = KVMigration(denoise_submesh=self.submeshes[2])

        self.stage_0.initialize()
        self.stage_1.initialize()
        self.stage_2.initialize(kv_migrator=self.kv_migrator)

        # Opt-in L1 migration for the TP prefill path. Mirrors Option B's
        # pattern. The migration helper walks each TP block on the slice
        # and moves matmul weights from DRAM to L1.
        if self.prefill_weights_l1 and self.prefill_tp_size > 1:
            from ._l1_migration import migrate_prefill_weights_to_l1

            migrate_prefill_weights_to_l1(self, mlp_only=self.prefill_weights_l1_mlp_only)

    # ------------------------------------------------------------------ #
    # End-to-end forward                                                 #
    # ------------------------------------------------------------------ #

    def run_inference(
        self,
        pixel_values: torch.Tensor,
        language_token_ids: torch.Tensor,
        noisy_actions: torch.Tensor,
        attention_mask_prefix: Optional[torch.Tensor] = None,
        attention_mask_joint: Optional[torch.Tensor] = None,
    ) -> Tuple["ttnn.Tensor", StageTimingsC]:
        """One end-to-end inference: returns (clean_actions, timings).

        Args:
            pixel_values:          torch [B, 3, H, W].
            language_token_ids:    torch [B, S_lang] int.
            noisy_actions:         torch [B, action_horizon_padded, action_dim]
                                   (action_horizon padded to tile alignment).
            attention_mask_prefix: torch [B, 1, S_prefix, S_prefix] for the
                                   VLM prefill self-attention (0 = unmasked).
                                   If None, an all-unmasked mask is built
                                   from the resulting prefix length.
            attention_mask_joint:  torch [B, 1, S_suffix_padded,
                                   S_prefix + S_suffix_padded] for the
                                   expert's joint attention over migrated
                                   prefix KV and the suffix tokens.

        Returns: (clean_actions ttnn.Tensor on the denoise submesh,
                  StageTimingsC).
        """
        if self.stage_0 is None or self.stage_1 is None or self.stage_2 is None:
            raise RuntimeError("Pi0_5PipelineC.run_inference called before initialize()")
        if self.kv_migrator is None:
            raise RuntimeError("KV migrator is not built; call initialize() first")

        t = StageTimingsC()
        t_total0 = time.perf_counter()

        # ---------- Stage 0: vision -----------------------------------------
        t0 = time.perf_counter()
        prefix_hidden_s0 = self.stage_0.forward(pixel_values, language_token_ids)
        t.stage_0_vision_ms = (time.perf_counter() - t0) * 1000

        # ---------- Transport 0 → 1 -----------------------------------------
        # In paired mode, prefill expects the input on chip 0 of the prefill
        # submesh (the chip owning VLM layer 0), NOT on the full 18-chip
        # submesh. The stage's first_chip_submesh accessor returns the right
        # target in both modes (full submesh in replicated mode, chip 0 in
        # paired mode), so this single line covers both paths.
        t0 = time.perf_counter()
        prefill_in_submesh = self.stage_1.first_chip_submesh
        prefix_hidden_s1 = send_activation_via_host(prefix_hidden_s0, prefill_in_submesh)
        ttnn.deallocate(prefix_hidden_s0)
        t.transport_0_to_1_ms = (time.perf_counter() - t0) * 1000

        # ---------- Stage 1: VLM prefill ------------------------------------
        # Mask must live on the same submesh as the prefill input — chip 0 in
        # paired mode; full prefill submesh in replicated mode.
        S_prefix = prefix_hidden_s1.shape[1]
        mask_prefix_s1 = self._build_or_upload_prefix_mask(attention_mask_prefix, S_prefix, prefill_in_submesh)
        t0 = time.perf_counter()
        h_after_1 = self.stage_1.forward(
            prefix_hidden_s1,
            attention_mask=mask_prefix_s1,
            use_cache=True,
        )
        t.stage_1_prefill_ms = (time.perf_counter() - t0) * 1000
        # h_after_1 is not consumed downstream in the no-prefill-residual
        # path; deallocate to free L1 on the prefill submesh.
        ttnn.deallocate(h_after_1)

        # ---------- KV migration (layer-paired, prefill → denoise) ----------
        t0 = time.perf_counter()
        per_layer_kv: List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]] = [None] * self.config.vlm_config.depth
        for global_idx, kv in self.stage_1.get_kv_cache():
            per_layer_kv[global_idx] = kv
        self.kv_migrator.migrate_layer_paired(per_layer_kv)
        prefix_kv_on_denoise = self.kv_migrator.as_list(self.config.vlm_config.depth)
        t.kv_migration_ms = (time.perf_counter() - t0) * 1000

        # ---------- Stage 2: denoise (full Euler loop) ----------------------
        # Suffix MLP and the first expert layer live on chip 0 of the denoise
        # submesh in paired mode, so x_t and the joint mask both need to land
        # there. In replicated mode this is the full denoise submesh.
        denoise_in_submesh = self.stage_2.first_chip_submesh
        noisy_on_denoise = self._upload_replicated(noisy_actions, denoise_in_submesh, dtype=ttnn.bfloat16)
        joint_mask_on_denoise = self._build_or_upload_joint_mask(
            attention_mask_joint,
            S_prefix=S_prefix,
            S_suffix_padded=noisy_on_denoise.shape[1],
            submesh=denoise_in_submesh,
        )

        t0 = time.perf_counter()
        clean_actions, step_ms = self._denoise_with_per_step_timing(
            noisy_on_denoise,
            prefix_kv_cache=prefix_kv_on_denoise,
            attention_mask=joint_mask_on_denoise,
        )
        t.stage_2_denoise_ms = (time.perf_counter() - t0) * 1000
        t.denoise_step_ms = step_ms

        t.total_ms = (time.perf_counter() - t_total0) * 1000
        return clean_actions, t

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _denoise_with_per_step_timing(
        self,
        noisy_actions: "ttnn.Tensor",
        prefix_kv_cache: List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]],
        attention_mask: "ttnn.Tensor",
    ) -> Tuple["ttnn.Tensor", List[float]]:
        """Identical to StageDenoise.denoise but records per-step wall time."""
        assert self.stage_2 is not None and self.stage_2.suffix is not None

        num_steps = self.stage_2.denoise_steps
        dt = -1.0 / num_steps

        x_t = noisy_actions
        x_t_owned = False
        step_times: List[float] = []

        for i in range(num_steps):
            t_step = 1.0 - i / num_steps
            B = x_t.shape[0]
            t_s0 = time.perf_counter()

            adarms_cond = self.stage_2.suffix.embed_adarms_cond(t_step, batch_size=B)
            suffix_h = self.stage_2.suffix.embed_actions(x_t)

            velocity_hidden = self.stage_2.slice.forward(
                suffix_h,
                adarms_cond,
                prefix_kv_cache=prefix_kv_cache,
                attention_mask=attention_mask,
            )
            ttnn.deallocate(suffix_h)
            ttnn.deallocate(adarms_cond)

            # In layer-paired mode the expert chain emits its output on the LAST
            # micro-submesh; the suffix MLP lives on chip 0, so we host-bounce
            # the hidden back before project_output. Mirrors StageDenoise.denoise().
            if (
                self.stage_2.layer_paired_l1
                and self.stage_2.micro_submeshes is not None
                and len(self.stage_2.micro_submeshes) > 1
            ):
                velocity_hidden_first = send_activation_via_host(velocity_hidden, self.stage_2.micro_submeshes[0])
                ttnn.deallocate(velocity_hidden)
                velocity_hidden = velocity_hidden_first

            v_t = self.stage_2.suffix.project_output(velocity_hidden)
            ttnn.deallocate(velocity_hidden)

            dx = ttnn.multiply(v_t, dt)
            ttnn.deallocate(v_t)
            x_t_new = ttnn.add(x_t, dx)
            ttnn.deallocate(dx)
            if x_t_owned:
                ttnn.deallocate(x_t)
            x_t = x_t_new
            x_t_owned = True

            step_times.append((time.perf_counter() - t_s0) * 1000)

        return x_t, step_times

    def _upload_replicated(
        self,
        t: torch.Tensor,
        submesh,
        dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=None,
    ) -> "ttnn.Tensor":
        if memory_config is None:
            memory_config = ttnn.L1_MEMORY_CONFIG
        if t.dtype != torch.float32:
            t = t.to(torch.float32)
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=layout,
            device=submesh,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submesh),
            memory_config=memory_config,
        )

    def _build_or_upload_prefix_mask(
        self,
        mask_torch: Optional[torch.Tensor],
        S_prefix: int,
        submesh,
    ) -> "ttnn.Tensor":
        if mask_torch is None:
            mask_torch = torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32)
        # SDPA requires the attention mask in DRAM (sdpa_device_operation.cpp:80).
        return self._upload_replicated(
            mask_torch,
            submesh,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _build_or_upload_joint_mask(
        self,
        mask_torch: Optional[torch.Tensor],
        S_prefix: int,
        S_suffix_padded: int,
        submesh,
    ) -> "ttnn.Tensor":
        if mask_torch is None:
            mask_torch = torch.zeros(1, 1, S_suffix_padded, S_prefix + S_suffix_padded, dtype=torch.float32)
        return self._upload_replicated(
            mask_torch,
            submesh,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
