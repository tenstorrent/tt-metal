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
    #
    # Master flag for "use paired on every stage that doesn't have an explicit
    # override." Suppressed for prefill when `prefill_tp_size > 1` (TP and paired
    # are mutually exclusive on prefill). Honored for denoise unless
    # `denoise_layer_paired_l1` is set explicitly.
    layer_paired_l1: bool = False
    # Explicit override for the denoise stage's paired placement. None = follow
    # the master `layer_paired_l1` flag. True = force denoise to paired even
    # when `layer_paired_l1=False` (e.g. when prefill runs TP=2 but denoise
    # still needs paired to fit L1 at depth=18). False = force replicated even
    # when the master flag is True.
    denoise_layer_paired_l1: Optional[bool] = None
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
    # Override the per-chip SigLIP layer split (sum must equal 27). When None
    # (default), `Pi0_5OptionCVisionSliceSplit` uses its hardcoded default
    # (4+4+4+4+3+3+3+2 on the 8-chip submesh). Set to `[7, 7, 7, 6]` for the
    # 4-chip vision submesh (PI0_OC_VISION_SHAPE="1,4"). Only consulted when
    # `device_siglip=True`.
    vision_layers_per_chip: Optional[List[int]] = None
    # Keep SigLIP pos_embed weight in DRAM even when `vision_weights_l1=True`.
    # pos_embed is fed to ttnn.embedding whose static CB region (~400 KB / core)
    # collides with adjacent L1 weight buffers on tight per-chip layouts (e.g.
    # 4-chip vision with 7 SigLIP layers + patch_embed on chip 0). Cost of
    # keeping it in DRAM is ~0.6 MB read once per inference — negligible.
    vision_keep_pos_embed_dram: bool = False
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
    # When True (and `prefill_weights_l1=True` and NOT mlp_only), migrate Q+O+MLP
    # but leave kv_proj in DRAM. kv_proj is REPLICATED across the (2,1) sub-mesh
    # (num_kv_heads=1 doesn't shard at TP=2) so it occupies ~2 MB / chip at
    # depth=18 × 2 layers/sub-mesh. Skipping it frees high-address L1 the
    # allocator otherwise spills into the matmul kernel's static CB region.
    prefill_weights_l1_skip_kv: bool = False
    # Non-TP layer-paired counterpart to `prefill_weights_l1_mlp_only`. When True
    # AND `layer_paired_l1=True`, the layer-paired loader keeps `o_proj` in DRAM
    # alongside the already-DRAM fused `wqkv`. Frees ~4.5 MB / chip — the
    # fragmentation slack the Tilize transient needs on the prefill forward path
    # at S=1024 with no TP. Has no effect when TP > 1 or layer_paired_l1=False.
    prefill_attn_dram: bool = False
    # When True (and `prefill_weights_l1=True` and `prefill_tp_size > 1`),
    # migrate MLP weights to L1 WIDTH_SHARDED instead of interleaved L1. Each
    # core owns a deterministic [K, N_per_chip / num_cores] slice at high-L1;
    # avoids the bank-fragmentation + kernel CB clash interleaved L1 hits.
    # Combined with TP=2 + `prefill_weights_l1_mlp_only=1`, this is the path
    # to fitting Gemma-2B MLP weights on L1 at S=1024. Default off.
    prefill_mlp_l1_width_sharded: bool = False
    # Grid (gx, gy) for the MLP width-shard. Default (8, 8) = 64 cores; with
    # TP=2 (per-chip N=8192) that's per-core [K, 128] = 272 KB at bf8.
    prefill_mlp_l1_width_sharded_grid: tuple = (8, 8)
    # D2D parent-mesh slice for prefill (Option A end-state). When True:
    #   - replaces stage_1.forward with Pi0_5OptionCVLMSliceParent.forward_real_block_chain
    #   - uses migrate_layer_paired_d2d (P2P fabric) instead of host-bounce KV migration
    #   - estimated savings vs current path: ~150-180 ms (inter-layer host bounces + KV migration)
    # Requires parent_mesh to be passed in (the galaxy parent the submeshes were carved from)
    # and the parent mesh to have been opened with set_fabric_config(FABRIC_1D).
    use_parent_mesh_slice: bool = False
    # D2D parent-mesh denoise slice for the expert chain (Step 3 of D2D
    # rollout). When True:
    #   - replaces stage_2's host-bouncing expert chain with
    #     Pi0_5OptionCExpertSliceParent (3 layers per chip × 6 chips, weights
    #     sharded on the parent mesh, P2P advance between expert chips)
    #   - all hops are same-column (denoise sits at parent col 3, rows 2..7)
    #     so a single fabric hop per transition — no multihop needed
    # Scaffolding stage: the slice is built and weights are uploaded, but the
    # forward + Euler wrap-back P2P land in follow-up commits. The flag is
    # plumbed now so the benchmark surface stays stable across the
    # incremental commits.
    use_parent_mesh_denoise: bool = False
    # Galaxy parent MeshDevice (8, 4). Required when use_parent_mesh_slice
    # OR use_parent_mesh_denoise is True.
    parent_mesh: Optional["ttnn.MeshDevice"] = None
    # When True, post-initialize walk the denoise stage's expert + suffix
    # slices and migrate every matmul weight + LN/mod weight from DRAM to L1.
    # Default False = today's behavior (uploads land in DRAM via the default
    # `_upload_l1_replicated` / `_upload_single_chip_l1` paths which still
    # default to DRAM despite the helper names). Mirrors the Option B
    # `weights_l1` flag scoped to the denoise stage. NO DRAM fallbacks once
    # this is enabled — see `_l1_migration.migrate_pipeline_denoise_to_l1`.
    # Validation agent: enable this in the probe alongside `denoise_mod_sharded`.
    denoise_weights_l1: bool = False
    # When True, the per-block adaRMS modulation Dense `[1024, 6144]` is
    # SHARDED across the denoise submesh along its 6144 output axis instead
    # of replicated. Cuts the dominant per-chip weight load by `submesh.size`
    # (6x at the full 6-chip denoise submesh). The forward path issues a
    # `ttnn.all_gather` to materialize the full mod output per layer before
    # the modulation; requires fabric init at parent-open.
    #
    # IMPORTANT: when `denoise_mod_sharded=True`, the caller MUST open the
    # parent mesh with `open_galaxy_mesh(enable_fabric=True)` so the fabric
    # is up for ttnn.all_gather. The pipeline does not own the mesh and
    # cannot enable fabric itself. Default False = today's behavior
    # (replicated mod weight, no fabric needed).
    #
    # Note: today this flag is honored only by the non-paired (replicated)
    # `Pi0_5OptionCExpertSlice` constructor — sharding doesn't apply to
    # single-chip submeshes used in the paired path. The mod weight on the
    # paired path is already chip-local (one layer's mod = one chip).
    denoise_mod_sharded: bool = False

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
            vision_layers_per_chip=self.vision_layers_per_chip,
            vision_keep_pos_embed_dram=self.vision_keep_pos_embed_dram,
        )
        # Prefill paired is suppressed by TP mode (mutually exclusive in StagePrefill).
        # Denoise paired follows the explicit override when set, otherwise the master.
        prefill_paired = self.layer_paired_l1 and self.prefill_tp_size == 1
        denoise_paired = (
            self.denoise_layer_paired_l1 if self.denoise_layer_paired_l1 is not None else self.layer_paired_l1
        )

        self.stage_1 = StagePrefill(
            s[1],
            self.submeshes[1],
            self.config,
            self.weights,
            layer_paired_l1=prefill_paired,
            prefill_tp_size=self.prefill_tp_size,
            attn_dram=self.prefill_attn_dram,
        )
        self.stage_2 = StageDenoise(
            s[2],
            self.submeshes[2],
            self.config,
            self.weights,
            denoise_steps=self.denoise_steps,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            layer_paired_l1=denoise_paired,
            layers_per_chip=self.expert_layers_per_chip,
            mod_sharded=self.denoise_mod_sharded,
        )
        self.kv_migrator = KVMigration(denoise_submesh=self.submeshes[2])

        self.stage_0.initialize()
        # Skip stage_1 init when using the parent-mesh slice — the
        # parent_mesh_slice replaces stage_1's forward and KV emission, so
        # stage_1's replicated weight uploads would just be wasted memory.
        # We still keep the stage_1 object around for first_chip_submesh
        # access (the transport landing site).
        if not self.use_parent_mesh_slice:
            self.stage_1.initialize()
        self.stage_2.initialize(kv_migrator=self.kv_migrator)

        # D2D parent-mesh slice for prefill. When enabled, this REPLACES the
        # host-bouncing layer-paired prefill path. We construct it here but
        # delegate the actual forward to it in run_inference().
        self.parent_mesh_slice = None
        if self.use_parent_mesh_slice:
            if self.parent_mesh is None:
                raise ValueError("use_parent_mesh_slice=True requires parent_mesh to be set")
            from .stages import PREFILL_SUBMESH_OFFSET, PREFILL_SUBMESH_SHAPE
            from .vlm_slice import Pi0_5OptionCVLMSliceParent

            self.parent_mesh_slice = Pi0_5OptionCVLMSliceParent(
                config=self.config,
                weights=self.weights,
                parent_mesh=self.parent_mesh,
                prefill_offset=PREFILL_SUBMESH_OFFSET,
                prefill_shape=PREFILL_SUBMESH_SHAPE,
                layer_range=(0, self.config.vlm_config.depth),
            )

        # D2D parent-mesh slice for denoise (expert chain). Scaffolding stage:
        # the slice is built so weights upload + per-chip placement is
        # validated under the benchmark surface, but the forward path is
        # still the host-bounce layer-paired expert chain. Switching the
        # expert forward to this slice is the next commit in the rollout.
        self.parent_mesh_denoise_slice = None
        if self.use_parent_mesh_denoise:
            if self.parent_mesh is None:
                raise ValueError("use_parent_mesh_denoise=True requires parent_mesh to be set")
            from .expert_slice import Pi0_5OptionCExpertSliceParent
            from .stages import (
                DENOISE_SUBMESH_OFFSET,
                DENOISE_SUBMESH_SHAPE,
                EXPERT_LAYERS_PER_DENOISE_CHIP,
            )

            self.parent_mesh_denoise_slice = Pi0_5OptionCExpertSliceParent(
                config=self.config,
                weights=self.weights,
                parent_mesh=self.parent_mesh,
                denoise_offset=DENOISE_SUBMESH_OFFSET,
                denoise_shape=DENOISE_SUBMESH_SHAPE,
                expert_layer_range=(0, self.config.expert_config.depth),
                layers_per_chip=EXPERT_LAYERS_PER_DENOISE_CHIP,
            )

        # Opt-in L1 migration for the TP prefill path. Mirrors Option B's
        # pattern. The migration helper walks each TP block on the slice
        # and moves matmul weights from DRAM to L1.
        if self.prefill_weights_l1 and self.prefill_tp_size > 1:
            from ._l1_migration import migrate_prefill_weights_to_l1

            migrate_prefill_weights_to_l1(
                self,
                mlp_only=self.prefill_weights_l1_mlp_only,
                skip_kv=self.prefill_weights_l1_skip_kv,
                mlp_width_sharded_grid=(
                    self.prefill_mlp_l1_width_sharded_grid if self.prefill_mlp_l1_width_sharded else None
                ),
            )

        # Opt-in L1 migration for the denoise (expert + suffix) path.
        # Mirrors the Option B `weights_l1` migration step scoped to stage 2.
        if self.denoise_weights_l1:
            from ._l1_migration import migrate_pipeline_denoise_to_l1

            migrate_pipeline_denoise_to_l1(self)

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
        if self.use_parent_mesh_slice:
            # Vision → parent mesh directly (one host bounce, no
            # intermediate prefill-submesh hop).
            prefix_hidden_s1 = self._lift_activation_to_parent_mesh(prefix_hidden_s0)
            prefill_in_submesh = None  # not used in this path
        else:
            prefill_in_submesh = self.stage_1.first_chip_submesh
            prefix_hidden_s1 = send_activation_via_host(prefix_hidden_s0, prefill_in_submesh)
        ttnn.deallocate(prefix_hidden_s0)
        t.transport_0_to_1_ms = (time.perf_counter() - t0) * 1000

        # ---------- Stage 1: VLM prefill ------------------------------------
        S_prefix = prefix_hidden_s1.shape[-2] if self.use_parent_mesh_slice else prefix_hidden_s1.shape[1]
        if self.use_parent_mesh_slice:
            # D2D parent-mesh path: activation already on the galaxy parent
            # at the first prefill chip's coord.
            t0 = time.perf_counter()
            h_after_1, per_layer_kv_on_parent = self.parent_mesh_slice.forward_real_block_chain(
                prefix_hidden_s1, return_kv_cache=True
            )
            t.stage_1_prefill_ms = (time.perf_counter() - t0) * 1000
            ttnn.deallocate(h_after_1)

            # D2D KV migration using the parent-mesh K/V tensors emitted by
            # the forward. This replaces 18 layers of host-bounce migration
            # with 18 layers of fabric P2P (microseconds vs ~120-150 ms).
            t0 = time.perf_counter()
            self.kv_migrator.migrate_layer_paired_d2d(per_layer_kv_on_parent)
            # Free the source KV tensors after migration.
            for k_t, v_t in per_layer_kv_on_parent:
                ttnn.deallocate(k_t)
                ttnn.deallocate(v_t)
            prefix_kv_on_denoise = self.kv_migrator.as_list(self.config.vlm_config.depth)
            t.kv_migration_ms = (time.perf_counter() - t0) * 1000
        else:
            # Host-bouncing layer-paired path (current default).
            # Mask must live on the same submesh as the prefill input — chip 0 in
            # paired mode; full prefill submesh in replicated mode.
            mask_prefix_s1 = self._build_or_upload_prefix_mask(attention_mask_prefix, S_prefix, prefill_in_submesh)
            t0 = time.perf_counter()
            h_after_1 = self.stage_1.forward(
                prefix_hidden_s1,
                attention_mask=mask_prefix_s1,
                use_cache=True,
            )
            t.stage_1_prefill_ms = (time.perf_counter() - t0) * 1000
            ttnn.deallocate(h_after_1)

            # ---------- KV migration (layer-paired, prefill → denoise) ----------
            t0 = time.perf_counter()
            per_layer_kv: List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]] = [None] * self.config.vlm_config.depth
            for global_idx, kv in self.stage_1.get_kv_cache():
                per_layer_kv[global_idx] = kv
            denoise_micro = (
                self.stage_2.micro_submeshes if self.stage_2 is not None and self.stage_2.layer_paired_l1 else None
            )
            self.kv_migrator.migrate_layer_paired(per_layer_kv, denoise_micro_submeshes=denoise_micro)
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

    def _lift_activation_to_parent_mesh(self, src: "ttnn.Tensor") -> "ttnn.Tensor":
        """Move a vision-submesh activation onto the GALAXY PARENT mesh,
        placed at the first prefill chip's parent coord.

        Used by the D2D parent-mesh prefill path: vision emits prefix_hidden
        on the vision submesh; the parent-mesh slice needs it sharded on the
        parent mesh with the live data at the first prefill chip's coord.
        Goes via host once (single cross-stage transition); the 17 inter-
        layer transitions that follow are pure fabric D2D.
        """
        from .stages import GALAXY_PARENT_SHAPE, PREFILL_SUBMESH_OFFSET

        # Read the vision-submesh tensor down to torch.
        shards = ttnn.get_device_tensors(src)
        # For replicated vision output, any shard is the same — take shard 0.
        host = ttnn.to_torch(shards[0])  # [1, S, hidden] (or [1, 1, S, hidden])
        # Normalize to 4D [1, 1, S, hidden]
        while host.ndim < 4:
            host = host.unsqueeze(0)
        # Build a parent-mesh-sharded tensor: 32 slots along dim 0, with
        # real data at the first prefill chip's parent linear index.
        devices_total = GALAXY_PARENT_SHAPE[0] * GALAXY_PARENT_SHAPE[1]
        lin0 = (PREFILL_SUBMESH_OFFSET[0]) * GALAXY_PARENT_SHAPE[1] + PREFILL_SUBMESH_OFFSET[1]
        full = torch.zeros((devices_total,) + tuple(host.shape[1:]), dtype=host.dtype)
        full[lin0] = host[0]
        return ttnn.from_torch(
            full,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.parent_mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.parent_mesh, dim=0),
        )

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
