# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pi0_5GLX1x8Pipeline — pi0.5 sample_actions on a single 1×8 mesh.

Three stages share the parent 1×8 mesh and run sequentially. No host-bounce
between stages and no fabric mesh sockets — all stage-to-stage data movement
is on-device CCL (ttnn.all_gather inside SigLIP DP, ttnn.all_reduce inside
StagePrefillTP8 MLP). Submeshes do not share tensor allocations, so keeping
every stage on the same mesh handle is what makes CCL handoff possible.

Stages:
    0. SigLIP DP — 3 real cameras + 5 zero dummies batch-sharded along dim 0
       (8 cameras, one per chip). Each chip runs the full SigLIP encoder
       (SigLIPCameraSlice) at bs=1. ttnn.all_gather(dim=0) replicates the
       (8, 256, 2048) output on every chip; ttnn.slice drops the 5 dummies.
    1. Prefill TP=8 — StagePrefillTP8 (col/row-parallel MLP, replicated attn,
       all_reduce per MLP block). Output (final_hidden, KV×18) replicated on
       all 8 chips.
    2. Denoise (replicated) — one ExpertChunkSlice(layer_range=(0, 18)) with
       implicit-replicated weights. 7 chips do redundant compute but
       latency = 1-chip; chip 0's copy is the output. 5-step Euler loop
       unrolled into the trace.

Trace machinery: capture full sample_actions on the parent mesh, replay via
(pipeline.py:568): pre-allocate persistent input buffers (pixel_values_buf,
lang_tokens_buf, x_t_fp32), eager warmup to JIT kernels, then a single
begin/end_trace_capture on the parent mesh. Replay = copy_host_to_device_tensor
into the buffers + ttnn.execute_trace + ttnn.to_torch on the output buffer.

Layout assumptions:
    * Parent mesh: 1×8 (open via mesh_setup.open_prefill_tp8_mesh(tp=8)).
    * All weights replicated EXCEPT prefill MLP (sharded by StagePrefillTP8).
    * num_cameras = 3 (production); padded to 8 for DP.
    * Denoise output replicated → ConcatMeshToTensor + [:1] slice to take chip 0.

Not in scope for v1 (TODO if needed):
    * PI0_UPSTREAM_MASKS path (per-layer attn masks + position-aware RoPE).
    * Per-stage *traced* sub-traces (would require pre-allocating 36 persistent
      KV buffers as inter-stage I/O). Per-stage *eager* timing is available via
      sample_actions_timed; trace-replay host-overhead split via
      sample_actions_traced_timed.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig, SuffixConfig
from models.experimental.pi0_5.tt.ttnn_gemma import (
    ada_rms_norm_no_gate_ttnn,
    ada_rms_norm_no_gate_precomputed_ttnn,
)
from models.experimental.pi0_5.tt.ttnn_pi0_5_model import (
    _MASK_VAL,
    _precompute_rope_table_torch,
)

from .expert_slice import ExpertChunkSlice
from .heads import _DenoiseHead, _PrefillHead
from .stage_prefill_tp8 import StagePrefillTP8
from .suffix_slice import SuffixSlice
from .vision_slice import SigLIPCameraSlice


_NUM_CHIPS_REQUIRED = 8
_DEFAULT_NUM_REAL_CAMS = 3  # production; overridable via PI0_NUM_CAMERAS env
_NUM_PATCHES = 256  # SigLIP 224/14 = 16 → 16² = 256 tokens per camera


class Pi0_5GLX1x8Pipeline:
    """End-to-end pi0.5 sample_actions on a 1×8 BH-Galaxy mesh.

    All stages share the same parent mesh device. Cross-stage data is
    replicated on the mesh (after SigLIP all_gather) so each stage's inputs
    are already in place — no submesh boundaries, no sockets, no host bounce.
    """

    def __init__(self, config: Pi0_5ModelConfig, weights: Dict[str, dict], mesh):
        if mesh.get_num_devices() != _NUM_CHIPS_REQUIRED:
            raise RuntimeError(
                f"Pi0_5GLX1x8Pipeline requires a {_NUM_CHIPS_REQUIRED}-chip mesh; got {mesh.get_num_devices()}"
            )
        self.mesh = mesh
        self.config = config
        self.num_denoising_steps = config.num_denoising_steps
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self._action_horizon_padded = ((config.action_horizon + 31) // 32) * 32

        self._patch_size = config.siglip_config.patch_size
        self._image_size = config.siglip_config.image_size
        self._vlm_hidden = config.vlm_config.width
        # Keep weights for set_num_denoising_steps() — the per-step block /
        # final modulation precomputes are derived from action_expert weights
        # and the timestep schedule, so they must be rebuilt whenever the
        # step count changes at runtime.
        self._weights = weights

        # ---- Stage 0: SigLIP DP — full encoder per chip (weights replicated) ----
        self.vision = SigLIPCameraSlice(
            config.siglip_config,
            weights["vlm_vision"],
            weights["vlm_projector"],
            mesh,
        )

        # ---- Stage 1: Prefill TP=8 — derives tp=8 from mesh.get_num_devices() ----
        # MLP gate/up/down weights sharded col/row across the mesh; attention
        # replicated. all_reduce(num_links=2) per MLP block sums down-proj partials.
        self.prefill = StagePrefillTP8(config, weights, mesh)

        # Lang token embedding lookup. _PrefillHead uploads embed_tokens with
        # no mesh_mapper → implicit replicate on every chip.
        self.prefill_head = _PrefillHead(weights["vlm_language"], mesh, self._vlm_hidden)

        # ---- Stage 2: Denoise — single chunk over all 18 layers, replicated ----
        suffix_cfg = SuffixConfig(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            expert_width=config.expert_config.width,
            pi05=True,
        )
        self.suffix = SuffixSlice(suffix_cfg, weights["pi0_projections"], mesh)
        self.denoise = ExpertChunkSlice(
            config.expert_config,
            weights["action_expert"],
            mesh,
            layer_range=(0, 18),
            max_seq_len=config.max_seq_len,
        )
        self.denoise_head = _DenoiseHead(weights["action_expert"], mesh)

        # ---- Pre-computed denoising schedule ----
        # All depend on self.num_denoising_steps. When the rollout switches N
        # at runtime (e.g. LIBERO sweep N=10 → N=5), set_num_denoising_steps()
        # must be called to rebuild these — otherwise the loop would run the
        # new step count but index entries built for the old schedule (e.g.
        # first 5 entries of a 10-step schedule → dt=-0.1 instead of the
        # correct -0.2, halving the denoise traversal and tanking accuracy).
        self._block_mods_per_step: List[List[Tuple["ttnn.Tensor", ...]]] = []
        self._final_mods_per_step: List[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = []
        self._build_denoise_schedule()

        # Real camera count comes from PI0_NUM_CAMERAS (production env) — 5
        # zero-dummy cams pad to 8 regardless. Range checked at upload time.
        self._num_real_cams = int(os.environ.get("PI0_NUM_CAMERAS", _DEFAULT_NUM_REAL_CAMS))
        if not 1 <= self._num_real_cams <= _NUM_CHIPS_REQUIRED:
            raise RuntimeError(f"PI0_NUM_CAMERAS={self._num_real_cams} out of range [1, {_NUM_CHIPS_REQUIRED}]")

        # Upstream-openpi-compat attention artifacts (prefix mask, position-
        # aware prefix/suffix RoPE, expert mask). Defaults to all-real
        # masks at init — backward-compatible with the perf test which
        # doesn't pass masks. The LIBERO adapter calls
        # `prepare_runtime_masks(img_masks, lang_masks)` per task to rebuild
        # these artifacts when the actual padding pattern is known.
        # Mirrors single-chip ttnn_pi0_5_model.py:_build_upstream_attn_artifacts.
        self._suffix_cos = None
        self._suffix_sin = None
        self._expert_attn_mask = None
        self._prefix_cos = None
        self._prefix_sin = None
        self._prefix_attn_mask = None
        self._artifact_mask_key = None  # (img_present_tuple, lang_real_count) — re-build trigger
        self._build_upstream_artifacts()  # init with all-real defaults

        # ---- Persistent buffers (pre-allocated for trace replay) ----
        # Allocated lazily on the first sample_actions call so the per-call
        # input shapes settle the buffer shape exactly once. After that,
        # copy_host_to_device_tensor refreshes contents without reallocation —
        # trace-safe (the buffer's tensor ID stays stable across replays).
        self.pixel_values_buf = None  # (8, H, W/patch, 3*patch) bf16 ROW_MAJOR — fold fast path
        self.lang_tokens_buf = None  # (1, lang_len) uint32 ROW_MAJOR

        # Persistent fp32 x_t buffer (replicated on all 8 chips). The Euler
        # loop reads + writes this in place so the buffer ID survives the
        # loop — required for trace replay.
        _zero_noise = torch.zeros(1, self._action_horizon_padded, self.action_dim, dtype=torch.float32)
        self.x_t_fp32 = ttnn.from_torch(
            _zero_noise,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Trace state.
        self._trace_id = None
        self._captured_actions = None
        # Staged-trace state (3 sub-traces on the same mesh).
        self._vision_trace_id = None
        self._prefill_trace_id = None
        self._denoise_trace_id = None
        self._staged_actions = None

    # ──────────── Stage-0 helpers (SigLIP DP) ────────────────────────────

    def _stack_and_fold_pixels(self, images: List[torch.Tensor]) -> torch.Tensor:
        """Stack N real cameras + (8-N) zero dummies, then host-pre-fold to the
        layout PatchEmbeddingTTNN._forward_fold accepts (the FAST path):

            BCHW (raw)
              → BHWC (channel-last)             permute (0, 2, 3, 1)
              → (B, H, W/patch, 3*patch)        reshape — saves a device reshape

        Returns a torch tensor uploaded ROW_MAJOR per ttnn.fold's input expectation.
        Mirrors _prefix_setup in test_perf_siglip_dp_vs_single.py.
        """
        n_real = len(images)
        if n_real != self._num_real_cams:
            raise RuntimeError(f"Pipeline configured for {self._num_real_cams} real cameras; got {n_real}")
        real = torch.cat(images, dim=0)  # (n_real, 3, H, W)
        if real.shape[0] < _NUM_CHIPS_REQUIRED:
            pad = torch.zeros(
                _NUM_CHIPS_REQUIRED - real.shape[0],
                real.shape[1],
                real.shape[2],
                real.shape[3],
                dtype=real.dtype,
            )
            real = torch.cat([real, pad], dim=0)  # (8, 3, H, W)
        x = real.permute(0, 2, 3, 1).contiguous()  # (8, H, W, 3)
        B, H, W, C = x.shape
        x = x.reshape(B, H, W // self._patch_size, C * self._patch_size).contiguous()
        return x

    def _run_vision_dp(self, pixel_values_ttnn: "ttnn.Tensor") -> "ttnn.Tensor":
        """Run SigLIP on all 8 chips at bs=1 per chip; gather then slice off dummies.

        Returns (n_real, 256, vlm_width) replicated on all 8 chips.
        """
        # Per-chip forward: input is (1, H, W/patch, 3*patch) ROW_MAJOR shard;
        # PatchEmbeddingTTNN._forward_fold detects the pre-reshape via the
        # last-dim == 3*patch branch and skips the device reshape.
        vision_out = self.vision.forward(pixel_values_ttnn)  # (1, 256, 2048) per chip

        # All-gather along batch dim → (8, 256, 2048) replicated on every chip.
        gathered = ttnn.all_gather(
            vision_out,
            dim=0,
            num_links=2,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(vision_out)

        # Drop the 5 dummy-camera rows. Result is replicated; each chip slices
        # its own copy of the same (n_real, 256, 2048) tensor.
        real = ttnn.slice(gathered, [0, 0, 0], [self._num_real_cams, _NUM_PATCHES, self._vlm_hidden])
        ttnn.deallocate(gathered)
        return real

    # ──────────── Stage-0b helper (prefix construction) ─────────────────

    def _build_prefix(self, vision_real: "ttnn.Tensor", lang_tokens) -> "ttnn.Tensor":
        """Reshape (n_real, 256, vlm_w) → (1, n_real·256, vlm_w); concat with
        lang embedding → (1, prefix_len, vlm_w). All replicated on the mesh.

        Builds the replicated [image ; lang] prefix on-device, but on
        the multi-chip parent mesh — every chip computes the same reshape +
        concat on its replicated copy.
        """
        n_cams = int(vision_real.shape[0])
        v = ttnn.reshape(vision_real, (1, n_cams * _NUM_PATCHES, self._vlm_hidden))
        lang_emb = self.prefill_head.embed_lang(lang_tokens)
        prefix = ttnn.concat([v, lang_emb], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        if prefix.layout != ttnn.TILE_LAYOUT:
            prefix = ttnn.to_layout(prefix, ttnn.TILE_LAYOUT)
        return prefix

    # ──────────── Stage-2 helper (final norm + project) ─────────────────

    def _apply_final_norm_and_project(
        self,
        expert_out: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        precomputed_final_mod: Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = None,
    ) -> "ttnn.Tensor":
        if precomputed_final_mod is not None:
            scale1, shift = precomputed_final_mod
            normed = ada_rms_norm_no_gate_precomputed_ttnn(
                expert_out, scale1, shift, self.config.expert_config.rms_norm_eps
            )
        else:
            normed = ada_rms_norm_no_gate_ttnn(
                expert_out,
                adarms_cond,
                self.denoise_head.mod_weight,
                self.denoise_head.mod_bias,
                self.config.expert_config.rms_norm_eps,
                self.denoise_head.core_grid,
            )
        return self.suffix.project_output(normed)

    # ──────────── Upstream-compat attention artifacts (unified builder) ──

    def _build_upstream_artifacts(self, img_masks=None, lang_masks=None):
        """Build all 6 attention artifacts based on optional masks.

        Args:
          img_masks: list of torch bool tensors (one per camera, each shape (1,)).
                     None → all real (default; matches perf test).
          lang_masks: torch bool tensor shape (1, 256). None → all real.

        SIDE EFFECT: deallocates current artifacts (if any) and rebuilds them.
        Sets self._suffix_cos, _suffix_sin, _expert_attn_mask, _prefix_cos,
        _prefix_sin, _prefix_attn_mask. When all real + prefix tile-aligned,
        _prefix_attn_mask is None (skip-mask fast path) and _prefix_cos/sin
        are None (prefill uses internal sequential RoPE).

        Mirrors single-chip ttnn_pi0_5_model.py:_build_upstream_attn_artifacts.
        """
        num_cams = self._num_real_cams
        suffix_padded = self._action_horizon_padded
        action_horizon = self.action_horizon
        vlm_head_dim = self.config.vlm_config.head_dim
        expert_head_dim = self.config.expert_config.head_dim
        max_seq_len = self.config.max_seq_len

        # Default masks: all real.
        if img_masks is None:
            img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(num_cams)]
        if lang_masks is None:
            lang_masks = torch.ones(1, 256, dtype=torch.bool)
        assert len(img_masks) == num_cams, f"need {num_cams} img masks, got {len(img_masks)}"

        # 1) Build pad_mask = concat(per-cam-pad-segments, lang_mask[0]).
        pad_segs = []
        for m in img_masks:
            real = bool(m.item()) if m.numel() == 1 else bool(m[0].item())
            pad_segs.append(torch.full((_NUM_PATCHES,), real, dtype=torch.bool))
        pad_segs.append(lang_masks[0].to(torch.bool))
        pad_mask = torch.cat(pad_segs, dim=0)
        prefix_len = pad_mask.shape[0]
        prefix_padded = ((prefix_len + 31) // 32) * 32
        prefix_real_count = int(pad_mask.sum().item())
        all_real_aligned = (prefix_real_count == prefix_len) and (prefix_padded == prefix_len)

        # 2) Prefix attention mask (None when all real + tile-aligned → fast SDPA path).
        if all_real_aligned:
            prefix_mask_4d = None
        else:
            pad_2d = pad_mask[:, None] & pad_mask[None, :]
            prefix_mask = torch.zeros(prefix_padded, prefix_padded, dtype=torch.bfloat16)
            prefix_mask[:prefix_len, :prefix_len].masked_fill_(~pad_2d, _MASK_VAL)
            if prefix_padded > prefix_len:
                prefix_mask[prefix_len:, :] = _MASK_VAL
                prefix_mask[:, prefix_len:] = _MASK_VAL
            prefix_mask_4d = prefix_mask.unsqueeze(0).unsqueeze(0)

        # 3) Prefix RoPE — only override when masking is needed (else prefill
        # stage's internal sequential cos/sin works fine and avoids extra alloc).
        if all_real_aligned:
            prefix_cos_4d = None
            prefix_sin_4d = None
        else:
            position_ids = torch.cumsum(pad_mask.to(torch.int64), dim=0) - 1
            position_ids = position_ids.clamp(min=0, max=max_seq_len - 1)
            cos_vlm, sin_vlm = _precompute_rope_table_torch(vlm_head_dim, max_seq_len)
            prefix_cos = cos_vlm[position_ids]
            prefix_sin = sin_vlm[position_ids]
            if prefix_padded > prefix_len:
                zc = torch.zeros(prefix_padded - prefix_len, vlm_head_dim, dtype=prefix_cos.dtype)
                zs = torch.zeros(prefix_padded - prefix_len, vlm_head_dim, dtype=prefix_sin.dtype)
                prefix_cos = torch.cat([prefix_cos, zc], dim=0)
                prefix_sin = torch.cat([prefix_sin, zs], dim=0)
            prefix_cos_4d = prefix_cos.unsqueeze(0).unsqueeze(0)
            prefix_sin_4d = prefix_sin.unsqueeze(0).unsqueeze(0)

        # 4) Suffix RoPE: offset by REAL prefix count (= prefix_len when all real).
        cos_exp, sin_exp = _precompute_rope_table_torch(expert_head_dim, max_seq_len)
        suffix_positions = (torch.arange(suffix_padded, dtype=torch.int64) + prefix_real_count).clamp(
            max=max_seq_len - 1
        )
        suffix_cos_4d = cos_exp[suffix_positions].unsqueeze(0).unsqueeze(0)
        suffix_sin_4d = sin_exp[suffix_positions].unsqueeze(0).unsqueeze(0)

        # 5) Expert cross-attention mask: block prefix-pad cols + suffix-tail pad.
        kv_total = prefix_padded + suffix_padded
        em = torch.zeros(suffix_padded, kv_total, dtype=torch.bfloat16)
        pad_blocked = (~pad_mask).nonzero(as_tuple=True)[0]
        if pad_blocked.numel() > 0:
            em[:, pad_blocked] = _MASK_VAL
        if prefix_padded > prefix_len:
            em[:, prefix_len:prefix_padded] = _MASK_VAL
        if suffix_padded > action_horizon:
            em[:, prefix_padded + action_horizon : kv_total] = _MASK_VAL
            em[action_horizon:suffix_padded, :] = _MASK_VAL
        expert_mask_4d = em.unsqueeze(0).unsqueeze(0)

        # Upload helpers.
        _rope_l1 = os.environ.get("PI0_ROPE_TABLES_L1", "").lower() in ("1", "true", "yes", "on")
        rope_mc = ttnn.L1_MEMORY_CONFIG if _rope_l1 else ttnn.DRAM_MEMORY_CONFIG

        def _up(host_t, mc=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(
                host_t.to(torch.bfloat16) if host_t.dtype != torch.bfloat16 else host_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=mc,
            )

        # Deallocate previous artifacts (idempotent on first call when all None).
        for attr in (
            "_suffix_cos",
            "_suffix_sin",
            "_expert_attn_mask",
            "_prefix_cos",
            "_prefix_sin",
            "_prefix_attn_mask",
        ):
            t = getattr(self, attr, None)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, attr, None)

        # Upload new artifacts.
        self._suffix_cos = _up(suffix_cos_4d, rope_mc)
        self._suffix_sin = _up(suffix_sin_4d, rope_mc)
        self._expert_attn_mask = _up(expert_mask_4d, ttnn.DRAM_MEMORY_CONFIG)
        if prefix_mask_4d is not None:
            self._prefix_attn_mask = _up(prefix_mask_4d, ttnn.DRAM_MEMORY_CONFIG)
        if prefix_cos_4d is not None:
            self._prefix_cos = _up(prefix_cos_4d, rope_mc)
            self._prefix_sin = _up(prefix_sin_4d, rope_mc)

        # Record the mask key so callers can skip rebuild when masks unchanged.
        img_present = tuple(bool(m.item()) if m.numel() == 1 else bool(m[0].item()) for m in img_masks)
        lang_real_count = int(lang_masks[0].to(torch.bool).sum().item())
        self._artifact_mask_key = (img_present, lang_real_count)

    def prepare_runtime_masks(self, img_masks, lang_masks):
        """Rebuild upstream artifacts for new runtime masks. Idempotent if
        masks match the last build (no-op). Call BEFORE capture_trace per task.
        """
        img_present = tuple(bool(m.item()) if m.numel() == 1 else bool(m[0].item()) for m in img_masks)
        lang_real_count = int(lang_masks[0].to(torch.bool).sum().item())
        key = (img_present, lang_real_count)
        if key == self._artifact_mask_key:
            return  # masks unchanged — current artifacts are valid
        self._build_upstream_artifacts(img_masks, lang_masks)

    # ──────────── TIER A: host-side modulation precompute ──────────────

    def _precompute_block_and_final_mods(self, weights: Dict[str, dict]) -> None:
        """Compute per-step, per-layer block modulations (sa1, ta, ga, sf1, tf, gf)
        and per-step final-norm modulations (scale+1, shift) on host (torch).
        Mirrors ttnn_pi0_5_model.py:_precompute_bs1_modulations TIER A path.
        Upload each as (1, 1, W) TILE bf16 → DRAM, replicated across the 1×8 mesh.

        These tensors are deterministic in the timestep schedule, so they're
        constant across calls. Using them in the denoise loop bypasses each
        block's per-step mod-Dense matmul (W→6W) and the final norm's mod-Dense
        (W→3W) — the dominant per-step host-driven device work.
        """
        from models.experimental.pi0_5.common.configs import SuffixConfig
        from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding

        cfg = self.config
        W = cfg.expert_config.width
        depth = cfg.expert_config.depth
        num_steps = self.num_denoising_steps
        ae_weights = weights["action_expert"]

        suffix_cfg = SuffixConfig(
            action_dim=cfg.action_dim,
            action_horizon=cfg.action_horizon,
            expert_width=W,
            pi05=True,
        )
        torch_suffix = Pi0_5SuffixEmbedding(suffix_cfg, weights["pi0_projections"])
        timesteps_t = torch.tensor([1.0 - i / num_steps for i in range(num_steps)], dtype=torch.bfloat16)
        adarms_cond_host: List[torch.Tensor] = []
        for i in range(num_steps):
            c = torch_suffix.embed_timestep_adarms(timesteps_t[i : i + 1]).to(torch.bfloat16)
            adarms_cond_host.append(c)

        per_layer_fused: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
        for layer_idx in range(depth):
            prefix = f"model.layers.{layer_idx}."
            w_pre_attn = ae_weights[f"{prefix}input_layernorm.dense.weight"]
            w_pre_ffw = ae_weights[f"{prefix}post_attention_layernorm.dense.weight"]
            fused_w = torch.cat([w_pre_attn, w_pre_ffw], dim=0).contiguous().to(torch.bfloat16)
            b_attn_key = f"{prefix}input_layernorm.dense.bias"
            b_ffw_key = f"{prefix}post_attention_layernorm.dense.bias"
            if b_attn_key in ae_weights:
                fused_b = (
                    torch.cat([ae_weights[b_attn_key], ae_weights[b_ffw_key]], dim=0).contiguous().to(torch.bfloat16)
                )
            else:
                fused_b = None
            per_layer_fused.append((fused_w, fused_b))

        def _host_pad_tile_upload(t: torch.Tensor) -> "ttnn.Tensor":
            t3d = t.unsqueeze(1).contiguous()  # (1, 1, W) matches _split_modulation_6
            return ttnn.from_torch(
                t3d,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        for step_idx in range(num_steps):
            cond = adarms_cond_host[step_idx]
            per_layer_step: List[Tuple["ttnn.Tensor", ...]] = []
            for layer_idx in range(depth):
                fused_w, fused_b = per_layer_fused[layer_idx]
                mod = torch.nn.functional.linear(cond, fused_w, fused_b)  # (1, 6W)
                sa = mod[:, 0 * W : 1 * W]
                ta = mod[:, 1 * W : 2 * W]
                ga = mod[:, 2 * W : 3 * W]
                sf = mod[:, 3 * W : 4 * W]
                tf = mod[:, 4 * W : 5 * W]
                gf = mod[:, 5 * W : 6 * W]
                sa1 = sa + 1.0
                sf1 = sf + 1.0
                per_layer_step.append(
                    (
                        _host_pad_tile_upload(sa1),
                        _host_pad_tile_upload(ta),
                        _host_pad_tile_upload(ga),
                        _host_pad_tile_upload(sf1),
                        _host_pad_tile_upload(tf),
                        _host_pad_tile_upload(gf),
                    )
                )
            self._block_mods_per_step.append(per_layer_step)

        final_w = ae_weights["model.norm.dense.weight"].contiguous().to(torch.bfloat16)
        final_b_t = ae_weights.get("model.norm.dense.bias")
        if final_b_t is not None:
            final_b_t = final_b_t.contiguous().to(torch.bfloat16)
        for step_idx in range(num_steps):
            cond = adarms_cond_host[step_idx]
            mod = torch.nn.functional.linear(cond, final_w, final_b_t)  # (1, 3W)
            scale = mod[:, 0 * W : 1 * W]
            shift = mod[:, 1 * W : 2 * W]
            scale1 = scale + 1.0
            self._final_mods_per_step.append(
                (
                    _host_pad_tile_upload(scale1),
                    _host_pad_tile_upload(shift),
                )
            )

    # ──────────── Persistent input buffers ──────────────────────────────

    def _ensure_persistent_input_buffers(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
    ) -> None:
        """Lazily allocate pixel_values_buf + lang_tokens_buf with the right
        shapes on the parent mesh. On subsequent calls, refresh contents
        in-place via copy_host_to_device_tensor (no reallocation → buffer
        IDs stable for trace replay)."""
        # Pixel values: host pre-permute + pre-reshape, batch-sharded (one
        # camera per chip — 3 real + 5 zero dummies).
        pixel_host = self._stack_and_fold_pixels(images)
        if self.pixel_values_buf is None:
            self.pixel_values_buf = ttnn.from_torch(
                pixel_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            host_t = ttnn.from_torch(
                pixel_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
            )
            ttnn.copy_host_to_device_tensor(host_t, self.pixel_values_buf)

        # Lang tokens: uint32 ROW_MAJOR replicated on the mesh.
        if self.lang_tokens_buf is None:
            self.lang_tokens_buf = ttnn.from_torch(
                lang_tokens.to(torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
            )
        else:
            host_t = ttnn.from_torch(
                lang_tokens.to(torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(host_t, self.lang_tokens_buf)

    def _build_denoise_schedule(self) -> None:
        """(Re)build the per-step denoising schedule tensors from
        self.num_denoising_steps. Called from __init__ and from
        set_num_denoising_steps() when the step count changes at runtime
        (e.g. LIBERO rollout sweeping N=10 → N=5)."""
        self._timesteps = [1.0 - i / self.num_denoising_steps for i in range(self.num_denoising_steps + 1)]
        self._dts = [self._timesteps[i + 1] - self._timesteps[i] for i in range(self.num_denoising_steps)]

        # Pre-compute adarms_cond per step on the mesh (replicated). Since the
        # timestep schedule is deterministic, these are constant across calls —
        # building once at init keeps the per-step loop body matmul-only
        # (no ttnn.from_torch inside the trace body).
        self._adarms_per_step: List["ttnn.Tensor"] = []
        for i in range(self.num_denoising_steps):
            t_ttnn = ttnn.from_torch(
                torch.tensor([self._timesteps[i]], dtype=torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
            )
            cond = self.suffix.embed_adarms_cond(t_ttnn)
            ttnn.deallocate(t_ttnn)
            self._adarms_per_step.append(cond)

        # TIER A precompute: per-step, per-layer block-mod tuples (sa1, ta, ga,
        # sf1, tf, gf) + per-step final-norm tuples (scale1, shift). Replicates
        # the single-chip ttnn_pi0_5_model.py:163 `_precompute_bs1_modulations`
        # path. All 6W and 3W mod-Dense matmuls happen on host once at init;
        # per-step device matmuls (18 layers × num_steps + num_steps final)
        # are eliminated at every inference. Saves ~5 ms / inference on this
        # pipeline (denoise 24.6 ms → ~20 ms target).
        self._block_mods_per_step = []
        self._final_mods_per_step = []
        self._precompute_block_and_final_mods(self._weights)

    def set_num_denoising_steps(self, num_steps: int) -> None:
        """Public setter: update num_denoising_steps AND rebuild the schedule
        tensors. Required when the rollout changes the step count at runtime
        (e.g. LIBERO sweeping N=10 → N=5). Bare assignment to
        `num_denoising_steps` without rebuilding leaves the loop indexing
        entries from the OLD schedule (e.g. first 5 entries of a 10-step
        schedule → dt=-0.1 instead of the correct -0.2 for N=5), tanking
        LIBERO accuracy on N=5.

        Also invalidates the trace cache (the trace recorded N_old denoise
        steps with N_old block_mods tensors — those tensor IDs no longer
        match the new schedule)."""
        if num_steps == self.num_denoising_steps:
            return
        self.num_denoising_steps = num_steps
        # Trace becomes stale — op count + precomputed tensor IDs changed.
        if self._trace_id is not None:
            ttnn.release_trace(self.mesh, self._trace_id)
            self._trace_id = None
        self._build_denoise_schedule()

    def _refresh_noise_buffer(self) -> None:
        """Refill self.x_t_fp32 in-place with fresh noise. Logical [:ah] gets
        N(0, 1); the rest is zero-padded to action_horizon_padded."""
        ah = self.action_horizon
        ah_padded = self._action_horizon_padded
        noise_pad = torch.zeros(1, ah_padded, self.action_dim, dtype=torch.float32)
        noise_pad[:, :ah, :] = torch.randn(1, ah, self.action_dim)
        host_t = ttnn.from_torch(noise_pad, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_t, self.x_t_fp32)

    # ──────────── Pure-device body (for trace capture + replay) ─────────

    def _sample_actions_device(self) -> "ttnn.Tensor":
        """Pure-device sample_actions body. Reads from persistent input
        buffers, writes the final actions into self.x_t_fp32 in place. No
        host I/O — every op records cleanly into a trace.
        """
        # ---- Stage 0: SigLIP DP ----
        vision_real = self._run_vision_dp(self.pixel_values_buf)

        # ---- Stage 0b: prefix construction ----
        prefix_embs = self._build_prefix(vision_real, self.lang_tokens_buf)
        ttnn.deallocate(vision_real)

        # ---- Stage 1: prefill TP=8 ----
        # When LIBERO single-arm masking is on, pass prefix_attn_mask + position-
        # aware RoPE override (cos/sin replicated across all 18 layers, since the
        # prefill stage's loop indexes per_chip_cos[i] per LAYER, not per chip).
        if self._prefix_cos is not None:
            num_layers = len(self.prefill.blocks)
            _final_hidden, per_layer_kv = self.prefill.run(
                prefix_embs,
                attention_mask=self._prefix_attn_mask,
                per_chip_cos=[self._prefix_cos] * num_layers,
                per_chip_sin=[self._prefix_sin] * num_layers,
            )
        else:
            _final_hidden, per_layer_kv = self.prefill.run(prefix_embs, attention_mask=self._prefix_attn_mask)
        ttnn.deallocate(prefix_embs)

        # ---- Stage 2: 5-step Euler denoise (replicated on all 8 chips) ----
        for i in range(self.num_denoising_steps):
            dt = self._dts[i]
            adarms_cond = self._adarms_per_step[i]
            block_mods = self._block_mods_per_step[i]
            final_mod = self._final_mods_per_step[i]

            # x_t fp32 → bf16 for the suffix embed_actions matmul.
            x_t_bf16 = ttnn.typecast(self.x_t_fp32, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            suffix_hidden = self.suffix.embed_actions(x_t_bf16)
            ttnn.deallocate(x_t_bf16)

            # Single chunk runs all 18 expert layers. KV is already replicated
            # on all 8 chips from the prefill stage, so the block reads it
            # directly — no migration needed.
            expert_out = self.denoise.forward(
                suffix_hidden,
                adarms_cond,
                per_layer_kv,
                attention_mask=self._expert_attn_mask,
                cos_override=self._suffix_cos,
                sin_override=self._suffix_sin,
                precomputed_mods=block_mods,
            )
            ttnn.deallocate(suffix_hidden)

            velocity_bf16 = self._apply_final_norm_and_project(expert_out, adarms_cond, precomputed_final_mod=final_mod)
            ttnn.deallocate(expert_out)

            # On-device Euler: x_t_fp32 ← x_t_fp32 + dt · velocity_fp32, IN-PLACE
            # via output_tensor=self.x_t_fp32. Keeps the buffer's tensor ID stable.
            velocity_fp32 = ttnn.typecast(velocity_bf16, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(velocity_bf16)
            velocity_scaled = ttnn.mul(velocity_fp32, dt, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(velocity_fp32)
            ttnn.add(self.x_t_fp32, velocity_scaled, output_tensor=self.x_t_fp32)
            ttnn.deallocate(velocity_scaled)

        # Deallocate the KV cache after the loop — not needed for next call
        # (prefill rebuilds it).
        for k, v in per_layer_kv:
            ttnn.deallocate(k)
            ttnn.deallocate(v)

        return self.x_t_fp32

    # ──────────── Per-stage timed eager (instrumentation) ──────────────

    def _sample_actions_device_timed(self) -> Tuple["ttnn.Tensor", Dict[str, float]]:
        """Same body as _sample_actions_device but with synchronize_device +
        perf_counter brackets between stages. Returns (x_t, timings_ms).

        Timings dict keys: vision_ms, prefix_ms, prefill_ms, denoise_ms,
        denoise_step_ms (list), compute_total_ms (sum of the above).
        Wall-clock — eager dispatch, NOT trace-replay timings. Use the
        proportions as a guide; absolute numbers will be larger than the
        traced total because of per-op host dispatch.
        """
        mesh = self.mesh
        t: Dict[str, float] = {"denoise_step_ms": []}

        # ---- Stage 0: SigLIP DP ----
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        vision_real = self._run_vision_dp(self.pixel_values_buf)
        ttnn.synchronize_device(mesh)
        t["vision_ms"] = (time.perf_counter() - t0) * 1000.0

        # ---- Stage 0b: prefix construction ----
        t0 = time.perf_counter()
        prefix_embs = self._build_prefix(vision_real, self.lang_tokens_buf)
        ttnn.deallocate(vision_real)
        ttnn.synchronize_device(mesh)
        t["prefix_ms"] = (time.perf_counter() - t0) * 1000.0

        # ---- Stage 1: prefill TP=8 ----
        t0 = time.perf_counter()
        _final_hidden, per_layer_kv = self.prefill.run(prefix_embs)
        ttnn.deallocate(prefix_embs)
        ttnn.synchronize_device(mesh)
        t["prefill_ms"] = (time.perf_counter() - t0) * 1000.0

        # ---- Stage 2: 5-step Euler denoise ----
        t_denoise_start = time.perf_counter()
        for i in range(self.num_denoising_steps):
            dt = self._dts[i]
            adarms_cond = self._adarms_per_step[i]
            block_mods = self._block_mods_per_step[i]
            final_mod = self._final_mods_per_step[i]

            t_step = time.perf_counter()
            x_t_bf16 = ttnn.typecast(self.x_t_fp32, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            suffix_hidden = self.suffix.embed_actions(x_t_bf16)
            ttnn.deallocate(x_t_bf16)
            expert_out = self.denoise.forward(
                suffix_hidden,
                adarms_cond,
                per_layer_kv,
                attention_mask=self._expert_attn_mask,
                cos_override=self._suffix_cos,
                sin_override=self._suffix_sin,
                precomputed_mods=block_mods,
            )
            ttnn.deallocate(suffix_hidden)
            velocity_bf16 = self._apply_final_norm_and_project(expert_out, adarms_cond, precomputed_final_mod=final_mod)
            ttnn.deallocate(expert_out)
            velocity_fp32 = ttnn.typecast(velocity_bf16, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(velocity_bf16)
            velocity_scaled = ttnn.mul(velocity_fp32, dt, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(velocity_fp32)
            ttnn.add(self.x_t_fp32, velocity_scaled, output_tensor=self.x_t_fp32)
            ttnn.deallocate(velocity_scaled)
            ttnn.synchronize_device(mesh)
            t["denoise_step_ms"].append((time.perf_counter() - t_step) * 1000.0)

        t["denoise_ms"] = (time.perf_counter() - t_denoise_start) * 1000.0

        for k, v in per_layer_kv:
            ttnn.deallocate(k)
            ttnn.deallocate(v)

        t["compute_total_ms"] = t["vision_ms"] + t["prefix_ms"] + t["prefill_ms"] + t["denoise_ms"]
        return self.x_t_fp32, t

    def sample_actions_timed(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Eager sample_actions with full timing breakdown.

        Times: input_upload_ms (pixel+lang+noise host→device),
               compute stages (via _sample_actions_device_timed),
               output_readback_ms (ttnn.to_torch),
               eager_total_ms (sum of all).
        """
        mesh = self.mesh
        timings: Dict[str, float] = {}

        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        self._ensure_persistent_input_buffers(images, lang_tokens)
        self._refresh_noise_buffer()
        ttnn.synchronize_device(mesh)
        timings["input_upload_ms"] = (time.perf_counter() - t0) * 1000.0

        out, stage_t = self._sample_actions_device_timed()
        timings.update(stage_t)

        t0 = time.perf_counter()
        x_t_final = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        timings["output_readback_ms"] = (time.perf_counter() - t0) * 1000.0

        ah = self.action_horizon
        actions = x_t_final[:1, :ah, :]

        timings["eager_total_ms"] = (
            timings["input_upload_ms"] + timings["compute_total_ms"] + timings["output_readback_ms"]
        )
        return actions, timings

    # ──────────── Trace capture + replay ────────────────────────────────

    def capture_trace(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        img_masks: Optional[List[torch.Tensor]] = None,
        lang_masks: Optional[torch.Tensor] = None,
    ) -> None:
        """One-time setup: optionally rebuild attention artifacts for the
        runtime masks, allocate persistent buffers, JIT-compile every kernel
        via an eager warmup, then capture the full _sample_actions_device body
        as a TTNN trace on the parent mesh's CQ 0.

        If img_masks/lang_masks are provided, the upstream attention artifacts
        (prefix mask, position-aware RoPE, suffix RoPE offset, expert mask)
        are rebuilt for these masks BEFORE warmup + capture. The new trace
        records reads from the freshly-allocated artifact tensors, so the
        old trace (if any) is released and must be re-captured. Default
        masks=None preserves backward-compat with the perf test which uses
        random inputs assumed all-real.
        """
        # 0. If masks provided and changed, release any prior trace and
        #    rebuild attention artifacts.
        if img_masks is not None and lang_masks is not None:
            img_present = tuple(bool(m.item()) if m.numel() == 1 else bool(m[0].item()) for m in img_masks)
            lang_real_count = int(lang_masks[0].to(torch.bool).sum().item())
            new_key = (img_present, lang_real_count)
            if new_key != self._artifact_mask_key:
                if self._trace_id is not None:
                    ttnn.release_trace(self.mesh, self._trace_id)
                    self._trace_id = None
                self._build_upstream_artifacts(img_masks, lang_masks)

        # 1. Stage the persistent inputs.
        self._ensure_persistent_input_buffers(images, lang_tokens)
        self._refresh_noise_buffer()

        # 2. Eager warmup — JIT-compile every op kernel. The trace allocator
        #    can't tolerate kernel JIT or lazy memcfg/pcfg cache builds inside
        #    the trace body, so we burn one full forward eagerly to populate
        #    all those caches.
        _ = self._sample_actions_device()

        # 3. Refresh noise (warmup mutated x_t_fp32) and capture the trace.
        self._refresh_noise_buffer()
        self._trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._captured_actions = self._sample_actions_device()
        ttnn.end_trace_capture(self.mesh, self._trace_id, cq_id=0)

    def sample_actions_traced(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Replay the captured trace on new inputs.

        Refreshes pixel_values_buf / lang_tokens_buf / x_t_fp32 in place via
        copy_host_to_device_tensor, runs ttnn.execute_trace, then reads chip 0's
        copy of the actions tensor and slices to the logical action_horizon.
        """
        if self._trace_id is None:
            raise RuntimeError("capture_trace() must be called before sample_actions_traced()")

        self._ensure_persistent_input_buffers(images, lang_tokens)
        self._refresh_noise_buffer()

        ttnn.execute_trace(self.mesh, self._trace_id, cq_id=0, blocking=True)

        # Output: the replicated x_t_fp32 lives on all 8 chips. ConcatMeshToTensor
        # along dim=0 stacks the 8 identical chip copies into a (8, ah_padded,
        # action_dim) host tensor; slice [:1] keeps one logical sample.
        x_t_final = ttnn.to_torch(
            self._captured_actions,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0),
        )
        ah = self.action_horizon
        return x_t_final[:1, :ah, :]

    # ──────────── 3-sub-trace capture (per-stage traced timing) ────────

    def _denoise_loop_device(self, per_layer_kv) -> None:
        """5-step Euler denoise loop body. Reads/writes self.x_t_fp32 in-place;
        per_layer_kv is the list of (k, v) tuples from the prefill stage.
        Pulled out of _sample_actions_device so the staged-denoise trace can
        reuse the exact same op sequence."""
        for i in range(self.num_denoising_steps):
            dt = self._dts[i]
            adarms_cond = self._adarms_per_step[i]
            block_mods = self._block_mods_per_step[i]
            final_mod = self._final_mods_per_step[i]
            x_t_bf16 = ttnn.typecast(self.x_t_fp32, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            suffix_hidden = self.suffix.embed_actions(x_t_bf16)
            ttnn.deallocate(x_t_bf16)
            expert_out = self.denoise.forward(
                suffix_hidden,
                adarms_cond,
                per_layer_kv,
                attention_mask=self._expert_attn_mask,
                cos_override=self._suffix_cos,
                sin_override=self._suffix_sin,
                precomputed_mods=block_mods,
            )
            ttnn.deallocate(suffix_hidden)
            velocity_bf16 = self._apply_final_norm_and_project(expert_out, adarms_cond, precomputed_final_mod=final_mod)
            ttnn.deallocate(expert_out)
            velocity_fp32 = ttnn.typecast(velocity_bf16, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(velocity_bf16)
            velocity_scaled = ttnn.mul(velocity_fp32, dt, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(velocity_fp32)
            ttnn.add(self.x_t_fp32, velocity_scaled, output_tensor=self.x_t_fp32)
            ttnn.deallocate(velocity_scaled)

    def capture_traces_staged(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
    ) -> None:
        """Capture three independent traces on the same 1×8 mesh:
            vision_trace   : pixel_values_buf  → vision_real (persistent)
            prefill_trace  : vision_real + lang → kv (persistent), then dealloc vision_real
            denoise_trace  : kv → x_t_fp32 (5-step Euler), then dealloc kv

        Persistent intermediates (vision_real after vision_trace, kv after
        prefill_trace) survive across trace boundaries because the trace
        allocator gives deterministic addresses on replay. Each trace cleans
        up its own input at the end of its body, so an iteration ends with no
        leftover stage allocations.

        Use sample_actions_traced_staged_timed() to replay + time each stage.
        """
        # 1. Persistent input buffers + initial noise.
        self._ensure_persistent_input_buffers(images, lang_tokens)
        self._refresh_noise_buffer()

        # 2. Eager warmup — JIT compile every kernel before any trace capture.
        _ = self._sample_actions_device()
        self._refresh_noise_buffer()

        # 3. Vision sub-trace.
        self._vision_trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        vision_real = self._run_vision_dp(self.pixel_values_buf)
        ttnn.end_trace_capture(self.mesh, self._vision_trace_id, cq_id=0)

        # 4. Prefill sub-trace (reads persistent vision_real, produces persistent kv,
        #    cleans up vision_real at the end so the next iter's vision_trace can re-allocate).
        self._prefill_trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        prefix_embs = self._build_prefix(vision_real, self.lang_tokens_buf)
        _final_hidden, per_layer_kv = self.prefill.run(prefix_embs)
        ttnn.deallocate(prefix_embs)
        ttnn.deallocate(vision_real)
        ttnn.end_trace_capture(self.mesh, self._prefill_trace_id, cq_id=0)

        # 5. Denoise sub-trace (reads persistent kv, runs 5-step Euler, cleans up kv).
        self._denoise_trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._denoise_loop_device(per_layer_kv)
        for k, v in per_layer_kv:
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        ttnn.end_trace_capture(self.mesh, self._denoise_trace_id, cq_id=0)

        self._staged_actions = self.x_t_fp32

    def sample_actions_traced_staged_timed(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Replay the 3 sub-traces sequentially with per-stage timing.

        Splits: input_upload | vision | prefill | denoise | output_readback.
        Each stage's time is wall-clock around ttnn.execute_trace with
        blocking=True — pure trace dispatch on device, no eager dispatch.
        """
        if self._vision_trace_id is None:
            raise RuntimeError("capture_traces_staged() must be called first")
        mesh = self.mesh
        timings: Dict[str, float] = {}

        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        self._ensure_persistent_input_buffers(images, lang_tokens)
        self._refresh_noise_buffer()
        ttnn.synchronize_device(mesh)
        timings["input_upload_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, self._vision_trace_id, cq_id=0, blocking=True)
        timings["vision_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, self._prefill_trace_id, cq_id=0, blocking=True)
        timings["prefill_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, self._denoise_trace_id, cq_id=0, blocking=True)
        timings["denoise_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        x_t_final = ttnn.to_torch(
            self._staged_actions,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )
        timings["output_readback_ms"] = (time.perf_counter() - t0) * 1000.0

        ah = self.action_horizon
        actions = x_t_final[:1, :ah, :]

        timings["compute_total_ms"] = timings["vision_ms"] + timings["prefill_ms"] + timings["denoise_ms"]
        timings["traced_total_ms"] = (
            timings["input_upload_ms"] + timings["compute_total_ms"] + timings["output_readback_ms"]
        )
        return actions, timings

    # ──────────── 2CQ trace replay (host-to-device on CQ1) ────────────

    def sample_actions_traced_2cq_loop(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        iters: int,
    ) -> Tuple[torch.Tensor, List[float]]:
        """Run `iters` iterations of trace replay with H2D input upload on CQ1
        overlapped with compute on CQ0. Exercised by the LIBERO ttnn_1x8 backend
        (libero_sim/libero_rollout.py).

        Requires the mesh to have been opened with num_command_queues=2.
        Returns (last_actions, per_iter_wall_clock_ms).

        Per-iter loop:
          1. wait_for_event(0, write_event) — CQ0 waits for CQ1's pre-stage
          2. execute_trace on CQ0 (non-blocking)
          3. record_event on CQ0 → op_event
          4. for next iter: wait_for_event(1, op_event); copy next inputs on
             CQ1; record write_event — CQ1 stages next chunk overlapped with
             CQ0 compute.
          5. synchronize_device + to_torch readback
        """
        if self._trace_id is None:
            raise RuntimeError("capture_trace() must be called first")
        if self.pixel_values_buf is None or self.lang_tokens_buf is None:
            raise RuntimeError("input buffers must be allocated via capture_trace first")
        mesh = self.mesh
        ah = self.action_horizon
        ah_padded = self._action_horizon_padded

        # Pre-stage host-side input tensors for iters+1 chunks (the +1 is the
        # initial pre-stage; the final chunk after the last execute doesn't
        # get re-uploaded). All allocations done up-front so no host overhead
        # leaks into the timed loop.
        host_chunks: List[Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]] = []
        for _ in range(iters + 1):
            pixel_host = self._stack_and_fold_pixels(images)
            h_pix = ttnn.from_torch(
                pixel_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            h_lang = ttnn.from_torch(
                lang_tokens.to(torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            noise_pad = torch.zeros(1, ah_padded, self.action_dim, dtype=torch.float32)
            noise_pad[:, :ah, :] = torch.randn(1, ah, self.action_dim)
            h_noise = ttnn.from_torch(noise_pad, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            host_chunks.append((h_pix, h_lang, h_noise))

        # Pre-stage chunk 0 inputs on CQ1.
        h0_pix, h0_lang, h0_noise = host_chunks[0]
        ttnn.copy_host_to_device_tensor(h0_pix, self.pixel_values_buf, cq_id=1)
        ttnn.copy_host_to_device_tensor(h0_lang, self.lang_tokens_buf, cq_id=1)
        ttnn.copy_host_to_device_tensor(h0_noise, self.x_t_fp32, cq_id=1)
        write_event = ttnn.record_event(mesh, 1)

        times_ms: List[float] = []
        last_actions = None
        for i in range(iters):
            t0 = time.perf_counter()
            # CQ0 waits until CQ1 finished staging this iter's inputs.
            ttnn.wait_for_event(0, write_event)
            ttnn.execute_trace(mesh, self._trace_id, cq_id=0, blocking=False)
            op_event = ttnn.record_event(mesh, 0)

            # Stage next iter's inputs on CQ1, overlapped with CQ0 compute.
            # CQ1 waits for op_event so it doesn't overwrite the input buffers
            # before the trace has consumed them.
            if i + 1 < iters:
                hn_pix, hn_lang, hn_noise = host_chunks[i + 1]
                ttnn.wait_for_event(1, op_event)
                ttnn.copy_host_to_device_tensor(hn_pix, self.pixel_values_buf, cq_id=1)
                ttnn.copy_host_to_device_tensor(hn_lang, self.lang_tokens_buf, cq_id=1)
                ttnn.copy_host_to_device_tensor(hn_noise, self.x_t_fp32, cq_id=1)
                write_event = ttnn.record_event(mesh, 1)

            # to_torch is itself blocking on _captured_actions, which serializes
            # implicitly after the trace. The explicit synchronize_device was
            # redundant and added host overhead.
            last_actions = ttnn.to_torch(
                self._captured_actions,
                mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
            )
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        if last_actions is None:
            raise RuntimeError("iters must be >= 1")
        return last_actions[:1, :ah, :], times_ms

    def sample_actions_traced_1cq_prestaged_loop(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        iters: int,
    ) -> Tuple[torch.Tensor, List[float]]:
        """Single-CQ trace replay with host_chunks PRE-STAGED before the timed
        loop (mirrors the 2CQ test's host-side amortization).

        Use this to isolate the actual H2D-DMA cost on CQ0 vs the host-prep
        cost. The standard `sample_actions_traced_timed` includes both per iter
        (host prep + DMA on CQ0 + compute + D2H). This variant pre-stages all
        host work outside the timed window — the per-iter window then contains:
            DMA dispatch + actual PCIe transfer (CQ0, serial before compute) +
            trace_exec + D2H readback.

        Compare to sample_actions_traced_2cq_loop:
            Both pre-stage host work outside the timed loop.
            1CQ: DMA serialized BEFORE compute on CQ0.
            2CQ: DMA on CQ1 in PARALLEL with compute on CQ0.
            Difference = how much of the actual PCIe DMA hides behind compute.
        """
        if self._trace_id is None:
            raise RuntimeError("capture_trace() must be called first")
        if self.pixel_values_buf is None or self.lang_tokens_buf is None:
            raise RuntimeError("input buffers must be allocated via capture_trace first")
        mesh = self.mesh
        ah = self.action_horizon
        ah_padded = self._action_horizon_padded

        # === Pre-stage host_chunks OUTSIDE the timed loop (same as 2CQ) ===
        host_chunks: List[Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]] = []
        for _ in range(iters):
            pixel_host = self._stack_and_fold_pixels(images)
            h_pix = ttnn.from_torch(
                pixel_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            h_lang = ttnn.from_torch(
                lang_tokens.to(torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            noise_pad = torch.zeros(1, ah_padded, self.action_dim, dtype=torch.float32)
            noise_pad[:, :ah, :] = torch.randn(1, ah, self.action_dim)
            h_noise = ttnn.from_torch(noise_pad, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            host_chunks.append((h_pix, h_lang, h_noise))

        times_ms: List[float] = []
        last_actions = None
        for i in range(iters):
            t0 = time.perf_counter()
            # DMA this iter's pre-prepped inputs to device on CQ0 (serial).
            hi_pix, hi_lang, hi_noise = host_chunks[i]
            ttnn.copy_host_to_device_tensor(hi_pix, self.pixel_values_buf)
            ttnn.copy_host_to_device_tensor(hi_lang, self.lang_tokens_buf)
            ttnn.copy_host_to_device_tensor(hi_noise, self.x_t_fp32)
            # Compute (still single-CQ — serial after DMAs above).
            ttnn.execute_trace(mesh, self._trace_id, cq_id=0, blocking=True)
            # D2H readback.
            last_actions = ttnn.to_torch(
                self._captured_actions,
                mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
            )
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        if last_actions is None:
            raise RuntimeError("iters must be >= 1")
        return last_actions[:1, :ah, :], times_ms

    def sample_actions_traced_timed(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Trace-replay with host-side overhead breakdown.

        Splits the replay loop into:
            input_upload_ms : pixel + lang + noise refresh via copy_host_to_device
            trace_exec_ms   : ttnn.execute_trace (compute on device, blocking)
            output_readback_ms : ttnn.to_torch (single concat)
            traced_total_ms : sum of the above

        Useful for separating "device compute" from "host glue" cost.
        """
        if self._trace_id is None:
            raise RuntimeError("capture_trace() must be called before sample_actions_traced_timed()")
        mesh = self.mesh
        timings: Dict[str, float] = {}

        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        self._ensure_persistent_input_buffers(images, lang_tokens)
        self._refresh_noise_buffer()
        ttnn.synchronize_device(mesh)
        timings["input_upload_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, self._trace_id, cq_id=0, blocking=True)
        timings["trace_exec_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        x_t_final = ttnn.to_torch(
            self._captured_actions,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )
        timings["output_readback_ms"] = (time.perf_counter() - t0) * 1000.0

        ah = self.action_horizon
        actions = x_t_final[:1, :ah, :]

        timings["traced_total_ms"] = (
            timings["input_upload_ms"] + timings["trace_exec_ms"] + timings["output_readback_ms"]
        )
        return actions, timings

    # ──────────── Eager public entry (debug / one-shot use) ─────────────

    def sample_actions(
        self,
        images: List[torch.Tensor],
        img_masks: Optional[List[torch.Tensor]] = None,  # unused in v1
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,  # unused in v1
        state: Optional[torch.Tensor] = None,  # unused in v1
    ) -> torch.Tensor:
        """Eager end-to-end sample_actions. Useful for first-call debugging
        and PCC verification before capture_trace() is wired up.

        Returns (1, action_horizon, action_dim) torch.fp32. The trace path
        (capture_trace + sample_actions_traced) is the production replay.
        """
        if lang_tokens is None:
            raise ValueError("lang_tokens required")

        self._ensure_persistent_input_buffers(images, lang_tokens)
        self._refresh_noise_buffer()
        out = self._sample_actions_device()
        x_t_final = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        ah = self.action_horizon
        return x_t_final[:1, :ah, :]
