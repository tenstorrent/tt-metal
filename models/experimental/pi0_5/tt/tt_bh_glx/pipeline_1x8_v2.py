# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pi0_5GLX1x8PipelineV2 — pi0.5 sample_actions on a 1×8 mesh with denoise
on a 1×1 submesh (eager, host-bounce KV, no trace).

Goal: prove the 1×1-submesh-for-denoise fix at N=5 LIBERO. Pure correctness
v1; perf optimizations (trace, sockets) deferred to v2.

Architecture difference vs the v1 pipeline_1x8.py:
- Vision DP + Prefill TP=8 stay on the full 1×8 parent mesh (unchanged).
- Denoise expert runs on a 1×1 submesh of the parent (chip 0 only).
- KV cache is host-bounced (extract chip 0's parent shard → fp32 host → bf8_b
  on the submesh) since cross-mesh op inputs are rejected by TTNN.
- bf8_b round-trip is bit-identical (verified by /home/tt-admin/.claude/jobs/
  dbeca188/tmp/probe_cross_mesh_shard.py), so handoff has zero quantization loss.

This v1 is eager (no trace). It will be slower per chunk (~30-100 ms host
bounce on top of compute), but proves the numerical fix. Add traces only after
LIBERO N=5 passes.
"""
from __future__ import annotations

import os
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
from .pipeline import _DenoiseHead, _PrefillHead
from .stage_prefill_tp4 import StagePrefillTP4
from .suffix_slice import SuffixSlice
from .vision_slice import SigLIPCameraSlice


_NUM_CHIPS_REQUIRED = 8
_DEFAULT_NUM_REAL_CAMS = 3
_NUM_PATCHES = 256


class Pi0_5GLX1x8PipelineV2:
    """1×8 parent mesh for vision + prefill, 1×1 submesh for denoise.

    Eager (no trace) implementation. Host-bounce KV between meshes.
    """

    def __init__(self, config: Pi0_5ModelConfig, weights: Dict[str, dict], mesh):
        """LAZY-SUBMESH variant: the 1×1 denoise submesh is created on FIRST
        sample_actions() call, AFTER vision DP completes on the parent. This
        avoids the parent-submesh CCL contention during the parent's all_gather.

        Denoise-side weights/components (SuffixSlice, ExpertChunkSlice,
        _DenoiseHead, TIER A modulations, suffix RoPE, expert mask) are also
        deferred and built lazily after the submesh exists.
        """
        import time as _time

        def _ilog(msg):
            print(f"[{_time.strftime('%T')}] v2.init: {msg}", flush=True)

        _ilog("entering __init__")
        if mesh.get_num_devices() != _NUM_CHIPS_REQUIRED:
            raise RuntimeError(
                f"Pi0_5GLX1x8PipelineV2 requires a {_NUM_CHIPS_REQUIRED}-chip mesh; got {mesh.get_num_devices()}"
            )
        self.mesh = mesh
        self._weights = weights  # stash for lazy denoise-side build
        self.denoise_submesh = None  # lazily created
        self.denoise_chip_index = 0
        self.config = config
        self.num_denoising_steps = config.num_denoising_steps
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self._action_horizon_padded = ((config.action_horizon + 31) // 32) * 32
        self._patch_size = config.siglip_config.patch_size
        self._image_size = config.siglip_config.image_size
        self._vlm_hidden = config.vlm_config.width
        _ilog(f"parent mesh: shape={mesh.shape}, num_devices={mesh.get_num_devices()}")

        # Grid assert moved to _build_denoise_lazy (submesh doesn't exist yet)

        # ── Stage 0: SigLIP DP — on parent ──────────────────────────────
        _ilog("building vision on parent")
        self.vision = SigLIPCameraSlice(
            config.siglip_config,
            weights["vlm_vision"],
            weights["vlm_projector"],
            mesh,
        )

        # ── Stage 1: Prefill TP=8 — on parent ────────────────────────────
        _ilog("building prefill on parent")
        self.prefill = StagePrefillTP4(config, weights, mesh)
        _ilog("building prefill_head on parent")
        self.prefill_head = _PrefillHead(weights["vlm_language"], mesh, self._vlm_hidden)

        # ── Stage 2: Denoise — DEFERRED to first sample_actions ──────────
        self.suffix = None
        self.denoise = None
        self.denoise_head = None

        # Schedule (host-side, no submesh needed)
        self._timesteps = [1.0 - i / self.num_denoising_steps for i in range(self.num_denoising_steps + 1)]
        self._dts = [self._timesteps[i + 1] - self._timesteps[i] for i in range(self.num_denoising_steps)]
        self._adarms_per_step: List["ttnn.Tensor"] = []
        self._block_mods_per_step: List[List[Tuple["ttnn.Tensor", ...]]] = []
        self._final_mods_per_step: List[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = []
        self._denoise_built = False
        _ilog("denoise stage DEFERRED (will build lazily on first sample_actions)")

        # Cam count
        self._num_real_cams = int(os.environ.get("PI0_NUM_CAMERAS", _DEFAULT_NUM_REAL_CAMS))
        if not 1 <= self._num_real_cams <= _NUM_CHIPS_REQUIRED:
            raise RuntimeError(f"PI0_NUM_CAMERAS={self._num_real_cams} out of range [1, {_NUM_CHIPS_REQUIRED}]")

        # Upstream attention artifacts. Prefix-side lives on parent; suffix-side
        # lives on the denoise submesh (since the denoise expert is the consumer).
        self._suffix_cos = None
        self._suffix_sin = None
        self._expert_attn_mask = None
        self._prefix_cos = None
        self._prefix_sin = None
        self._prefix_attn_mask = None
        self._artifact_mask_key = None
        # Upstream artifacts deferred too (suffix side needs the lazy submesh)
        _ilog("__init__ done (denoise + artifacts deferred to first sample_actions)")

    # ──────────────────── Stage helpers ────────────────────────────────

    def _stack_and_fold_pixels(self, images: List[torch.Tensor]) -> torch.Tensor:
        n_real = len(images)
        if n_real != self._num_real_cams:
            raise RuntimeError(f"Pipeline configured for {self._num_real_cams} real cameras; got {n_real}")
        real = torch.cat(images, dim=0)
        if real.shape[0] < _NUM_CHIPS_REQUIRED:
            pad = torch.zeros(
                _NUM_CHIPS_REQUIRED - real.shape[0],
                real.shape[1],
                real.shape[2],
                real.shape[3],
                dtype=real.dtype,
            )
            real = torch.cat([real, pad], dim=0)
        x = real.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        x = x.reshape(B, H, W // self._patch_size, C * self._patch_size).contiguous()
        return x

    def _run_vision_dp(self, pixel_values_ttnn) -> "ttnn.Tensor":
        vision_out = self.vision.forward(pixel_values_ttnn)
        gathered = ttnn.all_gather(
            vision_out,
            dim=0,
            num_links=2,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(vision_out)
        real = ttnn.slice(gathered, [0, 0, 0], [self._num_real_cams, _NUM_PATCHES, self._vlm_hidden])
        ttnn.deallocate(gathered)
        return real

    def _build_prefix(self, vision_real, lang_tokens) -> "ttnn.Tensor":
        n_cams = int(vision_real.shape[0])
        v = ttnn.reshape(vision_real, (1, n_cams * _NUM_PATCHES, self._vlm_hidden))
        lang_emb = self.prefill_head.embed_lang(lang_tokens)
        prefix = ttnn.concat([v, lang_emb], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        if prefix.layout != ttnn.TILE_LAYOUT:
            prefix = ttnn.to_layout(prefix, ttnn.TILE_LAYOUT)
        return prefix

    def _apply_final_norm_and_project(self, expert_out, adarms_cond, precomputed_final_mod=None):
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

    # ──────────── Upstream artifacts (mirrors v1 pipeline_1x8) ─────────

    def _build_upstream_artifacts(self, img_masks=None, lang_masks=None):
        """Build the 6 attention artifacts. Prefix-side on parent, suffix-side on
        denoise submesh. Mirrors pipeline_1x8.py:_build_upstream_artifacts.
        """
        num_cams = self._num_real_cams
        suffix_padded = self._action_horizon_padded
        action_horizon = self.action_horizon
        vlm_head_dim = self.config.vlm_config.head_dim
        expert_head_dim = self.config.expert_config.head_dim
        max_seq_len = self.config.max_seq_len

        if img_masks is None:
            img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(num_cams)]
        if lang_masks is None:
            lang_masks = torch.ones(1, 256, dtype=torch.bool)
        assert len(img_masks) == num_cams

        # 1) pad_mask
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

        # 2) Prefix attn mask (PARENT mesh)
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

        # 3) Prefix RoPE (PARENT mesh)
        if all_real_aligned:
            prefix_cos_4d = None
            prefix_sin_4d = None
        else:
            position_ids = (torch.cumsum(pad_mask.to(torch.int64), dim=0) - 1).clamp(min=0, max=max_seq_len - 1)
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

        # 4) Suffix RoPE (DENOISE SUBMESH)
        cos_exp, sin_exp = _precompute_rope_table_torch(expert_head_dim, max_seq_len)
        suffix_positions = (torch.arange(suffix_padded, dtype=torch.int64) + prefix_real_count).clamp(
            max=max_seq_len - 1
        )
        suffix_cos_4d = cos_exp[suffix_positions].unsqueeze(0).unsqueeze(0)
        suffix_sin_4d = sin_exp[suffix_positions].unsqueeze(0).unsqueeze(0)

        # 5) Expert cross-attn mask (DENOISE SUBMESH)
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

        _rope_l1 = os.environ.get("PI0_ROPE_TABLES_L1", "").lower() in ("1", "true", "yes", "on")
        rope_mc = ttnn.L1_MEMORY_CONFIG if _rope_l1 else ttnn.DRAM_MEMORY_CONFIG

        def _up_parent(host_t, mc=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(
                host_t.to(torch.bfloat16) if host_t.dtype != torch.bfloat16 else host_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=mc,
            )

        def _up_submesh(host_t, mc=ttnn.DRAM_MEMORY_CONFIG):
            if self.denoise_submesh is None:
                return None  # deferred until lazy build
            return ttnn.from_torch(
                host_t.to(torch.bfloat16) if host_t.dtype != torch.bfloat16 else host_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.denoise_submesh,
                memory_config=mc,
            )

        # Deallocate previous artifacts
        for attr, owner in (
            ("_suffix_cos", "submesh"),
            ("_suffix_sin", "submesh"),
            ("_expert_attn_mask", "submesh"),
            ("_prefix_cos", "parent"),
            ("_prefix_sin", "parent"),
            ("_prefix_attn_mask", "parent"),
        ):
            t = getattr(self, attr, None)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, attr, None)

        # Upload — prefix on parent, suffix on submesh
        self._suffix_cos = _up_submesh(suffix_cos_4d, rope_mc)
        self._suffix_sin = _up_submesh(suffix_sin_4d, rope_mc)
        self._expert_attn_mask = _up_submesh(expert_mask_4d, ttnn.DRAM_MEMORY_CONFIG)
        if prefix_mask_4d is not None:
            self._prefix_attn_mask = _up_parent(prefix_mask_4d, ttnn.DRAM_MEMORY_CONFIG)
        if prefix_cos_4d is not None:
            self._prefix_cos = _up_parent(prefix_cos_4d, rope_mc)
            self._prefix_sin = _up_parent(prefix_sin_4d, rope_mc)

        img_present = tuple(bool(m.item()) if m.numel() == 1 else bool(m[0].item()) for m in img_masks)
        lang_real_count = int(lang_masks[0].to(torch.bool).sum().item())
        self._artifact_mask_key = (img_present, lang_real_count)

    def prepare_runtime_masks(self, img_masks, lang_masks):
        img_present = tuple(bool(m.item()) if m.numel() == 1 else bool(m[0].item()) for m in img_masks)
        lang_real_count = int(lang_masks[0].to(torch.bool).sum().item())
        key = (img_present, lang_real_count)
        if key == self._artifact_mask_key:
            return
        self._build_upstream_artifacts(img_masks, lang_masks)

    # ──────────── TIER A precompute (on denoise submesh) ───────────────

    def _precompute_block_and_final_mods(self, weights: Dict[str, dict]) -> None:
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

        def _upload(t: torch.Tensor) -> "ttnn.Tensor":
            t3d = t.unsqueeze(1).contiguous()
            return ttnn.from_torch(
                t3d,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.denoise_submesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        for step_idx in range(num_steps):
            cond = adarms_cond_host[step_idx]
            per_layer_step: List[Tuple["ttnn.Tensor", ...]] = []
            for layer_idx in range(depth):
                fused_w, fused_b = per_layer_fused[layer_idx]
                mod = torch.nn.functional.linear(cond, fused_w, fused_b)
                sa, ta, ga = mod[:, 0:W], mod[:, W : 2 * W], mod[:, 2 * W : 3 * W]
                sf, tf, gf = mod[:, 3 * W : 4 * W], mod[:, 4 * W : 5 * W], mod[:, 5 * W : 6 * W]
                sa1 = sa + 1.0
                sf1 = sf + 1.0
                per_layer_step.append((_upload(sa1), _upload(ta), _upload(ga), _upload(sf1), _upload(tf), _upload(gf)))
            self._block_mods_per_step.append(per_layer_step)

        final_w = ae_weights["model.norm.dense.weight"].contiguous().to(torch.bfloat16)
        final_b_t = ae_weights.get("model.norm.dense.bias")
        if final_b_t is not None:
            final_b_t = final_b_t.contiguous().to(torch.bfloat16)
        for step_idx in range(num_steps):
            cond = adarms_cond_host[step_idx]
            mod = torch.nn.functional.linear(cond, final_w, final_b_t)
            scale, shift = mod[:, 0:W], mod[:, W : 2 * W]
            scale1 = scale + 1.0
            self._final_mods_per_step.append((_upload(scale1), _upload(shift)))

    def _build_denoise_lazy(self):
        """Build all denoise-side components on a 1×1 submesh of the parent.
        Call AFTER first vision DP completes to avoid parent-submesh CCL conflict.
        """
        if self._denoise_built:
            return
        import time as _time

        def _llog(msg):
            print(f"[{_time.strftime('%T')}] v2.lazy: {msg}", flush=True)

        _llog("creating 1×1 submesh on coord (0, 0)")
        self.denoise_submesh = self.mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        self.denoise_chip_index = 0
        parent_grid = self.mesh.compute_with_storage_grid_size()
        sub_grid = self.denoise_submesh.compute_with_storage_grid_size()
        assert (parent_grid.y == sub_grid.y) and (
            parent_grid.x == sub_grid.x
        ), f"Parent grid ≠ submesh grid: {parent_grid.y}×{parent_grid.x} vs {sub_grid.y}×{sub_grid.x}"
        _llog(f"submesh created (grid {sub_grid.y}×{sub_grid.x}); building suffix")
        suffix_cfg = SuffixConfig(
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            expert_width=self.config.expert_config.width,
            pi05=True,
        )
        self.suffix = SuffixSlice(suffix_cfg, self._weights["pi0_projections"], self.denoise_submesh)
        _llog("building denoise ExpertChunkSlice")
        self.denoise = ExpertChunkSlice(
            self.config.expert_config,
            self._weights["action_expert"],
            self.denoise_submesh,
            layer_range=(0, 18),
            max_seq_len=self.config.max_seq_len,
        )
        _llog("building denoise_head")
        self.denoise_head = _DenoiseHead(self._weights["action_expert"], self.denoise_submesh)

        _llog("precomputing adarms_cond per step")
        for i in range(self.num_denoising_steps):
            t_ttnn = ttnn.from_torch(
                torch.tensor([self._timesteps[i]], dtype=torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.denoise_submesh,
            )
            cond = self.suffix.embed_adarms_cond(t_ttnn)
            ttnn.deallocate(t_ttnn)
            self._adarms_per_step.append(cond)

        _llog("TIER A precompute (block + final mods on submesh)")
        self._precompute_block_and_final_mods(self._weights)

        _llog("rebuilding upstream artifacts (suffix-side now lives on submesh)")
        # Force rebuild — _build_upstream_artifacts uploads suffix_cos/sin/expert_attn_mask
        # to self.denoise_submesh, which only just got created.
        self._artifact_mask_key = None
        # Use the cached mask values from the last upstream_artifacts call (in __init__
        # we built with all-real defaults; if prepare_runtime_masks was called, those
        # values are now stale on the parent side but the prefix-side artifacts are
        # rebuilt below from the current pad_mask). Since we don't track them, just
        # rebuild with all-real defaults — sample_actions will call
        # prepare_runtime_masks again before the next chunk's denoise.
        self._build_upstream_artifacts()
        _llog("denoise lazy build done")
        self._denoise_built = True

    # ──────────── KV cache host-bounce (parent → submesh) ───────────────

    def _bounce_kv_to_submesh(self, parent_kv):
        """Extract chip {denoise_chip_index}'s KV from each parent layer,
        host-bounce as bf8_b onto the 1×1 denoise submesh. Returns list of
        (k, v) tuples ready for ExpertChunkSlice.forward.
        """
        ci = self.denoise_chip_index
        kv_sub = []
        for k_p, v_p in parent_kv:
            k_chip_host = ttnn.to_torch(ttnn.get_device_tensors(k_p)[ci]).float()
            v_chip_host = ttnn.to_torch(ttnn.get_device_tensors(v_p)[ci]).float()
            k_sub = ttnn.from_torch(
                k_chip_host,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.denoise_submesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            v_sub = ttnn.from_torch(
                v_chip_host,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.denoise_submesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            kv_sub.append((k_sub, v_sub))
        return kv_sub

    # ──────────── sample_actions (eager, no trace) ──────────────────────

    def sample_actions(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        img_masks: Optional[List[torch.Tensor]] = None,
        lang_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Eager end-to-end pi0.5 inference for one chunk.

        Returns: torch.Tensor of shape (1, action_horizon, action_dim).
        """
        import time as _time

        def _slog(msg):
            print(f"[{_time.strftime('%T')}] v2.sample: {msg}", flush=True)

        _slog("entering sample_actions")
        # Build parent-side artifacts (mask, prefix RoPE) IF masks provided.
        # Suffix-side artifacts will be built after the lazy submesh is created.
        if img_masks is not None and lang_masks is not None:
            if self.denoise_submesh is not None:
                _slog("prepare_runtime_masks (submesh exists)")
                self.prepare_runtime_masks(img_masks, lang_masks)
                _slog("prepare_runtime_masks done")
            else:
                _slog("delayed mask rebuild until lazy denoise build")
                self._pending_img_masks = img_masks
                self._pending_lang_masks = lang_masks

        # ── Upload pixel values (parent, batch-sharded one cam per chip) ─
        _slog("uploading pixel_values to parent")
        pixel_host = self._stack_and_fold_pixels(images)
        pixel_values = ttnn.from_torch(
            pixel_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _slog("pixel_values uploaded")

        # ── Upload lang tokens (parent, replicated) ──────────────────────
        _slog("uploading lang_tokens to parent")
        lang_tokens_tt = ttnn.from_torch(
            lang_tokens.to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _slog("lang_tokens uploaded")

        # ── Stage 0: SigLIP DP on parent ────────────────────────────────
        _slog("vision DP")
        vision_real = self._run_vision_dp(pixel_values)
        ttnn.deallocate(pixel_values)
        _slog("vision DP done")

        # ── Stage 0b: build prefix on parent ────────────────────────────
        _slog("building prefix")
        prefix_embs = self._build_prefix(vision_real, lang_tokens_tt)
        ttnn.deallocate(vision_real)
        ttnn.deallocate(lang_tokens_tt)
        _slog("prefix built")

        # ── Stage 1: Prefill TP=8 on parent ─────────────────────────────
        _slog(f"prefill (prefix_cos override = {self._prefix_cos is not None})")
        if self._prefix_cos is not None:
            num_layers = len(self.prefill.blocks)
            _final, parent_kv = self.prefill.run(
                prefix_embs,
                attention_mask=self._prefix_attn_mask,
                per_chip_cos=[self._prefix_cos] * num_layers,
                per_chip_sin=[self._prefix_sin] * num_layers,
            )
        else:
            _final, parent_kv = self.prefill.run(prefix_embs, attention_mask=self._prefix_attn_mask)
        ttnn.deallocate(prefix_embs)
        _slog(f"prefill done; {len(parent_kv)} KV layers on parent")

        # Match single-chip's tile-padding zero-fill (only when masked path active).
        if self._prefix_attn_mask is not None:
            _slog("fill_implicit_tile_padding on KV cache")
            parent_kv = [
                (ttnn.fill_implicit_tile_padding(k, 0.0), ttnn.fill_implicit_tile_padding(v, 0.0)) for k, v in parent_kv
            ]
            _slog("tile padding fill done")

        # ── LAZY denoise build (AFTER all parent CCL ops complete) ──────
        # Submesh existence blocks parent's all_gather/all_reduce, so we
        # defer submesh creation until vision DP + prefill (the parent's
        # CCL-using stages) are done.
        if not self._denoise_built:
            _slog("triggering lazy denoise build (after parent CCL done)")
            self._build_denoise_lazy()
            if hasattr(self, "_pending_img_masks"):
                _slog("applying pending runtime masks now that submesh exists")
                self.prepare_runtime_masks(self._pending_img_masks, self._pending_lang_masks)
                del self._pending_img_masks
                del self._pending_lang_masks
            _slog("lazy build complete")

        # ── HOST BOUNCE: parent KV → denoise submesh ────────────────────
        _slog("host-bouncing KV: parent → denoise submesh (18 layers × K+V = 36 round-trips)")
        per_layer_kv = self._bounce_kv_to_submesh(parent_kv)
        _slog(f"host bounce done; per_layer_kv has {len(per_layer_kv)} layers on submesh")
        # Free parent KV after bounce.
        for k, v in parent_kv:
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        _slog("parent KV deallocated")

        # ── Allocate x_t buffer on submesh ──────────────────────────────
        _slog("allocating x_t noise on submesh")
        ah = self.action_horizon
        ah_padded = self._action_horizon_padded
        noise = torch.zeros(1, ah_padded, self.action_dim, dtype=torch.float32)
        noise[:, :ah, :] = torch.randn(1, ah, self.action_dim, dtype=torch.float32)
        x_t_fp32 = ttnn.from_torch(
            noise,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.denoise_submesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _slog("x_t allocated")

        # ── Stage 2: 5-step Euler denoise on submesh ────────────────────
        _slog(f"starting {self.num_denoising_steps}-step Euler denoise")
        for i in range(self.num_denoising_steps):
            _slog(f"  denoise step {i}")
            dt = self._dts[i]
            adarms_cond = self._adarms_per_step[i]
            block_mods = self._block_mods_per_step[i]
            final_mod = self._final_mods_per_step[i]

            x_t_bf16 = ttnn.typecast(x_t_fp32, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
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

            velocity_bf16 = self._apply_final_norm_and_project(
                expert_out,
                adarms_cond,
                precomputed_final_mod=final_mod,
            )
            ttnn.deallocate(expert_out)
            velocity_fp32 = ttnn.typecast(velocity_bf16, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(velocity_bf16)
            velocity_scaled = ttnn.mul(velocity_fp32, dt, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(velocity_fp32)
            ttnn.add(x_t_fp32, velocity_scaled, output_tensor=x_t_fp32)
            ttnn.deallocate(velocity_scaled)

        _slog("denoise loop done")
        for k, v in per_layer_kv:
            ttnn.deallocate(k)
            ttnn.deallocate(v)

        # ── Read out actions from submesh ───────────────────────────────
        _slog("reading actions from submesh")
        x_t_final = ttnn.to_torch(x_t_fp32).float()
        ttnn.deallocate(x_t_fp32)
        actions = x_t_final[:1, :ah, :]

        # ── Tear down submesh so next chunk's parent CCL ops work ───────
        _slog("tearing down submesh (so next chunk's vision DP + prefill can run)")
        self._teardown_denoise_submesh()
        _slog(f"sample_actions returning shape={tuple(actions.shape)}")
        return actions

    def _teardown_denoise_submesh(self):
        """Close the submesh and clear all denoise-side device tensors.
        Next sample_actions will rebuild via _build_denoise_lazy.
        """
        if self.denoise_submesh is None:
            return
        for attr in (
            "_suffix_cos",
            "_suffix_sin",
            "_expert_attn_mask",
        ):
            t = getattr(self, attr, None)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, attr, None)
        self._adarms_per_step.clear()
        self._block_mods_per_step.clear()
        self._final_mods_per_step.clear()
        self.suffix = None
        self.denoise = None
        self.denoise_head = None
        try:
            ttnn.close_mesh_device(self.denoise_submesh)
        except Exception as _e:
            pass
        self.denoise_submesh = None
        self._denoise_built = False
        self._artifact_mask_key = None

    # ──────────── Cleanup ───────────────────────────────────────────────

    def close(self):
        """Explicit cleanup. Submesh must close before parent."""
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
        # TTNN handles tensor cleanup when the submesh closes, but be explicit
        # about ordering: caller must close the parent mesh (done outside).
