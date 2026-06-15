# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pi0_5GLXPipeline — end-to-end sample_actions across the 28-chip BH Galaxy
host-bounce pipeline.

Composition:
    vision      4 chips   (StageVision)
    prefill    18 chips   (StagePrefill)        + lang token embed on prefill[0]
    denoise     6 chips   (StageDenoise)        + replicated SuffixSlice + final adaRMS
                                                  on each denoise chip

Per-call sequence (sample_actions):
    1. Vision: 4-chip SigLIP + mm_projector → (B, 256, 2048) per camera on vision[3]
    2. Host-bounce vision_out → prefill[0]
    3. Build prefix: reshape (N_cams, 256, 2048) → (1, 256*N_cams, 2048); embed lang
       tokens on prefill[0] (vlm_embed_tokens); concat → (1, prefix_len, 2048)
    4. Prefill: 18-chip VLM → final_hidden, per_layer_kv on the prefill chips
    5. KV migration: layer-paired host-bounce + typecast to bf8_b, → denoise chips
    6. Denoise loop (N steps; host-side Euler integration of x_t):
       a. embed_actions(x_t) on denoise[0] via SuffixSlice
       b. embed_adarms_cond(t) on EACH denoise chip
       c. expert chain across 6 chips with host bounces
       d. final adaRMS norm on denoise[5] (ada_rms_norm_no_gate_ttnn)
       e. project_output(action_out_proj) on denoise[5]
       f. velocity → host → x_t ← x_t + dt·velocity (fp32 on host)
    7. Slice x_t (1, 32, 32) → (1, action_horizon, action_dim) and return

Differences vs single-chip ttnn_pi0_5_model.py:
    - batch_size = 1 only
    - No trace + 2CQ yet (runs eager; trace capture is scaffolded but not
      working — see capture_trace / Phase B.3). This is the main perf gap.
    - No keep_padded fast path in the denoise loop.

Already at parity with the single-chip path:
    - PI0_UPSTREAM_MASKS attention mask + position-aware RoPE (per-chip).
    - Precomputed per-(step, chip) adarms_cond (Phase B.2).
    - On-device fp32 Euler integration with an in-place x_t buffer (Phase B.1);
      x_t is refreshed via copy_host_to_device_tensor, not re-uploaded.
    - Per-op matmul/SDPA/dtype tuning is inherited from the shared ttnn_gemma /
      ttnn_siglip / ttnn_common building blocks (same env flags apply per chip).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig, SuffixConfig
from models.experimental.pi0_5.tt.ttnn_gemma import ada_rms_norm_no_gate_ttnn
from models.experimental.pi0_5.tt.ttnn_pi0_5_model import (
    _MASK_VAL,
    _precompute_rope_table_torch,
    use_upstream_masks,
)

from ._l1_migration import (
    denoise_l1_enabled,
    migrate_denoise_weights_to_l1,
    migrate_prefill_mlp_weights_to_l1,
    migrate_prefill_vlm_weights_to_l1,
    migrate_siglip_weights_to_l1,
    migrate_vlm_attn_to_l1,
    prefill_mlp_l1_enabled,
    siglip_attn_l1_enabled,
    siglip_l1_tensors,
    siglip_mlp_l1_enabled,
    vlm_attn_l1_enabled,
    vlm_attn_l1_tensors,
    prefill_mlp_l1_grid,
    prefill_mlp_l1_layout,
    prefill_mlp_l1_projs,
    prefill_vlm_l1_enabled,
    prefill_vlm_l1_projs,
    siglip_l1_enabled,
)
from .kv_migration import migrate_layer_paired
from .stage_denoise import StageDenoise
from .stage_prefill import StagePrefill
from .stage_vision import StageVision
from .stages import StageTimings
from .suffix_slice import SuffixSlice
from .transport import SocketTransport


_HIDDEN_SCALE_DTYPE = ttnn.bfloat16


def _upload_torch(tensor_torch: torch.Tensor, submesh, dtype=ttnn.bfloat16, mem_cfg=None, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor_torch,
        dtype=dtype,
        layout=layout,
        device=submesh,
        memory_config=mem_cfg if mem_cfg is not None else ttnn.DRAM_MEMORY_CONFIG,
    )


class _PrefillHead:
    """Lang token embedding + prefix concat — lives on prefill_per_chip[0]."""

    def __init__(self, vlm_language_weights: Dict[str, torch.Tensor], submesh, vlm_hidden: int):
        embed_w = vlm_language_weights.get("model.embed_tokens.weight") or vlm_language_weights.get("lm_head.weight")
        if embed_w is None:
            raise RuntimeError("vlm_embed_tokens weight not found")
        self.submesh = submesh
        self.vlm_hidden = vlm_hidden
        self.embed_scale = float(vlm_hidden) ** 0.5
        self.vlm_embed_tokens = ttnn.from_torch(
            embed_w,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,  # embeddings use row-major
            device=submesh,
        )

    def embed_lang(self, lang_tokens) -> "ttnn.Tensor":
        """Returns (B, lang_len, vlm_hidden) bf16 TILE.

        ``lang_tokens`` is either a torch.Tensor (eager path — uploaded inline)
        or a persistent ttnn.Tensor on self.submesh (trace path — references
        the pre-allocated buffer that copy_host_to_device_tensor refreshes
        between calls).
        """
        if isinstance(lang_tokens, torch.Tensor):
            tokens_ttnn = ttnn.from_torch(
                lang_tokens.to(torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.submesh,
            )
        else:
            tokens_ttnn = lang_tokens
        lang_emb = ttnn.embedding(tokens_ttnn, self.vlm_embed_tokens)
        if lang_emb.layout != ttnn.TILE_LAYOUT:
            lang_emb = ttnn.to_layout(lang_emb, ttnn.TILE_LAYOUT)
        return ttnn.mul(lang_emb, self.embed_scale)


class _DenoiseHead:
    """Final adaRMS norm weights for the last expert chip."""

    def __init__(self, expert_weights: Dict[str, torch.Tensor], submesh):
        from models.experimental.pi0_5.tt.ttnn_common import tensor_1d_to_2d_ttnn

        self.submesh = submesh
        mod_w = expert_weights["model.norm.dense.weight"]
        self.mod_weight = ttnn.from_torch(
            mod_w.T.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=submesh
        )
        mod_b = expert_weights.get("model.norm.dense.bias")
        self.mod_bias = tensor_1d_to_2d_ttnn(mod_b, submesh, dtype=ttnn.bfloat16) if mod_b is not None else None
        device_grid = submesh.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)


class Pi0_5GLXPipeline:
    """End-to-end host-bounced inference for pi0.5 across 28 BH Galaxy chips."""

    def __init__(self, config: Pi0_5ModelConfig, weights: Dict[str, dict], mesh_handles):
        self.config = config
        self.h = mesh_handles
        self.num_denoising_steps = config.num_denoising_steps
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self._action_horizon_padded = ((config.action_horizon + 31) // 32) * 32

        # Single SocketTransport instance shared across all stages + KV migration.
        # Caches one socket pair + receiver buffer per (src_chip, dst_chip, tag)
        # so the cross-chip wiring is paid once at first call and reused on
        # every subsequent sample_actions.
        self.transport = SocketTransport()

        # Per-stage executors (all share the same transport).
        self.stage_vision = StageVision(config.siglip_config, weights, mesh_handles, transport=self.transport)
        self.stage_prefill = StagePrefill(config, weights, mesh_handles, transport=self.transport)
        self.stage_denoise = StageDenoise(config, weights, mesh_handles, transport=self.transport)

        # Lang token embed on prefill[0].
        self.prefill_head = _PrefillHead(
            weights["vlm_language"], mesh_handles.prefill_per_chip[0], config.vlm_config.width
        )

        # Replicated suffix MLPs — one copy per denoise chip. Build SuffixConfig
        # from Pi0_5ModelConfig fields (mirrors ttnn_pi0_5_model.py:559).
        suffix_cfg = SuffixConfig(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            expert_width=config.expert_config.width,
            pi05=True,
        )
        self.suffix_slices: List[SuffixSlice] = [
            SuffixSlice(suffix_cfg, weights["pi0_projections"], chip) for chip in mesh_handles.denoise_per_chip
        ]

        # Final adaRMS norm + bias on the LAST denoise chip (last expert chunk).
        self.denoise_head = _DenoiseHead(weights["action_expert"], mesh_handles.denoise_per_chip[-1])

        # Persistent fp32 x_t buffer on denoise[0] for on-device Euler integration.
        # Allocated once at __init__; sample_actions refreshes its contents per call
        # via from_torch (or copy_host_to_device_tensor under trace mode).
        # NOTE: `self.x_t_fp32` is REASSIGNED each Euler step (ttnn.add creates a new
        # tensor). Trace mode in Phase B.3 will switch to ttnn.add with optional
        # output tensor to keep the same buffer.
        _zero_noise = torch.zeros(1, self._action_horizon_padded, self.action_dim, dtype=torch.float32)
        self.x_t_fp32 = ttnn.from_torch(
            _zero_noise,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_handles.denoise_per_chip[0],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Pre-compute timesteps + dt scalars (Python floats; ttnn.mul accepts them).
        self._timesteps = [1.0 - i / self.num_denoising_steps for i in range(self.num_denoising_steps + 1)]
        self._dts = [self._timesteps[i + 1] - self._timesteps[i] for i in range(self.num_denoising_steps)]

        # Pre-compute adarms_cond per (step, chip). Since timesteps are
        # deterministic, the conditioning tensors are constant across calls;
        # mirrors single-chip _precompute_bs1_adarms_cond. Each step gets its
        # own copy on each denoise chip so the denoise loop body indexes a
        # plain list — no per-step ttnn.from_torch + embed_adarms_cond chain
        # (which would block trace capture).
        self._adarms_per_step_per_chip: List[List["ttnn.Tensor"]] = []
        for i in range(self.num_denoising_steps):
            ts_now = self._timesteps[i]
            per_chip: List["ttnn.Tensor"] = []
            for chip, sl in zip(self.h.denoise_per_chip, self.suffix_slices):
                t_ttnn = ttnn.from_torch(
                    torch.tensor([ts_now], dtype=torch.float32),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=chip,
                )
                cond = sl.embed_adarms_cond(t_ttnn)
                ttnn.deallocate(t_ttnn)
                per_chip.append(cond)
            self._adarms_per_step_per_chip.append(per_chip)

        # Persistent host-side buffers for trace mode (Phase B.3). Stay as
        # torch tensors until first sample_actions call materializes the
        # ttnn versions on the right chips (via copy_host_to_device_tensor
        # under trace mode, or fresh from_torch in eager mode).
        self.pixel_values_buf = None  # on vision[0], (N_cams, 3, 224, 224) bf16
        self.lang_tokens_buf = None  # on prefill[0], (1, lang_len) uint32

        # Cached upstream-openpi compat artifacts (mask + position-aware RoPE).
        # Keyed by (img_present_tuple, lang_real_count, prefix_len). Same idea
        # as the single-chip Pi0_5ModelTTNN._cached_upstream_artifacts (see
        # ttnn_pi0_5_model.py:344). Built lazily on the first sample_actions
        # call and reused within a task. Each artifact set is replicated across
        # the appropriate stage's per-chip submeshes.
        self._upstream_cache_key = None
        self._upstream_per_chip = None  # dict with prefix_* (18-list) and expert_* (6-list)

        # Trace state (Phase B.3). _trace_id is set after capture_trace(). The
        # output tensor _captured_actions is what the trace's final ops write
        # to; sample_actions_traced reads from it after execute_trace.
        self._trace_id = None
        self._captured_actions = None

        # Per-stage trace state (Phase B.3 alt). Three separate traces on the
        # three stage submeshes (vision_submesh, prefill_submesh, denoise_submesh)
        # rather than one parent-mesh trace. Inter-stage transports (vision→prefill
        # socket, KV migration) run between traces, in Python — they aren't
        # captured. Caches:
        #   _vision_trace_id / _vision_out_buf   — trace + output tensor on vision_submesh
        #   _prefill_trace_id / _prefill_kv_buf  — trace + per-layer KV refs on prefill_submesh
        #   _denoise_trace_id                    — trace; output is self.x_t_fp32 on denoise_submesh
        # _prefix_kv_per_chip is cached after the first migration (lives across
        # subsequent calls; refreshed only when the prefill output changes).
        self._vision_trace_id = None
        self._vision_out_buf = None
        self._prefill_trace_id = None
        self._prefill_kv_buf = None
        self._denoise_trace_id = None
        self._prefix_kv_per_chip = None
        self._rope_l1 = os.environ.get("PI0_ROPE_TABLES_L1", "").lower() in ("1", "true", "yes", "on")

        # Opt-in: move SigLIP and VLM matmul weights to L1. These stages run once
        # per sample (not per denoise step), so they stay gated while we measure
        # L1 pressure and stage-latency impact.
        if siglip_l1_enabled():
            migrate_siglip_weights_to_l1(self.stage_vision)
        if prefill_vlm_l1_enabled():
            migrate_prefill_vlm_weights_to_l1(self.stage_prefill, prefill_vlm_l1_projs())

        # Move the static denoise-stage weights from DRAM into L1 so the N-step
        # Euler loop reads them on-chip instead of re-streaming ~93 MB/chip from
        # DRAM every step. Gated by PI0_GLX_DENOISE_L1 (default ON). The flag is
        # also threaded into KV migration so the per-call prefix KV lands in L1.
        self._denoise_l1 = denoise_l1_enabled()
        if self._denoise_l1:
            migrate_denoise_weights_to_l1(self.stage_denoise, self.suffix_slices, self.denoise_head)

        # Opt-in (PI0_GLX_PREFILL_MLP_L1=1): width-shard the prefill VLM MLP
        # weights (gate/up/down) into L1 for a width-shard-aware matmul. OFF by
        # default — the current matmul path expects interleaved weights.
        if prefill_mlp_l1_enabled():
            gx, gy = prefill_mlp_l1_grid()
            migrate_prefill_mlp_weights_to_l1(
                self.stage_prefill, prefill_mlp_l1_layout(), gx, gy, prefill_mlp_l1_projs()
            )

        # Per-tensor SigLIP L1 migration (e.g. PI0_GLX_SIGLIP_L1_TENSORS=wo,fc2).
        # fc1 and wqkv (N=4608) crash with CB-clash; wo and fc2 (N=1152) are
        # safe candidates. Empty/unset → no migration (production default).
        _siglip_tensors = list(siglip_l1_tensors())
        # Back-compat: bool flags fold into the allowlist
        if siglip_mlp_l1_enabled():
            _siglip_tensors += ["fc1", "fc2"]
        if siglip_attn_l1_enabled():
            _siglip_tensors += ["wqkv", "wo"]
        if _siglip_tensors:
            migrate_siglip_weights_to_l1(self.stage_vision, tensors=set(_siglip_tensors))

        # Opt-in: VLM PaliGemma attention (wqkv + o_proj) → L1.
        # Per-chip footprint ~9.7 MB / 168 MB usable. Default OFF.
        if vlm_attn_l1_enabled():
            migrate_vlm_attn_to_l1(self.stage_prefill, tensors=vlm_attn_l1_tensors())

    # ---- upstream-openpi compat artifacts -------------------------------

    def _upstream_key(
        self, img_masks: Optional[List[torch.Tensor]], lang_masks: Optional[torch.Tensor], prefix_len: int
    ):
        img_present = tuple(bool(m.item()) if m.numel() == 1 else bool(m[0].item()) for m in (img_masks or []))
        lang_real = int(lang_masks.to(torch.bool)[0].sum().item()) if lang_masks is not None else 0
        return (img_present, lang_real, prefix_len)

    def _build_and_upload_upstream_artifacts(
        self,
        img_masks: List[torch.Tensor],
        lang_masks: torch.Tensor,
        prefix_len: int,
    ) -> None:
        """Port of Pi0_5ModelTTNN._build_upstream_attn_artifacts adapted for the
        mesh model: each prefill chip gets its own (prefix_attn_mask, prefix_cos,
        prefix_sin) copy; each denoise chip gets its own (expert_attn_mask,
        suffix_cos, suffix_sin). Results cached on self._upstream_per_chip.
        """
        num_image_tokens = self.config.siglip_config.num_patches
        action_horizon = self.action_horizon
        suffix_padded = self._action_horizon_padded
        prefix_padded = ((prefix_len + 31) // 32) * 32
        vlm_head_dim = self.config.vlm_config.head_dim
        expert_head_dim = self.config.expert_config.head_dim
        max_seq_len = self.config.max_seq_len

        # 1) 1D prefix pad_mask (image-pad-aware + lang-pad-aware).
        pad_segments = []
        for m in img_masks:
            real = bool(m.item()) if m.numel() == 1 else bool(m[0].item())
            pad_segments.append(torch.full((num_image_tokens,), real, dtype=torch.bool))
        pad_segments.append(lang_masks[0].to(torch.bool))
        pad_mask = torch.cat(pad_segments, dim=0)
        assert pad_mask.shape[0] == prefix_len, (pad_mask.shape, prefix_len)
        prefix_real_count = int(pad_mask.sum().item())

        # 2) Prefix attention mask (additive bf16, finite -1e4 for masked).
        prefix_all_real_aligned = (prefix_real_count == prefix_len) and (prefix_padded == prefix_len)
        if prefix_all_real_aligned:
            prefix_mask_4d = None
        else:
            pad_2d = pad_mask[:, None] & pad_mask[None, :]
            pm = torch.zeros(prefix_padded, prefix_padded, dtype=torch.bfloat16)
            pm[:prefix_len, :prefix_len].masked_fill_(~pad_2d, _MASK_VAL)
            if prefix_padded > prefix_len:
                pm[prefix_len:, :] = _MASK_VAL
                pm[:, prefix_len:] = _MASK_VAL
            prefix_mask_4d = pm.unsqueeze(0).unsqueeze(0)

        # 3) Prefix RoPE at cumsum(pad)-1 positions.
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

        # 4) Suffix RoPE at prefix_real_count + [0..suffix_padded-1].
        cos_exp, sin_exp = _precompute_rope_table_torch(expert_head_dim, max_seq_len)
        suffix_positions = (torch.arange(suffix_padded, dtype=torch.int64) + prefix_real_count).clamp(
            max=max_seq_len - 1
        )
        suffix_cos_4d = cos_exp[suffix_positions].unsqueeze(0).unsqueeze(0)
        suffix_sin_4d = sin_exp[suffix_positions].unsqueeze(0).unsqueeze(0)

        # 5) Expert cross-attention mask (suffix→prefix+suffix).
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

        # Upload: replicate to each chip on the relevant submesh.
        rope_mc = ttnn.L1_MEMORY_CONFIG if self._rope_l1 else ttnn.DRAM_MEMORY_CONFIG

        def _up(host_t, chip, mc):
            return ttnn.from_torch(
                host_t.to(torch.bfloat16) if host_t.dtype != torch.bfloat16 else host_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=chip,
                memory_config=mc,
            )

        per_chip_prefix_mask = (
            [_up(prefix_mask_4d, c, ttnn.DRAM_MEMORY_CONFIG) for c in self.h.prefill_per_chip]
            if prefix_mask_4d is not None
            else None
        )
        per_chip_prefix_cos = [_up(prefix_cos_4d, c, rope_mc) for c in self.h.prefill_per_chip]
        per_chip_prefix_sin = [_up(prefix_sin_4d, c, rope_mc) for c in self.h.prefill_per_chip]
        per_chip_expert_mask = [_up(expert_mask_4d, c, ttnn.DRAM_MEMORY_CONFIG) for c in self.h.denoise_per_chip]
        per_chip_suffix_cos = [_up(suffix_cos_4d, c, rope_mc) for c in self.h.denoise_per_chip]
        per_chip_suffix_sin = [_up(suffix_sin_4d, c, rope_mc) for c in self.h.denoise_per_chip]

        self._upstream_per_chip = {
            "prefix_attn_mask": per_chip_prefix_mask,  # None or len-18 list
            "prefix_cos": per_chip_prefix_cos,
            "prefix_sin": per_chip_prefix_sin,
            "expert_attn_mask": per_chip_expert_mask,
            "suffix_cos": per_chip_suffix_cos,
            "suffix_sin": per_chip_suffix_sin,
            "prefix_real_count": prefix_real_count,
        }

    # ---- helpers ----------------------------------------------------------

    def _build_prefix(self, vision_out_on_p0: "ttnn.Tensor", lang_tokens_torch: torch.Tensor) -> "ttnn.Tensor":
        """Reshape vision (N_cams, 256, 2048) → (1, N_cams*256, 2048); embed lang; concat.

        Returns (1, prefix_len, vlm_width) on prefill[0].
        """
        v_shape = vision_out_on_p0.shape  # (N_cams, 256, vlm_width)
        n_cams = int(v_shape[0])
        num_patches = int(v_shape[1])
        vlm_w = int(v_shape[2])
        v = ttnn.reshape(vision_out_on_p0, (1, n_cams * num_patches, vlm_w))

        lang_emb = self.prefill_head.embed_lang(lang_tokens_torch)
        # Both v and lang_emb are TILE, bf16. Concat along seq dim.
        prefix = ttnn.concat([v, lang_emb], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        if prefix.layout != ttnn.TILE_LAYOUT:
            prefix = ttnn.to_layout(prefix, ttnn.TILE_LAYOUT)
        return prefix

    def _embed_adarms_per_chip(self, t_torch: torch.Tensor) -> List["ttnn.Tensor"]:
        """Compute adarms_cond on each denoise chip from a scalar timestep."""
        out: List["ttnn.Tensor"] = []
        for slice_, chip in zip(self.suffix_slices, self.h.denoise_per_chip):
            t_ttnn = ttnn.from_torch(
                t_torch.to(torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=chip,
            )
            out.append(slice_.embed_adarms_cond(t_ttnn))
        return out

    def _apply_final_norm_and_project(
        self, expert_out: "ttnn.Tensor", adarms_cond_on_last: "ttnn.Tensor"
    ) -> "ttnn.Tensor":
        """ada_rms_norm_no_gate + action_out_proj on the last denoise chip."""
        normed = ada_rms_norm_no_gate_ttnn(
            expert_out,
            adarms_cond_on_last,
            self.denoise_head.mod_weight,
            self.denoise_head.mod_bias,
            self.config.expert_config.rms_norm_eps,
            self.denoise_head.core_grid,
        )
        return self.suffix_slices[-1].project_output(normed)

    # ---- persistent input buffers + trace (Phase B.3) -------------------

    def _ensure_persistent_input_buffers(self, images: List[torch.Tensor], lang_tokens: torch.Tensor) -> None:
        """Lazily allocate self.pixel_values_buf + self.lang_tokens_buf the first
        time we see input of these shapes. Subsequent calls only need to refresh
        their contents via copy_host_to_device_tensor (no reallocation)."""
        pixel_values = torch.cat(images, dim=0)
        if self.pixel_values_buf is None:
            self.pixel_values_buf = ttnn.from_torch(
                pixel_values,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.h.vision_per_chip[0],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            host_t = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_t, self.pixel_values_buf)

        if self.lang_tokens_buf is None:
            self.lang_tokens_buf = ttnn.from_torch(
                lang_tokens.to(torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.h.prefill_per_chip[0],
            )
        else:
            host_t = ttnn.from_torch(lang_tokens.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_t, self.lang_tokens_buf)

    def _refresh_noise_buffer(self) -> None:
        """Re-fill self.x_t_fp32 in-place with fresh N(0,1) noise (logical [:ah], zero-padded)."""
        ah = self.action_horizon
        ah_padded = self._action_horizon_padded
        noise_pad = torch.zeros(1, ah_padded, self.action_dim, dtype=torch.float32)
        noise_pad[:, :ah, :] = torch.randn(1, ah, self.action_dim)
        host_t = ttnn.from_torch(noise_pad, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_t, self.x_t_fp32)

    def _sample_actions_device(self, upstream):
        """Pure-device sample_actions body. Reads from persistent input buffers
        (self.pixel_values_buf, self.lang_tokens_buf, self.x_t_fp32) and writes
        the final actions into self.x_t_fp32 in-place. No host I/O — suitable
        for trace capture.

        Returns the final actions tensor (= self.x_t_fp32 by reference) so the
        caller can either ttnn.to_torch it (eager) or remember it as the trace
        output handle (trace mode).

        PI0_GLX_TRACE_DEBUG=1 prints a marker after each stage enqueues and then
        drains that stage's submesh with ttnn.synchronize_device. A stall then
        localizes to either enqueue (the stage's "enqueued" line never prints)
        or execution (the run hangs at that stage's drain) — and tells us whether
        per-stage draining is enough to make the no-readback path complete (the
        eager path only completes because of its terminal to_torch)."""
        dbg = os.environ.get("PI0_GLX_TRACE_DEBUG", "").lower() in ("1", "true", "yes", "on")

        def _mark(stage: str, drain_mesh=None):
            if not dbg:
                return
            print(f"[glx-trace-dbg] {stage}: enqueued", flush=True)
            if drain_mesh is not None:
                ttnn.synchronize_device(drain_mesh)
                print(f"[glx-trace-dbg] {stage}: drained", flush=True)

        vision_out = self.stage_vision.run(self.pixel_values_buf)
        _mark("vision", self.h.vision_per_chip[-1])
        vision_out_p0 = self.transport.send(vision_out, self.h.prefill_per_chip[0], tag="v2p")
        _mark("v2p", self.h.prefill_per_chip[0])
        prefix_embs = self._build_prefix(vision_out_p0, self.lang_tokens_buf)
        _mark("build_prefix", self.h.prefill_per_chip[0])
        _final_hidden, per_layer_kv = self.stage_prefill.run(
            prefix_embs,
            attention_mask=None,
            position_ids=None,
            per_chip_attn_mask=(upstream["prefix_attn_mask"] if upstream is not None else None),
            per_chip_cos=(upstream["prefix_cos"] if upstream is not None else None),
            per_chip_sin=(upstream["prefix_sin"] if upstream is not None else None),
        )
        _mark("prefill", self.h.prefill_per_chip[-1])
        prefix_kv_per_chip = migrate_layer_paired(
            per_layer_kv, self.h.denoise_per_chip, transport=self.transport, to_l1=self._denoise_l1
        )
        _mark("kv_migration", self.h.denoise_per_chip[-1])
        self._run_denoise_loop_device(prefix_kv_per_chip, upstream)
        _mark("denoise", self.h.denoise_per_chip[0])
        return self.x_t_fp32

    def capture_trace(
        self,
        images: List[torch.Tensor],
        img_masks: Optional[List[torch.Tensor]],
        lang_tokens: torch.Tensor,
        lang_masks: Optional[torch.Tensor],
    ) -> None:
        """One-time setup: stage all persistent buffers + upstream artifacts,
        then capture the entire device-side compute as a TTNN trace on the
        parent mesh's CQ 0. After this completes, sample_actions_traced() can
        be called repeatedly with new inputs and will replay the trace.
        """
        # 1) Stage the persistent inputs.
        self._ensure_persistent_input_buffers(images, lang_tokens)
        self._refresh_noise_buffer()

        # 2) Build / cache upstream artifacts (depends on img_masks + lang_masks + prefix_len).
        upstream = None
        if use_upstream_masks() and img_masks is not None and lang_masks is not None:
            num_image_tokens = self.config.siglip_config.num_patches
            prefix_len = num_image_tokens * len(images) + int(lang_tokens.shape[-1])
            key = self._upstream_key(img_masks, lang_masks, prefix_len)
            if self._upstream_per_chip is None or self._upstream_cache_key != key:
                self._build_and_upload_upstream_artifacts(img_masks, lang_masks, prefix_len)
                self._upstream_cache_key = key
            upstream = self._upstream_per_chip

        # 3) Warm up once so JIT compiles all kernels before begin_trace_capture.
        # The trace allocator can't tolerate JIT compilation inside the trace body.
        _ = self._sample_actions_device(upstream)

        # 4) Refresh noise (warmup mutated x_t_fp32) and capture.
        self._refresh_noise_buffer()
        # Root the trace on the (7,4) compute submesh, NOT the (8,4) parent: a
        # trace's blocking finish defaults to the root mesh's full range, so a
        # parent-rooted trace would wait on the idle row-7 chips (no commands →
        # empty completion queue) and deadlock end_trace_capture. The compute
        # submesh is exactly the 28 commanded chips.
        self._trace_id = ttnn.begin_trace_capture(self.h.trace_root, cq_id=0)
        self._captured_actions = self._sample_actions_device(upstream)
        ttnn.end_trace_capture(self.h.trace_root, self._trace_id, cq_id=0)

    # ---- Per-stage trace fallback (Phase B.3 alt) -----------------------
    #
    # The whole-pipeline capture_trace() above did not complete on this 32-chip
    # parent MeshDevice (warmup hangs; root cause TBD — likely socket handshake
    # ordering across many submesh CQs). As a fallback, capture three SEPARATE
    # traces — one per stage submesh — and stitch them together in Python with
    # the inter-stage sockets / KV migration happening between traces.
    #
    # Layout:
    #   vision  trace on h.vision_submesh   (1,4)  — input self.pixel_values_buf
    #                                                  output self._vision_out_buf
    #   socket vision_per_chip[3] → prefill_per_chip[0] (Python-side, between traces)
    #   prefill trace on h.prefill_submesh  (6,3)  — input vision_out_p0
    #                                                  output (final_hidden, per_layer_kv)
    #   KV migration prefill chips → denoise chips (Python-side)
    #   denoise trace on h.denoise_submesh  (6,1)  — input self.x_t_fp32, prefix_kv
    #                                                  output self.x_t_fp32 (in-place)
    #
    # IMPORTANT — known limitation: the stage compute currently runs on per-chip
    # 1x1 submeshes (h.vision_per_chip[i], etc.) which have their own command
    # queues distinct from the stage submesh CQ. The trace capture below issues
    # the SAME per-chip compute path, so whether begin_trace_capture on the
    # stage submesh captures ops issued on its child 1x1 submeshes is the open
    # question. If it doesn't, the captured trace will be empty / replay will
    # be a no-op. This scaffolding lays out the call sites; validation is
    # deferred to a future debug session.

    def capture_per_stage_traces(
        self,
        images: List[torch.Tensor],
        img_masks: Optional[List[torch.Tensor]],
        lang_tokens: torch.Tensor,
        lang_masks: Optional[torch.Tensor],
    ) -> None:
        """Capture three per-stage traces (vision, prefill, denoise). Inter-stage
        transports run between traces. Must be called once before
        sample_actions_per_stage_traced.
        """
        # 1) Stage persistent inputs.
        self._ensure_persistent_input_buffers(images, lang_tokens)
        self._refresh_noise_buffer()

        # 2) Build / cache upstream artifacts.
        upstream = None
        if use_upstream_masks() and img_masks is not None and lang_masks is not None:
            num_image_tokens = self.config.siglip_config.num_patches
            prefix_len = num_image_tokens * len(images) + int(lang_tokens.shape[-1])
            key = self._upstream_key(img_masks, lang_masks, prefix_len)
            if self._upstream_per_chip is None or self._upstream_cache_key != key:
                self._build_and_upload_upstream_artifacts(img_masks, lang_masks, prefix_len)
                self._upstream_cache_key = key
            upstream = self._upstream_per_chip

        # 3) Warmup pass (JIT compiles all kernels). Each stage's compute runs
        # once eagerly so begin_trace_capture doesn't have to tolerate JIT.
        _ = self._sample_actions_device(upstream)
        self._refresh_noise_buffer()

        # 4) Vision trace.
        self._vision_trace_id = ttnn.begin_trace_capture(self.h.vision_submesh, cq_id=0)
        self._vision_out_buf = self.stage_vision.run(self.pixel_values_buf)
        ttnn.end_trace_capture(self.h.vision_submesh, self._vision_trace_id, cq_id=0)

        # Inter-stage transport vision→prefill (NOT in any trace — happens each call).
        vision_out_p0 = self.transport.send(self._vision_out_buf, self.h.prefill_per_chip[0], tag="v2p")

        # 5) Prefill trace.
        self._prefill_trace_id = ttnn.begin_trace_capture(self.h.prefill_submesh, cq_id=0)
        prefix_embs = self._build_prefix(vision_out_p0, self.lang_tokens_buf)
        _final_hidden, per_layer_kv = self.stage_prefill.run(
            prefix_embs,
            attention_mask=None,
            position_ids=None,
            per_chip_attn_mask=(upstream["prefix_attn_mask"] if upstream is not None else None),
            per_chip_cos=(upstream["prefix_cos"] if upstream is not None else None),
            per_chip_sin=(upstream["prefix_sin"] if upstream is not None else None),
        )
        self._prefill_kv_buf = per_layer_kv  # ttnn.Tensor refs per chip
        ttnn.end_trace_capture(self.h.prefill_submesh, self._prefill_trace_id, cq_id=0)

        # KV migration runs OUTSIDE traces — sockets between prefill chips and
        # denoise chips happen in Python.
        self._prefix_kv_per_chip = migrate_layer_paired(
            self._prefill_kv_buf, self.h.denoise_per_chip, transport=self.transport
        )

        # 6) Denoise trace. self.x_t_fp32 is read+written in-place each step.
        self._denoise_trace_id = ttnn.begin_trace_capture(self.h.denoise_submesh, cq_id=0)
        self._run_denoise_loop_device(self._prefix_kv_per_chip, upstream)
        ttnn.end_trace_capture(self.h.denoise_submesh, self._denoise_trace_id, cq_id=0)

    def sample_actions_per_stage_traced(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Replay the three per-stage traces, with inter-stage sockets + KV
        migration in Python between them.
        """
        if self._vision_trace_id is None or self._prefill_trace_id is None or self._denoise_trace_id is None:
            raise RuntimeError("capture_per_stage_traces() must be called before sample_actions_per_stage_traced()")

        # Refresh inputs into persistent buffers.
        self._ensure_persistent_input_buffers(images, lang_tokens)
        self._refresh_noise_buffer()

        # Replay vision.
        ttnn.execute_trace(self.h.vision_submesh, self._vision_trace_id, cq_id=0, blocking=False)

        # Vision→prefill socket (not captured).
        # NOTE: _vision_out_buf is what the vision trace writes to. The socket
        # send issues new ops on Python's CQ — they're sequenced after the
        # trace's writes by TTNN command-queue ordering on vision_per_chip[3].
        vision_out_p0 = self.transport.send(self._vision_out_buf, self.h.prefill_per_chip[0], tag="v2p")
        # (vision_out_p0 is the buffer the prefill trace expects to read from at
        # the start of its body — it must be the SAME tensor reference captured
        # during capture_per_stage_traces. send() returns the cached receiver
        # buffer for this (src, dst, tag), which is stable across calls.)

        # Replay prefill.
        ttnn.execute_trace(self.h.prefill_submesh, self._prefill_trace_id, cq_id=0, blocking=False)

        # KV migration (not captured).
        self._prefix_kv_per_chip = migrate_layer_paired(
            self._prefill_kv_buf, self.h.denoise_per_chip, transport=self.transport
        )

        # Replay denoise. self.x_t_fp32 holds the result after this returns.
        ttnn.execute_trace(self.h.denoise_submesh, self._denoise_trace_id, cq_id=0, blocking=True)

        ah = self.action_horizon
        return ttnn.to_torch(self.x_t_fp32)[:, :ah, :]

    def sample_actions_traced(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Replay the captured trace on new inputs. Must be preceded by
        capture_trace(). Refreshes pixel_values_buf, lang_tokens_buf, x_t_fp32
        via copy_host_to_device_tensor, executes the trace, reads the output.
        """
        if self._trace_id is None:
            raise RuntimeError("capture_trace() must be called before sample_actions_traced()")
        self._ensure_persistent_input_buffers(images, lang_tokens)
        self._refresh_noise_buffer()
        ttnn.execute_trace(self.h.trace_root, self._trace_id, cq_id=0, blocking=True)
        ah = self.action_horizon
        return ttnn.to_torch(self._captured_actions)[:, :ah, :]

    # ---- public entry ----------------------------------------------------

    def sample_actions(
        self,
        images: List[torch.Tensor],
        img_masks: Optional[List[torch.Tensor]] = None,  # unused in v1
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,  # unused in v1
        state: Optional[torch.Tensor] = None,  # unused for pi0.5
    ) -> Tuple[torch.Tensor, StageTimings]:
        """Run end-to-end inference. Returns (actions_torch, timings).

        Inputs:
            images: list of N_cams torch tensors, each (1, 3, 224, 224).
            lang_tokens: torch int64 tensor (1, lang_len) — typically lang_len=256.
        Output:
            actions_torch: (1, action_horizon=10, action_dim=32) float tensor.
            timings: StageTimings with per-stage ms.
        """
        if lang_tokens is None:
            raise ValueError("lang_tokens required")
        # Stack cameras into a single (N_cams, 3, H, W) tensor for the vision stage.
        pixel_values = torch.cat(images, dim=0)

        t = StageTimings()
        wall_start = time.perf_counter()

        # ---- Stage 0: vision ----
        t0 = time.perf_counter()
        vision_out = self.stage_vision.run(pixel_values)
        t.vision_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Transport 0→1: vision[3] → prefill[0] (fabric socket) ----
        t0 = time.perf_counter()
        vision_out_p0 = self.transport.send(vision_out, self.h.prefill_per_chip[0], tag="v2p")
        t.transport_v2p_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Build prefix embeddings + Stage 1: prefill ----
        prefix_embs = self._build_prefix(vision_out_p0, lang_tokens)
        prefix_len = int(prefix_embs.shape[1])

        # Upstream-openpi compat artifacts (gated by PI0_UPSTREAM_MASKS=1). Build
        # once per (img_present, lang_real_count, prefix_len) — cached across
        # subsequent sample_actions calls for the same prompt/scene shape.
        upstream = None
        if use_upstream_masks() and img_masks is not None and lang_masks is not None:
            key = self._upstream_key(img_masks, lang_masks, prefix_len)
            if self._upstream_per_chip is None or self._upstream_cache_key != key:
                self._build_and_upload_upstream_artifacts(img_masks, lang_masks, prefix_len)
                self._upstream_cache_key = key
            upstream = self._upstream_per_chip

        t0 = time.perf_counter()
        _final_hidden, per_layer_kv = self.stage_prefill.run(
            prefix_embs,
            attention_mask=None,
            position_ids=None,
            per_chip_attn_mask=(upstream["prefix_attn_mask"] if upstream is not None else None),
            per_chip_cos=(upstream["prefix_cos"] if upstream is not None else None),
            per_chip_sin=(upstream["prefix_sin"] if upstream is not None else None),
        )
        t.prefill_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Transport 1→2: layer-paired KV migration (fabric sockets) ----
        t0 = time.perf_counter()
        prefix_kv_per_chip = migrate_layer_paired(
            per_layer_kv, self.h.denoise_per_chip, transport=self.transport, to_l1=self._denoise_l1
        )
        t.kv_migration_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Stage 2: denoise loop (on-device fp32 Euler) ----
        ah = self.action_horizon
        ah_padded = self._action_horizon_padded

        # Initial noise: padded to ah_padded, only [:ah] is real (rest zero).
        # Refresh the persistent x_t_fp32 buffer in-place via copy_host_to_device_tensor
        # so the buffer ID stays stable (required for trace-replay compatibility).
        noise_pad = torch.zeros(1, ah_padded, self.action_dim, dtype=torch.float32)
        noise_pad[:, :ah, :] = torch.randn(1, ah, self.action_dim)
        host_noise = ttnn.from_torch(noise_pad, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_noise, self.x_t_fp32)

        # Run the device-side denoise loop. Same code path used by capture_trace.
        self._run_denoise_loop_device(prefix_kv_per_chip, upstream, timings=t)

        # ---- Final readback: pull x_t_fp32 to host and slice to logical action_horizon ----
        x_t_final = ttnn.to_torch(self.x_t_fp32)
        actions = x_t_final[:, :ah, :]

        t.total_ms = (time.perf_counter() - wall_start) * 1000.0
        return actions, t

    def _run_denoise_loop_device(self, prefix_kv_per_chip, upstream, timings=None):
        """Pure-device denoise loop. self.x_t_fp32 is read+written in-place each step
        so its Tensor ID stays stable across the loop (trace-replay-safe).

        No ttnn.from_torch, no ttnn.to_torch — every input is a persistent
        ttnn.Tensor on the right chip. Suitable for begin_trace_capture.
        """
        for i in range(self.num_denoising_steps):
            dt = self._dts[i]
            step_start = time.perf_counter() if timings is not None else 0.0

            # Cast persistent fp32 x_t → bf16 for the embed_actions matmul.
            x_t_bf16 = ttnn.typecast(self.x_t_fp32, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            suffix_hidden = self.suffix_slices[0].embed_actions(x_t_bf16)
            ttnn.deallocate(x_t_bf16)

            adarms_per_chip = self._adarms_per_step_per_chip[i]

            expert_out = self.stage_denoise.run_expert_chain(
                suffix_hidden,
                adarms_per_chip,
                prefix_kv_per_chip,
                attention_mask=None,
                position_ids=None,
                per_chip_attn_mask=(upstream["expert_attn_mask"] if upstream is not None else None),
                per_chip_cos=(upstream["suffix_cos"] if upstream is not None else None),
                per_chip_sin=(upstream["suffix_sin"] if upstream is not None else None),
            )

            velocity_bf16 = self._apply_final_norm_and_project(expert_out, adarms_per_chip[-1])

            # On-device Euler: x_t_fp32 ← x_t_fp32 + dt·velocity_fp32, IN-PLACE.
            # Ship velocity from denoise[-1] to denoise[0] via socket, then
            # typecast→mul→add. The final add writes back into self.x_t_fp32
            # via output_tensor= so the buffer's Tensor ID survives the loop.
            velocity_on_chip0 = self.transport.send(velocity_bf16, self.h.denoise_per_chip[0], tag="velocity_wrap")
            ttnn.deallocate(velocity_bf16)
            velocity_fp32 = ttnn.typecast(velocity_on_chip0, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
            velocity_scaled = ttnn.mul(velocity_fp32, dt, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(velocity_fp32)
            ttnn.add(self.x_t_fp32, velocity_scaled, output_tensor=self.x_t_fp32)
            ttnn.deallocate(velocity_scaled)

            if timings is not None:
                timings.denoise_step_ms.append((time.perf_counter() - step_start) * 1000.0)
        if timings is not None and not timings.denoise_step_ms:
            # Placeholder for trace-capture path which doesn't time per step.
            timings.denoise_step_ms = [0.0] * self.num_denoising_steps
