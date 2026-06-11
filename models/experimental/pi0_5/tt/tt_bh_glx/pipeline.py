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

Simplifications vs single-chip ttnn_pi0_5_model.py:
    - batch_size = 1 only
    - No PI0_UPSTREAM_MASKS (attention_mask = None, position_ids = None)
    - No keep_padded / precomputed_mod fast paths
    - Host-side Euler integration (no on-device fp32 accumulator)
    - x_t re-uploaded each step from host
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig, SuffixConfig
from models.experimental.pi0_5.tt.ttnn_gemma import ada_rms_norm_no_gate_ttnn

from .kv_migration import migrate_layer_paired
from .stage_denoise import StageDenoise
from .stage_prefill import StagePrefill
from .stage_vision import StageVision
from .stages import StageTimings
from .suffix_slice import SuffixSlice
from .transport import send_via_host


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

    def embed_lang(self, lang_tokens_torch: torch.Tensor) -> "ttnn.Tensor":
        """Returns (B, lang_len, vlm_hidden) bf16 TILE."""
        tokens_ttnn = ttnn.from_torch(
            lang_tokens_torch.to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.submesh,
        )
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

        # Per-stage executors.
        self.stage_vision = StageVision(config.siglip_config, weights, mesh_handles)
        self.stage_prefill = StagePrefill(config, weights, mesh_handles)
        self.stage_denoise = StageDenoise(config, weights, mesh_handles)

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
        ttnn.synchronize_device(self.h.vision_per_chip[-1])
        t.vision_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Transport 0→1: vision[3] → prefill[0] ----
        t0 = time.perf_counter()
        vision_out_p0 = send_via_host(vision_out, self.h.prefill_per_chip[0])
        ttnn.synchronize_device(self.h.prefill_per_chip[0])
        t.transport_v2p_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Build prefix embeddings + Stage 1: prefill ----
        prefix_embs = self._build_prefix(vision_out_p0, lang_tokens)
        t0 = time.perf_counter()
        _final_hidden, per_layer_kv = self.stage_prefill.run(prefix_embs, attention_mask=None, position_ids=None)
        ttnn.synchronize_device(self.h.prefill_per_chip[-1])
        t.prefill_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Transport 1→2: layer-paired KV migration ----
        t0 = time.perf_counter()
        prefix_kv_per_chip = migrate_layer_paired(per_layer_kv, self.h.denoise_per_chip)
        for chip in self.h.denoise_per_chip:
            ttnn.synchronize_device(chip)
        t.kv_migration_ms = (time.perf_counter() - t0) * 1000.0

        # ---- Stage 2: denoise loop ----
        # x_t starts as N(0,1) noise, host-padded to ah_padded.
        ah = self.action_horizon
        ah_padded = self._action_horizon_padded
        x_t = torch.zeros(1, ah_padded, self.action_dim, dtype=torch.float32)
        x_t[:, :ah, :] = torch.randn(1, ah, self.action_dim)

        timesteps = [1.0 - i / self.num_denoising_steps for i in range(self.num_denoising_steps + 1)]
        t.denoise_step_ms = []
        for i in range(self.num_denoising_steps):
            ts_now = timesteps[i]
            ts_next = timesteps[i + 1]
            dt = ts_next - ts_now

            step_start = time.perf_counter()

            # Upload x_t to denoise[0] (bf16 TILE).
            x_t_ttnn = _upload_torch(
                x_t, self.h.denoise_per_chip[0], dtype=ttnn.bfloat16, mem_cfg=ttnn.L1_MEMORY_CONFIG
            )

            # suffix.embed_actions on chip 0.
            suffix_hidden = self.suffix_slices[0].embed_actions(x_t_ttnn)
            ttnn.deallocate(x_t_ttnn)

            # adarms_cond replicated on each denoise chip.
            adarms_per_chip = self._embed_adarms_per_chip(torch.tensor([ts_now], dtype=torch.float32))

            # 18-layer expert chain across 6 chips.
            expert_out = self.stage_denoise.run_expert_chain(
                suffix_hidden, adarms_per_chip, prefix_kv_per_chip, attention_mask=None, position_ids=None
            )

            # Final adaRMS norm + action_out_proj on last denoise chip.
            velocity_ttnn = self._apply_final_norm_and_project(expert_out, adarms_per_chip[-1])
            ttnn.synchronize_device(self.h.denoise_per_chip[-1])

            # Pull velocity to host and do Euler step in fp32.
            velocity_torch = ttnn.to_torch(velocity_ttnn).to(torch.float32)
            # velocity_torch shape: (1, ah_padded, action_dim)
            x_t = x_t + dt * velocity_torch

            ttnn.deallocate(velocity_ttnn)
            for a in adarms_per_chip:
                ttnn.deallocate(a)

            t.denoise_step_ms.append((time.perf_counter() - step_start) * 1000.0)

        # ---- Slice physical → logical action_horizon ----
        actions = x_t[:, :ah, :]

        t.total_ms = (time.perf_counter() - wall_start) * 1000.0
        return actions, t
