# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pi0_5GLX1x8Pipeline — pi0.5 sample_actions on a single 1×8 mesh.

Three stages share the parent 1×8 mesh and run sequentially. No host-bounce
between stages and no fabric mesh sockets — all stage-to-stage data movement
is on-device CCL (ttnn.all_gather inside SigLIP DP, ttnn.all_reduce inside
StagePrefillTP4 MLP). Submeshes do not share tensor allocations, so keeping
every stage on the same mesh handle is what makes CCL handoff possible.

Stages:
    0. SigLIP DP — 3 real cameras + 5 zero dummies batch-sharded along dim 0
       (8 cameras, one per chip). Each chip runs the full SigLIP encoder
       (SigLIPCameraSlice) at bs=1. ttnn.all_gather(dim=0) replicates the
       (8, 256, 2048) output on every chip; ttnn.slice drops the 5 dummies.
    1. Prefill TP=8 — StagePrefillTP4 (col/row-parallel MLP, replicated attn,
       all_reduce per MLP block). Output (final_hidden, KV×18) replicated on
       all 8 chips.
    2. Denoise (replicated) — one ExpertChunkSlice(layer_range=(0, 18)) with
       implicit-replicated weights. 7 chips do redundant compute but
       latency = 1-chip; chip 0's copy is the output. 5-step Euler loop
       unrolled into the trace.

Trace machinery mirrors Pi0_5GLXPipeline.sample_actions_traced
(pipeline.py:568): pre-allocate persistent input buffers (pixel_values_buf,
lang_tokens_buf, x_t_fp32), eager warmup to JIT kernels, then a single
begin/end_trace_capture on the parent mesh. Replay = copy_host_to_device_tensor
into the buffers + ttnn.execute_trace + ttnn.to_torch on the output buffer.

Layout assumptions:
    * Parent mesh: 1×8 (open via mesh_setup.open_prefill_tp4_mesh(tp=8)).
    * All weights replicated EXCEPT prefill MLP (sharded by StagePrefillTP4).
    * num_cameras = 3 (production); padded to 8 for DP.
    * Denoise output replicated → ConcatMeshToTensor + [:1] slice to take chip 0.

Not in scope for v1 (TODO if needed):
    * PI0_UPSTREAM_MASKS path (per-layer attn masks + position-aware RoPE).
    * Per-stage timing breakdown.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import ttnn

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig, SuffixConfig
from models.experimental.pi0_5.tt.ttnn_gemma import ada_rms_norm_no_gate_ttnn

from .expert_slice import ExpertChunkSlice
from .pipeline import _DenoiseHead, _PrefillHead
from .stage_prefill_tp4 import StagePrefillTP4
from .suffix_slice import SuffixSlice
from .vision_slice import SigLIPCameraSlice


_NUM_CHIPS_REQUIRED = 8
_NUM_REAL_CAMS = 3
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
        self.prefill = StagePrefillTP4(config, weights, mesh)

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
                device=mesh,
            )
            cond = self.suffix.embed_adarms_cond(t_ttnn)
            ttnn.deallocate(t_ttnn)
            self._adarms_per_step.append(cond)

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
        self._num_real_cams = _NUM_REAL_CAMS

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

        Same logic as Pi0_5GLXPipeline._build_prefix (pipeline.py:393) but on
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
    ) -> "ttnn.Tensor":
        normed = ada_rms_norm_no_gate_ttnn(
            expert_out,
            adarms_cond,
            self.denoise_head.mod_weight,
            self.denoise_head.mod_bias,
            self.config.expert_config.rms_norm_eps,
            self.denoise_head.core_grid,
        )
        return self.suffix.project_output(normed)

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
        _final_hidden, per_layer_kv = self.prefill.run(prefix_embs)
        ttnn.deallocate(prefix_embs)

        # ---- Stage 2: 5-step Euler denoise (replicated on all 8 chips) ----
        for i in range(self.num_denoising_steps):
            dt = self._dts[i]
            adarms_cond = self._adarms_per_step[i]

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
            )
            ttnn.deallocate(suffix_hidden)

            velocity_bf16 = self._apply_final_norm_and_project(expert_out, adarms_cond)
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

    # ──────────── Trace capture + replay ────────────────────────────────

    def capture_trace(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
    ) -> None:
        """One-time setup: allocate persistent buffers, JIT-compile every
        kernel via an eager warmup, then capture the full _sample_actions_device
        body as a TTNN trace on the parent mesh's CQ 0. After this completes,
        sample_actions_traced() can be called repeatedly with new inputs.
        """
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
