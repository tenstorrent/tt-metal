# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Main PI0 model - TTNN Implementation (Inference Only)

This module assembles all PI0 components into a complete model:
    - PrefixEmbedding: Images + language → embeddings
    - SuffixEmbedding: State + actions + timestep → embeddings
    - PaliGemmaBackbone: VLM + Action Expert transformers

Architecture:
    1. Process images through SigLIP vision tower
    2. Embed language tokens through Gemma embeddings
    3. Concatenate to form prefix embeddings
    4. Prefill prefix, cache KV, denoise actions iteratively

Optimizations:
    1. Pre-computed timesteps
    2. Denoising loop stays entirely on device
    3. Single transfer at the end for final actions
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import ttnn

from models.experimental.pi0.common.configs import (
    PI0ModelConfig,
    PrefixConfig,
    SuffixConfig,
    PaliGemmaConfig,
    DenoiseConfig,
)
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from .ttnn_prefix import PrefixEmbeddingTTNN
from .ttnn_suffix import SuffixEmbeddingTTNN, convert_suffix_weights_to_ttnn
from .ttnn_paligemma import PaliGemmaBackboneTTNN


# openpi's additive-mask "neg inf" — see torch_pi0_model for the same constant.
# finfo.min overflows bfloat16 during the torch->ttnn cast and produces NaN.
_ATT_MASK_NEG_INF = -2.3819763e38


def _compute_prefix_padding_info(
    prefix_pad_ttnn: ttnn.Tensor,
    suffix_len: int,
    suffix_att_pattern: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> Tuple[int, Optional[ttnn.Tensor], Optional[ttnn.Tensor]]:
    """Compute prefix_offset, VLM attention mask, and expert cross-attention mask.

    Returns (prefix_offset, mask_vlm, mask_expert).

    mask_vlm: shape [1, 1, prefix_len, prefix_len] — used in forward_vlm so
        padded prefix positions do not contribute to the KV cache.
    mask_expert: shape [1, 1, suffix_len, prefix_len+suffix_len] — used in
        forward_expert for the suffix cross-attention over the cached prefix.

    Both masks are `None` when the prefix has no padding (all-zeros ≡ no mask).
    """
    prefix_pad = ttnn.to_torch(prefix_pad_ttnn)
    # Normalize to bool (B, L)
    prefix_pad = prefix_pad.reshape(prefix_pad.shape[0], -1).bool()
    prefix_len = prefix_pad.shape[1]
    prefix_offset = int(prefix_pad.long().sum(dim=1)[0].item())

    num_pad = prefix_len - prefix_offset
    if num_pad == 0:
        # No padding → all-zeros additive masks ≡ None for SDPA.
        return prefix_offset, None, None

    def _to_ttnn(t):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ----- VLM prefix self-attention mask [1, 1, prefix_len, prefix_len] -----
    # Prefix uses all-zero att pattern (bidirectional). Mask only hides padded
    # positions from being attended to (pad_2d).
    prefix_att_zeros = torch.zeros(prefix_pad.shape, dtype=torch.bool)
    cumsum_pre = torch.cumsum(prefix_att_zeros.long(), dim=1)
    att_2d_pre = cumsum_pre[:, None, :] <= cumsum_pre[:, :, None]
    pad_2d_pre = prefix_pad[:, None, :] * prefix_pad[:, :, None]
    prefix_2d = att_2d_pre & pad_2d_pre  # [1, prefix_len, prefix_len]
    add_pre = torch.zeros_like(prefix_2d, dtype=torch.float32)
    add_pre.masked_fill_(~prefix_2d, _ATT_MASK_NEG_INF)
    add_pre = add_pre.unsqueeze(1)  # [1, 1, prefix_len, prefix_len]
    mask_vlm = _to_ttnn(add_pre)

    # ----- Expert suffix cross-attention mask [1, 1, suffix_len, prefix_len+suffix_len] -----
    suffix_pad = torch.ones((1, suffix_len), dtype=torch.bool)
    suffix_att_bool = suffix_att_pattern.reshape(1, suffix_len).bool()
    full_pad = torch.cat([prefix_pad, suffix_pad], dim=1)
    full_att = torch.cat([prefix_att_zeros, suffix_att_bool], dim=1)
    cumsum = torch.cumsum(full_att.long(), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = full_pad[:, None, :] * full_pad[:, :, None]
    full_2d = att_2d & pad_2d
    suffix_2d = full_2d[:, -suffix_len:, :]  # [1, suffix_len, prefix_len+suffix_len]
    additive = torch.zeros_like(suffix_2d, dtype=torch.float32)
    additive.masked_fill_(~suffix_2d, _ATT_MASK_NEG_INF)
    additive = additive.unsqueeze(1)  # [1, 1, suffix_len, total_len]
    mask_expert = _to_ttnn(additive)

    return prefix_offset, mask_vlm, mask_expert


class PI0ModelTTNN:
    """
    Complete PI0 model implementation using TTNN.

    Maximizes execution on Tenstorrent hardware while keeping
    control flow and preprocessing on host.
    """

    def __init__(
        self,
        config: PI0ModelConfig,
        weight_loader: PI0WeightLoader,
        device: ttnn.Device,
    ):
        """
        Initialize PI0 model with TTNN.

        Args:
            config: Model configuration
            weight_loader: Loaded weights
            device: TTNN device
        """
        self.config = config
        self.weight_loader = weight_loader
        self.device = device

        # Initialize denoising config
        self.denoise_config = DenoiseConfig(
            num_steps=config.num_denoising_steps,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
        )

        pad_steps = ((self.denoise_config.num_steps + 31) // 32) * 32

        # Create timestep indices on device using ttnn.arange
        self.timestep_indices = ttnn.arange(0, pad_steps, 1, device=self.device, dtype=ttnn.bfloat16)

        # Pre-compute all timestep tensors for the denoising loop.
        # We compute the full sinusoidal embedding in float32 on host and cast
        # to bf16 for device storage. This avoids the bf16 precision loss that
        # occurs when multiplying large scaling factors (up to ~1570) by t in
        # bf16 on device — at t=1.0 the highest-frequency component has
        # scaling_factor≈1570 which rounds to 1568 in bf16, giving sin(1568)
        # instead of sin(1570.8), a difference of ~2.8 rad that completely
        # corrupts the high-frequency sinusoidal components.
        num_steps = self.denoise_config.num_steps
        self._precomputed_timesteps = []
        _expert_width = config.expert_config.width
        _min_period, _max_period = 4e-3, 4.0
        import math as _math

        _half = _expert_width // 2
        _fraction = torch.linspace(0.0, 1.0, _half, dtype=torch.float32)
        _period = _min_period * (_max_period / _min_period) ** _fraction
        _scaling = (1.0 / _period) * 2 * _math.pi  # shape [half_dim], float32
        for i in range(num_steps):
            t_val = 1.0 - i / num_steps
            sin_input = _scaling * t_val  # float32 multiplication — no precision loss
            time_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=0)  # [expert_width]
            time_emb = time_emb.unsqueeze(0)  # [1, expert_width]
            t_ttnn = ttnn.from_torch(
                time_emb,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._precomputed_timesteps.append(t_ttnn)

        x_t_torch = torch.randn(1, self.config.action_horizon, self.config.action_dim)
        self.x_t_ttnn = ttnn.from_torch(
            x_t_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Suffix attention-mask pattern for Pi0.5: [1, 0, 0, ..., 0]
        # (first action token has att=1, rest have att=0). Used to build the
        # additive cross-attention mask for real-padding LIBERO inputs.
        if self.config.pi05:
            self._suffix_att_pattern = torch.zeros(self.config.action_horizon, dtype=torch.bool)
            self._suffix_att_pattern[0] = True
        else:
            # Pi0 has an extra state token at position 0 with att=1,
            # then action_time token att=1, then zeros.
            self._suffix_att_pattern = torch.zeros(self.config.action_horizon + 1, dtype=torch.bool)
            self._suffix_att_pattern[0] = True
            self._suffix_att_pattern[1] = True

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all model components."""
        # Suffix embedding with TTNN weights
        suffix_config = SuffixConfig(
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            expert_width=self.config.expert_config.width,
            pi05=self.config.pi05,
        )
        pi0_weights = self.weight_loader.get_pi0_projections()

        # Convert weights to TTNN
        ttnn_weights = convert_suffix_weights_to_ttnn(pi0_weights, self.device)
        self.suffix_embedding = SuffixEmbeddingTTNN(suffix_config, ttnn_weights, self.device)

        # Backbone
        paligemma_config = PaliGemmaConfig(
            vlm_config=self.config.vlm_config,
            expert_config=self.config.expert_config,
            siglip_config=self.config.siglip_config,
            max_seq_len=self.config.max_seq_len,
        )
        weights = self.weight_loader.categorized_weights
        self.backbone = PaliGemmaBackboneTTNN(paligemma_config, weights, self.device)

        # Prefix embedding with backbone functions
        prefix_config = PrefixConfig(
            vlm_hidden_size=self.config.vlm_config.width,
            num_image_tokens=self.config.siglip_config.num_patches,
        )
        self.prefix_embedding = PrefixEmbeddingTTNN(
            prefix_config,
            self.device,
            embed_image_fn=self.backbone.embed_image,
            embed_language_fn=self.backbone.embed_language_tokens,
        )

    def embed_prefix(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Embed prefix (images + language) using TTNN.

        Args:
            images: List of input images (PyTorch)
            img_masks: Image validity masks (PyTorch)
            lang_tokens: Language token IDs (TTNN)
            lang_masks: Language masks (TTNN)

        Returns:
            Tuple of (embeddings, padding_mask, attention_mask) as TTNN tensors
        """
        return self.prefix_embedding.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    def embed_suffix(
        self,
        state: ttnn.Tensor,
        noisy_actions: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, Optional[ttnn.Tensor]]:
        """
        Embed suffix (state + noisy actions + timestep) using TTNN.

        Args:
            state: Robot state (TTNN)
            noisy_actions: Noisy actions (TTNN)
            timestep: Diffusion timestep (TTNN)

        Returns:
            Tuple of (embeddings, padding_mask, attention_mask, adarms_cond)
        """
        return self.suffix_embedding.embed_suffix(state, noisy_actions, timestep)

    def sample_actions(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample actions via denoising (TTNN inference).

        This runs the full denoising loop:
        1. Compute prefix embeddings (images + language) once
        2. Forward prefix through VLM and cache KV
        3. For each denoising step: compute suffix, forward through expert with cached KV

        Args:
            images: Input images (PyTorch)
            img_masks: Image masks (PyTorch)
            lang_tokens: Language tokens (PyTorch)
            lang_masks: Language masks (PyTorch)
            state: Robot state (PyTorch)

        Returns:
            Sampled actions (PyTorch)
        """
        batch_size = lang_tokens.shape[0]

        # Convert inputs to TTNN
        lang_tokens_ttnn = lang_tokens
        lang_masks_ttnn = lang_masks
        state_ttnn = state

        # Step 1: Embed prefix (images + language) using TTNN
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens_ttnn, lang_masks_ttnn)

        # Compute prefix_offset, VLM attention mask, and expert cross-attention
        # mask in one host sync of the small prefix_pad tensor (≤736 bools).
        # Both masks are None when the prefix has no padding.
        suffix_len = self._suffix_att_pattern.numel()
        prefix_offset, mask_vlm, mask_expert = _compute_prefix_padding_info(
            prefix_pad,
            suffix_len=suffix_len,
            suffix_att_pattern=self._suffix_att_pattern,
            device=self.device,
        )

        # Step 2: Forward prefix through VLM with proper padding mask and cache KV
        _, prefix_kv_cache = self.backbone.forward_vlm(prefix_embs, attention_mask=mask_vlm, use_cache=True)

        # Get timesteps using pure Python list (for control flow on host)
        num_steps = self.denoise_config.num_steps
        # Create timesteps as Python list: [1.0, 0.9, 0.8, ..., 0.0]
        timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]

        # Step 3: Sample initial noise
        x_t_ttnn = self.x_t_ttnn

        # Step 4: Denoising loop (stays on device!)
        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            # Use pre-computed timestep tensor (no slice/reshape per step)
            t_tensor = self._precomputed_timesteps[i]

            # Embed suffix (x_t_ttnn already on device - no transfer!)
            suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(state_ttnn, x_t_ttnn, t_tensor)

            # Forward through expert with cached prefix KV (pass adarms_cond for Pi0.5)
            expert_output, _ = self.backbone.forward_expert(
                suffix_embs,
                attention_mask=mask_expert,
                past_key_values=prefix_kv_cache,
                adarms_cond=adarms_cond,
                prefix_offset=prefix_offset,
            )

            # Extract action output (skip state token in PI0 mode)
            if not self.config.pi05:
                action_output = ttnn.slice(
                    expert_output, [0, 1, 0], [expert_output.shape[0], expert_output.shape[1], expert_output.shape[2]]
                )
            else:
                action_output = expert_output

            # Project to velocity
            velocity = self.suffix_embedding.project_output(action_output)

            # Euler step ON DEVICE (no transfer per step!)
            velocity_scaled = ttnn.mul(velocity, dt)
            x_t_ttnn = ttnn.add(x_t_ttnn, velocity_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)

            # Clear profiler buffer after each denoising step (~500 ops)
            # ReadDeviceProfiler removed for performance

        # Convert back to PyTorch only at the very end (1 transfer instead of 10!)
        return x_t_ttnn

    def _run_denoising_loop(
        self,
        state_ttnn: ttnn.Tensor,
        x_t_ttnn: ttnn.Tensor,
        prefix_kv_cache,
        prefix_offset: int = 0,
        mask_expert: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run the 10-step denoising loop. Factored out for trace capture."""
        num_steps = self.denoise_config.num_steps
        timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]

        for i in range(num_steps):
            dt = timesteps[i + 1] - timesteps[i]
            t_tensor = self._precomputed_timesteps[i]

            suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(state_ttnn, x_t_ttnn, t_tensor)

            expert_output, _ = self.backbone.forward_expert(
                suffix_embs,
                attention_mask=mask_expert,
                past_key_values=prefix_kv_cache,
                adarms_cond=adarms_cond,
                prefix_offset=prefix_offset,
            )

            if not self.config.pi05:
                action_output = ttnn.slice(
                    expert_output, [0, 1, 0], [expert_output.shape[0], expert_output.shape[1], expert_output.shape[2]]
                )
            else:
                action_output = expert_output

            velocity = self.suffix_embedding.project_output(action_output)
            velocity_scaled = ttnn.mul(velocity, dt)
            x_t_ttnn = ttnn.add(x_t_ttnn, velocity_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)

        return x_t_ttnn

    def setup_trace(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ):
        """
        Set up 2CQ + Trace for the denoising loop.

        Call this once to compile and capture. Then call execute_trace() for fast inference.
        """
        # Step 1: Run prefix (not traced — runs once per new observation)
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        suffix_len = self._suffix_att_pattern.numel()
        prefix_offset, mask_vlm, mask_expert = _compute_prefix_padding_info(
            prefix_pad,
            suffix_len=suffix_len,
            suffix_att_pattern=self._suffix_att_pattern,
            device=self.device,
        )
        self._trace_prefix_offset = prefix_offset
        self._trace_mask_expert = mask_expert

        _, self._trace_prefix_kv_cache = self.backbone.forward_vlm(prefix_embs, attention_mask=mask_vlm, use_cache=True)

        # Step 2: Compile pass — run denoising loop to JIT compile all kernels
        x_t_compile = self.x_t_ttnn
        self._run_denoising_loop(
            state,
            x_t_compile,
            self._trace_prefix_kv_cache,
            prefix_offset=prefix_offset,
            mask_expert=mask_expert,
        )

        # Step 3: Capture trace — re-run denoising loop under trace capture
        # The x_t tensor at self.x_t_ttnn address will be the input
        self._trace_x_t = self.x_t_ttnn
        self._trace_state = state

        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._trace_output = self._run_denoising_loop(
            self._trace_state,
            self._trace_x_t,
            self._trace_prefix_kv_cache,
            prefix_offset=prefix_offset,
            mask_expert=mask_expert,
        )
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)

        self._trace_id = trace_id

    def execute_trace(self) -> ttnn.Tensor:
        """Execute the captured denoising trace. Call setup_trace() first."""
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=True)
        return self._trace_output

    def sample_actions_traced(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample actions using 2CQ + Trace for the denoising loop.

        First call sets up the trace. Subsequent calls execute it.
        The prefix (SigLIP + VLM) runs normally each time.
        """
        # Run prefix (new observation each time)
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        suffix_len = self._suffix_att_pattern.numel()
        prefix_offset, mask_vlm, mask_expert = _compute_prefix_padding_info(
            prefix_pad,
            suffix_len=suffix_len,
            suffix_att_pattern=self._suffix_att_pattern,
            device=self.device,
        )

        _, prefix_kv_cache = self.backbone.forward_vlm(prefix_embs, attention_mask=mask_vlm, use_cache=True)

        if not hasattr(self, "_trace_id"):
            # First call: compile + capture trace
            # Compile pass
            x_t_compile = self.x_t_ttnn
            self._run_denoising_loop(
                state,
                x_t_compile,
                prefix_kv_cache,
                prefix_offset=prefix_offset,
                mask_expert=mask_expert,
            )

            # Store references for trace
            self._trace_prefix_kv_cache = prefix_kv_cache
            self._trace_prefix_offset = prefix_offset
            self._trace_mask_expert = mask_expert
            self._trace_x_t = self.x_t_ttnn
            self._trace_state = state

            # Capture trace
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            self._trace_output = self._run_denoising_loop(
                self._trace_state,
                self._trace_x_t,
                self._trace_prefix_kv_cache,
                prefix_offset=prefix_offset,
                mask_expert=mask_expert,
            )
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            self._trace_id = trace_id
        else:
            # Update prefix KV cache in-place at the same tensor addresses.
            # Copy new KV cache values to the trace's pre-allocated KV tensors.
            for i in range(len(prefix_kv_cache)):
                new_k, new_v = prefix_kv_cache[i]
                old_k, old_v = self._trace_prefix_kv_cache[i]
                ttnn.copy(new_k, old_k)
                ttnn.copy(new_v, old_v)

            # Trace captures prefix_offset as a Python int baked into
            # ttnn.slice coordinates. If the non-pad prefix length changes
            # across calls, the trace is no longer valid and callers must
            # release + recapture. Fail loudly instead of returning garbage.
            if prefix_offset != self._trace_prefix_offset:
                raise RuntimeError(
                    f"sample_actions_traced: prefix_offset changed "
                    f"({self._trace_prefix_offset} -> {prefix_offset}). "
                    f"Call release_trace() before sampling with a new token count."
                )
            # If a 4D additive mask was captured, update its values in-place.
            if (mask_expert is None) != (self._trace_mask_expert is None):
                raise RuntimeError(
                    "sample_actions_traced: expert attention mask toggled between "
                    "None and non-None. Call release_trace() before re-invoking."
                )
            if mask_expert is not None:
                ttnn.copy(mask_expert, self._trace_mask_expert)
                ttnn.deallocate(mask_expert)

        # Execute traced denoising loop
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=True)
        return self._trace_output

    def release_trace(self):
        """Release trace resources."""
        if hasattr(self, "_trace_id"):
            ttnn.release_trace(self.device, self._trace_id)
            del self._trace_id

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: ttnn.Device,
        config: Optional[PI0ModelConfig] = None,
    ) -> "PI0ModelTTNN":
        """
        Load pretrained PI0 model to TTNN device.

        Args:
            model_path: Path to model or HuggingFace model ID
            device: TTNN device
            config: Optional configuration override

        Returns:
            Loaded PI0 model
        """
        weight_loader = PI0WeightLoader(model_path)

        if config is None:
            config = PI0ModelConfig(
                action_dim=weight_loader.config.action_dim,
                action_horizon=weight_loader.config.action_horizon,
            )

        return cls(config, weight_loader, device)


# Default export
PI0Model = PI0ModelTTNN
