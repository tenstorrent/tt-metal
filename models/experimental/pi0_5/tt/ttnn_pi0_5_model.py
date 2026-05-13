# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 model (TTNN, inference).

Same denoising loop as PI0 but:
  - no state token in the suffix,
  - adarms_cond is passed into forward_expert,
  - no state-token slice on the expert output.
"""

from pathlib import Path
from typing import List, Optional, Union

import torch
import ttnn

from models.experimental.pi0_5.common.configs import (
    DenoiseConfig,
    PaliGemmaConfig,
    PrefixConfig,
    SuffixConfig,
)
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader as PI0WeightLoader
from models.experimental.pi0_5.tt.ttnn_prefix import PrefixEmbeddingTTNN

from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.tt.ttnn_suffix import (
    Pi0_5SuffixEmbeddingTTNN,
    convert_pi0_5_suffix_weights_to_ttnn,
)
from models.experimental.pi0_5.tt.ttnn_paligemma import Pi0_5PaliGemmaBackboneTTNN


class Pi0_5ModelTTNN:
    """TTNN PI0.5 model (inference)."""

    def __init__(
        self,
        config: Pi0_5ModelConfig,
        weight_loader: PI0WeightLoader,
        device: "ttnn.Device",
    ):
        assert config.pi05, "Pi0_5ModelTTNN requires config.pi05=True"
        self.config = config
        self.weight_loader = weight_loader
        self.device = device

        self.denoise_config = DenoiseConfig(
            num_steps=config.num_denoising_steps,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
        )

        pad_steps = ((self.denoise_config.num_steps + 31) // 32) * 32
        self.timestep_indices = ttnn.arange(0, pad_steps, 1, device=self.device, dtype=ttnn.bfloat16)

        # Initial noise tensor; resampled fresh on each sample_actions call to
        # match lerobot/openpi reference behavior (see sample_actions below).
        # Allocated once and reused as a destination buffer.
        x_t_torch = torch.randn(1, config.action_horizon, config.action_dim)
        self.x_t_ttnn = ttnn.from_torch(
            x_t_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        self._init_components()
        self._precompute_bs1_timestep_tensors()
        self._precompute_bs1_adarms_cond()

    def _precompute_bs1_timestep_tensors(self) -> None:
        num_steps = self.denoise_config.num_steps
        pad_steps = ((num_steps + 31) // 32) * 32
        idx = ttnn.to_layout(self.timestep_indices, ttnn.TILE_LAYOUT)
        vals = ttnn.multiply(idx, -1.0 / num_steps)
        ttnn.deallocate(idx)
        row = ttnn.add(vals, 1.0)
        ttnn.deallocate(vals)
        self._timesteps_row_ttnn = ttnn.reshape(row, (1, pad_steps))

        self._timestep_per_step_bs1: List["ttnn.Tensor"] = []
        for i in range(num_steps):
            t_i = ttnn.slice(self._timesteps_row_ttnn, [0, i], [1, i + 1])
            self._timestep_per_step_bs1.append(ttnn.reshape(t_i, (1,)))

    def _precompute_bs1_adarms_cond(self) -> None:
        """
        OPTIMIZATION: timesteps are deterministic (linspace 1.0 -> 0.0), so
        `adarms_cond = time_mlp_out(silu(time_mlp_in(sincos(t))))` is constant
        per step. Compute once at init and reuse for every inference call —
        removes sincos + 2 linears + silu from each denoise step.

        Only applicable when batch_size==1 (the dominant inference case).
        """
        num_steps = self.denoise_config.num_steps
        self._adarms_cond_per_step_bs1: List["ttnn.Tensor"] = []
        for i in range(num_steps):
            cond = self.suffix_embedding.embed_adarms_cond(self._timestep_per_step_bs1[i])
            self._adarms_cond_per_step_bs1.append(cond)

    def _init_components(self):
        suffix_config = SuffixConfig(
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            expert_width=self.config.expert_config.width,
            pi05=True,
        )
        pi0_weights = self.weight_loader.get_pi0_projections()
        ttnn_suffix_weights = convert_pi0_5_suffix_weights_to_ttnn(pi0_weights, self.device)
        self.suffix_embedding = Pi0_5SuffixEmbeddingTTNN(suffix_config, ttnn_suffix_weights, self.device)

        paligemma_config = PaliGemmaConfig(
            vlm_config=self.config.vlm_config,
            expert_config=self.config.expert_config,
            siglip_config=self.config.siglip_config,
            max_seq_len=self.config.max_seq_len,
        )
        weights = self.weight_loader.categorized_weights
        self.backbone = Pi0_5PaliGemmaBackboneTTNN(paligemma_config, weights, self.device)

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

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        return self.prefix_embedding.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    def embed_suffix(self, state, noisy_actions, timestep):
        return self.suffix_embedding.embed_suffix(state, noisy_actions, timestep)

    def sample_actions(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> "ttnn.Tensor":
        batch_size = lang_tokens.shape[0]

        prefix_embs, _, _ = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        # ttnn.embedding (called inside embed_prefix for language tokens) returns
        # ROW_MAJOR; ttnn.concat with the TILE image embeddings preserves that.
        # ttnn.rms_norm at the start of every VLM block requires TILE — convert
        # the concatenated prefix here before the VLM stack runs.
        if prefix_embs.layout != ttnn.TILE_LAYOUT:
            prefix_embs = ttnn.to_layout(prefix_embs, ttnn.TILE_LAYOUT)
        _, prefix_kv_cache = self.backbone.forward_vlm(prefix_embs, use_cache=True)

        num_steps = self.denoise_config.num_steps
        timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]

        timesteps_ttnn = None
        if batch_size != 1:
            pad_steps = ((num_steps + 31) // 32) * 32
            idx = ttnn.to_layout(self.timestep_indices, ttnn.TILE_LAYOUT)
            vals = ttnn.multiply(idx, -1.0 / num_steps)
            timesteps_ttnn = ttnn.add(vals, 1.0)
            timesteps_ttnn = ttnn.reshape(timesteps_ttnn, (1, pad_steps))
            ttnn.deallocate(idx)
            ttnn.deallocate(vals)

        # Resample fresh N(0, 1) noise each call — matches lerobot's
        # sample_noise (modeling_pi05.py:618) and the pytorch reference. Reusing
        # one fixed noise tensor across calls (the prior bug) made every chunk
        # in a rollout converge to the same flow-matching attractor, biasing
        # inference toward whatever modes that seed lands near.
        # Tests that need deterministic noise can set `self.resample_noise = False`
        # and pre-populate `self.x_t_ttnn` (see tests/perf/test_denoise_step_accuracy.py).
        if getattr(self, "resample_noise", True):
            fresh_noise = torch.randn(1, self.config.action_horizon, self.config.action_dim)
            x_t_ttnn = ttnn.from_torch(
                fresh_noise,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            x_t_ttnn = self.x_t_ttnn
        fast_path = batch_size == 1

        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            if fast_path:
                # OPTIMIZATION: adarms_cond is precomputed at init (deterministic
                # per step); only the action embedding depends on x_t.
                suffix_embs = self.suffix_embedding.embed_actions(x_t_ttnn)
                adarms_cond = self._adarms_cond_per_step_bs1[i]
            else:
                assert timesteps_ttnn is not None
                t_tensor = ttnn.slice(timesteps_ttnn, [0, i], [batch_size, i + 1])
                t_tensor = ttnn.reshape(t_tensor, (batch_size,))
                suffix_embs, _, _, adarms_cond = self.embed_suffix(state, x_t_ttnn, t_tensor)

            expert_output, _ = self.backbone.forward_expert(
                suffix_embs,
                adarms_cond=adarms_cond,
                past_key_values=prefix_kv_cache,
            )
            ttnn.deallocate(suffix_embs)

            velocity = self.suffix_embedding.project_output(expert_output)
            ttnn.deallocate(expert_output)

            velocity_scaled = ttnn.mul(velocity, dt)
            ttnn.deallocate(velocity)

            x_t_new = ttnn.add(x_t_ttnn, velocity_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(velocity_scaled)
            x_t_ttnn = x_t_new

        return x_t_ttnn

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: "ttnn.Device",
        config: Optional[Pi0_5ModelConfig] = None,
    ) -> "Pi0_5ModelTTNN":
        weight_loader = PI0WeightLoader(model_path)
        if config is None:
            config = Pi0_5ModelConfig(
                action_dim=weight_loader.config.action_dim,
                action_horizon=weight_loader.config.action_horizon,
            )
        return cls(config, weight_loader, device)
