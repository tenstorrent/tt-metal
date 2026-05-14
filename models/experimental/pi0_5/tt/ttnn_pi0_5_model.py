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
        self._precompute_bs1_modulations()
        # SDPA mask is built lazily on first sample_actions call (depends on
        # actual prefix kv length, which is fixed across replays).
        self._sdpa_attn_mask: Optional["ttnn.Tensor"] = None
        self._sdpa_mask_kv_len: int = 0

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

    def _precompute_bs1_modulations(self) -> None:
        """
        OPTIMIZATION (TIER A): adarms_cond is deterministic per step, so the
        per-block fused modulation Dense (W -> 6*W) and the final-norm Dense
        (W -> 3*W) all produce constant outputs. Precompute them at init and
        reuse across every inference call.

        Saves per inference:
          - 18 layers × 10 steps = 180 modulation matmuls  (~8 ms)
          - 360 (1→32) row tilizes that fed those matmuls   (~3 ms)
          - 360 scale_plus_one adds inside _modulated_rms_norm (we also
            pre-add the +1 here)
        """
        import ttnn as _ttnn
        from models.experimental.pi0_5.tt.ttnn_gemma import _split_modulation_6

        num_steps = self.denoise_config.num_steps
        blocks = self.backbone.expert_blocks
        # [step][layer] -> (sa1, ta, ga, sf1, tf, gf)
        self._block_mods_per_step: List[List[tuple]] = []
        # [step] -> (sf1_final, tf_final)
        self._final_mod_per_step: List[tuple] = []

        for step_idx in range(num_steps):
            cond = self._adarms_cond_per_step_bs1[step_idx]

            per_layer: List[tuple] = []
            for block in blocks:
                mod = _ttnn.linear(
                    cond,
                    block.mod_weight,
                    bias=block.mod_bias,
                    memory_config=_ttnn.DRAM_MEMORY_CONFIG,
                    core_grid=block.core_grid,
                    compute_kernel_config=block.mod_compute_kernel_config,
                )
                sa, ta, ga, sf, tf, gf = _split_modulation_6(mod)
                # Pre-add 1 to the scale tensors so rms_norm uses them directly.
                sa1 = _ttnn.add(sa, 1.0, memory_config=_ttnn.DRAM_MEMORY_CONFIG)
                sf1 = _ttnn.add(sf, 1.0, memory_config=_ttnn.DRAM_MEMORY_CONFIG)
                _ttnn.deallocate(sa)
                _ttnn.deallocate(sf)
                _ttnn.deallocate(mod)
                per_layer.append((sa1, ta, ga, sf1, tf, gf))
            self._block_mods_per_step.append(per_layer)

            # Final norm: separate Dense weight (3*W).
            mod_w = self.backbone.expert_final_norm_mod_weight
            mod_b = self.backbone.expert_final_norm_mod_bias
            mod = _ttnn.linear(
                cond,
                mod_w,
                bias=mod_b,
                memory_config=_ttnn.DRAM_MEMORY_CONFIG,
                core_grid=self.backbone.core_grid,
            )
            B = mod.shape[0]
            width3 = mod.shape[-1]
            width = width3 // 3
            mod3 = _ttnn.reshape(mod, (B, 1, width3))
            _ttnn.deallocate(mod)
            scale = mod3[:, :, 0:width]
            shift = mod3[:, :, width : 2 * width]
            # gate is discarded for final norm (no_gate variant).
            scale1 = _ttnn.add(scale, 1.0, memory_config=_ttnn.DRAM_MEMORY_CONFIG)
            _ttnn.deallocate(scale)
            _ttnn.deallocate(mod3)
            self._final_mod_per_step.append((scale1, shift))

    def _build_sdpa_phantom_mask(self, prefix_kv_len_logical: int) -> "ttnn.Tensor":
        """
        Build the SDPA attention mask used in the expert's keep_padded path.

        Math: the suffix is processed as logical=physical=64 (padded action_horizon).
        The prefix kv-cache is lifted to logical=physical=ceil_to_tile(prefix_len).
        The 16 prefix phantom positions and 14 suffix phantom positions in K must
        be masked out so attention to them doesn't pollute valid Q positions.

        Mask shape: (1, 1, 64, kv_len_total). Values: 0 = attend, -inf = mask.
        """
        action_horizon = self.config.action_horizon
        q_len_padded = ((action_horizon + 31) // 32) * 32
        prefix_padded = ((prefix_kv_len_logical + 31) // 32) * 32
        kv_total = prefix_padded + q_len_padded

        mask = torch.zeros(1, 1, q_len_padded, kv_total, dtype=torch.bfloat16)
        # Prefix phantom positions
        if prefix_padded > prefix_kv_len_logical:
            mask[:, :, :, prefix_kv_len_logical:prefix_padded] = float("-inf")
        # Suffix phantom positions
        if q_len_padded > action_horizon:
            suffix_phantom_start = prefix_padded + action_horizon
            mask[:, :, :, suffix_phantom_start:kv_total] = float("-inf")

        return ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

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

        # OPTIMIZATION (keep_padded): lift the prefix kv-cache to logical=physical
        # by zeroing the implicit tile padding. This eliminates the 360
        # UntilizeWithUnpadding(272→272) ops in the per-layer suffix-attention
        # concat path (~3 ms / chunk). The phantom positions become valid K/V
        # rows of zeros, which the SDPA mask zeros out.
        keep_padded_expert = batch_size == 1
        # Keep a reference to the original prefix kv-cache list so its storage
        # isn't reclaimed before any aliased lifted tensors are used.
        _prefix_kv_cache_original = prefix_kv_cache
        if keep_padded_expert and prefix_kv_cache is not None:
            prefix_kv_cache = [
                (
                    ttnn.fill_implicit_tile_padding(k, 0.0),
                    ttnn.fill_implicit_tile_padding(v, 0.0),
                )
                for k, v in prefix_kv_cache
            ]

            # Build the SDPA mask if not yet built (size depends on prefix_len).
            prefix_logical = prefix_kv_cache[0][0].shape[2]  # post-fill: logical=physical
            # Recover original (pre-lift) logical for mask construction. After
            # fill_implicit_tile_padding the logical==physical, but the *real*
            # token positions are the first `prefix_kv_len_logical` rows. We
            # cache by the padded size so trace replay reuses the same mask.
            if self._sdpa_attn_mask is None or self._sdpa_mask_kv_len != prefix_logical:
                # The original logical length is prefix_embs.shape[1] (pre-fill).
                orig_prefix_len = prefix_embs.shape[1]
                self._sdpa_attn_mask = self._build_sdpa_phantom_mask(orig_prefix_len)
                self._sdpa_mask_kv_len = prefix_logical

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

        # OPTIMIZATION (keep_padded): lift x_t LOGICAL shape from action_horizon
        # to the next tile-aligned size (50 → 64) with zeros in the new rows.
        # ttnn.pad changes the logical shape (fill_implicit_tile_padding only
        # zeros the implicit padding region — see fill_pad_device_operation.cpp
        # :22-26). The 14 phantom rows are masked out of SDPA by the prebuilt
        # attn mask, so they're benign mathematically.
        if keep_padded_expert:
            ah = self.config.action_horizon
            ah_padded = ((ah + 31) // 32) * 32
            if ah_padded > ah:
                x_t_ttnn = ttnn.pad(
                    x_t_ttnn,
                    padding=((0, 0), (0, ah_padded - ah), (0, 0)),
                    value=0.0,
                )

        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            if fast_path:
                # OPTIMIZATION: adarms_cond is precomputed at init (deterministic
                # per step); only the action embedding depends on x_t.
                suffix_embs = self.suffix_embedding.embed_actions(x_t_ttnn)
                adarms_cond = self._adarms_cond_per_step_bs1[i]
                precomputed_block_mods = self._block_mods_per_step[i]
                precomputed_final_mod = self._final_mod_per_step[i]
            else:
                assert timesteps_ttnn is not None
                t_tensor = ttnn.slice(timesteps_ttnn, [0, i], [batch_size, i + 1])
                t_tensor = ttnn.reshape(t_tensor, (batch_size,))
                suffix_embs, _, _, adarms_cond = self.embed_suffix(state, x_t_ttnn, t_tensor)
                precomputed_block_mods = None
                precomputed_final_mod = None

            expert_output, _ = self.backbone.forward_expert(
                suffix_embs,
                adarms_cond=adarms_cond,
                past_key_values=prefix_kv_cache,
                precomputed_block_mods=precomputed_block_mods,
                precomputed_final_mod=precomputed_final_mod,
                attention_mask=self._sdpa_attn_mask if keep_padded_expert else None,
                keep_padded=keep_padded_expert,
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
