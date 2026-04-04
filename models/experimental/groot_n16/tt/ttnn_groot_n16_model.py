# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GR00T N1.6-3B - Complete TTNN Model Implementation.

Assembles all components:
    1. SigLIP2 vision encoder (27 layers, 1152-dim)
    2. Pixel shuffle connector + MLP projection
    3. Qwen3-1.7B language model backbone (28 layers, 2048-dim, using layer 16)
    4. AlternateVLDiT action head (32 layers, 1536-dim)
    5. Embodiment-conditioned state/action encode/decode MLPs
    6. Flow matching inference loop (4 Euler integration steps)

Inference pipeline:
    images -> SigLIP2 -> pixel_shuffle -> MLP_connector -> Qwen3 (layer 16) -> backbone_features
    state -> state_encoder[embodiment] -> state_features
    for t in range(4):
        noisy_actions -> action_encoder[embodiment] -> action_features
        [state_features; action_features] + backbone_features -> AlternateVLDiT -> velocity
        actions += dt * velocity
    actions -> action_decoder[embodiment] -> predicted_actions
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import ttnn
from models.experimental.groot_n16.common.configs import Gr00tN16Config
from models.experimental.groot_n16.common.weight_loader import Gr00tN16WeightLoader
from models.experimental.groot_n16.tt.ttnn_common import (
    CORE_GRID_BH,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_params,
    to_tt_tensor,
)
from models.experimental.groot_n16.tt.ttnn_siglip2 import SigLIP2VisionEncoderTTNN
from models.experimental.groot_n16.tt.ttnn_dit import AlternateVLDiTTTNN
from models.experimental.groot_n16.tt.ttnn_embodiment import (
    CategorySpecificMLPTTNN,
    MultiEmbodimentActionEncoderTTNN,
)

logger = logging.getLogger(__name__)


class PixelShuffleConnectorTTNN:
    """
    Pixel shuffle downsampling + MLP connector.

    Downsamples vision tokens by ratio 0.5 (2x2 -> 1), then projects
    from 4*vision_dim to language_dim via 2-layer MLP.
    """

    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        connector_weights: Dict[str, torch.Tensor],
        device: Any,
    ):
        self.device = device
        self.vision_dim = vision_dim
        self.language_dim = language_dim

        # 2-layer MLP connector
        # Layer 0: 4*vision_dim -> language_dim (with GELU)
        # Layer 1: language_dim -> language_dim
        # Connector keys after stripping "backbone.model.mlp1." prefix:
        # 0.weight/bias, 1.weight/bias, 3.weight/bias (3-layer MLP, index 2 is GELU)
        for idx in ["0", "1", "3"]:
            w = connector_weights.get(f"{idx}.weight")
            b = connector_weights.get(f"{idx}.bias")
            if w is not None:
                setattr(self, f"proj{idx}_weight", preprocess_linear_weight(w, device))
                setattr(self, f"proj{idx}_bias",
                        preprocess_linear_bias(b, device) if b is not None else None)

    def pixel_shuffle_downsample(self, vision_features: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Pixel shuffle on CPU: [B, h*w, dim] -> [B, h/2 * w/2, 4*dim]

        This is done on CPU since it's a reshape operation before the MLP.
        """
        batch_size, seq_len, dim = vision_features.shape
        # Reshape to spatial: [B, h, w, dim]
        x = vision_features.reshape(batch_size, h, w, dim)
        # Downsample 2x2: [B, h/2, 2, w/2, 2, dim] -> [B, h/2, w/2, 4*dim]
        x = x.reshape(batch_size, h // 2, 2, w // 2, 2, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(batch_size, h // 2 * w // 2, 4 * dim)
        return x

    def __call__(self, vision_features: torch.Tensor, h: int, w: int) -> ttnn.Tensor:
        """
        Pixel shuffle + MLP projection.

        Args:
            vision_features: [batch, num_patches, vision_dim] on CPU
            h, w: spatial dimensions of patches (e.g., 16x16 for 256 patches)

        Returns:
            [batch, num_patches/4, language_dim] on device
        """
        # Pixel shuffle on CPU
        shuffled = self.pixel_shuffle_downsample(vision_features, h, w)
        # [batch, num_patches/4, 4*vision_dim]

        # Transfer to device
        shuffled_tt = to_tt_tensor(shuffled, self.device)

        # MLP layer 0
        h = ttnn.linear(
            shuffled_tt, self.proj0_weight, bias=self.proj0_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(shuffled_tt)

        # MLP layer 1
        h = ttnn.linear(
            h, self.proj1_weight, bias=self.proj1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH,
        )

        # GELU activation (index 2 in the Sequential)
        h = ttnn.gelu(h)

        # MLP layer 3
        output = ttnn.linear(
            h, self.proj3_weight, bias=self.proj3_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(h)

        return output


class Gr00tN16ModelTTNN:
    """
    Complete GR00T N1.6-3B model on TTNN.

    Implements the full VLA inference pipeline including flow matching.
    """

    def __init__(
        self,
        config: Gr00tN16Config,
        weight_loader: Gr00tN16WeightLoader,
        device: ttnn.Device,
    ):
        self.config = config
        self.device = device

        logger.info("Initializing GR00T N1.6-3B on TTNN...")
        t0 = time.time()

        # Load all weights
        weight_loader.load()

        # 1. Vision encoder (SigLIP2)
        logger.info("  Loading SigLIP2 vision encoder...")
        self.vision_encoder = SigLIP2VisionEncoderTTNN(
            config.backbone.vision,
            weight_loader.get_vision_weights(),
            device,
        )

        # 2. Pixel shuffle + MLP connector
        logger.info("  Loading pixel shuffle connector...")
        self.connector = PixelShuffleConnectorTTNN(
            config.backbone.vision.hidden_size,
            config.backbone.language.hidden_size,
            weight_loader.get_connector_weights(),
            device,
        )

        # 3. VL LayerNorm (applied to backbone features before DiT)
        logger.info("  Loading VL LayerNorm...")
        vl_ln_weights = weight_loader.get_vl_layernorm_weights()
        vl_ln_w = vl_ln_weights.get("weight")
        vl_ln_b = vl_ln_weights.get("bias")
        if vl_ln_w is not None:
            self.vl_ln_weight, self.vl_ln_bias = preprocess_layernorm_params(
                vl_ln_w, vl_ln_b, device,
            )

        # 4. AlternateVLDiT action head
        logger.info("  Loading AlternateVLDiT (32 layers)...")
        self.dit = AlternateVLDiTTTNN(
            config.dit,
            weight_loader.get_dit_weights(),
            device,
        )

        # 5. Embodiment MLPs
        logger.info("  Loading embodiment MLPs...")
        emb_cfg = config.embodiment

        self.state_encoder = CategorySpecificMLPTTNN(
            weight_loader.get_state_encoder_weights(),
            emb_cfg.max_num_embodiments,
            emb_cfg.max_state_dim,
            emb_cfg.state_hidden_dim,
            emb_cfg.state_output_dim,
            device,
        )

        self.action_encoder = MultiEmbodimentActionEncoderTTNN(
            weight_loader.get_action_encoder_weights(),
            emb_cfg,
            weight_loader.get_timestep_encoder_weights(),
            device,
        )

        self.action_decoder = CategorySpecificMLPTTNN(
            weight_loader.get_action_decoder_weights(),
            emb_cfg.max_num_embodiments,
            config.hidden_size,  # 1024 (DiT output dim)
            emb_cfg.state_hidden_dim,  # 1024
            emb_cfg.max_action_dim,  # 29
            device,
        )

        # 6. Positional embeddings for action tokens
        pos_embed = weight_loader.get_pos_embed()
        if pos_embed is not None:
            self.pos_embed = to_tt_tensor(pos_embed.unsqueeze(0), device)
        else:
            self.pos_embed = None

        # Note: Qwen3 backbone is handled separately via tt_transformers
        # For now, we compute backbone features on CPU/GPU and transfer
        self._backbone_model = None

        elapsed = time.time() - t0
        logger.info(f"  GR00T N1.6 initialized in {elapsed:.1f}s")

    def encode_vision(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """
        Encode images through SigLIP2 + pixel shuffle + connector.

        Args:
            pixel_values: [batch, 3, H, W]

        Returns:
            [batch, num_image_tokens, language_dim] on device
        """
        # SigLIP2 forward
        vision_features_tt = self.vision_encoder(pixel_values)

        # Transfer back to CPU for pixel shuffle
        vision_features_cpu = ttnn.to_torch(vision_features_tt)
        ttnn.deallocate(vision_features_tt)

        # Compute spatial dims
        num_patches = self.config.backbone.vision.num_patches
        h = w = int(num_patches ** 0.5)

        # Pixel shuffle + MLP connector
        image_tokens = self.connector(vision_features_cpu, h, w)

        return image_tokens

    def encode_backbone(
        self,
        image_tokens: ttnn.Tensor,
        text_tokens: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Run backbone (Qwen3) to get VL features.

        For the initial implementation, this uses the HuggingFace model on CPU/GPU
        and transfers features to device. The Qwen3 backbone will be ported to
        TTNN in a subsequent optimization pass.

        Args:
            image_tokens: [batch, num_image_tokens, language_dim] on device
            text_tokens: [batch, text_seq_len] token IDs

        Returns:
            [batch, total_seq_len, backbone_dim] backbone features on device
        """
        if self._backbone_model is not None:
            # Use HuggingFace model for backbone features
            # This is a temporary path until Qwen3 is ported to TTNN
            raise NotImplementedError(
                "Full Qwen3 backbone on TTNN not yet implemented. "
                "Use compute_backbone_features_cpu() instead."
            )

        raise NotImplementedError(
            "Backbone encoding requires either a CPU reference model or "
            "the Qwen3 TTNN implementation."
        )

    def run_flow_matching(
        self,
        backbone_features: ttnn.Tensor,
        state: torch.Tensor,
        embodiment_id: int = 0,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Flow matching inference: denoise actions over K steps.

        Args:
            backbone_features: [batch, seq_len, backbone_dim] from Qwen3 layer 16
            state: [batch, state_dim] robot proprioception (padded to max_state_dim)
            embodiment_id: Which robot embodiment
            image_mask: Optional mask for image vs text tokens

        Returns:
            [batch, action_horizon, action_dim] predicted actions (CPU)
        """
        batch_size = backbone_features.shape[0]
        K = self.config.num_inference_timesteps  # 4
        H = self.config.action_horizon  # 16
        action_dim = self.config.embodiment.max_action_dim  # 29
        dt = 1.0 / K

        # Apply VL LayerNorm to backbone features
        if hasattr(self, 'vl_ln_weight'):
            backbone_features = ttnn.layer_norm(
                backbone_features,
                weight=self.vl_ln_weight,
                bias=self.vl_ln_bias,
                epsilon=1e-6,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        # Encode state -> [batch, 1, input_embedding_dim]
        state_padded = F.pad(state, (0, self.config.embodiment.max_state_dim - state.shape[-1]))
        state_tt = to_tt_tensor(state_padded.unsqueeze(1), self.device)
        state_features = self.state_encoder(state_tt, embodiment_id)
        ttnn.deallocate(state_tt)

        # Initialize noisy actions from Gaussian noise
        actions = torch.randn(batch_size, H, action_dim)

        # Euler integration loop
        for step in range(K):
            t_cont = torch.tensor([step / K] * batch_size)

            # Transfer noisy actions to device
            actions_tt = to_tt_tensor(actions, self.device)

            # Encode actions with timestep
            action_features = self.action_encoder(
                actions_tt, t_cont, embodiment_id,
            )
            ttnn.deallocate(actions_tt)

            # Concatenate state + action features: [batch, 1+H, embedding_dim]
            dit_input = ttnn.concat([state_features, action_features], dim=1)
            ttnn.deallocate(action_features)

            # Add positional embeddings
            if self.pos_embed is not None:
                # Slice pos_embed to match sequence length
                seq_len = dit_input.shape[1]
                pos = ttnn.slice(
                    self.pos_embed,
                    [0, 0, 0],
                    [1, seq_len, self.pos_embed.shape[2]],
                )
                dit_input = ttnn.add(dit_input, pos, memory_config=ttnn.L1_MEMORY_CONFIG)

            # Get timestep embedding for AdaLN conditioning
            # The timestep embedding comes from the action encoder's timestep encoder
            timestep_emb = self.action_encoder.timestep_encoder(t_cont)

            # Run DiT
            dit_output = self.dit(
                dit_input, timestep_emb, backbone_features,
            )
            ttnn.deallocate(dit_input)
            ttnn.deallocate(timestep_emb)

            # Slice to get action tokens only (skip state token at position 0)
            action_out = ttnn.slice(
                dit_output,
                [0, 1, 0],
                [batch_size, 1 + H, dit_output.shape[2]],
            )
            ttnn.deallocate(dit_output)

            # Decode to velocity field
            velocity = self.action_decoder(action_out, embodiment_id)
            ttnn.deallocate(action_out)

            # Transfer velocity to CPU for Euler step
            velocity_cpu = ttnn.to_torch(velocity).to(torch.float32)
            ttnn.deallocate(velocity)

            # Euler step: actions = actions + dt * velocity
            actions = actions + dt * velocity_cpu.squeeze(0) if batch_size == 1 else actions + dt * velocity_cpu

        return actions

    def forward(
        self,
        pixel_values: torch.Tensor,
        text_tokens: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: int = 0,
    ) -> torch.Tensor:
        """
        Full forward pass: images + text + state -> actions.

        Args:
            pixel_values: [batch, 3, H, W] camera images
            text_tokens: [batch, text_seq_len] language instruction tokens
            state: [batch, state_dim] robot proprioception
            embodiment_id: Which robot embodiment

        Returns:
            [batch, action_horizon, action_dim] predicted actions
        """
        # Step 1: Vision encoding
        image_tokens = self.encode_vision(pixel_values)

        # Step 2: Backbone encoding (currently requires CPU reference)
        # TODO: implement Qwen3 on TTNN
        backbone_features = self.encode_backbone(image_tokens, text_tokens)

        # Step 3: Flow matching
        actions = self.run_flow_matching(
            backbone_features, state, embodiment_id,
        )

        return actions
