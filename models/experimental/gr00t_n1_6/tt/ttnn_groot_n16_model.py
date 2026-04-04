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
from models.experimental.gr00t_n1_6.common.configs import Gr00tN16Config
from models.experimental.gr00t_n1_6.common.weight_loader import Gr00tN16WeightLoader
from models.experimental.gr00t_n1_6.tt.ttnn_common import (
    CORE_GRID_BH,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_params,
    preprocess_rmsnorm_params,
    to_tt_tensor,
)
from models.experimental.gr00t_n1_6.tt.ttnn_siglip2 import SigLIP2VisionEncoderTTNN
from models.experimental.gr00t_n1_6.tt.ttnn_dit import AlternateVLDiTTTNN
from models.experimental.gr00t_n1_6.tt.ttnn_embodiment import (
    CategorySpecificMLPTTNN,
    MultiEmbodimentActionEncoderTTNN,
)
from models.experimental.gr00t_n1_6.tt.ttnn_qwen3 import Qwen3ModelTTNN

# Image placeholder token ID used by Eagle-Block2A backbone
IMAGE_TOKEN_ID = 151669

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
        # 0.weight [4608] (LayerNorm), 0.bias [4608]
        # 1.weight [2048, 4608] (Linear), 1.bias [2048]
        # 3.weight [2048, 2048] (Linear), 3.bias [2048]
        # Index 2 is GELU activation (no weights)

        # Layer 0: LayerNorm over pixel-shuffled features (4608 = 4 * 1152)
        ln_w = connector_weights.get("0.weight")
        ln_b = connector_weights.get("0.bias")
        if ln_w is not None:
            self.ln_weight, self.ln_bias = preprocess_layernorm_params(ln_w, ln_b, device)

        # Layer 1: Linear 4608 -> 2048
        w1 = connector_weights.get("1.weight")
        b1 = connector_weights.get("1.bias")
        if w1 is not None:
            self.proj1_weight = preprocess_linear_weight(w1, device)
            self.proj1_bias = preprocess_linear_bias(b1, device) if b1 is not None else None

        # Layer 3: Linear 2048 -> 2048
        w3 = connector_weights.get("3.weight")
        b3 = connector_weights.get("3.bias")
        if w3 is not None:
            self.proj3_weight = preprocess_linear_weight(w3, device)
            self.proj3_bias = preprocess_linear_bias(b3, device) if b3 is not None else None

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

        # Layer 0: LayerNorm over pixel-shuffled features
        h = ttnn.layer_norm(
            shuffled_tt, weight=self.ln_weight, bias=self.ln_bias,
            epsilon=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(shuffled_tt)

        # Layer 1: Linear 4608 -> 2048
        h = ttnn.linear(
            h, self.proj1_weight, bias=self.proj1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b, core_grid=CORE_GRID_BH,
        )

        # Layer 2: GELU activation
        h = ttnn.gelu(h)

        # Layer 3: Linear 2048 -> 2048
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

        # 7. Qwen3-1.7B language backbone (first 16 of 28 layers)
        logger.info("  Loading Qwen3-1.7B backbone (16 layers)...")
        self.qwen3 = Qwen3ModelTTNN(
            config.backbone.language,
            weight_loader.get_language_weights(),
            device,
        )

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

    @staticmethod
    def build_backbone_input_ids(
        text_tokens: torch.Tensor,
        num_image_tokens: int = 64,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Build input_ids for the Eagle backbone by appending image placeholders.

        Constructs: [text_tokens..., <IMG_CONTEXT> * num_image_tokens]

        Args:
            text_tokens: [batch, text_seq_len] language instruction token IDs.
            num_image_tokens: Number of image placeholder tokens (64 after pixel shuffle).

        Returns:
            input_ids: [batch, text_seq_len + num_image_tokens] with image placeholders.
            image_token_positions: List of sequence indices where image tokens are placed.
        """
        batch_size, text_len = text_tokens.shape
        img_placeholder = torch.full(
            (batch_size, num_image_tokens), IMAGE_TOKEN_ID, dtype=torch.long,
        )
        input_ids = torch.cat([text_tokens, img_placeholder], dim=1)
        image_token_positions = list(range(text_len, text_len + num_image_tokens))
        return input_ids, image_token_positions

    def encode_backbone(
        self,
        image_tokens: ttnn.Tensor,
        text_tokens: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Run Qwen3 backbone to get vision-language features.

        Assembles input_ids with image placeholders, splices vision features in,
        runs 16 Qwen3 layers, and returns hidden states at image token positions
        as the backbone features for the DiT action head.

        Args:
            image_tokens: [batch, num_image_tokens, language_dim] on device
                          (output of SigLIP2 + pixel shuffle + connector).
            text_tokens:  [batch, text_seq_len] language instruction token IDs on CPU.

        Returns:
            [batch, num_image_tokens, backbone_dim] backbone features on device.
        """
        num_image_tokens = self.config.backbone.num_image_tokens_per_frame  # 64

        # Build input_ids with image placeholders
        input_ids, image_positions = self.build_backbone_input_ids(
            text_tokens, num_image_tokens,
        )

        # Run Qwen3 forward (embedding + 16 layers + final norm)
        # Image features are spliced in at image_positions inside the model
        backbone_output = self.qwen3(
            input_ids,
            image_features=image_tokens,
            image_token_positions=image_positions,
        )
        # backbone_output: [batch, total_seq_len, hidden_size] on device

        # Extract features at image token positions only
        # These are the backbone features the DiT cross-attends to
        start_pos = image_positions[0]
        end_pos = image_positions[-1] + 1
        batch_size = input_ids.shape[0]
        hidden_size = backbone_output.shape[2]

        backbone_features = ttnn.slice(
            backbone_output,
            [0, start_pos, 0],
            [batch_size, end_pos, hidden_size],
        )
        ttnn.deallocate(backbone_output)

        return backbone_features

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
        # Step 1: Vision encoding (SigLIP2 + pixel shuffle + connector)
        image_tokens = self.encode_vision(pixel_values)

        # Step 2: Backbone encoding (Qwen3 layer 16)
        backbone_features = self.encode_backbone(image_tokens, text_tokens)
        ttnn.deallocate(image_tokens)

        # Step 3: Flow matching (4 Euler steps with DiT)
        actions = self.run_flow_matching(
            backbone_features, state, embodiment_id,
        )

        return actions
