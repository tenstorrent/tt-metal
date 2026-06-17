# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CosyVoice Flow-based Decoder - Conditional Flow Matching and Diffusion Transformer.

The flow decoder takes semantic tokens from the LLM backbone and generates
mel-spectrograms using a conditional flow matching (CFM) algorithm with
a Diffusion Transformer (DiT) architecture.
"""

from typing import Optional, Tuple

import torch
import ttnn
from loguru import logger


class TtCFMDecoder:
    """Conditional Flow Matching decoder using TTNN APIs.

    Implements the flow matching ODE solver for mel-spectrogram generation.
    Uses the Euler solver with cosine scheduling.
    """

    def __init__(self, config):
        self.config = config
        self.sigma_min = 1e-6
        self.solver = "euler"
        self.t_scheduler = "cosine"
        self.inference_cfg_rate = 0.7

    def compute_noise(self, t: float) -> float:
        """Compute noise level for time t using cosine schedule."""
        if self.t_scheduler == "cosine":
            return 1.0 - 0.5 * (1.0 + torch.cos(torch.tensor(t * torch.pi)))
        return t

    def solve_euler(
        self,
        x_t: ttnn.Tensor,
        t: float,
        dt: float,
        cond: ttnn.Tensor,
        embedding: ttnn.Tensor,
        estimator_fn,
    ) -> ttnn.Tensor:
        """Single Euler ODE step."""
        v_pred = estimator_fn(x_t, t, cond, embedding)
        return ttnn.add(x_t, ttnn.mul(v_pred, ttnn.full(x_t.shape, dt, dtype=ttnn.bfloat16)))

    def __call__(
        self,
        mu: ttnn.Tensor,
        mask: ttnn.Tensor,
        cond: ttnn.Tensor,
        embedding: ttnn.Tensor,
        n_timesteps: int = 25,
    ) -> ttnn.Tensor:
        """Run CFM decoder.

        Solves the probability flow ODE from noise to data distribution.

        Args:
            mu: Initial noise [batch, channels, time]
            mask: Padding mask [batch, 1, time]
            cond: Condition tensor [batch, channels, time]
            embedding: Speaker embedding [batch, spk_dim]
            n_timesteps: Number of ODE steps

        Returns:
            Generated mel-spectrogram [batch, 80, time]
        """
        # Initialize with noise
        x = mu

        # Time steps
        timesteps = torch.linspace(1.0, self.sigma_min, n_timesteps)

        for i in range(n_timesteps - 1):
            t = timesteps[i].item()
            dt = timesteps[i] - timesteps[i + 1]

            # Apply mask
            x = self.solve_euler(x, t, dt.item(), cond, embedding, self._estimator)
            x = ttnn.mul(x, mask)

        return x

    def _estimator(
        self,
        x_t: ttnn.Tensor,
        t: float,
        cond: ttnn.Tensor,
        embedding: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Velocity field estimator - called during ODE solving.

        Placeholder for the DiT-based velocity estimator.
        The actual DiT implementation replaces this.
        """
        return x_t  # Identity for initial implementation


class TtDiTBlock:
    """Diffusion Transformer (DiT) block using TTNN APIs."""

    def __init__(
        self,
        device: ttnn.Device,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        self.device = device
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Layer norm
        self.norm1_weight = ttnn.ones(hidden_size, device=device, layout=ttnn.TILE_LAYOUT)
        self.norm2_weight = ttnn.ones(hidden_size, device=device, layout=ttnn.TILE_LAYOUT)

        # Attention projections (initialized with random/default weights for Stage 1)
        self.qkv_weight = ttnn.ones((hidden_size, 3 * hidden_size), device=device,
                                     layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.proj_weight = ttnn.ones((hidden_size, hidden_size), device=device,
                                      layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.fc1_weight = ttnn.ones((hidden_size, mlp_hidden), device=device,
                                     layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.fc2_weight = ttnn.ones((mlp_hidden, hidden_size), device=device,
                                     layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def __call__(
        self,
        x: ttnn.Tensor,
        timestep_embed: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Apply DiT block.

        Args:
            x: Input tensor [batch, seq, hidden]
            timestep_embed: Optional timestep embedding

        Returns:
            Output tensor
        """
        # Pre-norm + attention with residual
        residual = x
        x = ttnn.layer_norm(x, weight=self.norm1_weight)
        x = ttnn.linear(x, self.qkv_weight)
        x = ttnn.linear(x, self.proj_weight)
        x = ttnn.add(x, residual)

        # Pre-norm + MLP with residual
        residual = x
        x = ttnn.layer_norm(x, weight=self.norm2_weight)
        x = ttnn.linear(x, self.fc1_weight)
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.fc2_weight)
        x = ttnn.add(x, residual)

        return x


class TtMaskedDiffWithXvec:
    """Masked diffusion model with speaker embedding (CosyVoice Flow Decoder).

    This is the main flow decoder module that wraps the DiT-based CFM decoder.
    It processes semantic tokens through an encoder and uses the CFM decoder
    to generate mel-spectrograms.
    """

    def __init__(
        self,
        device: ttnn.Device,
        config,
        state_dict: dict,
        tt_cache_path: Optional[str] = None,
    ):
        self.device = device
        self.config = config
        self.input_size = config.flow_input_size
        self.output_size = config.flow_output_size
        self.vocab_size = config.flow_vocab_size
        self.input_frame_rate = config.flow_input_frame_rate

        memory_config = ttnn.DRAM_MEMORY_CONFIG
        dtype = ttnn.bfloat16

        logger.info(f"Initializing CosyVoice flow decoder: "
                     f"vocab={self.vocab_size}, in={self.input_size}, out={self.output_size}")

        # Token embedding
        embed_weight = state_dict.get(
            "flow.input_embedding.weight",
            torch.zeros(self.vocab_size, self.input_size),
        )
        self.input_embedding = ttnn.as_tensor(
            embed_weight.unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )

        # Speaker embedding affine layer
        spk_weight = state_dict.get(
            "flow.spk_embed_affine_layer.weight",
            torch.zeros(config.flow_spk_embed_dim, self.output_size),
        )
        spk_bias = state_dict.get(
            "flow.spk_embed_affine_layer.bias",
            None,
        )
        self.spk_weight = ttnn.as_tensor(
            spk_weight.T.unsqueeze(0).unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )
        if spk_bias is not None:
            self.spk_bias = ttnn.as_tensor(
                spk_bias.unsqueeze(0).unsqueeze(0),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                dtype=dtype,
            )
        else:
            self.spk_bias = None

        # Encoder projection (from encoder output to mel output size)
        encoder_proj_weight = state_dict.get(
            "flow.encoder_proj.weight",
            torch.zeros(self.input_size, self.output_size),
        )
        self.encoder_proj_weight = ttnn.as_tensor(
            encoder_proj_weight.T.unsqueeze(0).unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )

        # CFM decoder
        self.cfm_decoder = TtCFMDecoder(config)

        # DiT blocks (simplified for Stage 1 - full implementation in Stage 2)
        self.dit_blocks = [
            TtDiTBlock(device, config.flow_in_channels, config.dit_num_heads)
            for _ in range(min(4, config.dit_n_blocks))
        ]

    def forward(
        self,
        speech_tokens: ttnn.Tensor,
        token_len: ttnn.Tensor,
        speaker_embedding: ttnn.Tensor,
        feat_len: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Forward pass through the flow decoder.

        Args:
            speech_tokens: Semantic token IDs [batch, seq_len]
            token_len: Token sequence lengths [batch]
            speaker_embedding: Speaker embedding [batch, spk_dim]
            feat_len: Target feature lengths [batch] (optional)

        Returns:
            Mel-spectrogram [batch, 80, time]
        """
        # Embed tokens
        x = ttnn.embedding(speech_tokens, self.input_embedding)

        # Apply mask
        mask = ttnn.ones(x.shape, device=self.device, dtype=ttnn.bfloat16)
        x = ttnn.mul(x, mask)

        # Encode with encoder projection
        x = ttnn.linear(x, self.encoder_proj_weight)

        # Process through CFM decoder
        # Rearrange to [batch, channels, time] for CFM
        x = ttnn.permute(x, (0, 2, 1))  # [batch, 80, seq]

        # Apply speaker embedding
        if speaker_embedding is not None:
            spk = ttnn.linear(speaker_embedding, self.spk_weight)
            if self.spk_bias is not None:
                spk = ttnn.add(spk, self.spk_bias)
            spk = ttnn.unsqueeze(spk, 2)  # [batch, 80, 1]
            x = ttnn.add(x, spk)

        # CFM decoding
        mu = ttnn.randn(x.shape, device=self.device, dtype=ttnn.bfloat16)
        mask_3d = ttnn.ones(x.shape, device=self.device, dtype=ttnn.bfloat16)
        mel = self.cfm_decoder(mu, mask_3d, x, speaker_embedding)

        return mel

    def inference(
        self,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
        flow_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run flow decoder inference.

        Args:
            token: Speech token IDs [batch, seq_len]
            token_len: Token length [batch]
            prompt_token: Prompt speech tokens [batch, prompt_len]
            prompt_token_len: Prompt token length [batch]
            prompt_feat: Prompt mel features [batch, 80, prompt_time]
            prompt_feat_len: Prompt feature length [batch]
            embedding: Speaker embedding [batch, spk_dim]
            flow_cache: Optional flow cache for streaming

        Returns:
            (mel_spectrogram, updated_cache)
        """
        # Combine prompt tokens with input tokens
        combined_tokens = torch.cat([prompt_token, token], dim=1)
        tt_tokens = ttnn.from_torch(combined_tokens, device=self.device, dtype=ttnn.uint32)
        tt_embedding = ttnn.from_torch(embedding, device=self.device, dtype=ttnn.bfloat16)

        # Run flow decoder
        mel = self.forward(tt_tokens, token_len, tt_embedding)

        # Convert back to torch
        mel_torch = ttnn.to_torch(mel)

        # Update flow cache
        new_cache = mel_torch[:, :, -4:] if flow_cache is not None else mel_torch[:, :, :0]

        return mel_torch, new_cache
