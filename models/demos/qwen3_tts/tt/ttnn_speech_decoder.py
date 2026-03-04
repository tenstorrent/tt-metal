# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Full TTNN implementation of Speech Tokenizer Decoder.

Uses TTNN ops for all operations including conv for trace compatibility:
- ttnn.conv1d for regular convolutions
- ttnn.conv_transpose2d with H=1 for transposed convolutions
- TTNN element-wise ops for snake activation
"""

from typing import Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.speech_tokenizer import SpeechTokenizerConfig
from models.demos.qwen3_tts.tt.ttnn_conv_decoder import TTNNConv1d, TTNNConvTranspose1d, ttnn_snake_activation


class TTNNConvNeXtBlock(LightweightModule):
    """TTNN ConvNeXt block for upsampler."""

    def __init__(
        self,
        device,
        channels: int,
        kernel_size: int = 7,
        intermediate_channels: int = None,
        weights: dict = None,
    ):
        super().__init__()
        self.device = device
        self.channels = channels
        self.intermediate_channels = intermediate_channels or channels * 4

        # Depthwise conv
        self.dwconv = None
        if weights and "dwconv.conv.weight" in weights:
            self.dwconv = TTNNConv1d(
                device=device,
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                weight=weights["dwconv.conv.weight"],
                bias_tensor=weights.get("dwconv.conv.bias"),
            )

        # Layer norm weights
        self.norm_weight = None
        self.norm_bias = None
        if weights and "norm.weight" in weights:
            self.norm_weight = ttnn.from_torch(
                weights["norm.weight"].unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.norm_bias = ttnn.from_torch(
                weights["norm.bias"].unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        # Pointwise convs (as linear)
        self.pwconv1_weight = None
        self.pwconv1_bias = None
        self.pwconv2_weight = None
        self.pwconv2_bias = None

        if weights:
            if "pwconv1.weight" in weights:
                # [out, in] for linear - transpose for TTNN linear
                w = weights["pwconv1.weight"].T.contiguous()
                self.pwconv1_weight = ttnn.from_torch(
                    w.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
            if "pwconv1.bias" in weights:
                self.pwconv1_bias = ttnn.from_torch(
                    weights["pwconv1.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
            if "pwconv2.weight" in weights:
                w = weights["pwconv2.weight"].T.contiguous()
                self.pwconv2_weight = ttnn.from_torch(
                    w.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
            if "pwconv2.bias" in weights:
                self.pwconv2_bias = ttnn.from_torch(
                    weights["pwconv2.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

        # Layer scale (gamma)
        self.gamma = None
        if weights and "gamma" in weights:
            self.gamma = ttnn.from_torch(
                weights["gamma"].view(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

    def forward(self, x: ttnn.Tensor, seq_len: int) -> Tuple[ttnn.Tensor, int]:
        """
        Forward pass.

        Args:
            x: Input [batch, 1, seq_len, channels] in NHWC format
            seq_len: Sequence length

        Returns:
            Output tensor and output length
        """
        residual = x

        # Depthwise conv (need NLC format)
        if self.dwconv:
            # [B, 1, L, C] -> [B, L, C] for conv1d
            x_nlc = ttnn.reshape(x, (x.shape[0], seq_len, self.channels))
            x_nlc, out_len = self.dwconv(x_nlc, seq_len)
            # Back to [B, 1, L, C]
            x = ttnn.reshape(x_nlc, (x.shape[0], 1, out_len, self.channels))
        else:
            out_len = seq_len

        # Layer norm
        if self.norm_weight is not None:
            x = ttnn.layer_norm(x, weight=self.norm_weight, bias=self.norm_bias)

        # Pointwise conv1 + GELU
        if self.pwconv1_weight is not None:
            x = ttnn.linear(x, self.pwconv1_weight, bias=self.pwconv1_bias)
            x = ttnn.gelu(x)

        # Pointwise conv2
        if self.pwconv2_weight is not None:
            x = ttnn.linear(x, self.pwconv2_weight, bias=self.pwconv2_bias)

        # Layer scale
        if self.gamma is not None:
            x = ttnn.mul(x, self.gamma)

        # Residual
        x = ttnn.add(x, residual)

        return x, out_len


class TTNNDecoderResidualBlock(LightweightModule):
    """TTNN residual block for conv decoder."""

    def __init__(
        self,
        device,
        channels: int,
        weights: dict,
        block_idx: int,
    ):
        super().__init__()
        self.device = device
        self.channels = channels

        # Activation 1 (snake)
        self.act1_alpha = None
        self.act1_beta = None
        act1_key = f"block.{block_idx}.act1"
        if f"{act1_key}.alpha" in weights:
            self.act1_alpha = ttnn.from_torch(
                weights[f"{act1_key}.alpha"].view(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.act1_beta = ttnn.from_torch(
                weights[f"{act1_key}.beta"].view(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        # Conv1 (dilated)
        conv1_weight = weights.get(f"block.{block_idx}.conv1.conv.weight")
        conv1_bias = weights.get(f"block.{block_idx}.conv1.conv.bias")
        self.conv1 = None
        if conv1_weight is not None:
            kernel_size = conv1_weight.shape[-1]
            self.conv1 = TTNNConv1d(
                device=device,
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                weight=conv1_weight,
                bias_tensor=conv1_bias,
            )

        # Activation 2 (snake)
        self.act2_alpha = None
        self.act2_beta = None
        act2_key = f"block.{block_idx}.act2"
        if f"{act2_key}.alpha" in weights:
            self.act2_alpha = ttnn.from_torch(
                weights[f"{act2_key}.alpha"].view(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.act2_beta = ttnn.from_torch(
                weights[f"{act2_key}.beta"].view(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        # Conv2 (1x1)
        conv2_weight = weights.get(f"block.{block_idx}.conv2.conv.weight")
        conv2_bias = weights.get(f"block.{block_idx}.conv2.conv.bias")
        self.conv2 = None
        if conv2_weight is not None:
            self.conv2 = TTNNConv1d(
                device=device,
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                weight=conv2_weight,
                bias_tensor=conv2_bias,
            )

    def forward(self, x: ttnn.Tensor, seq_len: int) -> Tuple[ttnn.Tensor, int]:
        """Forward pass."""
        residual = x
        out_len = seq_len

        # Snake activation 1
        if self.act1_alpha is not None:
            x = ttnn_snake_activation(x, self.act1_alpha, self.act1_beta)

        # Conv1
        if self.conv1 is not None:
            # [B, 1, L, C] -> [B, L, C]
            x_nlc = ttnn.reshape(x, (x.shape[0], out_len, x.shape[-1]))
            x_nlc, out_len = self.conv1(x_nlc, out_len)
            x = ttnn.reshape(x_nlc, (x.shape[0], 1, out_len, x.shape[-1]))

        # Snake activation 2
        if self.act2_alpha is not None:
            x = ttnn_snake_activation(x, self.act2_alpha, self.act2_beta)

        # Conv2
        if self.conv2 is not None:
            x_nlc = ttnn.reshape(x, (x.shape[0], out_len, x.shape[-1]))
            x_nlc, out_len = self.conv2(x_nlc, out_len)
            x = ttnn.reshape(x_nlc, (x.shape[0], 1, out_len, x.shape[-1]))

        # Residual
        x = ttnn.add(x, residual)

        return x, out_len


class TTNNDecoderBlock(LightweightModule):
    """TTNN decoder block with snake activation, upsample, and residual layers."""

    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        upsample_rate: int,
        weights: dict,
        num_residual_layers: int = 3,
    ):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_rate = upsample_rate

        # Snake activation before upsampling
        self.alpha = None
        self.beta = None
        if "alpha" in weights:
            self.alpha = ttnn.from_torch(
                weights["alpha"].view(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.beta = ttnn.from_torch(
                weights["beta"].view(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        # Upsample conv_transpose1d
        self.upsample_conv = None
        upsample_weight = weights.get("block.1.conv.weight")
        upsample_bias = weights.get("block.1.conv.bias")
        if upsample_weight is not None:
            kernel_size = upsample_weight.shape[-1]
            padding = (kernel_size - upsample_rate) // 2
            self.upsample_conv = TTNNConvTranspose1d(
                device=device,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=upsample_rate,
                padding=padding,
                weight=upsample_weight,
                bias_tensor=upsample_bias,
            )

        # Residual blocks
        self.residual_blocks = []
        for i in range(num_residual_layers):
            block_idx = 2 + i
            block = TTNNDecoderResidualBlock(device, out_channels, weights, block_idx)
            self.residual_blocks.append(block)

    def forward(self, x: ttnn.Tensor, seq_len: int) -> Tuple[ttnn.Tensor, int]:
        """Forward pass."""
        out_len = seq_len

        # Snake activation
        if self.alpha is not None:
            x = ttnn_snake_activation(x, self.alpha, self.beta)

        # Upsample
        if self.upsample_conv is not None:
            # x is [B, 1, L, C] - keep format for conv_transpose2d
            x, out_len = self.upsample_conv(x, out_len)

        # Residual blocks
        for block in self.residual_blocks:
            x, out_len = block(x, out_len)

        return x, out_len


class TTNNSpeechTokenizerDecoder(LightweightModule):
    """
    Full TTNN Speech Tokenizer Decoder for trace compatibility.

    All operations are in TTNN for tracing.
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        config: Optional[SpeechTokenizerConfig] = None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.config = config or SpeechTokenizerConfig()
        self.dtype = dtype

        # Store PyTorch weights for codebook lookup (done on host before tracing)
        self._load_codebook_weights(state_dict)

        # Pre-conv (conv1d) - use weight dimensions
        self.pre_conv = None
        self.pre_conv_out_channels = None
        if "pre_conv.conv.weight" in state_dict:
            weight = state_dict["pre_conv.conv.weight"]
            bias = state_dict.get("pre_conv.conv.bias")
            kernel_size = weight.shape[-1]
            in_channels = weight.shape[1]
            out_channels = weight.shape[0]
            self.pre_conv_out_channels = out_channels
            self.pre_conv = TTNNConv1d(
                device=device,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                weight=weight,
                bias_tensor=bias,
            )

        # Upsampler (ConvNeXt blocks)
        self.upsample_convs = []
        self.upsample_blocks = []
        for i, ratio in enumerate(self.config.upsampling_ratios):
            # Upsample conv
            prefix = f"upsample.{i}."
            conv_weight = state_dict.get(f"{prefix}0.conv.weight")
            conv_bias = state_dict.get(f"{prefix}0.conv.bias")
            if conv_weight is not None:
                in_ch = conv_weight.shape[1]
                out_ch = conv_weight.shape[0]
                kernel_size = conv_weight.shape[-1]
                padding = (kernel_size - ratio) // 2
                upsample = TTNNConvTranspose1d(
                    device=device,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=ratio,
                    padding=padding,
                    weight=conv_weight,
                    bias_tensor=conv_bias,
                )
                self.upsample_convs.append(upsample)

                # ConvNeXt block
                convnext_weights = {
                    k.replace(f"{prefix}1.", ""): v for k, v in state_dict.items() if k.startswith(f"{prefix}1.")
                }
                if convnext_weights:
                    block = TTNNConvNeXtBlock(device, out_ch, kernel_size=7, weights=convnext_weights)
                    self.upsample_blocks.append(block)

        # Initial decoder conv
        self.decoder_initial_conv = None
        if "decoder.0.conv.weight" in state_dict:
            weight = state_dict["decoder.0.conv.weight"]
            bias = state_dict.get("decoder.0.conv.bias")
            kernel_size = weight.shape[-1]
            in_ch = weight.shape[1]
            out_ch = weight.shape[0]
            self.decoder_initial_conv = TTNNConv1d(
                device=device,
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                weight=weight,
                bias_tensor=bias,
            )

        # Decoder blocks with upsampling
        self.decoder_blocks = []
        channels = self.config.decoder_dim
        for i, rate in enumerate(self.config.upsample_rates):
            block_prefix = f"decoder.{i + 1}."
            block_weights = {
                k.replace(block_prefix, ""): v for k, v in state_dict.items() if k.startswith(block_prefix)
            }

            if block_weights:
                # Get in/out channels from weights
                upsample_weight = block_weights.get("block.1.conv.weight")
                if upsample_weight is not None:
                    in_ch = upsample_weight.shape[0]  # conv_transpose: [I, O, K]
                    out_ch = upsample_weight.shape[1]
                else:
                    in_ch = channels
                    out_ch = channels // 2

                block = TTNNDecoderBlock(
                    device=device,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    upsample_rate=rate,
                    weights=block_weights,
                )
                self.decoder_blocks.append(block)
                channels = out_ch

        # Final snake activation
        self.final_alpha = None
        self.final_beta = None
        if "decoder.5.alpha" in state_dict:
            self.final_alpha = ttnn.from_torch(
                state_dict["decoder.5.alpha"].view(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            self.final_beta = ttnn.from_torch(
                state_dict["decoder.5.beta"].view(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        # Final conv
        self.final_conv = None
        if "decoder.6.conv.weight" in state_dict:
            weight = state_dict["decoder.6.conv.weight"]
            bias = state_dict.get("decoder.6.conv.bias")
            kernel_size = weight.shape[-1]
            self.final_conv = TTNNConv1d(
                device=device,
                in_channels=weight.shape[1],
                out_channels=weight.shape[0],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                weight=weight,
                bias_tensor=bias,
            )

    def _load_codebook_weights(self, state_dict: dict):
        """Load RVQ codebook weights for host-side lookup."""
        # RVQ First (semantic codebook)
        self.rvq_first_codebook = None
        first_key = "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"
        first_usage_key = "quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"
        if first_key in state_dict:
            embedding_sum = state_dict[first_key].clone()
            cluster_usage = state_dict.get(first_usage_key)
            if cluster_usage is not None:
                epsilon = 1e-5
                self.rvq_first_codebook = embedding_sum / cluster_usage.clamp(min=epsilon)[:, None]
            else:
                self.rvq_first_codebook = embedding_sum

        self.rvq_first_output_proj = state_dict.get("quantizer.rvq_first.output_proj.weight")

        # RVQ Rest codebooks (acoustic)
        self.rvq_rest_codebooks = []
        for i in range(self.config.num_quantizers - 1):
            key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"
            usage_key = f"quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"
            if key in state_dict:
                embedding_sum = state_dict[key].clone()
                cluster_usage = state_dict.get(usage_key)
                if cluster_usage is not None:
                    epsilon = 1e-5
                    normalized = embedding_sum / cluster_usage.clamp(min=epsilon)[:, None]
                    self.rvq_rest_codebooks.append(normalized)
                else:
                    self.rvq_rest_codebooks.append(embedding_sum)

        self.rvq_rest_output_proj = state_dict.get("quantizer.rvq_rest.output_proj.weight")

    def codebook_lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embeddings from RVQ codebooks (PyTorch, on host).

        This runs before TTNN operations, result is converted to TTNN tensor.

        Architecture (matching official qwen_tts):
        - rvq_first (semantic): 1 codebook -> output_proj (256 -> 512)
        - rvq_rest (acoustic): 15 codebooks -> sum -> output_proj (256 -> 512)
        - ADD the results (not concatenate!) -> [batch, seq_len, 512]

        Args:
            token_ids: [batch, num_quantizers, seq_len]

        Returns:
            embeddings: [batch, seq_len, 512] for NLC format (channels-last)
        """
        import torch.nn.functional as F

        batch_size, num_quantizers, seq_len = token_ids.shape
        device = token_ids.device

        # Process RVQ First (semantic) - in channels-first for conv1d
        rvq_first_emb = None
        if self.rvq_first_codebook is not None and num_quantizers > 0:
            codebook = self.rvq_first_codebook.to(device)
            ids = token_ids[:, 0, :]
            rvq_first_emb = F.embedding(ids, codebook)  # [batch, seq_len, 256]
            rvq_first_emb = rvq_first_emb.transpose(1, 2)  # [batch, 256, seq_len]

            if self.rvq_first_output_proj is not None:
                proj = self.rvq_first_output_proj.to(device)
                rvq_first_emb = F.conv1d(rvq_first_emb, proj)  # [batch, 512, seq_len]

        # Process RVQ Rest (acoustic) - sum all, then project
        rvq_rest_emb = None
        if len(self.rvq_rest_codebooks) > 0 and num_quantizers > 1:
            for i, codebook in enumerate(self.rvq_rest_codebooks):
                if i + 1 >= num_quantizers:
                    break
                codebook = codebook.to(device)
                ids = token_ids[:, i + 1, :]
                emb = F.embedding(ids, codebook)  # [batch, seq_len, 256]
                emb = emb.transpose(1, 2)  # [batch, 256, seq_len]
                if rvq_rest_emb is None:
                    rvq_rest_emb = emb
                else:
                    rvq_rest_emb = rvq_rest_emb + emb

            if rvq_rest_emb is not None and self.rvq_rest_output_proj is not None:
                proj = self.rvq_rest_output_proj.to(device)
                rvq_rest_emb = F.conv1d(rvq_rest_emb, proj)  # [batch, 512, seq_len]

        # ADD rvq_first and rvq_rest (not concatenate!)
        if rvq_first_emb is not None and rvq_rest_emb is not None:
            embeddings = rvq_first_emb + rvq_rest_emb  # [batch, 512, seq_len]
        elif rvq_first_emb is not None:
            embeddings = rvq_first_emb
        elif rvq_rest_emb is not None:
            embeddings = rvq_rest_emb
        else:
            raise ValueError("No codebook embeddings available")

        # Convert to channels-last (NLC) for TTNN
        embeddings = embeddings.transpose(1, 2)  # [batch, seq_len, 512]
        return embeddings

    def forward_ttnn(self, embeddings_tt: ttnn.Tensor, seq_len: int) -> Tuple[ttnn.Tensor, int]:
        """
        Forward pass through decoder (TTNN only, for tracing).

        Args:
            embeddings_tt: Input tensor in NHWC format [batch, 1, seq_len, latent_dim]
            seq_len: Input sequence length

        Returns:
            audio: Output audio tensor [batch, 1, out_len, 1]
            out_len: Output sequence length
        """
        out_len = seq_len
        x = embeddings_tt

        # Pre-conv (need NLC format)
        if self.pre_conv is not None:
            x_nlc = ttnn.reshape(x, (x.shape[0], out_len, x.shape[-1]))
            x_nlc, out_len = self.pre_conv(x_nlc, out_len)
            x = ttnn.reshape(x_nlc, (x.shape[0], 1, out_len, x_nlc.shape[-1]))

        # Upsampler
        for i, (upsample_conv, convnext_block) in enumerate(zip(self.upsample_convs, self.upsample_blocks)):
            # Upsample
            x, out_len = upsample_conv(x, out_len)
            # ConvNeXt block
            x, out_len = convnext_block.forward(x, out_len)

        # Initial decoder conv
        if self.decoder_initial_conv is not None:
            x_nlc = ttnn.reshape(x, (x.shape[0], out_len, x.shape[-1]))
            x_nlc, out_len = self.decoder_initial_conv(x_nlc, out_len)
            x = ttnn.reshape(x_nlc, (x.shape[0], 1, out_len, x_nlc.shape[-1]))

        # Decoder blocks
        for block in self.decoder_blocks:
            x, out_len = block.forward(x, out_len)

        # Final snake activation
        if self.final_alpha is not None:
            x = ttnn_snake_activation(x, self.final_alpha, self.final_beta)

        # Final conv
        if self.final_conv is not None:
            x_nlc = ttnn.reshape(x, (x.shape[0], out_len, x.shape[-1]))
            x_nlc, out_len = self.final_conv(x_nlc, out_len)
            x = ttnn.reshape(x_nlc, (x.shape[0], 1, out_len, x_nlc.shape[-1]))

        # Clamp to audio range [-1, 1]
        x = ttnn.clip(x, -1.0, 1.0)

        return x, out_len

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: codec tokens -> audio.

        Args:
            token_ids: [batch, num_quantizers, seq_len]

        Returns:
            audio: [batch, 1, num_samples]
        """
        # Codebook lookup on host (PyTorch)
        embeddings = self.codebook_lookup(token_ids)  # [batch, seq_len, latent_dim]
        seq_len = embeddings.shape[1]

        # Convert to TTNN tensor (NHWC format)
        embeddings_4d = embeddings.unsqueeze(1)  # [batch, 1, seq_len, latent_dim]
        embeddings_tt = ttnn.from_torch(
            embeddings_4d.to(torch.bfloat16),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Forward through decoder (TTNN)
        audio_tt, out_len = self.forward_ttnn(embeddings_tt, seq_len)

        # Convert back to PyTorch
        audio = ttnn.to_torch(audio_tt)  # [batch, 1, out_len, 1]
        audio = audio.squeeze(1).squeeze(-1)  # [batch, out_len]
        audio = audio.unsqueeze(1)  # [batch, 1, out_len]

        return audio


def test_ttnn_speech_decoder():
    """Test the full TTNN speech decoder."""
    from pathlib import Path

    from safetensors.torch import load_file

    print("Testing TTNN Speech Tokenizer Decoder...")

    # Load weights
    print("Loading weights...")
    from huggingface_hub import snapshot_download

    cache_path = Path(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["speech_tokenizer/*"]))
    st_path = cache_path / "speech_tokenizer" / "model.safetensors"
    state_dict = load_file(st_path)

    # Extract decoder weights (remove "decoder." prefix)
    # Keep quantizer weights for codebook lookup
    decoder_weights = {}
    prefix = "decoder."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            decoder_weights[k[len(prefix) :]] = v

    print(f"Loaded {len(decoder_weights)} decoder weights")
    print("Sample keys:", list(decoder_weights.keys())[:10])

    # Open device with larger L1 for conv operations
    device = ttnn.open_device(device_id=0, l1_small_size=131072)

    try:
        # Initialize decoder
        print("Initializing TTNN decoder...")
        decoder = TTNNSpeechTokenizerDecoder(device, decoder_weights)

        # Create test input (smaller seq_len to avoid L1 overflow during initial testing)
        batch_size = 1
        num_quantizers = 16
        seq_len = 32  # Small for initial testing

        token_ids = torch.randint(0, 2048, (batch_size, num_quantizers, seq_len))

        # Run forward
        print(f"Running forward with input shape: {token_ids.shape}")
        audio = decoder.forward(token_ids)
        print(f"Output audio shape: {audio.shape}")

        # Compute expected output length
        total_upsample = 1
        for r in decoder.config.upsampling_ratios:
            total_upsample *= r
        for r in decoder.config.upsample_rates:
            total_upsample *= r

        expected_samples = seq_len * total_upsample
        print(f"Expected samples: ~{expected_samples}")
        print(f"Actual samples: {audio.shape[-1]}")

        print("\nTTNN Speech Decoder test passed!")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_ttnn_speech_decoder()
