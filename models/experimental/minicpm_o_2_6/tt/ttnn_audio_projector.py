# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Audio Projector for MiniCPM-o-2_6.

The audio projector maps Whisper encoder outputs (1024d) to Qwen2.5 token embeddings (3584d).
Architecture matches the official MultiModalProjector with AvgPool1d followed by 2-layer MLP.
"""

import ttnn
from loguru import logger

try:
    from .common import (
        get_weights_memory_config,
        get_activations_memory_config,
    )
except ImportError:
    from common import (
        get_weights_memory_config,
        get_activations_memory_config,
    )


class TtnnAudioProjector:
    """
    TTNN implementation of MultiModalProjector for audio modality.

    Official MiniCPM-o-2_6 Architecture (matching modeling_minicpmo.py):
    - Linear1: 1024d → 3584d with bias and ReLU
    - Linear2: 3584d → 3584d with bias
    - AvgPool1d: Reduces sequence length by factor of 2 (no hidden dim reduction)

    Input: [batch, seq_len, 1024] (Whisper encoder output)
    Output: [batch, seq_len//2, 3584] (Qwen2.5 embedding space)
    """

    def __init__(
        self,
        device: ttnn.Device,
        input_dim: int = 1024,  # Whisper hidden size
        output_dim: int = 3584,  # Qwen2.5 hidden size
        pool_step: int = 2,  # Pooling step from config (default 2)
    ):
        """
        Initialize TTNN Audio Projector matching official MiniCPM-o-2_6 architecture.

        Args:
            device: TTNN device
            input_dim: Input dimension (Whisper: 1024)
            output_dim: Output dimension (Qwen2.5: 3584)
            pool_step: Pooling step (from config.audio_pool_step, default 2)
        """
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pool_step = pool_step

        # Linear layer dimensions (matching official MultiModalProjector)
        self.linear1_in = input_dim  # 1024
        self.linear1_out = output_dim  # 3584
        self.linear2_in = output_dim  # 3584
        self.linear2_out = output_dim  # 3584

        # Initialize weight tensors (will be loaded)
        self.linear1_weight = None
        self.linear1_bias = None
        self.linear2_weight = None
        self.linear2_bias = None

        logger.info(
            f"✅ TtnnAudioProjector initialized: {input_dim}d → {output_dim}d → {output_dim}d (pool step={pool_step})"
        )

    def load_weights(self, weights_dict: dict):
        """
        Load audio projector weights from dictionary matching official MiniCPM-o-2_6 format.

        Expected keys (from modeling_minicpmo.py):
        - 'audio_projection_layer.linear1.weight': [3584, 1024] (out_features, in_features)
        - 'audio_projection_layer.linear1.bias': [3584]
        - 'audio_projection_layer.linear2.weight': [3584, 3584] (out_features, in_features)
        - 'audio_projection_layer.linear2.bias': [3584]

        Note: TTNN expects [in_features, out_features] format, so we transpose the weights.
        """
        # Linear1 weights: [3584, 1024] → transpose to [1024, 3584] for TTNN
        if "audio_projection_layer.linear1.weight" in weights_dict:
            linear1_weight = weights_dict["audio_projection_layer.linear1.weight"]
            # Transpose for TTNN: [out_features, in_features] → [in_features, out_features]
            linear1_weight_ttnn = linear1_weight.t()
            self.linear1_weight = ttnn.from_torch(
                linear1_weight_ttnn,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=get_weights_memory_config(),
            )
            self.linear1_weight = ttnn.to_device(self.linear1_weight, self.device)
            self.linear1_weight = ttnn.to_layout(self.linear1_weight, ttnn.TILE_LAYOUT)
            logger.info(f"✅ Loaded linear1 weight: {linear1_weight.shape} → {linear1_weight_ttnn.shape}")
        else:
            logger.warning("⚠️ audio_projection_layer.linear1.weight not found")

        if "audio_projection_layer.linear1.bias" in weights_dict:
            linear1_bias = weights_dict["audio_projection_layer.linear1.bias"]
            self.linear1_bias = ttnn.from_torch(
                linear1_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=get_weights_memory_config(),
            )
            self.linear1_bias = ttnn.to_device(self.linear1_bias, self.device)
            self.linear1_bias = ttnn.to_layout(self.linear1_bias, ttnn.TILE_LAYOUT)
            logger.info(f"✅ Loaded linear1 bias: {linear1_bias.shape}")
        else:
            logger.warning("⚠️ audio_projection_layer.linear1.bias not found")

        # Linear2 weights: [3584, 3584] → transpose to [3584, 3584] for TTNN
        if "audio_projection_layer.linear2.weight" in weights_dict:
            linear2_weight = weights_dict["audio_projection_layer.linear2.weight"]
            # Transpose for TTNN: [out_features, in_features] → [in_features, out_features]
            linear2_weight_ttnn = linear2_weight.t()
            self.linear2_weight = ttnn.from_torch(
                linear2_weight_ttnn,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=get_weights_memory_config(),
            )
            self.linear2_weight = ttnn.to_device(self.linear2_weight, self.device)
            self.linear2_weight = ttnn.to_layout(self.linear2_weight, ttnn.TILE_LAYOUT)
            logger.info(f"✅ Loaded linear2 weight: {linear2_weight.shape} → {linear2_weight_ttnn.shape}")
        else:
            logger.warning("⚠️ audio_projection_layer.linear2.weight not found")

        if "audio_projection_layer.linear2.bias" in weights_dict:
            linear2_bias = weights_dict["audio_projection_layer.linear2.bias"]
            self.linear2_bias = ttnn.from_torch(
                linear2_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=get_weights_memory_config(),
            )
            self.linear2_bias = ttnn.to_device(self.linear2_bias, self.device)
            self.linear2_bias = ttnn.to_layout(self.linear2_bias, ttnn.TILE_LAYOUT)
            logger.info(f"✅ Loaded linear2 bias: {linear2_bias.shape}")
        else:
            logger.warning("⚠️ audio_projection_layer.linear2.bias not found")

    def forward(self, audio_features: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of audio projector matching official MiniCPM-o-2_6 architecture.

        Official flow from modeling_minicpmo.py:
        1. audio_states: [B, T, 1024] (Whisper encoder output)
        2. audio_embeds = self.audio_projection_layer(audio_states)  # [B, T, 3584] (Linear1 + ReLU + Linear2)
        3. audio_embeds = audio_embeds.transpose(1, 2)  # [B, T, 3584] → [B, 3584, T]
        4. audio_embeds = self.audio_avg_pooler(audio_embeds)  # [B, 3584, T] → [B, 3584, T/2]
        5. audio_embeds = audio_embeds.transpose(1, 2)  # [B, 3584, T/2] → [B, T/2, 3584]

        Args:
            audio_features: [batch, seq_len, 1024] Whisper encoder output

        Returns:
            projected_features: [batch, seq_len//pool_step, 3584] Qwen2.5 embedding space
        """
        # Move input to device first
        x = ttnn.to_device(audio_features, self.device)
        batch, seq_len, hidden_dim = x.shape

        # Ensure input has correct hidden dimension
        assert hidden_dim == self.input_dim, f"Input hidden dim {hidden_dim} != expected {self.input_dim}"

        # Step 1: Apply 2-layer MLP projection (1024 → 3584 → 3584)
        if self.linear1_weight is not None:
            # Convert to TILE layout for matmul
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

            # Linear1: 1024 → 3584 + bias + ReLU
            x = ttnn.linear(
                x, self.linear1_weight, bias=self.linear1_bias, memory_config=get_activations_memory_config()
            )
            x = ttnn.relu(x)

            # Linear2: 3584 → 3584 + bias
            if self.linear2_weight is not None:
                x = ttnn.linear(
                    x, self.linear2_weight, bias=self.linear2_bias, memory_config=get_activations_memory_config()
                )

        # Step 2: Transpose for pooling [B, T, 3584] → [B, 3584, T]
        x = ttnn.transpose(x, -2, -1)  # [batch, seq_len, hidden_dim] → [batch, hidden_dim, seq_len]

        # Step 3: Apply AvgPool1d using manual implementation (avoid pooling format issues)
        # Reshape and average manually to avoid tensor format compatibility issues
        batch_size, hidden_dim, seq_len = x.shape

        # Convert to format suitable for manual pooling
        x = ttnn.to_dtype(x, ttnn.bfloat16)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Ensure seq_len is divisible by pool_step, truncate if necessary
        effective_seq_len = (seq_len // self.pool_step) * self.pool_step
        if effective_seq_len < seq_len:
            # Truncate to make divisible by pool_step
            x = x[:, :, :effective_seq_len]

        # Reshape to group by pooling windows: [batch, hidden_dim, seq_len // pool_step, pool_step]
        new_seq_len = effective_seq_len // self.pool_step
        x_reshaped = ttnn.reshape(x, [batch_size, hidden_dim, new_seq_len, self.pool_step])

        # Average along the pooling dimension (last dim)
        x_pooled = ttnn.mean(x_reshaped, dim=-1)  # Average over pool_step dimension

        # Reshape back to [batch, hidden_dim, seq_len/pool_step]
        x = ttnn.reshape(x_pooled, [batch_size, hidden_dim, new_seq_len])

        # Convert back to TILE layout for consistency with rest of the model
        # Keep in bfloat16 to avoid device tensor dtype conversion issues
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Step 4: Transpose back [B, 3584, T/pool_step] → [B, T/pool_step, 3584]
        x = ttnn.transpose(x, -2, -1)

        return x


"""
TTNN implementation of Audio Projector for MiniCPM-o-2_6.

The audio projector maps Whisper encoder outputs (1024d) to Qwen2.5 token embeddings (3584d).
Architecture matches the official MultiModalProjector with AvgPool1d followed by 2-layer MLP.
"""

import ttnn
from loguru import logger

try:
    from .common import (
        get_weights_memory_config,
        get_activations_memory_config,
    )
except ImportError:
    from common import (
        get_weights_memory_config,
        get_activations_memory_config,
    )


class TtnnAudioProjector:
    """
    TTNN implementation of MultiModalProjector for audio modality.

    Official MiniCPM-o-2_6 Architecture (matching modeling_minicpmo.py):
    - Linear1: 1024d → 3584d with bias and ReLU
    - Linear2: 3584d → 3584d with bias
    - AvgPool1d: Reduces sequence length by factor of 2 (no hidden dim reduction)

    Input: [batch, seq_len, 1024] (Whisper encoder output)
    Output: [batch, seq_len//2, 3584] (Qwen2.5 embedding space)
    """

    def __init__(
        self,
        device: ttnn.Device,
        input_dim: int = 1024,  # Whisper hidden size
        output_dim: int = 3584,  # Qwen2.5 hidden size
        pool_step: int = 2,  # Pooling step from config (default 2)
    ):
        """
        Initialize TTNN Audio Projector matching official MiniCPM-o-2_6 architecture.

        Args:
            device: TTNN device
            input_dim: Input dimension (Whisper: 1024)
            output_dim: Output dimension (Qwen2.5: 3584)
            pool_step: Pooling step (from config.audio_pool_step, default 2)
        """
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pool_step = pool_step

        # Linear layer dimensions (matching official MultiModalProjector)
        self.linear1_in = input_dim  # 1024
        self.linear1_out = output_dim  # 3584
        self.linear2_in = output_dim  # 3584
        self.linear2_out = output_dim  # 3584

        # Initialize weight tensors (will be loaded)
        self.linear1_weight = None
        self.linear1_bias = None
        self.linear2_weight = None
        self.linear2_bias = None

        logger.info(
            f"✅ TtnnAudioProjector initialized: {input_dim}d → {output_dim}d → {output_dim}d (pool step={pool_step})"
        )

    def load_weights(self, weights_dict: dict):
        """
        Load audio projector weights from dictionary matching official MiniCPM-o-2_6 format.

        Expected keys (from modeling_minicpmo.py):
        - 'audio_projection_layer.linear1.weight': [3584, 1024] (out_features, in_features)
        - 'audio_projection_layer.linear1.bias': [3584]
        - 'audio_projection_layer.linear2.weight': [3584, 3584] (out_features, in_features)
        - 'audio_projection_layer.linear2.bias': [3584]

        Note: TTNN expects [in_features, out_features] format, so we transpose the weights.
        """
        # Linear1 weights: [3584, 1024] → transpose to [1024, 3584] for TTNN
        if "audio_projection_layer.linear1.weight" in weights_dict:
            linear1_weight = weights_dict["audio_projection_layer.linear1.weight"]
            # Transpose for TTNN: [out_features, in_features] → [in_features, out_features]
            linear1_weight_ttnn = linear1_weight.t()
            self.linear1_weight = ttnn.from_torch(
                linear1_weight_ttnn,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=get_weights_memory_config(),
            )
            self.linear1_weight = ttnn.to_device(self.linear1_weight, self.device)
            self.linear1_weight = ttnn.to_layout(self.linear1_weight, ttnn.TILE_LAYOUT)
            logger.info(f"✅ Loaded linear1 weight: {linear1_weight.shape} → {linear1_weight_ttnn.shape}")
        else:
            logger.warning("⚠️ audio_projection_layer.linear1.weight not found")

        if "audio_projection_layer.linear1.bias" in weights_dict:
            linear1_bias = weights_dict["audio_projection_layer.linear1.bias"]
            self.linear1_bias = ttnn.from_torch(
                linear1_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=get_weights_memory_config(),
            )
            self.linear1_bias = ttnn.to_device(self.linear1_bias, self.device)
            self.linear1_bias = ttnn.to_layout(self.linear1_bias, ttnn.TILE_LAYOUT)
            logger.info(f"✅ Loaded linear1 bias: {linear1_bias.shape}")
        else:
            logger.warning("⚠️ audio_projection_layer.linear1.bias not found")

        # Linear2 weights: [3584, 3584] → transpose to [3584, 3584] for TTNN
        if "audio_projection_layer.linear2.weight" in weights_dict:
            linear2_weight = weights_dict["audio_projection_layer.linear2.weight"]
            # Transpose for TTNN: [out_features, in_features] → [in_features, out_features]
            linear2_weight_ttnn = linear2_weight.t()
            self.linear2_weight = ttnn.from_torch(
                linear2_weight_ttnn,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=get_weights_memory_config(),
            )
            self.linear2_weight = ttnn.to_device(self.linear2_weight, self.device)
            self.linear2_weight = ttnn.to_layout(self.linear2_weight, ttnn.TILE_LAYOUT)
            logger.info(f"✅ Loaded linear2 weight: {linear2_weight.shape} → {linear2_weight_ttnn.shape}")
        else:
            logger.warning("⚠️ audio_projection_layer.linear2.weight not found")

        if "audio_projection_layer.linear2.bias" in weights_dict:
            linear2_bias = weights_dict["audio_projection_layer.linear2.bias"]
            self.linear2_bias = ttnn.from_torch(
                linear2_bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=get_weights_memory_config(),
            )
            self.linear2_bias = ttnn.to_device(self.linear2_bias, self.device)
            self.linear2_bias = ttnn.to_layout(self.linear2_bias, ttnn.TILE_LAYOUT)
            logger.info(f"✅ Loaded linear2 bias: {linear2_bias.shape}")
        else:
            logger.warning("⚠️ audio_projection_layer.linear2.bias not found")

    def forward(self, audio_features: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass of audio projector matching official MiniCPM-o-2_6 architecture.

        Official flow from modeling_minicpmo.py:
        1. audio_states: [B, T, 1024] (Whisper encoder output)
        2. audio_embeds = self.audio_projection_layer(audio_states)  # [B, T, 3584] (Linear1 + ReLU + Linear2)
        3. audio_embeds = audio_embeds.transpose(1, 2)  # [B, T, 3584] → [B, 3584, T]
        4. audio_embeds = self.audio_avg_pooler(audio_embeds)  # [B, 3584, T] → [B, 3584, T/2]
        5. audio_embeds = audio_embeds.transpose(1, 2)  # [B, 3584, T/2] → [B, T/2, 3584]

        Args:
            audio_features: [batch, seq_len, 1024] Whisper encoder output

        Returns:
            projected_features: [batch, seq_len//pool_step, 3584] Qwen2.5 embedding space
        """
        # Move input to device first
        x = ttnn.to_device(audio_features, self.device)
        batch, seq_len, hidden_dim = x.shape

        # Ensure input has correct hidden dimension
        assert hidden_dim == self.input_dim, f"Input hidden dim {hidden_dim} != expected {self.input_dim}"

        # Step 1: Apply 2-layer MLP projection (1024 → 3584 → 3584)
        if self.linear1_weight is not None:
            # Convert to TILE layout for matmul
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

            # Linear1: 1024 → 3584 + bias + ReLU
            x = ttnn.linear(
                x, self.linear1_weight, bias=self.linear1_bias, memory_config=get_activations_memory_config()
            )
            x = ttnn.relu(x)

            # Linear2: 3584 → 3584 + bias
            if self.linear2_weight is not None:
                x = ttnn.linear(
                    x, self.linear2_weight, bias=self.linear2_bias, memory_config=get_activations_memory_config()
                )

        # Step 2: Transpose for pooling [B, T, 3584] → [B, 3584, T]
        x = ttnn.transpose(x, -2, -1)  # [batch, seq_len, hidden_dim] → [batch, hidden_dim, seq_len]

        # Step 3: Apply AvgPool1d using manual implementation (avoid pooling format issues)
        # Reshape and average manually to avoid tensor format compatibility issues
        batch_size, hidden_dim, seq_len = x.shape

        # Convert to format suitable for manual pooling
        x = ttnn.to_dtype(x, ttnn.bfloat16)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Ensure seq_len is divisible by pool_step, truncate if necessary
        effective_seq_len = (seq_len // self.pool_step) * self.pool_step
        if effective_seq_len < seq_len:
            # Truncate to make divisible by pool_step
            x = x[:, :, :effective_seq_len]

        # Reshape to group by pooling windows: [batch, hidden_dim, seq_len // pool_step, pool_step]
        new_seq_len = effective_seq_len // self.pool_step
        x_reshaped = ttnn.reshape(x, [batch_size, hidden_dim, new_seq_len, self.pool_step])

        # Average along the pooling dimension (last dim)
        x_pooled = ttnn.mean(x_reshaped, dim=-1)  # Average over pool_step dimension

        # Reshape back to [batch, hidden_dim, seq_len/pool_step]
        x = ttnn.reshape(x_pooled, [batch_size, hidden_dim, new_seq_len])

        # Convert back to TILE layout for consistency with rest of the model
        # Keep in bfloat16 to avoid device tensor dtype conversion issues
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Step 4: Transpose back [B, 3584, T/pool_step] → [B, T/pool_step, 3584]
        x = ttnn.transpose(x, -2, -1)

        return x
