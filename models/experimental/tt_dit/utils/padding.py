# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional
import math
import ttnn


class PaddingConfig:
    """
    Configuration for model padding to enable tensor parallelism.

    This class handles the calculation and validation of padding requirements
    for attention heads and hidden dimensions to make them divisible by the
    tensor parallel factor.
    """

    def __init__(
        self, original_heads: int, target_heads: int, head_dim: int, tensor_parallel_factor: Optional[int] = None
    ):
        """
        Initialize padding configuration.

        Args:
            original_heads: Original number of attention heads
            target_heads: Target number of heads (must be >= original_heads)
            head_dim: Dimension per attention head (remains constant)
            tensor_parallel_factor: TP factor for validation (optional)
        """
        self.original_heads = original_heads
        self.target_heads = target_heads
        self.head_dim = head_dim

        # Calculate derived dimensions
        self.original_dim = original_heads * head_dim
        self.target_dim = target_heads * head_dim

        # Padding amounts
        self.head_padding = target_heads - original_heads
        self.dim_padding = self.target_dim - self.original_dim

        # Validation
        self._validate(tensor_parallel_factor)

    def _validate(self, tensor_parallel_factor: Optional[int]):
        """Validate padding configuration."""
        if self.target_heads < self.original_heads:
            raise ValueError(f"target_heads ({self.target_heads}) must be >= original_heads ({self.original_heads})")

        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")

        if tensor_parallel_factor is not None:
            if self.target_heads % tensor_parallel_factor != 0:
                raise ValueError(
                    f"target_heads ({self.target_heads}) must be divisible by "
                    f"tensor_parallel_factor ({tensor_parallel_factor})"
                )

    @classmethod
    def from_tensor_parallel_factor(
        cls, original_heads: int, head_dim: int, tensor_parallel_factor: int
    ) -> "PaddingConfig":
        """
        Create padding config automatically based on tensor parallel factor.

        Args:
            original_heads: Original number of attention heads
            head_dim: Dimension per attention head
            tensor_parallel_factor: Desired TP factor

        Returns:
            PaddingConfig with target_heads rounded up to be divisible by TP factor
        """
        target_heads = math.ceil(original_heads / tensor_parallel_factor) * tensor_parallel_factor
        return cls(original_heads, target_heads, head_dim, tensor_parallel_factor)

    def is_padding_needed(self) -> bool:
        """Return True if any padding is needed."""
        return self.head_padding > 0

    def __repr__(self) -> str:
        return (
            f"PaddingConfig(original_heads={self.original_heads}, "
            f"target_heads={self.target_heads}, head_dim={self.head_dim}, "
            f"dim_padding={self.dim_padding})"
        )


def pad_weight_tensor(
    weight: torch.Tensor, padding_config: PaddingConfig, pad_input_dim: bool = False, pad_output_dim: bool = False
) -> torch.Tensor:
    """
    Pad a weight tensor according to padding configuration.

    Args:
        weight: Weight tensor to pad (typically 2D: [input_dim, output_dim])
        padding_config: Padding configuration
        pad_input_dim: Whether to pad the input dimension
        pad_output_dim: Whether to pad the output dimension

    Returns:
        Padded weight tensor
    """
    if not padding_config.is_padding_needed():
        return weight

    padded_weight = weight.clone()

    # Pad input dimension (dimension 0 for transposed weights)
    if pad_input_dim and padding_config.dim_padding > 0:
        input_padding = torch.zeros(
            padding_config.dim_padding, weight.shape[1], dtype=weight.dtype, device=weight.device
        )
        padded_weight = torch.cat([padded_weight, input_padding], dim=0)

    # Pad output dimension (dimension 1 for transposed weights)
    if pad_output_dim and padding_config.dim_padding > 0:
        output_padding = torch.zeros(
            padded_weight.shape[0], padding_config.dim_padding, dtype=weight.dtype, device=weight.device
        )
        padded_weight = torch.cat([padded_weight, output_padding], dim=1)

    return padded_weight


def pad_bias_tensor(bias: torch.Tensor, padding_config: PaddingConfig) -> torch.Tensor:
    """
    Pad a bias tensor according to padding configuration.

    Args:
        bias: Bias tensor to pad
        padding_config: Padding configuration

    Returns:
        Padded bias tensor
    """
    if not padding_config.is_padding_needed():
        return bias

    if padding_config.dim_padding > 0:
        bias_padding = torch.zeros(padding_config.dim_padding, dtype=bias.dtype, device=bias.device)
        return torch.cat([bias, bias_padding], dim=-1)

    return bias


def pad_qkv_weights(
    q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor, padding_config: PaddingConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad QKV weight tensors for attention layers using structured padding.

    Args:
        q_weight: Query projection weight (in_dim, out_dim)
        k_weight: Key projection weight (in_dim, out_dim)
        v_weight: Value projection weight (in_dim, out_dim)
        padding_config: Padding configuration

    Returns:
        Tuple of padded (q_weight, k_weight, v_weight)
    """
    if not padding_config.is_padding_needed():
        return q_weight, k_weight, v_weight

    original_dim = padding_config.original_dim
    target_dim = padding_config.target_dim

    def pad_qkv_weight(weight):
        in_dim, out_dim = weight.shape
        mult = out_dim // original_dim
        assert mult == 3, f"Only 3-way fused QKV weight matrices are supported, given weight shape {weight.shape}"

        # Reshape: (in_dim, mult_factor * original_dim) -> (in_dim, mult_factor, original_dim)
        weight = weight.reshape(weight.shape[0], mult_factor, original_dim)

        # Pad output dimension: (in_dim, mult_factor, original_dim) -> (in_dim, mult_factor, target_dim)
        output_padding = torch.zeros(
            weight.shape[0], mult_factor, target_dim - original_dim, dtype=weight.dtype, device=weight.device
        )
        weight = torch.cat([weight, output_padding], dim=2)

        # Reshape back: (in_dim, mult_factor, target_dim) -> (in_dim, mult_factor * target_dim)
        weight = weight.reshape(weight.shape[0], -1)

        return weight

    padded_q = pad_qkv_weight(q_weight)
    padded_k = pad_qkv_weight(k_weight)
    padded_v = pad_qkv_weight(v_weight)

    return padded_q, padded_k, padded_v


def pad_qkv_biases(
    q_bias: Optional[torch.Tensor],
    k_bias: Optional[torch.Tensor],
    v_bias: Optional[torch.Tensor],
    padding_config: PaddingConfig,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Pad QKV bias tensors for attention layers using structured padding.

    Args:
        q_bias: Query projection bias (can be None)
        k_bias: Key projection bias (can be None)
        v_bias: Value projection bias (can be None)
        padding_config: Padding configuration

    Returns:
        Tuple of padded (q_bias, k_bias, v_bias)
    """
    if not padding_config.is_padding_needed():
        return q_bias, k_bias, v_bias

    original_dim = padding_config.original_dim
    target_dim = padding_config.target_dim

    def pad_qkv_bias(bias):
        if bias is None:
            return None

        orig_shape = bias.shape
        mult_factor = orig_shape[0] // original_dim
        assert mult_factor == 3, "Only 3-way fused QKV bias matrices are supported"

        bias = bias.reshape(mult_factor, original_dim)

        # Pad: (mult_factor, original_dim) -> (mult_factor, target_dim)
        bias_padding = torch.zeros(mult_factor, target_dim - original_dim, dtype=bias.dtype, device=bias.device)
        bias = torch.cat([bias, bias_padding], dim=1)

        # Reshape back: (mult_factor, target_dim) -> (mult_factor * target_dim,)
        bias = bias.reshape(orig_shape)

        return bias

    padded_q_bias = pad_qkv_bias(q_bias)
    padded_k_bias = pad_qkv_bias(k_bias)
    padded_v_bias = pad_qkv_bias(v_bias)

    return padded_q_bias, padded_k_bias, padded_v_bias


def get_padded_vision_seq_len(N, num_devices):
    divisor = ttnn.TILE_SIZE * num_devices

    # Calculate padding needed to make seq_len divisible by both tile size and num_devices
    padded_seq_len = math.ceil(N / divisor) * divisor
    padding = padded_seq_len - N
    shard_size = padded_seq_len // num_devices
    return padded_seq_len


def pad_vision_seq_parallel(tensor, num_devices):
    """
    Sequence parallelism shards the vision tensor in dim2.
    dim2 must be divisible by tile size and num_devices.
    """
    seq_len = tensor.shape[2]
    padded_seq_len = get_padded_vision_seq_len(seq_len, num_devices)
    pad_len = padded_seq_len - seq_len

    # Pad the sequence length dimension (dim2) on the right
    if pad_len > 0:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len))

    return tensor
