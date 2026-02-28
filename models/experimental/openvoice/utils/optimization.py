# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
TTNN optimization utilities for OpenVoice.

Provides memory configuration, sharding, and compute configuration helpers
for optimal performance on Tenstorrent hardware.
"""

from typing import Any, Optional, Tuple

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False


def get_l1_memory_config(
    shard_strategy: Optional[str] = None,
    shard_shape: Optional[Tuple[int, int]] = None,
) -> Any:
    """
    Get L1 memory configuration for activations.

    L1 is faster than DRAM but has limited size (~1MB per core).
    Use for intermediate activations that fit in L1.

    Args:
        shard_strategy: Optional sharding strategy ("height", "width", "block")
        shard_shape: Shape of each shard (height, width)

    Returns:
        Memory configuration for L1
    """
    if not TTNN_AVAILABLE:
        return None

    if shard_strategy is None:
        return ttnn.L1_MEMORY_CONFIG

    # Create sharded L1 config
    strategy_map = {
        "height": ttnn.ShardStrategy.HEIGHT,
        "width": ttnn.ShardStrategy.WIDTH,
        "block": ttnn.ShardStrategy.BLOCK,
    }
    strategy = strategy_map.get(shard_strategy, ttnn.ShardStrategy.HEIGHT)

    return ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=8, x=8),  # Use 8x8 grid
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def get_dram_memory_config() -> Any:
    """
    Get DRAM memory configuration for weights and large tensors.

    DRAM has more capacity but higher latency than L1.
    Use for weights and tensors that don't fit in L1.

    Returns:
        Memory configuration for DRAM
    """
    if not TTNN_AVAILABLE:
        return None
    return ttnn.DRAM_MEMORY_CONFIG


def get_compute_config(
    device: Any,
    math_fidelity: str = "HiFi4",
    fp32_acc: bool = False,
) -> Any:
    """
    Get compute kernel configuration for optimal performance.

    Args:
        device: TTNN device
        math_fidelity: "HiFi4" (highest), "HiFi2", "LoFi" (fastest)
        fp32_acc: Use FP32 accumulation for higher precision

    Returns:
        Compute kernel configuration
    """
    if not TTNN_AVAILABLE or device is None:
        return None

    fidelity_map = {
        "HiFi4": ttnn.MathFidelity.HiFi4,
        "HiFi2": ttnn.MathFidelity.HiFi2,
        "LoFi": ttnn.MathFidelity.LoFi,
    }
    fidelity = fidelity_map.get(math_fidelity, ttnn.MathFidelity.HiFi4)

    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=not fp32_acc,
        fp32_dest_acc_en=fp32_acc,
    )


def get_sdpa_program_config(
    q_chunk_size: int = 128,
    k_chunk_size: int = 128,
) -> Any:
    """
    Get program configuration for scaled dot product attention.

    Args:
        q_chunk_size: Chunk size for query sequence
        k_chunk_size: Chunk size for key sequence

    Returns:
        SDPA program configuration
    """
    if not TTNN_AVAILABLE:
        return None

    return ttnn.transformer.SDPAProgramConfig(
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )


def fused_attention(
    query: Any,
    key: Any,
    value: Any,
    attn_mask: Optional[Any] = None,
    scale: Optional[float] = None,
    is_causal: bool = False,
    device: Optional[Any] = None,
    memory_config: Optional[Any] = None,
    compute_config: Optional[Any] = None,
) -> Any:
    """
    Perform fused scaled dot product attention using FlashAttention-2.

    This is more efficient than manual matmul + softmax + matmul.

    Args:
        query: Query tensor [B, num_heads, seq_len, head_dim]
        key: Key tensor [B, num_heads, seq_len, head_dim]
        value: Value tensor [B, num_heads, seq_len, head_dim]
        attn_mask: Optional attention mask [B, 1, seq_len, seq_len]
        scale: Optional scale factor (default: 1/sqrt(head_dim))
        is_causal: Whether to apply causal masking
        device: TTNN device
        memory_config: Output memory configuration
        compute_config: Compute kernel configuration

    Returns:
        Attention output [B, num_heads, seq_len, head_dim]
    """
    if not TTNN_AVAILABLE:
        import torch.nn.functional as F

        # PyTorch fallback
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            scale=scale,
            is_causal=is_causal,
        )

    # Get default configs
    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG

    if compute_config is None and device is not None:
        compute_config = get_compute_config(device, "HiFi4")

    return ttnn.transformer.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        memory_config=memory_config,
        compute_kernel_config=compute_config,
    )


def apply_memory_config(tensor: Any, memory_config: Any, device: Any) -> Any:
    """
    Apply memory configuration to a tensor.

    Args:
        tensor: Input tensor
        memory_config: Target memory configuration
        device: TTNN device

    Returns:
        Tensor with new memory configuration
    """
    if not TTNN_AVAILABLE or device is None:
        return tensor

    return ttnn.to_memory_config(tensor, memory_config)


def optimize_for_inference(model_components: dict, device: Any) -> dict:
    """
    Apply inference optimizations to model components.

    This should be called after loading weights but before inference.

    Args:
        model_components: Dictionary of model components
        device: TTNN device

    Returns:
        Optimized model components
    """
    if not TTNN_AVAILABLE or device is None:
        return model_components

    # Get optimal compute config
    compute_config = get_compute_config(device, "HiFi4", fp32_acc=False)

    # Store compute config in components for use during forward pass
    model_components["_compute_config"] = compute_config
    model_components["_l1_memory_config"] = ttnn.L1_MEMORY_CONFIG
    model_components["_dram_memory_config"] = ttnn.DRAM_MEMORY_CONFIG

    return model_components
