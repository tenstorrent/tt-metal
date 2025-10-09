# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Language Model Head building block for output projection.

This module provides the final projection layer that converts hidden states
to vocabulary logits in language models. It supports distributed computation
across multiple devices with weight sharding.
"""

import math
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch

import ttnn

from ..ccl import AllReduceImplConfig, AllReduceSpec, CCLManager, all_reduce_forward


@dataclass
class LMHeadSpec:
    """
    Mathematical specification for the language model head.

    The LM head performs a linear projection from hidden dimension to
    vocabulary size, optionally followed by distributed reduction.
    """

    hidden_dim: int
    vocab_size: int
    num_devices: int = 1
    # Padded vocab size for efficient tensor operations
    padded_vocab_size: Optional[int] = None
    # Whether this is a galaxy configuration (affects sharding)
    is_galaxy: bool = False
    # Maximum columns per device to avoid L1 OOM
    max_columns_per_device: int = 16384
    # Add weight tying support
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.padded_vocab_size is None:
            # Pad to nearest multiple of 32 for tile alignment
            self.padded_vocab_size = ((self.vocab_size + 31) // 32) * 32

    def validate(self):
        """Validate spec constraints."""
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.num_devices > 0, "num_devices must be positive"
        assert self.max_columns_per_device > 0, "max_columns_per_device must be positive"
        assert self.padded_vocab_size >= self.vocab_size, "padded_vocab_size must be >= vocab_size"


@dataclass
class LMHeadImplConfig:
    """
    TTNN-specific implementation configuration for LM head.

    Contains device-specific optimizations and memory configurations
    for the output projection operation.
    """

    # Data types
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    output_dtype: ttnn.DataType = ttnn.bfloat8_b
    ccl_dtype: ttnn.DataType = ttnn.bfloat16

    # Compute configuration
    compute_kernel_config: Optional[dict] = None

    # Memory configurations
    output_memory_config: Optional[ttnn.MemoryConfig] = None
    weight_memory_config: Optional[ttnn.MemoryConfig] = None

    # Program configurations for each weight split
    program_configs: Optional[List[dict]] = None

    # CCL configuration
    num_reduce_scatter_links: int = 1
    num_all_gather_links: int = 2
    use_composite: bool = True

    # Sharding strategy
    shard_strategy: Literal["column", "row", "block"] = "column"

    # Tile configuration
    tile_padded_batch_rows: int = 32


def get_default_impl_config(
    spec: LMHeadSpec, device: str, mode: Literal["prefill", "decode"] = "prefill", strategy: str = "default"
) -> LMHeadImplConfig:
    """
    Return default implementation configuration for LM head.

    Args:
        spec: LM head specification
        device: Target device (e.g., "N150", "N300", "T3000", "TG")
        mode: Execution mode (prefill or decode)
        strategy: Performance strategy hint

    Returns:
        Default LMHeadImplConfig for the specified device and mode
    """
    # Base compute kernel config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    if device.startswith("N150"):
        # Single device, no distribution needed
        return LMHeadImplConfig(
            compute_kernel_config=compute_kernel_config,
            output_memory_config=ttnn.L1_MEMORY_CONFIG,
            weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_reduce_scatter_links=1,
            num_all_gather_links=1,
            use_composite=False,
        )
    elif device.startswith("N300"):
        # Two devices, basic sharding
        return LMHeadImplConfig(
            compute_kernel_config=compute_kernel_config,
            output_memory_config=ttnn.L1_MEMORY_CONFIG,
            weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_reduce_scatter_links=1,
            num_all_gather_links=2,
            use_composite=True,
        )
    elif device.startswith("T3000") or device == "T3K":
        # 8 devices, optimized sharding
        return LMHeadImplConfig(
            compute_kernel_config=compute_kernel_config,
            output_memory_config=ttnn.L1_MEMORY_CONFIG,
            weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_reduce_scatter_links=2,
            num_all_gather_links=2,
            use_composite=True,
            output_dtype=ttnn.bfloat8_b if mode == "decode" else ttnn.bfloat16,
        )
    elif device == "TG" or spec.is_galaxy:
        # Galaxy configuration with advanced sharding
        return LMHeadImplConfig(
            compute_kernel_config=compute_kernel_config,
            output_memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.L1_MEMORY_CONFIG,
            weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_reduce_scatter_links=1,
            num_all_gather_links=2,
            use_composite=True,
            shard_strategy="block" if spec.hidden_dim > 4096 else "column",
        )
    else:
        # Conservative defaults
        return LMHeadImplConfig(
            compute_kernel_config=compute_kernel_config,
            output_memory_config=ttnn.L1_MEMORY_CONFIG,
            weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


def prepare_weights(
    weight: torch.Tensor,
    spec: LMHeadSpec,
    impl_config: LMHeadImplConfig,
    mesh_device: ttnn.Device,
) -> List[ttnn.Tensor]:
    """
    Prepare and shard weights for distributed LM head computation.

    Args:
        weight: PyTorch weight tensor of shape (vocab_size, hidden_dim)
        spec: LM head specification
        impl_config: Implementation configuration
        mesh_device: Device mesh for distribution

    Returns:
        List of sharded weight tensors on device
    """
    # Transpose weight to (hidden_dim, vocab_size) for matmul
    weight = weight.t()

    # Calculate splits based on max columns per device
    size_per_device = spec.vocab_size // spec.num_devices
    if spec.is_galaxy:
        size_per_device = spec.padded_vocab_size // spec.num_devices

    num_splits = math.ceil(size_per_device / spec.max_columns_per_device)
    split_sizes = [min(size_per_device, spec.max_columns_per_device)] * (num_splits - 1)
    split_sizes.append(size_per_device - sum(split_sizes))  # Remaining columns

    output_weights = []

    if spec.is_galaxy:
        # Galaxy mode: pad and shard across 2D mesh
        padded_weight = torch.zeros(spec.hidden_dim, spec.padded_vocab_size)
        padded_weight[:, : spec.vocab_size] = weight

        # Use 2D mesh sharding for galaxy
        memory_config = impl_config.weight_memory_config or (
            ttnn.DRAM_MEMORY_CONFIG
            if spec.hidden_dim == 2048
            else ttnn.create_sharded_memory_config(
                shape=(spec.hidden_dim // 4, spec.padded_vocab_size // 8),
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.BLOCK,
            )
        )

        tt_weight = ttnn.as_tensor(
            padded_weight.unsqueeze(0).unsqueeze(0),  # Add batch dims
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(3, 2)),
            layout=ttnn.TILE_LAYOUT,
            dtype=impl_config.weight_dtype,
            memory_config=memory_config,
        )
        output_weights.append(tt_weight)
    else:
        # Standard mode: split and shard across devices
        for i, split_size in enumerate(split_sizes):
            # Gather splits from all devices for this chunk
            device_splits = []
            for device in range(spec.num_devices):
                start = device * size_per_device + sum(split_sizes[:i])
                end = start + split_size
                device_splits.append(weight[:, start:end])

            # Concatenate splits from all devices
            combined_split = torch.cat(device_splits, dim=-1)

            # Create memory config for this split
            memory_config = impl_config.weight_memory_config or ttnn.create_sharded_memory_config(
                shape=(spec.hidden_dim, math.ceil(combined_split.shape[-1] / spec.num_devices)),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
            )

            # Convert to TT tensor and shard across devices
            tt_weight = ttnn.as_tensor(
                combined_split.unsqueeze(0).unsqueeze(0),  # Add batch dims
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                layout=ttnn.TILE_LAYOUT,
                dtype=impl_config.weight_dtype,
                memory_config=memory_config,
            )
            output_weights.append(tt_weight)

    return output_weights


def lm_head_forward(
    hidden_states: ttnn.Tensor,
    weights: List[ttnn.Tensor],
    spec: LMHeadSpec,
    impl_config: LMHeadImplConfig,
    mesh_device: ttnn.Device,
    ccl_manager: CCLManager,
) -> ttnn.Tensor:
    """
    Forward pass of the language model head.

    Performs distributed matrix multiplication to project hidden states
    to vocabulary logits, with optional all-reduce across devices.

    Args:
        hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)
        weights: List of sharded weight tensors
        spec: LM head specification
        impl_config: Implementation configuration
        mesh_device: Device mesh
        ccl_manager: CCL manager for distributed operations

    Returns:
        Output logits of shape (batch, seq_len, vocab_size)
    """
    outputs = []

    # Process each weight split
    for i, weight in enumerate(weights):
        # Get program config for this split
        program_config = (
            impl_config.program_configs[i]
            if impl_config.program_configs and i < len(impl_config.program_configs)
            else None
        )

        # Perform linear projection
        output = ttnn.linear(
            hidden_states,
            weight,
            compute_kernel_config=impl_config.compute_kernel_config,
            program_config=program_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=impl_config.output_dtype,
        )

        # Convert from sharded to interleaved if needed
        if output.is_sharded():
            output = ttnn.sharded_to_interleaved(
                output, memory_config=impl_config.output_memory_config or ttnn.L1_MEMORY_CONFIG
            )

        outputs.append(output)

    # Concatenate outputs from all splits
    if len(outputs) > 1:
        output = ttnn.concat(outputs, dim=-1, memory_config=impl_config.output_memory_config or ttnn.L1_MEMORY_CONFIG)
    else:
        output = outputs[0]

    # Perform all-reduce across devices if multi-device
    if spec.num_devices > 1:
        # Set up all-reduce specs
        all_reduce_spec = AllReduceSpec(
            mesh_shape=tuple(mesh_device.shape),
            cluster_axis=1 if spec.is_galaxy else None,
            reduce_dim=3 if spec.is_galaxy else 0,
        )

        all_reduce_impl = AllReduceImplConfig(
            num_reduce_scatter_links=impl_config.num_reduce_scatter_links,
            num_all_gather_links=impl_config.num_all_gather_links,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=impl_config.ccl_dtype,
            sharded=False,
            use_composite=impl_config.use_composite,
        )

        output = all_reduce_forward(
            output,
            mesh_device,
            ccl_manager,
            all_reduce_spec,
            all_reduce_impl,
        )

    return output


def prefill_forward(
    hidden_states: ttnn.Tensor,
    weights: List[ttnn.Tensor],
    spec: LMHeadSpec,
    impl_config: LMHeadImplConfig,
    mesh_device: ttnn.Device,
    ccl_manager: CCLManager,
) -> ttnn.Tensor:
    """
    Prefill mode forward pass for LM head.

    Processes entire sequence at once with optimizations for prefill.

    Args:
        hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)
        weights: List of sharded weight tensors
        spec: LM head specification
        impl_config: Implementation configuration
        mesh_device: Device mesh
        ccl_manager: CCL manager for distributed operations

    Returns:
        Output logits of shape (batch, seq_len, vocab_size)
    """
    return lm_head_forward(
        hidden_states,
        weights,
        spec,
        impl_config,
        mesh_device,
        ccl_manager,
    )


def decode_forward(
    hidden_states: ttnn.Tensor,
    weights: List[ttnn.Tensor],
    spec: LMHeadSpec,
    impl_config: LMHeadImplConfig,
    mesh_device: ttnn.Device,
    ccl_manager: CCLManager,
) -> ttnn.Tensor:
    """
    Decode mode forward pass for LM head.

    Processes single token with optimizations for autoregressive generation.

    Args:
        hidden_states: Input tensor of shape (batch, 1, hidden_dim)
        weights: List of sharded weight tensors
        spec: LM head specification
        impl_config: Implementation configuration
        mesh_device: Device mesh
        ccl_manager: CCL manager for distributed operations

    Returns:
        Output logits of shape (batch, 1, vocab_size)
    """
    # Decode mode may use different dtype for efficiency
    if impl_config.output_dtype != hidden_states.dtype:
        hidden_states = ttnn.typecast(hidden_states, impl_config.output_dtype)

    return lm_head_forward(
        hidden_states,
        weights,
        spec,
        impl_config,
        mesh_device,
        ccl_manager,
    )
