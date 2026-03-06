# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Unified MoE configuration for both GPT-OSS and DeepSeek backends."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import ttnn


@dataclass
class MoEUnifiedConfig:
    """Unified configuration for MoE forward pass supporting both GPT-OSS and DeepSeek backends.

    This configuration provides a single interface to control chunking, all-to-all operations,
    expert computation, and memory management across different MoE implementations.
    """

    # Chunking configuration
    enable_chunking: bool = False
    chunk_size: Optional[int] = None  # None means no chunking
    chunk_dim: int = 0  # 0=batch, 2=sequence

    # Dispatch configuration for all_to_all_dispatch
    dispatch_config: Dict = field(default_factory=dict)
    # Expected keys:
    # {
    #     "cluster_axis": 0,
    #     "memory_config": ttnn.L1_MEMORY_CONFIG or ttnn.DRAM_MEMORY_CONFIG,
    #     "num_links": 4,
    #     "topology": ttnn.Topology.Ring,
    #     "output_concat_dim": 2,
    # }

    # Expert execution type
    expert_type: str = "routed"  # "routed", "throughput_decode", "throughput_prefill"
    expert_config: Dict = field(default_factory=dict)  # Backend-specific expert configuration

    # Combine configuration for all_to_all_combine
    combine_config: Dict = field(default_factory=dict)
    # Expected keys:
    # {
    #     "cluster_axis": 0,
    #     "memory_config": ttnn.L1_MEMORY_CONFIG or ttnn.DRAM_MEMORY_CONFIG,
    #     "num_links": 4,
    #     "topology": ttnn.Topology.Ring,
    #     "output_shard_dim": 2 or -1,
    # }

    # All-reduce configuration (GPT-OSS only)
    enable_all_reduce: bool = False
    all_reduce_config: Dict = field(default_factory=dict)
    # Expected keys:
    # {
    #     "cluster_axis": 1,  # Column-wise for GPT-OSS
    #     "num_links": 4,
    #     "topology": ttnn.Topology.Ring,
    #     "memory_config": ttnn.L1_MEMORY_CONFIG or ttnn.DRAM_MEMORY_CONFIG,
    # }

    # Memory configurations
    input_memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    output_memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    intermediate_memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG


def get_gptoss_decode_config() -> MoEUnifiedConfig:
    """GPT-OSS decode configuration with exact ttnn parameters."""
    return MoEUnifiedConfig(
        enable_chunking=False,
        dispatch_config={
            "cluster_axis": 0,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "num_links": 4,
            "topology": ttnn.Topology.Ring,
            "output_concat_dim": 2,
        },
        expert_type="throughput_decode",
        combine_config={
            "cluster_axis": 0,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "num_links": 4,
            "topology": ttnn.Topology.Ring,
            "output_shard_dim": 2,
        },
        enable_all_reduce=True,
        all_reduce_config={
            "cluster_axis": 1,  # Column-wise
            "num_links": 4,
            "topology": ttnn.Topology.Ring,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
        },
        input_memory_config=ttnn.L1_MEMORY_CONFIG,
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Tests expect DRAM
        intermediate_memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def get_gptoss_prefill_config() -> MoEUnifiedConfig:
    """GPT-OSS prefill configuration."""
    return MoEUnifiedConfig(
        enable_chunking=False,  # Start without chunking, add if needed
        dispatch_config={
            "cluster_axis": 0,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "num_links": 4,
            "topology": ttnn.Topology.Ring,
            "output_concat_dim": 2,
        },
        expert_type="throughput_prefill",
        combine_config={
            "cluster_axis": 0,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "num_links": 4,
            "topology": ttnn.Topology.Ring,
            "output_shard_dim": 2,
        },
        enable_all_reduce=True,
        all_reduce_config={
            "cluster_axis": 1,
            "num_links": 4,
            "topology": ttnn.Topology.Ring,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        },
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def get_deepseek_config(mode: str) -> MoEUnifiedConfig:
    """DeepSeek configuration."""
    is_decode = mode == "decode"
    memory_config = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG

    return MoEUnifiedConfig(
        enable_chunking=not is_decode,  # Chunk for prefill
        chunk_size=32,  # Will be dynamically adjusted
        chunk_dim=0,  # Batch dimension chunking
        dispatch_config={
            "cluster_axis": 0,
            "memory_config": memory_config,
            "num_links": 1,
            "topology": ttnn.Topology.Linear,
            "output_concat_dim": 2,  # Use positive index for token dimension
        },
        expert_type="routed",
        combine_config={
            "cluster_axis": 0,
            "memory_config": memory_config,
            "num_links": 1,
            "topology": ttnn.Topology.Linear,
            "output_shard_dim": 2,  # Use positive index for token dimension
        },
        enable_all_reduce=False,  # DeepSeek handles externally
        input_memory_config=memory_config,
        output_memory_config=memory_config,
        intermediate_memory_config=memory_config,
    )
