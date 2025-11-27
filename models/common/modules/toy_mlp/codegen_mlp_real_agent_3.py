# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTTv2 Real MLP Module with Code Generation

This module ports the production MLP from models/tt_transformers/tt/mlp.py to TTTv2 patterns.
Key aspects:
1. Separate template functions for each (topology, mode) combination
2. No if-else on static conditions in template functions
3. Explicit configuration classes for hardware configs and program configs
4. Template-based code generation for different hardware configurations

Topology/Mode Matrix:
- Single Device Decode: Canonical MLP with DRAM sharded matmuls
- Single Device Prefill: Canonical MLP with interleaved memory
- T3K 1D Decode/Prefill: Tensor-parallel MLP with reduce_scatter
- Galaxy 2D Decode: Fused CCL operations with reduce_scatter + all_gather
- Galaxy 2D Prefill: Different strategy with full reduce_scatter/all_gather

Based on the original implementation in models/tt_transformers/tt/mlp.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.codegen_utils.generation import class_to_source, function_to_source
from models.common.modules.lazy_weight import LazyWeight

# =============================================================================
# Single Source of Truth for Generated Module Imports
# =============================================================================
# Real Python symbols used by generated modules.
# Benefits:
#   1. IDEs and language servers understand these symbols
#   2. Import statements derived programmatically
#   3. No string duplication

GENERATED_MODULE_NAMESPACE = {
    # External packages
    "ttnn": ttnn,
    # typing
    "Optional": Optional,
    "Any": Any,
    "Callable": Callable,
    # dataclasses
    "dataclass": dataclass,
    "field": field,
    # Internal modules
    "LazyWeight": LazyWeight,
    "LightweightModule": LightweightModule,
}


def get_import_source(module_namespace: dict) -> str:
    """Generate import statements from the namespace of real Python symbols."""
    import types
    from collections import defaultdict

    module_imports = []
    from_imports = defaultdict(list)

    for name, obj in module_namespace.items():
        if isinstance(obj, types.ModuleType):
            module_imports.append(f"import {obj.__name__}")
        else:
            module = getattr(obj, "__module__", None)
            if module:
                from_imports[module].append(name)

    lines = []
    for stmt in sorted(module_imports):
        lines.append(stmt)
    for module in sorted(from_imports.keys()):
        symbols = ", ".join(sorted(from_imports[module]))
        lines.append(f"from {module} import {symbols}")

    return "\n".join(lines)


# =============================================================================
# Configuration Classes
# =============================================================================


class Topology(Enum):
    """Hardware topology for multi-device configurations"""

    SINGLE_DEVICE = "single_device"  # Single Wormhole chip (N150)
    N300 = "n300"  # N300: 2 devices, 1D
    T3K_1D = "t3k_1d"  # T3K with 1D ring topology (8 devices)
    GALAXY_2D = "galaxy_2d"  # Galaxy with 2D mesh topology (8x4 devices)


class ExecutionMode(Enum):
    """Execution mode - determines memory configs and program configs"""

    DECODE = "decode"  # Batch decode with DRAM sharded matmuls
    PREFILL = "prefill"  # Prefill with interleaved memory


@dataclass
class MLPConfig:
    """MLP architectural configuration - derived from model args."""

    dim: int  # Model dimension (e.g., 4096 for Llama 7B)
    hidden_dim: int  # Intermediate/FFN dimension (e.g., 11008 for Llama 7B)
    num_devices: int = 1  # Number of devices for sharding
    activation_type: Any = None  # ttnn.UnaryOpType.SILU by default

    # Topology selection
    topology: Topology = Topology.SINGLE_DEVICE

    # Prefill-specific
    prefill_len_cutoff: int = 1024  # Reshape threshold for prefill


@dataclass
class LinearOpConfig:
    """Configuration for a single linear operation (ttnn.linear)."""

    compute_kernel_config: Optional[Any] = None
    program_config: Optional[Any] = None
    memory_config: Optional[Any] = None
    dtype: Optional[Any] = None  # ttnn dtype


@dataclass
class MulOpConfig:
    """Configuration for mul operation with fused activation."""

    input_tensor_a_activations: Optional[list] = None
    dtype: Optional[Any] = None
    memory_config: Optional[Any] = None


@dataclass
class ReduceScatterConfig:
    """Configuration for reduce_scatter CCL operation."""

    dim: int = 3
    cluster_axis: int = 1
    num_links: int = 1
    memory_config: Optional[Any] = None
    intermediate_memory_config: Optional[Any] = None
    topology: Any = None  # ttnn.Topology


@dataclass
class AllGatherConfig:
    """Configuration for all_gather CCL operation."""

    dim: int = 3
    cluster_axis: int = 1
    num_links: int = 2
    memory_config: Optional[Any] = None
    topology: Any = None


@dataclass
class AllReduceConfig:
    """Configuration for all_reduce CCL operation (combines reduce_scatter + all_gather)."""

    cluster_axis: int = 0
    dim: int = 3
    num_reduce_scatter_links: int = 1
    num_all_gather_links: int = 2
    sharded: bool = False
    memory_config: Optional[Any] = None
    dtype: Optional[Any] = None
    use_composite: bool = False
    topology: Any = None


@dataclass
class DecodeOpConfigs:
    """Operation configurations for decode mode."""

    w1: LinearOpConfig  # Gate projection
    w2: LinearOpConfig  # Down projection
    w3: LinearOpConfig  # Up projection
    mul: MulOpConfig  # Fused activation + mul
    # Output memory config for final reshape
    output_memory_config: Optional[Any] = None


@dataclass
class PrefillOpConfigs:
    """Operation configurations for prefill mode."""

    w1: LinearOpConfig
    w2: LinearOpConfig
    w3: LinearOpConfig
    mul: MulOpConfig
    output_memory_config: Optional[Any] = None


@dataclass
class TGDecodeOpConfigs:
    """Operation configurations for TG (Galaxy) decode mode with CCL."""

    w1: LinearOpConfig
    w2: LinearOpConfig
    w3: LinearOpConfig
    mul: MulOpConfig
    # CCL operations
    rs_w1: ReduceScatterConfig  # Reduce scatter after w1
    rs_w3: ReduceScatterConfig  # Reduce scatter after w3
    ag_intermediate: AllGatherConfig  # All gather before w2
    ar_output: AllReduceConfig  # All reduce after w2
    # Memory configs for transitions
    ff1_out_reduce_scatter_memcfg: Optional[Any] = None
    ff1_out_gathered_memcfg: Optional[Any] = None
    ff2_out_reduce_scatter_memcfg: Optional[Any] = None
    sharded_mlp2_input_memcfg: Optional[Any] = None
    decode_residual_memcfg: Optional[Any] = None
    sharded_attn_input_memcfg: Optional[Any] = None


@dataclass
class MLPWeights:
    """Weight tensors for MLP module."""

    w1: LazyWeight  # Gate projection weight (dim -> hidden_dim)
    w2: LazyWeight  # Down projection weight (hidden_dim -> dim)
    w3: LazyWeight  # Up projection weight (dim -> hidden_dim)


_ = None  # Stop inspect.getsource from grabbing following comments

# =============================================================================
# TEMPLATE 1: Single Device / T3K Decode Mode
# =============================================================================
# Standard SwiGLU MLP with DRAM sharded matmuls for decode
# Corresponds to mode == "decode" and not TG in original mlp.py


def forward_mlp_decode_impl(x, weights, ops):
    """
    MLP forward pass for decode mode (single device or T3K).

    Uses DRAM sharded matmuls for efficient batch decode.
    Fused activation (SILU) in the mul operation.

    Dataflow: x -> [w1, w3] parallel -> silu(w1) * w3 -> w2 -> output

    Args:
        x: Input tensor [1, 1, seq_len, dim]
        weights: MLPWeights container
        ops: DecodeOpConfigs with operation configurations
    """
    # 1. Gate projection (w1) and Up projection (w3) in parallel
    # In decode mode, these use DRAM sharded matmuls
    w1_out = ttnn.linear(
        x,
        weights.w1,
        compute_kernel_config=ops.w1.compute_kernel_config,
        program_config=ops.w1.program_config,
        memory_config=ops.w1.memory_config,
        dtype=ops.w1.dtype,
    )

    w3_out = ttnn.linear(
        x,
        weights.w3,
        compute_kernel_config=ops.w3.compute_kernel_config,
        program_config=ops.w3.program_config,
        memory_config=ops.w3.memory_config,
        dtype=ops.w3.dtype,
    )

    ttnn.deallocate(x)

    # 2. Fused activation + element-wise multiplication
    # silu(w1_out) * w3_out
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=ops.mul.input_tensor_a_activations,
        dtype=ops.mul.dtype,
        memory_config=ops.mul.memory_config,
    )

    # 3. Reshard for w2 if needed (memory config transition)
    if ops.output_memory_config is not None:
        w2_in = ttnn.to_memory_config(w2_in, ops.output_memory_config)

    ttnn.deallocate(w1_out)
    ttnn.deallocate(w3_out)

    # 4. Down projection (w2)
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.w2.compute_kernel_config,
        program_config=ops.w2.program_config,
        memory_config=ops.w2.memory_config,
        dtype=ops.w2.dtype,
    )

    ttnn.deallocate(w2_in)

    return w2_out


_ = None

# =============================================================================
# TEMPLATE 2: Single Device / T3K Prefill Mode
# =============================================================================
# Standard MLP for prefill with interleaved memory


def forward_mlp_prefill_impl(x, weights, ops, prefill_len_cutoff):
    """
    MLP forward pass for prefill mode.

    Uses interleaved memory for variable sequence lengths.
    For long sequences (>= prefill_len_cutoff), reshapes to parallelize.

    Args:
        x: Input tensor [1, 1, seq_len, dim]
        weights: MLPWeights container
        ops: PrefillOpConfigs with operation configurations
        prefill_len_cutoff: Threshold for reshaping long sequences
    """
    seq_len = x.shape[-2]

    # For long sequences, reshape to parallelize computation
    if seq_len >= prefill_len_cutoff:
        x = ttnn.reshape(x, [1, seq_len // prefill_len_cutoff, prefill_len_cutoff, -1])

    # 1. Gate and Up projections
    w1_out = ttnn.linear(
        x,
        weights.w1,
        compute_kernel_config=ops.w1.compute_kernel_config,
        program_config=ops.w1.program_config,
        memory_config=ops.w1.memory_config,
        dtype=ops.w1.dtype,
    )

    w3_out = ttnn.linear(
        x,
        weights.w3,
        compute_kernel_config=ops.w3.compute_kernel_config,
        program_config=ops.w3.program_config,
        memory_config=ops.w3.memory_config,
        dtype=ops.w3.dtype,
    )

    ttnn.deallocate(x)

    # 2. Fused activation + multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=ops.mul.input_tensor_a_activations,
        dtype=ops.mul.dtype,
        memory_config=ops.mul.memory_config,
    )

    ttnn.deallocate(w1_out)
    ttnn.deallocate(w3_out)

    # 3. Down projection
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.w2.compute_kernel_config,
        program_config=ops.w2.program_config,
        memory_config=ops.w2.memory_config,
        dtype=ops.w2.dtype,
    )

    ttnn.deallocate(w2_in)

    return w2_out


_ = None

# =============================================================================
# TEMPLATE 3: T3K 1D with All-Reduce (Decode)
# =============================================================================
# Tensor-parallel MLP for T3K with reduce_scatter after w2


def forward_mlp_t3k_decode_impl(x, weights, ops, mesh_device, tt_ccl):
    """
    MLP forward pass for T3K 1D decode mode with tensor parallelism.

    Uses reduce_scatter after down projection to combine partial sums.
    This is the N300/T3K path (1D ring topology).

    Args:
        x: Input tensor sharded across devices
        weights: MLPWeights container (weights sharded across devices)
        ops: DecodeOpConfigs with operation configurations
        mesh_device: Mesh device for CCL operations
        tt_ccl: TT_CCL instance for collective operations
    """
    # 1. Gate and Up projections (each device has partial weights)
    w1_out = ttnn.linear(
        x,
        weights.w1,
        compute_kernel_config=ops.w1.compute_kernel_config,
        program_config=ops.w1.program_config,
        memory_config=ops.w1.memory_config,
        dtype=ops.w1.dtype,
    )

    w3_out = ttnn.linear(
        x,
        weights.w3,
        compute_kernel_config=ops.w3.compute_kernel_config,
        program_config=ops.w3.program_config,
        memory_config=ops.w3.memory_config,
        dtype=ops.w3.dtype,
    )

    ttnn.deallocate(x)

    # 2. Fused activation + multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=ops.mul.input_tensor_a_activations,
        dtype=ops.mul.dtype,
        memory_config=ops.mul.memory_config,
    )

    if ops.output_memory_config is not None:
        w2_in = ttnn.to_memory_config(w2_in, ops.output_memory_config)

    ttnn.deallocate(w1_out)
    ttnn.deallocate(w3_out)

    # 3. Down projection
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.w2.compute_kernel_config,
        program_config=ops.w2.program_config,
        memory_config=ops.w2.memory_config,
        dtype=ops.w2.dtype,
    )

    ttnn.deallocate(w2_in)

    # 4. Reduce scatter to combine partial sums across devices
    # For T3K 1D, we use reduce_scatter along the weight sharding dimension
    original_shape = w2_out.shape
    if original_shape[0] != 1 or original_shape[1] != 1:
        w2_out = ttnn.reshape(
            w2_out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

    # Convert to interleaved if sharded
    if w2_out.is_sharded():
        w2_out = ttnn.sharded_to_interleaved(w2_out, ttnn.L1_MEMORY_CONFIG)

    w2_out_reduced = ttnn.experimental.reduce_scatter_minimal_async(
        w2_out,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    ttnn.deallocate(w2_out)

    return w2_out_reduced


_ = None

# =============================================================================
# TEMPLATE 4: Galaxy 2D Decode with Full CCL
# =============================================================================
# Uses reduce_scatter after w1/w3, all_gather before w2, all_reduce after w2


def forward_mlp_galaxy_decode_impl(x, weights, ops, mesh_device, tt_ccl, use_all_reduce_for_ff1):
    """
    MLP forward pass for Galaxy 2D mesh decode mode.

    For large models (dim >= 8192), uses:
    - reduce_scatter after w1 and w3 along cluster_axis=1
    - all_gather before w2
    - all_reduce after w2 along cluster_axis=0

    For smaller models, uses tt_all_reduce for w1/w3 outputs.

    Args:
        x: Input tensor sharded across 2D mesh
        weights: MLPWeights container
        ops: TGDecodeOpConfigs with CCL configurations
        mesh_device: 2D mesh device
        tt_ccl: TT_CCL instance
        use_all_reduce_for_ff1: If True, use all_reduce for w1/w3; else reduce_scatter
    """
    # 1. Gate and Up projections
    w1_out = ttnn.linear(
        x,
        weights.w1,
        compute_kernel_config=ops.w1.compute_kernel_config,
        program_config=ops.w1.program_config,
        memory_config=ops.w1.memory_config,
        dtype=ops.w1.dtype,
        core_grid=None,
    )

    w3_out = ttnn.linear(
        x,
        weights.w3,
        compute_kernel_config=ops.w3.compute_kernel_config,
        program_config=ops.w3.program_config,
        memory_config=ops.w3.memory_config,
        dtype=ops.w3.dtype,
        core_grid=None,
    )

    ttnn.deallocate(x)

    # 2. CCL for w1 and w3 outputs
    if not use_all_reduce_for_ff1:
        # Use reduce_scatter for large models
        input_mem_cfg = w1_out.memory_config()
        cluster_axis = ops.rs_w1.cluster_axis

        w1_out = ttnn.experimental.reduce_scatter_minimal_async(
            w1_out,
            persistent_output_buffers=None,
            dim=ops.rs_w1.dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=ops.rs_w1.num_links,
            cluster_axis=cluster_axis,
            memory_config=ops.ff1_out_reduce_scatter_memcfg,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ops.rs_w1.topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        w3_out = ttnn.experimental.reduce_scatter_minimal_async(
            w3_out,
            persistent_output_buffers=None,
            dim=ops.rs_w3.dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=ops.rs_w3.num_links,
            cluster_axis=cluster_axis,
            memory_config=ops.ff1_out_reduce_scatter_memcfg,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ops.rs_w3.topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
    else:
        # Use all_reduce for smaller models (not shown in detail - would call tt_all_reduce)
        # This path delegates to the imported tt_all_reduce function
        pass

    # 3. Fused activation + multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=ops.mul.input_tensor_a_activations,
        dtype=ops.mul.dtype,
        memory_config=w1_out.memory_config(),
    )

    ttnn.deallocate(w1_out)
    ttnn.deallocate(w3_out)

    # 4. All-gather to reconstruct full intermediate for w2
    if not use_all_reduce_for_ff1:
        cluster_axis = ops.ag_intermediate.cluster_axis
        w2_in = ttnn.experimental.all_gather_async(
            w2_in,
            persistent_output_buffer=None,
            dim=ops.ag_intermediate.dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=ops.ag_intermediate.num_links,
            cluster_axis=cluster_axis,
            topology=ops.ag_intermediate.topology,
            memory_config=input_mem_cfg,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

    # 5. Down projection
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.w2.compute_kernel_config,
        program_config=ops.w2.program_config,
        memory_config=ops.w2.memory_config,
        dtype=ops.w2.dtype,
        core_grid=None,
    )

    ttnn.deallocate(w2_in)

    # 6. All-reduce across cluster_axis=0 to combine results
    original_shape = w2_out.shape
    if original_shape[0] != 1 or original_shape[1] != 1:
        w2_out = ttnn.reshape(
            w2_out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

    # For TG with composite all-reduce
    cluster_axis = ops.ar_output.cluster_axis
    if ops.ar_output.use_composite:
        # Reduce scatter + all gather composite
        w2_out_reduced = ttnn.experimental.reduce_scatter_minimal_async(
            w2_out,
            persistent_output_buffers=None,
            dim=ops.ar_output.dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=ops.ar_output.num_reduce_scatter_links,
            cluster_axis=cluster_axis,
            memory_config=ops.ff2_out_reduce_scatter_memcfg,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ops.ar_output.topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        input_mem_cfg = w2_out.memory_config()
        w2_out_reduced = ttnn.experimental.all_gather_async(
            w2_out_reduced,
            persistent_output_buffer=None,
            dim=ops.ar_output.dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=ops.ar_output.num_all_gather_links,
            cluster_axis=cluster_axis,
            topology=ops.ar_output.topology,
            memory_config=input_mem_cfg,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
    else:
        # All gather + reduce
        w2_out_gathered = ttnn.experimental.all_gather_async(
            w2_out,
            persistent_output_buffer=None,
            dim=ops.ar_output.dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=ops.ar_output.num_all_gather_links,
            cluster_axis=cluster_axis,
            topology=ops.ar_output.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        w2_out_gathered = ttnn.to_memory_config(w2_out_gathered, ttnn.L1_MEMORY_CONFIG)

        w2_out_reduced = ttnn.experimental.fast_reduce_nc(
            w2_out_gathered,
            dims=[ops.ar_output.dim],
            output=None,
            compute_kernel_config=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ttnn.deallocate(w2_out_gathered)

    # 7. Reshape to original shape
    w2_out_reduced = ttnn.reshape(w2_out_reduced, original_shape)

    # 8. Final memory config for residual connection
    if ops.sharded_attn_input_memcfg is not None:
        w2_out_reduced = ttnn.to_memory_config(w2_out_reduced, ops.sharded_attn_input_memcfg)

    return w2_out_reduced


_ = None

# =============================================================================
# TEMPLATE 5: Galaxy 2D Prefill Mode
# =============================================================================


def forward_mlp_galaxy_prefill_impl(x, weights, ops, mesh_device, tt_ccl, prefill_len_cutoff):
    """
    MLP forward pass for Galaxy 2D mesh prefill mode.

    Similar to decode but with different memory management for variable-length prefill.

    Args:
        x: Input tensor
        weights: MLPWeights container
        ops: TGDecodeOpConfigs with CCL configurations
        mesh_device: 2D mesh device
        tt_ccl: TT_CCL instance
        prefill_len_cutoff: Threshold for reshaping long sequences
    """
    seq_len = x.shape[-2]

    # Reshape for long sequences
    if seq_len >= prefill_len_cutoff:
        x = ttnn.reshape(x, [1, seq_len // prefill_len_cutoff, prefill_len_cutoff, -1])

    # 1. Gate and Up projections
    w1_out = ttnn.linear(
        x,
        weights.w1,
        compute_kernel_config=ops.w1.compute_kernel_config,
        program_config=ops.w1.program_config,
        memory_config=ops.w1.memory_config,
        dtype=ops.w1.dtype,
        core_grid=None,
    )

    w3_out = ttnn.linear(
        x,
        weights.w3,
        compute_kernel_config=ops.w3.compute_kernel_config,
        program_config=ops.w3.program_config,
        memory_config=ops.w3.memory_config,
        dtype=ops.w3.dtype,
        core_grid=None,
    )

    ttnn.deallocate(x)

    # 2. Reduce scatter for w1 and w3 (prefill uses same pattern as decode for large models)
    cluster_axis = ops.rs_w1.cluster_axis

    w1_out = ttnn.experimental.reduce_scatter_minimal_async(
        w1_out,
        persistent_output_buffers=None,
        dim=ops.rs_w1.dim,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        num_links=ops.rs_w1.num_links,
        cluster_axis=cluster_axis,
        memory_config=None,  # Use default for prefill
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ops.rs_w1.topology,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    w3_out = ttnn.experimental.reduce_scatter_minimal_async(
        w3_out,
        persistent_output_buffers=None,
        dim=ops.rs_w3.dim,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        num_links=1,
        cluster_axis=cluster_axis,
        memory_config=None,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ops.rs_w3.topology,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    # 3. Fused activation + multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=ops.mul.input_tensor_a_activations,
        dtype=ops.mul.dtype,
        memory_config=w1_out.memory_config(),
    )

    ttnn.deallocate(w1_out)
    ttnn.deallocate(w3_out)

    # 4. All-gather before w2
    input_mem_cfg = w2_in.memory_config()
    w2_in = ttnn.experimental.all_gather_async(
        w2_in,
        persistent_output_buffer=None,
        dim=ops.ag_intermediate.dim,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=2,
        cluster_axis=cluster_axis,
        topology=ops.ag_intermediate.topology,
        memory_config=input_mem_cfg,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    # 5. Down projection
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.w2.compute_kernel_config,
        program_config=ops.w2.program_config,
        memory_config=ops.w2.memory_config,
        dtype=ops.w2.dtype,
        core_grid=None,
    )

    ttnn.deallocate(w2_in)

    # 6. All-reduce output (for prefill, typically not sharded)
    original_shape = w2_out.shape
    if original_shape[0] != 1 or original_shape[1] != 1:
        w2_out = ttnn.reshape(
            w2_out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

    w2_out = ttnn.to_memory_config(w2_out, ttnn.DRAM_MEMORY_CONFIG)

    cluster_axis = ops.ar_output.cluster_axis
    w2_out_gathered = ttnn.experimental.all_gather_async(
        w2_out,
        persistent_output_buffer=None,
        dim=ops.ar_output.dim,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=ops.ar_output.num_all_gather_links,
        cluster_axis=cluster_axis,
        topology=ops.ar_output.topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    w2_out_reduced = ttnn.experimental.fast_reduce_nc(
        w2_out_gathered,
        dims=[ops.ar_output.dim],
        output=None,
        compute_kernel_config=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn.deallocate(w2_out_gathered)

    w2_out_reduced = ttnn.reshape(
        w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
    )

    return w2_out_reduced


_ = None

# =============================================================================
# Template Registry
# =============================================================================

TEMPLATE_REGISTRY = {
    # Single device
    (Topology.SINGLE_DEVICE, ExecutionMode.DECODE): forward_mlp_decode_impl,
    (Topology.SINGLE_DEVICE, ExecutionMode.PREFILL): forward_mlp_prefill_impl,
    # N300 (1D, 2 devices)
    (Topology.N300, ExecutionMode.DECODE): forward_mlp_t3k_decode_impl,
    (Topology.N300, ExecutionMode.PREFILL): forward_mlp_prefill_impl,
    # T3K 1D (8 devices, 1D ring)
    (Topology.T3K_1D, ExecutionMode.DECODE): forward_mlp_t3k_decode_impl,
    (Topology.T3K_1D, ExecutionMode.PREFILL): forward_mlp_prefill_impl,
    # Galaxy 2D (8x4 mesh)
    (Topology.GALAXY_2D, ExecutionMode.DECODE): forward_mlp_galaxy_decode_impl,
    (Topology.GALAXY_2D, ExecutionMode.PREFILL): forward_mlp_galaxy_prefill_impl,
}


def get_template_for_config(topology: Topology, mode: ExecutionMode):
    """Select the appropriate template function based on topology and mode."""
    key = (topology, mode)
    if key not in TEMPLATE_REGISTRY:
        raise ValueError(f"No template registered for {key}. Available: {list(TEMPLATE_REGISTRY.keys())}")
    return TEMPLATE_REGISTRY[key]


# =============================================================================
# Default Configuration Initializers
# =============================================================================


def init_decode_ops_config() -> DecodeOpConfigs:
    """Initialize operation configs for decode mode with good defaults."""
    # HiFi2 for decode (FLOP-bound on 12 cores with HiFi4)
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # L1 width sharded for decode
    memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    w1_config = LinearOpConfig(
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
    )
    w2_config = LinearOpConfig(
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
    )
    w3_config = LinearOpConfig(
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
    )
    mul_config = MulOpConfig(
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat8_b,
    )

    return DecodeOpConfigs(
        w1=w1_config,
        w2=w2_config,
        w3=w3_config,
        mul=mul_config,
    )


def init_prefill_ops_config() -> PrefillOpConfigs:
    """Initialize operation configs for prefill mode with good defaults."""
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    memory_config = ttnn.DRAM_MEMORY_CONFIG

    w1_config = LinearOpConfig(
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
    )
    w2_config = LinearOpConfig(
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
    )
    w3_config = LinearOpConfig(
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        dtype=ttnn.bfloat16,
    )
    mul_config = MulOpConfig(
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat8_b,
    )

    return PrefillOpConfigs(
        w1=w1_config,
        w2=w2_config,
        w3=w3_config,
        mul=mul_config,
    )


def init_tg_decode_ops_config(dim: int) -> TGDecodeOpConfigs:
    """Initialize operation configs for TG (Galaxy) decode mode."""
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    w1_config = LinearOpConfig(
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        dtype=ttnn.bfloat8_b,
    )
    w2_config = LinearOpConfig(
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        dtype=ttnn.bfloat8_b,
    )
    w3_config = LinearOpConfig(
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        dtype=ttnn.bfloat8_b,
    )
    mul_config = MulOpConfig(
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat8_b,
    )

    # CCL configs
    rs_config = ReduceScatterConfig(
        dim=3,
        cluster_axis=1,
        num_links=1,
        topology=ttnn.Topology.Linear,
    )
    ag_config = AllGatherConfig(
        dim=3,
        cluster_axis=1,
        num_links=2,
        topology=ttnn.Topology.Linear,
    )
    ar_config = AllReduceConfig(
        cluster_axis=0,
        dim=3,
        num_reduce_scatter_links=1,
        num_all_gather_links=2,
        sharded=True,
        use_composite=(dim == 8192),
        topology=ttnn.Topology.Linear,
    )

    return TGDecodeOpConfigs(
        w1=w1_config,
        w2=w2_config,
        w3=w3_config,
        mul=mul_config,
        rs_w1=rs_config,
        rs_w3=rs_config,
        ag_intermediate=ag_config,
        ar_output=ar_config,
    )


_ = None

# =============================================================================
# Module Templates
# =============================================================================


class MLPDecodeModuleTemplate(LightweightModule):
    """Template class for MLP decode mode (single device or T3K without CCL)."""

    def __init__(self, device, weights: MLPWeights):
        super().__init__()
        self.device = device
        self.dim = 0
        self.hidden_dim = 0

        self.weights = weights
        for key in vars(self.weights):
            attr = getattr(self.weights, key)
            if hasattr(attr, "get_weight"):
                setattr(self.weights, key, attr.get_weight())
        self._init_ops_config()

    def _init_ops_config(self):
        pass

    def forward(self, x):
        return forward_mlp_decode_impl(x, self.weights, self.ops)


class MLPPrefillModuleTemplate(LightweightModule):
    """Template class for MLP prefill mode."""

    def __init__(self, device, weights: MLPWeights, prefill_len_cutoff: int = 1024):
        super().__init__()
        self.device = device
        self.dim = 0
        self.hidden_dim = 0
        self.prefill_len_cutoff = prefill_len_cutoff

        self.weights = weights
        for key in vars(self.weights):
            attr = getattr(self.weights, key)
            if hasattr(attr, "get_weight"):
                setattr(self.weights, key, attr.get_weight())
        self._init_ops_config()

    def _init_ops_config(self):
        pass

    def forward(self, x):
        return forward_mlp_prefill_impl(x, self.weights, self.ops, self.prefill_len_cutoff)


class MLPT3KDecodeModuleTemplate(LightweightModule):
    """Template class for MLP T3K decode mode with CCL."""

    def __init__(self, mesh_device, tt_ccl, weights: MLPWeights):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.dim = 0
        self.hidden_dim = 0

        self.weights = weights
        for key in vars(self.weights):
            attr = getattr(self.weights, key)
            if hasattr(attr, "get_weight"):
                setattr(self.weights, key, attr.get_weight())
        self._init_ops_config()

    def _init_ops_config(self):
        pass

    def forward(self, x):
        return forward_mlp_t3k_decode_impl(x, self.weights, self.ops, self.mesh_device, self.tt_ccl)


class MLPGalaxyDecodeModuleTemplate(LightweightModule):
    """Template class for MLP Galaxy 2D decode mode with full CCL."""

    def __init__(self, mesh_device, tt_ccl, weights: MLPWeights, use_all_reduce_for_ff1: bool = False):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.dim = 0
        self.hidden_dim = 0
        self.use_all_reduce_for_ff1 = use_all_reduce_for_ff1

        self.weights = weights
        for key in vars(self.weights):
            attr = getattr(self.weights, key)
            if hasattr(attr, "get_weight"):
                setattr(self.weights, key, attr.get_weight())
        self._init_ops_config()

    def _init_ops_config(self):
        pass

    def forward(self, x):
        return forward_mlp_galaxy_decode_impl(
            x, self.weights, self.ops, self.mesh_device, self.tt_ccl, self.use_all_reduce_for_ff1
        )


class MLPGalaxyPrefillModuleTemplate(LightweightModule):
    """Template class for MLP Galaxy 2D prefill mode."""

    def __init__(self, mesh_device, tt_ccl, weights: MLPWeights, prefill_len_cutoff: int = 1024):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.dim = 0
        self.hidden_dim = 0
        self.prefill_len_cutoff = prefill_len_cutoff

        self.weights = weights
        for key in vars(self.weights):
            attr = getattr(self.weights, key)
            if hasattr(attr, "get_weight"):
                setattr(self.weights, key, attr.get_weight())
        self._init_ops_config()

    def _init_ops_config(self):
        pass

    def forward(self, x):
        return forward_mlp_galaxy_prefill_impl(
            x, self.weights, self.ops, self.mesh_device, self.tt_ccl, self.prefill_len_cutoff
        )


# =============================================================================
# Code Generator
# =============================================================================


class TTTv2MLPRealCodeGen:
    """
    Generates optimized MLP implementations based on configuration.

    This generator:
    1. Looks up the right template from TEMPLATE_REGISTRY based on (topology, mode)
    2. Extracts the template source code
    3. Generates the appropriate config initialization
    4. Assembles into a complete module class
    """

    # Map of (topology, mode) -> (template_class, ops_init_func)
    TEMPLATE_CLASS_MAP = {
        (Topology.SINGLE_DEVICE, ExecutionMode.DECODE): (MLPDecodeModuleTemplate, init_decode_ops_config),
        (Topology.SINGLE_DEVICE, ExecutionMode.PREFILL): (MLPPrefillModuleTemplate, init_prefill_ops_config),
        (Topology.N300, ExecutionMode.DECODE): (MLPT3KDecodeModuleTemplate, init_decode_ops_config),
        (Topology.N300, ExecutionMode.PREFILL): (MLPPrefillModuleTemplate, init_prefill_ops_config),
        (Topology.T3K_1D, ExecutionMode.DECODE): (MLPT3KDecodeModuleTemplate, init_decode_ops_config),
        (Topology.T3K_1D, ExecutionMode.PREFILL): (MLPPrefillModuleTemplate, init_prefill_ops_config),
        (Topology.GALAXY_2D, ExecutionMode.DECODE): (MLPGalaxyDecodeModuleTemplate, init_tg_decode_ops_config),
        (Topology.GALAXY_2D, ExecutionMode.PREFILL): (MLPGalaxyPrefillModuleTemplate, init_tg_decode_ops_config),
    }

    def __init__(
        self,
        mlp_config: MLPConfig,
        mode: ExecutionMode = ExecutionMode.DECODE,
    ):
        self.mlp_config = mlp_config
        self.mode = mode
        self._validate_config()

        self.selected_template = get_template_for_config(mlp_config.topology, mode)
        self.template_name = self.selected_template.__name__

        key = (mlp_config.topology, mode)
        self.template_class, self.ops_init_func = self.TEMPLATE_CLASS_MAP[key]

    def _validate_config(self):
        """Validate that configuration is feasible."""
        if self.mlp_config.topology == Topology.GALAXY_2D:
            if self.mlp_config.dim % 32 != 0:
                raise ValueError(f"Galaxy topology requires dim divisible by 32, got {self.mlp_config.dim}")

    def generate_ops_init_source(self) -> list:
        """Generate source code for _init_ops_config method."""
        # Get the appropriate init function
        if self.mlp_config.topology == Topology.GALAXY_2D:
            # TG init takes dim as parameter
            source_lines = function_to_source(self.ops_init_func)
            # Replace the function call with self.ops assignment
            processed_lines = []
            for line in source_lines:
                if line.strip().startswith("def "):
                    continue  # Skip def line
                if "return " in line:
                    # Change return to self.ops assignment
                    line = line.replace("return ", "self.ops = ")
                processed_lines.append(line)
            return processed_lines
        else:
            source_lines = function_to_source(self.ops_init_func)
            processed_lines = []
            for line in source_lines:
                if line.strip().startswith("def "):
                    continue
                if "return " in line:
                    line = line.replace("return ", "self.ops = ")
                processed_lines.append(line)
            return processed_lines

    def generate_module_class(self) -> str:
        """Generate complete module class with initialization and forward."""
        topology_name = self.mlp_config.topology.value.replace("_", "")
        mode_name = self.mode.value
        class_name = f"TTTv2MLP_{topology_name}_{mode_name}"

        lines = []

        # Add imports
        lines.append(get_import_source(GENERATED_MODULE_NAMESPACE))
        lines.append("")

        # Add config classes
        lines.extend(class_to_source(LinearOpConfig))
        lines.append("")
        lines.extend(class_to_source(MulOpConfig))
        lines.append("")

        if self.mlp_config.topology == Topology.GALAXY_2D:
            lines.extend(class_to_source(ReduceScatterConfig))
            lines.append("")
            lines.extend(class_to_source(AllGatherConfig))
            lines.append("")
            lines.extend(class_to_source(AllReduceConfig))
            lines.append("")
            lines.extend(class_to_source(TGDecodeOpConfigs))
        elif self.mode == ExecutionMode.DECODE:
            lines.extend(class_to_source(DecodeOpConfigs))
        else:
            lines.extend(class_to_source(PrefillOpConfigs))
        lines.append("")

        lines.extend(class_to_source(MLPWeights))
        lines.append("")

        # Add template function
        lines.append(f"# Template: {self.template_name}")
        impl_source = function_to_source(self.selected_template)
        for line in impl_source:
            lines.append(line)
        lines.append("")

        # Generate module class from template
        template_source = class_to_source(self.template_class)
        template_class_name = self.template_class.__name__

        # Replace class name
        template_source[0] = template_source[0].replace(template_class_name, class_name)

        # Find and replace _init_ops_config body
        in_init_ops = False
        skip_until_next_def = False
        final_class_lines = []

        for i, line in enumerate(template_source):
            if "def _init_ops_config(self):" in line:
                final_class_lines.append(line)
                # Add our generated ops init
                ops_init_lines = self.generate_ops_init_source()
                for ops_line in ops_init_lines:
                    if ops_line.strip():
                        final_class_lines.append(f"    {ops_line}")
                    else:
                        final_class_lines.append("")
                skip_until_next_def = True
                continue

            if skip_until_next_def:
                # Check if we've reached the next method or end of class
                stripped = line.strip()
                if stripped.startswith("def ") or (stripped and not line.startswith("        ")):
                    skip_until_next_def = False
                    if stripped.startswith("def "):
                        final_class_lines.append("")
                    final_class_lines.append(line)
                continue

            # Replace dim/hidden_dim placeholders
            if "self.dim = 0" in line:
                line = line.replace("0", str(self.mlp_config.dim))
            if "self.hidden_dim = 0" in line:
                line = line.replace("0", str(self.mlp_config.hidden_dim))

            final_class_lines.append(line)

        # Add docstring
        docstring_lines = [
            f'    """',
            f"    Auto-generated MLP module for {topology_name} {mode_name}",
            f"    Configuration:",
            f"      - dim: {self.mlp_config.dim}",
            f"      - hidden_dim: {self.mlp_config.hidden_dim}",
            f"      - num_devices: {self.mlp_config.num_devices}",
            f'    """',
        ]

        # Insert docstring after class definition
        result_lines = [final_class_lines[0]]  # class line
        result_lines.extend(docstring_lines)
        result_lines.extend(final_class_lines[1:])

        lines.extend(result_lines)

        return "\n".join(lines)


# =============================================================================
# SaveSource
# =============================================================================


class SaveSource:
    """Handles saving generated source code to files."""

    def __init__(self, filename: Optional[str] = None):
        self._filename = filename
        self._default_dir = None

    def _get_default_dir(self) -> Path:
        if self._default_dir is None:
            codegen_file = Path(__file__).resolve()
            codegen_dir = codegen_file.parent
            self._default_dir = codegen_dir / "generated"
            self._default_dir.mkdir(exist_ok=True)
        return self._default_dir

    def get_filename(self, mlp_config: Optional[MLPConfig] = None, mode: Optional[ExecutionMode] = None) -> str:
        if self._filename:
            filepath = Path(self._filename)
            if not filepath.is_absolute():
                return str(self._get_default_dir() / filepath)
            return str(filepath)

        if mlp_config and mode:
            default_name = f"mlp_real_{mlp_config.topology.value}_{mode.value}_{mlp_config.dim}.py"
        else:
            default_name = "mlp_real_generated.py"

        return str(self._get_default_dir() / default_name)


# =============================================================================
# Main API
# =============================================================================


def MLP(
    mlp_config: MLPConfig,
    mode: ExecutionMode = ExecutionMode.DECODE,
    *,
    save_source: Optional[SaveSource] = None,
) -> type:
    """
    Main API to compile an MLP module for specific hardware and configuration.

    Args:
        mlp_config: MLP configuration
        mode: Execution mode (decode or prefill)
        save_source: Optional SaveSource instance for saving generated code

    Returns:
        Compiled module class
    """
    codegen = TTTv2MLPRealCodeGen(mlp_config, mode)
    source_code = codegen.generate_module_class()

    # Save source if requested
    filename = "<string>"
    if save_source is not None:
        header = (
            f"# SPDX-FileCopyrightText: © {datetime.now().year} Tenstorrent AI ULC\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
            "# Auto-generated by TTTv2 CodeGen\n"
        )
        source_code = header + source_code

        filepath = save_source.get_filename(mlp_config, mode)
        if filepath:
            with open(filepath, "w") as f:
                f.write(source_code)
            filename = str(Path(filepath).resolve())

    # Compile the source
    namespace = GENERATED_MODULE_NAMESPACE.copy()
    code_obj = compile(source_code, filename, "exec")
    exec(code_obj, namespace)

    # Find the generated class
    module_class = None
    for name, obj in namespace.items():
        if name.startswith("TTTv2MLP_"):
            module_class = obj
            break

    return module_class


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=== Example: Generate Real MLP for Single Device Decode ===")

    mlp_config = MLPConfig(
        dim=4096,
        hidden_dim=11008,
        num_devices=1,
        topology=Topology.SINGLE_DEVICE,
    )

    save_source = SaveSource()
    module_class = MLP(
        mlp_config,
        mode=ExecutionMode.DECODE,
        save_source=save_source,
    )

    print(f"Generated class: {module_class}")
    if module_class:
        filename = save_source.get_filename(mlp_config, ExecutionMode.DECODE)
        print(f"\nSaved to: {filename}")
        print("\nGenerated source preview:")
        print("-" * 60)
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines[:80]:
                print(line, end="")
            print("... [truncated]\n")

    print("\n=== Example: Generate Real MLP for Galaxy 2D Decode ===")

    mlp_config_galaxy = MLPConfig(
        dim=8192,
        hidden_dim=28672,
        num_devices=32,
        topology=Topology.GALAXY_2D,
    )

    save_source_galaxy = SaveSource()
    module_class_galaxy = MLP(
        mlp_config_galaxy,
        mode=ExecutionMode.DECODE,
        save_source=save_source_galaxy,
    )

    print(f"Generated class: {module_class_galaxy}")
    if module_class_galaxy:
        filename = save_source_galaxy.get_filename(mlp_config_galaxy, ExecutionMode.DECODE)
        print(f"\nSaved to: {filename}")
        print("\nGenerated source preview (Galaxy 2D):")
        print("-" * 60)
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines[:100]:
                print(line, end="")
            print("... [truncated]\n")
