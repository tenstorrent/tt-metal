"""
TTTv2 Real MLP Module with Code Generation

This module brings the production MLP implementation from models/tt_transformers/tt/mlp.py
to the TTTv2 framework with code generation support.

Key differences from the toy example (codegen_mlp.py):
1. Real weight sharding patterns (ShardTensor2dMesh)
2. Real CCL operations (reduce_scatter_minimal_async, all_gather_async, tt_all_reduce)
3. Real program configs from model_config
4. Real dtype optimizations from DECODERS_OPTIMIZATIONS

APPROACH: "Codegen Selects, Doesn't Compose"
- Each (topology, mode, dim_class) combination gets its own complete template function
- Templates are readable, standalone implementations
- Codegen's job is to SELECT the right template based on configuration
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.codegen_utils.generation import class_to_source, function_to_source
from models.common.modules.lazy_weight import LazyWeight
from models.tt_transformers.tt.ccl import tt_all_reduce

# =============================================================================
# Single Source of Truth for Generated Module Imports
# =============================================================================

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
    "tt_all_reduce": tt_all_reduce,
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

    SINGLE_DEVICE = "single_device"  # Single Wormhole chip
    T3K_1D = "t3k_1d"  # TG with 1D ring topology
    GALAXY_2D = "galaxy_2d"  # Galaxy with 2D mesh topology


class DimClass(Enum):
    """Dimension class for template selection.

    In the real MLP, dim >= 8192 triggers different CCL patterns.
    """

    SMALL = "small"  # dim < 8192
    LARGE = "large"  # dim >= 8192


@dataclass
class MLPConfig:
    """MLP-specific configuration matching the real MLP args pattern."""

    dim: int  # Model hidden dimension (e.g., 4096, 8192)
    hidden_dim: int  # Intermediate dimension after gate/up projection
    unpadded_hidden_dim: int  # Original hidden dim before padding
    num_devices: int = 1  # Number of devices for sharding
    is_galaxy: bool = False  # Whether running on Galaxy topology
    cluster_shape: Tuple[int, int] = (1, 1)  # Mesh shape for sharding
    activation: str = "silu"  # Activation function
    prefill_len_cutoff: int = 1024  # Cutoff for prefill reshaping
    # Derived topology and dim class
    topology: Topology = Topology.SINGLE_DEVICE
    dim_class: DimClass = DimClass.SMALL

    def __post_init__(self):
        # Derive topology from is_galaxy
        if self.is_galaxy:
            self.topology = Topology.GALAXY_2D
        elif self.num_devices > 1:
            self.topology = Topology.T3K_1D
        else:
            self.topology = Topology.SINGLE_DEVICE

        # Derive dim class
        self.dim_class = DimClass.LARGE if self.dim >= 8192 else DimClass.SMALL


@dataclass
class OpConfig:
    """Configuration for a TTNN operation."""

    compute_kernel_config: Optional[Any] = None
    memory_config: Optional[Any] = None
    dtype: Optional[Any] = None
    program_config: Optional[Any] = None
    core_grid: Optional[Any] = None


@dataclass
class MLPOpConfigs:
    """Container for MLP operation configurations.

    Maps to the real MLP's use of model_config for op configurations.
    """

    ff1_ff3: OpConfig  # Gate and up projection config
    ff2: OpConfig  # Down projection config
    activation_type: Any = field(default_factory=lambda: ttnn.UnaryOpType.SILU)
    activation_dtype: Optional[Any] = None


@dataclass
class CCLConfig:
    """CCL-specific configuration for multi-device operations."""

    num_reduce_scatter_links: int = 1
    num_all_gather_links: int = 2
    cluster_axis: int = 1
    ccl_dtype: Any = field(default_factory=lambda: ttnn.bfloat16)


@dataclass
class MLPWeights:
    """Container for MLP weights.

    Follows the real MLP's w1, w2, w3 naming:
    - w1: gate_proj (projects to hidden_dim)
    - w2: down_proj (projects back to dim)
    - w3: up_proj (projects to hidden_dim)
    """

    w1: LazyWeight  # gate_proj
    w2: LazyWeight  # down_proj
    w3: LazyWeight  # up_proj


@dataclass
class MemoryConfigs:
    """Memory configurations for different stages of the MLP."""

    ff1_out_reduce_scatter: Optional[Any] = None
    ff1_out_gathered: Optional[Any] = None
    sharded_mlp2_input: Optional[Any] = None
    ff2_out_reduce_scatter: Optional[Any] = None
    sharded_attn_input: Optional[Any] = None
    decode_residual: Optional[Any] = None


_ = None  # Stop inspect.getsource from grabbing following comments

# =============================================================================
# TEMPLATE 1: Non-TG Decode MLP
# =============================================================================
# Standard decode path for single device or T3K with DRAM sharded matmuls


def forward_mlp_non_tg_decode_impl(x, weights, ops, mem_configs, mesh_device, tt_ccl, args):
    """
    Non-Galaxy decode MLP forward pass.

    Uses DRAM sharded matmuls with HiFi2 (drops 1 bit of activations but avoids
    being FLOP-bound on 12 cores with HiFi4).

    Dataflow: x -> [w1, w3] -> silu(w1) * w3 -> w2 -> all_reduce -> output

    Args:
        x: Input tensor [1, 1, seq_len, dim]
        weights: MLPWeights container with w1, w2, w3
        ops: MLPOpConfigs for operation configurations
        mem_configs: MemoryConfigs for memory layouts
        mesh_device: Device mesh for multi-device operations
        tt_ccl: CCL instance for collective operations
        args: Model args with topology info
    """
    memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    # 1. Gate projection (w1)
    w1_out = ttnn.linear(
        x,
        weights.w1,
        dtype=ops.activation_dtype or ttnn.bfloat16,
        core_grid=None,
        compute_kernel_config=ops.ff1_ff3.compute_kernel_config,
        program_config=ops.ff1_ff3.program_config,
        memory_config=memory_config,
    )

    # 2. Up projection (w3)
    w3_out = ttnn.linear(
        x,
        weights.w3,
        dtype=ops.activation_dtype or ttnn.bfloat16,
        core_grid=None,
        compute_kernel_config=ops.ff1_ff3.compute_kernel_config,
        program_config=ops.ff1_ff3.program_config,
        memory_config=memory_config,
    )
    ttnn.deallocate(x)

    # 3. Activation + element-wise multiplication
    # SiLU is fused into the mul operation via input_tensor_a_activations
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ops.activation_type],
        dtype=ops.activation_dtype or ttnn.bfloat8_b,
        memory_config=w1_out.memory_config(),
    )

    # 4. Memory config adjustment for w2 (may use different core grid)
    w2_in = ttnn.to_memory_config(w2_in, mem_configs.sharded_mlp2_input)

    ttnn.deallocate(w3_out)
    ttnn.deallocate(w1_out)

    # 5. Down projection (w2)
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.ff2.compute_kernel_config,
        dtype=ops.activation_dtype or ttnn.bfloat16,
        program_config=ops.ff2.program_config,
        memory_config=memory_config,
        core_grid=None,
    )
    ttnn.deallocate(w2_in)

    # 6. All-reduce across devices
    w2_out_reduced = tt_all_reduce(
        w2_out,
        mesh_device,
        tt_ccl,
        cluster_axis=0,
        dim=3,
        num_reduce_scatter_links=args.num_reduce_scatter_links,
        num_all_gather_links=args.num_all_gather_links,
        sharded=True,
        memory_config=w2_out.memory_config(),
        dtype=args.ccl_dtype,
        use_composite=False,
        topology=args.ccl_topology(),
    )

    # 7. Reshape to ensure dim 0 and 1 are 1
    original_shape = w2_out_reduced.shape
    w2_out_reduced = ttnn.reshape(
        w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
    )

    # 8. Final memory config
    w2_out_reduced = ttnn.to_memory_config(w2_out_reduced, mem_configs.decode_residual)

    return w2_out_reduced


_ = None  # Stop inspect.getsource from grabbing following comments

# =============================================================================
# TEMPLATE 2: Non-TG Prefill MLP
# =============================================================================


def forward_mlp_non_tg_prefill_impl(x, weights, ops, mem_configs, mesh_device, tt_ccl, args, seq_len):
    """
    Non-Galaxy prefill MLP forward pass.

    For long sequences (>= prefill_len_cutoff), reshapes input to fit on device
    and parallelize computation.

    Args:
        x: Input tensor [1, 1, seq_len, dim]
        weights: MLPWeights container
        ops: MLPOpConfigs
        mem_configs: MemoryConfigs
        mesh_device: Device mesh
        tt_ccl: CCL instance
        args: Model args
        seq_len: Sequence length for program config selection
    """
    memory_config = ttnn.DRAM_MEMORY_CONFIG

    # 1. Gate projection (w1)
    w1_out = ttnn.linear(
        x,
        weights.w1,
        dtype=ops.activation_dtype or ttnn.bfloat16,
        core_grid=None,
        compute_kernel_config=ops.ff1_ff3.compute_kernel_config,
        program_config=ops.ff1_ff3.program_config,
        memory_config=memory_config,
    )

    # 2. Up projection (w3)
    w3_out = ttnn.linear(
        x,
        weights.w3,
        dtype=ops.activation_dtype or ttnn.bfloat16,
        core_grid=None,
        compute_kernel_config=ops.ff1_ff3.compute_kernel_config,
        program_config=ops.ff1_ff3.program_config,
        memory_config=memory_config,
    )
    ttnn.deallocate(x)

    # 3. Activation + element-wise multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ops.activation_type],
        dtype=ops.activation_dtype or ttnn.bfloat8_b,
        memory_config=w1_out.memory_config(),
    )

    ttnn.deallocate(w3_out)
    ttnn.deallocate(w1_out)

    # 4. Down projection (w2)
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.ff2.compute_kernel_config,
        dtype=ops.activation_dtype or ttnn.bfloat16,
        program_config=ops.ff2.program_config,
        memory_config=memory_config,
        core_grid=None,
    )
    ttnn.deallocate(w2_in)

    # 5. All-reduce across devices
    w2_out_reduced = tt_all_reduce(
        w2_out,
        mesh_device,
        tt_ccl,
        cluster_axis=0,
        dim=3,
        num_reduce_scatter_links=args.num_reduce_scatter_links,
        num_all_gather_links=args.num_all_gather_links,
        sharded=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=args.ccl_dtype,
        use_composite=False,
        topology=args.ccl_topology(),
    )

    # 6. Reshape
    original_shape = w2_out_reduced.shape
    w2_out_reduced = ttnn.reshape(
        w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
    )

    return w2_out_reduced


_ = None  # Stop inspect.getsource from grabbing following comments

# =============================================================================
# TEMPLATE 3: TG (Galaxy) Decode MLP - Small Dim (< 8192)
# =============================================================================
# Uses tt_all_reduce instead of reduce_scatter + all_gather pattern


def forward_mlp_tg_decode_small_impl(x, weights, ops, mem_configs, mesh_device, tt_ccl, args):
    """
    Galaxy decode MLP for small dimensions (< 8192).

    Uses tt_all_reduce pattern which is more efficient for smaller dimensions.

    Dataflow:
        x -> [w1, w3] -> all_reduce(w1), all_reduce(w3)
          -> silu(w1) * w3 -> w2 -> all_reduce -> output
    """
    memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    # 1. Gate projection (w1)
    w1_out = ttnn.linear(
        x,
        weights.w1,
        dtype=ttnn.bfloat8_b,
        core_grid=None,
        compute_kernel_config=ops.ff1_ff3.compute_kernel_config,
        program_config=ops.ff1_ff3.program_config,
        memory_config=memory_config,
    )

    # 2. Up projection (w3)
    w3_out = ttnn.linear(
        x,
        weights.w3,
        dtype=ttnn.bfloat8_b,
        core_grid=None,
        compute_kernel_config=ops.ff1_ff3.compute_kernel_config,
        program_config=ops.ff1_ff3.program_config,
        memory_config=memory_config,
    )
    ttnn.deallocate(x)

    # 3. All-reduce for w1_out and w3_out (small dim uses all_reduce instead of RS+AG)
    w1_out = tt_all_reduce(
        w1_out,
        mesh_device,
        tt_ccl,
        cluster_axis=1,
        num_all_gather_links=2,
        sharded=True,
        topology=args.ccl_topology(),
        memory_config=mem_configs.ff1_out_gathered,
    )
    w3_out = tt_all_reduce(
        w3_out,
        mesh_device,
        tt_ccl,
        cluster_axis=1,
        num_all_gather_links=2,
        sharded=True,
        topology=args.ccl_topology(),
        memory_config=mem_configs.ff1_out_gathered,
    )

    # 4. Activation + element-wise multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ops.activation_type],
        dtype=ops.activation_dtype or ttnn.bfloat8_b,
        memory_config=w1_out.memory_config(),
    )

    ttnn.deallocate(w3_out)
    ttnn.deallocate(w1_out)

    # 5. Down projection (w2)
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.ff2.compute_kernel_config,
        dtype=args.ccl_dtype,
        program_config=ops.ff2.program_config,
        memory_config=memory_config,
        core_grid=None,
    )
    ttnn.deallocate(w2_in)

    # 6. All-reduce across cluster_axis=0 with dim=0 for small TG
    w2_out_reduced = tt_all_reduce(
        w2_out,
        mesh_device,
        tt_ccl,
        cluster_axis=0,
        dim=0,  # dim=0 for small TG
        num_reduce_scatter_links=args.num_reduce_scatter_links,
        num_all_gather_links=args.num_all_gather_links,
        sharded=True,
        memory_config=mem_configs.ff2_out_reduce_scatter,
        dtype=args.ccl_dtype,
        use_composite=False,
        topology=args.ccl_topology(),
    )

    # 7. Reshape
    original_shape = w2_out_reduced.shape
    w2_out_reduced = ttnn.reshape(
        w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
    )

    # 8. Final memory config
    w2_out_reduced = ttnn.to_memory_config(w2_out_reduced, mem_configs.sharded_attn_input)

    return w2_out_reduced


_ = None  # Stop inspect.getsource from grabbing following comments

# =============================================================================
# TEMPLATE 4: TG (Galaxy) Decode MLP - Large Dim (>= 8192)
# =============================================================================
# Uses reduce_scatter_minimal_async + all_gather_async pattern


def forward_mlp_tg_decode_large_impl(x, weights, ops, mem_configs, mesh_device, tt_ccl, args):
    """
    Galaxy decode MLP for large dimensions (>= 8192).

    Uses reduce_scatter_minimal_async followed by all_gather_async pattern
    for more efficient communication on large hidden dimensions.

    Dataflow:
        x -> [w1, w3] -> reduce_scatter(w1), reduce_scatter(w3)
          -> silu(w1) * w3 -> all_gather -> w2 -> all_reduce -> output
    """
    memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    input_mem_cfg = None  # Will be captured from w1_out

    # 1. Gate projection (w1)
    w1_out = ttnn.linear(
        x,
        weights.w1,
        dtype=ttnn.bfloat8_b,
        core_grid=None,
        compute_kernel_config=ops.ff1_ff3.compute_kernel_config,
        program_config=ops.ff1_ff3.program_config,
        memory_config=memory_config,
    )

    # 2. Up projection (w3)
    w3_out = ttnn.linear(
        x,
        weights.w3,
        dtype=ttnn.bfloat8_b,
        core_grid=None,
        compute_kernel_config=ops.ff1_ff3.compute_kernel_config,
        program_config=ops.ff1_ff3.program_config,
        memory_config=memory_config,
    )
    ttnn.deallocate(x)

    # Capture memory config for later use
    input_mem_cfg = w1_out.memory_config()

    # 3. Reduce scatter for w1_out
    cluster_axis = 1
    w1_out = ttnn.experimental.reduce_scatter_minimal_async(
        w1_out,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        num_links=args.num_reduce_scatter_links,
        cluster_axis=cluster_axis,
        memory_config=mem_configs.ff1_out_reduce_scatter,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    # 4. Reduce scatter for w3_out
    w3_out = ttnn.experimental.reduce_scatter_minimal_async(
        w3_out,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        num_links=1,
        cluster_axis=cluster_axis,
        memory_config=mem_configs.ff1_out_reduce_scatter,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    # 5. Activation + element-wise multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ops.activation_type],
        dtype=ops.activation_dtype or ttnn.bfloat8_b,
        memory_config=w1_out.memory_config(),
    )

    ttnn.deallocate(w3_out)
    ttnn.deallocate(w1_out)

    # 6. All-gather to reconstruct full intermediate
    w2_in = ttnn.experimental.all_gather_async(
        w2_in,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=2,
        cluster_axis=1,
        topology=ttnn.Topology.Linear,
        memory_config=input_mem_cfg,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

    # 7. Down projection (w2)
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.ff2.compute_kernel_config,
        dtype=args.ccl_dtype,
        program_config=ops.ff2.program_config,
        memory_config=memory_config,
        core_grid=None,
    )
    ttnn.deallocate(w2_in)

    # 8. All-reduce across cluster_axis=0 with composite mode for large dim
    w2_out_reduced = tt_all_reduce(
        w2_out,
        mesh_device,
        tt_ccl,
        cluster_axis=0,
        dim=3,
        num_reduce_scatter_links=args.num_reduce_scatter_links,
        num_all_gather_links=args.num_all_gather_links,
        sharded=True,
        memory_config=mem_configs.ff2_out_reduce_scatter,
        dtype=args.ccl_dtype,
        use_composite=True,  # Use composite for large dim
        topology=args.ccl_topology(),
    )

    # 9. Reshape
    original_shape = w2_out_reduced.shape
    w2_out_reduced = ttnn.reshape(
        w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
    )

    # 10. Final memory config
    w2_out_reduced = ttnn.to_memory_config(w2_out_reduced, mem_configs.sharded_attn_input)

    return w2_out_reduced


_ = None  # Stop inspect.getsource from grabbing following comments

# =============================================================================
# TEMPLATE 5: TG (Galaxy) Prefill MLP
# =============================================================================
# Uses reduce_scatter_minimal_async + all_gather_async pattern for prefill


def forward_mlp_tg_prefill_impl(x, weights, ops, mem_configs, mesh_device, tt_ccl, args, seq_len):
    """
    Galaxy prefill MLP forward pass.

    Uses reduce_scatter + all_gather pattern for prefill regardless of dim size.

    Args:
        x: Input tensor (may be reshaped for long sequences)
        weights: MLPWeights container
        ops: MLPOpConfigs
        mem_configs: MemoryConfigs
        mesh_device: Device mesh
        tt_ccl: CCL instance
        args: Model args
        seq_len: Sequence length
    """
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    input_mem_cfg = None

    # 1. Gate projection (w1)
    w1_out = ttnn.linear(
        x,
        weights.w1,
        dtype=ttnn.bfloat8_b,
        core_grid=None,
        compute_kernel_config=ops.ff1_ff3.compute_kernel_config,
        program_config=ops.ff1_ff3.program_config,
        memory_config=memory_config,
    )

    # 2. Up projection (w3)
    w3_out = ttnn.linear(
        x,
        weights.w3,
        dtype=ttnn.bfloat8_b,
        core_grid=None,
        compute_kernel_config=ops.ff1_ff3.compute_kernel_config,
        program_config=ops.ff1_ff3.program_config,
        memory_config=memory_config,
    )
    ttnn.deallocate(x)

    input_mem_cfg = w1_out.memory_config()

    # 3. Reduce scatter for w1_out
    cluster_axis = 1
    w1_out = ttnn.experimental.reduce_scatter_minimal_async(
        w1_out,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        num_links=args.num_reduce_scatter_links,
        cluster_axis=cluster_axis,
        memory_config=None,  # Uses default for prefill
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    # 4. Reduce scatter for w3_out
    w3_out = ttnn.experimental.reduce_scatter_minimal_async(
        w3_out,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        num_links=1,
        cluster_axis=cluster_axis,
        memory_config=None,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    # 5. Activation + element-wise multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ops.activation_type],
        dtype=ops.activation_dtype or ttnn.bfloat8_b,
        memory_config=w1_out.memory_config(),
    )

    ttnn.deallocate(w3_out)
    ttnn.deallocate(w1_out)

    # 6. All-gather to reconstruct full intermediate
    w2_in = ttnn.experimental.all_gather_async(
        w2_in,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=2,
        cluster_axis=1,
        topology=ttnn.Topology.Linear,
        memory_config=input_mem_cfg,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    # 7. Down projection (w2)
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.ff2.compute_kernel_config,
        dtype=args.ccl_dtype,
        program_config=ops.ff2.program_config,
        memory_config=memory_config,
        core_grid=None,
    )
    ttnn.deallocate(w2_in)

    # 8. All-reduce
    w2_out_reduced = tt_all_reduce(
        w2_out,
        mesh_device,
        tt_ccl,
        cluster_axis=0,
        dim=3,
        num_reduce_scatter_links=args.num_reduce_scatter_links,
        num_all_gather_links=args.num_all_gather_links,
        sharded=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=args.ccl_dtype,
        use_composite=True if args.dim == 8192 else False,
        topology=args.ccl_topology(),
    )

    # 9. Reshape
    original_shape = w2_out_reduced.shape
    w2_out_reduced = ttnn.reshape(
        w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
    )

    return w2_out_reduced


_ = None  # Stop inspect.getsource from grabbing following comments

# =============================================================================
# Template Registry
# =============================================================================
# Maps (topology, mode, dim_class) -> template function

TEMPLATE_REGISTRY = {
    # Non-TG decode
    (Topology.SINGLE_DEVICE, "decode", DimClass.SMALL): forward_mlp_non_tg_decode_impl,
    (Topology.SINGLE_DEVICE, "decode", DimClass.LARGE): forward_mlp_non_tg_decode_impl,
    (Topology.T3K_1D, "decode", DimClass.SMALL): forward_mlp_non_tg_decode_impl,
    (Topology.T3K_1D, "decode", DimClass.LARGE): forward_mlp_non_tg_decode_impl,
    # Non-TG prefill
    (Topology.SINGLE_DEVICE, "prefill", DimClass.SMALL): forward_mlp_non_tg_prefill_impl,
    (Topology.SINGLE_DEVICE, "prefill", DimClass.LARGE): forward_mlp_non_tg_prefill_impl,
    (Topology.T3K_1D, "prefill", DimClass.SMALL): forward_mlp_non_tg_prefill_impl,
    (Topology.T3K_1D, "prefill", DimClass.LARGE): forward_mlp_non_tg_prefill_impl,
    # TG decode - depends on dim class
    (Topology.GALAXY_2D, "decode", DimClass.SMALL): forward_mlp_tg_decode_small_impl,
    (Topology.GALAXY_2D, "decode", DimClass.LARGE): forward_mlp_tg_decode_large_impl,
    # TG prefill - same for all dim classes
    (Topology.GALAXY_2D, "prefill", DimClass.SMALL): forward_mlp_tg_prefill_impl,
    (Topology.GALAXY_2D, "prefill", DimClass.LARGE): forward_mlp_tg_prefill_impl,
}


def get_template_for_config(topology: Topology, mode: str, dim_class: DimClass):
    """Select the appropriate template function based on configuration."""
    key = (topology, mode, dim_class)
    if key not in TEMPLATE_REGISTRY:
        raise ValueError(f"No template registered for {key}. Available: {list(TEMPLATE_REGISTRY.keys())}")
    return TEMPLATE_REGISTRY[key]


# =============================================================================
# Op Config Initializers
# =============================================================================


def init_ops_config_non_tg_decode_impl(model_config, layer_num, args):
    """Initialize op configs for non-TG decode mode."""
    from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

    layer_num = max(layer_num, 0)

    # Get dtype from DECODERS_OPTIMIZATIONS
    activation_dtype = model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
        decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
    )
    ff1_3_compute_kernel_cfg = model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
        decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=args
    )
    ff2_compute_kernel_cfg = model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
        decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=args
    )

    # Program configs from model_config
    pc_1_3 = model_config["DECODE_MLP_W1_W3_PRG_CONFIG"]
    pc_2 = model_config["DECODE_MLP_W2_PRG_CONFIG"]

    ff1_ff3_config = OpConfig(
        compute_kernel_config=ff1_3_compute_kernel_cfg,
        program_config=pc_1_3,
    )
    ff2_config = OpConfig(
        compute_kernel_config=ff2_compute_kernel_cfg,
        program_config=pc_2,
    )

    activation_type = args.mlp_activation_type if hasattr(args, "mlp_activation_type") else ttnn.UnaryOpType.SILU

    return MLPOpConfigs(
        ff1_ff3=ff1_ff3_config,
        ff2=ff2_config,
        activation_type=activation_type,
        activation_dtype=activation_dtype,
    )


def init_ops_config_non_tg_prefill_impl(model_config, layer_num, args, seq_len):
    """Initialize op configs for non-TG prefill mode."""
    from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

    layer_num = max(layer_num, 0)

    activation_dtype = model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
        decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
    )
    ff1_3_compute_kernel_cfg = model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
        decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=args
    )
    ff2_compute_kernel_cfg = model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
        decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=args
    )

    # Prefill program configs are functions of seq_len
    pc_1_3 = model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"](seq_len)
    pc_2 = model_config["PREFILL_MLP_W2_PRG_CONFIG"](seq_len)

    ff1_ff3_config = OpConfig(
        compute_kernel_config=ff1_3_compute_kernel_cfg,
        program_config=pc_1_3,
    )
    ff2_config = OpConfig(
        compute_kernel_config=ff2_compute_kernel_cfg,
        program_config=pc_2,
    )

    activation_type = args.mlp_activation_type if hasattr(args, "mlp_activation_type") else ttnn.UnaryOpType.SILU

    return MLPOpConfigs(
        ff1_ff3=ff1_ff3_config,
        ff2=ff2_config,
        activation_type=activation_type,
        activation_dtype=activation_dtype,
    )


def init_ops_config_tg_decode_impl(model_config, layer_num, args):
    """Initialize op configs for TG decode mode."""
    from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

    layer_num = max(layer_num, 0)

    activation_dtype = model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
        decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
    )
    ff1_3_compute_kernel_cfg = model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
        decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=args
    )
    ff2_compute_kernel_cfg = model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
        decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=args
    )

    # TG program configs depend on dim
    pc_1_3 = model_config["FF1_3_TG_PROGCFG"] if args.dim >= 4096 else None
    pc_2 = model_config["FF2_TG_PROGCFG"] if args.dim >= 4096 else None

    ff1_ff3_config = OpConfig(
        compute_kernel_config=ff1_3_compute_kernel_cfg,
        program_config=pc_1_3,
    )
    ff2_config = OpConfig(
        compute_kernel_config=ff2_compute_kernel_cfg,
        program_config=pc_2,
    )

    activation_type = args.mlp_activation_type if hasattr(args, "mlp_activation_type") else ttnn.UnaryOpType.SILU

    return MLPOpConfigs(
        ff1_ff3=ff1_ff3_config,
        ff2=ff2_config,
        activation_type=activation_type,
        activation_dtype=activation_dtype,
    )


def init_ops_config_tg_prefill_impl(model_config, layer_num, args, seq_len):
    """Initialize op configs for TG prefill mode."""
    from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

    layer_num = max(layer_num, 0)

    activation_dtype = model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
        decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
    )
    ff1_3_compute_kernel_cfg = model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
        decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=args
    )
    ff2_compute_kernel_cfg = model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
        decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=args
    )

    # TG prefill uses prefill program configs
    pc_1_3 = model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"](seq_len)
    pc_2 = model_config["PREFILL_MLP_W2_PRG_CONFIG"](seq_len)

    ff1_ff3_config = OpConfig(
        compute_kernel_config=ff1_3_compute_kernel_cfg,
        program_config=pc_1_3,
    )
    ff2_config = OpConfig(
        compute_kernel_config=ff2_compute_kernel_cfg,
        program_config=pc_2,
    )

    activation_type = args.mlp_activation_type if hasattr(args, "mlp_activation_type") else ttnn.UnaryOpType.SILU

    return MLPOpConfigs(
        ff1_ff3=ff1_ff3_config,
        ff2=ff2_config,
        activation_type=activation_type,
        activation_dtype=activation_dtype,
    )


def init_memory_configs_decode_impl(model_config, is_galaxy):
    """Initialize memory configs for decode mode."""
    if is_galaxy:
        return MemoryConfigs(
            ff1_out_reduce_scatter=model_config.get("FF1_OUT_REDUCE_SCATTER_MEMCFG"),
            ff1_out_gathered=model_config.get("FF1_OUT_GATHERED_MEMCFG"),
            sharded_mlp2_input=model_config.get("SHARDED_MLP2_INPUT_MEMCFG"),
            ff2_out_reduce_scatter=model_config.get("FF2_OUT_REDUCE_SCATTER_MEMCFG"),
            sharded_attn_input=model_config.get("SHARDED_ATTN_INPUT_MEMCFG"),
            decode_residual=model_config.get("DECODE_RESIDUAL_MEMCFG"),
        )
    else:
        return MemoryConfigs(
            sharded_mlp2_input=model_config.get("SHARDED_MLP2_INPUT_MEMCFG"),
            decode_residual=model_config.get("DECODE_RESIDUAL_MEMCFG"),
        )


# =============================================================================
# Module Template Classes
# =============================================================================


class MLPModuleDecodeTemplate(LightweightModule):
    """
    Template class for decode-mode MLP module.
    Handles both TG and non-TG topologies.
    """

    def __init__(self, mesh_device, tt_ccl, args, weights: MLPWeights, model_config, layer_num):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.layer_num = layer_num

        # Initialize weights
        self.weights = weights
        for key in ["w1", "w2", "w3"]:
            attr = getattr(self.weights, key)
            if hasattr(attr, "get_weight"):
                setattr(self.weights, key, attr.get_weight())

        self._init_ops_config()
        self._init_memory_configs()

    def _init_ops_config(self):
        pass  # Will be generated

    def _init_memory_configs(self):
        pass  # Will be generated

    def forward(self, x):
        # Will be replaced with appropriate template
        pass


class MLPModulePrefillTemplate(LightweightModule):
    """
    Template class for prefill-mode MLP module.
    Handles both TG and non-TG topologies.
    """

    def __init__(self, mesh_device, tt_ccl, args, weights: MLPWeights, model_config, layer_num):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.layer_num = layer_num
        self.prefill_len_cutoff = args.prefill_len_cutoff

        # Initialize weights
        self.weights = weights
        for key in ["w1", "w2", "w3"]:
            attr = getattr(self.weights, key)
            if hasattr(attr, "get_weight"):
                setattr(self.weights, key, attr.get_weight())

    def forward(self, x, seq_len):
        # Reshape for long sequences
        if seq_len >= self.prefill_len_cutoff:
            x = ttnn.reshape(x, [1, seq_len // self.prefill_len_cutoff, self.prefill_len_cutoff, -1])

        # Initialize ops config with seq_len
        self._init_ops_config(seq_len)

        # Will be replaced with appropriate template

    def _init_ops_config(self, seq_len):
        pass  # Will be generated


# =============================================================================
# Code Generator
# =============================================================================


class TTTv2RealMLPCodeGen:
    """
    Generates optimized MLP implementations based on configuration.

    This generator follows the "Codegen Selects" approach:
    1. Looks up the right template from TEMPLATE_REGISTRY
    2. Extracts template source code
    3. Generates appropriate config initialization
    4. Assembles into a complete module class
    """

    def __init__(
        self,
        mlp_config: MLPConfig,
        mode: str = "decode",
    ):
        self.mlp_config = mlp_config
        self.mode = mode
        self._validate_config()

        # Select the template
        self.selected_template = get_template_for_config(mlp_config.topology, mode, mlp_config.dim_class)
        self.template_name = self.selected_template.__name__

    def _validate_config(self):
        """Validate configuration."""
        if self.mlp_config.is_galaxy and self.mlp_config.hidden_dim % 32 != 0:
            raise ValueError(f"Galaxy requires hidden_dim divisible by 32, got {self.mlp_config.hidden_dim}")

    def generate_module_class(self) -> str:
        """Generate complete module class with initialization and forward."""
        topology_name = self.mlp_config.topology.value.replace("_", "")
        dim_class_name = self.mlp_config.dim_class.value
        class_name = f"TTTv2MLP_{topology_name}_{self.mode}_{dim_class_name}"

        lines = []

        # Imports
        lines.append(get_import_source(GENERATED_MODULE_NAMESPACE))
        lines.append("")

        # Config classes
        lines.extend(class_to_source(OpConfig))
        lines.append("")
        lines.extend(class_to_source(MLPOpConfigs))
        lines.append("")
        lines.extend(class_to_source(MemoryConfigs))
        lines.append("")
        lines.extend(class_to_source(MLPWeights))
        lines.append("")

        # Template function
        lines.append(
            f"# Template: {self.template_name} (selected for {self.mlp_config.topology.value}, {self.mode}, {self.mlp_config.dim_class.value})"
        )
        impl_source = function_to_source(self.selected_template)
        for line in impl_source:
            lines.append(line)
        lines.append("")

        # Module class
        if self.mode == "decode":
            self._generate_decode_class(lines, class_name)
        else:
            self._generate_prefill_class(lines, class_name)

        return "\n".join(lines)

    def _generate_decode_class(self, lines: list, class_name: str):
        """Generate decode-mode module class."""
        lines.append(f"class {class_name}(LightweightModule):")
        lines.append(f'    """')
        lines.append(f"    Auto-generated MLP module for {self.mlp_config.topology.value} decode mode")
        lines.append(f"    Configuration:")
        lines.append(f"      - dim: {self.mlp_config.dim}")
        lines.append(f"      - hidden_dim: {self.mlp_config.hidden_dim}")
        lines.append(f"      - is_galaxy: {self.mlp_config.is_galaxy}")
        lines.append(f'    """')
        lines.append("")
        lines.append("    def __init__(self, mesh_device, tt_ccl, args, weights, model_config, layer_num):")
        lines.append("        super().__init__()")
        lines.append("        self.mesh_device = mesh_device")
        lines.append("        self.tt_ccl = tt_ccl")
        lines.append("        self.args = args")
        lines.append("        self.dim = args.dim")
        lines.append("        self.model_config = model_config")
        lines.append("        self.layer_num = layer_num")
        lines.append("")
        lines.append("        self.weights = weights")
        lines.append('        for key in ["w1", "w2", "w3"]:')
        lines.append("            attr = getattr(self.weights, key)")
        lines.append('            if hasattr(attr, "get_weight"):')
        lines.append("                setattr(self.weights, key, attr.get_weight())")
        lines.append("")
        lines.append("        self._init_ops_config()")
        lines.append("        self._init_memory_configs()")
        lines.append("")

        # Generate _init_ops_config
        self._generate_decode_ops_config(lines)

        # Generate _init_memory_configs
        self._generate_memory_configs(lines)

        # Generate forward
        lines.append("    def forward(self, x):")
        lines.append(
            f"        return {self.template_name}(x, self.weights, self.ops, self.mem_configs, self.mesh_device, self.tt_ccl, self.args)"
        )
        lines.append("")

    def _generate_prefill_class(self, lines: list, class_name: str):
        """Generate prefill-mode module class."""
        lines.append(f"class {class_name}(LightweightModule):")
        lines.append(f'    """')
        lines.append(f"    Auto-generated MLP module for {self.mlp_config.topology.value} prefill mode")
        lines.append(f"    Configuration:")
        lines.append(f"      - dim: {self.mlp_config.dim}")
        lines.append(f"      - hidden_dim: {self.mlp_config.hidden_dim}")
        lines.append(f"      - is_galaxy: {self.mlp_config.is_galaxy}")
        lines.append(f'    """')
        lines.append("")
        lines.append("    def __init__(self, mesh_device, tt_ccl, args, weights, model_config, layer_num):")
        lines.append("        super().__init__()")
        lines.append("        self.mesh_device = mesh_device")
        lines.append("        self.tt_ccl = tt_ccl")
        lines.append("        self.args = args")
        lines.append("        self.dim = args.dim")
        lines.append("        self.model_config = model_config")
        lines.append("        self.layer_num = layer_num")
        lines.append("        self.prefill_len_cutoff = args.prefill_len_cutoff")
        lines.append("")
        lines.append("        self.weights = weights")
        lines.append('        for key in ["w1", "w2", "w3"]:')
        lines.append("            attr = getattr(self.weights, key)")
        lines.append('            if hasattr(attr, "get_weight"):')
        lines.append("                setattr(self.weights, key, attr.get_weight())")
        lines.append("")

        # Generate _init_ops_config (takes seq_len)
        self._generate_prefill_ops_config(lines)

        # Generate forward
        lines.append("    def forward(self, x, seq_len):")
        lines.append("        # Reshape for long sequences")
        lines.append("        if seq_len >= self.prefill_len_cutoff:")
        lines.append(
            "            x = ttnn.reshape(x, [1, seq_len // self.prefill_len_cutoff, self.prefill_len_cutoff, -1])"
        )
        lines.append("")
        lines.append("        # Initialize ops config with seq_len")
        lines.append("        self._init_ops_config(seq_len)")
        lines.append("")
        lines.append(
            f"        return {self.template_name}(x, self.weights, self.ops, self.mem_configs, self.mesh_device, self.tt_ccl, self.args, seq_len)"
        )
        lines.append("")

    def _generate_decode_ops_config(self, lines: list):
        """Generate _init_ops_config for decode mode."""
        lines.append("    def _init_ops_config(self):")
        lines.append("        from models.tt_transformers.tt.model_config import OpGroup, TensorGroup")
        lines.append("")
        lines.append("        layer_num = max(self.layer_num, 0)")
        lines.append("")
        lines.append('        activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(')
        lines.append("            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION")
        lines.append("        )")
        lines.append('        ff1_3_compute = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(')
        lines.append("            decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args")
        lines.append("        )")
        lines.append('        ff2_compute = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(')
        lines.append("            decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=self.args")
        lines.append("        )")
        lines.append("")

        if self.mlp_config.is_galaxy:
            lines.append('        pc_1_3 = self.model_config["FF1_3_TG_PROGCFG"] if self.dim >= 4096 else None')
            lines.append('        pc_2 = self.model_config["FF2_TG_PROGCFG"] if self.dim >= 4096 else None')
        else:
            lines.append('        pc_1_3 = self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"]')
            lines.append('        pc_2 = self.model_config["DECODE_MLP_W2_PRG_CONFIG"]')

        lines.append("")
        lines.append("        ff1_ff3_config = OpConfig(")
        lines.append("            compute_kernel_config=ff1_3_compute,")
        lines.append("            program_config=pc_1_3,")
        lines.append("        )")
        lines.append("        ff2_config = OpConfig(")
        lines.append("            compute_kernel_config=ff2_compute,")
        lines.append("            program_config=pc_2,")
        lines.append("        )")
        lines.append("")
        lines.append(
            '        activation_type = self.args.mlp_activation_type if hasattr(self.args, "mlp_activation_type") else ttnn.UnaryOpType.SILU'
        )
        lines.append("")
        lines.append("        self.ops = MLPOpConfigs(")
        lines.append("            ff1_ff3=ff1_ff3_config,")
        lines.append("            ff2=ff2_config,")
        lines.append("            activation_type=activation_type,")
        lines.append("            activation_dtype=activation_dtype,")
        lines.append("        )")
        lines.append("")

    def _generate_prefill_ops_config(self, lines: list):
        """Generate _init_ops_config for prefill mode."""
        lines.append("    def _init_ops_config(self, seq_len):")
        lines.append("        from models.tt_transformers.tt.model_config import OpGroup, TensorGroup")
        lines.append("")
        lines.append("        layer_num = max(self.layer_num, 0)")
        lines.append("")
        lines.append('        activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(')
        lines.append("            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION")
        lines.append("        )")
        lines.append('        ff1_3_compute = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(')
        lines.append("            decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args")
        lines.append("        )")
        lines.append('        ff2_compute = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(')
        lines.append("            decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=self.args")
        lines.append("        )")
        lines.append("")
        lines.append('        pc_1_3 = self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"](seq_len)')
        lines.append('        pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG"](seq_len)')
        lines.append("")
        lines.append("        ff1_ff3_config = OpConfig(")
        lines.append("            compute_kernel_config=ff1_3_compute,")
        lines.append("            program_config=pc_1_3,")
        lines.append("        )")
        lines.append("        ff2_config = OpConfig(")
        lines.append("            compute_kernel_config=ff2_compute,")
        lines.append("            program_config=pc_2,")
        lines.append("        )")
        lines.append("")
        lines.append(
            '        activation_type = self.args.mlp_activation_type if hasattr(self.args, "mlp_activation_type") else ttnn.UnaryOpType.SILU'
        )
        lines.append("")
        lines.append("        self.ops = MLPOpConfigs(")
        lines.append("            ff1_ff3=ff1_ff3_config,")
        lines.append("            ff2=ff2_config,")
        lines.append("            activation_type=activation_type,")
        lines.append("            activation_dtype=activation_dtype,")
        lines.append("        )")
        lines.append("")
        lines.append("        # Initialize memory configs (no seq_len dependency)")
        lines.append("        self._init_memory_configs()")
        lines.append("")

        # Add memory config init for prefill
        self._generate_memory_configs(lines)

    def _generate_memory_configs(self, lines: list):
        """Generate _init_memory_configs."""
        lines.append("    def _init_memory_configs(self):")
        if self.mlp_config.is_galaxy:
            lines.append("        self.mem_configs = MemoryConfigs(")
            lines.append('            ff1_out_reduce_scatter=self.model_config.get("FF1_OUT_REDUCE_SCATTER_MEMCFG"),')
            lines.append('            ff1_out_gathered=self.model_config.get("FF1_OUT_GATHERED_MEMCFG"),')
            lines.append('            sharded_mlp2_input=self.model_config.get("SHARDED_MLP2_INPUT_MEMCFG"),')
            lines.append('            ff2_out_reduce_scatter=self.model_config.get("FF2_OUT_REDUCE_SCATTER_MEMCFG"),')
            lines.append('            sharded_attn_input=self.model_config.get("SHARDED_ATTN_INPUT_MEMCFG"),')
            lines.append('            decode_residual=self.model_config.get("DECODE_RESIDUAL_MEMCFG"),')
            lines.append("        )")
        else:
            lines.append("        self.mem_configs = MemoryConfigs(")
            lines.append('            sharded_mlp2_input=self.model_config.get("SHARDED_MLP2_INPUT_MEMCFG"),')
            lines.append('            decode_residual=self.model_config.get("DECODE_RESIDUAL_MEMCFG"),')
            lines.append("        )")
        lines.append("")


# =============================================================================
# SaveSource and Main API
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

    def get_filename(self, mlp_config: Optional[MLPConfig] = None, mode: str = "decode") -> Optional[str]:
        if self._filename:
            filepath = Path(self._filename)
            if not filepath.is_absolute():
                return str(self._get_default_dir() / filepath)
            return str(filepath)

        if mlp_config:
            topology_name = mlp_config.topology.value.replace("_", "")
            dim_class = mlp_config.dim_class.value
            default_name = f"mlp_real_{topology_name}_{mode}_{dim_class}_{mlp_config.dim}.py"
        else:
            default_name = "mlp_real_single_device_decode_small_4096.py"

        return str(self._get_default_dir() / default_name)


def MLP(
    mlp_config: MLPConfig,
    *,
    mode: str = "decode",
    save_source: Optional[SaveSource] = None,
) -> type:
    """
    Main API to compile a real MLP module for specific hardware and configuration.

    Args:
        mlp_config: MLP configuration.
        mode: "decode" or "prefill"
        save_source: Optional SaveSource instance for saving generated code.

    Returns:
        Compiled module class
    """
    codegen = TTTv2RealMLPCodeGen(mlp_config, mode=mode)
    source_code = codegen.generate_module_class()

    # Save source if requested
    filename = "<string>"
    if save_source is not None:
        header = (
            f"# SPDX-FileCopyrightText: © {datetime.now().year} Tenstorrent AI ULC\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
            "# Auto-generated by TTTv2 Real MLP CodeGen\n"
        )
        source_code = header + source_code

        filepath = save_source.get_filename(mlp_config, mode)
        if filepath:
            with open(filepath, "w") as f:
                f.write(source_code)
            filename = str(Path(filepath).resolve())

    # Compile
    namespace = GENERATED_MODULE_NAMESPACE.copy()
    code_obj = compile(source_code, filename, "exec")
    exec(code_obj, namespace)

    # Find generated class
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
    print("=== Example 1: Non-TG Decode MLP ===")

    mlp_config = MLPConfig(
        dim=4096,
        hidden_dim=11008,
        unpadded_hidden_dim=11008,
        num_devices=1,
        is_galaxy=False,
    )

    save_source = SaveSource()
    module_class = MLP(mlp_config, mode="decode", save_source=save_source)

    print(f"Generated class: {module_class}")
    if module_class:
        filename = save_source.get_filename(mlp_config, "decode")
        if filename:
            with open(filename, "r") as f:
                print("\n".join(f.readlines()[:80]))
            print("... [truncated]\n")

    print("\n=== Example 2: TG Decode MLP (Large Dim) ===")

    mlp_config_tg = MLPConfig(
        dim=8192,
        hidden_dim=28672,
        unpadded_hidden_dim=28672,
        num_devices=32,
        is_galaxy=True,
        cluster_shape=(4, 8),
    )

    save_source_tg = SaveSource()
    module_class_tg = MLP(mlp_config_tg, mode="decode", save_source=save_source_tg)

    print(f"Generated class: {module_class_tg}")
    if module_class_tg:
        filename_tg = save_source_tg.get_filename(mlp_config_tg, "decode")
        if filename_tg:
            with open(filename_tg, "r") as f:
                print("\n".join(f.readlines()[:100]))
            print("... [truncated]\n")

    print("\n=== Example 3: TG Prefill MLP ===")

    module_class_prefill = MLP(mlp_config_tg, mode="prefill", save_source=SaveSource())
    print(f"Generated class: {module_class_prefill}")
