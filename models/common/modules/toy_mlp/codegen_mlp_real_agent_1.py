"""
TTTv2 Real MLP Module with Code Generation

This file brings the real MLP implementation from models/tt_transformers/tt/mlp.py
into the TTTv2 codegen pattern following tttv2_design_proposal.md and tttv2_module_design.md.

Key design points:
1. Template functions are real runnable Python functions (no self)
2. Minimal if-else in templates - static conditions are handled by template selection
3. No torch dependency - uses LightweightModule
4. Codegen selects the right template based on (topology, mode) configuration
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
    """
    Generate import statements from the namespace of real Python symbols.
    """
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
    T3K_1D = "t3k_1d"  # T3K with 1D ring topology
    GALAXY_2D = "galaxy_2d"  # Galaxy with 2D mesh topology


class Mode(Enum):
    """Execution mode"""

    DECODE = "decode"
    PREFILL = "prefill"


@dataclass
class MLPModelConfig:
    """
    Model configuration parameters that affect MLP behavior.
    These come from the model_config dict in the original mlp.py.
    """

    dim: int  # Model dimension (hidden size)
    hidden_dim: int  # Intermediate size (after padding if applicable)
    num_devices: int = 1
    is_galaxy: bool = False
    prefill_len_cutoff: int = 1024  # 512 for Blackhole, 1024 for Wormhole

    # CCL settings
    num_reduce_scatter_links: int = 1
    num_all_gather_links: int = 2
    ccl_dtype: Any = None  # Will be set to ttnn.bfloat8_b or similar

    # Activation type
    activation_type: Any = None  # Will be set to ttnn.UnaryOpType.SILU

    def __post_init__(self):
        if self.ccl_dtype is None:
            self.ccl_dtype = ttnn.bfloat8_b
        if self.activation_type is None:
            self.activation_type = ttnn.UnaryOpType.SILU


@dataclass
class OpConfig:
    """Configuration for a TTNN operation"""

    compute_kernel_config: Optional[Any] = None
    memory_config: Optional[Any] = None
    dtype: Optional[Any] = None
    program_config: Optional[Any] = None


@dataclass
class MLPOpConfigs:
    """Container for MLP operation configurations - used for single device / simple topologies"""

    w1: OpConfig  # gate projection
    w2: OpConfig  # down projection
    w3: OpConfig  # up projection
    w2_in_memory_config: Optional[Any] = None  # Memory config for w2 input
    activation_type: Any = None

    def __post_init__(self):
        if self.activation_type is None:
            self.activation_type = ttnn.UnaryOpType.SILU


@dataclass
class TGMLPOpConfigs:
    """Container for Galaxy TG MLP operation configurations"""

    w1: OpConfig
    w2: OpConfig
    w3: OpConfig

    # CCL configurations
    reduce_scatter_memory_config: Optional[Any] = None
    all_gather_memory_config: Optional[Any] = None
    all_reduce_memory_config: Optional[Any] = None

    # CCL settings
    cluster_axis: int = 1
    num_reduce_scatter_links: int = 1
    num_all_gather_links: int = 2

    activation_type: Any = None
    ccl_dtype: Any = None

    def __post_init__(self):
        if self.activation_type is None:
            self.activation_type = ttnn.UnaryOpType.SILU
        if self.ccl_dtype is None:
            self.ccl_dtype = ttnn.bfloat8_b


@dataclass
class MLPWeights:
    """Container for MLP weights"""

    w1: LazyWeight  # gate_proj weight
    w2: LazyWeight  # down_proj weight
    w3: LazyWeight  # up_proj weight


_ = None  # Stop inspect.getsource from grabbing following comments


# =============================================================================
# TEMPLATE 1: Single Device / T3K Decode
# =============================================================================
# Standard decode path without TG CCL operations


def forward_mlp_decode_impl(x, weights, ops, model_cfg):
    """
    MLP forward pass for decode mode on single device or T3K.

    Dataflow: x -> [w1, w3] -> silu(w1) * w3 -> w2 -> output

    Args:
        x: Input tensor
        weights: Container with weight tensors (w1, w2, w3)
        ops: MLPOpConfigs with operation configurations
        model_cfg: MLPModelConfig with model parameters
    """
    # W1 (gate) and W3 (up) projections - DRAM sharded matmuls
    w1_out = ttnn.linear(
        x,
        weights.w1,
        dtype=ops.w1.dtype,
        compute_kernel_config=ops.w1.compute_kernel_config,
        program_config=ops.w1.program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )

    w3_out = ttnn.linear(
        x,
        weights.w3,
        dtype=ops.w3.dtype,
        compute_kernel_config=ops.w3.compute_kernel_config,
        program_config=ops.w3.program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )
    ttnn.deallocate(x)

    # Activation and element-wise multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ops.activation_type],
        dtype=ops.w2.dtype or ttnn.bfloat8_b,
        memory_config=w1_out.memory_config(),
    )

    # w2 may use a different core grid, reshard if needed
    if ops.w2_in_memory_config is not None:
        w2_in = ttnn.to_memory_config(w2_in, ops.w2_in_memory_config)

    ttnn.deallocate(w3_out)
    ttnn.deallocate(w1_out)

    # W2 (down) projection
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.w2.compute_kernel_config,
        dtype=ops.w2.dtype,
        program_config=ops.w2.program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )
    ttnn.deallocate(w2_in)

    return w2_out


_ = None


# =============================================================================
# TEMPLATE 2: Single Device / T3K Prefill
# =============================================================================


def forward_mlp_prefill_impl(x, weights, ops, model_cfg, seq_len):
    """
    MLP forward pass for prefill mode on single device or T3K.

    For large sequence lengths, reshapes input to fit on device and parallelize.

    Args:
        x: Input tensor
        weights: Container with weight tensors (w1, w2, w3)
        ops: MLPOpConfigs with operation configurations
        model_cfg: MLPModelConfig with model parameters
        seq_len: Sequence length for prefill
    """
    # Reshape for large sequences
    if seq_len >= model_cfg.prefill_len_cutoff:
        x = ttnn.reshape(x, [1, seq_len // model_cfg.prefill_len_cutoff, model_cfg.prefill_len_cutoff, -1])

    # W1 and W3 projections
    w1_out = ttnn.linear(
        x,
        weights.w1,
        dtype=ops.w1.dtype,
        compute_kernel_config=ops.w1.compute_kernel_config,
        program_config=ops.w1.program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    w3_out = ttnn.linear(
        x,
        weights.w3,
        dtype=ops.w3.dtype,
        compute_kernel_config=ops.w3.compute_kernel_config,
        program_config=ops.w3.program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(x)

    # Activation and multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ops.activation_type],
        dtype=ops.w2.dtype or ttnn.bfloat8_b,
        memory_config=w1_out.memory_config(),
    )

    ttnn.deallocate(w3_out)
    ttnn.deallocate(w1_out)

    # W2 projection
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.w2.compute_kernel_config,
        dtype=ops.w2.dtype,
        program_config=ops.w2.program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(w2_in)

    return w2_out


_ = None


# =============================================================================
# TEMPLATE 3: Galaxy TG Decode with All-Reduce
# =============================================================================
# For smaller models (dim < 8192) on Galaxy, uses all_reduce instead of reduce_scatter


def forward_mlp_tg_decode_allreduce_impl(x, weights, ops, model_cfg, tt_ccl, mesh_device):
    """
    MLP forward pass for Galaxy TG decode mode using all_reduce.

    Used for smaller models (dim < 8192) where all_reduce is more efficient.

    Args:
        x: Input tensor (sharded across devices)
        weights: Container with weight tensors
        ops: TGMLPOpConfigs with operation configurations
        model_cfg: MLPModelConfig with model parameters
        tt_ccl: TT_CCL instance for collective operations
        mesh_device: Mesh device for multi-device operations
    """
    from models.tt_transformers.tt.ccl import tt_all_reduce

    # W1 and W3 projections
    w1_out = ttnn.linear(
        x,
        weights.w1,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=ops.w1.compute_kernel_config,
        program_config=ops.w1.program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )

    w3_out = ttnn.linear(
        x,
        weights.w3,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=ops.w3.compute_kernel_config,
        program_config=ops.w3.program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )
    ttnn.deallocate(x)

    # All-reduce for W1 and W3 outputs
    w1_out = tt_all_reduce(
        w1_out,
        mesh_device,
        tt_ccl,
        cluster_axis=1,
        num_all_gather_links=2,
        sharded=True,
        memory_config=ops.all_reduce_memory_config,
    )
    w3_out = tt_all_reduce(
        w3_out,
        mesh_device,
        tt_ccl,
        cluster_axis=1,
        num_all_gather_links=2,
        sharded=True,
        memory_config=ops.all_reduce_memory_config,
    )

    # Activation and multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ops.activation_type],
        dtype=ops.w2.dtype or ttnn.bfloat8_b,
        memory_config=w1_out.memory_config(),
    )

    ttnn.deallocate(w3_out)
    ttnn.deallocate(w1_out)

    # W2 projection
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.w2.compute_kernel_config,
        dtype=ops.ccl_dtype,
        program_config=ops.w2.program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )
    ttnn.deallocate(w2_in)

    # Final all-reduce
    w2_out_reduced = tt_all_reduce(
        w2_out,
        mesh_device,
        tt_ccl,
        cluster_axis=0,
        dim=0,
        num_reduce_scatter_links=ops.num_reduce_scatter_links,
        num_all_gather_links=ops.num_all_gather_links,
        sharded=True,
        memory_config=ops.all_reduce_memory_config,
        dtype=ops.ccl_dtype,
    )

    # Reshape to ensure dim 0 and 1 are 1
    original_shape = w2_out_reduced.shape
    w2_out_reduced = ttnn.reshape(
        w2_out_reduced,
        (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1]),
    )

    return w2_out_reduced


_ = None


# =============================================================================
# TEMPLATE 4: Galaxy TG Decode with Reduce-Scatter (Large Models)
# =============================================================================
# For large models (dim == 8192) on Galaxy, uses reduce_scatter + all_gather


def forward_mlp_tg_decode_reducescatter_impl(x, weights, ops, model_cfg, tt_ccl, mesh_device):
    """
    MLP forward pass for Galaxy TG decode mode using reduce_scatter + all_gather.

    Used for large models (dim == 8192) where reduce_scatter is more efficient.

    Args:
        x: Input tensor (sharded across devices)
        weights: Container with weight tensors
        ops: TGMLPOpConfigs with operation configurations
        model_cfg: MLPModelConfig with model parameters
        tt_ccl: TT_CCL instance for collective operations
        mesh_device: Mesh device for multi-device operations
    """
    from models.tt_transformers.tt.ccl import tt_all_reduce

    input_mem_cfg = None

    # W1 and W3 projections
    w1_out = ttnn.linear(
        x,
        weights.w1,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=ops.w1.compute_kernel_config,
        program_config=ops.w1.program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )

    w3_out = ttnn.linear(
        x,
        weights.w3,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=ops.w3.compute_kernel_config,
        program_config=ops.w3.program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )
    ttnn.deallocate(x)

    input_mem_cfg = w1_out.memory_config()
    cluster_axis = 1

    # Reduce scatter for W1
    w1_out = ttnn.experimental.reduce_scatter_minimal_async(
        w1_out,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        num_links=ops.num_reduce_scatter_links,
        cluster_axis=cluster_axis,
        memory_config=ops.reduce_scatter_memory_config,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    # Reduce scatter for W3
    w3_out = ttnn.experimental.reduce_scatter_minimal_async(
        w3_out,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        num_links=1,
        cluster_axis=cluster_axis,
        memory_config=ops.reduce_scatter_memory_config,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    # Activation and multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ops.activation_type],
        dtype=ops.w2.dtype or ttnn.bfloat8_b,
        memory_config=w1_out.memory_config(),
    )

    ttnn.deallocate(w3_out)
    ttnn.deallocate(w1_out)

    # All-gather before W2
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

    # W2 projection
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.w2.compute_kernel_config,
        dtype=ops.ccl_dtype,
        program_config=ops.w2.program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )
    ttnn.deallocate(w2_in)

    # Final all-reduce
    w2_out_reduced = tt_all_reduce(
        w2_out,
        mesh_device,
        tt_ccl,
        cluster_axis=0,
        dim=3,
        num_reduce_scatter_links=ops.num_reduce_scatter_links,
        num_all_gather_links=ops.num_all_gather_links,
        sharded=True,
        memory_config=ops.all_reduce_memory_config,
        dtype=ops.ccl_dtype,
        use_composite=True,
    )

    # Reshape to ensure dim 0 and 1 are 1
    original_shape = w2_out_reduced.shape
    w2_out_reduced = ttnn.reshape(
        w2_out_reduced,
        (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1]),
    )

    return w2_out_reduced


_ = None


# =============================================================================
# TEMPLATE 5: Galaxy TG Prefill
# =============================================================================


def forward_mlp_tg_prefill_impl(x, weights, ops, model_cfg, tt_ccl, mesh_device, seq_len):
    """
    MLP forward pass for Galaxy TG prefill mode.

    Args:
        x: Input tensor (sharded across devices)
        weights: Container with weight tensors
        ops: TGMLPOpConfigs with operation configurations
        model_cfg: MLPModelConfig with model parameters
        tt_ccl: TT_CCL instance for collective operations
        mesh_device: Mesh device for multi-device operations
        seq_len: Sequence length for prefill
    """
    from models.tt_transformers.tt.ccl import tt_all_reduce

    # Reshape for large sequences
    if seq_len >= model_cfg.prefill_len_cutoff:
        x = ttnn.reshape(x, [1, seq_len // model_cfg.prefill_len_cutoff, model_cfg.prefill_len_cutoff, -1])

    input_mem_cfg = None

    # W1 and W3 projections
    w1_out = ttnn.linear(
        x,
        weights.w1,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=ops.w1.compute_kernel_config,
        program_config=ops.w1.program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    w3_out = ttnn.linear(
        x,
        weights.w3,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=ops.w3.compute_kernel_config,
        program_config=ops.w3.program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(x)

    input_mem_cfg = w1_out.memory_config()
    cluster_axis = 1

    # Reduce scatter for W1
    w1_out = ttnn.experimental.reduce_scatter_minimal_async(
        w1_out,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        num_links=ops.num_reduce_scatter_links,
        cluster_axis=cluster_axis,
        memory_config=None,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    # Reduce scatter for W3
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

    # Activation and multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ops.activation_type],
        dtype=ops.w2.dtype or ttnn.bfloat8_b,
        memory_config=w1_out.memory_config(),
    )

    ttnn.deallocate(w3_out)
    ttnn.deallocate(w1_out)

    # All-gather before W2
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

    # W2 projection
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.w2.compute_kernel_config,
        dtype=ops.ccl_dtype,
        program_config=ops.w2.program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(w2_in)

    # Final all-reduce
    w2_out_reduced = tt_all_reduce(
        w2_out,
        mesh_device,
        tt_ccl,
        cluster_axis=0,
        dim=3,
        num_reduce_scatter_links=ops.num_reduce_scatter_links,
        num_all_gather_links=ops.num_all_gather_links,
        sharded=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ops.ccl_dtype,
        use_composite=True if model_cfg.dim == 8192 else False,
    )

    # Reshape to ensure dim 0 and 1 are 1
    original_shape = w2_out_reduced.shape
    w2_out_reduced = ttnn.reshape(
        w2_out_reduced,
        (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1]),
    )

    return w2_out_reduced


_ = None


# =============================================================================
# TEMPLATE 6: T3K Decode with All-Reduce
# =============================================================================


def forward_mlp_t3k_decode_impl(x, weights, ops, model_cfg, tt_ccl, mesh_device):
    """
    MLP forward pass for T3K decode mode with all-reduce.

    Args:
        x: Input tensor
        weights: Container with weight tensors
        ops: MLPOpConfigs with operation configurations
        model_cfg: MLPModelConfig with model parameters
        tt_ccl: TT_CCL instance for collective operations
        mesh_device: Mesh device for multi-device operations
    """
    from models.tt_transformers.tt.ccl import tt_all_reduce

    # W1 and W3 projections
    w1_out = ttnn.linear(
        x,
        weights.w1,
        dtype=ops.w1.dtype,
        compute_kernel_config=ops.w1.compute_kernel_config,
        program_config=ops.w1.program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )

    w3_out = ttnn.linear(
        x,
        weights.w3,
        dtype=ops.w3.dtype,
        compute_kernel_config=ops.w3.compute_kernel_config,
        program_config=ops.w3.program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )
    ttnn.deallocate(x)

    # Activation and multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ops.activation_type],
        dtype=ops.w2.dtype or ttnn.bfloat8_b,
        memory_config=w1_out.memory_config(),
    )

    # Reshard for W2 if needed
    if ops.w2_in_memory_config is not None:
        w2_in = ttnn.to_memory_config(w2_in, ops.w2_in_memory_config)

    ttnn.deallocate(w3_out)
    ttnn.deallocate(w1_out)

    # W2 projection
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.w2.compute_kernel_config,
        dtype=ops.w2.dtype,
        program_config=ops.w2.program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )
    ttnn.deallocate(w2_in)

    # All-reduce across devices
    w2_out_reduced = tt_all_reduce(
        w2_out,
        mesh_device,
        tt_ccl,
        cluster_axis=0,
        dim=3,
        sharded=True,
        memory_config=w2_out.memory_config(),
        dtype=model_cfg.ccl_dtype,
    )

    # Reshape to ensure dim 0 and 1 are 1
    original_shape = w2_out_reduced.shape
    w2_out_reduced = ttnn.reshape(
        w2_out_reduced,
        (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1]),
    )

    return w2_out_reduced


_ = None


# =============================================================================
# TEMPLATE 7: T3K Prefill with All-Reduce
# =============================================================================


def forward_mlp_t3k_prefill_impl(x, weights, ops, model_cfg, tt_ccl, mesh_device, seq_len):
    """
    MLP forward pass for T3K prefill mode with all-reduce.

    Args:
        x: Input tensor
        weights: Container with weight tensors
        ops: MLPOpConfigs with operation configurations
        model_cfg: MLPModelConfig with model parameters
        tt_ccl: TT_CCL instance for collective operations
        mesh_device: Mesh device for multi-device operations
        seq_len: Sequence length for prefill
    """
    from models.tt_transformers.tt.ccl import tt_all_reduce

    # Reshape for large sequences
    if seq_len >= model_cfg.prefill_len_cutoff:
        x = ttnn.reshape(x, [1, seq_len // model_cfg.prefill_len_cutoff, model_cfg.prefill_len_cutoff, -1])

    # W1 and W3 projections
    w1_out = ttnn.linear(
        x,
        weights.w1,
        dtype=ops.w1.dtype,
        compute_kernel_config=ops.w1.compute_kernel_config,
        program_config=ops.w1.program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    w3_out = ttnn.linear(
        x,
        weights.w3,
        dtype=ops.w3.dtype,
        compute_kernel_config=ops.w3.compute_kernel_config,
        program_config=ops.w3.program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(x)

    # Activation and multiplication
    w2_in = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ops.activation_type],
        dtype=ops.w2.dtype or ttnn.bfloat8_b,
        memory_config=w1_out.memory_config(),
    )

    ttnn.deallocate(w3_out)
    ttnn.deallocate(w1_out)

    # W2 projection
    w2_out = ttnn.linear(
        w2_in,
        weights.w2,
        compute_kernel_config=ops.w2.compute_kernel_config,
        dtype=ops.w2.dtype,
        program_config=ops.w2.program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(w2_in)

    # All-reduce across devices
    w2_out_reduced = tt_all_reduce(
        w2_out,
        mesh_device,
        tt_ccl,
        cluster_axis=0,
        dim=3,
        sharded=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=model_cfg.ccl_dtype,
    )

    # Reshape to ensure dim 0 and 1 are 1
    original_shape = w2_out_reduced.shape
    w2_out_reduced = ttnn.reshape(
        w2_out_reduced,
        (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1]),
    )

    return w2_out_reduced


_ = None


# =============================================================================
# Template Registry
# =============================================================================

# Key: (topology, mode, model_size_class)
# model_size_class: "small" (dim < 8192) or "large" (dim >= 8192)
TEMPLATE_REGISTRY = {
    # Single device - no CCL needed
    (Topology.SINGLE_DEVICE, Mode.DECODE, "any"): forward_mlp_decode_impl,
    (Topology.SINGLE_DEVICE, Mode.PREFILL, "any"): forward_mlp_prefill_impl,
    # T3K - uses all-reduce
    (Topology.T3K_1D, Mode.DECODE, "any"): forward_mlp_t3k_decode_impl,
    (Topology.T3K_1D, Mode.PREFILL, "any"): forward_mlp_t3k_prefill_impl,
    # Galaxy TG - different strategies based on model size
    (Topology.GALAXY_2D, Mode.DECODE, "small"): forward_mlp_tg_decode_allreduce_impl,
    (Topology.GALAXY_2D, Mode.DECODE, "large"): forward_mlp_tg_decode_reducescatter_impl,
    (Topology.GALAXY_2D, Mode.PREFILL, "any"): forward_mlp_tg_prefill_impl,
}


def get_model_size_class(dim: int) -> str:
    """Determine model size class based on dimension."""
    return "large" if dim >= 8192 else "small"


def get_template_for_config(topology: Topology, mode: Mode, dim: int = 4096):
    """
    Select the appropriate template function based on topology, mode, and model size.
    """
    size_class = get_model_size_class(dim)

    # Try specific size class first
    key = (topology, mode, size_class)
    if key in TEMPLATE_REGISTRY:
        return TEMPLATE_REGISTRY[key]

    # Fall back to "any" size class
    key = (topology, mode, "any")
    if key in TEMPLATE_REGISTRY:
        return TEMPLATE_REGISTRY[key]

    raise ValueError(
        f"No template registered for {(topology, mode, size_class)}. " f"Available: {list(TEMPLATE_REGISTRY.keys())}"
    )


# =============================================================================
# Op Config Initializers
# =============================================================================


def init_ops_config_single_device():
    """Config initializer for single device MLP."""
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    w1_config = OpConfig(
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
    )
    w2_config = OpConfig(
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
    )
    w3_config = OpConfig(
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
    )

    return MLPOpConfigs(
        w1=w1_config,
        w2=w2_config,
        w3=w3_config,
        activation_type=ttnn.UnaryOpType.SILU,
    )


def init_ops_config_tg():
    """Config initializer for Galaxy TG MLP."""
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    w1_config = OpConfig(
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat8_b,
    )
    w2_config = OpConfig(
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat8_b,
    )
    w3_config = OpConfig(
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat8_b,
    )

    return TGMLPOpConfigs(
        w1=w1_config,
        w2=w2_config,
        w3=w3_config,
        activation_type=ttnn.UnaryOpType.SILU,
        ccl_dtype=ttnn.bfloat8_b,
    )


# =============================================================================
# Module Templates
# =============================================================================


class MLPModuleSingleDeviceTemplate(LightweightModule):
    """Template class for single device MLP module."""

    def __init__(self, device, weights: MLPWeights, model_cfg: MLPModelConfig):
        super().__init__()
        self.device = device
        self.model_cfg = model_cfg

        # Initialize weights
        self.weights = weights
        for key in vars(self.weights):
            attr = getattr(self.weights, key)
            if hasattr(attr, "get_weight"):
                setattr(self.weights, key, attr.get_weight())

        self._init_ops_config()

    def _init_ops_config(self):
        pass

    def forward(self, x, mode="decode"):
        if mode == "decode":
            return forward_mlp_decode_impl(x, self.weights, self.ops, self.model_cfg)
        else:
            seq_len = x.shape[-2]
            return forward_mlp_prefill_impl(x, self.weights, self.ops, self.model_cfg, seq_len)


class MLPModuleT3KTemplate(LightweightModule):
    """Template class for T3K MLP module with CCL."""

    def __init__(
        self,
        device,
        weights: MLPWeights,
        model_cfg: MLPModelConfig,
        tt_ccl,
        mesh_device,
    ):
        super().__init__()
        self.device = device
        self.model_cfg = model_cfg
        self.tt_ccl = tt_ccl
        self.mesh_device = mesh_device

        # Initialize weights
        self.weights = weights
        for key in vars(self.weights):
            attr = getattr(self.weights, key)
            if hasattr(attr, "get_weight"):
                setattr(self.weights, key, attr.get_weight())

        self._init_ops_config()

    def _init_ops_config(self):
        pass

    def forward(self, x, mode="decode"):
        if mode == "decode":
            return forward_mlp_t3k_decode_impl(x, self.weights, self.ops, self.model_cfg, self.tt_ccl, self.mesh_device)
        else:
            seq_len = x.shape[-2]
            return forward_mlp_t3k_prefill_impl(
                x, self.weights, self.ops, self.model_cfg, self.tt_ccl, self.mesh_device, seq_len
            )


class MLPModuleTGTemplate(LightweightModule):
    """Template class for Galaxy TG MLP module with CCL."""

    def __init__(
        self,
        device,
        weights: MLPWeights,
        model_cfg: MLPModelConfig,
        tt_ccl,
        mesh_device,
    ):
        super().__init__()
        self.device = device
        self.model_cfg = model_cfg
        self.tt_ccl = tt_ccl
        self.mesh_device = mesh_device

        # Initialize weights
        self.weights = weights
        for key in vars(self.weights):
            attr = getattr(self.weights, key)
            if hasattr(attr, "get_weight"):
                setattr(self.weights, key, attr.get_weight())

        self._init_ops_config()

    def _init_ops_config(self):
        pass

    def forward(self, x, mode="decode"):
        if mode == "decode":
            if self.model_cfg.dim >= 8192:
                return forward_mlp_tg_decode_reducescatter_impl(
                    x, self.weights, self.ops, self.model_cfg, self.tt_ccl, self.mesh_device
                )
            else:
                return forward_mlp_tg_decode_allreduce_impl(
                    x, self.weights, self.ops, self.model_cfg, self.tt_ccl, self.mesh_device
                )
        else:
            seq_len = x.shape[-2]
            return forward_mlp_tg_prefill_impl(
                x, self.weights, self.ops, self.model_cfg, self.tt_ccl, self.mesh_device, seq_len
            )


# =============================================================================
# Code Generator
# =============================================================================


class TTTv2RealMLPCodeGen:
    """
    Generates optimized MLP implementations based on the real mlp.py.

    This generator:
    1. Looks up the right template from TEMPLATE_REGISTRY based on (topology, mode, size)
    2. Extracts the template source code
    3. Generates the appropriate config initialization
    4. Assembles into a complete module class
    """

    def __init__(
        self,
        model_cfg: MLPModelConfig,
        topology: Topology = Topology.SINGLE_DEVICE,
        mode: Mode = Mode.DECODE,
    ):
        self.model_cfg = model_cfg
        self.topology = topology
        self.mode = mode
        self._validate_config()

        # Select templates for both decode and prefill
        self.decode_template = get_template_for_config(topology, Mode.DECODE, model_cfg.dim)
        self.prefill_template = get_template_for_config(topology, Mode.PREFILL, model_cfg.dim)

    def _validate_config(self):
        """Validate that configuration is feasible"""
        if self.topology == Topology.GALAXY_2D and not self.model_cfg.is_galaxy:
            raise ValueError("Galaxy topology requires is_galaxy=True in model_cfg")

    def _get_template_class(self):
        """Get the appropriate template class based on topology."""
        if self.topology == Topology.SINGLE_DEVICE:
            return MLPModuleSingleDeviceTemplate
        elif self.topology == Topology.T3K_1D:
            return MLPModuleT3KTemplate
        else:  # GALAXY_2D
            return MLPModuleTGTemplate

    def _get_ops_init_function(self):
        """Get the appropriate ops config initializer."""
        if self.topology == Topology.GALAXY_2D:
            return init_ops_config_tg
        else:
            return init_ops_config_single_device

    def generate_ops_init_source(self) -> list:
        """Generate source code for _init_ops_config method."""
        init_func = self._get_ops_init_function()
        source_lines = function_to_source(init_func)

        # Process the body lines
        processed_lines = []
        for line in source_lines:
            current_line = line

            # Change return to assignment
            if "return MLPOpConfigs" in current_line:
                current_line = current_line.replace("return MLPOpConfigs", "self.ops = MLPOpConfigs")
            elif "return TGMLPOpConfigs" in current_line:
                current_line = current_line.replace("return TGMLPOpConfigs", "self.ops = TGMLPOpConfigs")

            processed_lines.append(current_line)

        return processed_lines[1:]  # Skip def line

    def generate_module_class(self) -> str:
        """Generate complete module class with initialization and forward."""
        topology_name = self.topology.value.replace("_", "")
        class_name = f"TTTv2MLP_{topology_name}_dim{self.model_cfg.dim}"

        class_lines = []

        # Add imports
        class_lines.append(get_import_source(GENERATED_MODULE_NAMESPACE))
        class_lines.append("")

        # Add config classes
        class_lines.extend(class_to_source(OpConfig))
        class_lines.append("")

        if self.topology == Topology.GALAXY_2D:
            class_lines.extend(class_to_source(TGMLPOpConfigs))
        else:
            class_lines.extend(class_to_source(MLPOpConfigs))
        class_lines.append("")

        class_lines.extend(class_to_source(MLPWeights))
        class_lines.append("")

        class_lines.extend(class_to_source(MLPModelConfig))
        class_lines.append("")

        # Add template functions
        class_lines.append(f"# Decode template: {self.decode_template.__name__}")
        for line in function_to_source(self.decode_template):
            class_lines.append(line)
        class_lines.append("")

        if self.prefill_template != self.decode_template:
            class_lines.append(f"# Prefill template: {self.prefill_template.__name__}")
            for line in function_to_source(self.prefill_template):
                class_lines.append(line)
            class_lines.append("")

        # Generate the module class
        template_class = self._get_template_class()
        template_source = class_to_source(template_class)

        # Replace class name and inject ops init
        final_lines = []
        skip = False

        for i, line in enumerate(template_source):
            if i == 0:
                # Replace class name
                final_lines.append(line.replace(template_class.__name__, class_name))
                # Add docstring
                final_lines.append(f'    """')
                final_lines.append(f"    Auto-generated MLP module for {topology_name}")
                final_lines.append(f"    Model dimension: {self.model_cfg.dim}")
                final_lines.append(f"    Hidden dimension: {self.model_cfg.hidden_dim}")
                final_lines.append(f'    """')
            elif "def _init_ops_config(self):" in line:
                final_lines.append(line)
                ops_init_lines = self.generate_ops_init_source()
                for l in ops_init_lines:
                    if l.strip():
                        final_lines.append(f"    {l}")
                    else:
                        final_lines.append("")
                final_lines.append("")
                skip = True
            elif skip:
                # Skip until next method or end of class
                if line.strip().startswith("def ") or (
                    line.strip() != "" and not line.startswith("        ") and not line.startswith("    def")
                ):
                    skip = False
                    if line.strip():
                        final_lines.append(line)
            else:
                final_lines.append(line)

        class_lines.extend(final_lines)

        return "\n".join(class_lines)


# =============================================================================
# Main API
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

    def get_filename(
        self, model_cfg: Optional[MLPModelConfig] = None, topology: Optional[Topology] = None
    ) -> Optional[str]:
        if self._filename:
            filepath = Path(self._filename)
            if not filepath.is_absolute():
                return str(self._get_default_dir() / filepath)
            return str(filepath)

        if model_cfg and topology:
            topo_name = topology.value.replace("_", "")
            default_name = f"mlp_real_{topo_name}_dim{model_cfg.dim}.py"
        else:
            default_name = "mlp_real_generated.py"

        return str(self._get_default_dir() / default_name)

    def save(
        self,
        source_code: str,
        model_cfg: Optional[MLPModelConfig] = None,
        topology: Optional[Topology] = None,
    ) -> None:
        filepath = self.get_filename(model_cfg, topology)
        if not filepath:
            return

        with open(filepath, "w") as f:
            years = datetime.now().year
            f.write(f"# SPDX-FileCopyrightText: © {years} Tenstorrent AI ULC\n")
            f.write("# SPDX-License-Identifier: Apache-2.0\n")
            f.write("# Auto-generated by TTTv2 CodeGen from codegen_mlp_real.py\n")
            f.write(source_code)


def RealMLP(
    model_cfg: MLPModelConfig,
    topology: Topology = Topology.SINGLE_DEVICE,
    *,
    save_source: Optional[SaveSource] = None,
) -> type:
    """
    Main API to compile a real MLP module for specific hardware and configuration.

    This generates code that matches the behavior of models/tt_transformers/tt/mlp.py
    but in the TTTv2 codegen pattern.

    Args:
        model_cfg: Model configuration parameters
        topology: Hardware topology (SINGLE_DEVICE, T3K_1D, GALAXY_2D)
        save_source: Optional SaveSource instance for saving generated code

    Returns:
        Compiled module class
    """
    codegen = TTTv2RealMLPCodeGen(model_cfg, topology)
    source_code = codegen.generate_module_class()

    filename = "<string>"
    if save_source is not None:
        header = (
            f"# SPDX-FileCopyrightText: © {datetime.now().year} Tenstorrent AI ULC\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
            "# Auto-generated by TTTv2 CodeGen from codegen_mlp_real.py\n"
        )
        source_code = header + source_code

        filepath = save_source.get_filename(model_cfg, topology)
        if filepath:
            with open(filepath, "w") as f:
                f.write(source_code)
            filename = str(Path(filepath).resolve())

    # Compile the source
    namespace = GENERATED_MODULE_NAMESPACE.copy()
    # Add additional imports needed by templates
    namespace["tt_all_reduce"] = None  # Will be imported at runtime

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
    print("=== Example 1: Single Device MLP (Llama 7B) ===")

    model_cfg = MLPModelConfig(
        dim=4096,
        hidden_dim=11008,
        num_devices=1,
        is_galaxy=False,
    )

    save_source = SaveSource()
    module_class = RealMLP(
        model_cfg,
        topology=Topology.SINGLE_DEVICE,
        save_source=save_source,
    )

    print(f"Generated class: {module_class}")
    filename = save_source.get_filename(model_cfg, Topology.SINGLE_DEVICE)
    if filename:
        print(f"Saved to: {filename}")
        with open(filename, "r") as f:
            lines = f.readlines()[:60]
            print("\nGenerated source preview:")
            print("-" * 60)
            print("".join(lines))
            print("... [truncated]\n")

    print("\n=== Example 2: Galaxy TG MLP (Llama 70B) ===")

    model_cfg_tg = MLPModelConfig(
        dim=8192,
        hidden_dim=28672,
        num_devices=32,
        is_galaxy=True,
    )

    save_source_tg = SaveSource()
    module_class_tg = RealMLP(
        model_cfg_tg,
        topology=Topology.GALAXY_2D,
        save_source=save_source_tg,
    )

    print(f"Generated class: {module_class_tg}")
    filename_tg = save_source_tg.get_filename(model_cfg_tg, Topology.GALAXY_2D)
    if filename_tg:
        print(f"Saved to: {filename_tg}")
        with open(filename_tg, "r") as f:
            lines = f.readlines()[:80]
            print("\nGenerated source preview:")
            print("-" * 60)
            print("".join(lines))
            print("... [truncated]\n")

    print("\n=== Example 3: T3K MLP ===")

    model_cfg_t3k = MLPModelConfig(
        dim=4096,
        hidden_dim=14336,
        num_devices=8,
        is_galaxy=False,
    )

    save_source_t3k = SaveSource()
    module_class_t3k = RealMLP(
        model_cfg_t3k,
        topology=Topology.T3K_1D,
        save_source=save_source_t3k,
    )

    print(f"Generated class: {module_class_t3k}")
    filename_t3k = save_source_t3k.get_filename(model_cfg_t3k, Topology.T3K_1D)
    if filename_t3k:
        print(f"Saved to: {filename_t3k}")
