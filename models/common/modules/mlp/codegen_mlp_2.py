"""
TTTv2 MLP Module with Code Generation - APPROACH 2: Compose

=== APPROACH 2: "Codegen Composes, Doesn't Just Select" ===

This file demonstrates the "micro-pieces" approach where:
- The MLP is broken into composable micro-pieces (gate_up, activation_mul, down)
- Each piece has variants for different topologies
- Codegen COMPOSES the right pieces together based on configuration

Trade-offs:
+ No code duplication - shared pieces are written once
+ Adding a new variant often means adding just ONE new piece
+ For large modules (attention), this scales better
+ Easier to test individual pieces in isolation
- More abstraction overhead
- Harder to see the complete flow at a glance
- Composition logic can get complex

See codegen_mlp.py for the alternative "Codegen Selects" approach.

=== Micro-Pieces for MLP ===

The MLP forward pass is decomposed into:
1. gate_up_projection: Compute gate and up projections
2. activation_mul: Apply activation and element-wise multiply
3. down_projection: Final projection to output dimension

Each piece can have topology-specific variants:
- gate_up_projection_canonical: Two separate ttnn.linear calls
- gate_up_projection_fused_ccl: Uses double_matmul_line_reduce_scatter
- etc.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import ttnn
from models.common.modules.codegen_utils.generation import function_to_source
from models.common.modules.lazy_weight import LazyWeight

# =============================================================================
# Configuration Classes (same as codegen_mlp.py)
# =============================================================================


class Topology(Enum):
    SINGLE_DEVICE = "single_device"
    T3K_1D = "t3k_1d"
    GALAXY_2D = "galaxy_2d"


@dataclass
class MLPConfig:
    hidden_size: int
    intermediate_size: int
    activation: str = "silu"
    dropout: float = 0.0
    topology: Topology = Topology.SINGLE_DEVICE


@dataclass
class OpConfig:
    compute_kernel_config: Optional[Any] = None
    memory_config: Optional[Any] = None
    dtype: Optional[Any] = None
    program_config: Optional[Any] = None


@dataclass
class MLPOpConfigs:
    gate: OpConfig
    up: OpConfig
    down: OpConfig
    activation: Callable


@dataclass
class CCLOpConfigs:
    fused_gate_up: OpConfig
    reduce_scatter: OpConfig
    down: OpConfig
    cluster_axis: int = 1
    num_links: int = 1
    activation: Callable = field(default_factory=lambda: ttnn.silu)


@dataclass
class MLPWeights:
    gate_proj_weight: LazyWeight
    gate_proj_bias: LazyWeight
    up_proj_weight: LazyWeight
    up_proj_bias: LazyWeight
    down_proj_weight: LazyWeight
    down_proj_bias: LazyWeight


# =============================================================================
# MICRO-PIECE 1: Gate + Up Projection
# =============================================================================
# Multiple variants for different topologies


def gate_up_projection_canonical(x, weights, ops):
    """
    Canonical gate+up projection: two separate linear operations.
    Returns: (gate_output, up_output, x_to_deallocate_or_None)
    """
    gate = ttnn.linear(
        x,
        weights.gate_proj_weight,
        bias=weights.gate_proj_bias,
        compute_kernel_config=ops.gate.compute_kernel_config,
        memory_config=ops.gate.memory_config,
        dtype=ops.gate.dtype,
        program_config=ops.gate.program_config,
    )
    up = ttnn.linear(
        x,
        weights.up_proj_weight,
        bias=weights.up_proj_bias,
        compute_kernel_config=ops.up.compute_kernel_config,
        memory_config=ops.up.memory_config,
        dtype=ops.up.dtype,
        program_config=ops.up.program_config,
    )
    return gate, up, None  # No deallocation needed here


def gate_up_projection_fused_ccl(x, weights, ops, tt_ccl):
    """
    Fused gate+up projection with CCL for Galaxy 2D.
    Uses double_matmul_line_reduce_scatter for efficiency.
    Returns: (gate_reduced, up_out, x_to_deallocate)
    """
    gate_reduced, up_out = tt_ccl.double_matmul_line_reduce_scatter(
        x,
        weights.gate_proj_weight,
        weights.up_proj_weight,
        cluster_axis=ops.cluster_axis,
        num_links=ops.num_links,
        RS_memory_config=ops.reduce_scatter.memory_config,
        compute_kernel_config=ops.fused_gate_up.compute_kernel_config,
        dtype=ops.fused_gate_up.dtype,
        program_config=ops.fused_gate_up.program_config,
        memory_config=ops.fused_gate_up.memory_config,
    )
    return gate_reduced, up_out, x  # x should be deallocated


# =============================================================================
# MICRO-PIECE 2: Activation + Multiplication
# =============================================================================


def activation_mul_canonical(gate, up, ops):
    """
    Canonical activation + mul: apply activation to gate, then multiply.
    Returns: (intermediate, gate_to_dealloc, up_to_dealloc)
    """
    gate_activated = ops.activation(gate)
    intermediate = ttnn.mul(gate_activated, up)
    return intermediate, None, None  # In-place activation, no extra dealloc


def activation_mul_fused_ccl(gate_reduced, up_out, ops, tt_ccl):
    """
    Fused activation + mul for Galaxy 2D.
    Includes reduce_scatter for up and fused activation in mul.
    Returns: (intermediate, gate_to_dealloc, up_to_dealloc)
    """
    # First, reduce scatter for up projection
    up_reduced = tt_ccl.line_reduce_scatter(
        up_out,
        cluster_axis=ops.cluster_axis,
        num_links=ops.num_links,
        memory_config=ops.reduce_scatter.memory_config,
    )

    # Fused activation + mul
    # todo)) we could refactor this even further by calling activation_mul_canonical() here
    intermediate = ttnn.mul(
        gate_reduced,
        up_reduced,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ops.fused_gate_up.dtype,
        memory_config=ops.reduce_scatter.memory_config,
    )

    return intermediate, gate_reduced, up_reduced  # Both need deallocation


# =============================================================================
# MICRO-PIECE 3: Down Projection
# =============================================================================


def down_projection_canonical(intermediate, weights, ops):
    """
    Canonical down projection: single linear operation.
    Returns: (output, intermediate_to_dealloc)
    """
    output = ttnn.linear(
        intermediate,
        weights.down_proj_weight,
        bias=weights.down_proj_bias,
        compute_kernel_config=ops.down.compute_kernel_config,
        memory_config=ops.down.memory_config,
        dtype=ops.down.dtype,
        program_config=ops.down.program_config,
    )
    return output, None  # intermediate handled by caller


def down_projection_ccl(intermediate, weights, ops, tt_ccl):
    """
    Down projection with CCL all-gather and all-reduce for Galaxy 2D.
    Returns: (output, intermediate_to_dealloc)
    """
    # All-gather to reconstruct full intermediate
    intermediate_gathered = tt_ccl.line_all_gather(
        intermediate,
        dim=3,
        cluster_axis=ops.cluster_axis,
        num_links=ops.num_links,
        memory_config=ops.down.memory_config,
    )

    # Down projection
    output = ttnn.linear(
        intermediate_gathered,
        weights.down_proj_weight,
        bias=weights.down_proj_bias,
        compute_kernel_config=ops.down.compute_kernel_config,
        memory_config=ops.down.memory_config,
        dtype=ops.down.dtype,
        program_config=ops.down.program_config,
    )
    ttnn.deallocate(intermediate_gathered)

    # All-reduce to combine results
    output = tt_ccl.line_all_reduce(
        output,
        cluster_axis=0,
        num_links=ops.num_links,
        memory_config=ops.down.memory_config,
    )

    return output, intermediate  # intermediate should be deallocated


# =============================================================================
# Piece Registry: Maps (piece_name, topology) -> piece function
# =============================================================================

PIECE_REGISTRY = {
    # Gate+Up projection variants
    ("gate_up", Topology.SINGLE_DEVICE): gate_up_projection_canonical,
    ("gate_up", Topology.T3K_1D): gate_up_projection_canonical,
    ("gate_up", Topology.GALAXY_2D): gate_up_projection_fused_ccl,
    # Activation+Mul variants
    ("activation_mul", Topology.SINGLE_DEVICE): activation_mul_canonical,
    ("activation_mul", Topology.T3K_1D): activation_mul_canonical,
    ("activation_mul", Topology.GALAXY_2D): activation_mul_fused_ccl,
    # Down projection variants
    ("down", Topology.SINGLE_DEVICE): down_projection_canonical,
    ("down", Topology.T3K_1D): down_projection_canonical,
    ("down", Topology.GALAXY_2D): down_projection_ccl,
}


def get_piece(piece_name: str, topology: Topology):
    """Get a micro-piece function for the given topology."""
    key = (piece_name, topology)
    if key not in PIECE_REGISTRY:
        raise ValueError(f"No piece registered for {key}")
    return PIECE_REGISTRY[key]


# =============================================================================
# Composed Forward Functions (Generated by Codegen)
# =============================================================================
# These show what the composed output looks like.
# In practice, codegen would generate these by stitching pieces together.


def forward_mlp_composed_canonical(x, weights, ops):
    """
    Composed MLP forward for single device.
    Generated by composing: gate_up_canonical + activation_mul_canonical + down_canonical
    """
    # Piece 1: Gate + Up projection
    gate, up, dealloc1 = gate_up_projection_canonical(x, weights, ops)
    if dealloc1 is not None:
        ttnn.deallocate(dealloc1)

    # Piece 2: Activation + Mul
    intermediate, dealloc2, dealloc3 = activation_mul_canonical(gate, up, ops)
    if dealloc2 is not None:
        ttnn.deallocate(dealloc2)
    if dealloc3 is not None:
        ttnn.deallocate(dealloc3)

    # Piece 3: Down projection
    output, dealloc4 = down_projection_canonical(intermediate, weights, ops)
    if dealloc4 is not None:
        ttnn.deallocate(dealloc4)

    return output


def forward_mlp_composed_galaxy(x, weights, ops, tt_ccl):
    """
    Composed MLP forward for Galaxy 2D.
    Generated by composing: gate_up_fused_ccl + activation_mul_fused_ccl + down_ccl
    """
    # Piece 1: Gate + Up projection (fused with reduce scatter)
    gate_reduced, up_out, dealloc1 = gate_up_projection_fused_ccl(x, weights, ops, tt_ccl)
    if dealloc1 is not None:
        ttnn.deallocate(dealloc1)

    # Piece 2: Activation + Mul (with reduce scatter for up)
    intermediate, dealloc2, dealloc3 = activation_mul_fused_ccl(gate_reduced, up_out, ops, tt_ccl)
    if dealloc2 is not None:
        ttnn.deallocate(dealloc2)
    if dealloc3 is not None:
        ttnn.deallocate(dealloc3)

    # Piece 3: Down projection (with all-gather and all-reduce)
    output, dealloc4 = down_projection_ccl(intermediate, weights, ops, tt_ccl)
    if dealloc4 is not None:
        ttnn.deallocate(dealloc4)

    return output


# =============================================================================
# Code Generator: Composes Pieces into Complete Implementations
# =============================================================================


class TTTv2MLPCodeGenCompose:
    """
    Generates MLP implementations by COMPOSING micro-pieces.

    APPROACH 2: "Codegen Composes"

    This generator:
    1. Looks up each micro-piece from PIECE_REGISTRY based on topology
    2. Composes them into a complete forward function
    3. Generates appropriate glue code (deallocation, variable passing)
    4. Produces a complete module class

    Key insight: pieces are SMALL and REUSABLE. Adding a new topology
    might only require adding 1-2 new pieces, not a complete template.
    """

    def __init__(self, mlp_config: MLPConfig, mode: str = "decode"):
        self.mlp_config = mlp_config
        self.mode = mode
        self.topology = mlp_config.topology

        # Collect the pieces we need
        self.gate_up_piece = get_piece("gate_up", self.topology)
        self.activation_mul_piece = get_piece("activation_mul", self.topology)
        self.down_piece = get_piece("down", self.topology)

        # Determine if we need CCL (tt_ccl parameter)
        self.needs_ccl = self.topology == Topology.GALAXY_2D

    def generate_forward_function(self) -> str:
        """
        Generate the composed forward function by stitching pieces together.

        This is the core of the "compose" approach - we take the source of
        each piece and combine them, handling the interfaces between pieces.
        """
        lines = []

        # Function signature
        if self.needs_ccl:
            lines.append("def forward_mlp_impl(x, weights, ops, tt_ccl):")
        else:
            lines.append("def forward_mlp_impl(x, weights, ops):")

        lines.append('    """')
        lines.append(f"    Composed MLP forward for {self.topology.value}.")
        lines.append(
            f"    Pieces: {self.gate_up_piece.__name__} + {self.activation_mul_piece.__name__} + {self.down_piece.__name__}"
        )
        lines.append('    """')

        # Piece 1: Gate + Up
        lines.append("    # === Piece 1: Gate + Up Projection ===")
        if self.needs_ccl:
            lines.append("    gate, up, _dealloc1 = gate_up_projection_fused_ccl(x, weights, ops, tt_ccl)")
        else:
            lines.append("    gate, up, _dealloc1 = gate_up_projection_canonical(x, weights, ops)")
        lines.append("    if _dealloc1 is not None:")
        lines.append("        ttnn.deallocate(_dealloc1)")
        lines.append("")

        # Piece 2: Activation + Mul
        lines.append("    # === Piece 2: Activation + Multiplication ===")
        if self.needs_ccl:
            lines.append("    intermediate, _dealloc2, _dealloc3 = activation_mul_fused_ccl(gate, up, ops, tt_ccl)")
        else:
            lines.append("    intermediate, _dealloc2, _dealloc3 = activation_mul_canonical(gate, up, ops)")
        lines.append("    if _dealloc2 is not None:")
        lines.append("        ttnn.deallocate(_dealloc2)")
        lines.append("    if _dealloc3 is not None:")
        lines.append("        ttnn.deallocate(_dealloc3)")
        lines.append("")

        # Piece 3: Down projection
        lines.append("    # === Piece 3: Down Projection ===")
        if self.needs_ccl:
            lines.append("    output, _dealloc4 = down_projection_ccl(intermediate, weights, ops, tt_ccl)")
        else:
            lines.append("    output, _dealloc4 = down_projection_canonical(intermediate, weights, ops)")
        lines.append("    if _dealloc4 is not None:")
        lines.append("        ttnn.deallocate(_dealloc4)")
        lines.append("")

        lines.append("    return output")

        return "\n".join(lines)

    def generate_module_source(self) -> str:
        """Generate complete module source code with composed forward."""
        lines = []

        # Header
        lines.append(f"# Auto-generated MLP module for {self.topology.value}")
        lines.append(f"# Generated using APPROACH 2: Compose")
        lines.append(
            f"# Pieces: {self.gate_up_piece.__name__}, {self.activation_mul_piece.__name__}, {self.down_piece.__name__}"
        )
        lines.append("")

        # Add imports (would include ttnn, piece functions, etc.)
        lines.append("import ttnn")
        lines.append("")

        # Add piece function sources
        lines.append("# === Micro-Pieces ===")
        lines.append("")

        # Include the specific pieces we're using
        for piece_fn in [self.gate_up_piece, self.activation_mul_piece, self.down_piece]:
            piece_source = function_to_source(piece_fn)
            lines.extend(piece_source)
            lines.append("")

        # Add the composed forward function
        lines.append("# === Composed Forward Function ===")
        lines.append("")
        lines.append(self.generate_forward_function())
        lines.append("")

        # Add module class
        lines.append("# === Module Class ===")
        lines.append("")

        class_name = f"TTTv2MLP_{self.topology.value}"
        if self.needs_ccl:
            lines.append(f"class {class_name}(LightweightModule):")
            lines.append(f'    """MLP module for {self.topology.value} with CCL support."""')
            lines.append("")
            lines.append("    def __init__(self, device, weights, tt_ccl):")
            lines.append("        super().__init__()")
            lines.append("        self.device = device")
            lines.append("        self.weights = weights")
            lines.append("        self.tt_ccl = tt_ccl")
            lines.append("        self._init_ops_config()")
            lines.append("")
            lines.append("    def _init_ops_config(self):")
            lines.append("        # TODO: Initialize CCLOpConfigs")
            lines.append("        pass")
            lines.append("")
            lines.append("    def forward(self, x):")
            lines.append("        return forward_mlp_impl(x, self.weights, self.ops, self.tt_ccl)")
        else:
            lines.append(f"class {class_name}(LightweightModule):")
            lines.append(f'    """MLP module for {self.topology.value}."""')
            lines.append("")
            lines.append("    def __init__(self, device, weights):")
            lines.append("        super().__init__()")
            lines.append("        self.device = device")
            lines.append("        self.weights = weights")
            lines.append("        self._init_ops_config()")
            lines.append("")
            lines.append("    def _init_ops_config(self):")
            lines.append("        # TODO: Initialize MLPOpConfigs")
            lines.append("        pass")
            lines.append("")
            lines.append("    def forward(self, x):")
            lines.append("        return forward_mlp_impl(x, self.weights, self.ops)")

        return "\n".join(lines)


# =============================================================================
# Comparison: Approach 1 vs Approach 2
# =============================================================================

"""
=== SUMMARY: When to Use Each Approach ===

APPROACH 1 ("Codegen Selects" - codegen_mlp.py):
- Better when templates are relatively small (< 50 lines)
- Better when variants differ significantly in structure
- Easier to read and understand individual templates
- Lower abstraction overhead
- Use for: Small modules, highly divergent implementations

APPROACH 2 ("Codegen Composes" - this file):
- Better when templates are large (100+ lines)
- Better when variants share significant common code
- Easier to add new variants (just add the differing piece)
- Higher abstraction overhead but less code duplication
- Use for: Large modules like attention, incremental topology additions

=== The Dynamic Balance ===

The "right" approach depends on:
1. Module size: Small → Select, Large → Compose
2. Variant similarity: Similar → Compose, Different → Select
3. Team familiarity: New team → Select (easier to read), Experienced → Compose
4. Evolution rate: Stable → Select, Rapidly changing → Compose

For TTTv2 MLP specifically:
- Current state: ~50 lines per template → Select is fine
- If we add 5+ topologies with variations: Consider switching to Compose
- If we add prefill/decode/chunked variants: Compose becomes more attractive

For TTTv2 Attention (hundreds of lines):
- Compose approach likely better from the start
- QKV projection, attention computation, output projection as pieces
- Each piece can have topology-specific variants
"""


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=== APPROACH 2: Codegen Composes ===\n")

    # Example 1: Generate for single device
    print("--- Single Device ---")
    config_single = MLPConfig(
        hidden_size=4096,
        intermediate_size=11008,
        topology=Topology.SINGLE_DEVICE,
    )
    gen_single = TTTv2MLPCodeGenCompose(config_single)
    print(gen_single.generate_forward_function())
    print("\n")

    # Example 2: Generate for Galaxy 2D
    print("--- Galaxy 2D ---")
    config_galaxy = MLPConfig(
        hidden_size=4096,
        intermediate_size=11008,
        topology=Topology.GALAXY_2D,
    )
    gen_galaxy = TTTv2MLPCodeGenCompose(config_galaxy)
    print(gen_galaxy.generate_forward_function())
    print("\n")

    # Show the piece composition
    print("--- Piece Composition Summary ---")
    print(f"Single Device uses:")
    print(f"  - gate_up: {get_piece('gate_up', Topology.SINGLE_DEVICE).__name__}")
    print(f"  - activation_mul: {get_piece('activation_mul', Topology.SINGLE_DEVICE).__name__}")
    print(f"  - down: {get_piece('down', Topology.SINGLE_DEVICE).__name__}")
    print()
    print(f"Galaxy 2D uses:")
    print(f"  - gate_up: {get_piece('gate_up', Topology.GALAXY_2D).__name__}")
    print(f"  - activation_mul: {get_piece('activation_mul', Topology.GALAXY_2D).__name__}")
    print(f"  - down: {get_piece('down', Topology.GALAXY_2D).__name__}")
