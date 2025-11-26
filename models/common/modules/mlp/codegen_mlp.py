"""
TTTv2 MLP Module with Code Generation

This example demonstrates:
1. How a high-level MLP module can generate specialized implementations
2. Template-based code generation for different hardware configurations
3. Compile-time optimization based on tensor shapes and hardware constraints

=== APPROACH 1: "Codegen Selects, Doesn't Compose" ===

This file demonstrates the "multiple templates" approach where:
- Each topology/mode combination gets its own complete template function
- Codegen's job is to SELECT the right template based on configuration
- Templates are readable, standalone implementations (no micro-composition)

Trade-offs:
+ Each template is self-contained and readable
+ Easy to understand what a specific config produces
+ No abstraction overhead or composition complexity
- Code duplication between templates (e.g., down projection is similar across templates)
- Adding a new variant means copying & modifying an entire template
- For large modules (attention), this could mean 300+ line templates with lots of duplication

See codegen_mlp_2.py for the alternative "Codegen Composes" approach.
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

# =============================================================================
# Single Source of Truth for Generated Module Imports
# =============================================================================
# This dict of REAL Python symbols is the single source of truth.
# Benefits:
#   1. IDEs and language servers understand these symbols (autocomplete, hover, etc.)
#   2. We derive import statements programmatically from the symbols
#   3. No string duplication or manual synchronization

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

    Groups imports by module for cleaner output.
    """
    import types
    from collections import defaultdict

    # Group symbols by their source module
    module_imports = []  # "import X" statements
    from_imports = defaultdict(list)  # "from X import a, b, c" grouped by X

    for name, obj in module_namespace.items():
        if isinstance(obj, types.ModuleType):
            module_imports.append(f"import {obj.__name__}")
        else:
            module = getattr(obj, "__module__", None)
            if module:
                from_imports[module].append(name)

    # Build the output
    lines = []

    # Module imports first
    for stmt in sorted(module_imports):
        lines.append(stmt)

    # Then from imports, grouped and sorted
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


@dataclass
class MLPConfig:
    """MLP-specific configuration"""

    hidden_size: int
    intermediate_size: int
    activation: str = "silu"
    dropout: float = 0.0
    # New: topology selection for template routing
    topology: Topology = Topology.SINGLE_DEVICE


@dataclass
class OpConfig:
    """Configuration for a TTNN operation"""

    compute_kernel_config: Optional[Any] = None
    memory_config: Optional[Any] = None
    dtype: Optional[Any] = None
    program_config: Optional[Any] = None


@dataclass
class MLPOpConfigs:
    """Container for MLP operation configurations"""

    gate: OpConfig
    up: OpConfig
    down: OpConfig
    activation: Callable


@dataclass
class CCLOpConfigs:
    """Container for CCL (Collective Communication Library) operation configurations.

    Used by fused multi-device templates like forward_mlp_fused_ccl_impl.
    """

    # Fused double matmul + reduce scatter config
    fused_gate_up: OpConfig
    # Reduce scatter for up projection result
    reduce_scatter: OpConfig
    # Down projection config
    down: OpConfig
    # CCL-specific settings
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


_ = None  # Stop inspect.getsource from grabbing following comments

# =============================================================================
# TEMPLATE 1: Canonical MLP (Single Device / Simple Topology)
# =============================================================================
# This is the standard SwiGLU-style MLP: gate, up, activation, mul, down


# Template function for MLP forward pass
# [INFO] We want to avoid if-else in the template function:
# def forward_mlp_impl(x, weights, ops, mode, topology):
# if topology == "galaxy" and mode == "decode":
#     return _forward_mlp_galaxy_decode(...)
# elif topology == "galaxy" and mode == "prefill":
#     return _forward_mlp_galaxy_prefill(...)
# else:
#     return _forward_mlp_canonical(...)
def forward_mlp_impl(x, weights, ops):
    """
    Canonical MLP forward pass for single-device or simple topologies.

    Dataflow: x -> [gate, up] -> activation(gate) * up -> down -> output

    Args:
        x: Input tensor
        weights: Container with weight tensors (gate_proj, up_proj, down_proj)
        ops: Configuration object containing operation configs
    """
    # 1. Gate and Up Projections (parallel)
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

    # 2. Activation on gate
    gate = ops.activation(gate)

    # 3. Element-wise multiplication
    intermediate = ttnn.mul(gate, up)

    # 4. Down Projection
    output = ttnn.linear(
        intermediate,
        weights.down_proj_weight,
        bias=weights.down_proj_bias,
        compute_kernel_config=ops.down.compute_kernel_config,
        memory_config=ops.down.memory_config,
        dtype=ops.down.dtype,
        program_config=ops.down.program_config,
    )

    return output


_ = None  # Stop inspect.getsource from grabbing following comments

# =============================================================================
# TEMPLATE 2: Fused CCL MLP (Galaxy 2D Mesh Topology)
# =============================================================================
# This template uses fused kernels for multi-device execution:
# - double_matmul_line_reduce_scatter: fuses gate+up matmuls with reduce_scatter
# - Designed for Galaxy's 2D mesh topology (8x4 devices)
#
# Note: This is a SEPARATE template, not a parameterized version of the canonical one.
# The dataflow is structurally different and cannot be expressed as config changes.


def forward_mlp_fused_ccl_impl(x, weights, ops, tt_ccl):
    """
    Fused CCL MLP forward pass for Galaxy 2D mesh topology.

    Uses fused kernels that combine matmul + collective operations for
    better performance on multi-device configurations.

    Dataflow:
        x -> double_matmul_line_reduce_scatter(gate, up)
          -> reduce_scatter(up_result)
          -> mul(gate_reduced, up_reduced) with fused activation
          -> all_gather -> down -> reduce -> output

    Args:
        x: Input tensor (sharded across devices)
        weights: Container with weight tensors
        ops: CCLOpConfigs with fused operation configurations
        tt_ccl: TT_CCL instance for collective operations
    """
    # 1. Fused gate + up projections with reduce_scatter
    # This single call replaces: gate_matmul, up_matmul, reduce_scatter(gate)
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
    ttnn.deallocate(x)

    # 2. Reduce scatter for up projection result
    up_reduced = tt_ccl.line_reduce_scatter(
        up_out,
        cluster_axis=ops.cluster_axis,
        num_links=ops.num_links,
        memory_config=ops.reduce_scatter.memory_config,
    )
    ttnn.deallocate(up_out)

    # 3. Element-wise multiplication with fused activation
    # Note: activation is fused into the mul operation for efficiency
    intermediate = ttnn.mul(
        gate_reduced,
        up_reduced,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ops.fused_gate_up.dtype,
        memory_config=ops.reduce_scatter.memory_config,
    )
    ttnn.deallocate(gate_reduced)
    ttnn.deallocate(up_reduced)

    # 4. All-gather to reconstruct full intermediate
    intermediate_gathered = tt_ccl.line_all_gather(
        intermediate,
        dim=3,
        cluster_axis=ops.cluster_axis,
        num_links=ops.num_links,
        memory_config=ops.down.memory_config,
    )
    ttnn.deallocate(intermediate)

    # 5. Down projection
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

    # 6. All-reduce to combine results across devices
    output = tt_ccl.line_all_reduce(
        output,
        cluster_axis=0,  # Reduce across the other axis
        num_links=ops.num_links,
        memory_config=ops.down.memory_config,
    )

    return output


_ = None  # Stop inspect.getsource from grabbing following comments

# =============================================================================
# Template Registry: Maps (topology, mode) -> template function
# =============================================================================
# This is the "selection" mechanism - codegen looks up the right template.

TEMPLATE_REGISTRY = {
    # Single device: use canonical template
    (Topology.SINGLE_DEVICE, "decode"): forward_mlp_impl,
    (Topology.SINGLE_DEVICE, "prefill"): forward_mlp_impl,
    # T3K 1D ring: use canonical (could add specialized later)
    (Topology.T3K_1D, "decode"): forward_mlp_impl,
    (Topology.T3K_1D, "prefill"): forward_mlp_impl,
    # Galaxy 2D mesh: use fused CCL template
    (Topology.GALAXY_2D, "decode"): forward_mlp_fused_ccl_impl,
    (Topology.GALAXY_2D, "prefill"): forward_mlp_impl,  # Prefill might use different strategy
}


def get_template_for_config(topology: Topology, mode: str = "decode"):
    """
    Select the appropriate template function based on topology and mode.

    This is the core of the "codegen selects" approach - the template
    registry maps configurations to complete, standalone implementations.
    """
    key = (topology, mode)
    if key not in TEMPLATE_REGISTRY:
        raise ValueError(f"No template registered for {key}. Available: {list(TEMPLATE_REGISTRY.keys())}")
    return TEMPLATE_REGISTRY[key]


def init_ops_config_impl():
    """Config initializer for canonical MLP template."""
    # Hardware-specific configuration
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    # Memory configuration
    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.DRAM
    )

    # Op Configs
    gate_config = OpConfig(
        compute_kernel_config=compute_kernel_config, memory_config=memory_config, dtype=ttnn.bfloat16
    )
    up_config = OpConfig(compute_kernel_config=compute_kernel_config, memory_config=memory_config, dtype=ttnn.bfloat16)
    down_config = OpConfig(
        compute_kernel_config=compute_kernel_config, memory_config=memory_config, dtype=ttnn.bfloat16
    )
    return MLPOpConfigs(gate=gate_config, up=up_config, down=down_config, activation=ttnn.silu)


def init_ccl_ops_config_impl():
    """Config initializer for fused CCL MLP template (Galaxy 2D)."""
    # High-fidelity compute for accuracy
    compute_kernel_config_hifi = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    # Sharded memory config for multi-device
    sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1
    )

    # Reduce scatter output config
    reduce_scatter_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1
    )

    # Fused gate+up config
    fused_gate_up_config = OpConfig(
        compute_kernel_config=compute_kernel_config_hifi,
        memory_config=sharded_memory_config,
        dtype=ttnn.bfloat8_b,  # Lower precision for intermediate
    )

    # Reduce scatter config
    rs_config = OpConfig(
        memory_config=reduce_scatter_config,
        dtype=ttnn.bfloat8_b,
    )

    # Down projection config
    down_config = OpConfig(
        compute_kernel_config=compute_kernel_config_hifi,
        memory_config=sharded_memory_config,
        dtype=ttnn.bfloat8_b,
    )

    return CCLOpConfigs(
        fused_gate_up=fused_gate_up_config,
        reduce_scatter=rs_config,
        down=down_config,
        cluster_axis=1,
        num_links=1,
        activation=ttnn.silu,
    )


class MLPModuleTemplate(LightweightModule):
    """
    Template class for MLP module (canonical single-device).
    This class is used to generate the final module.
    """

    def __init__(self, device, weights: MLPWeights):
        super().__init__()
        self.device = device
        # Placeholders - will be replaced by generator
        self.hidden_size = 0
        self.intermediate_size = 0
        self.activation = "silu"

        # Initialize weights
        self.weights = weights
        for key in vars(self.weights):
            attr = getattr(self.weights, key)
            if hasattr(attr, "get_weight"):
                setattr(self.weights, key, attr.get_weight())
        self._init_ops_config()

    def _init_ops_config(self):
        pass

    def forward(self, x):
        return forward_mlp_impl(x, self.weights, self.ops)


class MLPModuleCCLTemplate(LightweightModule):
    """
    Template class for MLP module (Galaxy 2D with CCL).
    Uses fused CCL operations for multi-device execution.
    """

    def __init__(self, device, weights: MLPWeights, tt_ccl):
        super().__init__()
        self.device = device
        self.tt_ccl = tt_ccl
        # Placeholders - will be replaced by generator
        self.hidden_size = 0
        self.intermediate_size = 0
        self.activation = "silu"

        # Initialize weights
        self.weights = weights
        for key in vars(self.weights):
            attr = getattr(self.weights, key)
            if hasattr(attr, "get_weight"):
                setattr(self.weights, key, attr.get_weight())
        self._init_ops_config()

    def _init_ops_config(self):
        pass

    def forward(self, x):
        return forward_mlp_fused_ccl_impl(x, self.weights, self.ops, self.tt_ccl)


class TTTv2MLPCodeGen:
    """
    Generates optimized MLP implementations based on configuration.

    APPROACH 1: "Codegen Selects"

    This generator:
    1. Looks up the right template from TEMPLATE_REGISTRY based on (topology, mode)
    2. Extracts the template source code
    3. Generates the appropriate config initialization
    4. Assembles into a complete module class

    The key insight: templates are COMPLETE implementations, not composable pieces.
    Adding a new variant means adding a new template to the registry.
    """

    def __init__(
        self,
        mlp_config: MLPConfig,
        device=None,
        mode: str = "decode",
        please_gen_prefetcher=False,
        prefetcher_core_config=None,
    ):
        self.mlp_config = mlp_config
        self.device = device
        self.mode = mode
        self._validate_config()

        # Select the template based on configuration
        self.selected_template = get_template_for_config(mlp_config.topology, mode)
        self.template_name = self.selected_template.__name__

    def _validate_config(self):
        """Validate that configuration is feasible"""
        # Validate topology-specific constraints
        if self.mlp_config.topology == Topology.GALAXY_2D:
            # Galaxy requires specific hidden size alignment for 2D sharding
            if self.mlp_config.hidden_size % 32 != 0:
                raise ValueError(
                    f"Galaxy topology requires hidden_size divisible by 32, got {self.mlp_config.hidden_size}"
                )

    def generate_ops_init_source(self) -> list:
        """Generate source code for _init_ops_config method.

        Selects the appropriate config initializer based on the selected template.
        - Canonical templates use MLPOpConfigs
        - CCL templates use CCLOpConfigs
        """
        # Choose the right init function based on template
        if self.template_name == "forward_mlp_fused_ccl_impl":
            source_lines = function_to_source(init_ccl_ops_config_impl)
            config_class = "CCLOpConfigs"
        else:
            source_lines = function_to_source(init_ops_config_impl)
            config_class = "MLPOpConfigs"

        # Process the body lines
        processed_lines = []
        for line in source_lines:
            current_line = line

            # Inject correct activation
            if "activation=ttnn.silu" in current_line:
                if self.mlp_config.activation == "silu":
                    activation = "ttnn.silu"
                elif self.mlp_config.activation == "gelu":
                    activation = "ttnn.gelu"
                elif self.mlp_config.activation == "relu":
                    activation = "ttnn.relu"
                else:
                    activation = f"ttnn.{self.mlp_config.activation}"
                current_line = current_line.replace("activation=ttnn.silu", f"activation={activation}")

            # Change return to assignment
            if f"return {config_class}" in current_line:
                current_line = current_line.replace(f"return {config_class}", f"self.ops = {config_class}")

            processed_lines.append(current_line)

        return processed_lines[1:]  # Skip def line

    def generate_module_class(self) -> str:
        """Generate complete module class with initialization and forward.

        APPROACH 1: Select the right template based on topology/mode.
        The generated class uses the selected template's forward function.
        """
        # Determine device/topology name for class naming
        topology_name = self.mlp_config.topology.value.replace("_", "")
        device_name = f"wormhole_{topology_name}"

        # Build complete class
        class_lines = []

        # Add imports from single source of truth
        class_lines.append(get_import_source(GENERATED_MODULE_NAMESPACE))
        class_lines.append("")

        # Add config classes based on selected template
        class_lines.extend(class_to_source(OpConfig))
        class_lines.append("")

        if self.template_name == "forward_mlp_fused_ccl_impl":
            class_lines.extend(class_to_source(CCLOpConfigs))
        else:
            class_lines.extend(class_to_source(MLPOpConfigs))
        class_lines.append("")

        class_lines.extend(class_to_source(MLPWeights))
        class_lines.append("")

        # Add the SELECTED template function source code
        # This is the key: we emit the specific template chosen by the registry
        impl_source = function_to_source(self.selected_template)
        class_lines.append(
            f"# Template: {self.template_name} (selected for {self.mlp_config.topology.value}, {self.mode})"
        )
        for line in impl_source:
            class_lines.append(line)
        class_lines.append("")

        # Generate the main module class from the appropriate template
        if self.template_name == "forward_mlp_fused_ccl_impl":
            template_source = class_to_source(MLPModuleCCLTemplate)
            template_class_name = "MLPModuleCCLTemplate"
        else:
            template_source = class_to_source(MLPModuleTemplate)
            template_class_name = "MLPModuleTemplate"

        # Remove the original docstring from template if present
        # We assume docstring is at the beginning of the body
        # A simple heuristic: if lines 1 start with """, skip until end of docstring
        body_start_idx = 1
        if len(template_source) > 1 and template_source[1].strip().startswith('"""'):
            # Find end of docstring
            for i in range(1, len(template_source)):
                if template_source[i].strip().endswith('"""') and (i > 1 or len(template_source[i].strip()) > 3):
                    body_start_idx = i + 1
                    break
                elif template_source[i].strip() == '"""' and i > 1:
                    body_start_idx = i + 1
                    break

        # Construct final lines
        new_class_name = f"TTTv2MLP_{device_name}"
        final_lines = [template_source[0]]  # Class definition

        # Add new docstring
        config_doc = [
            f'    """',
            f"    Auto-generated MLP module for {device_name}",
            f"    Configuration:",
            f"      - Hidden size: {self.mlp_config.hidden_size}",
            f"      - Intermediate size: {self.mlp_config.intermediate_size}",
            f"      - Activation: {self.mlp_config.activation}",
            f'    """',
        ]
        final_lines.extend(config_doc)

        # Process body lines
        skip = False
        for line in template_source[body_start_idx:]:
            if "self.hidden_size = 0" in line:
                final_lines.append(line.replace("0", str(self.mlp_config.hidden_size)))
            elif "self.intermediate_size = 0" in line:
                final_lines.append(line.replace("0", str(self.mlp_config.intermediate_size)))
            elif 'self.activation = "silu"' in line:
                final_lines.append(line.replace('"silu"', f"'{self.mlp_config.activation}'"))
            elif "def _init_ops_config(self):" in line:
                # final_lines.append(line) # Don't append def line again as function_to_source does not include it?
                # wait function_to_source DOES include def.
                # generate_ops_init_source returns body lines now.

                final_lines.append(line)  # Append the def line from template
                ops_init_lines = self.generate_ops_init_source()
                # Indent them. function_to_source body lines have 4 space indent.
                # We need them inside the method, so +4 spaces?
                # function_to_source returns:
                # def ...:
                #     """..."""
                #     body...

                # generate_ops_init_source returns:
                #     """..."""
                #     body...

                # We want:
                #     def _init_ops_config(self):
                #         """..."""
                #         body...

                # So we need to indent whatever generate_ops_init_source returns by 4 spaces (since it's already indented for function level, but we are inside class)
                # But wait, function_to_source returns lines that are already indented 4 spaces relative to def?
                # Yes.

                for l in ops_init_lines:
                    # It already has 4 spaces indent from function_to_source logic
                    # We are putting it inside a class, so we need 4 spaces more.
                    if l.strip():
                        # l is like "    x = 1"
                        final_lines.append(f"    {l}")
                    else:
                        final_lines.append("")
                final_lines.append("")  # Add blank line after _init_ops_config
                skip = True  # Skip the 'pass' or body of template _init_ops_config
            elif skip:
                # Assuming _init_ops_config is the last method or we look for next dedent/def
                if line.strip().startswith("def ") or (line.strip() != "" and not line.startswith("        ")):
                    skip = False
                    if line.strip():
                        final_lines.append(line)
                else:
                    pass  # Skip body
            else:
                final_lines.append(line)

        # Replace class name in the definition line
        final_lines[0] = final_lines[0].replace(template_class_name, new_class_name)

        class_lines.extend(final_lines)

        return "\n".join(class_lines)


class SaveSource:
    """Handles saving generated source code to files."""

    def __init__(self, filename: Optional[str] = None):
        """
        Initialize save source handler.

        Args:
            filename: Optional filename. If None, defaults to a generated subdirectory
                     relative to codegen_mlp.py. Can be a relative or absolute path.
                     If None, defaults to generated/mlp_wormhole_b0_{hidden_size}.py
        """
        self._filename = filename
        self._default_dir = None

    def _get_default_dir(self) -> Path:
        """Get the default generated directory."""
        if self._default_dir is None:
            # Get directory of codegen_mlp.py
            codegen_file = Path(__file__).resolve()
            codegen_dir = codegen_file.parent
            self._default_dir = codegen_dir / "generated"
            # Ensure directory exists
            self._default_dir.mkdir(exist_ok=True)
        return self._default_dir

    def get_filename(self, mlp_config: Optional[MLPConfig] = None) -> Optional[str]:
        """
        Get the filename, computing default if needed.

        Args:
            mlp_config: Optional MLP config to use for default filename generation.

        Returns:
            Filename path or None if not saving.
        """
        if self._filename:
            # If filename is provided, resolve it relative to default dir if it's relative
            filepath = Path(self._filename)
            if not filepath.is_absolute():
                return str(self._get_default_dir() / filepath)
            return str(filepath)

        # Generate default filename based on config
        if mlp_config:
            default_name = f"mlp_wormhole_b0_{mlp_config.hidden_size}.py"
        else:
            default_name = "mlp_wormhole_b0_4096.py"

        return str(self._get_default_dir() / default_name)

    def save(self, source_code: str, mlp_config: Optional[MLPConfig] = None) -> None:
        """
        Save source code to file.

        Args:
            source_code: The generated source code to save.
            mlp_config: Optional MLP config for filename generation.
        """
        filepath = self.get_filename(mlp_config)
        if not filepath:
            return

        with open(filepath, "w") as f:
            years = datetime.now().year
            f.write(f"# SPDX-FileCopyrightText: © {years} Tenstorrent AI ULC\n")
            f.write("# SPDX-License-Identifier: Apache-2.0\n")
            f.write("# Auto-generated by TTTv2 CodeGen\n")
            f.write(source_code)


def MLP(
    mlp_config: MLPConfig,
    *,
    gen_format: str = "class",
    save_source: Optional[SaveSource] = None,
) -> Tuple[type, str]:
    """
    Main API to compile an MLP module for specific hardware and configuration.

    Args:
        mlp_config: MLP configuration.
        gen_format: Generation format ("class" or "function").
        save_source: Optional SaveSource instance. If provided, saves generated code to file.
                    If None (default), no file is saved.

    Returns:
        - Compiled module class or pure function
        - Generated source code
    """

    # Create code generator
    codegen = TTTv2MLPCodeGen(mlp_config)

    # Generate source code
    if gen_format == "class":
        source_code = codegen.generate_module_class()
    elif gen_format == "function":
        # For function format, we return the template implementation
        # But simpler to just rely on class format for now or adapt similarly
        # body = codegen.generate_forward_function()
        # This is not quite right for "function" mode as it lacks context, but sticking to "class" mode primarily.
        # The previous logic for function mode was a bit hacked too.
        # Let's just return the implementation function source.
        raise ValueError(f"TODO: Implement function format")

    else:
        raise ValueError(f"Invalid generation format: {gen_format}")

    # Save source if save_source handler is provided
    filename = "<string>"
    if save_source is not None:
        # Prepend header to source_code so compiled code line numbers match file
        header = (
            f"# SPDX-FileCopyrightText: © {datetime.now().year} Tenstorrent AI ULC\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
            "# Auto-generated by TTTv2 CodeGen\n"
        )
        source_code = header + source_code

        # Write to file (source_code now includes header)
        filepath = save_source.get_filename(mlp_config)
        if filepath:
            with open(filepath, "w") as f:
                f.write(source_code)
            filename = str(Path(filepath).resolve())

    # Compile the source into a class using namespace from single source of truth
    namespace = GENERATED_MODULE_NAMESPACE.copy()
    code_obj = compile(source_code, filename, "exec")
    exec(code_obj, namespace)

    # Find the generated class
    module_class = None
    if gen_format == "class":
        for name, obj in namespace.items():
            if name.startswith("TTTv2MLP_"):
                module_class = obj
                break
    elif gen_format == "function":
        module_class = namespace.get("forward_mlp_impl")

    return module_class


# Example usage and demonstration
if __name__ == "__main__":
    # Example 1: Generate MLP for Wormhole with specific config
    print("=== Example 1: Llama-style MLP for Wormhole ===")

    # Hardware config removed from here

    mlp_config = MLPConfig(
        hidden_size=4096,
        intermediate_size=11008,  # Llama 2 7B size
        activation="silu",
        dropout=0.0,
    )

    save_source = SaveSource()
    module_class = MLP(
        mlp_config,
        save_source=save_source,
    )

    print(f"Generated class: {module_class}")
    if module_class:
        print("\nGenerated source preview:")
        print("-" * 60)

        filename = save_source.get_filename(mlp_config)
        if filename:
            with open(filename, "r") as f:
                print("\n".join(f.readlines()[:50]))
            print("... [truncated]\n")

        # Test instantiation
        try:
            device = ttnn.open_device(device_id=0)
            print("Device opened.")
        except Exception as e:
            print(f"Could not open device (this is expected in CI if no device): {e}")
            exit(0)

        # Create MLPWeights
        # Shapes:
        # gate: (hidden, intermediate) -> torch linear (intermediate, hidden)
        # up: (hidden, intermediate) -> torch linear (intermediate, hidden)
        # down: (intermediate, hidden) -> torch linear (hidden, intermediate)
        import torch

        intermediate_size = mlp_config.intermediate_size
        hidden_size = mlp_config.hidden_size

        cache_dir = Path("/tmp/tttv2_mlp_cache")
        cache_dir.mkdir(exist_ok=True)

        # Create source tensors (or callables for lazy loading)
        def make_weight(shape):
            return torch.randn(shape, dtype=torch.bfloat16)

        def make_bias(shape):
            return torch.randn(shape, dtype=torch.bfloat16).reshape(1, 1, 1, -1)

        # New LazyWeight interface: explicit parameters, cache-friendly
        weights = MLPWeights(
            gate_proj_weight=LazyWeight(
                source=make_weight((hidden_size, intermediate_size)),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                cache_dir=cache_dir,
                weight_name="gate_proj_weight",
            ),
            gate_proj_bias=LazyWeight(
                source=make_bias((intermediate_size,)),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            ),
            up_proj_weight=LazyWeight(
                source=make_weight((hidden_size, intermediate_size)),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                cache_dir=cache_dir,
                weight_name="up_proj_weight",
            ),
            up_proj_bias=LazyWeight(
                source=make_bias((intermediate_size,)),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            ),
            down_proj_weight=LazyWeight(
                source=make_weight((intermediate_size, hidden_size)),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                cache_dir=cache_dir,
                weight_name="down_proj_weight",
            ),
            down_proj_bias=LazyWeight(
                source=make_bias((hidden_size,)),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            ),
        )

        print("Instantiating module with state_dict...")
        # Note: module_class is the generated class
        # Source-level debugging is enabled by compile() with filename above
        model = module_class(device, weights=weights)

        print("Running forward...")
        x = ttnn.from_torch(
            torch.randn(1, 1, 32, hidden_size, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        # Verify lazy loading: cache file should NOT exist before forward if we cleaned it,
        # but wait, we are passing LazyWeight to MLPWeights.
        # forward_mlp_impl accesses .gate_proj_weight property -> calls get_weight() -> loads/creates.
        # So it should work transparently.

        out = model(x)
        print(f"Output shape: {out.shape}")

        print("Checking cache creation...")
        if any(cache_dir.glob("gate_proj_weight_*.tensorbin")):
            print("Cache created successfully!")
        else:
            print("Cache NOT created!")

        ttnn.close_device(device)
        print("Device closed.")
