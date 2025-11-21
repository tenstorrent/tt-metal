"""
TTTv2 MLP Module with Code Generation

This example demonstrates:
1. How a high-level MLP module can generate specialized implementations
2. Template-based code generation for different hardware configurations
3. Compile-time optimization based on tensor shapes and hardware constraints
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import ttnn


@dataclass
class MLPConfig:
    """MLP-specific configuration"""

    hidden_size: int
    intermediate_size: int
    activation: str = "silu"
    dropout: float = 0.0


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


def generate_kernel_config(name: str, kernel_config, indent: str = "") -> list:
    """Generate hardware-specific configuration setup"""
    # Alias for generate_hardware_config_source to match previous rename request
    # Reusing the logic above
    lines = [
        "# Hardware-specific configuration",
        f"{name} = ttnn.WormholeComputeKernelConfig(",
        f"  math_fidelity=ttnn.MathFidelity.HiFi4,",  # Defaulting
        f"  fp32_dest_acc_en=True,",  # Defaulting
        "   packer_l1_acc=True",
        ")",
        "",
    ]
    lines = [f"{indent}{line}" for line in lines]
    return lines


def generate_memory_config_source(name: str, mem_config, indent: str = ""):
    lines = [
        "# Memory configuration",
        f"{name} = ttnn.MemoryConfig(",
        f"  memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,",  # Defaulting
        f"  buffer_type=ttnn.BufferType.L1",
        ")",
        "",
    ]
    lines = [f"{indent}{line}" for line in lines]

    return lines


def generate_optimized_source(func_name: str, args: list, func_def: ast.FunctionDef, indent: str = "") -> str:
    """Generate complete optimized source code"""

    # Extract function body
    # unparse usually re-indents to 4 spaces.
    body_code = ast.unparse(func_def.body)
    # ast.unparse on list of stmts might not exist in older python, but unparse on FunctionDef gives whole function.

    # Let's stick to unparsing the function then splitting, but handle indentation more robustly.
    full_code = ast.unparse(func_def)
    body_lines = full_code.split("\n")[1:]  # Skip def line

    # The body_lines from ast.unparse will have 4-space indentation relative to def.
    # We want to put them inside our generated function which is also indented.

    local_indent = "    "  # 4 spaces
    source_lines = [
        f'def {func_name}({", ".join(args)}):',
        f'{local_indent}"""Generated from introspected function"""',
    ]

    for line in body_lines:
        # line already has 4 spaces from unparse if it was indented in original function?
        # Yes, ast.unparse formats code with standard indentation.
        # So line is like "    x = 1"
        # We want to preserve that relative indentation.
        source_lines.append(line)

    source_lines = [f"{indent}{line}" for line in source_lines]
    return source_lines


def function_to_source(func: Callable, class_name: str = "GeneratedClass", indent: str = "") -> str:
    """
    Convert a function to optimized source code by:
    1. Extracting the original source
    2. Analyzing for optimization opportunities
    3. Generating enhanced source with context
    """

    # Get original source
    original_source = inspect.getsource(func)
    # Dedent to ensure clean parsing if function was nested or indented
    original_source = textwrap.dedent(original_source)

    # Parse AST for analysis
    tree = ast.parse(original_source)

    # Extract function details
    func_def = tree.body[0]
    func_name = func_def.name
    args = [arg.arg for arg in func_def.args.args]

    # Analyze function body for TTNN operations
    # ttnn_ops = find_ttnn_operations(func_def)

    # Generate optimized source with full context
    source_lines = generate_optimized_source(func_name, args, func_def, indent)

    return source_lines


# Template function for MLP forward pass
def forward_mlp_impl(x, weights, ops):
    """
    Pure functional implementation of MLP forward pass.

    Args:
        x: Input tensor
        weights: Container/object with weight tensors (gate_proj, up_proj, down_proj)
        ops: Configuration object containing operation configs (gate, up, down, activation)
    """
    # MLP Forward Pass
    # 1. Gate and Up Projections
    # todo)) how to handle fused alternatives
    gate = ttnn.linear(
        x,
        weights.gate_proj_weight,
        bias=weights.gate_proj_bias,
        compute_kernel_config=ops.gate.compute_kernel_config,
        memory_config=ops.gate.memory_config,
        dtype=ops.gate.dtype,
        program_config=ops.gate.program_config
        # todo)) for prefetcher
        # sub_device_id=ops.gate.sub_device_id
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

    # 2. Activation
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


class TTTv2MLPCodeGen:
    """
    Generates optimized MLP implementations based on configuration.
    This is the core of the code generation approach.
    """

    def __init__(self, mlp_config: MLPConfig, device=None, please_gen_prefetcher=False, prefetcher_core_config=None):
        self.mlp_config = mlp_config
        self.device = device
        self._validate_config()

    def _validate_config(self):
        """Validate that configuration is feasible"""

    def generate_forward_function(self) -> str:
        """
        Generate specialized forward function based on configuration.
        """
        return "return forward_mlp_impl(x, self.weights, self.ops)"

    def generate_ops_init_source(self) -> list:
        """Generate source code for _init_ops_config method"""
        lines = []

        # Configure memory and compute based on hardware
        lines.extend(self._generate_hardware_config())

        # Determine activation function string
        if self.mlp_config.activation == "silu":
            activation = "ttnn.silu"
        elif self.mlp_config.activation == "gelu":
            activation = "ttnn.gelu"
        elif self.mlp_config.activation == "relu":
            activation = "ttnn.relu"
        else:
            activation = f"ttnn.{self.mlp_config.activation}"

        # Generate OpConfigs for each operation
        lines.extend(
            [
                "# Op Configs",
                "gate_config = OpConfig(",
                "    compute_kernel_config=compute_kernel_config,",
                "    memory_config=memory_config,",
                "    dtype=ttnn.bfloat16",
                ")",
                "up_config = OpConfig(",
                "    compute_kernel_config=compute_kernel_config,",
                "    memory_config=memory_config,",
                "    dtype=ttnn.bfloat16",
                ")",
                "down_config = OpConfig(",
                "    compute_kernel_config=compute_kernel_config,",
                "    memory_config=memory_config,",
                "    dtype=ttnn.bfloat16",
                ")",
                f"self.ops = MLPOpConfigs(gate=gate_config, up=up_config, down=down_config, activation={activation})",
                "",
            ]
        )

        return lines

    def _generate_hardware_config(self) -> list:
        """Generate hardware-specific configuration setup"""
        # Just passing None for now as we hardcoded defaults in the function
        lines = generate_kernel_config("compute_kernel_config", None)
        lines.extend(generate_memory_config_source("memory_config", None))

        return lines

    def generate_module_class(self) -> str:
        """Generate complete module class with initialization and forward"""

        device_name = "wormhole_b0"  # Hardcoded default for now as we don't have hw_config

        # Build complete class
        class_lines = [
            "from dataclasses import dataclass",
            "from typing import Optional, Any, Callable",
            "",
            "class LightweightModule:",
            '    """LightweightModule to replace nn.Module and remove torch dependency"""',
            "    def __call__(self, *args, **kwargs):",
            "        return self.forward(*args, **kwargs)",
            "",
            "@dataclass",
            "class OpConfig:",
            "    compute_kernel_config: Optional[Any] = None",
            "    memory_config: Optional[Any] = None",
            "    dtype: Optional[Any] = None",
            "    program_config: Optional[Any] = None",
            "",
            "@dataclass",
            "class MLPOpConfigs:",
            "    gate: OpConfig",
            "    up: OpConfig",
            "    down: OpConfig",
            "    activation: Callable",
            "",
            "@dataclass",
            "class MLPWeights:",
            "    gate_proj_weight: Any",
            "    gate_proj_bias: Any",
            "    up_proj_weight: Any",
            "    up_proj_bias: Any",
            "    down_proj_weight: Any",
            "    down_proj_bias: Any",
            "",
        ]

        # Add the implementation function source code as a free function (not in class)
        impl_source = function_to_source(forward_mlp_impl)
        for line in impl_source:
            class_lines.append(line)
        class_lines.append("")

        # Continue with class definition
        class_lines.extend(
            [
                f"class TTTv2MLP_{device_name}(LightweightModule):",
                f'    """',
                f"    Auto-generated MLP module for {device_name}",
                f"    Configuration:",
                f"      - Hidden size: {self.mlp_config.hidden_size}",
                f"      - Intermediate size: {self.mlp_config.intermediate_size}",
                f"      - Activation: {self.mlp_config.activation}",
                f'    """',
                f"",
                f"    def __init__(self, device):",
                f"        super().__init__()",
                f"        self.device = device",
                f"        self.hidden_size = {self.mlp_config.hidden_size}",
                f"        self.intermediate_size = {self.mlp_config.intermediate_size}",
                f"        self.activation = '{self.mlp_config.activation}'",
                f"        ",
                f"        # Initialize weights",
                f"        self._init_weights()",
                f"        self._init_ops_config()",
                # todo)) { for prefetcher
                # f"        self.prefetcher = TtLlamaPrefetcherSetup(device, 1, 1, mode='decode')", if self.please_gen_prefetcher else ""
                # f"        self.prefetcher.insert_tensor(self.weights.gate_proj_weight)",
                # f"        self.prefetcher.insert_tensor(self.weights.up_proj_weight)",
                # f"        self.prefetcher.insert_tensor(self.weights.down_proj_weight)",
                # }
                f"",
                f"    def _init_weights(self):",
                f'        """Initialize projection weights"""',
                f"        import ttnn",
                f"",
                f"        # Gate Projection",
                f"        gate_proj_weight = ttnn.create_weight(",
                f"            shape=[{self.mlp_config.hidden_size}, {self.mlp_config.intermediate_size}],",
                f"            dtype=ttnn.bfloat16,",
                f"            device=self.device",
                f"        )",
                f"        gate_proj_bias = ttnn.create_bias(",
                f"            shape=[{self.mlp_config.intermediate_size}],",
                f"            dtype=ttnn.bfloat16,",
                f"            device=self.device",
                f"        )",
                f"",
                f"        # Up Projection",
                f"        up_proj_weight = ttnn.create_weight(",
                f"            shape=[{self.mlp_config.hidden_size}, {self.mlp_config.intermediate_size}],",
                f"            dtype=ttnn.bfloat16,",
                f"            device=self.device",
                f"        )",
                f"        up_proj_bias = ttnn.create_bias(",
                f"            shape=[{self.mlp_config.intermediate_size}],",
                f"            dtype=ttnn.bfloat16,",
                f"            device=self.device",
                f"        )",
                f"",
                f"        # Down Projection",
                f"        down_proj_weight = ttnn.create_weight(",
                f"            shape=[{self.mlp_config.intermediate_size}, {self.mlp_config.hidden_size}],",
                f"            dtype=ttnn.bfloat16,",
                f"            device=self.device",
                f"        )",
                f"        down_proj_bias = ttnn.create_bias(",
                f"            shape=[{self.mlp_config.hidden_size}],",
                f"            dtype=ttnn.bfloat16,",
                f"            device=self.device",
                f"        )",
                f"",
                f"        self.weights = MLPWeights(",
                f"            gate_proj_weight=gate_proj_weight, gate_proj_bias=gate_proj_bias,",
                f"            up_proj_weight=up_proj_weight, up_proj_bias=up_proj_bias,",
                f"            down_proj_weight=down_proj_weight, down_proj_bias=down_proj_bias",
                f"        )",
                f"",
                f"        self.training = False",
                f"",
            ]
        )

        # Add _init_ops_config method
        class_lines.append("    def _init_ops_config(self):")
        ops_config_lines = self.generate_ops_init_source()
        for line in ops_config_lines:
            if line.strip():
                class_lines.append(f"        {line}")
            else:
                class_lines.append("")
        class_lines.append("")

        # Add the forward function
        forward_src = self.generate_forward_function()
        forward_lines = forward_src.split("\n")

        # Add definition of forward
        class_lines.append("    def forward(self, x):")

        for line in forward_lines:
            if line.strip():
                class_lines.append(f"        {line}")
            else:
                class_lines.append("")

        return "\n".join(class_lines)


def MLP(
    mlp_config: MLPConfig,
    *,
    # Removed hw_config
    gen_format: str = "class",
    save_source: bool = False,
    filename: str = "",
) -> Tuple[type, str]:
    """
    Main API to compile an MLP module for specific hardware and configuration.

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
        body = codegen.generate_forward_function()
        source_code = function_to_source(forward_mlp_impl)
        # This is not quite right for "function" mode as it lacks context, but sticking to "class" mode primarily.
        # The previous logic for function mode was a bit hacked too.
        # Let's just return the implementation function source.

    else:
        raise ValueError(f"Invalid generation format: {gen_format}")

    # Save source if requested
    if save_source:
        with open(filename, "w") as f:
            f.write("# Auto-generated by TTTv2 CodeGen\n")
            f.write("import ttnn\n\n")
            f.write(source_code)

    # Compile the source into a class
    namespace = {"ttnn": ttnn, "Optional": Optional, "Any": Any, "Callable": Callable}
    # Execute with single namespace to avoid scope issues with type resolution
    exec(source_code, namespace)

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

    module_class = MLP(
        mlp_config,
        gen_format="class",
        save_source=True,
        filename=f"mlp_wormhole_b0_{mlp_config.hidden_size}.py",
    )

    print(f"Generated class: {module_class}")
    if module_class:
        print("\nGenerated source preview:")
        print("-" * 60)

        with open(f"mlp_wormhole_b0_{mlp_config.hidden_size}.py", "r") as f:
            print("\n".join(f.readlines()[:50]))
        print("... [truncated]\n")
