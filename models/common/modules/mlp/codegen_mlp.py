"""
TTTv2 MLP Module with Code Generation

This example demonstrates:
1. How a high-level MLP module can generate specialized implementations
2. Template-based code generation for different hardware configurations
3. Compile-time optimization based on tensor shapes and hardware constraints
"""

import inspect
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import ttnn
from models.common.modules.codegen_utils.generation import class_to_source, function_to_source


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


@dataclass
class MLPWeights:
    gate_proj_weight: Any
    gate_proj_bias: Any
    up_proj_weight: Any
    up_proj_bias: Any
    down_proj_weight: Any
    down_proj_bias: Any


class LightweightModule:
    """LightweightModule to replace nn.Module and remove torch dependency"""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


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


def module_imports_template():
    pass


def init_ops_config_impl():
    # Hardware-specific configuration
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    # Memory configuration
    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1
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


class MLPModuleTemplate(LightweightModule):
    """
    Template class for MLP module.
    This class is used to generate the final module.
    """

    def __init__(self, device):
        super().__init__()
        self.device = device
        # Placeholders - will be replaced by generator
        self.hidden_size = 0
        self.intermediate_size = 0
        self.activation = "silu"

        # Initialize weights
        self._init_weights()
        self._init_ops_config()

    def _init_weights(self):
        """Initialize projection weights"""
        # Gate Projection
        self.gate_proj_weight = ttnn.create_weight(
            shape=[self.hidden_size, self.intermediate_size],
            dtype=ttnn.bfloat16,
            device=self.device,
        )
        self.gate_proj_bias = ttnn.create_bias(shape=[self.intermediate_size], dtype=ttnn.bfloat16, device=self.device)

        # Up Projection
        self.up_proj_weight = ttnn.create_weight(
            shape=[self.hidden_size, self.intermediate_size],
            dtype=ttnn.bfloat16,
            device=self.device,
        )
        self.up_proj_bias = ttnn.create_bias(shape=[self.intermediate_size], dtype=ttnn.bfloat16, device=self.device)

        # Down Projection
        self.down_proj_weight = ttnn.create_weight(
            shape=[self.intermediate_size, self.hidden_size],
            dtype=ttnn.bfloat16,
            device=self.device,
        )
        self.down_proj_bias = ttnn.create_bias(shape=[self.hidden_size], dtype=ttnn.bfloat16, device=self.device)

        self.weights = MLPWeights(
            gate_proj_weight=self.gate_proj_weight,
            gate_proj_bias=self.gate_proj_bias,
            up_proj_weight=self.up_proj_weight,
            up_proj_bias=self.up_proj_bias,
            down_proj_weight=self.down_proj_weight,
            down_proj_bias=self.down_proj_bias,
        )

    def _init_ops_config(self):
        pass

    def forward(self, x):
        return forward_mlp_impl(x, self.weights, self.ops)


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

    def generate_ops_init_source(self) -> list:
        """Generate source code for _init_ops_config method"""
        # Introspect init_ops_config_impl
        source_lines = function_to_source(init_ops_config_impl)

        # We need to process the body lines to inject activation function
        # The source_lines from function_to_source includes "def init_ops_config_impl():" and indentation

        processed_lines = []
        for line in source_lines:
            current_line = line
            if "activation=ttnn.silu" in current_line:
                # Inject correct activation
                if self.mlp_config.activation == "silu":
                    activation = "ttnn.silu"
                elif self.mlp_config.activation == "gelu":
                    activation = "ttnn.gelu"
                elif self.mlp_config.activation == "relu":
                    activation = "ttnn.relu"
                else:
                    activation = f"ttnn.{self.mlp_config.activation}"

                current_line = current_line.replace("activation=ttnn.silu", f"activation={activation}")

            if "return MLPOpConfigs" in current_line:
                # Change return to assignment to self.ops
                current_line = current_line.replace("return MLPOpConfigs", "self.ops = MLPOpConfigs")

            processed_lines.append(current_line)

        return processed_lines[1:]  # Skip def line

    def generate_module_class(self) -> str:
        """Generate complete module class with initialization and forward"""

        device_name = "wormhole_b0"  # Hardcoded default for now as we don't have hw_config

        # Build complete class
        class_lines = []

        # Add imports
        import_src = inspect.getsource(module_imports_template)
        import_src = textwrap.dedent(import_src)
        # Split lines and skip def line
        import_lines = import_src.split("\n")[1:]
        # Dedent body lines (they are indented inside the function)
        # Assuming 4 spaces indentation for function body
        import_lines = [line[4:] if line.startswith("    ") else line for line in import_lines]

        class_lines.extend(import_lines)
        class_lines.append("")

        # Add template classes
        class_lines.extend(class_to_source(LightweightModule))
        class_lines.append("")
        class_lines.extend(class_to_source(OpConfig))
        class_lines.append("")
        class_lines.extend(class_to_source(MLPOpConfigs))
        class_lines.append("")
        class_lines.extend(class_to_source(MLPWeights))
        class_lines.append("")

        # Add the implementation function source code as a free function (not in class)
        impl_source = function_to_source(forward_mlp_impl)
        for line in impl_source:
            class_lines.append(line)
        class_lines.append("")

        # Generate the main module class from template
        template_source = class_to_source(MLPModuleTemplate)

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
        final_lines[0] = final_lines[0].replace("MLPModuleTemplate", new_class_name)

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
            f.write("# Auto-generated by TTTv2 CodeGen\n")
            f.write("import ttnn\n\n")
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
    if save_source is not None:
        save_source.save(source_code, mlp_config)

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
