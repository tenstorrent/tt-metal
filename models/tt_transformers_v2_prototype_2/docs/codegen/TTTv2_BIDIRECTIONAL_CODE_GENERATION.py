"""
TTTv2 Bidirectional Code Generation

Demonstrates how to:
1. Generate source code from actual functions (introspection)
2. Create functions from source code strings (compilation)
3. Keep both in sync for TTTv2 modules
"""

import ast
import inspect
from typing import Any, Callable, Dict, Optional, Tuple


class FunctionSourceGenerator:
    """Generate source code from function objects with TTTv2 optimizations"""

    def __init__(self, module_context: Dict[str, Any]):
        self.context = module_context

    def function_to_source(self, func: Callable, class_name: str = "GeneratedClass") -> str:
        """
        Convert a function to optimized source code by:
        1. Extracting the original source
        2. Analyzing for optimization opportunities
        3. Generating enhanced source with context
        """

        # Get original source
        original_source = inspect.getsource(func)

        # Parse AST for analysis
        tree = ast.parse(original_source)

        # Extract function details
        func_def = tree.body[0]
        func_name = func_def.name
        args = [arg.arg for arg in func_def.args.args]

        # Analyze function body for TTNN operations
        ttnn_ops = self._find_ttnn_operations(func_def)

        # Generate optimized source with full context
        source = self._generate_optimized_source(func_name, args, func_def, ttnn_ops, class_name)

        return source

    def _find_ttnn_operations(self, func_def: ast.FunctionDef) -> list:
        """Find all TTNN operations in the function"""

        class TTNNVisitor(ast.NodeVisitor):
            def __init__(self):
                self.ttnn_calls = []

            def visit_Call(self, node):
                # Look for ttnn.* calls
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "ttnn":
                        self.ttnn_calls.append(
                            {"op": node.func.attr, "args": len(node.args), "keywords": [kw.arg for kw in node.keywords]}
                        )
                self.generic_visit(node)

        visitor = TTNNVisitor()
        visitor.visit(func_def)
        return visitor.ttnn_calls

    def _generate_optimized_source(
        self, func_name: str, args: list, func_def: ast.FunctionDef, ttnn_ops: list, class_name: str
    ) -> str:
        """Generate complete optimized source code"""

        # Extract function body
        body_lines = ast.unparse(func_def).split("\n")[1:]  # Skip def line
        body = "\n".join(body_lines)

        # Build complete class source
        source_lines = [
            f"class {class_name}(TTTModuleBase):",
            f'    """Auto-generated from function with TTTv2 optimizations"""',
            f"",
            f"    # Extracted configuration from context",
        ]

        # Add configuration from context
        for key, value in self.context.items():
            if isinstance(value, (int, float, bool, str)):
                source_lines.append(f"    {key.upper()} = {repr(value)}")

        source_lines.extend(
            [
                f"",
                f"    def __init__(self, device):",
                f"        super().__init__(device)",
                f"        self.device = device",
                f"        self._setup_configs()",
                f"",
                f"    def _setup_configs(self):",
                f'        """Initialize configurations for TTNN operations"""',
                f"        import ttnn",
                f"",
            ]
        )

        # Add configuration setup based on detected TTNN ops
        if any(op["op"] == "linear" for op in ttnn_ops):
            source_lines.extend(
                [
                    f"        # Linear operation configuration",
                    f"        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(",
                    f"            math_fidelity=ttnn.MathFidelity.HiFi4,",
                    f'            fp32_dest_acc_en={self.context.get("fp32_acc", True)},',
                    f"            packer_l1_acc=True",
                    f"        )",
                    f"",
                    f"        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(",
                    f'            compute_with_storage_grid_size={self.context.get("grid_size", (8, 7))},',
                    f'            in0_block_w={self.context.get("in0_block_w", 2)},',
                    f'            out_subblock_h={self.context.get("out_subblock_h", 1)},',
                    f'            out_subblock_w={self.context.get("out_subblock_w", 4)},',
                    f'            per_core_M={self.context.get("per_core_M", 128)},',
                    f'            per_core_N={self.context.get("per_core_N", 128)}',
                    f"        )",
                    f"",
                    f"        self.memory_config = ttnn.MemoryConfig(",
                    f"            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,",
                    f"            buffer_type=ttnn.BufferType.L1",
                    f"        )",
                    f"",
                ]
            )

        # Add the forward method with original function body
        source_lines.extend(
            [
                f'    def {func_name}({", ".join(args)}):',
                f'        """Generated from introspected function"""',
            ]
        )

        # Indent and add function body
        for line in body.split("\n"):
            if line.strip():
                source_lines.append(f"        {line}")
            else:
                source_lines.append("")

        return "\n".join(source_lines)


class SourceFunctionCompiler:
    """Compile source code strings into executable functions"""

    def __init__(self, global_context: Optional[Dict[str, Any]] = None):
        self.global_context = global_context or {}

    def source_to_function(self, source: str, func_name: str = "forward") -> Callable:
        """
        Compile source code string into an executable function
        """

        # Parse source to extract function
        tree = ast.parse(source)

        # Find the target function in the class
        target_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                target_func = node
                break

        if not target_func:
            raise ValueError(f"Function {func_name} not found in source")

        # Create a module from the source
        compiled = compile(source, "<generated>", "exec")

        # Execute in a namespace
        namespace = {"TTTModuleBase": TTTModuleBase, **self.global_context}
        exec(compiled, namespace)

        # Extract the class and get the method
        for name, obj in namespace.items():
            if isinstance(obj, type) and issubclass(obj, TTTModuleBase):
                # Create instance and get method
                instance = obj(device="dummy")
                method = getattr(instance, func_name)

                # Return unbound version
                return method.__func__

        raise ValueError("Could not extract function from source")

    def source_to_class(self, source: str) -> type:
        """Compile source code string into a class"""

        # Compile and execute
        compiled = compile(source, "<generated>", "exec")
        namespace = {"TTTModuleBase": TTTModuleBase, **self.global_context}
        exec(compiled, namespace)

        # Find and return the class
        for name, obj in namespace.items():
            if isinstance(obj, type) and issubclass(obj, TTTModuleBase) and name != "TTTModuleBase":
                return obj

        raise ValueError("No class found in source")


class TTTModuleBase:
    """Base class for TTT modules"""

    def __init__(self, device):
        self.device = device


class BidirectionalCodeGenerator:
    """
    Combines introspection and compilation for bidirectional code generation
    """

    def __init__(self, hw_config: Dict[str, Any]):
        self.hw_config = hw_config
        self.source_gen = FunctionSourceGenerator(hw_config)
        self.compiler = SourceFunctionCompiler({"ttnn": MockTTNN()})

    def create_optimized_module(self, name: str, forward_func: Callable, config: Dict[str, Any]) -> Tuple[type, str]:
        """
        Create both a functional class and its source code from a function
        """

        # Merge config with hardware config
        full_context = {**self.hw_config, **config}
        self.source_gen.context = full_context

        # Generate source from function
        source = self.source_gen.function_to_source(forward_func, name)

        # Compile source back to class
        generated_class = self.compiler.source_to_class(source)

        # Attach source to class
        generated_class._generated_source = source

        return generated_class, source

    def modify_and_regenerate(self, original_source: str, modifications: Dict[str, Any]) -> Tuple[type, str]:
        """
        Modify source code and regenerate class
        """

        # Parse source
        tree = ast.parse(original_source)

        # Apply modifications (simplified example)
        modifier = SourceModifier(modifications)
        modified_tree = modifier.visit(tree)

        # Generate new source
        new_source = ast.unparse(modified_tree)

        # Compile to class
        new_class = self.compiler.source_to_class(new_source)
        new_class._generated_source = new_source

        return new_class, new_source


class SourceModifier(ast.NodeTransformer):
    """Modify AST based on configuration changes"""

    def __init__(self, modifications: Dict[str, Any]):
        self.mods = modifications

    def visit_Assign(self, node):
        # Modify assignments for configuration values
        if isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in self.mods:
                # Replace with new value
                node.value = ast.Constant(value=self.mods[name])
        return node


# Mock TTNN for demonstration
class MockTTNN:
    class WormholeComputeKernelConfig:
        def __init__(self, **kwargs):
            pass

    class MatmulMultiCoreReuseMultiCastProgramConfig:
        def __init__(self, **kwargs):
            pass

    class MemoryConfig:
        def __init__(self, **kwargs):
            pass

    class TensorMemoryLayout:
        BLOCK_SHARDED = "BLOCK_SHARDED"

    class BufferType:
        L1 = "L1"

    class MathFidelity:
        HiFi4 = "HiFi4"

    @staticmethod
    def linear(x, weight, bias=None, **kwargs):
        return x  # Mock return

    @staticmethod
    def to_memory_config(x, config):
        return x


# Example demonstrating bidirectional generation
if __name__ == "__main__":
    # Hardware configuration
    hw_config = {
        "device_name": "wormhole_b0",
        "grid_size": (8, 7),
        "fp32_acc": True,
        "in0_block_w": 2,
        "out_subblock_h": 1,
        "out_subblock_w": 4,
        "per_core_M": 128,
        "per_core_N": 128,
    }

    # Define a forward function
    def forward(self, x, weight, bias=None):
        """Optimized forward pass for wormhole_b0"""
        import ttnn

        # Ensure optimal memory layout
        if x.memory_config() != self.memory_config:
            x = ttnn.to_memory_config(x, self.memory_config)

        # Execute with pre-optimized configuration
        output = ttnn.linear(
            x,
            weight,
            bias=bias,
            compute_kernel_config=self.compute_kernel_config,
            program_config=self.program_config,
            memory_config=self.memory_config,
            dtype=ttnn.bfloat16,
        )

        return output

    # Create bidirectional generator
    generator = BidirectionalCodeGenerator(hw_config)

    print("=== Example 1: Function to Source to Class ===")
    print("Original function:", forward.__name__)

    # Generate optimized module from function
    LinearClass, source = generator.create_optimized_module(
        "OptimizedLinear", forward, {"in_features": 4096, "out_features": 4096}
    )

    print("\nGenerated source code:")
    print("-" * 60)
    print(source[:800] + "\n... [truncated]")

    print(f"\nGenerated class: {LinearClass}")
    print(f"Class has forward method: {hasattr(LinearClass, 'forward')}")

    # Example 2: Modify generated source and regenerate
    print("\n\n=== Example 2: Modify Source and Regenerate ===")

    modifications = {"GRID_SIZE": (4, 4), "PER_CORE_M": 256, "PER_CORE_N": 256}  # Smaller grid  # Larger per-core work

    ModifiedClass, modified_source = generator.modify_and_regenerate(source, modifications)

    print("Modified configurations in source:")
    for line in modified_source.split("\n"):
        if any(key in line for key in modifications.keys()):
            print(f"  {line.strip()}")

    # Example 3: Extract function from existing source
    print("\n\n=== Example 3: Extract Function from Source ===")

    existing_source = """
class ExistingModule(TTTModuleBase):
    def forward(self, x, weight, bias=None):
        import ttnn
        # Custom implementation
        result = ttnn.linear(x, weight, bias=bias)
        return result * 2  # Some custom scaling
"""

    compiler = SourceFunctionCompiler({"ttnn": MockTTNN()})
    extracted_func = compiler.source_to_function(existing_source, "forward")

    print(f"Successfully extracted function: {extracted_func}")
    print(f"Function name: {extracted_func.__name__}")

    # Show complete workflow
    print("\n\n=== Complete Workflow Summary ===")
    print("1. Start with Python function")
    print("2. Introspect to generate optimized source code")
    print("3. Source includes hardware-specific configurations")
    print("4. Compile source back to executable class")
    print("5. Both function and source are kept in sync")
    print("6. Can modify source and regenerate class")
    print("7. Can extract functions from existing source")
