"""
TTTv2 Integrated Bidirectional Code Generation

This example shows how bidirectional code generation integrates with
the broader TTTv2 metaclass system to provide seamless optimization.
"""

import ast
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Type


@dataclass
class OptimizationContext:
    """Context for code optimization decisions"""

    hw_config: Dict[str, Any]
    module_config: Dict[str, Any]
    optimization_level: int = 2
    debug_mode: bool = False


class TTTSmartMeta(type):
    """
    Advanced metaclass that provides bidirectional code generation:
    - Creates optimized implementations from high-level descriptions
    - Generates source code that can be inspected and modified
    - Allows runtime modification and recompilation
    """

    _registry = {}

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Check if we have a prototype function
        prototype_func = namespace.get("_prototype_forward")
        optimization_context = kwargs.get("optimization_context")

        if prototype_func and optimization_context:
            # Generate optimized implementation from prototype
            optimized_namespace, source = mcs._optimize_from_prototype(name, prototype_func, optimization_context)
            namespace.update(optimized_namespace)

            # Store bidirectional mapping
            mcs._registry[name] = {
                "prototype": prototype_func,
                "source": source,
                "context": optimization_context,
                "optimized_methods": list(optimized_namespace.keys()),
            }

        # Create class
        cls = super().__new__(mcs, name, bases, namespace)

        # Attach introspection methods
        cls.get_generated_source = lambda: mcs._registry.get(name, {}).get("source", "")
        cls.get_optimization_context = lambda: mcs._registry.get(name, {}).get("context")
        cls.regenerate_with_config = lambda new_config: mcs._regenerate_class(cls, new_config)

        return cls

    @staticmethod
    def _optimize_from_prototype(
        class_name: str, prototype: Callable, context: OptimizationContext
    ) -> Tuple[Dict[str, Any], str]:
        """Generate optimized implementation from prototype function"""

        # Analyze prototype
        sig = inspect.signature(prototype)
        params = list(sig.parameters.keys())

        # Get prototype source and analyze
        proto_source = inspect.getsource(prototype)
        proto_ast = ast.parse(proto_source)

        # Identify optimization opportunities
        optimizer = TTTCodeOptimizer(context)
        optimized_ast = optimizer.optimize(proto_ast)

        # Generate both namespace and source
        namespace = {}
        source_lines = [f"class {class_name}(TTTModuleBase):"]

        # Add configuration constants
        source_lines.extend(
            [
                f'    """Optimized implementation for {context.hw_config["device"]}"""',
                f"",
                f"    # Hardware configuration",
            ]
        )

        for key, value in context.hw_config.items():
            if isinstance(value, (int, float, bool, str, tuple, list)):
                namespace[key.upper()] = value
                source_lines.append(f"    {key.upper()} = {repr(value)}")

        source_lines.append("")

        # Generate __init__ method
        init_source = [
            "    def __init__(self, device):",
            "        super().__init__(device)",
            "        self._init_optimized_configs()",
            "",
        ]
        source_lines.extend(init_source)

        def __init__(self, device):
            super(self.__class__.__bases__[0], self).__init__(device)
            self._init_optimized_configs()

        namespace["__init__"] = __init__

        # Generate config initialization
        config_source, init_configs_func = optimizer.generate_config_init()
        source_lines.extend(config_source)
        namespace["_init_optimized_configs"] = init_configs_func

        # Generate optimized forward method
        forward_source = optimizer.generate_optimized_forward(prototype, optimized_ast)
        source_lines.extend(forward_source)

        # Create actual forward function
        forward_func = mcs._compile_forward_function(forward_source, context)
        namespace["forward"] = forward_func

        # Add utility methods
        source_lines.extend(
            [
                "",
                "    def estimate_performance(self):",
                '        """Estimate performance metrics"""',
                f"        flops = {optimizer.estimate_flops()}",
                f"        memory_bandwidth = {optimizer.estimate_bandwidth()}",
                '        return {"flops": flops, "bandwidth": memory_bandwidth}',
            ]
        )

        def estimate_performance(self):
            return {"flops": optimizer.estimate_flops(), "bandwidth": optimizer.estimate_bandwidth()}

        namespace["estimate_performance"] = estimate_performance

        # Combine source
        full_source = "\n".join(source_lines)

        return namespace, full_source

    @staticmethod
    def _compile_forward_function(source_lines: list, context: OptimizationContext) -> Callable:
        """Compile forward function from source"""

        # Extract just the forward method
        forward_start = None
        forward_end = None
        indent_level = 0

        for i, line in enumerate(source_lines):
            if "def forward(" in line:
                forward_start = i
                indent_level = len(line) - len(line.lstrip())
            elif forward_start is not None and line.strip() and len(line) - len(line.lstrip()) <= indent_level:
                if not line.strip().startswith(('"""', "'''")):
                    forward_end = i
                    break

        if forward_start is None:
            raise ValueError("No forward method found")

        if forward_end is None:
            forward_end = len(source_lines)

        # Create standalone function
        func_lines = []
        for line in source_lines[forward_start:forward_end]:
            # Remove class method indentation
            if line.strip():
                func_lines.append(line[4:])  # Remove 4 spaces
            else:
                func_lines.append("")

        func_source = "\n".join(func_lines)

        # Compile in a namespace with necessary imports
        namespace = {"ttnn": MockTTNN(), **context.module_config}

        exec(func_source, namespace)
        return namespace["forward"]

    @classmethod
    def _regenerate_class(mcs, original_class: Type, new_config: Dict[str, Any]) -> Type:
        """Regenerate class with new configuration"""

        # Get original registration
        original_name = original_class.__name__
        registration = mcs._registry.get(original_name)

        if not registration:
            raise ValueError(f"Class {original_name} was not generated by TTTSmartMeta")

        # Update context
        new_context = OptimizationContext(
            hw_config={**registration["context"].hw_config, **new_config},
            module_config=registration["context"].module_config,
            optimization_level=registration["context"].optimization_level,
        )

        # Generate new class with updated name
        new_name = f"{original_name}_Regenerated"

        # Create new class
        new_class = TTTSmartMeta(
            new_name,
            original_class.__bases__,
            {"_prototype_forward": registration["prototype"]},
            optimization_context=new_context,
        )

        return new_class


class TTTCodeOptimizer:
    """Optimizes code based on hardware and module configuration"""

    def __init__(self, context: OptimizationContext):
        self.context = context

    def optimize(self, tree: ast.AST) -> ast.AST:
        """Apply optimizations to AST"""
        # Simplified - would apply various optimization passes
        return tree

    def generate_config_init(self) -> Tuple[list, Callable]:
        """Generate configuration initialization"""

        source = [
            "    def _init_optimized_configs(self):",
            '        """Initialize hardware-optimized configurations"""',
            "        import ttnn",
            "",
        ]

        if self.context.hw_config.get("device") == "wormhole_b0":
            source.extend(
                [
                    "        self.compute_config = ttnn.WormholeComputeKernelConfig(",
                    "            math_fidelity=ttnn.MathFidelity.HiFi4,",
                    "            fp32_dest_acc_en=True,",
                    "            packer_l1_acc=True",
                    "        )",
                    "",
                    "        self.memory_config = ttnn.MemoryConfig(",
                    "            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,",
                    "            buffer_type=ttnn.BufferType.L1",
                    "        )",
                ]
            )

        def _init_optimized_configs(self):
            import ttnn

            self.compute_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
            )
            self.memory_config = ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1
            )

        return source, _init_optimized_configs

    def generate_optimized_forward(self, prototype: Callable, optimized_ast: ast.AST) -> list:
        """Generate optimized forward method"""

        # Get prototype source
        proto_source = inspect.getsource(prototype)
        proto_lines = proto_source.strip().split("\n")[1:]  # Skip def line

        source = [
            "    def forward(self, x, weight, bias=None):",
            '        """Hardware-optimized forward pass"""',
            "        import ttnn",
            "",
            "        # Optimization: Ensure optimal memory layout",
            '        if hasattr(x, "memory_config") and x.memory_config() != self.memory_config:',
            "            x = ttnn.to_memory_config(x, self.memory_config)",
            "",
        ]

        # Add optimized implementation
        if "linear" in str(proto_source):
            source.extend(
                [
                    "        # Optimized linear operation",
                    "        output = ttnn.linear(",
                    "            x, weight, bias=bias,",
                    "            compute_kernel_config=self.compute_config,",
                    "            memory_config=self.memory_config",
                    "        )",
                    "        return output",
                ]
            )
        else:
            # Add prototype body with adjustments
            for line in proto_lines:
                source.append(f"        {line}")

        return source

    def estimate_flops(self) -> int:
        """Estimate FLOPS for the operation"""
        # Simplified estimation
        if "in_features" in self.context.module_config:
            m = self.context.module_config.get("out_features", 1)
            n = self.context.module_config["in_features"]
            return 2 * m * n  # Simplified

        return 1000000  # Default

    def estimate_bandwidth(self) -> float:
        """Estimate memory bandwidth requirement"""
        return 100.0  # GB/s simplified


class TTTModuleBase:
    """Base class for TTT modules"""

    def __init__(self, device):
        self.device = device


# Mock TTNN for demonstration
class MockTTNN:
    class WormholeComputeKernelConfig:
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
        return x

    @staticmethod
    def to_memory_config(x, config):
        return x


# Demonstration
if __name__ == "__main__":
    print("=== TTTv2 Integrated Bidirectional Code Generation ===\n")

    # Step 1: Define a prototype forward function
    def prototype_linear_forward(self, x, weight, bias=None):
        """Prototype forward pass - will be optimized"""
        import ttnn

        # Simple prototype - metaclass will optimize this
        return ttnn.linear(x, weight, bias=bias)

    # Step 2: Create optimization context
    context = OptimizationContext(
        hw_config={"device": "wormhole_b0", "compute_grid": (8, 7), "l1_memory": 1024 * 1024},
        module_config={"in_features": 4096, "out_features": 4096, "bias": True},
        optimization_level=2,
    )

    # Step 3: Create optimized class using metaclass
    OptimizedLinear = TTTSmartMeta(
        "OptimizedLinear",
        (TTTModuleBase,),
        {"_prototype_forward": prototype_linear_forward},
        optimization_context=context,
    )

    print(f"Created class: {OptimizedLinear}")
    print(f"\nOptimization context: {OptimizedLinear.get_optimization_context()}")

    # Step 4: Inspect generated source
    generated_source = OptimizedLinear.get_generated_source()
    print("\nGenerated source code:")
    print("-" * 60)
    print(generated_source[:600] + "\n... [truncated]")

    # Step 5: Use the optimized class
    device = "wormhole_b0"
    module = OptimizedLinear(device)
    perf = module.estimate_performance()
    print(f"\nEstimated performance: {perf}")

    # Step 6: Regenerate with different configuration
    print("\n=== Regenerating with Modified Configuration ===")

    new_config = {"compute_grid": (4, 4), "optimization_level": 3}  # Smaller grid  # Higher optimization

    RegeneratedLinear = OptimizedLinear.regenerate_with_config(new_config)

    print(f"\nRegenerated class: {RegeneratedLinear}")
    print("\nUpdated configuration in source:")

    new_source = RegeneratedLinear.get_generated_source()
    for line in new_source.split("\n"):
        if "COMPUTE_GRID" in line:
            print(f"  {line.strip()}")

    # Step 7: Show the complete workflow
    print("\n=== Complete Bidirectional Workflow ===")
    print("1. Start with prototype function (high-level intent)")
    print("2. Metaclass analyzes and optimizes based on hardware")
    print("3. Generates both executable class AND readable source")
    print("4. Source can be inspected, modified, and regenerated")
    print("5. Changes to configuration trigger regeneration")
    print("6. All optimizations are traceable and debuggable")

    print("\nThis enables TTTv2 to:")
    print("- Support 100+ models with hardware-specific optimizations")
    print("- Maintain readable, debuggable generated code")
    print("- Allow model developers to understand optimizations")
    print("- Enable iterative refinement of implementations")
