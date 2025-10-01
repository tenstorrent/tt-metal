"""
TTTv2 Metaclass-based Code Generation

This demonstrates how metaclasses can generate both functional classes
and their equivalent source code for deployment.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type


@dataclass
class HardwareConfig:
    """Hardware configuration"""

    device_name: str
    compute_grid: Tuple[int, int]
    supports_fp32_acc: bool
    optimal_tile_size: int
    l1_memory_per_core: int


@dataclass
class ModuleConfig:
    """Configuration for a TTT module"""

    in_features: int
    out_features: int
    bias: bool = True
    activation: Optional[str] = None


class CodeCaptureMeta(type):
    """Metaclass that captures generated code while creating classes"""

    generated_sources: Dict[str, str] = {}

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Extract configuration if provided
        module_config = kwargs.get("module_config")
        hw_config = kwargs.get("hw_config")

        if module_config and hw_config:
            # Generate optimized methods and source code
            generated_namespace, source_code = mcs._generate_optimized_code(name, module_config, hw_config)

            # Merge with provided namespace
            namespace.update(generated_namespace)

            # Store generated source
            mcs.generated_sources[name] = source_code

        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)

        # Attach source code to the class
        cls._generated_source = mcs.generated_sources.get(name, "")

        return cls

    @staticmethod
    def _generate_optimized_code(
        class_name: str, module_config: ModuleConfig, hw_config: HardwareConfig
    ) -> Tuple[Dict[str, Any], str]:
        """Generate both namespace entries and source code"""

        # Calculate optimal configuration
        shard_config = CodeCaptureMeta._calculate_sharding(module_config, hw_config)

        # Generate source code
        source_lines = [
            f"class {class_name}(TTTLinearBase):",
            f'    """Auto-generated Linear layer optimized for {hw_config.device_name}"""',
            f"",
            f"    # Hardware-specific constants",
            f'    DEVICE = "{hw_config.device_name}"',
            f"    IN_FEATURES = {module_config.in_features}",
            f"    OUT_FEATURES = {module_config.out_features}",
            f"    USE_BIAS = {module_config.bias}",
            f"",
            f"    # Pre-computed optimal configuration",
            f'    GRID_SIZE = {shard_config["grid_size"]}',
            f'    PER_CORE_M = {shard_config["per_core_M"]}',
            f'    PER_CORE_N = {shard_config["per_core_N"]}',
            f'    IN0_BLOCK_W = {shard_config["in0_block_w"]}',
            f'    OUT_SUBBLOCK_H = {shard_config["out_subblock_h"]}',
            f'    OUT_SUBBLOCK_W = {shard_config["out_subblock_w"]}',
            f"",
            f"    def __init__(self, device):",
            f"        super().__init__(device)",
            f"        self._setup_configs()",
            f"",
            f"    def _setup_configs(self):",
            f'        """Initialize hardware-specific configurations"""',
            f"        import ttnn",
            f"",
            f"        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(",
            f"            math_fidelity=ttnn.MathFidelity.HiFi4,",
            f"            math_approx_mode=True,",
            f"            fp32_dest_acc_en={hw_config.supports_fp32_acc},",
            f"            packer_l1_acc=True",
            f"        )",
            f"",
            f"        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(",
            f"            compute_with_storage_grid_size=self.GRID_SIZE,",
            f"            in0_block_w=self.IN0_BLOCK_W,",
            f"            out_subblock_h=self.OUT_SUBBLOCK_H,",
            f"            out_subblock_w=self.OUT_SUBBLOCK_W,",
            f"            per_core_M=self.PER_CORE_M,",
            f"            per_core_N=self.PER_CORE_N,",
            f"            transpose_mcast=False,",
            f"            fused_activation={CodeCaptureMeta._get_activation_string(module_config.activation)}",
            f"        )",
            f"",
            f"        self.memory_config = ttnn.MemoryConfig(",
            f"            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,",
            f"            buffer_type=ttnn.BufferType.L1",
            f"        )",
            f"",
            f"    def forward(self, x, weight, bias=None):",
            f'        """Optimized forward pass for {hw_config.device_name}"""',
            f"        import ttnn",
            f"",
            f"        # Ensure optimal memory layout",
            f"        if x.memory_config() != self.memory_config:",
            f"            x = ttnn.to_memory_config(x, self.memory_config)",
            f"",
            f"        # Execute with pre-optimized configuration",
            f"        output = ttnn.linear(",
            f"            x, weight, bias=bias,",
            f"            compute_kernel_config=self.compute_kernel_config,",
            f"            program_config=self.program_config,",
            f"            memory_config=self.memory_config,",
            f"            dtype=ttnn.bfloat16",
            f"        )",
            f"",
            f"        return output",
            f"",
            f"    @classmethod",
            f"    def get_optimal_weight_layout(cls):",
            f'        """Return optimal weight layout for this configuration"""',
            f"        return {{",
            f'            "shape": ({module_config.out_features}, {module_config.in_features}),',
            f'            "layout": "TILE_LAYOUT",',
            f'            "memory_config": "DRAM_INTERLEAVED",',
            f'            "dtype": "bfloat16"',
            f"        }}",
        ]

        source_code = "\n".join(source_lines)

        # Generate namespace entries (actual methods)
        namespace = {}

        # Class attributes
        namespace["DEVICE"] = hw_config.device_name
        namespace["IN_FEATURES"] = module_config.in_features
        namespace["OUT_FEATURES"] = module_config.out_features
        namespace["USE_BIAS"] = module_config.bias
        namespace["GRID_SIZE"] = shard_config["grid_size"]
        namespace["PER_CORE_M"] = shard_config["per_core_M"]
        namespace["PER_CORE_N"] = shard_config["per_core_N"]
        namespace["IN0_BLOCK_W"] = shard_config["in0_block_w"]
        namespace["OUT_SUBBLOCK_H"] = shard_config["out_subblock_h"]
        namespace["OUT_SUBBLOCK_W"] = shard_config["out_subblock_w"]

        # Methods
        def _setup_configs(self):
            import ttnn

            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=True,
                fp32_dest_acc_en=hw_config.supports_fp32_acc,
                packer_l1_acc=True,
            )

            self.program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=self.GRID_SIZE,
                in0_block_w=self.IN0_BLOCK_W,
                out_subblock_h=self.OUT_SUBBLOCK_H,
                out_subblock_w=self.OUT_SUBBLOCK_W,
                per_core_M=self.PER_CORE_M,
                per_core_N=self.PER_CORE_N,
                transpose_mcast=False,
                fused_activation=None,  # Simplified for example
            )

            self.memory_config = ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer_type=ttnn.BufferType.L1
            )

        def forward(self, x, weight, bias=None):
            import ttnn

            if x.memory_config() != self.memory_config:
                x = ttnn.to_memory_config(x, self.memory_config)

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

        @classmethod
        def get_optimal_weight_layout(cls):
            return {
                "shape": (module_config.out_features, module_config.in_features),
                "layout": "TILE_LAYOUT",
                "memory_config": "DRAM_INTERLEAVED",
                "dtype": "bfloat16",
            }

        namespace["_setup_configs"] = _setup_configs
        namespace["forward"] = forward
        namespace["get_optimal_weight_layout"] = get_optimal_weight_layout

        return namespace, source_code

    @staticmethod
    def _calculate_sharding(module_config: ModuleConfig, hw_config: HardwareConfig) -> Dict[str, Any]:
        """Calculate optimal sharding configuration"""
        m = module_config.out_features
        n = module_config.in_features

        # Simple sharding calculation
        grid_x = min(hw_config.compute_grid[0], (m + 127) // 128)
        grid_y = min(hw_config.compute_grid[1], (n + 127) // 128)

        return {
            "grid_size": (grid_x, grid_y),
            "per_core_M": (m + grid_x - 1) // grid_x,
            "per_core_N": (n + grid_y - 1) // grid_y,
            "in0_block_w": 2,
            "out_subblock_h": 1,
            "out_subblock_w": 4,
        }

    @staticmethod
    def _get_activation_string(activation: Optional[str]) -> str:
        """Convert activation to string representation"""
        if activation == "gelu":
            return "ttnn.UnaryOpType.GELU"
        elif activation == "relu":
            return "ttnn.UnaryOpType.RELU"
        else:
            return "None"


class TTTLinearBase:
    """Base class for TTT Linear modules"""

    def __init__(self, device):
        self.device = device

    def get_resource_usage(self) -> Dict[str, Any]:
        """Estimate resource usage"""
        # This would be overridden by generated classes
        return {}


class TTTModuleGenerator:
    """High-level API for generating TTT modules with metaclasses"""

    def __init__(self, hw_config: HardwareConfig):
        self.hw_config = hw_config

    def generate_linear(
        self, name: str, in_features: int, out_features: int, bias: bool = True, activation: Optional[str] = None
    ) -> Type[TTTLinearBase]:
        """Generate an optimized Linear module class"""

        module_config = ModuleConfig(
            in_features=in_features, out_features=out_features, bias=bias, activation=activation
        )

        # Create class with metaclass
        generated_class = CodeCaptureMeta(
            name, (TTTLinearBase,), {}, module_config=module_config, hw_config=self.hw_config
        )

        return generated_class

    def save_generated_source(self, module_class: Type, filepath: str):
        """Save the generated source code to file"""
        source = getattr(module_class, "_generated_source", "")
        if source:
            with open(filepath, "w") as f:
                f.write('"""Auto-generated by TTTv2 Metaclass Generator"""\n\n')
                f.write("import ttnn\n")
                f.write("from typing import Optional\n\n")
                f.write(source)
        else:
            raise ValueError(f"No generated source found for {module_class}")


class TTTModelBuilder:
    """Builder that uses metaclasses to construct complete models"""

    def __init__(self, model_name: str, hw_config: HardwareConfig):
        self.model_name = model_name
        self.hw_config = hw_config
        self.generator = TTTModuleGenerator(hw_config)
        self.generated_modules = {}

    def add_linear(self, name: str, in_features: int, out_features: int, **kwargs):
        """Add a linear layer to the model"""
        module_class = self.generator.generate_linear(f"{self.model_name}_{name}", in_features, out_features, **kwargs)
        self.generated_modules[name] = module_class
        return module_class

    def build_model(self) -> Tuple[Type, str]:
        """Build the complete model class and source code"""

        # Generate model source code
        model_source_lines = [
            f"class {self.model_name}Model:",
            f'    """Auto-generated model for {self.hw_config.device_name}"""',
            f"",
            f"    def __init__(self, device):",
            f"        self.device = device",
            f"        self.modules = {{}}",
            f"",
        ]

        # Add module initialization
        for name, module_cls in self.generated_modules.items():
            model_source_lines.append(f'        self.modules["{name}"] = {module_cls.__name__}(device)')

        # Add forward method
        model_source_lines.extend(
            [
                f"",
                f"    def forward(self, x, weights):",
                f'        """Forward pass through all modules"""',
                f"        # This is a simplified example",
                f"        for name, module in self.modules.items():",
                f"            if name in weights:",
                f'                x = module.forward(x, weights[name]["weight"], weights[name].get("bias"))',
                f"        return x",
                f"",
                f"    def get_total_resources(self):",
                f'        """Calculate total resource usage"""',
                f'        total = {{"l1": 0, "cores": 0}}',
                f"        for module in self.modules.values():",
                f"            usage = module.get_resource_usage()",
                f'            total["l1"] += usage.get("l1_memory_bytes", 0)',
                f'            total["cores"] = max(total["cores"], usage.get("active_cores", 0))',
                f"        return total",
            ]
        )

        model_source = "\n".join(model_source_lines)

        # Create the actual model class
        model_namespace = {
            "modules": self.generated_modules,
            "__init__": self._create_init_method(),
            "forward": self._create_forward_method(),
            "get_total_resources": self._create_resource_method(),
        }

        ModelClass = type(self.model_name + "Model", (), model_namespace)

        # Combine all sources
        complete_source = self._combine_sources(model_source)

        return ModelClass, complete_source

    def _create_init_method(self):
        def __init__(self, device):
            self.device = device
            self.modules = {}
            for name, module_cls in self.__class__.modules.items():
                self.modules[name] = module_cls(device)

        return __init__

    def _create_forward_method(self):
        def forward(self, x, weights):
            for name, module in self.modules.items():
                if name in weights:
                    x = module.forward(x, weights[name]["weight"], weights[name].get("bias"))
            return x

        return forward

    def _create_resource_method(self):
        def get_total_resources(self):
            total = {"l1": 0, "cores": 0}
            for module in self.modules.values():
                usage = module.get_resource_usage()
                total["l1"] += usage.get("l1_memory_bytes", 0)
                total["cores"] = max(total["cores"], usage.get("active_cores", 0))
            return total

        return get_total_resources

    def _combine_sources(self, model_source: str) -> str:
        """Combine all generated sources into one file"""
        combined = [
            '"""',
            f"Auto-generated model implementation for {self.model_name}",
            f"Target device: {self.hw_config.device_name}",
            '"""',
            "",
            "import ttnn",
            "from typing import Optional, Dict",
            "",
            "",
            "# Base class",
            inspect.getsource(TTTLinearBase),
            "",
            "",
        ]

        # Add all module sources
        for name, module_cls in self.generated_modules.items():
            combined.append(f"# Module: {name}")
            combined.append(module_cls._generated_source)
            combined.append("")
            combined.append("")

        # Add model class
        combined.append("# Model class")
        combined.append(model_source)

        return "\n".join(combined)


# Demonstration of the metaclass approach
if __name__ == "__main__":
    # Define hardware configuration
    wh_b0 = HardwareConfig(
        device_name="wormhole_b0",
        compute_grid=(8, 7),
        supports_fp32_acc=True,
        optimal_tile_size=32,
        l1_memory_per_core=1024 * 1024,
    )

    # Example 1: Generate individual module
    print("Example 1: Individual Module Generation")
    print("=" * 80)

    generator = TTTModuleGenerator(wh_b0)

    # Generate a Linear module class
    LinearQProj = generator.generate_linear("LinearQProj", in_features=4096, out_features=4096, bias=False)

    # The class is fully functional
    print(f"Generated class: {LinearQProj}")
    print(f"Class attributes: IN_FEATURES={LinearQProj.IN_FEATURES}, OUT_FEATURES={LinearQProj.OUT_FEATURES}")

    # And we can extract its source code
    print("\nGenerated source code:")
    print(LinearQProj._generated_source[:500] + "...")

    # Example 2: Build complete model
    print("\n\nExample 2: Complete Model Generation")
    print("=" * 80)

    builder = TTTModelBuilder("Llama2_7B", wh_b0)

    # Add layers
    builder.add_linear("q_proj", 4096, 4096, bias=False)
    builder.add_linear("k_proj", 4096, 1024, bias=False)
    builder.add_linear("v_proj", 4096, 1024, bias=False)
    builder.add_linear("o_proj", 4096, 4096, bias=False)
    builder.add_linear("mlp_gate", 4096, 11008, bias=False, activation="gelu")
    builder.add_linear("mlp_up", 4096, 11008, bias=False)
    builder.add_linear("mlp_down", 11008, 4096, bias=False)

    # Build the model
    ModelClass, complete_source = builder.build_model()

    print(f"Generated model class: {ModelClass}")
    print("\nComplete source code preview:")
    print(complete_source[:1000] + "\n... [truncated] ...")

    # Save the generated source
    with open("/tmp/generated_llama2_7b_wh_b0.py", "w") as f:
        f.write(complete_source)
    print("\nSaved complete source to: /tmp/generated_llama2_7b_wh_b0.py")
