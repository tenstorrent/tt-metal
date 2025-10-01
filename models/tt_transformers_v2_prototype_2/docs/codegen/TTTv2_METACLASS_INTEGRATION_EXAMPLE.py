"""
TTTv2 Metaclass Integration Example

Shows how the metaclass approach integrates with the broader TTTv2
module system and enables dynamic optimization with source code export.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type


@dataclass
class TTTModuleSpec:
    """Specification for a TTT module"""

    module_type: str
    config: Dict[str, Any]
    hw_constraints: Optional[Dict[str, Any]] = None


class TTTOptimizingMeta(type):
    """
    Advanced metaclass that:
    1. Generates optimized implementations
    2. Captures source code
    3. Provides runtime introspection
    4. Enables hot-swapping of implementations
    """

    # Registry of all generated modules
    _module_registry: Dict[str, Type] = {}
    _source_registry: Dict[str, str] = {}
    _optimization_metadata: Dict[str, Dict] = {}

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Extract TTT-specific configuration
        module_spec = kwargs.get("module_spec")
        hw_config = kwargs.get("hw_config")
        optimize = kwargs.get("optimize", True)

        if module_spec and hw_config and optimize:
            # Run optimization pipeline
            opt_namespace, source, metadata = mcs._optimize_module(name, bases, namespace, module_spec, hw_config)
            namespace.update(opt_namespace)

            # Store metadata
            mcs._source_registry[name] = source
            mcs._optimization_metadata[name] = metadata
        else:
            # No optimization, use as-is
            source = mcs._extract_source(name, namespace)
            mcs._source_registry[name] = source

        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)

        # Register the module
        mcs._module_registry[name] = cls

        # Attach TTT-specific attributes
        cls._ttt_source = mcs._source_registry.get(name, "")
        cls._ttt_metadata = mcs._optimization_metadata.get(name, {})
        cls._ttt_spec = module_spec

        return cls

    @classmethod
    def _optimize_module(
        mcs, name: str, bases: tuple, namespace: dict, module_spec: TTTModuleSpec, hw_config: dict
    ) -> tuple:
        """Run optimization pipeline on module"""

        metadata = {"original_spec": module_spec, "hardware": hw_config, "optimizations_applied": []}

        # Choose optimization strategy based on module type
        if module_spec.module_type == "linear":
            opt_namespace, source = mcs._optimize_linear(name, module_spec.config, hw_config, metadata)
        elif module_spec.module_type == "attention":
            opt_namespace, source = mcs._optimize_attention(name, module_spec.config, hw_config, metadata)
        elif module_spec.module_type == "layernorm":
            opt_namespace, source = mcs._optimize_layernorm(name, module_spec.config, hw_config, metadata)
        else:
            # Fallback to generic optimization
            opt_namespace = namespace
            source = mcs._extract_source(name, namespace)

        # Apply common optimizations
        opt_namespace = mcs._apply_common_optimizations(opt_namespace, module_spec, hw_config, metadata)

        return opt_namespace, source, metadata

    @staticmethod
    def _optimize_linear(name: str, config: dict, hw_config: dict, metadata: dict) -> tuple:
        """Optimize linear layer implementation"""

        in_features = config["in_features"]
        out_features = config["out_features"]
        bias = config.get("bias", True)

        # Calculate optimal sharding
        grid_x, grid_y = hw_config["compute_grid"]
        cores_m = min(grid_x, (out_features + 127) // 128)
        cores_n = min(grid_y, (in_features + 127) // 128)

        metadata["optimizations_applied"].extend(["optimal_sharding", "memory_layout_optimization", "kernel_selection"])

        # Generate optimized implementation
        namespace = {
            "IN_FEATURES": in_features,
            "OUT_FEATURES": out_features,
            "USE_BIAS": bias,
            "CORES_M": cores_m,
            "CORES_N": cores_n,
        }

        def forward(self, x, weight, bias=None):
            import ttnn

            # Pre-computed optimal config
            compute_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
            )

            program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(self.CORES_M, self.CORES_N),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=(out_features + self.CORES_M - 1) // self.CORES_M,
                per_core_N=(in_features + self.CORES_N - 1) // self.CORES_N,
            )

            return ttnn.linear(
                x, weight, bias=bias, compute_kernel_config=compute_config, program_config=program_config
            )

        namespace["forward"] = forward

        # Generate source code
        source = f'''
class {name}(TTTModuleBase):
    """Optimized Linear layer for {hw_config['device_name']}"""

    IN_FEATURES = {in_features}
    OUT_FEATURES = {out_features}
    USE_BIAS = {bias}
    CORES_M = {cores_m}
    CORES_N = {cores_n}

    def forward(self, x, weight, bias=None):
        import ttnn

        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True
        )

        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(self.CORES_M, self.CORES_N),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M={(out_features + cores_m - 1) // cores_m},
            per_core_N={(in_features + cores_n - 1) // cores_n},
        )

        return ttnn.linear(
            x, weight, bias=bias,
            compute_kernel_config=compute_config,
            program_config=program_config
        )
'''

        return namespace, source

    @staticmethod
    def _optimize_attention(name: str, config: dict, hw_config: dict, metadata: dict) -> tuple:
        """Optimize attention implementation"""

        hidden_dim = config["hidden_dim"]
        num_heads = config["num_heads"]
        use_flash = hidden_dim >= 2048 and hw_config.get("supports_flash_attention", True)

        metadata["optimizations_applied"].append("flash_attention" if use_flash else "decomposed_attention")

        namespace = {
            "HIDDEN_DIM": hidden_dim,
            "NUM_HEADS": num_heads,
            "USE_FLASH": use_flash,
        }

        if use_flash:

            def forward(self, q, k, v, mask=None):
                import ttnn

                return ttnn.transformer.flash_attention(
                    q, k, v, is_causal=True, scale=1.0 / (self.HIDDEN_DIM // self.NUM_HEADS) ** 0.5
                )

        else:

            def forward(self, q, k, v, mask=None):
                import ttnn

                scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
                scores = scores * (1.0 / (self.HIDDEN_DIM // self.NUM_HEADS) ** 0.5)
                if mask is not None:
                    scores = scores + mask
                probs = ttnn.softmax(scores, dim=-1)
                return ttnn.matmul(probs, v)

        namespace["forward"] = forward

        source = f'''
class {name}(TTTModuleBase):
    """Optimized Attention for {hw_config['device_name']}"""

    HIDDEN_DIM = {hidden_dim}
    NUM_HEADS = {num_heads}
    USE_FLASH = {use_flash}

    def forward(self, q, k, v, mask=None):
        import ttnn
        {"return ttnn.transformer.flash_attention(q, k, v, is_causal=True, scale=1.0 / (self.HIDDEN_DIM // self.NUM_HEADS) ** 0.5)" if use_flash else """scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
        scores = scores * (1.0 / (self.HIDDEN_DIM // self.NUM_HEADS) ** 0.5)
        if mask is not None:
            scores = scores + mask
        probs = ttnn.softmax(scores, dim=-1)
        return ttnn.matmul(probs, v)"""}
'''

        return namespace, source

    @staticmethod
    def _optimize_layernorm(name: str, config: dict, hw_config: dict, metadata: dict) -> tuple:
        """Optimize LayerNorm implementation"""

        normalized_shape = config["normalized_shape"]
        eps = config.get("eps", 1e-5)

        metadata["optimizations_applied"].append("fused_layernorm")

        namespace = {
            "NORMALIZED_SHAPE": normalized_shape,
            "EPS": eps,
        }

        def forward(self, x, weight, bias):
            import ttnn

            return ttnn.layer_norm(x, weight=weight, bias=bias, eps=self.EPS)

        namespace["forward"] = forward

        source = f'''
class {name}(TTTModuleBase):
    """Optimized LayerNorm for {hw_config['device_name']}"""

    NORMALIZED_SHAPE = {normalized_shape}
    EPS = {eps}

    def forward(self, x, weight, bias):
        import ttnn
        return ttnn.layer_norm(x, weight=weight, bias=bias, eps=self.EPS)
'''

        return namespace, source

    @staticmethod
    def _apply_common_optimizations(
        namespace: dict, module_spec: TTTModuleSpec, hw_config: dict, metadata: dict
    ) -> dict:
        """Apply optimizations common to all module types"""

        # Add resource estimation
        def estimate_resources(self):
            # Simplified resource estimation
            base_memory = 1024  # Base overhead

            if hasattr(self, "IN_FEATURES") and hasattr(self, "OUT_FEATURES"):
                # Linear layer estimation
                weight_memory = self.IN_FEATURES * self.OUT_FEATURES * 2  # bfloat16
                activation_memory = 32 * max(self.IN_FEATURES, self.OUT_FEATURES) * 2
                return {
                    "memory_bytes": base_memory + weight_memory + activation_memory,
                    "compute_cycles": self.IN_FEATURES * self.OUT_FEATURES,
                }

            return {"memory_bytes": base_memory, "compute_cycles": 1000}

        namespace["estimate_resources"] = estimate_resources

        # Add profiling hooks
        def profile_forward(self, *args, **kwargs):
            import time

            start = time.perf_counter()
            result = self.forward(*args, **kwargs)
            end = time.perf_counter()
            self._last_forward_time = end - start
            return result

        namespace["profile_forward"] = profile_forward

        metadata["optimizations_applied"].append("resource_estimation")
        metadata["optimizations_applied"].append("profiling_hooks")

        return namespace

    @staticmethod
    def _extract_source(name: str, namespace: dict) -> str:
        """Extract source code from namespace"""
        # Simplified source extraction
        return f"# Source for {name}\n# [Generated from namespace]"

    @classmethod
    def get_module(mcs, name: str) -> Optional[Type]:
        """Get a registered module by name"""
        return mcs._module_registry.get(name)

    @classmethod
    def export_all_sources(mcs, directory: str):
        """Export all generated sources to a directory"""
        os.makedirs(directory, exist_ok=True)

        for module_name, source in mcs._source_registry.items():
            filepath = os.path.join(directory, f"{module_name}.py")
            with open(filepath, "w") as f:
                f.write('"""Auto-generated by TTTv2"""\n\n')
                f.write("from tttv2_base import TTTModuleBase\n")
                f.write("import ttnn\n\n")
                f.write(source)

    @classmethod
    def create_deployment_package(mcs, modules: List[str], output_path: str):
        """Create a deployment package with selected modules"""

        package_source = '''"""
TTTv2 Deployment Package
Auto-generated optimized modules
"""

import ttnn
from typing import Dict, Optional, Any


class TTTModuleBase:
    """Base class for all TTT modules"""

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def estimate_resources(self) -> Dict[str, Any]:
        return {'memory_bytes': 0, 'compute_cycles': 0}


'''

        # Add selected modules
        for module_name in modules:
            if module_name in mcs._source_registry:
                package_source += f"\n\n# Module: {module_name}\n"
                package_source += mcs._source_registry[module_name]
                package_source += "\n"

        # Add loader
        package_source += '''

# Module loader
def load_module(module_name: str):
    """Load a module by name"""
    return globals().get(module_name)


# Available modules
AVAILABLE_MODULES = [
'''
        for module_name in modules:
            package_source += f'    "{module_name}",\n'

        package_source += "]\n"

        # Write package
        with open(output_path, "w") as f:
            f.write(package_source)

        return output_path


class TTTModuleBase:
    """Base class for TTT modules"""


class TTTv2ModelFactory:
    """
    High-level factory for creating complete models using the metaclass system
    """

    def __init__(self, model_name: str, hw_config: Dict[str, Any]):
        self.model_name = model_name
        self.hw_config = hw_config
        self.modules = {}

    def add_module(
        self, name: str, module_type: str, config: Dict[str, Any], hw_constraints: Optional[Dict[str, Any]] = None
    ) -> Type:
        """Add a module to the model"""

        module_spec = TTTModuleSpec(module_type=module_type, config=config, hw_constraints=hw_constraints)

        # Generate unique class name
        class_name = f"{self.model_name}_{name}_{module_type}"

        # Create the module class using metaclass
        module_class = TTTOptimizingMeta(
            class_name, (TTTModuleBase,), {}, module_spec=module_spec, hw_config=self.hw_config, optimize=True
        )

        self.modules[name] = module_class
        return module_class

    def create_model(self) -> Type:
        """Create the complete model class"""

        modules = self.modules

        class Model:
            def __init__(self, device):
                self.device = device
                self.layers = {}
                for name, module_cls in modules.items():
                    self.layers[name] = module_cls()

            def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                # Simplified forward pass
                x = inputs["input"]

                # Process through layers in order
                if "attention" in self.layers:
                    x = self.layers["attention"].forward(x, x, x)

                if "ffn" in self.layers:
                    x = self.layers["ffn"].forward(x, inputs["ffn_weight"])

                return {"output": x}

            def get_optimizations_summary(self) -> Dict[str, List[str]]:
                """Get summary of optimizations applied to each module"""
                summary = {}
                for name, layer in self.layers.items():
                    if hasattr(layer, "_ttt_metadata"):
                        summary[name] = layer._ttt_metadata.get("optimizations_applied", [])
                return summary

            def export_optimized_source(self, filepath: str):
                """Export the optimized source code for deployment"""
                TTTOptimizingMeta.create_deployment_package(list(modules.keys()), filepath)

        # Set the name
        Model.__name__ = f"{self.model_name}Model"

        return Model


# Example usage demonstrating integration
if __name__ == "__main__":
    # Hardware configuration
    hw_config = {
        "device_name": "wormhole_b0",
        "compute_grid": (8, 7),
        "supports_flash_attention": True,
        "l1_memory_per_core": 1024 * 1024,
    }

    # Create a transformer model using the factory
    factory = TTTv2ModelFactory("OptimizedTransformer", hw_config)

    # Add attention module
    attention_cls = factory.add_module(
        name="self_attention", module_type="attention", config={"hidden_dim": 4096, "num_heads": 32}
    )

    # Add FFN modules
    ffn_up_cls = factory.add_module(
        name="ffn_up", module_type="linear", config={"in_features": 4096, "out_features": 11008, "bias": False}
    )

    ffn_down_cls = factory.add_module(
        name="ffn_down", module_type="linear", config={"in_features": 11008, "out_features": 4096, "bias": False}
    )

    # Add LayerNorm
    ln_cls = factory.add_module(
        name="layer_norm", module_type="layernorm", config={"normalized_shape": [4096], "eps": 1e-5}
    )

    # Create the model
    TransformerModel = factory.create_model()

    # Demonstrate usage
    print(f"Created model: {TransformerModel.__name__}")
    print("\nRegistered modules:")
    for name, cls in factory.modules.items():
        print(f"  {name}: {cls.__name__}")
        if hasattr(cls, "_ttt_metadata"):
            optimizations = cls._ttt_metadata.get("optimizations_applied", [])
            print(f"    Optimizations: {', '.join(optimizations)}")

    # Show generated source for one module
    print("\nGenerated source for ffn_up module:")
    print(ffn_up_cls._ttt_source[:600] + "...")

    # Export all sources
    print("\nExporting all module sources...")
    TTTOptimizingMeta.export_all_sources("/tmp/tttv2_generated_modules")
    print("Exported to: /tmp/tttv2_generated_modules/")

    # Create deployment package
    print("\nCreating deployment package...")
    deployment_path = "/tmp/optimized_transformer_deploy.py"
    TTTOptimizingMeta.create_deployment_package(list(factory.modules.keys()), deployment_path)
    print(f"Created deployment package: {deployment_path}")
