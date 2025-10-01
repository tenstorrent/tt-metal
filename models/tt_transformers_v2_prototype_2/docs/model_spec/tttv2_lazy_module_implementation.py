"""
TTTv2 Lazy Module Implementation Design

This module demonstrates a practical implementation of lazy module construction
specifically designed for TTTv2's needs with 100+ models.
"""

import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import psutil


# Resource tracking
@dataclass
class ResourceRequirements:
    """Estimated resource requirements for a module"""

    memory_mb: float
    compute_flops: float
    params_count: int
    activation_memory_mb: float
    supports_device_types: List[str]


# Base interface for all TTTv2 modules
class TTTModule(ABC):
    """Base interface that all TTT modules implement"""

    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    @abstractmethod
    def get_resource_requirements(self) -> ResourceRequirements:
        pass


# Lazy module wrapper
class LazyModule:
    """
    Lazy wrapper for TTTv2 modules that defers materialization.

    Key features:
    - Lightweight metadata storage
    - Resource estimation without materialization
    - Thread-safe materialization
    - Memory pressure awareness
    """

    # Class-level memory tracker
    _memory_tracker = {"allocated": 0, "limit": psutil.virtual_memory().total * 0.8}  # 80% of system memory
    _lock = threading.Lock()

    def __init__(
        self, module_class: Type[TTTModule], config: Dict[str, Any], name: Optional[str] = None, priority: int = 0
    ):
        """
        Initialize lazy module.

        Args:
            module_class: The TTT module class to instantiate
            config: Configuration dictionary for the module
            name: Optional name for tracking
            priority: Priority for eviction (higher = keep longer)
        """
        self.module_class = module_class
        self.config = config
        self.name = name or f"{module_class.__name__}_{id(self)}"
        self.priority = priority

        # Lazy state
        self._module: Optional[TTTModule] = None
        self._resource_requirements: Optional[ResourceRequirements] = None
        self._materialized = False
        self._device = None

        # Estimate resources without creating module
        self._resource_requirements = self._estimate_resources()

    def _estimate_resources(self) -> ResourceRequirements:
        """Estimate resources without creating the module"""
        # Module classes can provide static estimation method
        if hasattr(self.module_class, "estimate_resources"):
            return self.module_class.estimate_resources(self.config)

        # Default estimation based on common patterns
        hidden_dim = self.config.get("hidden_dim", 0)
        num_heads = self.config.get("num_heads", 0)

        # Simple heuristics for transformer modules
        if "attention" in self.module_class.__name__.lower():
            params = 4 * hidden_dim * hidden_dim  # Q, K, V, O projections
            memory_mb = params * 4 / 1e6  # 4 bytes per param
            flops = 2 * params  # Rough estimate
        elif "ffn" in self.module_class.__name__.lower():
            intermediate_dim = self.config.get("intermediate_dim", 4 * hidden_dim)
            params = 2 * hidden_dim * intermediate_dim + intermediate_dim * hidden_dim
            memory_mb = params * 4 / 1e6
            flops = 2 * params
        else:
            # Generic fallback
            params = hidden_dim * hidden_dim
            memory_mb = params * 4 / 1e6
            flops = params

        return ResourceRequirements(
            memory_mb=memory_mb,
            compute_flops=flops,
            params_count=params,
            activation_memory_mb=memory_mb * 0.1,  # 10% for activations
            supports_device_types=["cpu", "cuda", "ttnn"],
        )

    @property
    def is_materialized(self) -> bool:
        """Check if module is materialized"""
        return self._materialized

    @property
    def resource_requirements(self) -> ResourceRequirements:
        """Get resource requirements without materialization"""
        return self._resource_requirements

    def can_materialize(self, device: str = "cpu") -> bool:
        """Check if module can be materialized given current resources"""
        required_memory = self._resource_requirements.memory_mb * 1e6

        with self._lock:
            available_memory = self._memory_tracker["limit"] - self._memory_tracker["allocated"]
            return available_memory >= required_memory

    def materialize(self, device: Optional[str] = None, force: bool = False) -> TTTModule:
        """
        Materialize the module.

        Args:
            device: Device to materialize on
            force: Force materialization even if memory is low

        Returns:
            Materialized module

        Raises:
            MemoryError: If insufficient memory and force=False
        """
        if self._materialized and self._module is not None:
            return self._module

        device = device or self._device or "cpu"
        required_memory = self._resource_requirements.memory_mb * 1e6

        # Check memory availability
        if not force and not self.can_materialize(device):
            raise MemoryError(
                f"Insufficient memory to materialize {self.name}. " f"Required: {required_memory / 1e9:.2f}GB"
            )

        with self._lock:
            # Double-check after acquiring lock
            if self._materialized and self._module is not None:
                return self._module

            # Create module
            self._module = self.module_class(**self.config, device=device)
            self._materialized = True
            self._device = device

            # Update memory tracking
            self._memory_tracker["allocated"] += required_memory

        return self._module

    def release(self):
        """Release the materialized module to free memory"""
        if self._materialized and self._module is not None:
            with self._lock:
                required_memory = self._resource_requirements.memory_mb * 1e6
                self._memory_tracker["allocated"] -= required_memory

                # Clear module
                del self._module
                self._module = None
                self._materialized = False

    def __call__(self, *args, **kwargs):
        """Forward pass with automatic materialization"""
        if not self._materialized:
            self.materialize()
        return self._module(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to materialized module"""
        if name.startswith("_"):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

        # Materialize if accessing module methods
        if not self._materialized:
            self.materialize()

        return getattr(self._module, name)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize lazy module specification"""
        return {
            "module_class": f"{self.module_class.__module__}.{self.module_class.__name__}",
            "config": self.config,
            "name": self.name,
            "priority": self.priority,
            "resource_requirements": {
                "memory_mb": self._resource_requirements.memory_mb,
                "params_count": self._resource_requirements.params_count,
            },
        }


# Lazy model composition
class LazyModel:
    """
    Compose multiple lazy modules into a model.

    Features:
    - Partial materialization
    - Memory-aware scheduling
    - Priority-based eviction
    """

    def __init__(self, name: str):
        self.name = name
        self.modules: Dict[str, LazyModule] = {}
        self._materialization_order: List[str] = []

    def add_module(self, name: str, module: LazyModule):
        """Add a lazy module to the model"""
        self.modules[name] = module
        self._materialization_order.append(name)

    def get_total_resources(self) -> ResourceRequirements:
        """Calculate total resource requirements"""
        total_memory = 0
        total_flops = 0
        total_params = 0

        for module in self.modules.values():
            req = module.resource_requirements
            total_memory += req.memory_mb
            total_flops += req.compute_flops
            total_params += req.params_count

        return ResourceRequirements(
            memory_mb=total_memory,
            compute_flops=total_flops,
            params_count=total_params,
            activation_memory_mb=total_memory * 0.1,
            supports_device_types=["cpu", "cuda", "ttnn"],
        )

    def materialize_partial(self, module_names: List[str], device: str = "cpu") -> Dict[str, TTTModule]:
        """Materialize only specific modules"""
        materialized = {}

        for name in module_names:
            if name in self.modules:
                module = self.modules[name].materialize(device)
                materialized[name] = module

        return materialized

    def materialize_all(self, device: str = "cpu", max_memory_mb: Optional[float] = None):
        """Materialize all modules with memory limit"""
        if max_memory_mb is None:
            max_memory_mb = LazyModule._memory_tracker["limit"] / 1e6

        materialized_memory = 0
        for name in self._materialization_order:
            module = self.modules[name]
            req_memory = module.resource_requirements.memory_mb

            if materialized_memory + req_memory <= max_memory_mb:
                module.materialize(device)
                materialized_memory += req_memory
            else:
                print(f"Warning: Skipping {name} due to memory limit")

    def release_modules(self, module_names: List[str]):
        """Release specific modules to free memory"""
        for name in module_names:
            if name in self.modules:
                self.modules[name].release()


# Builder pattern for lazy models
class LazyModelBuilder:
    """Fluent interface for building lazy models"""

    def __init__(self, name: str):
        self.model = LazyModel(name)
        self._current_layer_idx = 0

    def add_attention(self, **config) -> "LazyModelBuilder":
        """Add attention layer"""
        from tt_transformers_v2.attention import MultiHeadAttention

        name = f"layer_{self._current_layer_idx}_attention"
        self.model.add_module(name, LazyModule(MultiHeadAttention, config, name=name, priority=2))
        return self

    def add_ffn(self, **config) -> "LazyModelBuilder":
        """Add FFN layer"""
        from tt_transformers_v2.ffn import SwiGLU

        name = f"layer_{self._current_layer_idx}_ffn"
        self.model.add_module(name, LazyModule(SwiGLU, config, name=name, priority=1))
        return self

    def add_norm(self, **config) -> "LazyModelBuilder":
        """Add normalization layer"""
        from tt_transformers_v2.normalization import RMSNorm

        name = f"layer_{self._current_layer_idx}_norm"
        self.model.add_module(name, LazyModule(RMSNorm, config, name=name, priority=0))
        return self

    def next_layer(self) -> "LazyModelBuilder":
        """Move to next layer"""
        self._current_layer_idx += 1
        return self

    def build(self) -> LazyModel:
        """Build the lazy model"""
        return self.model


# Memory pressure manager
class MemoryPressureManager:
    """
    Manage memory pressure across multiple lazy models.

    Features:
    - Global memory tracking
    - Automatic eviction
    - Priority-based retention
    """

    def __init__(self, memory_limit_mb: Optional[float] = None):
        self.memory_limit_mb = memory_limit_mb or (psutil.virtual_memory().total / 1e6 * 0.8)
        self.models: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._lock = threading.Lock()

    def register_model(self, model: LazyModel):
        """Register a model for memory management"""
        self.models[model.name] = model

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage by model"""
        usage = {}

        for name, model in self.models.items():
            model_memory = 0
            for module_name, module in model.modules.items():
                if module.is_materialized:
                    model_memory += module.resource_requirements.memory_mb

            usage[name] = model_memory

        return usage

    def evict_if_needed(self, required_mb: float) -> bool:
        """Evict modules if needed to free memory"""
        current_usage = sum(self.get_memory_usage().values())

        if current_usage + required_mb <= self.memory_limit_mb:
            return True

        # Find modules to evict (lowest priority first)
        eviction_candidates = []
        for model in self.models.values():
            for name, module in model.modules.items():
                if module.is_materialized:
                    eviction_candidates.append((module.priority, name, module))

        # Sort by priority (ascending)
        eviction_candidates.sort(key=lambda x: x[0])

        # Evict until enough memory is available
        freed_memory = 0
        needed_memory = current_usage + required_mb - self.memory_limit_mb

        for priority, name, module in eviction_candidates:
            if freed_memory >= needed_memory:
                break

            module.release()
            freed_memory += module.resource_requirements.memory_mb

        return freed_memory >= needed_memory


# Example usage
def demo_lazy_modules():
    """Demonstrate lazy module usage"""

    # Create a lazy model
    builder = LazyModelBuilder("llama-70b")

    # Build model with 80 layers
    for i in range(80):
        builder.add_attention(hidden_dim=8192, num_heads=64, num_kv_heads=8).add_ffn(
            hidden_dim=8192, intermediate_dim=28672
        ).add_norm(hidden_dim=8192).next_layer()

    model = builder.build()

    # Check total resources without materialization
    total_resources = model.get_total_resources()
    print(f"Total model size: {total_resources.memory_mb / 1024:.2f} GB")
    print(f"Total parameters: {total_resources.params_count / 1e9:.2f}B")

    # Materialize only first few layers
    model.materialize_partial(["layer_0_attention", "layer_0_ffn", "layer_0_norm"])

    # Use memory pressure manager
    manager = MemoryPressureManager(memory_limit_mb=32000)  # 32GB limit
    manager.register_model(model)

    # Check if we can materialize more
    can_materialize = manager.evict_if_needed(1000)  # Need 1GB
    print(f"Can materialize 1GB more: {can_materialize}")


if __name__ == "__main__":
    demo_lazy_modules()
