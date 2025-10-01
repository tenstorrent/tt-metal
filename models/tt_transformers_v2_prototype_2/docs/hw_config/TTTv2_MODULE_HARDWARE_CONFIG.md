# Module-Centric Hardware Configuration Design

## Core Idea

Each building block module is responsible for providing its own hardware-specific default configurations, making the code more maintainable and easier to understand.

## Design

### 1. Module Hardware Configuration Interface

```python
# tt_transformers_v2/src/building_blocks/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModuleHardwareConfig:
    """Hardware configuration for a specific module."""
    precision: Dict[str, str]  # Weight and activation precisions
    compute: Dict[str, Any]    # Compute kernel configurations
    memory: Dict[str, Any]     # Memory placement and sharding
    device_overrides: Dict[str, Dict[str, Any]] = None  # Device-specific overrides

class HardwareConfigurableModule(ABC):
    """Base class for modules that provide hardware configurations."""

    @classmethod
    @abstractmethod
    def get_default_hw_config(
        cls,
        device_type: str,
        mode: str = "performance",
        model_size: Optional[int] = None,
        **kwargs
    ) -> ModuleHardwareConfig:
        """
        Get default hardware configuration for this module.

        Args:
            device_type: Device name (e.g., "N150", "T3K")
            mode: "performance" or "accuracy"
            model_size: Optional model size for size-based optimizations
            **kwargs: Additional module-specific parameters

        Returns:
            Module-specific hardware configuration
        """
        pass
```

### 2. Example: MultiHeadAttention with Built-in Hardware Config

```python
# tt_transformers_v2/src/building_blocks/attention/mha.py
import ttnn
from ..base import HardwareConfigurableModule, ModuleHardwareConfig

class MultiHeadAttention(HardwareConfigurableModule):
    """Multi-head attention with integrated hardware configuration knowledge."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        device: Optional[Any] = None,
        hw_config: Optional[ModuleHardwareConfig] = None,
        hw_mode: str = "performance",
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.device = device

        # Use provided config or get defaults for this module
        if hw_config is None and device is not None:
            device_type = self._detect_device_type(device)
            hw_config = self.get_default_hw_config(
                device_type=device_type,
                mode=hw_mode,
                hidden_dim=hidden_dim,
                num_heads=num_heads
            )

        self.hw_config = hw_config
        self._init_weights()

    @classmethod
    def get_default_hw_config(
        cls,
        device_type: str,
        mode: str = "performance",
        model_size: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        **kwargs
    ) -> ModuleHardwareConfig:
        """
        Get attention-specific hardware configuration.

        This method encapsulates all the attention-specific hardware
        optimization knowledge from TTTv1.
        """
        # Base configuration for attention
        base_config = ModuleHardwareConfig(
            precision={
                "qkv_weights": "BFP8",
                "o_weights": "BFP8",
                "activations": "BF16",
                "kv_cache": "BFP8",
            },
            compute={
                "qkv_matmul": {
                    "math_fidelity": "HiFi2",
                    "fp32_dest_acc_en": True,
                    "packer_l1_acc": True,
                },
                "attn_scores": {
                    "math_fidelity": "HiFi4",  # Higher fidelity for attention scores
                    "fp32_dest_acc_en": True,
                },
                "attn_output": {
                    "math_fidelity": "HiFi2",
                    "fp32_dest_acc_en": True,
                },
            },
            memory={
                "qkv_weights_placement": "DRAM",
                "o_weights_placement": "DRAM",
                "kv_cache_placement": "L1" if device_type == "N150" else "DRAM",
                "use_width_sharding": True,
            }
        )

        # Apply mode-specific adjustments
        if mode == "accuracy":
            # Accuracy mode: use higher precision
            base_config.precision.update({
                "qkv_weights": "BF16",
                "o_weights": "BF16",
                "kv_cache": "BF16",
            })
            base_config.compute["attn_scores"]["math_fidelity"] = "HiFi4"

        # Device-specific optimizations
        device_overrides = {
            "N150": {
                # N150 has limited L1, adjust accordingly
                "memory": {
                    "kv_cache_placement": "L1" if hidden_dim and hidden_dim <= 2048 else "DRAM",
                },
                "compute": {
                    "qkv_matmul": {"fp32_dest_acc_en": False},  # Save L1 on N150
                }
            },
            "T3K": {
                # T3K can handle larger KV cache in L1
                "memory": {
                    "kv_cache_placement": "L1",
                    "use_height_sharding": True,  # T3K benefits from 2D sharding
                },
            },
            "TG": {
                # Galaxy-specific optimizations
                "precision": {
                    "kv_cache": "BFP8",  # Always use BFP8 KV cache on Galaxy
                },
                "memory": {
                    "use_all_gather": True,
                },
            },
        }

        # Model size specific adjustments (e.g., for 70B+ models)
        if model_size and model_size > 70e9:
            # Large models need different precision trade-offs
            if mode == "accuracy":
                # Even in accuracy mode, large models use BFP8 attention
                base_config.precision["qkv_weights"] = "BFP8"
                base_config.precision["kv_cache"] = "BFP8"

        # Apply device-specific overrides
        if device_type in device_overrides:
            overrides = device_overrides[device_type]
            for category, settings in overrides.items():
                if category in base_config.__dict__:
                    base_config.__dict__[category].update(settings)

        base_config.device_overrides = device_overrides
        return base_config

    def _init_weights(self):
        """Initialize weights using hardware configuration."""
        if self.hw_config is None:
            # Fallback to default types
            weight_dtype = ttnn.bfloat16
        else:
            # Get dtype from hardware config
            weight_dtype = self._get_ttnn_dtype(
                self.hw_config.precision.get("qkv_weights", "BF16")
            )

        # Initialize QKV weights
        self.wq = self._create_parameter(
            (self.hidden_dim, self.hidden_dim),
            dtype=weight_dtype,
            placement=self.hw_config.memory.get("qkv_weights_placement", "DRAM")
        )
        # ... initialize other weights

    def forward(self, x, start_pos=0, mask=None, mode="decode"):
        """Forward pass using hardware-optimized operations."""
        # Get compute configs for this operation
        qkv_compute_config = self._get_compute_kernel_config(
            self.hw_config.compute.get("qkv_matmul", {})
        ) if self.hw_config else None

        # QKV projection with hardware-specific compute config
        q = ttnn.matmul(x, self.wq, compute_kernel_config=qkv_compute_config)
        # ... rest of attention computation
```

### 3. Example: SwiGLU FFN with Built-in Hardware Config

```python
# tt_transformers_v2/src/building_blocks/ffn/gated_mlp.py
class SwiGLU(HardwareConfigurableModule):
    """SwiGLU FFN with integrated hardware configuration."""

    @classmethod
    def get_default_hw_config(
        cls,
        device_type: str,
        mode: str = "performance",
        model_size: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        intermediate_dim: Optional[int] = None,
        **kwargs
    ) -> ModuleHardwareConfig:
        """
        FFN-specific hardware configuration.

        Encapsulates FFN optimization knowledge:
        - FFNs are typically compute-bound, benefit from lower precision
        - W1/W3 can often fuse with activation functions
        - Different patterns for prefill vs decode
        """
        base_config = ModuleHardwareConfig(
            precision={
                "w1_weights": "BFP4" if mode == "performance" else "BFP8",
                "w2_weights": "BFP8",
                "w3_weights": "BFP4" if mode == "performance" else "BFP8",
                "activations": "BF16",
            },
            compute={
                "w1_matmul": {
                    "math_fidelity": "LoFi" if mode == "performance" else "HiFi2",
                    "fp32_dest_acc_en": False,  # FFN can use lower precision accumulation
                    "fused_activation": "SILU",  # Fuse SiLU with W1
                },
                "w3_matmul": {
                    "math_fidelity": "LoFi" if mode == "performance" else "HiFi2",
                    "fp32_dest_acc_en": False,
                },
                "w2_matmul": {
                    "math_fidelity": "HiFi2",  # Output projection needs higher fidelity
                    "fp32_dest_acc_en": True,
                },
            },
            memory={
                "weights_placement": "DRAM",
                "activations_placement": "L1",
                "use_width_sharding": True,
                "gate_ffn_fusion": True,  # Fuse W1/W3 weights storage
            }
        )

        # Model-specific adjustments
        if model_size and model_size > 70e9:
            # Large models always use BFP4 for W1/W3, even in accuracy mode
            base_config.precision["w1_weights"] = "BFP4"
            base_config.precision["w3_weights"] = "BFP4"
            base_config.compute["w1_matmul"]["math_fidelity"] = "LoFi"
            base_config.compute["w3_matmul"]["math_fidelity"] = "LoFi"

        # Device-specific optimizations
        if device_type == "N150" and hidden_dim and hidden_dim > 8192:
            # Large FFNs on N150 need special handling
            base_config.memory["use_height_sharding"] = True

        return base_config
```

### 4. Integration with AutoHardwareConfig

```python
# tt_transformers_v2/src/hardware/auto_config.py
from typing import Dict, Any

class AutoHardwareConfig:
    """Enhanced auto configuration that leverages module defaults."""

    @staticmethod
    def from_model_spec(
        model_spec: ModelSpec,
        device: str,
        mode: str = "performance"
    ) -> Dict[str, ModuleHardwareConfig]:
        """
        Create hardware configs for all modules in a model spec.

        Returns:
            Dictionary mapping module types to their hardware configs
        """
        configs = {}
        model_size = AutoHardwareConfig._estimate_model_size(model_spec)

        # Get attention config
        configs["attention"] = MultiHeadAttention.get_default_hw_config(
            device_type=device,
            mode=mode,
            model_size=model_size,
            hidden_dim=model_spec.hidden_dim,
            num_heads=model_spec.num_heads
        )

        # Get FFN config
        configs["ffn"] = SwiGLU.get_default_hw_config(
            device_type=device,
            mode=mode,
            model_size=model_size,
            hidden_dim=model_spec.hidden_dim,
            intermediate_dim=model_spec.intermediate_dim
        )

        # Get normalization config
        configs["norm"] = RMSNorm.get_default_hw_config(
            device_type=device,
            mode=mode,
            hidden_dim=model_spec.hidden_dim
        )

        return configs
```

### 5. Usage Examples

```python
# Example 1: Module with automatic hardware config
attention = MultiHeadAttention(
    hidden_dim=4096,
    num_heads=32,
    device=ttnn_device,  # Hardware config auto-determined
)

# Example 2: Module with explicit mode
attention = MultiHeadAttention(
    hidden_dim=4096,
    num_heads=32,
    device=ttnn_device,
    hw_mode="accuracy"  # Use accuracy presets
)

# Example 3: Query module for its optimal config
config = MultiHeadAttention.get_default_hw_config(
    device_type="T3K",
    mode="performance",
    hidden_dim=4096,
    num_heads=32
)
print(f"Recommended QKV precision: {config.precision['qkv_weights']}")
print(f"Recommended KV cache placement: {config.memory['kv_cache_placement']}")

# Example 4: Model factory using module defaults
class ModelFactoryEnhanced:
    @staticmethod
    def from_spec(spec: ModelSpec, device: str, mode: str = "performance"):
        # Each module gets its own optimized config
        attn = MultiHeadAttention(
            hidden_dim=spec.hidden_dim,
            num_heads=spec.num_heads,
            device=device,
            hw_mode=mode
        )

        ffn = SwiGLU(
            hidden_dim=spec.hidden_dim,
            intermediate_dim=spec.intermediate_dim,
            device=device,
            hw_mode=mode
        )

        # Modules handle their own hardware optimization
        return Model(attention=attn, ffn=ffn, ...)
```

## Benefits of Module-Centric Approach

### 1. **Locality of Knowledge**
Hardware optimization knowledge lives with the module that uses it:
```python
# In attention module - all attention-specific optimizations in one place
class MultiHeadAttention:
    @classmethod
    def get_default_hw_config(cls, ...):
        # Attention needs HiFi4 for scores computation
        # KV cache placement depends on model size
        # QKV can use BFP8 for most models
        # ... all attention-specific knowledge here
```

### 2. **Discoverability**
Developers can easily find hardware configs by looking at the module:
```python
# Want to know FFN hardware configs? Look at FFN module!
help(SwiGLU.get_default_hw_config)
# Shows all FFN-specific optimizations and rationale
```

### 3. **Modularity**
Each module can be tested with its hardware configs independently:
```python
# Test attention with different hardware configs
for device in ["N150", "T3K", "TG"]:
    config = MultiHeadAttention.get_default_hw_config(device)
    # Test module with this config
```

### 4. **Progressive Learning**
New developers can understand one module at a time:
- Start with understanding MultiHeadAttention
- Its hardware configs are right there
- No need to understand global configuration system first

### 5. **Easy Extension**
Adding a new module with hardware configs is self-contained:
```python
class FlashAttention(HardwareConfigurableModule):
    @classmethod
    def get_default_hw_config(cls, device_type: str, ...):
        # Flash attention specific optimizations
        # All knowledge about flash attention hardware in one place
```

## Comparison with Centralized Approach

### Centralized (Original Design)
```python
# Hardware configs in separate module
DevicePresets.PRESETS[DeviceType.N150][PrecisionMode.ACCURACY] = {
    "attention": {...},
    "ffn": {...},
    # All configs in one place
}
```

### Module-Centric (New Design)
```python
# Hardware configs with the module
class MultiHeadAttention:
    @classmethod
    def get_default_hw_config(cls, device_type="N150", mode="accuracy"):
        # Attention-specific configs defined here
```

## Migration Path

Existing modules can gradually adopt this pattern:
```python
# Phase 1: Add get_default_hw_config to modules
# Phase 2: Update ModelFactory to use module defaults
# Phase 3: Deprecate centralized config (keep as fallback)
```

This module-centric approach provides better code organization while maintaining the simplicity of the API and all the benefits of automatic hardware configuration.
