# TTTv2 Hardware Configuration Design

## Motivation

Decouple hardware-specific configurations from model specifications to:
1. Provide smart defaults based on TTTv1 learnings
2. Allow minimal code for standard configurations
3. Enable advanced tuning when needed
4. Support multiple device types with single model specification

## Key Patterns from TTTv1 Analysis

### 1. Device-Specific Configurations
- Different max prefill chunk sizes per model/device combo
- Device-specific core grids and memory layouts
- Precision/fidelity settings that vary by device capabilities

### 2. Common Configuration Categories
- **Precision Settings**: BFP4, BFP8, BF16 for different tensor groups
- **Math Fidelity**: LoFi, HiFi2, HiFi4 with various options
- **Memory Configs**: DRAM vs L1, sharded vs interleaved
- **Compute Kernels**: Different configs for different operations
- **Parallelism**: Tensor parallel, pipeline parallel settings

### 3. Model-Specific Overrides
- 70B+ models use BFP4 MLPs even in accuracy mode
- Specific models need different attention precision
- Per-layer configurations (e.g., decoder-specific settings)

## Proposed Hardware Configuration API

### 1. Device Presets

```python
# tt_transformers_v2/src/hardware/device_presets.py
from dataclasses import dataclass
from typing import Dict, Optional, Any
from enum import Enum

class DeviceType(Enum):
    N150 = "N150"
    N300 = "N300"
    T3K = "T3K"
    TG = "TG"
    P100 = "P100"
    P150 = "P150"
    P300 = "P300"
    P150x4 = "P150x4"

class PrecisionMode(Enum):
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    CUSTOM = "custom"

@dataclass
class ComputeConfig:
    """Compute kernel configuration."""
    math_fidelity: str = "HiFi2"
    fp32_dest_acc_en: bool = True
    packer_l1_acc: bool = True
    math_approx_mode: bool = False

@dataclass
class PrecisionConfig:
    """Precision settings for different tensor groups."""
    mlp_weights: str = "BFP8"
    mlp_activations: str = "BF16"
    attention_weights: str = "BFP8"
    attention_activations: str = "BF16"
    kv_cache: str = "BFP8"

@dataclass
class MemoryConfig:
    """Memory placement and sharding configuration."""
    weights_placement: str = "DRAM"
    activations_placement: str = "L1"
    kv_cache_placement: str = "DRAM"
    use_width_sharding: bool = True
    use_height_sharding: bool = False

@dataclass
class ParallelismConfig:
    """Parallelism configuration."""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel: bool = False

@dataclass
class DeviceConfig:
    """Complete device configuration."""
    device_type: DeviceType
    precision: PrecisionConfig
    compute: Dict[str, ComputeConfig]  # Per-operation compute configs
    memory: MemoryConfig
    parallelism: ParallelismConfig

    # Device-specific limits
    max_prefill_chunk_size: int
    max_batch_size: int
    l1_memory_size: int
    dram_bandwidth: float

    # Optional overrides
    custom_configs: Dict[str, Any] = None

class DevicePresets:
    """Pre-configured device settings based on TTTv1 learnings."""

    # Default precision configs
    ACCURACY_PRECISION = PrecisionConfig(
        mlp_weights="BF16",
        mlp_activations="BF16",
        attention_weights="BF16",
        attention_activations="BF16",
        kv_cache="BF16"
    )

    PERFORMANCE_PRECISION = PrecisionConfig(
        mlp_weights="BFP4",
        mlp_activations="BF16",
        attention_weights="BFP8",
        attention_activations="BF16",
        kv_cache="BFP8"
    )

    # Device-specific presets
    PRESETS = {
        DeviceType.N150: {
            PrecisionMode.ACCURACY: DeviceConfig(
                device_type=DeviceType.N150,
                precision=ACCURACY_PRECISION,
                compute={
                    "mlp": ComputeConfig(math_fidelity="HiFi2", fp32_dest_acc_en=False),
                    "attention": ComputeConfig(math_fidelity="HiFi4"),
                    "sdpa": ComputeConfig(math_fidelity="HiFi4"),
                },
                memory=MemoryConfig(
                    weights_placement="DRAM",
                    activations_placement="L1",
                    kv_cache_placement="L1",
                    use_width_sharding=True
                ),
                parallelism=ParallelismConfig(tensor_parallel_size=1),
                max_prefill_chunk_size=4096,
                max_batch_size=32,
                l1_memory_size=1499136,  # 1.5MB
                dram_bandwidth=384.0,    # GB/s
            ),
            PrecisionMode.PERFORMANCE: DeviceConfig(
                device_type=DeviceType.N150,
                precision=PERFORMANCE_PRECISION,
                compute={
                    "mlp": ComputeConfig(math_fidelity="LoFi"),
                    "attention": ComputeConfig(math_fidelity="HiFi2"),
                    "sdpa": ComputeConfig(math_fidelity="HiFi2"),
                },
                memory=MemoryConfig(
                    weights_placement="DRAM",
                    activations_placement="L1",
                    kv_cache_placement="DRAM",
                    use_width_sharding=True
                ),
                parallelism=ParallelismConfig(tensor_parallel_size=1),
                max_prefill_chunk_size=4096,
                max_batch_size=32,
                l1_memory_size=1499136,
                dram_bandwidth=384.0,
            ),
        },
        DeviceType.T3K: {
            PrecisionMode.ACCURACY: DeviceConfig(
                device_type=DeviceType.T3K,
                precision=ACCURACY_PRECISION,
                compute={
                    "mlp": ComputeConfig(math_fidelity="HiFi2", fp32_dest_acc_en=True),
                    "attention": ComputeConfig(math_fidelity="HiFi4"),
                    "sdpa": ComputeConfig(math_fidelity="HiFi4"),
                },
                memory=MemoryConfig(
                    weights_placement="DRAM",
                    activations_placement="L1",
                    kv_cache_placement="L1",
                    use_width_sharding=True
                ),
                parallelism=ParallelismConfig(tensor_parallel_size=8),
                max_prefill_chunk_size=32768,
                max_batch_size=32,
                l1_memory_size=1499136,
                dram_bandwidth=384.0 * 8,  # 8 devices
            ),
            PrecisionMode.PERFORMANCE: DeviceConfig(
                device_type=DeviceType.T3K,
                precision=PERFORMANCE_PRECISION,
                compute={
                    "mlp": ComputeConfig(math_fidelity="LoFi"),
                    "attention": ComputeConfig(math_fidelity="HiFi2"),
                    "sdpa": ComputeConfig(math_fidelity="HiFi2"),
                },
                memory=MemoryConfig(
                    weights_placement="DRAM",
                    activations_placement="L1",
                    kv_cache_placement="DRAM",
                    use_width_sharding=True
                ),
                parallelism=ParallelismConfig(tensor_parallel_size=8),
                max_prefill_chunk_size=32768,
                max_batch_size=32,
                l1_memory_size=1499136,
                dram_bandwidth=384.0 * 8,
            ),
        },
        # Add more device presets...
    }

    @classmethod
    def get_config(
        cls,
        device: str,
        mode: PrecisionMode = PrecisionMode.PERFORMANCE
    ) -> DeviceConfig:
        """Get device configuration with smart defaults."""
        device_type = DeviceType(device)

        if device_type not in cls.PRESETS:
            raise ValueError(f"No preset for device {device}")

        if mode not in cls.PRESETS[device_type]:
            raise ValueError(f"No {mode.value} preset for device {device}")

        return cls.PRESETS[device_type][mode]
```

### 2. Model-Specific Overrides

```python
# tt_transformers_v2/src/hardware/model_overrides.py
from typing import Dict, Optional
from .device_presets import DeviceConfig, PrecisionConfig

class ModelHardwareOverrides:
    """Model-specific hardware configuration overrides."""

    # Model size thresholds
    LARGE_MODEL_THRESHOLD = 70e9  # 70B parameters

    @staticmethod
    def apply_model_overrides(
        config: DeviceConfig,
        model_spec: "ModelSpec"
    ) -> DeviceConfig:
        """Apply model-specific overrides to device config."""

        # Calculate approximate model size
        model_size = model_spec.vocab_size * model_spec.hidden_dim
        model_size += model_spec.num_layers * (
            4 * model_spec.hidden_dim * model_spec.hidden_dim +  # MLP
            4 * model_spec.hidden_dim * model_spec.hidden_dim    # Attention
        )

        # Large models (70B+) use BFP4 MLPs even in accuracy mode
        if model_size > ModelHardwareOverrides.LARGE_MODEL_THRESHOLD:
            config.precision.mlp_weights = "BFP4"
            config.compute["mlp"].math_fidelity = "LoFi"

        # Llama 3 models can use BFP8 attention in accuracy mode
        if "llama-3" in model_spec._metadata.get("family", "").lower():
            if config.precision.attention_weights == "BF16":
                config.precision.attention_weights = "BFP8"
                config.precision.kv_cache = "BFP8"

        # Model-specific prefill chunk size adjustments
        prefill_overrides = {
            "Llama-3.1-8B": {"N150": 4096, "N300": 65536, "T3K": 131072},
            "Llama-3.1-70B": {"T3K": 32768, "TG": 131072},
            # Add more model-specific overrides...
        }

        model_name = model_spec._metadata.get("name", "")
        if model_name in prefill_overrides:
            device_name = config.device_type.value
            if device_name in prefill_overrides[model_name]:
                config.max_prefill_chunk_size = prefill_overrides[model_name][device_name]

        return config
```

### 3. Simplified API for Model Developers

```python
# tt_transformers_v2/src/hardware/auto_config.py
from typing import Optional, Union
from .device_presets import DevicePresets, PrecisionMode, DeviceConfig
from .model_overrides import ModelHardwareOverrides

class AutoHardwareConfig:
    """Automatic hardware configuration with minimal user input."""

    @staticmethod
    def from_device(
        device: str,
        mode: Union[str, PrecisionMode] = "performance",
        model_spec: Optional["ModelSpec"] = None
    ) -> DeviceConfig:
        """
        Create hardware config with smart defaults.

        Args:
            device: Device name (e.g., "N150", "T3K")
            mode: "performance" or "accuracy"
            model_spec: Optional model specification for model-specific tuning

        Returns:
            Complete hardware configuration

        Example:
            # Minimal configuration - just specify device
            hw_config = AutoHardwareConfig.from_device("N150")

            # With accuracy mode
            hw_config = AutoHardwareConfig.from_device("T3K", mode="accuracy")

            # With model-specific optimizations
            hw_config = AutoHardwareConfig.from_device(
                "T3K",
                mode="performance",
                model_spec=model_spec
            )
        """
        if isinstance(mode, str):
            mode = PrecisionMode(mode)

        # Get base configuration
        config = DevicePresets.get_config(device, mode)

        # Apply model-specific overrides if model spec provided
        if model_spec is not None:
            config = ModelHardwareOverrides.apply_model_overrides(config, model_spec)

        return config

    @staticmethod
    def custom(
        device: str,
        precision: Optional[Dict[str, str]] = None,
        compute: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> DeviceConfig:
        """
        Create custom hardware configuration.

        Example:
            hw_config = AutoHardwareConfig.custom(
                device="N150",
                precision={"mlp_weights": "BFP8", "attention_weights": "BF16"},
                compute={"mlp": {"math_fidelity": "HiFi4"}},
                max_prefill_chunk_size=8192
            )
        """
        # Start with performance defaults
        config = DevicePresets.get_config(device, PrecisionMode.PERFORMANCE)

        # Override precision settings
        if precision:
            for key, value in precision.items():
                setattr(config.precision, key, value)

        # Override compute settings
        if compute:
            for op, settings in compute.items():
                if op in config.compute:
                    for key, value in settings.items():
                        setattr(config.compute[op], key, value)

        # Override other settings
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config
```

### 4. Integration with Building Blocks

```python
# Example: How building blocks use hardware config
from tt_transformers_v2.hardware import AutoHardwareConfig

class MultiHeadAttention:
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        device: Optional[Any] = None,
        hw_config: Optional[DeviceConfig] = None
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Auto-detect hardware config if not provided
        if hw_config is None and device is not None:
            device_name = self._detect_device_name(device)
            hw_config = AutoHardwareConfig.from_device(device_name)

        self.hw_config = hw_config

        # Use hardware config for initialization
        self._init_weights()

    def _init_weights(self):
        # Use precision from hardware config
        weight_dtype = self._get_ttnn_dtype(self.hw_config.precision.attention_weights)

        # Initialize weights with proper dtype
        self.wq = self._create_parameter(
            shape=(self.hidden_dim, self.hidden_dim),
            dtype=weight_dtype
        )

    def forward(self, x):
        # Use compute config for operations
        compute_config = self.hw_config.compute.get("attention")

        # Apply compute kernel config
        q = ttnn.matmul(
            x,
            self.wq,
            compute_kernel_config=self._get_compute_kernel_config(compute_config)
        )
        # ...
```

## Usage Examples

### 1. Minimal Configuration

```python
from tt_transformers_v2 import ModelFactory, ModelSpec
from tt_transformers_v2.hardware import AutoHardwareConfig

# Just specify device - everything else is automatic
model_spec = ModelSpec.uniform(
    vocab_size=32000,
    hidden_dim=4096,
    num_layers=32,
    num_heads=32,
    intermediate_dim=11008
)

# Automatic hardware config
model = ModelFactory.from_spec(
    model_spec,
    device="N150"  # Just device name!
)
```

### 2. With Mode Selection

```python
# High accuracy mode
model = ModelFactory.from_spec(
    model_spec,
    device="T3K",
    hw_mode="accuracy"  # Use accuracy presets
)

# Or explicitly create config
hw_config = AutoHardwareConfig.from_device("T3K", mode="accuracy")
model = ModelFactory.from_spec(model_spec, hw_config=hw_config)
```

### 3. Model-Aware Configuration

```python
# Hardware config automatically adjusts for model
hw_config = AutoHardwareConfig.from_device(
    "T3K",
    mode="performance",
    model_spec=model_spec  # 70B model gets BFP4 MLPs automatically
)

model = ModelFactory.from_spec(model_spec, hw_config=hw_config)
```

### 4. Custom Configuration

```python
# Fine-grained control when needed
hw_config = AutoHardwareConfig.custom(
    device="N150",
    precision={
        "mlp_weights": "BFP8",      # Override MLP precision
        "attention_weights": "BF16"   # Keep attention in BF16
    },
    compute={
        "mlp": {"math_fidelity": "HiFi4"}  # Use HiFi4 for MLP
    },
    max_prefill_chunk_size=8192  # Custom prefill size
)

model = ModelFactory.from_spec(model_spec, hw_config=hw_config)
```

## Benefits

1. **Zero Configuration**: Just specify device name for standard models
2. **Smart Defaults**: Incorporates all learnings from TTTv1
3. **Model-Aware**: Automatically adjusts for model size and architecture
4. **Progressive Disclosure**: Simple API with full control when needed
5. **Type Safety**: Strongly typed configuration objects
6. **Extensibility**: Easy to add new devices and presets

## Migration from TTTv1

```python
# TTTv1 style (complex)
model_config = {}
model_config["COMPUTE_KERNEL_CONFIG_HIFI2"] = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
# ... hundreds more lines of config ...

# TTTv2 style (simple)
model = ModelFactory.from_spec(model_spec, device="N150")
# That's it! All the complexity is handled internally
```

This design achieves the goal of minimal hardware configuration code while preserving the ability to fine-tune when needed.
