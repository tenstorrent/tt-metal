# Hardware Configuration and Tensor Cache Design for TTTv2

## Overview

Hardware configuration happens after module specialization (prefill/decode), with each module having access to default configs for every TTNN op on every supported device. The design allows direct op-level configuration now while leaving room for future abstraction.

## Design Principles

1. **Practical First**: Direct TTNN op configuration (no over-abstraction initially)
2. **Device-Aware Defaults**: Each module knows optimal configs for its ops per device
3. **Override Capability**: Users can override specific op configs when needed
4. **Cache Integration**: Tensor caches for weights (at config time) and activations (at compile time)

## Hardware Configuration Flow

```
Module Spec → Execution Strategy → Hardware Config → Tensor Caches
    ↓              ↓                    ↓                  ↓
 (math only)  (prefill/decode)   (TTNN op configs)  (weight/activation)
```

## Detailed Design

### 1. Hardware Configuration Layer

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import ttnn

@dataclass
class TTNNOpConfig:
    """Configuration for a single TTNN operation"""
    op_name: str
    memory_config: ttnn.MemoryConfig
    kernel_config: Optional[ttnn.MatmulConfig] = None
    dtype: str = "bfloat16"
    layout: str = "TILE"
    sharding: Optional[Dict[str, Any]] = None

@dataclass
class ModuleHardwareConfig:
    """Hardware configuration for an entire module"""
    device: str
    optimization_target: str = "balanced"  # "latency", "memory", "balanced"

    # Op-level configs - can override defaults per op
    op_configs: Dict[str, TTNNOpConfig] = field(default_factory=dict)

    # Cache configs
    weight_cache_dtype: str = "bfloat16"
    activation_cache_policy: str = "auto"  # "always", "never", "auto"

    def get_op_config(self, op_name: str) -> Optional[TTNNOpConfig]:
        """Get config for specific op, if overridden"""
        return self.op_configs.get(op_name)
```

### 2. Module Implementation with Hardware Config

```python
class MultiHeadAttentionWithHardware:
    """Attention module with hardware configuration after specialization"""

    # Default configs per device
    DEFAULT_CONFIGS = {
        "ttnn:n150": {
            "prefill": {
                "qkv_matmul": TTNNOpConfig(
                    op_name="qkv_matmul",
                    memory_config=ttnn.MemoryConfig(
                        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
                        buffer_type=ttnn.BufferType.DRAM
                    ),
                    kernel_config=ttnn.MatmulConfig(
                        compute_kernel_type="LARGE_TILE",
                        fp32_accumulation=True
                    ),
                    dtype="bfloat16"
                ),
                "attention_matmul": TTNNOpConfig(
                    op_name="attention_matmul",
                    memory_config=ttnn.MemoryConfig(
                        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                        buffer_type=ttnn.BufferType.L1
                    ),
                    dtype="bfloat16"
                )
            },
            "decode": {
                "q_matmul": TTNNOpConfig(
                    op_name="q_matmul",
                    memory_config=ttnn.MemoryConfig(
                        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        buffer_type=ttnn.BufferType.L1
                    ),
                    kernel_config=ttnn.MatmulConfig(
                        compute_kernel_type="SMALL_TILE",
                        fp32_accumulation=False  # Lower precision OK for decode
                    ),
                    dtype="bfloat16"
                ),
                "kv_cache": TTNNOpConfig(
                    op_name="kv_cache",
                    memory_config=ttnn.MemoryConfig(
                        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        buffer_type=ttnn.BufferType.DRAM
                    ),
                    dtype="bfloat8_b"  # Lower precision for cache
                )
            }
        },
        "ttnn:n300": {
            # Different optimal configs for N300
            # ...
        }
    }

    def __init__(self,
                 spec: AttentionSpec,
                 device: str,
                 hw_config: Optional[ModuleHardwareConfig] = None):
        self.spec = spec
        self.device = device

        # Use provided config or create default
        if hw_config is None:
            hw_config = ModuleHardwareConfig(device=device)
        self.hw_config = hw_config

        # Step 1: Specialize into prefill/decode
        self._specialize_execution_modes()

        # Step 2: Apply hardware configs to specialized implementations
        self._apply_hardware_configs()

        # Step 3: Setup tensor caches
        self._setup_tensor_caches()

    def _specialize_execution_modes(self):
        """Create prefill and decode implementations"""
        self.prefill_impl = self._create_prefill_implementation()
        self.decode_impl = self._create_decode_implementation()

    def _create_prefill_implementation(self):
        """Create prefill ops (before hardware config)"""
        return {
            'qkv_proj': None,  # Will be configured in _apply_hardware_configs
            'out_proj': None,
            'scale': 1.0 / math.sqrt(self.spec.head_dim)
        }

    def _apply_hardware_configs(self):
        """Apply hardware configs to each op"""
        device_configs = self.DEFAULT_CONFIGS.get(self.device, self.DEFAULT_CONFIGS["ttnn:n150"])

        # Configure prefill ops
        prefill_configs = device_configs["prefill"]

        # QKV projection for prefill
        qkv_config = self.hw_config.get_op_config("qkv_matmul") or prefill_configs["qkv_matmul"]
        self.prefill_impl['qkv_proj'] = ttnn.Linear(
            self.spec.hidden_dim,
            3 * self.spec.hidden_dim,
            bias=False,
            memory_config=qkv_config.memory_config,
            kernel_config=qkv_config.kernel_config,
            dtype=qkv_config.dtype,
            output_layout=qkv_config.layout
        )

        # Configure decode ops
        decode_configs = device_configs["decode"]

        # Q projection for decode (separate, optimized for single token)
        q_config = self.hw_config.get_op_config("q_matmul") or decode_configs["q_matmul"]
        self.decode_impl['q_proj'] = ttnn.Linear(
            self.spec.hidden_dim,
            self.spec.hidden_dim,
            bias=False,
            memory_config=q_config.memory_config,
            kernel_config=q_config.kernel_config,
            dtype=q_config.dtype
        )

    def override_op_config(self, op_name: str, new_config: TTNNOpConfig):
        """Allow users to override specific op configs"""
        self.hw_config.op_configs[op_name] = new_config
        # Re-apply configs
        self._apply_hardware_configs()
```

### 3. Tensor Cache Design

```python
@dataclass
class TensorCacheConfig:
    """Configuration for tensor caching"""
    cache_weights: bool = True
    cache_activations: bool = True
    weight_cache_dtype: str = "bfloat16"
    activation_cache_dtype: str = "bfloat16"
    cache_layout: str = "TILE"
    shard_strategy: Optional[str] = None

class TensorCacheManager:
    """Manages hardware tensor caches for weights and activations"""

    def __init__(self, device: str, cache_config: TensorCacheConfig):
        self.device = device
        self.cache_config = cache_config
        self.weight_caches = {}
        self.activation_caches = {}

    def cache_module_weights(self, module_name: str, reference_weights: Dict[str, torch.Tensor]):
        """Cache weights during hardware config time"""
        cached_weights = {}

        for weight_name, weight_tensor in reference_weights.items():
            # Convert to TT tensor format
            tt_weight = ttnn.from_torch(
                weight_tensor,
                device=self.device,
                dtype=self.cache_config.weight_cache_dtype,
                layout=self.cache_config.cache_layout
            )

            # Apply sharding if needed
            if self.cache_config.shard_strategy:
                tt_weight = self._apply_sharding(tt_weight, self.cache_config.shard_strategy)

            # Store in cache
            cached_weights[weight_name] = tt_weight

        self.weight_caches[module_name] = cached_weights
        return cached_weights

    def setup_activation_cache(self, module_name: str, activation_shapes: Dict[str, tuple]):
        """Setup activation caches (at compile time)"""
        if not self.cache_config.cache_activations:
            return {}

        activation_caches = {}

        for act_name, shape in activation_shapes.items():
            # Pre-allocate activation buffer
            cache_buffer = ttnn.allocate_tensor(
                shape=shape,
                device=self.device,
                dtype=self.cache_config.activation_cache_dtype,
                layout=self.cache_config.cache_layout
            )

            activation_caches[act_name] = cache_buffer

        self.activation_caches[module_name] = activation_caches
        return activation_caches

    def _apply_sharding(self, tensor, strategy):
        """Apply sharding strategy to tensor"""
        if strategy == "column":
            return ttnn.shard_tensor_to_cores(tensor, dim=1)
        elif strategy == "row":
            return ttnn.shard_tensor_to_cores(tensor, dim=0)
        # More sharding strategies...
        return tensor
```

### 4. Integration with Module

```python
class MultiHeadAttentionComplete:
    """Complete attention module with all components"""

    def __init__(self,
                 spec: AttentionSpec,
                 device: str,
                 hw_config: Optional[ModuleHardwareConfig] = None,
                 reference_module=None):

        self.spec = spec
        self.device = device
        self.hw_config = hw_config or ModuleHardwareConfig(device=device)

        # Step 1: Execution specialization
        self._specialize_execution_modes()

        # Step 2: Hardware configuration
        self._apply_hardware_configs()

        # Step 3: Weight caching (if reference provided)
        if reference_module:
            self._cache_weights_from_reference(reference_module)

        # Step 4: Setup for activation caching (happens at compile)
        self._prepare_activation_caching()

    def _cache_weights_from_reference(self, reference_module):
        """Convert and cache reference weights"""
        cache_manager = TensorCacheManager(
            self.device,
            TensorCacheConfig(
                weight_cache_dtype=self.hw_config.weight_cache_dtype
            )
        )

        # Extract weights from reference
        reference_weights = {
            'qkv_weight': reference_module.qkv_proj.weight,
            'out_weight': reference_module.out_proj.weight
        }

        # Cache them
        self.cached_weights = cache_manager.cache_module_weights(
            f"attention_{id(self)}",
            reference_weights
        )

        # Update ops to use cached weights
        self.prefill_impl['qkv_proj'].set_weight(self.cached_weights['qkv_weight'])

    def _prepare_activation_caching(self):
        """Prepare activation cache setup (actual allocation at compile)"""
        self.activation_shapes = {
            'qkv_output': (None, None, 3 * self.spec.hidden_dim),  # Batch/seq determined at compile
            'attention_scores': (None, self.spec.num_heads, None, None),
            'attention_output': (None, None, self.spec.hidden_dim)
        }

    def compile(self, sample_input):
        """Compile-time setup including activation caches"""
        batch_size, seq_len = sample_input.shape[:2]

        # Now we know concrete shapes - setup activation caches
        cache_manager = TensorCacheManager(self.device, TensorCacheConfig())

        concrete_shapes = {
            'qkv_output': (batch_size, seq_len, 3 * self.spec.hidden_dim),
            'attention_scores': (batch_size, self.spec.num_heads, seq_len, seq_len),
            'attention_output': (batch_size, seq_len, self.spec.hidden_dim)
        }

        self.activation_caches = cache_manager.setup_activation_cache(
            f"attention_{id(self)}",
            concrete_shapes
        )

    def forward(self, hidden_states, use_cache=True):
        """Forward with optional activation caching"""
        if use_cache and hasattr(self, 'activation_caches'):
            # Use pre-allocated buffers
            qkv_output = self.activation_caches['qkv_output']
            self.prefill_impl['qkv_proj'](hidden_states, output=qkv_output)
            # ... rest of forward using cached buffers
        else:
            # Normal forward without caching
            qkv_output = self.prefill_impl['qkv_proj'](hidden_states)
            # ...
```

### 5. Future Extensibility Points

```python
# Future: TTNN Op abstraction layer
class TTNNOpAbstraction:
    """Future abstraction over TTNN ops (not implemented yet)"""

    @staticmethod
    def create_matmul(input_dim: int, output_dim: int,
                     optimization_target: str) -> TTNNOp:
        """Future: Auto-configure TTNN ops based on high-level intent"""
        # This would hide TTNN-specific details
        pass

# Future: Declarative hardware configs
hardware_config = """
optimization_target: latency
ops:
  qkv_matmul:
    precision: high
    memory: L1
    tiling: aggressive
  attention_scores:
    precision: medium
    memory: DRAM
"""

# Future: Auto-tuning
class AutoTuner:
    """Future: Automatically find optimal configs"""

    def tune_module(self, module, sample_inputs, target_metric="latency"):
        # Try different configs and measure
        pass
```

## Usage Examples

### Basic Usage
```python
# Simple - use all defaults
attention = MultiHeadAttentionComplete(
    spec=AttentionSpec(hidden_dim=4096, num_heads=32),
    device="ttnn:n150"
)
```

### Override Specific Op
```python
# Override just the QKV matmul config
hw_config = ModuleHardwareConfig(device="ttnn:n150")
hw_config.op_configs["qkv_matmul"] = TTNNOpConfig(
    op_name="qkv_matmul",
    memory_config=ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1
    ),
    dtype="float32"  # Higher precision
)

attention = MultiHeadAttentionComplete(spec, device, hw_config)
```

### With Weight Caching
```python
# Load reference model and cache weights
reference = load_huggingface_model("meta-llama/Llama-3-8b")
attention = MultiHeadAttentionComplete(
    spec=AttentionSpec(4096, 32),
    device="ttnn:n150",
    reference_module=reference.model.layers[0].self_attn
)
```

## Benefits

1. **Practical**: Direct TTNN op configuration without over-abstraction
2. **Flexible**: Can override individual ops when needed
3. **Efficient**: Weight caching at config time, activation caching at compile
4. **Extensible**: Clear points for future abstraction
5. **Device-Aware**: Optimal defaults per device

This design provides the right level of control for current needs while leaving clear extension points for future TTNN op abstraction.
