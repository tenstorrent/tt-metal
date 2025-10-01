# TTTv2 Module Specification and Hardware Configuration Design

## Prompt and summary

### Prompt
Let's focus on cracking these big design points in more
details: 1) model spec to module spec; 2) hardware configuration
for model and module. Both are needed before any model could be
run on TT hardware; however, it is important for the success of
TTTv2 to decouple them and modularize (into modules) as much as
reasonable. One idea is to build TTTv2 model implementations in
two steps: 1) consume model spec (custom made in python or from a
 standard format such as huggingface) to build model structures,
instanciate (partially) modules, connect them into compute
graphs; by this time, the developers could try to run this model
implementation on a specified TT hardware because each module by
default provides hardware configuration by default for each
supported hardware device; and 2) the developers can follow the
error messages (if any) to debug the model implementation by
adding more module instances to the model impl or troubleshoot
accuracy issues by isolate-testing module instances; or to debug
performance issues by getting a performance profile and adjust
hardware configuraitons. What do you think about this developers
experience?

### Summary
I really like this developer experience vision! Here's why it
  works so well:

  Key Strengths:

  1. Progressive Disclosure
    - Start simple: Model just works with defaults
    - Add complexity only when needed
    - Each step builds on previous understanding
  2. Clear Separation of Concerns
    - Architecture Spec: Pure mathematical description (what to
  compute)
    - Hardware Config: Execution details (how to compute it)
    - Developers can focus on one at a time
  3. Actionable Feedback Loop
  Run → Profile → Identify Issue → Get Suggestion → Apply Fix →
  Verify
  3. Every error comes with a solution!
  4. Modular Hardware Configs
    - Each module provides its own sensible defaults
    - Configs can be overridden at any granularity
    - Hardware knowledge lives with the module that needs it

  The Developer Journey:

  Day 1 (ML Researcher)
  ```python
  model = TTTModel.from_huggingface("llama-3")
  output = model.generate("Hello")  # Just works!
  ```

  Day 2 (Debugging)
  ```python
  profile = model.profile(input)
  # "Layer 15 attention is 10x slower than expected"
  # "Try: Use HiFi4 precision for attention scores"
  ```

  Day 3 (Optimization)
  ```python
  model.layers[15].attention.apply_hw_config(
      score_dtype="HiFi4"
  )  # 10x speedup!
  ```

  Why This Design Succeeds:

  1. Fails Fast with Solutions: When things go wrong, developers
  get specific, actionable guidance
  2. Supports Different Expertise Levels:
    - Beginners: Use defaults
    - Intermediate: Apply preset optimizations
    - Advanced: Fine-tune specific modules
    - Expert: Write custom kernels
  3. Maintains TTTv2's Library Philosophy: The two-step approach
  is a pattern, not a framework requirement. Advanced users can
  still directly instantiate modules with custom configs.

  This design achieves the perfect balance between ease of use
  and performance optimization, making TTTv2 accessible to ML
  researchers while still enabling hardware engineers to extract
  maximum performance.

## Overview

The proposed two-step developer experience separates architectural concerns from hardware optimization, enabling rapid prototyping followed by performance tuning.

## Developer Experience Vision

### Step 1: Architecture-First Development
```python
# Start with just the model architecture
model_spec = ModelSpec.from_huggingface("meta-llama/Llama-3-8b")
model = TTTModel.from_spec(model_spec)

# Run immediately with default hardware configs
output = model.generate("Hello world", device="ttnn:0")
# Works! But might not be optimal
```

### Step 2: Performance Optimization
```python
# Profile to find bottlenecks
profile = model.profile(sample_input)
print(profile.bottlenecks())
# > Layer 15 attention: 45ms (expected: 5ms)
# > Suggested: Use HiFi4 for attention scores

# Apply targeted optimizations
model.layers[15].attention.apply_hw_config(
    TTNNConfig(score_dtype="HiFi4", kv_cache_mode="paged")
)
```

## Detailed Design

### 1. Model Spec to Module Spec Mapping

#### Layer 1: Architecture Specification (What)
```python
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum

@dataclass
class ArchitectureSpec:
    """Pure mathematical/structural definition - no hardware details"""

    # Model-level architecture
    model_type: str  # "decoder", "encoder-decoder", "multimodal"
    vocab_size: int
    max_sequence_length: int

    # Layer specifications
    layers: List['LayerSpec']

    # Connections (for non-sequential models)
    connections: Optional[List[Tuple[str, str]]] = None

@dataclass
class LayerSpec:
    """Specification for a single layer"""
    layer_id: str
    layer_type: str  # "transformer", "attention", "ffn", "norm", etc.

    # Mathematical parameters only
    params: Dict[str, Any]

    # No hardware details here!
    # Bad: dtype, sharding, kernel_type
    # Good: hidden_dim, num_heads, activation_fn

class TransformerLayerSpec(LayerSpec):
    """Specialized spec for transformer layers"""
    def __init__(self, layer_id: str, hidden_dim: int, num_heads: int,
                 ffn_dim: int, activation: str = "swish"):
        super().__init__(
            layer_id=layer_id,
            layer_type="transformer",
            params={
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "ffn_dim": ffn_dim,
                "activation": activation,
                "norm_type": "rmsnorm",
                "norm_eps": 1e-5
            }
        )

# Example: Building architecture spec
def llama3_architecture_spec() -> ArchitectureSpec:
    """Pure architecture - what the model computes"""
    layers = []

    # Embedding
    layers.append(LayerSpec(
        layer_id="embedding",
        layer_type="embedding",
        params={"vocab_size": 32000, "hidden_dim": 4096}
    ))

    # Transformer layers
    for i in range(32):
        layers.append(TransformerLayerSpec(
            layer_id=f"layer_{i}",
            hidden_dim=4096,
            num_heads=32,
            ffn_dim=11008,
            activation="swish"
        ))

    # Output
    layers.append(LayerSpec(
        layer_id="lm_head",
        layer_type="linear",
        params={"input_dim": 4096, "output_dim": 32000}
    ))

    return ArchitectureSpec(
        model_type="decoder",
        vocab_size=32000,
        max_sequence_length=8192,
        layers=layers
    )
```

#### Layer 2: Module Instantiation
```python
class ModuleFactory:
    """Maps architecture specs to TTT module instances"""

    @staticmethod
    def create_module(layer_spec: LayerSpec, device: Optional[str] = None):
        """Create module with default hardware config"""

        # Module type dispatch
        if layer_spec.layer_type == "transformer":
            return ModuleFactory._create_transformer_layer(layer_spec, device)
        elif layer_spec.layer_type == "attention":
            return ModuleFactory._create_attention(layer_spec, device)
        elif layer_spec.layer_type == "ffn":
            return ModuleFactory._create_ffn(layer_spec, device)
        # ... more types

    @staticmethod
    def _create_transformer_layer(spec: LayerSpec, device: str):
        """Create a complete transformer layer"""
        from tt_transformers_v2.patterns import TransformerLayer
        from tt_transformers_v2.attention import MultiHeadAttention
        from tt_transformers_v2.ffn import SwiGLU
        from tt_transformers_v2.normalization import RMSNorm

        params = spec.params

        # Create submodules with default configs
        attention = MultiHeadAttention(
            hidden_dim=params["hidden_dim"],
            num_heads=params["num_heads"],
            device=device
            # Hardware config is auto-selected!
        )

        ffn = SwiGLU(
            hidden_dim=params["hidden_dim"],
            intermediate_dim=params["ffn_dim"],
            device=device
        )

        norm = RMSNorm(
            hidden_dim=params["hidden_dim"],
            eps=params["norm_eps"],
            device=device
        )

        return TransformerLayer(
            attention=attention,
            ffn=ffn,
            norm=norm,
            layer_id=spec.layer_id
        )
```

### 2. Hardware Configuration Design

#### Principle: Progressive Disclosure
```python
class HardwareConfigurable:
    """Base class for hardware-aware modules"""

    def __init__(self, *args, device=None, hw_config=None, **kwargs):
        # Step 1: Use defaults if no config provided
        if hw_config is None:
            hw_config = self.get_default_hw_config(device)

        self.hw_config = hw_config
        self._validate_hw_config()

    @classmethod
    def get_default_hw_config(cls, device: str) -> 'HardwareConfig':
        """Get sensible defaults for this device"""
        if device.startswith("ttnn"):
            return cls.get_ttnn_defaults()
        elif device.startswith("cuda"):
            return cls.get_cuda_defaults()
        else:
            return cls.get_cpu_defaults()

    def apply_hw_config(self, new_config: 'HardwareConfig'):
        """Apply new hardware config (for step 2 optimization)"""
        self.hw_config = new_config
        self._recompile()

@dataclass
class HardwareConfig:
    """Hardware-specific execution parameters"""
    # Computation precision
    compute_dtype: str = "bfloat16"
    accumulate_dtype: str = "float32"

    # Memory layout
    layout: str = "row_major"  # or "tile_major" for TTNN
    sharding_strategy: Optional[str] = None

    # Kernel selection
    kernel_variant: str = "auto"  # or "flash", "fused", etc.

    # Memory optimization
    activation_checkpointing: bool = False
    gradient_accumulation_dtype: Optional[str] = None

    # Device-specific
    device_specific: Dict[str, Any] = field(default_factory=dict)
```

#### Module-Specific Hardware Defaults
```python
class MultiHeadAttention(BaseAttention, HardwareConfigurable):
    """Attention with smart hardware defaults"""

    @classmethod
    def get_ttnn_defaults(cls) -> HardwareConfig:
        """TTNN-optimized defaults for attention"""
        return HardwareConfig(
            compute_dtype="bfloat16",
            accumulate_dtype="float32",
            layout="tile_major",
            kernel_variant="flash",
            device_specific={
                "score_dtype": "HiFi4",  # High precision for scores
                "qkv_dtype": "bfloat16",
                "use_fused_qkv": True,
                "kv_cache_dtype": "bfloat16",
                "kv_cache_layout": "paged"
            }
        )

    def __init__(self, hidden_dim: int, num_heads: int, device=None, hw_config=None):
        # First init hardware config
        super().__init__(device=device, hw_config=hw_config)

        # Then init attention parameters
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Create weights with hardware-aware layout
        self._create_parameters()

    def _create_parameters(self):
        """Create parameters with hardware-optimal layout"""
        if self.hw_config.device_specific.get("use_fused_qkv", False):
            # Single fused QKV projection
            self.qkv_proj = self._create_weight(
                [self.hidden_dim, 3 * self.hidden_dim],
                dtype=self.hw_config.compute_dtype,
                layout=self.hw_config.layout
            )
        else:
            # Separate Q, K, V projections
            self.q_proj = self._create_weight(...)
            self.k_proj = self._create_weight(...)
            self.v_proj = self._create_weight(...)
```

### 3. Two-Step Development Flow

#### Step 1: Get It Working
```python
class TTTModel:
    """Model that separates architecture from hardware"""

    @classmethod
    def from_spec(cls, arch_spec: ArchitectureSpec, device: str = "cpu"):
        """Create model from architecture spec with default hw configs"""
        modules = {}

        for layer_spec in arch_spec.layers:
            # Create with defaults - just works!
            module = ModuleFactory.create_module(layer_spec, device)
            modules[layer_spec.layer_id] = module

        return cls(modules, arch_spec, device)

    def forward(self, input_ids, **kwargs):
        """Run forward pass with default configs"""
        x = self.modules["embedding"](input_ids)

        for i in range(self.num_layers):
            x = self.modules[f"layer_{i}"](x, **kwargs)

        return self.modules["lm_head"](x)

    def check_hardware_compatibility(self):
        """Proactive error checking"""
        issues = []

        for name, module in self.modules.items():
            # Check if module can run on target device
            compat = module.check_device_compatibility(self.device)
            if not compat.is_compatible:
                issues.append({
                    'module': name,
                    'issue': compat.reason,
                    'suggestion': compat.suggestion
                })

        return issues
```

#### Step 2: Make It Fast
```python
class PerformanceOptimizer:
    """Tools for step 2 optimization"""

    def profile(self, model: TTTModel, sample_input) -> 'ProfileReport':
        """Profile model to find bottlenecks"""
        with ProfileContext() as profiler:
            model(sample_input)

        return ProfileReport(profiler.get_results())

class ProfileReport:
    """Actionable performance insights"""

    def bottlenecks(self) -> List[Dict[str, Any]]:
        """Get bottlenecks with suggestions"""
        bottlenecks = []

        for layer_name, metrics in self.metrics.items():
            expected_time = self._get_expected_time(layer_name)
            actual_time = metrics['duration_ms']

            if actual_time > expected_time * 1.5:  # 50% slower
                bottlenecks.append({
                    'layer': layer_name,
                    'actual_ms': actual_time,
                    'expected_ms': expected_time,
                    'suggestions': self._get_optimization_suggestions(layer_name, metrics)
                })

        return bottlenecks

    def _get_optimization_suggestions(self, layer: str, metrics: Dict) -> List[str]:
        """Hardware-specific optimization suggestions"""
        suggestions = []

        # Memory-bound?
        if metrics['memory_bandwidth_utilization'] > 0.9:
            suggestions.append("Try activation checkpointing")
            suggestions.append("Consider sharding across chips")

        # Compute-bound?
        if metrics['compute_utilization'] < 0.5:
            suggestions.append("Enable kernel fusion")
            suggestions.append("Use specialized kernels (flash, fused)")

        # Precision issues?
        if 'numeric_errors' in metrics:
            suggestions.append("Increase precision for this layer")
            suggestions.append(f"Try HiFi4 for scores (currently {metrics['dtype']})")

        return suggestions

# Developer applies suggestions
def optimize_model(model: TTTModel, profile: ProfileReport):
    """Apply optimizations based on profile"""

    for bottleneck in profile.bottlenecks():
        layer_name = bottleneck['layer']
        module = model.modules[layer_name]

        # Try suggested optimizations
        if "Try HiFi4 for scores" in bottleneck['suggestions']:
            if hasattr(module, 'attention'):
                new_config = module.attention.hw_config.copy()
                new_config.device_specific['score_dtype'] = 'HiFi4'
                module.attention.apply_hw_config(new_config)

        # More optimization logic...
```

### 4. Error Messages and Debugging

```python
class HardwareError(Exception):
    """Helpful hardware configuration errors"""

    def __init__(self, module: str, issue: str, suggestion: str):
        self.module = module
        self.issue = issue
        self.suggestion = suggestion

        super().__init__(
            f"\nModule: {module}\n"
            f"Issue: {issue}\n"
            f"Suggestion: {suggestion}\n"
            f"Documentation: https://docs.tt.com/ttt/hw-config#{module}"
        )

# Example errors developers might see
class ShardingError(HardwareError):
    def __init__(self, module: str, tensor_shape: tuple, available_memory: int):
        super().__init__(
            module=module,
            issue=f"Tensor {tensor_shape} exceeds chip memory ({available_memory}MB)",
            suggestion=(
                f"Options:\n"
                f"1. Enable sharding: module.apply_hw_config(shard='column')\n"
                f"2. Use smaller batch size\n"
                f"3. Enable activation checkpointing"
            )
        )
```

### 5. Progressive Learning Path

```python
# Beginner: Just works
model = TTTModel.from_huggingface("meta-llama/Llama-3-8b")
output = model.generate("Hello")

# Intermediate: Basic optimization
profile = model.profile(sample_input)
model.apply_optimization_preset("balanced")  # memory vs speed

# Advanced: Fine-grained control
model.layers[15].attention.apply_hw_config(
    HardwareConfig(
        compute_dtype="bfloat16",
        device_specific={
            "score_dtype": "HiFi4",
            "qkv_layout": "column_parallel",
            "kv_cache_mode": "paged",
            "page_size": 256
        }
    )
)

# Expert: Custom kernels
@register_kernel("ttnn", "attention")
def my_custom_attention_kernel(q, k, v, config):
    # Custom TTNN kernel implementation
    pass
```

## Benefits of This Design

1. **Low Barrier to Entry**: Model works immediately with defaults
2. **Progressive Optimization**: Only optimize what needs it
3. **Clear Separation**: Architecture vs execution concerns
4. **Actionable Feedback**: Errors come with solutions
5. **Hardware Abstraction**: Same model code works across devices
6. **Performance Transparency**: Know why something is slow

## Summary

This two-step approach provides the best developer experience:
- **Step 1**: Focus on getting the architecture right
- **Step 2**: Optimize performance based on profiling

The key insight is that modules provide sensible hardware defaults, so models "just work" out of the box, while still allowing fine-grained optimization when needed.
