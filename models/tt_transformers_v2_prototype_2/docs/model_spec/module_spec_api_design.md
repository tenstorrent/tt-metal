# TTTv2 Module Specification API Design

## Core Principle: Fixed Public API with Internal Distribution

The key insight is that each module exposes a stable, minimal API of essential parameters while internally handling the complex mapping to TTNN operations.

## Design Option 1: Fixed Parameter API (Your Proposal)

### Module Spec as Data Structure

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass(frozen=True)  # Immutable for clarity
class AttentionModuleSpec:
    """Public API for attention module specification"""
    # Essential mathematical parameters only
    hidden_dim: int
    num_heads: int

    # Optional parameters with sensible defaults
    head_dim: Optional[int] = None  # Computed if not provided
    dropout: float = 0.0
    max_seq_length: int = 8192

    def __post_init__(self):
        # Validate and compute derived parameters
        if self.head_dim is None:
            object.__setattr__(self, 'head_dim', self.hidden_dim // self.num_heads)

        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert self.head_dim * self.num_heads == self.hidden_dim, "head_dim * num_heads must equal hidden_dim"

@dataclass(frozen=True)
class FFNModuleSpec:
    """Public API for FFN module specification"""
    hidden_dim: int
    intermediate_dim: int
    activation: str = "swiglu"
    dropout: float = 0.0

@dataclass(frozen=True)
class NormModuleSpec:
    """Public API for normalization specification"""
    hidden_dim: int
    eps: float = 1e-5
    norm_type: str = "rmsnorm"
```

### Module Implementation with Internal Distribution

```python
class MultiHeadAttention(BaseAttention):
    """Attention module that distributes spec to TTNN ops"""

    def __init__(self, spec: AttentionModuleSpec, device=None, hw_config=None):
        self.spec = spec  # Store immutable spec
        self.device = device
        self.hw_config = hw_config or self.get_default_hw_config(device)

        # Distribute to internal TTNN operations
        self._create_ttnn_ops()

    def _create_ttnn_ops(self):
        """Distribute spec parameters to TTNN operations"""
        # The module decides how to map spec to ops

        # Example: QKV projection distribution
        if self.hw_config.fuse_qkv:
            self.qkv_proj = ttnn.Linear(
                in_features=self.spec.hidden_dim,
                out_features=3 * self.spec.hidden_dim,
                bias=False,
                # Hardware-specific parameters
                dtype=self.hw_config.compute_dtype,
                memory_config=self._get_memory_config(),
                kernel_config=self._get_kernel_config()
            )
        else:
            # Separate projections
            self.q_proj = ttnn.Linear(self.spec.hidden_dim, self.spec.hidden_dim)
            self.k_proj = ttnn.Linear(self.spec.hidden_dim, self.spec.hidden_dim)
            self.v_proj = ttnn.Linear(self.spec.hidden_dim, self.spec.hidden_dim)

        # Output projection
        self.out_proj = ttnn.Linear(
            self.spec.hidden_dim,
            self.spec.hidden_dim,
            bias=False
        )

        # Attention-specific ops
        self.scale = 1.0 / math.sqrt(self.spec.head_dim)

    def _get_memory_config(self):
        """Determine memory layout based on spec and hardware"""
        if self.spec.hidden_dim > 8192 and self.device == "ttnn":
            # Large models need sharding
            return ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                buffer_type=ttnn.BufferType.L1
            )
        else:
            return ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttnn.BufferType.DRAM
            )
```

## Design Option 2: Builder Pattern with Progressive Refinement

```python
class AttentionBuilder:
    """Progressive builder for attention specification"""

    def __init__(self):
        self._params = {}
        self._constraints = []

    def hidden_dim(self, value: int) -> 'AttentionBuilder':
        self._params['hidden_dim'] = value
        self._constraints.append(lambda: value > 0 and value % 32 == 0)
        return self

    def num_heads(self, value: int) -> 'AttentionBuilder':
        self._params['num_heads'] = value
        # Add constraint that depends on hidden_dim
        self._constraints.append(
            lambda: self._params['hidden_dim'] % value == 0
        )
        return self

    def build(self) -> AttentionModuleSpec:
        # Validate all constraints
        for constraint in self._constraints:
            assert constraint(), "Constraint validation failed"

        return AttentionModuleSpec(**self._params)

# Usage
spec = AttentionBuilder() \
    .hidden_dim(4096) \
    .num_heads(32) \
    .build()
```

## Design Option 3: Protocol-Based Specification

```python
from typing import Protocol

class AttentionSpecProtocol(Protocol):
    """Protocol defining what an attention spec must provide"""
    @property
    def hidden_dim(self) -> int: ...

    @property
    def num_heads(self) -> int: ...

    @property
    def head_dim(self) -> int: ...

    def validate(self) -> None:
        """Validate the specification"""
        ...

# Multiple implementations can satisfy the protocol
class MinimalAttentionSpec:
    def __init__(self, hidden_dim: int, num_heads: int):
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._hidden_dim // self._num_heads

    def validate(self):
        assert self._hidden_dim % self._num_heads == 0

class RichAttentionSpec:
    """Richer spec with more parameters"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @property
    def hidden_dim(self) -> int:
        return self.config['hidden_dim']

    # ... other protocol methods
```

## Design Option 4: Hierarchical Specification

```python
@dataclass
class BaseModuleSpec:
    """Base specification all modules share"""
    module_id: str
    hidden_dim: int

    def validate(self):
        assert self.hidden_dim > 0

@dataclass
class ComputeModuleSpec(BaseModuleSpec):
    """Specification for modules that do computation"""
    precision: str = "auto"  # auto, high, balanced, fast

    def get_compute_requirements(self) -> Dict[str, Any]:
        return {
            'flops': self._estimate_flops(),
            'memory_bandwidth': self._estimate_bandwidth()
        }

@dataclass
class AttentionModuleSpec(ComputeModuleSpec):
    """Attention-specific parameters"""
    num_heads: int
    attention_type: str = "standard"  # standard, flash, linear, sparse

    def _estimate_flops(self):
        seq_len_estimate = 2048  # Default estimate
        return 2 * seq_len_estimate * self.hidden_dim * self.hidden_dim
```

## Recommended Approach: Hybrid Design

Combine the best aspects of each option:

```python
# 1. Core immutable specs (Option 1)
@dataclass(frozen=True)
class AttentionSpec:
    """Immutable public API"""
    hidden_dim: int
    num_heads: int
    # Only essential params in public API

    def validate(self):
        assert self.hidden_dim % self.num_heads == 0

# 2. Module handles distribution
class MultiHeadAttention:
    def __init__(self, spec: AttentionSpec, device=None, **kwargs):
        self.spec = spec
        self._setup_ttnn_ops()

    def _setup_ttnn_ops(self):
        """Module decides how to distribute to ops"""
        # Complex internal logic hidden from users

        # Example: Dynamic op selection based on spec
        if self.spec.num_heads > 64:
            # Use specialized kernel for many heads
            self._setup_grouped_query_attention()
        else:
            self._setup_standard_attention()

# 3. Factory for common patterns
class ModuleSpecFactory:
    @staticmethod
    def attention_from_config(config: Dict) -> AttentionSpec:
        """Convert various configs to our spec"""
        if 'n_heads' in config:  # GPT style
            return AttentionSpec(
                hidden_dim=config['n_embd'],
                num_heads=config['n_heads']
            )
        elif 'num_attention_heads' in config:  # HF style
            return AttentionSpec(
                hidden_dim=config['hidden_size'],
                num_heads=config['num_attention_heads']
            )

# 4. Reference tracking (your idea)
@dataclass
class ModuleSpecWithReference:
    """Spec with optional reference for debugging"""
    spec: AttentionSpec
    reference_module: Optional[str] = None  # e.g., "model.layers.0.self_attn"
    reference_source: Optional[str] = None   # e.g., "huggingface"

    def compare_with_reference(self, ref_outputs, our_outputs):
        """For debugging accuracy issues"""
        pass
```

## Benefits of Fixed API Approach

1. **Stable Interface**: Users only need to know essential parameters
2. **Internal Flexibility**: Modules can evolve implementation without breaking API
3. **Clear Boundaries**: Spec (what) vs implementation (how) separation
4. **Type Safety**: Strong typing with dataclasses/protocols
5. **Debugging Support**: Reference association for accuracy testing

## Example Usage

```python
# Step 1: Create spec (no hardware details!)
attention_spec = AttentionSpec(hidden_dim=4096, num_heads=32)

# Step 2: Module handles all complexity
attention = MultiHeadAttention(attention_spec, device="ttnn:0")
# Module internally:
# - Computes head_dim
# - Decides QKV fusion
# - Selects memory layout
# - Chooses optimal kernel
# - Configures sharding

# Step 3: Optional - track reference for debugging
tracked_attention = ModuleSpecWithReference(
    spec=attention_spec,
    reference_module="model.layers.15.self_attn",
    reference_source="pytorch"
)

# During debugging:
if accuracy < threshold:
    ref_output = get_reference_output(tracked_attention.reference_module)
    our_output = attention(input)
    diff = compare_outputs(ref_output, our_output)
```

This design achieves the perfect balance of simplicity for users while maintaining flexibility for module implementers.
