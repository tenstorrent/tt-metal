# TTTv2 Design Proposal

## Problem Analysis

The core issue with TTTv1 is the **N×M×P explosion**:
- N models × M platforms × P tests = exponential complexity
- Every change requires validating all combinations
- Adding model #11 requires testing against models 1-10

## Design Philosophy

**"Library, not Framework"** - TTTv2 should be a collection of building blocks that models consume, not a framework that controls models.

## Proposed Architecture

### 1. Core Library Structure

```
tt_transformers_v2/
├── __init__.py               # Top-level public API
├── src/                      # Core library source code
│   ├── __init__.py
│   ├── building_blocks/      # Pure functional components
│   │   ├── __init__.py
│   │   ├── attention/
│   │   │   ├── __init__.py
│   │   │   ├── mha.py       # Multi-head attention
│   │   │   ├── gqa.py       # Grouped-query attention
│   │   │   ├── flash.py     # Flash attention variants
│   │   │   ├── sliding.py   # Sliding window attention
│   │   │   └── rotary.py    # RoPE implementation
│   │   ├── ffn/
│   │   │   ├── __init__.py
│   │   │   ├── mlp.py       # Standard MLP
│   │   │   ├── gated_mlp.py # SwiGLU, GeGLU variants
│   │   │   └── moe.py       # Mixture of experts
│   │   ├── embeddings/
│   │   │   ├── __init__.py
│   │   │   ├── token.py     # Token embeddings
│   │   │   └── position.py  # Position embeddings
│   │   └── normalization/
│   │       ├── __init__.py
│   │       ├── layernorm.py
│   │       └── rmsnorm.py
│   │
│   ├── patterns/             # Common architectural patterns
│   │   ├── __init__.py
│   │   ├── decoder_layer.py  # Pre-built decoder patterns
│   │   ├── encoder_layer.py  # Pre-built encoder patterns
│   │   └── cross_attention.py # Cross-attention patterns
│   │
│   ├── hardware/             # Hardware abstraction layer
│   │   ├── __init__.py
│   │   ├── device_config.py  # Device capabilities
│   │   ├── memory_planner.py # Memory optimization
│   │   ├── kernel_library.py # Optimized kernels
│   │   └── sharding.py       # Tensor sharding strategies
│   │
│   ├── interfaces/           # Standard interfaces
│   │   ├── __init__.py
│   │   ├── model_spec.py     # Model architecture specifications
│   │   ├── model_factory.py  # Factory for creating models from specs
│   │   ├── weight_format.py  # Weight format specifications
│   │   ├── generation.py     # Generation interfaces (if needed)
│   │   └── serving.py        # Serving interfaces (vLLM, etc.)
│   │
│   └── testing/              # Testing utilities
│       ├── __init__.py
│       ├── test_suite.py     # TestSuite context manager & builder
│       ├── module_tests.py   # Core test implementations
│       ├── correctness.py    # Correctness test utilities
│       ├── performance.py    # Performance measurement utilities
│       └── shape_inference.py # Auto shape detection from modules
│
├── models/                   # Reference model implementations
│   ├── __init__.py
│   ├── llama3/              # LLaMA 3 reference implementation
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── config.py
│   └── mistral/             # Mistral reference implementation
│       ├── __init__.py
│       ├── model.py
│       └── config.py
│
└── tests/                    # Unit tests for real-world configurations
    ├── __init__.py
    ├── attention/           # Tests for attention variants
    │   ├── __init__.py
    │   ├── test_llama_attention.py    # LLaMA's specific attention config
    │   ├── test_mistral_attention.py  # Mistral's sliding window config
    │   └── test_gpt_attention.py      # GPT's standard MHA config
    ├── ffn/                 # Tests for FFN variants
    │   ├── __init__.py
    │   ├── test_swiglu.py   # SwiGLU as used in LLaMA
    │   └── test_geglu.py    # GeGLU as used in other models
    └── integration/         # Integration tests
        ├── __init__.py
        ├── test_decoder_patterns.py
        └── test_memory_efficiency.py
```

### 2. Model Implementation Pattern

TTTv2 includes select reference model implementations in the `models/` directory. These serve as:
- Examples of best practices for using TTTv2
- Migration templates for external models
- Test beds for new building blocks

Developers can use the reference models as templates to implement their own models and submit them to the model zoo.

All models are implemented in a way that consumes TTTv2 as a library, for example:

```python
# models/llama3/model.py
from tt_transformers_v2 import attention, ffn, normalization, patterns, embeddings

class LLaMA3Model:
    """Model implementation using TTTv2 building blocks."""

    def __init__(self, config: LLaMA3Config, device):
        self.config = config
        self.device = device

        # Direct construction from building blocks
        self.embedding = embeddings.TokenEmbedding(
            config.vocab_size, config.hidden_dim, device=device
        )

        self.layers = [
            patterns.DecoderLayer(
                attention=attention.MultiHeadAttention(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    rope_theta=config.rope_theta,
                    device=device,
                ),
                ffn=ffn.SwiGLU(
                    hidden_dim=config.hidden_dim,
                    intermediate_dim=config.intermediate_dim,
                    device=device,
                ),
                norm=normalization.RMSNorm(config.hidden_dim, device=device),
                residual_pattern="post_norm",
                device=device,
            )
            for _ in range(config.num_layers)
        ]

        self.final_norm = normalization.RMSNorm(config.hidden_dim, device=device)

    def __call__(self, input_ids, attention_mask=None):
        # Forward pass without PyTorch dependency
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.final_norm(x)
```

**Key Points:**
- Models live in separate repositories/packages
- Models pip install specific TTTv2 versions
- No model-specific code in TTTv2
- Models are free to organize as they wish

### 3. Strict Public API Control via __init__.py

The public API is **strictly controlled** through `__init__.py` files using `__all__`:

```python
# tt_transformers_v2/__init__.py
"""TTTv2: Modular transformer building blocks for Tenstorrent hardware."""

# Only these imports are allowed - everything else is private
__all__ = [
    # Module namespaces
    "attention",
    "ffn",
    "normalization",
    "embeddings",
    "patterns",
    "interfaces",
    "hardware",
    # Direct class access for convenience
    "MultiHeadAttention",
    "SwiGLU",
    "RMSNorm",
    "DecoderLayer",
    # Version
    "__version__",
]

# Version info
__version__ = "2.0.0"

# Controlled re-exports from src
from .src.building_blocks import attention, ffn, normalization, embeddings
from .src import patterns, interfaces, hardware

# Convenience imports (only for items in __all__)
from .src.building_blocks.attention import MultiHeadAttention
from .src.building_blocks.ffn import SwiGLU
from .src.building_blocks.normalization import RMSNorm
from .src.patterns import DecoderLayer

# Prevent direct src access
import sys
_module = sys.modules[__name__]
_module.__dict__["src"] = None  # Block direct access to src
```

```python
# tt_transformers_v2/src/building_blocks/__init__.py
"""Building blocks for transformer models."""

# Strict control - only these submodules are exposed
__all__ = ["attention", "ffn", "embeddings", "normalization"]

from . import attention
from . import ffn
from . import embeddings
from . import normalization

# Private modules/functions stay private
from . import _utils  # Not in __all__, stays private
```

```python
# tt_transformers_v2/src/building_blocks/attention/__init__.py
"""Attention mechanisms for transformers."""

# Strict API - only these are public
__all__ = [
    "MultiHeadAttention",
    "GroupedQueryAttention",
    "FlashAttention",
    "SlidingWindowAttention",
    "rotary",
    "BaseAttention",  # For extension only
]

from .mha import MultiHeadAttention
from .gqa import GroupedQueryAttention
from .flash import FlashAttention
from .sliding import SlidingWindowAttention
from .base import BaseAttention
from . import rotary

# Private implementation details not exposed
from ._attention_utils import _compute_attention_scores  # Stays private
from ._kernels import _optimized_attention_kernel  # Stays private
```

### API Control Benefits

1. **Prevents Breaking Changes**
```python
# Users cannot do this:
from tt_transformers_v2.src.building_blocks.attention._utils import some_internal_func
# ❌ ImportError - not in public API

# They must use public API:
from tt_transformers_v2 import attention
# ✓ Works - public API
```

2. **Clear Contracts**
```python
# Everything in __all__ is:
# - Public API
# - Documented
# - Stable across minor versions
# - Tested

# Everything NOT in __all__ is:
# - Private implementation
# - Can change without notice
# - Not documented for external use
```

3. **Tooling Support**
```python
# IDEs and linters respect __all__
from tt_transformers_v2 import *  # Only imports what's in __all__

# Type checkers understand the boundary
# Documentation generators only show public API
```

### 4. API Enforcement Strategies

#### Python Conventions
1. **Leading underscore for private modules/functions**: `_utils.py`, `_internal.py`
2. **__all__ in every __init__.py**: Explicitly lists public API
3. **Type stubs (.pyi)**: Can further restrict what's visible to type checkers

#### Testing Enforcement
```python
# tests/test_public_api.py
def test_public_api_complete():
    """Ensure all documented APIs are in __all__"""
    import tt_transformers_v2
    public_api = set(tt_transformers_v2.__all__)
    # Verify against documentation

def test_no_private_imports():
    """Ensure private modules can't be imported"""
    with pytest.raises(ImportError):
        from tt_transformers_v2.src._internal import something
```

#### Documentation
```python
# Only document what's in __all__
# Sphinx/pdoc can be configured to respect __all__
```

### 5. Key Design Decisions

#### A. Strict Dependency Rules
```
src/building_blocks → depends only on TTNN
src/patterns → depends on building_blocks
src/hardware → depends on TTNN
src/interfaces → depends on building_blocks
src/testing → depends on all above

models/ → depends on src/ (reference implementations)
tests/ → depends on src/ (tests real configurations)

External Models → depend on TTTv2 (versioned)
TTTv2 src/ → knows nothing about external models
```

#### B. Configuration Philosophy
- **No model-specific configs in TTTv2**
- Building blocks accept generic parameters
- Models translate their configs to TTTv2 parameters

#### C. Testing Strategy

**Three-tier testing approach:**

1. **Unit Tests (in src/testing/)** - Generic component tests
```python
# src/testing/test_attention_correctness.py
class TestAttentionCorrectness:
    def test_attention_math(self):
        # Test mathematical correctness

    def test_attention_gradients(self):
        # Test gradient computation
```

2. **Configuration Tests (in tests/)** - Real-world configurations
```python
# tests/attention/test_llama_attention.py
from tt_transformers_v2 import attention

class TestLLaMAAttention:
    def test_llama_32_heads_4k_dim(self):
        # Test exact configuration used in LLaMA-7B
        attn = attention.MultiHeadAttention(
            hidden_dim=4096,
            num_heads=32,
            rope_theta=10000.0
        )
        # Verify correctness and performance
```

3. **Model-Driven Testing** - Context Manager + Builder Pattern
```python
# Both internal and external models can use the testing API

# tt_transformers_v2/models/llama3/model.py (internal reference)
# OR
# external_models/llama3/model.py (external model)
from tt_transformers_v2.testing import TestSuite

class LLaMA3Model:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.attention = attention.MultiHeadAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            device=device,
        )
        self.ffn = ffn.SwiGLU(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
            device=device,
        )
        self.norm = normalization.RMSNorm(config.hidden_dim, device=device)

# Test model with context manager + builder pattern
model = LLaMA3Model(config, device)

with TestSuite(model) as suite:
    # Pass module instances directly - type safe!
    suite.test(model.attention) \
        .tolerance(1e-3) \
        .expect(latency_ms=5.0, memory_mb=512)

    suite.test(model.ffn) \
        .tolerance(1e-3) \
        .expect(latency_ms=3.0, memory_mb=256)

    suite.test(model.norm) \
        .tolerance(1e-5)

# Tests run automatically on context exit
# Shape detection is automatic from model instance!

# Can also test nested modules
with TestSuite(model) as suite:
    for i, layer in enumerate(model.layers):
        suite.test(layer.attention, name=f"layer_{i}_attn") \
            .expect(latency_ms=5.0)
```

#### D. Testing API Benefits

**Why Context Manager + Builder Pattern?**

1. **Type Safety**: Using module instances instead of strings
   ```python
   suite.test(model.attention)  # ✓ IDE autocomplete, type checking
   suite.test("atention")       # ✗ Would fail at runtime
   ```

2. **Auto Shape Detection**: Introspects model instance
   ```python
   # No need to specify input shapes!
   # TestSuite detects from model.attention.hidden_dim
   ```

3. **Fluent Interface**: Natural, readable test specs
   ```python
   suite.test(model.ffn).tolerance(1e-3).expect(latency_ms=3.0)
   ```

4. **Zero Boilerplate**: No decorators or methods to implement

5. **Flexible**: Test any module, even nested ones
   ```python
   suite.test(model.layers[0].attention)
   suite.test(model.encoder.mlp)
   ```

#### E. Versioning & Compatibility

**Version Guarantee Matrix:**
```
TTTv2 2.0.x → TTTv2 2.0.y: Binary compatible (patch)
TTTv2 2.0.x → TTTv2 2.1.x: Source compatible (minor)
TTTv2 2.x.x → TTTv2 3.x.x: Migration required (major)
```

**Model Pinning:**
```toml
# external_models/llama3/requirements.txt
tt_transformers_v2==2.1.*  # Pin to minor version
```

### 4. Hardware Abstraction

TTTv2 uses a module-centric approach where each building block provides its own hardware configuration defaults:

```python
# Each module knows its optimal hardware configuration
class MultiHeadAttention(HardwareConfigurableModule):
    @classmethod
    def get_default_hw_config(cls, device_type: str, mode: str = "performance"):
        # Attention-specific hardware optimizations
        # e.g., HiFi4 for attention scores, BFP8 for QKV weights
        return ModuleHardwareConfig(...)

# Simple usage - modules auto-configure
attention = MultiHeadAttention(
    hidden_dim=4096,
    num_heads=32,
    device=ttnn_device,  # Hardware config auto-determined
)

# Or specify mode
attention = MultiHeadAttention(
    hidden_dim=4096,
    num_heads=32,
    device=ttnn_device,
    hw_mode="accuracy"
)

# Query module for its recommendations
config = MultiHeadAttention.get_default_hw_config("T3K", mode="performance")
print(f"Recommended precision: {config.precision['qkv_weights']}")
```

**Key Benefits:**
- Hardware knowledge lives with the module that uses it
- Easy to discover optimal settings for each component
- Progressive learning - understand one module at a time
- Maintains simplicity while enabling fine control

### 5. Extension Points

TTTv2 provides clear extension mechanisms:

```python
# Custom attention variant
from tt_transformers_v2 import attention
from tt_transformers_v2.attention import BaseAttention

class SparseAttention(BaseAttention):
    def compute_attention(self, q, k, v, mask):
        # Custom sparse attention logic
        pass

# Use directly - no registration needed
sparse_attn = SparseAttention(config)
```

### 6. Model Specification & Factory

TTTv2 provides a clean separation between model architecture specification and hardware-specific configurations:

```python
# Pure architecture specification - no hardware details
from tt_transformers_v2.interfaces import ModelSpec, AttentionSpec, FFNSpec
from tt_transformers_v2.interfaces import ModelFactory, ActivationType, NormType

# Convert from HuggingFace
from transformers import AutoConfig
hf_config = AutoConfig.from_pretrained("meta-llama/Llama-3-8b")
model_spec = ModelFactory.from_huggingface(hf_config.to_dict())

# Or create custom specification
model_spec = ModelSpec.uniform(
    vocab_size=32000,
    hidden_dim=4096,
    num_layers=32,
    num_heads=32,
    intermediate_dim=11008,
    activation=ActivationType.SILU,
    norm_type=NormType.RMS_NORM,
)

# Create model with hardware-specific optimizations
model = ModelFactory.from_spec(model_spec, device=ttnn_device)
```

**Benefits:**
- Model architecture is hardware-agnostic
- Easy conversion from HuggingFace configs
- Hardware optimizations applied separately
- Developers can still manually construct models

### 7. Migration

We will not provide migration tools. We will migrate select models to TTTv2.

The rest of the models should migrate to TTTv2 by their owners. We will provide migration guides for once we have migrated select models ourselves.

## Benefits of This Design

1. **Scalability**: Adding model N+1 doesn't affect models 1-N
2. **Testability**: Components tested in isolation
3. **Flexibility**: Models choose their building blocks
4. **Maintainability**: Clear ownership boundaries
5. **Performance**: Hardware abstraction enables optimization
6. **Simplicity**: No tribal knowledge needed

## Success Metrics

- Time to add new model: < 1 week for single developer
- Test suite runtime: O(1) not O(N²)
- Breaking changes: < 1 per year
- Model performance: Within 5% of hand-optimized
- Code reuse: > 80% across models

## Example: Adding a New Model

```bash
# Developer workflow
1. pip install tt_transformers_v2==2.1.*
2. Create model adapter using building blocks
3. Test model in isolation
4. Submit model to model zoo (separate repo)
5. No TTTv2 changes needed!
```

This design achieves the goal of supporting 100+ models by making TTTv2 a stable foundation that models build upon, rather than a monolithic framework that owns all models.

## To-Be-Designed Areas

### 1. Code Coverage in Testing

**Challenge**: How do we ensure that all building blocks are adequately tested across different configurations used by models?

**Requirements**:
- Track which building block configurations are tested by models
- Identify untested configurations or parameter combinations
- Generate coverage reports showing:
  - Which modules have test coverage
  - Which parameter ranges are tested
  - Which combinations are missing

**Open Questions**:
- Should we track coverage at the module level or parameter level?
- How to aggregate coverage across all models using TTTv2?
- Should coverage influence CI/CD decisions?

**Potential Approach**:
```python
# Automatic coverage tracking in TestSuite
with TestSuite(model, track_coverage=True) as suite:
    suite.test(model.attention).expect(latency_ms=5.0)
    # Automatically records that MultiHeadAttention with
    # hidden_dim=4096, num_heads=32 was tested

# Generate coverage report
coverage_report = suite.get_coverage_report()
# Shows: attention.MultiHeadAttention tested with:
#   - hidden_dim: [4096, 8192] (missing: < 2048)
#   - num_heads: [32, 64] (missing: 8, 16)
```

### 2. Model Specification & Factory

Model specification to model implementation still has a gap. For example, on TT device, we typically use different implementation for prefill and decode. Model architecture specification does not typically contain such details. Such details could be considered as part of the model implementation and thus be grouped with hardware-specific configurations? Or it could handled specially because it is a rather stable strategy to specialize forward into prefill_forward and decode_forward?

### 3. Hardware Abstraction and configuration

Needs more detailed design.
Also need to design details for hardware specific tensor caches!

### (finally) double check against the "Library, not Framework" design philosophy

### 4. Code generation

[forward function](TTTv2_BIDIRECTIONAL_CODE_GENERATION.py) at line 348 can be a free function that can be used to generate the source code for the class. It is currently used to generate the source code for the class.

This approach could be combined with the metaclass approach to generate the source code for the class. Or we could make it simpler by specializing a compile_function for each specialized forward function. Such compile_function could generate the specialized forward function that contains concretized configurations based on the tensor shape and hardware configurations
