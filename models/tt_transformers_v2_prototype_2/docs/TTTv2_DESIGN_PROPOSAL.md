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

#### Model Specification to Model Implementation Gap
Model specification to model implementation still has a gap. For example, on TT device, we typically use different implementation for prefill and decode. Model architecture specification does not typically contain such details. Such details could be considered as part of the model implementation and thus be grouped with hardware-specific configurations? Or it could handled specially because it is a rather stable strategy to specialize forward into prefill_forward and decode_forward?

##### Examples of dataflow extraction for Qwen2.5-VL
Based on the model structure you provided for Qwen2.5-VL, I've created several approaches
  to extract dataflow information from HuggingFace models:

  Summary of Methods:

  1. Forward Hooks Method (dataflow_extraction_methods.py):
    - Uses PyTorch's register_forward_hook to trace execution
    - Captures input/output shapes and module execution order
    - Works with any HuggingFace model
  2. Static Analysis (qwen25_vl_dataflow_extractor.py):
    - Analyzes the model structure without running it
    - Specifically tailored for Qwen2.5-VL architecture
    - Provides detailed dataflow for visual pipeline, language pipeline, and fusion points
  3. Dynamic Tracing (extract_hf_model_dataflow.py):
    - Intercepts actual forward() method calls
    - Captures the true execution flow including dynamic branches
    - Builds a call graph showing module dependencies
  4. Practical Examples (practical_dataflow_example.py):
    - Simple, ready-to-use functions for dataflow extraction
    - Works with any HuggingFace model
    - Includes visualization and export capabilities

  Key Insights for Qwen2.5-VL Dataflow:

  From the model structure, the dataflow follows this pattern:

  Visual Pipeline:
  Pixel Values → Conv3D Patch Embed → Rotary Pos Emb → 32 Vision Blocks → Patch Merger
  [B,3,T,H,W] → [B,1280,*,*,*] → [B,patches,1280] → [B,merged,2048]

  Language Pipeline:
  Input IDs → Token Embeddings → 36 Decoder Layers → Layer Norm → LM Head
  [B,seq] → [B,seq,2048] → [B,seq+visual,2048] → [B,seq+visual,151936]

  Fusion Point:
  - Visual features from merger are concatenated with text embeddings
  - Language decoder layers perform cross-attention over both modalities

  The official ways to extract this information are:
  1. TorchScript tracing - Creates a static graph but may miss dynamic behavior
  2. ONNX export - Provides a standardized computational graph
  3. Forward hooks - Most flexible, captures actual runtime behavior
  4. Model introspection - Analyze the model structure programmatically

From the above example, I would like to propose the design for TTTv2 to completely separate the model architecture specification from the model implementation including the model spec extraction. Model implementation is responsible for the model spec extraction and then the mapping from the extract model spec to instances of TTTv2 building blocks and patterns. TTTv2 provides extension machenisms to register new modules and patterns, such that they can leverage the rest of TTTv2 (including building blocks, patterns, testing API, etc.).

#### Designing TTTv2 Model Specification to be Standard-Agnostic

  Refer to [this doc for details](models/tt_transformers_v2_prototype_2/docs/model_spec/MODEL_SPEC_STANDARDS_ANALYSIS.md)

  I've created a design that addresses your concern about being tied to external standards.
  The key principles are:

  1. Adapter Pattern: Each standard gets an adapter that converts to TTTv2's minimal internal
   format
  2. Minimal Internal Spec: TTTv2 only needs layer types, parameters, and connections
  3. Plugin System: Users can add support for new formats without modifying TTTv2
  4. Lazy Conversion: Convert only what's needed when it's needed

  The design allows:
  model = TTTModelFactory.from_format("huggingface", hf_config)
  model = TTTModelFactory.from_format("onnx", onnx_model)
  model = TTTModelFactory.from_format("custom", custom_data)

  This way, TTTv2 remains flexible and isn't locked into any particular standard, while still
   benefiting from the ecosystem of existing formats.

  We need to first provide model spec for each TTTv2 building block and pattern, such that the model implementation can have a clean interface to specify the module specification during model instantiation. The model factory will be a good addition on top of the model spec to create a model instance automatically from a external-standard model spec.

#### Model Specification & Factory

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

#### Hardware Abstraction

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

#### Extension Points

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

#### API Control Benefits

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

#### F. Module specification API
Based on the current design, we can assume that the model implementation will be responsible for the model spec extraction and then the mapping from the extract model spec to instances of TTTv2 building blocks and patterns, including the dataflow and the module specification.

We need to further design the details of the module specification API for each module within TTTv2 such that the model implementation can have a clean interface to specify the module specification during model instantiation.

Idea: from model spec to module spec is a good time to associate each TTTv2 module with a reference module from the model spec. For example, if model spec is constructed from huggingface model, we can associate each TTTv2 module with a reference module from the huggingface model spec. If the model spec is custom made, the deverlopers will need to provide a reference model for each TTTv2 module -- in this case, the reference module association should be optional. Maybe the only required reference module would be determined by the debugging process.

##### Core Design: Fixed Public API
  [Refer to this doc for details](models/tt_transformers_v2_prototype_2/docs/model_spec/module_spec_api_design.md) and this code for details: [reference_tracking_example.py](models/tt_transformers_v2_prototype_2/docs/model_spec/reference_tracking_example.py)

  @dataclass(frozen=True)
  class AttentionSpec:
      """Minimal, stable public API"""
      hidden_dim: int
      num_heads: int
      # That's it! No hardware details

  Benefits:
  1. Stable Interface: Users only provide essential math
  parameters
  2. Internal Freedom: Module implementations can evolve without
  breaking changes
  3. Clear Separation: Spec defines "what", module decides "how"

  Key Enhancement: Reference Tracking

  Your idea about associating TTT modules with reference
  implementations is brilliant for debugging:

  attention_spec = AttentionSpec(
      hidden_dim=4096,
      num_heads=32,
      reference_path="model.layers.15.self_attn",  # From HF
  model
      reference_type="huggingface"
  )

  During debugging, automatically compare outputs
  attention.compare_with_reference(test_input)
  > Max difference: 1.2e-5
  > All close: True ✓

  Alternative Approaches Considered:

  1. Builder Pattern: More verbose but allows validation during
  construction
  2. Protocol-Based: Maximum flexibility but less discoverable
  3. Hierarchical: Good for complex inheritance but adds
  complexity

  Recommendation: Hybrid Approach

  Combine your fixed API idea with:
  - Immutable dataclasses for specs (clarity)
  - Optional reference tracking (debugging)
  - Module-specific parameter distribution (flexibility)

  This gives us:
  - Simple for users: Just provide math parameters
  - Powerful for debugging: Reference comparison built-in
  - Flexible for implementers: Modules control internal details

  The key insight is that modules are the right abstraction level
   for hiding complexity - they know both their mathematical
  requirements AND hardware optimization strategies.

  ##### Potential issue: Model Spec Extraction Gap

  The design acknowledges but doesn't fully address the gap between model architecture
  specification and implementation details. The separation between "what"
  (architecture) and "how" (prefill vs decode strategies) needs clearer boundaries.

  Key Design Decisions:

  [Refer to this doc for details](models/tt_transformers_v2_prototype_2/docs/model_spec/execution_strategy_design.md) and this code for details: [prefill_decode_example.py](models/tt_transformers_v2_prototype_2/docs/model_spec/prefill_decode_example.py)

  1. Prefill/Decode as First-Class Citizens
   Natural API for TT hardware
  attention.prefill_forward(prompt_tokens)
  attention.decode_forward(new_token, cache_position)
  2. Specialization at Module Creation
  def __init__(self, spec, device):
      Automatically create both implementations
      self._create_prefill_implementation()  # Optimized for
  throughput
      self._create_decode_implementation()   # Optimized for
  latency + cache
  3. Extensible Through Strategy Pattern
  Default strategy (baked into TTTv2)
  strategy = PrefillDecodeStrategy()

  Future strategies can be added without refactoring
  strategy = SpeculativeDecodingStrategy()  # Future addition
  strategy = ContinuousBatchingStrategy()   # Another future
  addition

  Why This Works:

  1. Clean Separation: Module specs stay pure (just math),
  execution strategies handle the "how"
  2. Progressive Disclosure:
    - Simple API: prefill_forward() / decode_forward()
    - Auto-detection: forward() picks the right mode
    - Advanced: Custom strategies for future needs
  3. Hardware Optimization: Each mode can have completely
  different implementations:
    - Prefill: Large tile sizes, batch processing
    - Decode: Small tiles, KV cache, single token
  4. Future-Proof: New execution strategies can be added without
  changing existing module interfaces

  The design successfully bridges the gap by introducing an
  "execution strategy" layer between pure specs and hardware
  configs, making prefill/decode specialization feel native to
  TTTv2 while keeping doors open for future innovations.

#### G. Hardware Abstraction and configuration

[Refer to this doc for details](hw_config/hardware_config_detailed_design.md) and this code for details: [hardware_config_example.py](hw_config/hardware_config_example.py)

Key Design Decisions:

  1. Configuration After Specialization
  Module Spec → Prefill/Decode Specialization → Hardware Config →
   Tensor Caches
  1. Hardware config applies to already-specialized ops,
  maintaining clean separation
  2. Direct TTNN Op Configuration
    - No over-abstraction initially (pragmatic approach)
    - Each module has device-specific defaults for its TTNN ops
    - Users can override individual ops when needed
  3. Tensor Cache Strategy
    - Weight caches: Created at config time when converting from
  reference model
    - Activation caches: Pre-allocated at compile time when
  shapes are known
    - Both use hardware-optimized layouts and dtypes

  Why This Works Well:

  1. Practical: Direct access to TTNN op configs without
  unnecessary abstraction layers
  2. Device-Aware: Each module knows optimal settings per device:
  "ttnn:n150": {"prefill": {"use_dram": True}}
  "ttnn:n300": {"prefill": {"use_dram": False}}
  3. Override Flexibility:
  Override just what you need
  hw_config_overrides = {
      'qkv_matmul': {'dtype': 'float32'}  # Higher precision for this op only
  }
  4. Future-Proof: Clear extension points for future TTNN
  abstraction:
    - Could add high-level optimization targets later
    - Could add auto-tuning capabilities
    - Could add declarative config languages

  Tensor Cache Benefits:

  - Weight caches: Avoid repeated PyTorch→TTNN conversions
  - Activation caches: Reuse buffers across forward passes
  - Hardware-optimized: Use appropriate dtypes/layouts per device

  This design gives you the control needed today while leaving
  room for tomorrow's abstractions!

#### H. Code generation

[forward function](codegen/TTTv2_BIDIRECTIONAL_CODE_GENERATION.py) at line 348 can be a free function that can be used to generate the source code for the class. It is currently used to generate the source code for the class. This free function is like a template! Future TTNN ops could be hidden behind a free function as well that serves as a template for the source code generation, which gives us a chance to adjust the APIs and have room for working around TTNN limitations.

This approach could be combined with the metaclass approach to generate the source code for the class. Or we could make it simpler by specializing a compile_function for each specialized forward function. Such compile_function could generate the specialized forward function that contains concretized configurations based on the tensor shape and hardware configurations

### 6. Migration

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

## Future Design Options

  #### (finally) double check against the "Library, not Framework" design philosophy

  #### Community Governance - Clear process for accepting new patterns/modules

  Refer to [this doc for details](models/tt_transformers_v2_prototype_2/docs/community_governance_design.md)

  Community governance is critical for TTTv2's success in supporting 100+ models. Key considerations:

  1. **API Stability vs Innovation**
     - Strict semantic versioning with experimental namespaces
     - Modules can graduate from experimental → contrib → core
     - Breaking changes only allowed in major versions

  2. **Quality Control Through Tiers**
     ```python
     class ModuleTier(Enum):
         CORE = "core"              # TTT team maintained
         CONTRIB = "contrib"        # Community reviewed
         COMMUNITY = "community"    # Community maintained
         EXPERIMENTAL = "experimental"  # No guarantees
     ```

  3. **Namespacing for Scale**
     ```python
     # Hierarchical names prevent conflicts
     attention="stanford-nlp/attention/efficient/linear-v2"
     attention="deepmind/attention/perceiver-ar"
     ```

  4. **Automated Governance**
     - Performance regression detection
     - Dependency conflict checking
     - Abandoned module detection
     - Compatibility matrix generation

  5. **Clear Decision Process**
     - Technical Steering Committee for core changes
     - Module maintainers for contrib
     - Automated merge for experimental (if tests pass)

  6. **Certification Levels**
     - Functionally tested
     - Performance verified
     - Hardware validated
     - Production certified

  Impact on Design:
  - Clear package structure: src/ (stable), contrib/, experimental/
  - Versioned module specifications
  - Policy as code for automated enforcement
  - Performance contracts to prevent regressions


  #### Explicit Interface Contracts

  Instead of relying on duck typing, define explicit protocols/interfaces:

  from typing import Protocol

  class TTTModule(Protocol):
      """Base protocol all TTT modules must implement"""
      def forward(self, x: Tensor, **kwargs) -> Tensor: ...
      def get_config(self) -> Dict[str, Any]: ...
      def estimate_memory(self, input_shape: Tuple) -> int: ...

  #### Declarative Model Construction

  Instead of manual wiring, consider a declarative approach:

  More maintainable for 100+ models
  model_def = {
      "layers": [
          {"type": "embedding", "vocab_size": 32000, "dim": 4096},
          {"type": "transformer", "repeat": 32, "config": {...}},
          {"type": "lm_head", "vocab_size": 32000}
      ]
  }
  model = ModelFactory.from_definition(model_def, device=device)

  #### Strengthen the Library Philosophy

  - Remove the models/ directory from TTTv2 core - reference implementations should live in
  separate repos
  - This reinforces that TTTv2 doesn't "own" any models

  #### Version-Specific Test Suites

  Models can pin their test expectations to TTTv2 versions
  with TestSuite(model, version="2.1") as suite:
      suite.test(model.attention).expect_compatible_with("2.1")

  #### Lazy Module Construction

  Refer to [this doc for details](models/tt_transformers_v2_prototype_2/docs/model_spec/lazy_module_construction_design.md) and this code for details: [tttv2_lazy_module_implementation.py](models/tt_transformers_v2_prototype_2/docs/model_spec/tttv2_lazy_module_implementation.py)

  For better memory efficiency with 100+ models:
  Modules aren't materialized until needed
  attention = LazyModule(MultiHeadAttention, config={...})
  attention.materialize(device)  # Only when needed

  #### Extension Registration Ambiguity

  Key Design Points:

  1. No Registration Required for Basic Use
    - Custom modules work immediately through inheritance
    - No barriers to entry - just inherit and use
  class MyAttention(BaseAttention):
      def compute_attention(self, q, k, v, mask=None):
          # Custom logic

  Use immediately!
  attn = MyAttention(hidden_dim=4096, num_heads=32)
  2. Optional Registration for Enhanced Features
    - Discovery through search/list APIs
    - Automatic documentation generation
    - Configuration validation
    - Performance hints and metadata
    - Community sharing capabilities
  3. Benefits of Registration
    - Discoverability: ttt list-modules --category attention
    - Documentation: Auto-generated from metadata
    - Validation: Config checking against schema
    - CLI Support: ttt search "flash attention"
    - Community Hub: Share and discover modules
  4. Flexible Usage Patterns
  Mix registered and unregistered modules
  model = ModelBuilder()
      .add_layer(
          attention="flash-attention-v3",  # Registered by name
          ffn=MyCustomFFN,                 # Direct class
  reference
          norm="ttnn-fused-rmsnorm"        # Registered by name
      )
  5. Clear Documentation
    - The registry is presented as a "value-add" feature
    - Examples show both patterns working side-by-side
    - No confusion about what's required vs optional

  This design maintains the flexibility of direct instantiation
  while providing a pathway for community building and module
  discovery, resolving the ambiguity in the original proposal.

  #### interface to netmap or other higher level model interfaces
netmap: https://docs.google.com/document/d/1KmaawWqk4nZ_cBQoc8LBE6EYbXwjjD8A8FBA7vTen_g/edit?tab=t.0

netmap wants to build compute graph with ttnn ops. The design in this doc can apply to ttnn ops as well! The other contribution that netmap has is that it can help people create sharding specs but I think programmatical ways such as the CuTe DSL is preferrable in scaling scenarioes.

Other ideas in netmap like the use of ttnn trace tool to trace HF models is also interesting and also adoptable by TTTv2.

TTTv2 as a library provides the flexibility to choose different higher level model interfaces --> we can build one too!

#### refactor hardware configs of TTNN ops for each module

After we get things working with the current design, we should spend some time to study and refactor the hardware configs of TTNN ops for each module. Maybe there is a smaller number of patterns for config than there are models! This could be a good thing to create an abstraction layer for hardware configs -- patching up the only leak in the abstract interface in this design.

For exampl, instead of specifying config for a TTNN matmul op, we could say RingMatmulConfig, BcastMatMul, etc.

#### Code Coverage in Testing

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
