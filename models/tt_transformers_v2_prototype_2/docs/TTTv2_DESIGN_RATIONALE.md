# TTTv2 Design Rationale

## How This Design Solves the Core Problems

### Problem 1: Models × Platforms × Tests Explosion

**TTTv1 Problem:**
```
Adding LLaMA-3 requires testing:
- LLaMA-3 works with all platforms ✓
- LLaMA-3 doesn't break Mistral ✓
- LLaMA-3 doesn't break Gemma ✓
- LLaMA-3 doesn't break Qwen ✓
- ... (test with all 10+ existing models)
```

**TTTv2 Solution:**
```
Adding LLaMA-3 requires testing:
- LLaMA-3 works with TTTv2 2.1.* ✓
- Done! (No cross-model testing needed)
```

### Problem 2: Tribal Knowledge Required

**TTTv1 Reality:**
- "Don't change that function, it breaks Mistral's generation"
- "This optimization only works on Wormhole for models < 30B"
- "You need to test with Sarah's branch for the new attention"

**TTTv2 Reality:**
- Building blocks have contracts (input/output/behavior)
- Hardware abstraction hides platform details
- Stable releases with semantic versioning

### Design Principles Mapped to Goals

| Goal | Design Solution |
|------|----------------|
| **Modular** | Separate building_blocks/ with single responsibilities |
| **Composable** | Blocks combine via patterns/, models choose combinations |
| **Readable** | Clear directory structure, one concept per file |
| **Reproducible** | Pinned versions, deterministic behaviors |
| **Maintainable** | Models outside TTTv2, clear ownership |
| **Releasable** | Semantic versioning, compatibility guarantees |

## Key Architectural Decisions

### 1. Building Blocks vs Monolithic Modules

**Why Building Blocks?**
```python
# TTTv1: Monolithic
class Attention:
    def __init__(self, config):
        # 500 lines handling all attention variants
        if config.model_type == "llama":
            # LLaMA-specific code
        elif config.model_type == "mistral":
            # Mistral-specific code
        # ... grows with each model

# TTTv2: Composable
# building_blocks/attention/mha.py
class MultiHeadAttention:
    # 100 lines, just MHA

# building_blocks/attention/gqa.py
class GroupedQueryAttention:
    # 100 lines, just GQA

# Models choose what they need
```

### 2. Hardware Abstraction Layer

**Why Abstract Hardware?**
```python
# TTTv1: Hardware details leak everywhere
if is_wormhole_b0():
    tile_size = 32
    use_special_kernel = True
elif is_grayskull():
    tile_size = 16
    use_special_kernel = False

# TTTv2: Hardware abstraction
config = DeviceConfig.auto_detect()
optimal_params = config.get_optimal_params("attention")
```

### 3. Models as External Consumers

**Why External?**
- **Independence**: Models can evolve without touching TTTv2
- **Versioning**: Models pin to stable TTTv2 versions
- **Testing**: Models test themselves, TTTv2 tests itself
- **Ownership**: Clear responsibility boundaries

### 4. No Model-Specific Code in Core

**TTTv1 Anti-pattern:**
```python
# In core TTT code
if self.config.model_type == "llama":
    scale = 1.0 / math.sqrt(self.head_dim)
elif self.config.model_type == "gpt2":
    scale = 1.0 / math.sqrt(self.head_dim * 2)  # GPT2 quirk
```

**TTTv2 Pattern:**
```python
# In core TTTv2
class Attention:
    def __init__(self, scale_factor):
        self.scale = scale_factor  # Model provides

# In model code
attention = Attention(scale_factor=1.0/math.sqrt(head_dim))
```

## Testing Philosophy

### Three-Tier Testing Strategy

#### 1. Component Testing (src/testing/)
```
src/testing/
├── test_correctness.py    # Math is correct
├── test_performance.py    # Meets speed targets
├── test_memory.py         # Memory usage bounds
└── test_compatibility.py  # API stability
```

#### 2. Configuration Testing (tests/)
```
tests/
├── attention/
│   ├── test_llama_attention.py    # LLaMA's exact configs
│   ├── test_mistral_attention.py  # Mistral's sliding window
│   └── test_gpt_attention.py      # GPT's standard MHA
└── ffn/
    ├── test_swiglu.py             # Real SwiGLU configs
    └── test_standard_mlp.py       # Real MLP configs
```

#### 3. Model Testing (models/ or external)
```
models/llama3/
├── test_accuracy.py       # Model accuracy
├── test_generation.py     # Generation quality
└── test_performance.py    # End-to-end speed
```

### Key Testing Insights

1. **Component Tests**: "Does MultiHeadAttention work correctly for any valid config?"
2. **Configuration Tests**: "Does MultiHeadAttention work for LLaMA's specific config?"
3. **Model Tests**: "Does the complete LLaMA-3 model work correctly?"

This separation ensures:
- Components are robust across all valid inputs
- Real-world configurations are explicitly tested
- Models can trust that their specific use cases work

## Migration Strategy

### Phase 1: Parallel Development
- TTTv2 developed alongside TTTv1
- New models use TTTv2
- Existing models stay on TTTv1

### Phase 2: Gradual Migration
- Migration tools for common patterns
- Model-by-model migration
- Both versions supported

### Phase 3: Deprecation
- TTTv1 deprecated after all models migrate
- Long deprecation window
- Clear migration paths

## Success Metrics Validation

| Metric | How Design Achieves It |
|--------|----------------------|
| Add model < 1 week | Use building blocks, no core changes |
| O(1) test time | Test components independently |
| < 1 breaking change/year | Semantic versioning, stable APIs |
| 5% of hand-optimized | Hardware abstraction enables optimization |
| 80% code reuse | Shared building blocks across models |

## Conclusion

This design transforms TTT from a monolithic framework that owns all models to a modular library that models consume. This fundamental shift solves the scaling problem by making the complexity O(N) instead of O(N²).
