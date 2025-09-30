# TTTv2 Structure Summary

## Directory Organization

```
tt_transformers_v2/
├── src/                    # Core library (the actual TTTv2)
│   ├── building_blocks/    # Atomic components
│   ├── patterns/          # Common compositions
│   ├── hardware/          # HW abstraction
│   ├── interfaces/        # Standard interfaces
│   └── testing/           # Testing utilities
│
├── models/                # Reference implementations
│   ├── llama3/           # Maintained by TTTv2 team
│   └── mistral/          # Maintained by TTTv2 team
│
└── tests/                 # Real-world config tests
    ├── attention/        # Test actual attention configs
    ├── ffn/              # Test actual FFN configs
    └── integration/      # Integration tests
```

## Key Principles

### 1. Clear Separation
- **src/**: Pure library code, no model-specific logic
- **models/**: Select reference implementations we maintain
- **tests/**: Tests for configurations actually used in models

### 2. Import Patterns

**Clean public API via __init__.py files:**
```python
# Both internal and external models use the same clean imports
from tt_transformers_v2 import attention, ffn, normalization, patterns

# Or specific imports
from tt_transformers_v2.attention import MultiHeadAttention
from tt_transformers_v2.ffn import SwiGLU
from tt_transformers_v2.normalization import RMSNorm

# The src/ directory is hidden from users
# __init__.py files handle the mapping:
# tt_transformers_v2/attention/__init__.py imports from ../src/building_blocks/attention/
```

### 3. Testing Strategy

| Directory | Purpose | Example |
|-----------|---------|---------|
| `src/testing/` | Test framework & utilities | `TestConfig`, `run_module_tests()` |
| `tests/` | Pre-written config tests | "Does LLaMA's 32-head attention work?" |
| Model code | Test specifications | Model specifies what to test via `get_test_configs()` |

**Key Innovation: Context Manager + Builder Pattern Testing**
```python
# Create model and test with fluent API
model = MyModel(config)

with TestSuite(model) as suite:
    # Pass module instances directly - type safe!
    suite.test(model.attention) \
        .tolerance(1e-3) \
        .expect(latency_ms=5.0, memory_mb=512)

    suite.test(model.ffn) \
        .expect(latency_ms=3.0)  # Auto-detects shapes!

# Tests run automatically on context exit
```

Benefits:
- **Type Safety**: Direct module references with IDE support
- **Auto Shape Detection**: Introspects model instance
- **Fluent Interface**: Natural method chaining
- **Zero Boilerplate**: No decorators or special methods
- **Flexible**: Test any module, even nested ones
- **Context Safety**: Automatic execution and cleanup

### 4. Migration Approach

- We maintain select models in `models/` as references
- These serve as examples for external model migration
- External models remain outside TTTv2 repository
- Migration guides based on our reference implementations

## Benefits of This Structure

1. **Clean separation**: Library vs models vs tests
2. **Clear ownership**: TTTv2 team owns src/ and select models/
3. **Realistic testing**: tests/ contains real-world configurations
4. **Migration examples**: models/ provides migration templates
5. **Versioning clarity**: External models depend on public API only
6. **Clean imports**: No `.src.` in import paths thanks to __init__.py
7. **API stability**: Public API defined in __init__.py files
8. **Implementation flexibility**: Can reorganize src/ without breaking public API

## Strict Public API Control

The public API is **strictly controlled** through `__init__.py` files:

### Control Mechanisms

1. **`__all__` in every `__init__.py`**: Explicitly defines public API
2. **Private modules use `_` prefix**: `_utils.py`, `_internal.py`
3. **No direct src/ access**: Top-level `__init__.py` hides internal structure

### How It Works

```python
# ✅ ALLOWED - Public API only
from tt_transformers_v2 import attention
from tt_transformers_v2.attention import MultiHeadAttention

# ❌ BLOCKED - Internal implementation
from tt_transformers_v2.src.building_blocks import attention  # No!
from tt_transformers_v2.attention._utils import internal_func  # No!
```

### Benefits

1. **API Stability**: Only items in `__all__` are public; everything else can change
2. **Clear Contracts**: Public vs private is explicit, not implicit
3. **Tool Support**: IDEs, linters, and docs respect `__all__`
4. **Prevents Accidents**: Can't accidentally depend on internal implementation
5. **Easier Refactoring**: Internal changes don't break external users

### Example API Definition

```python
# tt_transformers_v2/attention/__init__.py
__all__ = [
    "MultiHeadAttention",    # ✓ Public class
    "GroupedQueryAttention", # ✓ Public class
    "BaseAttention",         # ✓ Public base class for extensions
    # _attention_utils       # ✗ Not exported, stays private
    # _compute_scores        # ✗ Not exported, stays private
]
```
