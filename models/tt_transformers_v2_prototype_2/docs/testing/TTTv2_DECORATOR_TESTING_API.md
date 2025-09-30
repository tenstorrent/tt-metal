# Decorator-Based Testing API for TTTv2

## Design Goals
- Clean, declarative syntax
- Minimal boilerplate
- Clear association between modules and their tests
- Easy to discover what's being tested

## Proposed Decorator API

### Option 1: Class-Level Decorator with Module Names

```python
from tt_transformers_v2.testing import test_modules, TestSpec

@test_modules(
    attention=TestSpec(
        input_shape=(1, 2048, 4096),
        tolerance=1e-3,
        performance_targets={"latency_ms": 5.0, "memory_mb": 512},
    ),
    ffn=TestSpec(
        input_shape=(1, 2048, 4096),
        tolerance=1e-3,
        performance_targets={"latency_ms": 3.0, "memory_mb": 256},
    ),
    norm=TestSpec(
        input_shape=(1, 2048, 4096),
        tolerance=1e-5,
    ),
)
class LLaMA3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # These attribute names match the decorator keys
        self.attention = attention.MultiHeadAttention(...)
        self.ffn = ffn.SwiGLU(...)
        self.norm = normalization.RMSNorm(...)
```

### Option 2: Property Decorators

```python
from tt_transformers_v2.testing import test_module

class LLaMA3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._attention = attention.MultiHeadAttention(...)
        self._ffn = ffn.SwiGLU(...)
        self._norm = normalization.RMSNorm(...)

    @property
    @test_module(
        input_shape=(1, 2048, 4096),
        tolerance=1e-3,
        performance_targets={"latency_ms": 5.0, "memory_mb": 512},
    )
    def attention(self):
        return self._attention

    @property
    @test_module(
        input_shape=(1, 2048, 4096),
        tolerance=1e-3,
        performance_targets={"latency_ms": 3.0, "memory_mb": 256},
    )
    def ffn(self):
        return self._ffn
```

### Option 3: Descriptor-Based (Most Elegant)

```python
from tt_transformers_v2.testing import TestableModule

class LLaMA3Model(nn.Module):
    # Declare testable modules at class level
    attention = TestableModule(
        "attention.MultiHeadAttention",
        input_shape=(1, 2048, 4096),
        tolerance=1e-3,
        performance_targets={"latency_ms": 5.0, "memory_mb": 512},
    )

    ffn = TestableModule(
        "ffn.SwiGLU",
        input_shape=(1, 2048, 4096),
        tolerance=1e-3,
        performance_targets={"latency_ms": 3.0, "memory_mb": 256},
    )

    def __init__(self, config):
        super().__init__()
        # TestableModule descriptor handles the instantiation
        self.attention.build(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            rope_theta=config.rope_theta,
        )
        self.ffn.build(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
        )
```

### Option 4: Builder Pattern with Chaining

```python
from tt_transformers_v2.testing import with_tests

class LLaMA3Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Chain test specs with module creation
        self.attention = (
            with_tests(input_shape=(1, 2048, 4096), tolerance=1e-3)
            .performance(latency_ms=5.0, memory_mb=512)
            .build(attention.MultiHeadAttention(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                rope_theta=config.rope_theta,
            ))
        )

        self.ffn = (
            with_tests(input_shape=(1, 2048, 4096), tolerance=1e-3)
            .performance(latency_ms=3.0, memory_mb=256)
            .build(ffn.SwiGLU(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
            ))
        )
```

### Option 5: Context Manager (My Favorite)

```python
from tt_transformers_v2.testing import TestContext

class LLaMA3Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Context manager captures test specs
        with TestContext(self) as test:
            # Build and automatically register for testing
            self.attention = test(
                attention.MultiHeadAttention(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    rope_theta=config.rope_theta,
                ),
                input_shape=(1, 2048, 4096),
                tolerance=1e-3,
                latency_ms=5.0,
                memory_mb=512,
            )

            self.ffn = test(
                ffn.SwiGLU(
                    hidden_dim=config.hidden_dim,
                    intermediate_dim=config.intermediate_dim,
                ),
                input_shape=(1, 2048, 4096),
                tolerance=1e-3,
                latency_ms=3.0,
                memory_mb=256,
            )

            # Modules without test() are not tested
            self.norm = normalization.RMSNorm(config.hidden_dim)
```

## Implementation for Option 1 (Class Decorator)

```python
# tt_transformers_v2/src/testing/decorators.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TestSpec:
    input_shape: tuple
    tolerance: float = 1e-3
    expected_output_shape: Optional[tuple] = None
    performance_targets: Optional[Dict[str, float]] = None

def test_modules(**module_specs: TestSpec):
    """Decorator to specify which modules to test in a model."""
    def decorator(cls):
        # Store test specs on the class
        cls._test_specs = module_specs

        # Add method to retrieve test configs
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Validate that specified modules exist
            for module_name in module_specs:
                if not hasattr(self, module_name):
                    raise AttributeError(
                        f"Model {cls.__name__} has test spec for '{module_name}' "
                        f"but no such attribute exists"
                    )

        cls.__init__ = new_init

        # Add test discovery method
        def get_test_configs(self):
            configs = {}
            for module_name, spec in self._test_specs.items():
                module = getattr(self, module_name)
                configs[f"{self.__class__.__name__}_{module_name}"] = (module, spec)
            return configs

        cls.get_test_configs = get_test_configs
        return cls

    return decorator

# Usage
@test_modules(
    attention=TestSpec((1, 2048, 4096), tolerance=1e-3),
    ffn=TestSpec((1, 2048, 4096), tolerance=1e-3),
)
class LLaMA3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = attention.MultiHeadAttention(...)
        self.ffn = ffn.SwiGLU(...)
```

## Running Tests

```python
# Automatic test discovery
from tt_transformers_v2.testing import discover_and_run_tests

# Find all classes with @test_modules decorator
results = discover_and_run_tests("models.llama3")

# Or explicit
model = LLaMA3Model(config)
for name, (module, spec) in model.get_test_configs().items():
    print(f"Testing {name}...")
    test_module_with_spec(module, spec)
```

## Benefits of Decorator Approach

1. **Declarative**: Test specs are clearly visible at class level
2. **Validated**: Can check that module names match actual attributes
3. **Discoverable**: Easy to find all models with tests
4. **DRY**: No need to implement `get_test_configs` in each model
5. **Flexible**: Can mix with other testing approaches

## Advanced Usage

```python
# Conditional testing based on model size
@test_modules(
    attention=TestSpec(
        input_shape=lambda self: (1, 2048, self.config.hidden_dim),
        tolerance=1e-3,
        performance_targets=lambda self: {
            "latency_ms": 5.0 if self.config.model_size == "7b" else 10.0
        }
    )
)
class ScalableModel(nn.Module):
    pass
```
