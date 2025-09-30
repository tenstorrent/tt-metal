# TTTv2 Testing API Design

## Goal
Allow model developers to specify and run tests for their specific module configurations without writing new test code.

## Proposed API

### Option 1: Test Registry Pattern

```python
# models/llama3/model.py
from tt_transformers_v2 import attention, ffn, normalization
from tt_transformers_v2.testing import register_module_tests

class LLaMA3Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Build model components
        self.attention = attention.MultiHeadAttention(
            hidden_dim=4096,
            num_heads=32,
            rope_theta=10000.0,
        )

        self.ffn = ffn.SwiGLU(
            hidden_dim=4096,
            intermediate_dim=11008,
        )

        # Register these specific configurations for testing
        register_module_tests(self, "llama3_7b")

# Or more explicit registration
@register_module_tests("llama3_7b")
def get_test_modules():
    return {
        "attention": attention.MultiHeadAttention(
            hidden_dim=4096,
            num_heads=32,
            rope_theta=10000.0,
        ),
        "ffn": ffn.SwiGLU(
            hidden_dim=4096,
            intermediate_dim=11008,
        ),
        "norm": normalization.RMSNorm(
            hidden_dim=4096,
            eps=1e-6,
        ),
    }

# Run tests
$ pytest tt_transformers_v2/tests --model-config llama3_7b
```

### Option 2: Test Specification Files

```python
# models/llama3/test_config.py
from tt_transformers_v2.testing import ModuleTestSpec

LLAMA3_7B_TESTS = [
    ModuleTestSpec(
        name="llama3_7b_attention",
        module_class="attention.MultiHeadAttention",
        config={
            "hidden_dim": 4096,
            "num_heads": 32,
            "rope_theta": 10000.0,
        },
        expected_performance={
            "latency_ms": 5.0,
            "memory_mb": 512,
        },
        test_cases=["correctness", "gradients", "performance"],
    ),
    ModuleTestSpec(
        name="llama3_7b_ffn",
        module_class="ffn.SwiGLU",
        config={
            "hidden_dim": 4096,
            "intermediate_dim": 11008,
        },
        expected_performance={
            "latency_ms": 3.0,
            "memory_mb": 256,
        },
    ),
]

# Run via pytest
$ pytest tt_transformers_v2/tests --test-spec models/llama3/test_config.py
```

### Option 3: Inline Test Decorators (My Preference)

```python
# models/llama3/model.py
from tt_transformers_v2 import attention, ffn, normalization
from tt_transformers_v2.testing import test_module

class LLaMA3Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Decorate modules with test specifications
        self.attention = test_module(
            attention.MultiHeadAttention(
                hidden_dim=4096,
                num_heads=32,
                rope_theta=10000.0,
            ),
            test_name="llama3_7b_attention",
            expected_performance={
                "latency_ms": 5.0,
                "memory_mb": 512,
                "accuracy": 1e-3,
            },
            test_batch_size=1,
            test_seq_len=2048,
        )

        self.ffn = test_module(
            ffn.SwiGLU(
                hidden_dim=4096,
                intermediate_dim=11008,
            ),
            test_name="llama3_7b_ffn",
            expected_performance={
                "latency_ms": 3.0,
                "memory_mb": 256,
            },
        )

# The test_module decorator:
# 1. Wraps the module without changing behavior
# 2. Registers it for testing with metadata
# 3. Can be discovered and tested automatically
```

### Option 4: Test Methods on Model Class

```python
# models/llama3/model.py
from tt_transformers_v2.testing import ModuleTest, run_module_tests

class LLaMA3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... build model ...

    def get_module_tests(self) -> List[ModuleTest]:
        """Define tests for this model's module configurations."""
        return [
            ModuleTest(
                module=self.attention,
                name="llama3_attention",
                input_shape=(1, 2048, 4096),
                expected_output_shape=(1, 2048, 4096),
                tolerance=1e-3,
            ),
            ModuleTest(
                module=self.ffn,
                name="llama3_ffn",
                input_shape=(1, 2048, 4096),
                expected_output_shape=(1, 2048, 4096),
                tolerance=1e-3,
            ),
        ]

# Run tests
model = LLaMA3Model(config)
results = run_module_tests(model.get_module_tests())
```

## Recommended Approach: Hybrid of 3 & 4

```python
# tt_transformers_v2/testing/module_testing.py
@dataclass
class TestConfig:
    """Configuration for testing a module instance."""
    name: str
    input_shape: Tuple[int, ...]
    expected_output_shape: Optional[Tuple[int, ...]] = None
    tolerance: float = 1e-3
    performance_targets: Optional[Dict[str, float]] = None
    test_cases: List[str] = field(default_factory=lambda: ["correctness", "performance"])

class TestableModule:
    """Mixin for models that want to expose testable modules."""

    def get_test_configs(self) -> Dict[str, Tuple[nn.Module, TestConfig]]:
        """Return modules and their test configurations."""
        raise NotImplementedError

# models/llama3/model.py
class LLaMA3Model(nn.Module, TestableModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Build modules
        self.attention = attention.MultiHeadAttention(...)
        self.ffn = ffn.SwiGLU(...)
        self.norm = normalization.RMSNorm(...)

    def get_test_configs(self):
        """Specify which modules to test and how."""
        return {
            "llama3_7b_attention": (
                self.attention,
                TestConfig(
                    name="llama3_7b_attention",
                    input_shape=(1, 2048, self.config.hidden_dim),
                    tolerance=1e-3,
                    performance_targets={
                        "latency_ms": 5.0,
                        "memory_mb": 512,
                    },
                ),
            ),
            "llama3_7b_ffn": (
                self.ffn,
                TestConfig(
                    name="llama3_7b_ffn",
                    input_shape=(1, 2048, self.config.hidden_dim),
                    tolerance=1e-3,
                    performance_targets={
                        "latency_ms": 3.0,
                        "memory_mb": 256,
                    },
                ),
            ),
        }

# Running tests
from tt_transformers_v2.testing import run_module_tests

model = LLaMA3Model(config)
results = run_module_tests(model)

# Or via pytest plugin
$ pytest --test-model models/llama3/model.py::LLaMA3Model
```

## Test Implementation

```python
# tt_transformers_v2/src/testing/module_tester.py
def run_module_tests(model: TestableModule) -> TestResults:
    """Run all tests for modules specified by the model."""
    results = TestResults()

    for test_name, (module, config) in model.get_test_configs().items():
        print(f"Testing {test_name}...")

        # Run correctness test
        if "correctness" in config.test_cases:
            result = test_correctness(module, config)
            results.add(test_name, "correctness", result)

        # Run performance test
        if "performance" in config.test_cases:
            result = test_performance(module, config)
            results.add(test_name, "performance", result)

        # Compare against targets
        if config.performance_targets:
            verify_performance_targets(result, config.performance_targets)

    return results

def test_correctness(module: nn.Module, config: TestConfig):
    """Test mathematical correctness of module."""
    # Generate test input
    x = torch.randn(config.input_shape)

    # Forward pass
    y = module(x)

    # Check output shape
    if config.expected_output_shape:
        assert y.shape == config.expected_output_shape

    # Check numerics (no NaN/inf)
    assert torch.isfinite(y).all()

    # Gradient check
    check_gradients(module, x)

    return CorrectnessResult(passed=True)
```

## Benefits

1. **No new test code needed** - Reuse existing test implementations
2. **Model-specific configs** - Test exactly what the model uses
3. **Performance tracking** - Specify expected performance
4. **CI integration** - Can run in CI for all registered models
5. **Discoverability** - Easy to find what's being tested

## Usage in CI

```yaml
# .github/workflows/test_models.yml
- name: Test LLaMA-3 Modules
  run: |
    python -m tt_transformers_v2.testing.run_model_tests \
      --model models/llama3/model.py::LLaMA3Model \
      --device wormhole_b0
```
