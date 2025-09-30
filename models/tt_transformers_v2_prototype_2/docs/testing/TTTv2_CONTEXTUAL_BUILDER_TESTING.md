# Context Manager + Builder Pattern Testing API

## Core Idea
Create a testing context from an instantiated model that can automatically infer shapes and provide a fluent interface for specifying test parameters.

## Design

### Basic Usage

```python
from tt_transformers_v2.testing import TestSuite

# Create model instance
model = LLaMA3Model(config)

# Build test suite using context manager + builder pattern
with TestSuite(model) as suite:
    # Pass actual module instances - no strings!
    suite.test(model.attention) \
        .tolerance(1e-3) \
        .expect(latency_ms=5.0, memory_mb=512)

    suite.test(model.ffn) \
        .tolerance(1e-3) \
        .expect(latency_ms=3.0, memory_mb=256)

    suite.test(model.norm) \
        .tolerance(1e-5)  # No performance expectations for this one

# Tests are automatically run when context exits
# Or run explicitly
results = suite.run()
```

### Advanced Usage with Custom Inputs

```python
with TestSuite(model, batch_size=2, seq_len=4096) as suite:
    # Can override auto-detected shapes
    suite.test(model.attention) \
        .input_shape((2, 4096, 4096)) \
        .tolerance(1e-3) \
        .expect(latency_ms=10.0)

    # Test with different sequence lengths
    suite.test(model.ffn) \
        .with_seq_lengths([512, 1024, 2048, 4096]) \
        .tolerance(1e-3) \
        .expect_scaling("linear")  # Expects linear scaling with seq_len

    # Test optional module if it exists
    if hasattr(model, 'sparse_attention'):
        suite.test(model.sparse_attention) \
            .tolerance(1e-4)

    # Can also test submodules directly
    suite.test(model.attention.key_proj, name="attention.key_proj") \
        .tolerance(1e-3)
```

### Implementation

```python
# tt_transformers_v2/src/testing/test_suite.py
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from dataclasses import dataclass, field

@dataclass
class TestResult:
    module_name: str
    passed: bool
    metrics: Dict[str, float]
    errors: List[str] = field(default_factory=list)

class ModuleTestBuilder:
    """Fluent interface for building test specifications."""

    def __init__(self, suite: 'TestSuite', module: nn.Module, name: Optional[str] = None):
        self.suite = suite
        self.module = module
        self.module_name = name or self._get_module_name(module)
        self._tolerance = 1e-3
        self._expected = {}
        self._input_shape = None
        self._seq_lengths = None
        self._skip = False

        # Auto-detect shape from model
        self._auto_detect_shape()

    def _get_module_name(self, module: nn.Module) -> str:
        """Get a readable name for the module."""
        # Try to find the attribute name in the model
        for name, mod in self.suite.model.named_modules():
            if mod is module:
                return name
        # Fallback to class name
        return module.__class__.__name__

    def _auto_detect_shape(self):
        """Automatically infer input shape from module configuration."""
        # Try to infer from module attributes
        if hasattr(self.module, 'hidden_dim'):
            hidden_dim = self.module.hidden_dim
            self._input_shape = (
                self.suite.batch_size,
                self.suite.seq_len,
                hidden_dim
            )
        elif hasattr(self.module, 'in_features'):
            # For linear layers
            in_features = self.module.in_features
            self._input_shape = (
                self.suite.batch_size,
                self.suite.seq_len,
                in_features
            )
        # Add more detection logic as needed

    def tolerance(self, tol: float) -> 'ModuleTestBuilder':
        """Set numerical tolerance for this test."""
        self._tolerance = tol
        return self

    def expect(self, **kwargs) -> 'ModuleTestBuilder':
        """Set performance expectations."""
        self._expected.update(kwargs)
        return self

    def input_shape(self, shape: tuple) -> 'ModuleTestBuilder':
        """Override auto-detected input shape."""
        self._input_shape = shape
        return self

    def with_seq_lengths(self, lengths: List[int]) -> 'ModuleTestBuilder':
        """Test with multiple sequence lengths."""
        self._seq_lengths = lengths
        return self

    def if_exists(self) -> 'ModuleTestBuilder':
        """Only test if module exists (for optional components)."""
        if not hasattr(self.suite.model, self.module_name):
            self._skip = True
        return self

    def expect_scaling(self, scaling_type: str) -> 'ModuleTestBuilder':
        """Expect specific scaling behavior (linear, quadratic, etc)."""
        self._expected['scaling'] = scaling_type
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Register this test with the suite
        if not self._skip:
            self.suite._register_test(self)


class TestSuite:
    """Context manager for building and running module tests."""

    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 1,
        seq_len: int = 2048,
        device: Optional[torch.device] = None,
        auto_run: bool = True
    ):
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device or next(model.parameters()).device
        self.auto_run = auto_run
        self._tests: List[ModuleTestBuilder] = []
        self._results: List[TestResult] = []

    def test(self, module: nn.Module, name: Optional[str] = None) -> ModuleTestBuilder:
        """Start building a test for the given module instance."""
        return ModuleTestBuilder(self, module, name)

    def _register_test(self, test_builder: ModuleTestBuilder):
        """Register a test specification."""
        self._tests.append(test_builder)

    def run(self) -> List[TestResult]:
        """Run all registered tests."""
        self._results.clear()

        for test in self._tests:
            print(f"Testing {test.module_name}...")
            result = self._run_single_test(test)
            self._results.append(result)

            if result.passed:
                print(f"  ✓ Passed")
            else:
                print(f"  ✗ Failed: {', '.join(result.errors)}")

        return self._results

    def _run_single_test(self, test: ModuleTestBuilder) -> TestResult:
        """Run a single module test."""
        module = test.module
        errors = []
        metrics = {}

        try:
            # Generate test input
            if test._input_shape:
                x = torch.randn(test._input_shape, device=self.device)
            else:
                errors.append(f"Could not determine input shape for {test.module_name}")
                return TestResult(test.module_name, False, metrics, errors)

            # Test different sequence lengths if specified
            if test._seq_lengths:
                for seq_len in test._seq_lengths:
                    x_seq = torch.randn(
                        test._input_shape[0],
                        seq_len,
                        test._input_shape[2],
                        device=self.device
                    )
                    self._test_forward(module, x_seq, test, errors, metrics)
            else:
                self._test_forward(module, x, test, errors, metrics)

            # Check performance expectations
            for metric, expected in test._expected.items():
                if metric in metrics:
                    if metrics[metric] > expected:
                        errors.append(
                            f"{metric}: {metrics[metric]:.2f} > expected {expected}"
                        )

        except Exception as e:
            errors.append(f"Exception: {str(e)}")

        return TestResult(
            module_name=test.module_name,
            passed=len(errors) == 0,
            metrics=metrics,
            errors=errors
        )

    def _test_forward(self, module, x, test, errors, metrics):
        """Test forward pass of a module."""
        # Correctness test
        y = module(x)

        if not torch.isfinite(y).all():
            errors.append("Output contains NaN or Inf")

        # Performance test (simplified)
        import time
        start_time = time.time()
        for _ in range(10):
            _ = module(x)
        torch.cuda.synchronize() if x.is_cuda else None
        elapsed = (time.time() - start_time) / 10 * 1000  # ms

        metrics['latency_ms'] = elapsed

        # Memory usage (simplified)
        if x.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = module(x)
            metrics['memory_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.auto_run and exc_type is None:
            self.run()


# Additional helper for even simpler usage
def test_model(model: nn.Module, **modules_to_test):
    """Quick testing with minimal setup."""
    with TestSuite(model) as suite:
        for module_name, expectations in modules_to_test.items():
            test = suite.test(module_name).tolerance(1e-3)
            if expectations:
                test.expect(**expectations)

    return suite._results

# Usage:
results = test_model(
    model,
    attention={"latency_ms": 5.0, "memory_mb": 512},
    ffn={"latency_ms": 3.0},
    norm={},  # Test with defaults
)
```

## Benefits of Module Instance Approach

1. **Type Safety**: IDE autocomplete and type checking
   ```python
   suite.test(model.attention)  # ✓ IDE knows this is a Module
   suite.test("atention")       # ✗ Typo not caught until runtime
   ```

2. **Direct References**: No string-based lookups
   ```python
   # Test any module, even nested ones
   suite.test(model.layers[0].attention)
   suite.test(model.encoder.layers[5].mlp)
   ```

3. **Refactoring-Friendly**: Rename attributes without breaking tests
   ```python
   # If you rename self.attention → self.mha
   # This still works: suite.test(model.mha)
   ```

4. **Zero Shape Configuration**: Auto-detects from actual module
5. **Fluent Interface**: Natural method chaining
6. **Model-Aware**: Full introspection capabilities

## Integration with CI/CD

```yaml
# .github/workflows/test_models.yml
- name: Test Model Configurations
  run: |
    python -c "
    from models.llama3 import LLaMA3Model
    from tt_transformers_v2.testing import TestSuite

    model = LLaMA3Model.from_pretrained('llama3-7b')
    with TestSuite(model) as suite:
        suite.test('attention').tolerance(1e-3).expect(latency_ms=5.0)
        suite.test('ffn').tolerance(1e-3).expect(latency_ms=3.0)
    "
```

## Future Extensions

1. **Comparison Testing**
   ```python
   suite.test("attention").compare_with("reference_attention")
   ```

2. **Profiling Integration**
   ```python
   suite.test("ffn").profile(warmup=10, iterations=100)
   ```

3. **Batch Testing**
   ```python
   suite.test_all_modules().matching("*attention*").tolerance(1e-3)
   ```
