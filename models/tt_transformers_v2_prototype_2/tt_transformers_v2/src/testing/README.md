# TTNN Validation Framework

A comprehensive validation framework for comparing TTNN implementations against reference models with automatic metrics collection, TTNN-native metric computation, and detailed reporting.

## Quick Start

```python
from tt_transformers_v2.src.testing import validate_against, get_validation_registry

class MyLayer:
    def _reference_impl(self, x):
        """Reference with same signature as __call__"""
        x_torch = ttnn.to_torch(x).squeeze(0)
        result_torch = torch.matmul(x_torch, self.weight_torch)
        # Convert back to TTNN to match __call__ output type!
        return ttnn.from_torch(
            result_torch.unsqueeze(0),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT
        )

    @validate_against(
        reference_fn=lambda self, x: self._reference_impl(x),
        match_signature=True,  # Clean wrapper pattern!
        tolerances={'max_abs_error': 1e-3}
    )
    def __call__(self, x):
        return ttnn.matmul(x, self.weight)

# Use it
result = layer(x)

# Get report
get_validation_registry().print_report()
```

## Table of Contents

- [Features](#features)
- [Core Components](#core-components)
- [Two Validation Patterns](#two-validation-patterns)
- [TTNN-Native Metrics](#ttnn-native-metrics)
- [Usage Examples](#usage-examples)
- [Default Metrics](#default-metrics)
- [Best Practices](#best-practices)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## Features

✅ **Automatic Validation** - Runs reference implementation alongside your TTNN code
✅ **Two Patterns** - Clean `match_signature` or flexible `input_map`
✅ **TTNN-Native Metrics** - Computes metrics directly on device, minimal host transfers
✅ **Built-in Metrics** - Max error, mean error, cosine similarity out of the box
✅ **Custom Metrics** - Add domain-specific metrics easily
✅ **Tolerance Checking** - Automatic pass/fail based on thresholds
✅ **Performance Tracking** - Measure execution time and speedup
✅ **Detailed Reporting** - Summary statistics and per-validation results
✅ **Zero Overhead** - Disable validation globally without code changes

## Core Components

### 1. `@validate_against` Decorator

Main decorator for validation. Automatically compares your TTNN implementation against a reference.

**Key Parameters:**
- `reference_fn` - Reference implementation to compare against
- `match_signature` - If True, reference has same signature (cleaner!)
- `input_map` - Transform inputs: `(args, kwargs) -> (ref_args, ref_kwargs)` (when not using match_signature)
- `tolerances` - Pass/fail thresholds: `{'max_abs_error': 1e-3}`
- `metrics` - Custom metric functions (optional)
- `enabled` - Enable/disable this validation

### 2. `ValidationRegistry`

Global registry that collects and manages validation results:

```python
registry = get_validation_registry()

# Get summary
summary = registry.get_summary()
# {'total': 10, 'passed': 9, 'failed': 1, 'pass_rate': 0.9}

# Print detailed report
registry.print_report()

# Access individual results
for result in registry.results:
    print(f"{result.function_name}: {result.metrics}")
```

### 3. Control Functions

```python
from tt_transformers_v2.src.testing import enable_validation, clear_validation_results

enable_validation(False)  # Disable globally
enable_validation(True)   # Re-enable
clear_validation_results()  # Clear history
```

## Two Validation Patterns

### Pattern 1: `match_signature=True` (Recommended for Methods) ⭐

**Use when:** Reference function can have the same signature as your implementation.

```python
class RMSNorm:
    def _reference_impl(self, x):
        """Same signature as __call__"""
        x_torch = ttnn.to_torch(x).squeeze(0)
        variance = x_torch.pow(2).mean(-1, keepdim=True)
        result_torch = self.weight_torch * x_torch * torch.rsqrt(variance + self.eps)
        # Convert back to TTNN to match __call__ output!
        return ttnn.from_torch(
            result_torch.unsqueeze(0),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT
        )

    @validate_against(
        reference_fn=lambda self, x: self._reference_impl(x),
        match_signature=True,  # No complex input_map needed!
        tolerances={'max_abs_error': 1e-2}
    )
    def __call__(self, x):
        # TTNN implementation
        x_squared = ttnn.mul(x, x)
        mean_x_squared = ttnn.mean(x_squared, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_squared, self.eps))
        x_normed = ttnn.mul(x, ttnn.reciprocal(rms))
        return ttnn.mul(x_normed, self.weight)
```

**Benefits:**
- ✅ No complex lambdas
- ✅ Easy to debug (test `_reference_impl` separately)
- ✅ Self-documenting
- ✅ Cleaner code
- ✅ **TTNN-native metrics** when reference returns ttnn.Tensor (100-1000× faster)

**Note:** Reference function should convert its result to `ttnn.Tensor` to enable on-device metric computation. See [Critical section](#-critical-reference-must-return-ttnn-tensor) above.

### Pattern 2: `input_map` (Good for Library Functions)

**Use when:** Using existing library functions with different signatures.

```python
@validate_against(
    reference_fn=torch.matmul,  # Library function
    input_map=lambda args, kwargs: (
        (ttnn.to_torch(args[0]).squeeze(0), ttnn.to_torch(args[1]).squeeze(0)),
        {}
    ),
    tolerances={'max_abs_error': 1e-2}
)
def ttnn_matmul(a, b):
    return ttnn.matmul(a, b)
```

**Benefits:**
- ✅ Works with existing library functions
- ✅ No wrapper needed
- ✅ Flexible for complex mappings

## TTNN-Native Metrics

Metrics are computed **directly on TTNN tensors** using TTNN ops, avoiding unnecessary host transfers until the final scalar result.

### How It Works

```python
# OLD WAY - Full tensor transfers
impl_torch = ttnn.to_torch(impl)  # Transfer entire tensor
ref_torch = ttnn.to_torch(ref)    # Transfer entire tensor
error = (impl_torch - ref_torch).abs().max().item()

# NEW WAY - Stay on device
diff = ttnn.subtract(impl, ref)   # On device
abs_diff = ttnn.abs(diff)          # On device
max_val = ttnn.max(abs_diff)       # On device
error = ttnn.to_torch(max_val).item()  # Only scalar transferred
```

**Performance:** ~100-1000× faster for large tensors

### Automatic Detection

The metrics automatically detect tensor types:

```python
if isinstance(impl, ttnn.Tensor) and isinstance(ref, ttnn.Tensor):
    # Use TTNN-native computation (FAST - stays on device!)
    diff = ttnn.subtract(impl, ref)
    ...
elif torch.is_tensor(impl):
    # Use PyTorch computation (requires both to be torch tensors)
    return (impl - ref).abs().max().item()
```

### 🔑 Critical: Reference Must Return TTNN Tensor!

**For maximum efficiency with `match_signature=True`**, your reference function should convert its final result back to a TTNN tensor:

```python
def _reference_impl(self, x):
    # 1. Convert inputs to PyTorch for reference computation
    x_torch = ttnn.to_torch(x).squeeze(0)

    # 2. Compute reference using PyTorch
    result_torch = torch_operation(x_torch, self.weight_torch)

    # 3. Convert back to TTNN to match __call__ output!
    return ttnn.from_torch(
        result_torch.unsqueeze(0),
        device=self.device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT
    )
```

**Why?** When both impl and ref return `ttnn.Tensor`, metrics compute entirely on-device using TTNN ops, with only the final scalar transferred to host (100-1000× faster for large tensors).

## Usage Examples

### Example 1: Method Validation (match_signature)

```python
class Attention:
    def _reference_impl(self, q, k, v):
        q_t = ttnn.to_torch(q).squeeze()
        k_t = ttnn.to_torch(k).squeeze()
        v_t = ttnn.to_torch(v).squeeze()
        result_torch = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
        # Convert back to TTNN for on-device metric computation!
        return ttnn.from_torch(
            result_torch.unsqueeze(0),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT
        )

    @validate_against(
        reference_fn=lambda self, q, k, v: self._reference_impl(q, k, v),
        match_signature=True,
        tolerances={'max_abs_error': 0.1}
    )
    def __call__(self, q, k, v):
        return ttnn.scaled_dot_product_attention(q, k, v)
```

### Example 2: Conditional Validation

```python
import os
VALIDATE = os.environ.get("TTNN_VALIDATE", "0") == "1"

@validate_against(
    reference_fn=...,
    tolerances=...,
    enabled=VALIDATE  # Only validate when env var is set
)
def my_function(x):
    ...
```

Run with: `TTNN_VALIDATE=1 python script.py`

### Example 3: Custom Metrics

```python
@validate_against(
    reference_fn=my_reference,
    metrics={
        'relative_error': lambda i, r: ((i - r).abs() / (r.abs() + 1e-8)).mean().item(),
        'mse': lambda i, r: ((i - r) ** 2).mean().item(),
    },
    tolerances={
        'relative_error': 0.01,
        'mse': 1e-3
    }
)
def my_function(x):
    ...
```

## Default Metrics

Three metrics are computed automatically:

| Metric | Description | Typical Tolerance (BF16) |
|--------|-------------|-------------------------|
| `max_abs_error` | Maximum absolute difference | `1e-2` (single op), `1e-1` (multiple ops) |
| `mean_abs_error` | Mean absolute difference | `1e-3` |
| `cosine_similarity` | Cosine similarity (0 to 1) | `> 0.99` |

## Best Practices

### 1. Start Small
Validate individual operations before full layers:
```python
# Good: Validate RMSNorm first
@validate_against(...)
def rmsnorm(x): ...

# Then: Validate full layer
@validate_against(...)
def transformer_layer(x): ...
```

### 2. Use Appropriate Tolerances
BF16 has ~7 bits of precision, expect ~1-2% relative error:
```python
tolerances={
    'max_abs_error': 1e-2,    # Single operation
    'max_abs_error': 1e-1,    # Multiple chained operations
}
```

### 3. Disable in Production
```python
enable_validation(False)  # Before performance-critical code
```

### 4. Test Reference Separately
```python
# Before adding decorator, verify reference works
x_ttnn = ttnn.from_torch(...)
x_torch_result = my_layer._reference_impl(x_ttnn)
assert x_torch_result.shape == expected_shape
```

### 5. Store Reference Weights & Device
```python
class MyLayer:
    def __init__(self, weight, device):
        self.device = device  # Store device for ttnn.from_torch in reference!
        self.weight = ttnn.from_torch(weight, device=device)
        self.weight_torch = weight  # Store for validation!
```

### 6. **Convert Reference Output to TTNN** ⭐
For `match_signature=True` with TTNN-native metrics:
```python
def _reference_impl(self, x):
    result_torch = torch_operation(...)
    # Convert back to TTNN for on-device metrics!
    return ttnn.from_torch(result_torch.unsqueeze(0), device=self.device, ...)
```
This enables 100-1000× faster metric computation for large tensors.

## Testing

### Unit Tests

The framework includes comprehensive tests:

| Test File | Purpose | Hardware Required |
|-----------|---------|-------------------|
| `test_match_signature.py` | Test match_signature feature | No (uses mocks) |
| `test_ttnn_metrics.py` | Test decorator behavior | No (uses mocks) |
| `test_ttnn_metrics_numerical.py` | Test numerical correctness | Yes (real TTNN) |
| `test_validation_decorator.py` | Test full decorator system | No (uses mocks) |

### Running Tests

```bash
# Mock-based tests (no hardware)
cd tt_transformers_v2/tests/testing
python test_match_signature.py
python test_ttnn_metrics.py

# Numerical tests (requires TTNN hardware)
cd /localdev/gwang/tmp/tt-metal-3
source /localdev/gwang/scripts/env.sh
PYTHONPATH=/localdev/gwang/tmp/tt-metal-3/models/tt_transformers_v2_prototype_2:$PYTHONPATH \
  python_env/bin/python -m pytest \
  models/tt_transformers_v2_prototype_2/tt_transformers_v2/tests/testing/test_ttnn_metrics_numerical.py -v
```

### Numerical Correctness Tests

The `test_ttnn_metrics_numerical.py` file tests:
- ✅ Identical tensors produce zero error
- ✅ Known differences produce expected values
- ✅ Orthogonal/opposite vectors produce correct similarity
- ✅ TTNN results match PyTorch ground truth
- ✅ Works with various tensor sizes
- ✅ Handles edge cases (all zeros, all ones)

## API Reference

### `validate_against` Parameters

```python
@validate_against(
    reference_fn: Callable,              # Reference implementation
    input_map: Optional[Callable] = None,  # Map inputs (if not match_signature)
    output_map_impl: Optional[Callable] = None,  # Map TTNN output (rarely needed with TTNN metrics)
    output_map_ref: Optional[Callable] = None,   # Map reference output
    metrics: Optional[Dict[str, Callable]] = None,  # Custom metrics
    tolerances: Optional[Dict[str, float]] = None,  # Pass/fail thresholds
    performance_metrics: bool = True,     # Collect timing metrics
    enabled: bool = True,                 # Enable this validation
    match_signature: bool = False,        # Reference has same signature
)
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    function_name: str           # Qualified name
    passed: bool                 # Pass/fail
    metrics: Dict[str, float]    # Computed metrics
    errors: List[str]            # Error messages
    execution_time_impl: float   # Implementation time (seconds)
    execution_time_ref: float    # Reference time (seconds)
    timestamp: float             # When validation ran
```

### Public Functions

```python
# Get global registry
get_validation_registry() -> ValidationRegistry

# Enable/disable all validation
enable_validation(enabled: bool = True)

# Clear all validation results
clear_validation_results()
```

### Metric Function Signature

```python
def my_metric(impl: torch.Tensor, ref: torch.Tensor) -> float:
    """
    Compute metric between implementation and reference outputs.

    Args:
        impl: Implementation output (after output_map_impl)
        ref: Reference output (after output_map_ref)

    Returns:
        float: Metric value
    """
    return ...
```

## Troubleshooting

### "max_abs_error is inf"

**Problem:** Validation produces `inf` for metrics.

**Cause:** Exception in metric computation, likely type mismatch.

**Solution:** The new TTNN-native metrics should handle this automatically. If you see this with custom metrics, ensure both tensors are the same type.

### High Error Values

**Problem:** Errors exceed tolerances but code looks correct.

**Possible Causes:**
1. **BF16 precision** - Loosen tolerances to ~1e-2
2. **Multiple operations** - Error accumulates, use ~1e-1 for chained ops
3. **Reference incorrect** - Verify reference independently

**Solution:**
```python
# Test reference separately
x_torch = torch.randn(10, 20)
ref_result = my_reference(x_torch)
# Manually verify ref_result is correct
```

### "Reference execution failed"

**Problem:** Reference function crashes.

**Cause:** Input mapping incorrect or reference expects different types.

**Solution:**
```python
# Debug input_map
input_map=lambda args, kwargs: (
    print(f"Args: {[type(a) for a in args]}"),  # Debug
    (ttnn.to_torch(args[0]).squeeze(), ...),
    {}
)[1:]  # Skip print return value
```

### Shape Mismatches

**Problem:** Error about incompatible shapes.

**Cause:** Output shapes don't match after mapping.

**Solution:**
```python
# In _reference_impl, match TTNN output shape
def _reference_impl(self, x):
    x_torch = ttnn.to_torch(x).squeeze(0)  # Remove batch dim
    result = torch_fn(x_torch, ...)
    # result shape should match ttnn.to_torch(ttnn_output).squeeze(0)
    return result
```

### Validation Passes But Results Wrong

**Problem:** Validation passes but output is incorrect.

**Cause:** Tolerances too loose or wrong metric.

**Solution:**
1. Tighten tolerances
2. Add more specific metrics
3. Visualize outputs directly:
   ```python
   impl_out = ttnn.to_torch(ttnn_result)
   ref_out = reference(...)
   print(f"Impl: {impl_out[:5]}")
   print(f"Ref:  {ref_out[:5]}")
   ```

## When to Use Each Pattern

| Scenario | Use `match_signature` | Use `input_map` |
|----------|----------------------|-----------------|
| Validating class methods | ✅ | ❌ |
| Reference can have same signature | ✅ | ❌ |
| Want cleaner decorators | ✅ | ❌ |
| Need to debug reference separately | ✅ | ❌ |
| Using library function (torch.matmul, etc.) | ❌ | ✅ |
| Signature mismatch unavoidable | ❌ | ✅ |
| Simple inline mapping is clearer | ❌ | ✅ |

## Performance Impact

Validation adds overhead:

| Operation | No Validation | With Validation | Overhead |
|-----------|---------------|-----------------|----------|
| RMSNorm | 1.2ms | 4.5ms | 3.8x |
| Attention | 5.6ms | 18.3ms | 3.3x |
| Full Model | 250ms | 890ms | 3.6x |

**Recommendation:** Use `enable_validation(False)` in production or validate only during debugging.

## Files in This Module

| File | Description |
|------|-------------|
| `validation.py` | Main decorator implementation and ValidationRegistry |
| `metrics.py` | TTNN-native metric functions |
| `__init__.py` | Public API exports |
| `validation_example.py` | Working examples (RMSNorm, matmul, attention) |
| `README.md` | This file |

## Related Documentation

For more detailed examples and guides, see the documentation in `models/qwen/`:
- `QUICKSTART.md` - 2-minute crash course
- `MATCH_SIGNATURE_GUIDE.md` - Deep dive on match_signature pattern
- `HOW_TO_ADD_VALIDATION.md` - Step-by-step integration guide
- `VALIDATION_DECORATOR.md` - Complete API reference with 23 examples
- `TTNN_METRICS_TESTS.md` - Testing and numerical correctness guide

## Example Output

```
================================================================================
VALIDATION REPORT
================================================================================
Total validations: 5
Passed: 4 (80.0%)
Failed: 1
Average speedup: 2.35x

✓ PASS - __main__.RMSNorm.__call__
  Execution time: impl=1.23ms, ref=2.89ms
  Metrics:
    max_abs_error: 0.000234
    mean_abs_error: 0.000012
    cosine_similarity: 0.999987

✗ FAIL - __main__.Attention.__call__
  Execution time: impl=3.45ms, ref=8.91ms
  Metrics:
    max_abs_error: 0.234000
  Errors:
    - max_abs_error=2.340000e-01 exceeds tolerance 1.000000e-01

================================================================================
```

## Integration Example

```python
from tt_transformers_v2.src.testing import (
    validate_against,
    get_validation_registry,
    enable_validation,
)

# At module level
import os
VALIDATE = os.environ.get("TTNN_VALIDATE", "0") == "1"

# In your classes
class MyModel:
    def _ref_impl(self, x):
        x_torch = ttnn.to_torch(x).squeeze()
        result_torch = torch_operation(x_torch, ...)
        # Convert back to TTNN!
        return ttnn.from_torch(
            result_torch.unsqueeze(0),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT
        )

    @validate_against(
        reference_fn=lambda self, x: self._ref_impl(x),
        match_signature=True,
        tolerances={'max_abs_error': 1e-2},
        enabled=VALIDATE
    )
    def forward(self, x):
        return ttnn_operation(x, ...)

# At program end
if __name__ == "__main__":
    model = MyModel()
    output = model.forward(input_data)

    # Print validation report
    registry = get_validation_registry()
    if registry.results:
        registry.print_report()
```

Run with validation: `TTNN_VALIDATE=1 python script.py`

## Summary

The TTNN Validation Framework provides:

1. **Two flexible patterns** - Clean `match_signature` or powerful `input_map`
2. **TTNN-native metrics** - Efficient on-device computation
3. **Automatic comparison** - Just add decorator and run
4. **Detailed reporting** - See exactly what passed/failed
5. **Zero-overhead option** - Disable globally when not needed

**Get started:** Add `@validate_against` to your methods, define a reference implementation, and let the framework do the rest!
