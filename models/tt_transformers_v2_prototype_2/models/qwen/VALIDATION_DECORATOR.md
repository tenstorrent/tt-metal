# Validation Decorator System

A comprehensive decorator system for validating TTNN implementations against reference models with automatic metrics collection and performance tracking.

## Overview

The `@validate_against` decorator allows you to:
- Compare TTNN implementations against PyTorch or other reference implementations
- Automatically collect accuracy metrics (max error, mean error, cosine similarity, etc.)
- Track performance metrics (execution time, speedup)
- Define custom input/output mappings between implementations
- Set tolerance thresholds for automatic pass/fail validation
- Generate detailed validation reports

## Basic Usage

```python
from ds_r1_qwen import validate_against

@validate_against(
    reference_fn=torch.nn.functional.layer_norm,
    input_map=lambda args, kwargs: ((ttnn.to_torch(args[0]),), kwargs),
    output_map_impl=lambda x: ttnn.to_torch(x),
    tolerances={'max_abs_error': 1e-3}
)
def my_ttnn_layer_norm(x, normalized_shape, eps=1e-5):
    # Your TTNN implementation
    return ttnn.layer_norm(x, normalized_shape, eps)
```

## Decorator Parameters

### `reference_fn: Callable`
The reference implementation to compare against. Typically a PyTorch function.

```python
reference_fn=torch.matmul
reference_fn=torch.nn.functional.softmax
reference_fn=my_custom_reference_function
```

### `input_map: Optional[Callable]`
Maps decorated function inputs to reference function inputs.

**Signature**: `(args, kwargs) -> (ref_args, ref_kwargs)`

```python
# Example: Convert TTNN tensors to PyTorch
input_map=lambda args, kwargs: (
    (ttnn.to_torch(args[0]).squeeze(), ttnn.to_torch(args[1]).squeeze()),
    {}
)

# Example: Extract attributes from self
input_map=lambda args, kwargs: (
    (ttnn.to_torch(args[1]).squeeze(), args[0].weight_torch),
    {'eps': args[0].eps}
)
```

If `None`, inputs are passed as-is to the reference function.

### `output_map_impl: Optional[Callable]`
Maps decorated function output to a comparable format.

**Signature**: `(output) -> comparable_output`

```python
# Convert TTNN to PyTorch tensor
output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0)

# Extract specific field from tuple
output_map_impl=lambda x: x[0]

# Convert to numpy for comparison
output_map_impl=lambda x: ttnn.to_torch(x).numpy()
```

If `None`, output is used as-is.

### `output_map_ref: Optional[Callable]`
Maps reference function output to a comparable format.

**Signature**: `(output) -> comparable_output`

```python
output_map_ref=lambda x: x.squeeze(0)
output_map_ref=lambda x: x.detach()
```

If `None`, output is used as-is.

### `metrics: Optional[Dict[str, Callable]]`
Custom metrics to compute. Adds to or overrides default metrics.

**Signature**: Each metric function: `(impl_output, ref_output) -> float`

**Default metrics**:
- `max_abs_error`: Maximum absolute difference
- `mean_abs_error`: Mean absolute difference
- `cosine_similarity`: Cosine similarity between flattened tensors

```python
metrics={
    'relative_error': lambda impl, ref: ((impl - ref).abs() / (ref.abs() + 1e-8)).mean().item(),
    'mse': lambda impl, ref: ((impl - ref) ** 2).mean().item(),
    'pearson_correlation': lambda impl, ref: torch.corrcoef(
        torch.stack([impl.flatten(), ref.flatten()])
    )[0, 1].item(),
}
```

### `tolerances: Optional[Dict[str, float]]`
Tolerance thresholds for pass/fail validation. If any metric exceeds its tolerance, validation fails.

```python
tolerances={
    'max_abs_error': 1e-3,        # Must be <= 0.001
    'mean_abs_error': 1e-4,       # Must be <= 0.0001
    'cosine_similarity': 0.999,   # Must be >= 0.999 (for similarity metrics)
}
```

Note: For similarity metrics like cosine_similarity, you might want to check if value < threshold to fail.

### `performance_metrics: bool = True`
Whether to collect execution time metrics.

### `enabled: bool = True`
Whether this specific validation is enabled.

## Complete Examples

### Example 1: RMSNorm Validation

```python
class RMSNorm:
    def __init__(self, weight: torch.Tensor, eps: float, device):
        self.eps = eps
        self.weight = ttnn.from_torch(weight.unsqueeze(0).unsqueeze(0), device=device)
        self.weight_torch = weight  # Keep for reference

    @validate_against(
        reference_fn=lambda x, w, eps: w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps),
        input_map=lambda args, kwargs: (
            (ttnn.to_torch(args[1]).squeeze(0), args[0].weight_torch, args[0].eps),
            {}
        ),
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
        tolerances={'max_abs_error': 1e-2, 'mean_abs_error': 1e-3}
    )
    def __call__(self, x):
        x_squared = ttnn.mul(x, x)
        mean_x_squared = ttnn.mean(x_squared, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_squared, self.eps))
        x_normed = ttnn.mul(x, ttnn.reciprocal(rms))
        return ttnn.mul(x_normed, self.weight)
```

### Example 2: Attention Validation

```python
def reference_attention(q, k, v, mask, scale):
    """Reference PyTorch attention"""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    attn = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

@validate_against(
    reference_fn=reference_attention,
    input_map=lambda args, kwargs: (
        (
            ttnn.to_torch(args[0]).squeeze(0),  # q
            ttnn.to_torch(args[1]).squeeze(0),  # k
            ttnn.to_torch(args[2]).squeeze(0),  # v
            args[3],  # mask (already torch)
            args[4]   # scale
        ),
        {}
    ),
    output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
    metrics={
        'max_abs_error': lambda i, r: (i - r).abs().max().item(),
        'attention_entropy': lambda i, r: -(i * torch.log(i + 1e-9)).sum(-1).mean().item(),
    },
    tolerances={'max_abs_error': 0.1}
)
def ttnn_attention(q, k, v, mask, scale):
    scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
    scores = ttnn.mul(scores, scale)
    if mask is not None:
        scores = ttnn.add(scores, mask)
    attn = ttnn.softmax(scores, dim=-1)
    return ttnn.matmul(attn, v)
```

### Example 3: Method Validation with Self Access

```python
class Attention:
    def __init__(self, wq, wk, wv, wo, device):
        self.wq = ttnn.from_torch(wq, device=device)
        self.wk = ttnn.from_torch(wk, device=device)
        self.wv = ttnn.from_torch(wv, device=device)
        self.wo = ttnn.from_torch(wo, device=device)
        # Keep reference weights
        self.wq_torch = wq
        self.wk_torch = wk
        self.wv_torch = wv
        self.wo_torch = wo

    @validate_against(
        reference_fn=lambda x, wq, wk, wv, wo: torch.matmul(
            torch.nn.functional.softmax(
                torch.matmul(torch.matmul(x, wq), torch.matmul(x, wk).transpose(-2, -1)),
                dim=-1
            ),
            torch.matmul(x, wv)
        ) @ wo,
        input_map=lambda args, kwargs: (
            (
                ttnn.to_torch(args[1]).squeeze(0),  # x
                args[0].wq_torch,
                args[0].wk_torch,
                args[0].wv_torch,
                args[0].wo_torch
            ),
            {}
        ),
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
        tolerances={'max_abs_error': 1e-1}
    )
    def __call__(self, x):
        q = ttnn.matmul(x, self.wq)
        k = ttnn.matmul(x, self.wk)
        v = ttnn.matmul(x, self.wv)
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
        attn = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(attn, v)
        return ttnn.matmul(out, self.wo)
```

## Validation Registry

### Get Validation Results

```python
from ds_r1_qwen import get_validation_registry

registry = get_validation_registry()

# Get summary statistics
summary = registry.get_summary()
print(f"Passed: {summary['passed']}/{summary['total']}")
print(f"Pass rate: {summary['pass_rate']*100:.1f}%")
print(f"Average speedup: {summary['avg_speedup']:.2f}x")

# Access individual results
for result in registry.results:
    print(f"{result.function_name}: {'PASS' if result.passed else 'FAIL'}")
    print(f"  Metrics: {result.metrics}")
    print(f"  Impl time: {result.execution_time_impl*1000:.2f}ms")
    print(f"  Ref time: {result.execution_time_ref*1000:.2f}ms")
```

### Print Validation Report

```python
# Print detailed report
registry.print_report()
```

Output:
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

✗ FAIL - __main__.ttnn_attention
  Execution time: impl=3.45ms, ref=8.91ms
  Metrics:
    max_abs_error: 0.234000
  Errors:
    - max_abs_error=2.340000e-01 exceeds tolerance 1.000000e-01

================================================================================
```

### Control Validation

```python
from ds_r1_qwen import enable_validation, clear_validation_results

# Disable all validation
enable_validation(False)

# Your code runs without validation overhead
model(input_data)

# Re-enable validation
enable_validation(True)

# Clear previous results
clear_validation_results()
```

## Advanced Usage

### Conditional Validation

```python
import os

ENABLE_VALIDATION = os.environ.get("ENABLE_VALIDATION", "0") == "1"

@validate_against(
    reference_fn=torch.matmul,
    input_map=...,
    output_map_impl=...,
    enabled=ENABLE_VALIDATION  # Only validate if env var is set
)
def my_function(x):
    return ttnn.matmul(x, x)
```

### Per-Layer Validation

```python
class TransformerLayer:
    def __init__(self, layer_id, ...):
        self.layer_id = layer_id
        # Only validate first 2 layers
        self.validate = layer_id < 2

    @validate_against(
        reference_fn=...,
        enabled=lambda: self.validate  # Note: doesn't work, use enabled parameter
        ...
    )
    def __call__(self, x):
        ...
```

### Custom Metrics

```python
def psnr(impl, ref):
    """Peak Signal-to-Noise Ratio"""
    mse = ((impl - ref) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    max_val = ref.abs().max().item()
    return 20 * math.log10(max_val / math.sqrt(mse))

def ssim_metric(impl, ref):
    """Structural Similarity Index"""
    # Simplified SSIM for 1D
    mu_impl = impl.mean()
    mu_ref = ref.mean()
    sigma_impl = impl.std()
    sigma_ref = ref.std()
    sigma_impl_ref = ((impl - mu_impl) * (ref - mu_ref)).mean()

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = ((2 * mu_impl * mu_ref + c1) * (2 * sigma_impl_ref + c2)) / \
           ((mu_impl ** 2 + mu_ref ** 2 + c1) * (sigma_impl ** 2 + sigma_ref ** 2 + c2))
    return ssim.item()

@validate_against(
    reference_fn=my_reference,
    metrics={
        'psnr': psnr,
        'ssim': ssim_metric,
        'max_abs_error': lambda i, r: (i - r).abs().max().item()
    },
    tolerances={
        'psnr': 40.0,  # Must be >= 40 dB
        'ssim': 0.95,  # Must be >= 0.95
        'max_abs_error': 1e-3
    }
)
def my_function(x):
    ...
```

## Best Practices

1. **Keep reference implementations simple**: Reference functions should be correct and readable, not optimized.

2. **Use appropriate tolerances**: BF16 typically has ~1e-2 to 1e-3 precision. Set tolerances accordingly.

3. **Map carefully**: Ensure input/output mappings preserve semantic meaning. Shape mismatches are a common issue.

4. **Validate incrementally**: Start with small components (RMSNorm, matmul) before validating full layers.

5. **Use custom metrics**: Default metrics may not capture domain-specific requirements (e.g., attention patterns).

6. **Disable in production**: Use `enable_validation(False)` or environment variables to disable validation overhead in production.

7. **Profile with validation off**: Validation adds overhead. Profile performance with validation disabled.

## Troubleshooting

### "Reference execution failed"
- Check that input_map correctly transforms inputs to reference function's expected format
- Verify reference function works standalone with the mapped inputs

### "Output mapping failed"
- Check tensor shapes match expected dimensions
- Verify data types are compatible (e.g., TTNN -> Torch conversion)
- Look for squeeze/unsqueeze mismatches

### "Metric X failed"
- Ensure metric function handles edge cases (NaN, Inf, empty tensors)
- Check that metric operates on the right shapes (flattened vs. multidimensional)

### High error values
- Verify reference implementation is correct
- Check for numerical instability (division by zero, large exponentials)
- Consider BF16 precision limitations
- Ensure operations are in the same order (associativity matters for floats)

### Validation passes but results look wrong
- Check if tolerance is too loose
- Add more specific metrics (per-channel, per-position analysis)
- Visualize outputs directly instead of relying only on aggregate metrics

## Integration with Existing Code

To add validation to existing code:

1. Import validation utilities:
```python
from ds_r1_qwen import validate_against, get_validation_registry
```

2. Add decorator to methods/functions you want to validate

3. Run your code normally

4. Print report at the end:
```python
get_validation_registry().print_report()
```

That's it! Validation happens automatically during execution.
