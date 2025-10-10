# Validation Decorator - Quick Start

- QUICKSTART.md - 2-minute crash course, copy-paste examples
- README_VALIDATION.md - Overview, architecture, integration guide
- VALIDATION_DECORATOR.md - Complete API reference (23 examples)
- HOW_TO_ADD_VALIDATION.md - Step-by-step for existing code
- validation_example.py - Working examples (RMSNorm, matmul, attention)
- test_validation_decorator.py - 11 unit tests
- ds_r1_qwen.py - Model implementation with decorator system

## 30-Second Example

```python
from ds_r1_qwen import validate_against, get_validation_registry

@validate_against(
    reference_fn=torch.nn.functional.layer_norm,
    input_map=lambda args, kwargs: ((ttnn.to_torch(args[0]).squeeze(),), kwargs),
    output_map_impl=lambda x: ttnn.to_torch(x).squeeze(),
    tolerances={'max_abs_error': 1e-3}
)
def my_function(x):
    return ttnn.layer_norm(x)

# Use it
result = my_function(x)

# Get report
get_validation_registry().print_report()
```

## Core Concept

```
Your TTNN Code → @validate_against → Auto-compares with reference → Collects metrics → Reports results
```

## The 4 Key Parameters

1. **`reference_fn`** - What to compare against (e.g., `torch.matmul`)
2. **`input_map`** - How to convert your inputs: `(args, kwargs) -> (ref_args, ref_kwargs)`
3. **`output_map_impl`** - How to convert your output: `(ttnn_output) -> comparable_output`
4. **`tolerances`** - Pass/fail thresholds: `{'max_abs_error': 1e-3}`

## Common Pattern: Method Validation

```python
class MyLayer:
    def __init__(self, weight, device):
        self.weight = ttnn.from_torch(weight, device=device)
        self.weight_torch = weight  # ← Store this for validation!

    @validate_against(
        reference_fn=lambda x, w: torch.matmul(x, w),
        input_map=lambda args, kwargs: (
            (ttnn.to_torch(args[1]).squeeze(0), args[0].weight_torch),  # args[0] = self
            {}
        ),
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
        tolerances={'max_abs_error': 1e-2}
    )
    def __call__(self, x):
        return ttnn.matmul(x, self.weight)
```

## input_map Template

```python
input_map=lambda args, kwargs: (
    (
        ttnn.to_torch(args[0]).squeeze(0),  # First arg converted
        ttnn.to_torch(args[1]).squeeze(0),  # Second arg converted
        args[2],                            # Third arg passed as-is
    ),
    {}  # Or kwargs dict
)
```

For methods, remember `args[0]` is `self`!

```python
input_map=lambda args, kwargs: (
    (
        ttnn.to_torch(args[1]).squeeze(0),  # First actual arg (args[0] is self)
        args[0].some_attribute,              # Access self attributes
    ),
    {}
)
```

## Default Metrics (Always Computed)

- `max_abs_error` - Max absolute difference
- `mean_abs_error` - Mean absolute difference
- `cosine_similarity` - Cosine similarity (0 to 1)

## Add Custom Metrics

```python
@validate_against(
    reference_fn=...,
    metrics={
        'relative_error': lambda impl, ref: ((impl - ref).abs() / (ref.abs() + 1e-8)).mean().item(),
    },
    tolerances={'relative_error': 0.01},
    ...
)
```

## Control Validation

```python
from ds_r1_qwen import enable_validation, clear_validation_results

enable_validation(False)  # Disable
enable_validation(True)   # Enable
clear_validation_results()  # Clear history
```

## Get Results

```python
registry = get_validation_registry()

# Summary
summary = registry.get_summary()
print(f"Pass rate: {summary['pass_rate']*100:.1f}%")

# Full report
registry.print_report()

# Individual results
for result in registry.results:
    print(f"{result.function_name}: {result.passed}")
    print(f"  Metrics: {result.metrics}")
```

## Use Environment Variable

```python
import os
VALIDATE = os.environ.get("VALIDATE", "0") == "1"

@validate_against(..., enabled=VALIDATE)
def my_function(x):
    ...
```

```bash
VALIDATE=1 python script.py  # Runs with validation
python script.py              # Runs without validation
```

## Typical Tolerances (BF16)

```python
tolerances={
    'max_abs_error': 1e-2,    # Single op
    'max_abs_error': 1e-1,    # Multiple ops
    'mean_abs_error': 1e-3,   # More lenient
}
```

## Files

- `ds_r1_qwen.py` - Implementation (decorator lives here)
- `VALIDATION_DECORATOR.md` - Full API docs
- `HOW_TO_ADD_VALIDATION.md` - Step-by-step integration guide
- `validation_example.py` - Working examples
- `test_validation_decorator.py` - Unit tests

## Test It

```bash
# Run tests
python test_validation_decorator.py

# Run examples
python validation_example.py
```

## Troubleshooting

**Error: "Reference execution failed"**
→ Check `input_map` transforms correctly

**Error: "Output mapping failed"**
→ Check shapes match: `print(ttnn.to_torch(output).shape)`

**High errors but code looks right**
→ BF16 precision ~1e-2, loosen tolerances

**Validation passing but results wrong**
→ Tolerances too loose, visualize outputs directly

## Full Example: RMSNorm

```python
class RMSNorm:
    def __init__(self, weight: torch.Tensor, eps: float, device):
        self.eps = eps
        self.weight = ttnn.from_torch(weight.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16)
        self.weight_torch = weight  # Store for validation

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

# Use it
rms = RMSNorm(weight, eps=1e-6, device)
output = rms(x)

# Check validation
get_validation_registry().print_report()
```

## That's It!

Three steps:
1. Add `@validate_against` decorator
2. Define mappings
3. Call `get_validation_registry().print_report()`

Everything else is automatic.
