# match_signature Pattern Guide

The `match_signature=True` parameter allows you to write wrapper reference functions with the same signature as your implementation, eliminating the need for complex `input_map` and `output_map` lambdas.

## Problem It Solves

**Old way (complex lambdas):**

```python
class RMSNorm:
    def __init__(self, weight, device):
        self.weight = ttnn.from_torch(weight, device=device)
        self.weight_torch = weight  # Store for reference
        self.eps = 1e-6

    @validate_against(
        reference_fn=torch_rms_norm,  # Different signature!
        input_map=lambda args, kwargs: (
            # args[0] = self, args[1] = x
            (ttnn.to_torch(args[1]).squeeze(0), args[0].weight_torch),
            {"eps": args[0].eps}
        ),
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
        output_map_ref=lambda x: x,
        tolerances={'max_abs_error': 1e-3}
    )
    def __call__(self, x):
        return ttnn.layer_norm(x, ...)
```

**Problems:**
- Complex lambda with `args[0]`, `args[1]` indexing
- Easy to get mapping wrong
- Hard to debug
- Not reusable

## Solution: match_signature=True

**New way (clean wrapper):**

```python
class RMSNorm:
    def __init__(self, weight, device):
        self.weight = ttnn.from_torch(weight, device=device)
        self.weight_torch = weight
        self.eps = 1e-6

    def _reference_impl(self, x):
        """Reference with SAME signature as __call__"""
        x_torch = ttnn.to_torch(x).squeeze(0)
        return torch_rms_norm(x_torch, self.weight_torch, self.eps)

    @validate_against(
        reference_fn=lambda self, x: self._reference_impl(x),
        match_signature=True,  # Magic happens here!
        tolerances={'max_abs_error': 1e-3}
    )
    def __call__(self, x):
        return ttnn.layer_norm(x, ...)
```

**Benefits:**
✅ No complex lambdas
✅ Same signature = same args passed
✅ Easy to debug (test `_reference_impl` separately)
✅ Self-documenting
✅ Clean decorator

## How It Works

When `match_signature=True`:

1. Decorator calls `reference_fn(*args, **kwargs)` with **exact same args** as implementation
2. No input mapping needed
3. Outputs are compared after mapping (you **still need `output_map_impl`** to convert TTNN → torch!)

**⚠️ IMPORTANT**: `match_signature=True` only affects **input** handling. You almost always still need `output_map_impl` to convert TTNN outputs to torch tensors for comparison!

```python
# Inside decorator:
if match_signature:
    ref_output = reference_fn(*args, **kwargs)  # Same args!
elif input_map:
    ref_args, ref_kwargs = input_map(args, kwargs)
    ref_output = reference_fn(*ref_args, **ref_kwargs)
```

## Pattern 1: Method with Reference Method

```python
class MyLayer:
    def __init__(self, weight, device):
        self.weight = ttnn.from_torch(weight, device=device)
        self.weight_torch = weight

    def _torch_reference(self, x):
        """Pure PyTorch reference"""
        x_torch = ttnn.to_torch(x).squeeze(0)
        result = torch.matmul(x_torch, self.weight_torch)
        return result  # Returns torch.Tensor

    @validate_against(
        reference_fn=lambda self, x: self._torch_reference(x),
        match_signature=True,
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),  # CRITICAL: TTNN -> torch
        # No output_map_ref needed - reference already returns torch
        tolerances={'max_abs_error': 1e-2}
    )
    def __call__(self, x):
        """TTNN implementation"""
        return ttnn.matmul(x, self.weight)
```

## Pattern 2: Method with Inline Lambda

```python
class Attention:
    def __init__(self, wq, wk, wv, device):
        self.wq = ttnn.from_torch(wq, device=device)
        self.wq_torch = wq
        # ... etc

    @validate_against(
        reference_fn=lambda self, q, k, v: torch.nn.functional.scaled_dot_product_attention(
            ttnn.to_torch(q).squeeze(0),
            ttnn.to_torch(k).squeeze(0),
            ttnn.to_torch(v).squeeze(0)
        ),
        match_signature=True,
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
        tolerances={'max_abs_error': 0.1}
    )
    def __call__(self, q, k, v):
        """TTNN attention"""
        return ttnn.scaled_dot_product_attention(q, k, v)
```

## Pattern 3: Standalone Function

For standalone functions (not methods), you can still use `match_signature`:

```python
def reference_matmul(a, b):
    """Reference that handles TTNN inputs"""
    a_torch = ttnn.to_torch(a).squeeze(0)
    b_torch = ttnn.to_torch(b).squeeze(0)
    return torch.matmul(a_torch, b_torch)

@validate_against(
    reference_fn=reference_matmul,
    match_signature=True,
    output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
    tolerances={'max_abs_error': 1e-2}
)
def ttnn_matmul(a, b):
    return ttnn.matmul(a, b)
```

## When to Use match_signature

✅ **Use match_signature when:**
- Reference function has same signature as implementation
- You can write a wrapper that takes the same args
- Method validation where you can add a `_reference_impl` method
- You want cleaner, more readable decorators

❌ **Don't use match_signature when:**
- Reference function is from a library with different signature (use `input_map`)
- Mapping is simple and inline lambda is clearer
- Need to pass different args to reference

## Comparison Table

| Feature | match_signature=True | input_map |
|---------|---------------------|-----------|
| **Cleanliness** | ✅ Very clean | ❌ Complex lambdas |
| **Debuggability** | ✅ Easy to test wrapper | ❌ Lambda hard to debug |
| **Library functions** | ❌ Need wrapper | ✅ Use directly |
| **Reusability** | ✅ Reference method reusable | ❌ Lambda not reusable |
| **Setup** | Need reference method | No setup |

## Complete Example: Both Patterns

```python
class RMSNorm:
    """Shows both patterns for comparison"""

    def __init__(self, weight, device):
        self.weight = ttnn.from_torch(weight, device=device)
        self.weight_torch = weight
        self.eps = 1e-6

    # Pattern 1: match_signature (recommended for methods)
    def _reference_impl(self, x):
        x_torch = ttnn.to_torch(x).squeeze(0)
        variance = x_torch.pow(2).mean(-1, keepdim=True)
        return self.weight_torch * x_torch * torch.rsqrt(variance + self.eps)

    @validate_against(
        reference_fn=lambda self, x: self._reference_impl(x),
        match_signature=True,
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
        tolerances={'max_abs_error': 1e-2}
    )
    def forward(self, x):
        x_squared = ttnn.mul(x, x)
        mean_x_squared = ttnn.mean(x_squared, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_squared, self.eps))
        x_normed = ttnn.mul(x, ttnn.reciprocal(rms))
        return ttnn.mul(x_normed, self.weight)


# Pattern 2: input_map (for library functions)
@validate_against(
    reference_fn=torch.matmul,  # Library function
    input_map=lambda args, kwargs: (
        (ttnn.to_torch(args[0]).squeeze(0), ttnn.to_torch(args[1]).squeeze(0)),
        {}
    ),
    output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
    tolerances={'max_abs_error': 1e-2}
)
def ttnn_matmul(a, b):
    return ttnn.matmul(a, b)
```

## Common Mistakes ⚠️

### Mistake 1: Forgetting `output_map_impl`

```python
# ❌ WRONG - Will produce inf errors!
@validate_against(
    reference_fn=lambda self, x: self._reference_impl(x),
    match_signature=True,
    # Missing output_map_impl!
    tolerances={'max_abs_error': 1e-3}
)
def __call__(self, x):
    return ttnn.matmul(x, self.weight)  # Returns TTNN tensor
```

The reference returns torch tensor, implementation returns TTNN tensor. Without `output_map_impl`, metrics compare incompatible types → `inf` error!

```python
# ✅ CORRECT
@validate_against(
    reference_fn=lambda self, x: self._reference_impl(x),
    match_signature=True,
    output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),  # ← Add this!
    tolerances={'max_abs_error': 1e-3}
)
```

**Rule**: `match_signature=True` only affects **inputs**. You still need `output_map_impl` for **outputs**!

### Mistake 2: Shape Mismatches

```python
# ❌ WRONG - Shape mismatch
def _reference_impl(self, x):
    x_torch = ttnn.to_torch(x)  # [1, 1, seq, hidden]
    return torch_fn(x_torch, ...)  # Returns [1, 1, seq, hidden]

@validate_against(
    output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),  # [1, seq, hidden]
    # Shapes don't match: [1, seq, hidden] vs [1, 1, seq, hidden]
)
```

```python
# ✅ CORRECT - Consistent shapes
def _reference_impl(self, x):
    x_torch = ttnn.to_torch(x).squeeze(0)  # [seq, hidden]
    return torch_fn(x_torch, ...)  # [seq, hidden]

@validate_against(
    output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),  # [seq, hidden]
    # Shapes match!
)
```

## Tips

1. **Name reference methods clearly**: Use `_reference_impl`, `_torch_reference`, or `_golden` prefix

2. **Test reference separately**:
```python
# Before adding decorator, test reference works
x_ttnn = ttnn.from_torch(...)
x_torch_result = my_layer._reference_impl(x_ttnn)
print(x_torch_result)  # Should work
```

3. **ALWAYS use output_map_impl**: Even with `match_signature`, you almost always need `output_map_impl` to convert TTNN → torch for comparison

4. **Lambda vs method**: For simple cases, inline lambda is fine:
```python
reference_fn=lambda self, x: torch_rms_norm(ttnn.to_torch(x).squeeze(0), self.weight_torch)
```

For complex logic, use a method:
```python
def _reference_impl(self, x):
    # Complex conversion logic
    ...
    return result
```

## Migration Guide

**From old input_map pattern:**

```python
# Before
@validate_against(
    reference_fn=torch_fn,
    input_map=lambda args, kwargs: ((ttnn.to_torch(args[1]).squeeze(0), args[0].weight), {}),
    output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
)
def __call__(self, x):
    ...
```

**To match_signature pattern:**

```python
# After
def _reference_impl(self, x):
    x_torch = ttnn.to_torch(x).squeeze(0)
    return torch_fn(x_torch, self.weight)

@validate_against(
    reference_fn=lambda self, x: self._reference_impl(x),
    match_signature=True,
    output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
)
def __call__(self, x):
    ...
```

**Changes:**
1. Add `_reference_impl` method with same signature as `__call__`
2. Move conversion logic into `_reference_impl`
3. Set `match_signature=True`
4. Remove `input_map`
5. Keep `output_map_impl` if needed (usually yes for TTNN → torch conversion)

## Troubleshooting

### "max_abs_error is inf"

**Problem**: Validation produces `inf` or `nan` for metrics.

**Cause**: Missing `output_map_impl` - trying to compare TTNN tensor with torch tensor directly.

**Solution**: Add `output_map_impl`:
```python
@validate_against(
    reference_fn=lambda self, x: self._reference_impl(x),
    match_signature=True,
    output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),  # ← ADD THIS!
    tolerances={'max_abs_error': 1e-3}
)
```

### "Shape mismatch in metrics"

**Problem**: Error about incompatible shapes.

**Cause**: Reference output shape doesn't match implementation output shape after mapping.

**Solution**: Make sure both outputs have same shape after mapping:
```python
# In _reference_impl:
x_torch = ttnn.to_torch(x).squeeze(0)  # Match the squeeze in output_map_impl!
result = torch_fn(x_torch, ...)
return result  # Should be [seq, hidden]

# In decorator:
output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0)  # [seq, hidden]
```

### "Reference execution failed"

**Problem**: Reference function crashes.

**Cause**: Reference expects torch tensor but receives TTNN tensor.

**Solution**: Convert in `_reference_impl`:
```python
def _reference_impl(self, x):
    x_torch = ttnn.to_torch(x).squeeze(0)  # Convert first!
    return torch_fn(x_torch, ...)
```

## Summary

`match_signature=True` is syntactic sugar that makes decorators cleaner when you can write a reference function with the same signature as your implementation. It's especially powerful for method validation where you can add a `_reference_impl` method to your class.

**Key points**:
- ✅ `match_signature=True` simplifies **input** handling
- ⚠️ You **still need `output_map_impl`** for output conversion
- ✅ Test `_reference_impl` separately before adding decorator

**When in doubt**: Try `match_signature` first. If it feels awkward, fall back to `input_map`.
