# Changelog: match_signature Feature

## Summary

Added `match_signature=True` parameter to `@validate_against` decorator, enabling cleaner validation decorators when the reference function has the same signature as the implementation.

## What Changed

### New Parameter

```python
def validate_against(
    reference_fn: Callable,
    input_map: Optional[Callable] = None,
    output_map_impl: Optional[Callable] = None,
    output_map_ref: Optional[Callable] = None,
    metrics: Optional[Dict[str, Callable]] = None,
    tolerances: Optional[Dict[str, float]] = None,
    performance_metrics: bool = True,
    enabled: bool = True,
    match_signature: bool = False,  # ← NEW!
):
```

### Behavior

When `match_signature=True`:
- Reference function is called with **exact same args/kwargs** as implementation
- No `input_map` needed
- Eliminates complex lambda indexing (`args[0]`, `args[1]`, etc.)

## Migration Guide

### Before (Old Pattern)

```python
class RMSNorm:
    @validate_against(
        reference_fn=torch_rms_norm,
        input_map=lambda args, kwargs: (
            (ttnn.to_torch(args[1]).squeeze(0), args[0].weight_torch),
            {"eps": args[0].eps}
        ),
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
        tolerances={'max_abs_error': 1e-3}
    )
    def __call__(self, x):
        return ttnn.layer_norm(x, ...)
```

### After (New Pattern)

```python
class RMSNorm:
    def _reference_impl(self, x):
        """Reference with same signature"""
        x_torch = ttnn.to_torch(x).squeeze(0)
        return torch_rms_norm(x_torch, self.weight_torch, self.eps)

    @validate_against(
        reference_fn=lambda self, x: self._reference_impl(x),
        match_signature=True,  # ← Clean!
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
        tolerances={'max_abs_error': 1e-3}
    )
    def __call__(self, x):
        return ttnn.layer_norm(x, ...)
```

## Benefits

✅ **Cleaner code** - No complex `input_map` lambdas
✅ **Easier to debug** - Reference method can be tested separately
✅ **More readable** - Clear what reference does
✅ **Self-documenting** - `_reference_impl` name explains purpose
✅ **Backward compatible** - Old `input_map` pattern still works

## Files Changed

### Core Implementation
- `ds_r1_qwen.py`: Added `match_signature` parameter and logic

### Documentation
- `MATCH_SIGNATURE_GUIDE.md`: Complete guide with examples (NEW)
- `QUICKSTART.md`: Updated with match_signature examples
- `CHANGELOG_MATCH_SIGNATURE.md`: This file (NEW)

### Examples
- `validation_example.py`: Updated RMSNorm to show both patterns
- `test_match_signature.py`: Unit tests for new feature (NEW)

## Testing

All tests pass:
```bash
python test_match_signature.py
# ================================================================================
# MATCH_SIGNATURE FEATURE TESTS
# ================================================================================
# Test 1: match_signature with method
#   ✓ match_signature with method works!
# Test 2: match_signature vs input_map comparison
#   ✓ Both patterns work correctly!
# Test 3: match_signature with multiple args
#   ✓ Multiple args with match_signature works!
# Test 4: match_signature with kwargs
#   ✓ Kwargs with match_signature works!
# ================================================================================
# TESTS COMPLETE: 4 passed, 0 failed
```

## When to Use

### Use `match_signature=True` when:
- Validating class methods
- Reference function can have same signature
- You want cleaner decorators

### Use `input_map` when:
- Reference is a library function with different signature
- Mapping is simple and inline lambda is clear
- Need to transform args significantly

## Examples

### Example 1: Simple Method

```python
class Layer:
    def _ref(self, x):
        return torch.matmul(ttnn.to_torch(x).squeeze(), self.weight_torch)

    @validate_against(
        reference_fn=lambda self, x: self._ref(x),
        match_signature=True,
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(),
        tolerances={'max_abs_error': 1e-2}
    )
    def __call__(self, x):
        return ttnn.matmul(x, self.weight)
```

### Example 2: Multiple Args

```python
class Attention:
    def _ref(self, q, k, v):
        q_t = ttnn.to_torch(q).squeeze()
        k_t = ttnn.to_torch(k).squeeze()
        v_t = ttnn.to_torch(v).squeeze()
        return torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)

    @validate_against(
        reference_fn=lambda self, q, k, v: self._ref(q, k, v),
        match_signature=True,
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(),
        tolerances={'max_abs_error': 0.1}
    )
    def __call__(self, q, k, v):
        return ttnn.scaled_dot_product_attention(q, k, v)
```

### Example 3: With Kwargs

```python
class Norm:
    def _ref(self, x, eps=1e-6):
        x_t = ttnn.to_torch(x).squeeze()
        return x_t / (x_t.norm() + eps)

    @validate_against(
        reference_fn=lambda self, x, eps=1e-6: self._ref(x, eps),
        match_signature=True,
        output_map_impl=lambda x: ttnn.to_torch(x).squeeze(),
        tolerances={'max_abs_error': 1e-3}
    )
    def __call__(self, x, eps=1e-6):
        return ttnn.normalize(x, eps)
```

## Implementation Details

The decorator checks `match_signature` first:

```python
# Inside decorator wrapper
if match_signature:
    # Reference function has same signature, call with same args/kwargs
    ref_args, ref_kwargs = args, kwargs
elif input_map:
    # Use custom input mapping
    ref_args, ref_kwargs = input_map(args, kwargs)
else:
    # Pass through as-is
    ref_args, ref_kwargs = args, kwargs
```

## Backward Compatibility

✅ **Fully backward compatible**
- All existing code with `input_map` works unchanged
- `match_signature=False` by default
- No breaking changes

## Future Enhancements

Potential future improvements:
- Auto-detect signature match (inspect module)
- Helper decorators: `@validate_with_wrapper(reference_method_name)`
- Automatic `output_map_impl` for common cases

## See Also

- `MATCH_SIGNATURE_GUIDE.md` - Complete guide with patterns
- `QUICKSTART.md` - Quick reference
- `VALIDATION_DECORATOR.md` - Full API docs
- `validation_example.py` - Working examples
