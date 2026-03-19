# API Contract: softmax

Golden tests import and call the operation as documented below.
Any pipeline-generated implementation MUST match this contract for golden tests to pass.

## Import Path

```python
from ttnn.operations.softmax import softmax
```

## Function Signature

```python
def softmax(
    input_tensor: ttnn.Tensor,    # bfloat16, TILE_LAYOUT, 4D
    dim: int = -1,                # dimension along which softmax is computed
    *,
    numeric_stable: bool = True,  # subtract max before exp for numerical stability
) -> ttnn.Tensor
```

## Valid Call Patterns

| Pattern | Example | Test File |
|---------|---------|-----------|
| Default (dim=-1, stable) | `softmax(t)` | test_golden_shapes.py |
| Explicit dim=-1 | `softmax(t, dim=-1)` | test_golden_modes.py |
| Explicit dim=-2 | `softmax(t, dim=-2)` | test_golden_modes.py |
| Unstable mode | `softmax(t, numeric_stable=False)` | test_golden_modes.py |
| All explicit | `softmax(t, dim=-2, numeric_stable=False)` | test_golden_modes.py |

## Input Requirements

| Property | Requirement |
|----------|-------------|
| dtype | bfloat16 |
| layout | TILE_LAYOUT |
| rank | 4 (N, C, H, W) |
| H, W | divisible by 32 (tile-aligned) |
| memory | DRAM or L1 |

## Output Requirements

| Property | Requirement |
|----------|-------------|
| shape | Same as input |
| dtype | bfloat16 |
| layout | TILE_LAYOUT |

## Validation (Python-side, pre-device)

The operation MUST raise `ValueError` or `RuntimeError` for:
- Input dtype is not bfloat16
- Input layout is not TILE_LAYOUT
- Input tensor rank is less than 2
- dim is not -1 or -2

## Numerical Tolerances

| Test category | rtol | atol |
|---------------|------|------|
| Standard shapes | 0.02 | 0.1 |
| Near-zero inputs | 0.03 | 0.15 |
| Large magnitudes | 0.02 | 0.1 |
| numeric_stable=False | 0.05 | 0.2 |
