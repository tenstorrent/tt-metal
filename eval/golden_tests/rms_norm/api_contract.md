# API Contract: rms_norm

Golden tests import and call the operation as documented below.
Any pipeline-generated implementation MUST match this contract for golden tests to pass.

## Import Path

```python
from ttnn.operations.rms_norm import rms_norm
```

## Function Signature

```python
def rms_norm(
    input_tensor: ttnn.Tensor,      # Input tensor, bfloat16 or float32
    *,
    gamma: Optional[ttnn.Tensor] = None,  # Optional scale, shape (1,1,1,W) where W = input's last dim
    epsilon: float = 1e-6,                # Small constant for numerical stability
) -> ttnn.Tensor
```

## Math Definition

```
RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma
```

Normalization is performed along the **last dimension** (W). When gamma is None, the scale step is skipped.

For float32 input, fp32 accumulation must be used throughout (no intermediate bfloat16 downcast).

## Valid Call Patterns

| Pattern | Example | Test File |
|---------|---------|-----------|
| Minimal (no optional params) | `rms_norm(x)` | test_golden_modes.py |
| With gamma | `rms_norm(x, gamma=g)` | test_golden_shapes.py |
| Custom epsilon | `rms_norm(x, epsilon=1e-5)` | test_golden_modes.py |
| Full invocation | `rms_norm(x, gamma=g, epsilon=1e-5)` | test_golden_shapes.py |

## Input Requirements

| Property | Requirement |
|----------|-------------|
| dtype | bfloat16 or float32 |
| layout | ROW_MAJOR_LAYOUT or TILE_LAYOUT |
| memory | DRAM_MEMORY_CONFIG |
| rank | >= 2 (4D tensors are standard: N, C, H, W) |
| shape | H, W divisible by 32 for TILE_LAYOUT |

## Gamma Requirements

| Property | Requirement |
|----------|-------------|
| shape | (1, 1, 1, W) where W matches input's last dimension |
| dtype | Same as input tensor |
| layout | ROW_MAJOR_LAYOUT |

## Output Requirements

| Property | Requirement |
|----------|-------------|
| shape | Same as input |
| dtype | Same as input (bfloat16 or float32) |
| layout | Same as input |

## Validation (Python-side, pre-device)

The operation MUST raise `ValueError` or `RuntimeError` for:
- Input tensor with rank < 2
- Gamma tensor whose last dimension does not match input's last dimension

## Numerical Tolerances

| Test category | dtype | rtol | atol |
|---------------|-------|------|------|
| Standard | bfloat16 | 0.01 | 0.05 |
| Standard | float32 | 0.001 | 0.01 |
| Near-zero inputs | bfloat16 | 0.02 | 0.10 |
