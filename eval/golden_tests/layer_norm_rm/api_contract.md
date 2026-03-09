# API Contract: layer_norm_rm

Golden tests import and call the operation as documented below.
Any pipeline-generated implementation MUST match this contract for golden tests to pass.

## Import Path

```python
from ttnn.operations.layer_norm_rm import layer_norm_rm
```

## Function Signature

```python
def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,   # Optional scale, shape (1, 1, 1, W)
    beta: ttnn.Tensor = None,    # Optional shift, shape (1, 1, 1, W)
    *,
    epsilon: float = 1e-5,       # Keyword-only, default 1e-5
) -> ttnn.Tensor
```

## Valid Call Patterns

| Pattern | Example | Test File |
|---------|---------|-----------|
| Pure normalization (no affine) | `layer_norm_rm(input)` | test_golden_modes.py |
| With gamma only | `layer_norm_rm(input, gamma)` | test_golden_validation.py |
| Full affine | `layer_norm_rm(input, gamma, beta)` | test_golden_shapes.py |
| Custom epsilon | `layer_norm_rm(input, gamma, beta, epsilon=1e-3)` | test_golden_modes.py |

## Input Requirements

| Property | Requirement |
|----------|-------------|
| input dtype | bfloat16 |
| input layout | ROW_MAJOR_LAYOUT |
| input memory | DRAM_MEMORY_CONFIG |
| input shape | (N, C, H, W) with H, W divisible by 32 |
| gamma shape | (1, 1, 1, W) matching input W |
| beta shape | (1, 1, 1, W) matching input W |

## Output Requirements

| Property | Requirement |
|----------|-------------|
| shape | Same as input |
| dtype | bfloat16 |
| layout | ROW_MAJOR_LAYOUT |

## Validation (Python-side, pre-device)

The operation MUST raise `ValueError` or `RuntimeError` for:
- Non-bfloat16 input dtype
- TILE_LAYOUT input
- gamma shape width != input width
- beta shape width != input width

## Numerical Tolerances

| Test category | rtol | atol |
|---------------|------|------|
| Standard shapes | 0.02 | 0.1 |
| Near-zero inputs | 0.05 | 0.15 |
