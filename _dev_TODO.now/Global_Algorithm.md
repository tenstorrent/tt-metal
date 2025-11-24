# Layer 1: Algorithm

Pure mathematical transformation semantics.

## Concepts

```yaml
tensor:
  properties:
    - { name: shape, type: "tuple[uint32]" }
    - { name: dtype, type: DataType }
    - { name: rank, expr: "len(shape)" }

dtypes:
  - { name: BFLOAT16, size: 2 }
  - { name: FLOAT32, size: 4 }
  - { name: UINT16, size: 2 }
  - { name: UINT32, size: 4 }

shape_notation:
  - "[B, C, H, W]"     # 4D: Batch, Channel, Height, Width
  - "shape[i]"         # Size of dimension i
  - "shape[-1]"        # Last dimension
```

## Precondition Schema

```yaml
precondition:
  id: string           # Unique identifier (P1, A1, etc.)
  entity: string       # Target entity path (Input.shape, Output.dtype)
  attr: string         # Attribute being checked (rank, value)
  rel: string          # Relation: ==, !=, <, <=, >, >=, %, in
  value: any           # Expected value
  expr: string         # Alternative: expression for computed value
```

## Postcondition Schema

```yaml
postcondition:
  id: string
  entity: string
  rel: string
  expr: string         # Expression defining the result
```

## Transformation Patterns

```yaml
patterns:
  elementwise_unary:
    rule: "Output[i] = f(Input[i]) for all i"
    shape_invariant: "Output.shape == Input.shape"

  transpose:
    rule: "Output[..., i, j] = Input[..., j, i]"
    shape_change: "dims swapped"

  layout_reorder:
    hwc_to_chw: "Output[b, c, h, w] = Input[b, h, w, c]"
    chw_to_hwc: "Output[b, h, w, c] = Input[b, c, h, w]"
```

## Invariants

```yaml
invariants:
  - { id: I1, rule: "output.volume == input.volume (for reshapes)" }
  - { id: I2, rule: "output.dtype == input.dtype (unless explicit cast)" }
  - { id: D1, rule: "Same inputs + same seed = same outputs" }
  - { id: D2, rule: "No side effects on input tensors" }
```
