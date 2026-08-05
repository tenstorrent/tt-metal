---
name: tensor-shape
description: >-
  Convert LLK lib/API tile-size args to ckernel::TensorShape and maintain TRISC
  TensorShape coverage. Use when adding TensorShape parameters, replacing
  face_r_dim/num_faces, editing LLK_VALIDATE_TENSOR_SHAPE_*, regenerating
  tensor_shape_coverage_*.h, or reviewing TensorShape PRs.
argument-hint: "[convert|coverage|review] [op-or-path]"
user_invocable: true
---

# /tensor-shape — TensorShape Plumbing & Coverage

## Usage

```
/tensor-shape convert unpack_A
/tensor-shape coverage harvest unpack_A
/tensor-shape review
```

## Mandatory First Step

Read the canonical reference and follow it:

```
Read(".claude/references/tensor-shape.md")
```

For metal-layer propagation after signature changes, also read `.claude/references/metal-integration.md`.

## Modes

### convert

Plumb `TensorShape` through an LLK op (lib → metal API → tests/fuser):

1. Replace `face_r_dim` / `num_faces` (and similar) with `ckernel::TensorShape`.
2. Include `tensor_shape.h` + the matching `tensor_shape_coverage_{unpack|math|pack}.h`.
3. Add `LLK_VALIDATE_TENSOR_SHAPE_{UNPACK|MATH|PACK}("_llk_foo_...", tensor_shape)` — **string literal** name; **no** API registry edit in `coverage.h`.
4. Prefer `num_faces_r_dim` / `num_faces_c_dim` for new logic; keep flat `total_num_faces()` only when intentionally preserving old mop behavior.
5. Update WH + BH together; Quasar only when in scope.
6. Metal wrappers: `get_operand_tensor_shape(operand_id)`.
7. Tests/fuser: `make_tensor_shape` / `cpp_tensor_shape`; never legacy helper for 32×16.

### coverage

Discover or regenerate TRISC allowlists:

1. Do **not** add a per-API enum entry.
2. Run discovery with `TT_LLK_DISABLE_ASSERTS=1` + DPRINT, then:
   `helpers/tensor_shape_coverage_parser.py harvest|emit|summary`
3. New shape → named `TENSOR_SHAPE_FR*_NF*` in `tensor_shape_coverage.h` if needed + TRISC predicate update via emit.
4. New harvestable `fn_name` → add to `MATH_FUNCTIONS` / `UNPACK_FUNCTIONS` in the parser only.

### review

Apply the **Review Checklist** in `.claude/references/tensor-shape.md`. Especially correct the misconception that every new API must be listed in `coverage.h` — coverage is TRISC-scoped; call sites only need the validate macro + string literal.

## Key Files

| Role | Path |
|------|------|
| Struct / helpers | `common/tensor_shape.h` |
| Macros / named shapes | `common/tensor_shape_coverage.h` |
| TRISC tables | `common/tensor_shape_coverage_{math,unpack,pack}.h` |
| Parser | `tests/python_tests/helpers/tensor_shape_coverage_parser.py` |
| Python TileShape | `tests/python_tests/helpers/tile_shape.py` |
| Metal operand helper | `tt_metal/hw/ckernels/{arch}/metal/llk_io/llk_operands.h` |
