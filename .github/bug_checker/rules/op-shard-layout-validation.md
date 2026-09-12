# Missing Shard and Layout Validation at Op Entry

## Description

A TTNN op's device operation is handed tensors whose `Layout` (`TILE` vs
`ROW_MAJOR`) and `MemoryConfig` (interleaved vs height/width/block sharded) are
chosen by the *caller*. The op's program factory, however, is usually written
against one specific combination — it computes core ranges from a shard grid,
sizes circular buffers from a shard shape, or indexes tiles assuming tile
layout.

When the validation hook does not reject the combinations the factory cannot
handle, the factory runs anyway on an input it was never designed for. Because
nothing asserts, the failure surfaces as **silently wrong output, or a hang**,
far from the op that caused it — instead of a clean `TT_FATAL` naming the
unsupported configuration. This was the single largest bug bucket in an audit of
merged `fix` PRs: 74 of them were an op learning to reject, or correctly handle,
a layout or memory-config combination it had previously accepted and mangled.

The most common concrete failure is calling `.shard_spec().value()` on a tensor
that is interleaved. `shard_spec()` returns an optional; dereferencing it on an
interleaved tensor is undefined behaviour, and even when it happens not to
crash, the derived core grid is meaningless. Across
`ttnn/cpp/ttnn/operations/`, `.shard_spec().value()` outnumbers
`.shard_spec().has_value()` by roughly eight to one, so the unchecked
dereference is very much the default habit.

**Scope — this rule is about presence of an entry-level compatibility check, not
about shape arithmetic.** Reshape volume equality, ROW_MAJOR width-of-8
alignment, TILE 32-divisibility, and MoE tile-distribution tables are covered by
the separate `reshape-dim-check` rule; do not re-report those here. This rule
fires on an op that *reads* layout, memory-config, or shard-spec fields to build
its program without first *validating* that it supports what it was given.

## What to Look For

1. **`.shard_spec().value()` without a sharded guard**: any dereference of the
   optional shard spec that is not dominated by `is_sharded()`,
   `shard_spec().has_value()`, or an equivalent `TT_FATAL`. This includes
   dereferences in the validation hook itself, and in helpers the hook calls.

2. **Validation hook that never mentions layout or memory layout**: a
   `validate_on_program_cache_miss()` / `validate()` that checks dtype, storage
   type, and shapes but never constrains `layout()` or
   `memory_config().memory_layout()`, while the corresponding program factory
   branches on, or assumes, one of them. If the factory has a
   `TILE`-only tile-indexing loop, the validator must require `Layout::TILE`.

3. **Partial `TensorMemoryLayout` coverage**: an op that handles
   `HEIGHT_SHARDED` and `INTERLEAVED` but silently falls through for
   `WIDTH_SHARDED` / `BLOCK_SHARDED` — typically an `if/else if` chain or a
   `switch` with no `default:` that raises. Every enumerator the op does not
   support must be rejected explicitly.

4. **Unvalidated input/output memory-config agreement**: ops that require input
   and output to share a memory layout, shard shape, shard orientation, or core
   grid, but only derive the output config from the input without asserting the
   relationship — and multi-input ops that never check the operands' shard specs
   are mutually compatible.

5. **New op or new program factory with no layout/shard precondition at all**:
   an added `DeviceOperation` whose validate hook is empty, trivial, or
   dtype-only. Confirm the factory's actual assumptions and require them
   upfront, rather than relying on callers to pass the right thing.

## Bad Code Examples

```cpp
// BUG: shard_spec() is empty for an interleaved input — this dereferences a
// disengaged optional. Nothing in validate() requires the input to be sharded.
void MyOp::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "Input must be BFLOAT16");
    const auto& shard_spec = input.shard_spec().value();
    TT_FATAL(shard_spec.shape[0] % tt::constants::TILE_HEIGHT == 0, "Bad shard height");
}
```

```cpp
// BUG: the factory indexes the input tile-by-tile, but validate never requires
// TILE layout — a ROW_MAJOR input produces silently wrong output.
void MyOp::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Input must be allocated");
    // no constraint on input.layout()
}
```

```cpp
// BUG: WIDTH_SHARDED and BLOCK_SHARDED fall through this chain with
// num_cores left at 0, which later divides by zero or hangs the program.
uint32_t num_cores = 0;
if (input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
    num_cores = input.shard_spec().value().num_cores();
} else if (input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED) {
    num_cores = compute_grid.x * compute_grid.y;
}
```

```cpp
// BUG: output config is derived from the input without ever checking the two
// operands are shard-compatible; mismatched grids produce garbage.
void MyOp::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input_a;
    const auto& b = tensor_args.input_b;
    TT_FATAL(a.padded_shape() == b.padded_shape(), "Shapes must match");
    // a and b may have different memory layouts, shard grids, or orientations
}
```

## Good Code Examples

```cpp
// GOOD: the sharded branch is guarded, so shard_spec() is only dereferenced
// when it is engaged, and the interleaved case is handled explicitly.
void MyOp::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "Input must be BFLOAT16");
    if (input.is_sharded()) {
        const auto& shard_spec = input.shard_spec().value();
        TT_FATAL(
            shard_spec.shape[0] % tt::constants::TILE_HEIGHT == 0,
            "Shard height {} must be a multiple of the tile height",
            shard_spec.shape[0]);
    }
}
```

```cpp
// GOOD: the layout the factory actually requires is asserted at entry, with a
// message that names the unsupported configuration.
void MyOp::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input must be on device");
    TT_FATAL(
        input.layout() == Layout::TILE,
        "my_op: only TILE layout is supported, got {}. Convert the input with "
        "ttnn::to_layout before calling this op.",
        input.layout());
}
```

```cpp
// GOOD: every TensorMemoryLayout enumerator is either handled or rejected.
uint32_t num_cores = 0;
switch (input.memory_config().memory_layout()) {
    case TensorMemoryLayout::HEIGHT_SHARDED:
        num_cores = input.shard_spec().value().num_cores();
        break;
    case TensorMemoryLayout::INTERLEAVED:
        num_cores = compute_grid.x * compute_grid.y;
        break;
    default:
        TT_THROW(
            "my_op: unsupported memory layout {}. Only HEIGHT_SHARDED and "
            "INTERLEAVED are supported.",
            input.memory_config().memory_layout());
}
```

```cpp
// GOOD: the operands' shard compatibility is an explicit precondition.
void MyOp::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input_a;
    const auto& b = tensor_args.input_b;
    TT_FATAL(a.padded_shape() == b.padded_shape(), "Shapes must match");
    TT_FATAL(
        a.memory_config().memory_layout() == b.memory_config().memory_layout(),
        "my_op: operands must share a memory layout, got {} and {}",
        a.memory_config().memory_layout(),
        b.memory_config().memory_layout());
    if (a.is_sharded()) {
        TT_FATAL(
            a.shard_spec().value() == b.shard_spec().value(),
            "my_op: sharded operands must have identical shard specs");
    }
}
```
