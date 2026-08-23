# Program Cache Hash Collision

## Description

TTNN's program cache keys a compiled `Program` by a hash computed from a device
operation's attributes and input tensor specs (`compute_program_hash()`, or the
default hash for ops that don't override it). If two structurally different
invocations of the same op hash to the same key, the program cache returns the
*first* compiled program for the *second* call — the kernels, CBs, and static
runtime args from call #1 are silently reused for call #2's shapes/dtypes/config.
This does not crash; it produces wrong output or a downstream assert far from
the actual bug.

Root cause is almost always the same shape: the hash implementation includes
some but not all of the fields that actually change kernel generation. Common
gaps are optional tensors (present vs. absent), broadcast/alignment flags that
differ per operand, and shape fields hashed by rank or volume instead of the
specific dims that affect tiling and padding.

Tenstorrent hit this at scale: PR #45821 fixed a program-cache hashing bug in
one CCL op, and thirteen more ops (all-gather, reduce-scatter, dropout, minimal
matmul + strided reduce-scatter, neighbor pad, slice reshard, and others) each
needed their own follow-up fix for the identical root cause, because their hash
implementations had the same class of gap.

## What to Look For

1. **New or modified `compute_program_hash()`**: check every attribute and
   input tensor field the op's `create()` / program factory actually reads to
   decide kernel structure. If a field is read in `create()` but not folded
   into the hash, two calls that differ only in that field collide.

2. **Optional inputs/outputs**: ops with an optional bias, residual, mask, or
   output tensor must hash "provided" vs. "not provided" as a distinct case —
   not just hash the provided tensor's shape when present and skip the field
   entirely when absent.

3. **Per-operand broadcast/alignment flags**: binary/ternary ops where each
   operand can independently broadcast must include each operand's broadcast
   decision in the hash, not just the output shape (two operand pairs can
   alias to the same output shape via different broadcast paths).

4. **Shape hashed by rank/volume instead of dims**: hashing
   `shape.rank()` or `shape.volume()` instead of the actual per-dimension
   values means different-rank shapes with identical padding, or same-volume
   shapes with different tiling, collide.

5. **New `DeviceOperation` with no explicit hash override**: silently
   inherits a default hash. Confirm the default is specific enough for this
   op's actual configuration space before assuming it's fine.

6. **Hashing a buffer address** — the inverse mistake: *including* a field that
   must not be in the key. Folding `buffer()->address()` into the hash puts the
   allocation itself into the cache key, so every reallocation produces a new
   entry: unbounded cache growth, a fresh compile on essentially every call, and
   the cache stops functioning as a cache. An address is never a structural
   property. If a hash is reaching for one, the thing actually needed is a
   per-dispatch refresh (a `BufferBinding`, or a re-apply in
   `override_runtime_arguments`), not a key field.

7. **`ProgramDescriptor::custom_program_hash` misuse**: this replaces the
   descriptor's *structural* hash outright. It is **not** the op's
   `compute_program_hash`, and setting it discards the structural keying the
   framework would otherwise derive. You almost never want it — treat any new
   use as needing explicit justification.

8. **Excluding shape while the program bakes a per-core work distribution**: a
   hash that omits shape or volume lets one cached program be shared across
   different work splits, so per-core tile counts baked at the first miss are
   wrong for later calls. Excluding a field is only safe if it is also
   re-applied per dispatch — there is no third category.

## Bad Code Examples

```cpp
// BUG: hash only accounts for rank, not the actual padded dims — a
// different-rank shape with identical padding collides with this one
size_t compute_program_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        operation_attributes.some_flag,
        input_tensor.padded_shape().rank());
}
```

```cpp
// BUG: optional bias tensor's presence isn't part of the hash — a call
// with bias and a call without bias can hash identically if the other
// fields happen to match
size_t compute_program_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        input_tensor_a.padded_shape(),
        input_tensor_b.padded_shape());
    // `bias` is read in create() but never enters the hash
}
```

```cpp
// BUG: only the output shape is hashed; two different per-operand
// broadcast decisions that alias to the same output shape collide
size_t compute_program_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(output_shape);
}
```

```cpp
// BUG: hashing a buffer address puts the allocation in the cache key. Every
// reallocation is a brand-new entry, so the cache grows without bound and
// recompiles on essentially every call. The address needed a per-dispatch
// refresh (a BufferBinding), not a key field.
size_t compute_program_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        input_tensor.padded_shape(),
        input_tensor.buffer()->address());
}
```

```cpp
// BUG: custom_program_hash replaces the descriptor's STRUCTURAL hash. This
// is not compute_program_hash — setting it here throws away the framework's
// structural keying and silently narrows the key to one attribute.
ProgramDescriptor desc = build_descriptor(...);
desc.custom_program_hash = tt::stl::hash::hash_objects_with_default_seed(attrs.mode);
```

## Good Code Examples

```cpp
// GOOD: hashes the actual padded shape (not just rank), so
// different-rank shapes with identical padding get distinct keys
size_t compute_program_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        operation_attributes.some_flag,
        input_tensor.padded_shape());
}
```

```cpp
// GOOD: optional tensor's presence is an explicit part of the hash
size_t compute_program_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        input_tensor_a.padded_shape(),
        input_tensor_b.padded_shape(),
        bias.has_value(),
        bias.has_value() ? bias->padded_shape() : ttnn::Shape{});
}
```

```cpp
// GOOD: each operand's broadcast alignment is folded in independently
size_t compute_program_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        input_tensor_a.padded_shape(),
        input_tensor_b.padded_shape(),
        broadcast_alignment_a,
        broadcast_alignment_b);
}
```

```cpp
// GOOD: only structural properties are hashed. The buffer address is handled
// as a per-dispatch binding instead of a key field, so the cache stays small
// and the address is refreshed on every hit.
size_t compute_program_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        input_tensor.padded_shape(),
        input_tensor.dtype(),
        input_tensor.memory_config());
}

// ... and in the factory:
kernel.emplace_runtime_args(core, {input_tensor.buffer(), num_tiles});
```

```cpp
// GOOD: the shape the work split is derived from is part of the key, so a
// different split is a cache miss rather than a silently-reused program.
size_t compute_program_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        operation_attributes.mode,
        input_tensor.padded_shape(),
        input_tensor.memory_config());
}
```
