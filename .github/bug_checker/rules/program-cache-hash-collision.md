# Program Cache Hash Collision

## Description

TTNN's program cache keys a compiled `Program` by a `ProgramCacheKey` that holds
**both** a 64-bit hash **and** an exact canonical key. The two are not
equivalent, and the difference is what this rule is about.

**The default path is protected.** An op that does not override
`compute_program_hash()` gets the default reflection-based hash over
`(operation_attributes, tensor_args)`. If two structurally different calls
happen to produce the same 64-bit hash, the canonical key still differs, so the
lookup **misses** and the op simply compiles a second program. A default-hash
collision therefore costs a redundant rebuild — it does **not** produce a wrong
result. Do not report fixed default-hash collision cases as correctness bugs.

**Custom hashes opt out of that safety net.** Overriding
`compute_program_hash()` replaces the structural comparison with whatever the
override returns (beyond the operation identity). There is no canonical key left
to disambiguate, so two structurally different invocations that the override
maps to the same value are a genuine **wrong cache hit**: the program cache
returns the *first* compiled program for the *second* call, and the kernels, CBs,
and static runtime args from call #1 are silently reused for call #2's
shapes/dtypes/config. This does not crash; it produces wrong output or a
downstream assert far from the actual bug.

So the target of this rule is narrow and specific: **a custom
`compute_program_hash()` that omits a structural field the program factory
actually reads.** Common gaps are optional tensors (present vs. absent),
broadcast/alignment flags that differ per operand, and shapes reduced to a
single scalar (rank or volume) instead of the dims that drive tiling and
padding.

Two distinct pieces of history, which should not be conflated:

- **Issue #45821** is the *default*-hash case: `PermuteDeviceOperation` has no
  custom hash, and the boost-style combiner in `hash_objects` mapped the logical
  shapes `[3, 17, 1, 1]` and `[1, 152, 1, 1]` to the same 64-bit value, causing a
  wrong cache hit in `test_batch_norm`. The cause was a **weak hash combiner**,
  not an omitted field, and the canonical-key comparison described above is what
  now prevents that class from producing wrong results. It is background, not an
  example of what this rule detects.
- **The ~13 per-op fixes** (all-gather, reduce-scatter, dropout, minimal matmul +
  strided reduce-scatter, neighbor pad, slice reshard, and others) are the real
  evidence for this rule: each op had written its own `compute_program_hash()`
  and each had omitted a field its factory depended on.

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

4. **Shape reduced to a single scalar instead of its dims**. Two separate and
   independent failure modes, which should not be conflated:

   - **Hashing `shape.rank()`**: rank is a *discriminator between different
     ranks*, so it distinguishes `[32, 64]` from `[1, 32, 64]`. What it cannot
     do is separate shapes of the **same rank but different dimensions** —
     `[32, 64]`, `[64, 32]`, and `[128, 256]` all hash identically because only
     the number `2` is folded in. That is the collision: same rank, different
     dims.
   - **Hashing `shape.volume()`**: volume separates different element counts,
     but aliases shapes with the **same volume and different tiling** —
     `[32, 64]` and `[64, 32]` both have volume 2048, yet tile into different
     grids and generate different kernels.

   In both cases the fix is the same — hash the actual per-dimension values —
   but they are distinct aliasing conditions and a rule that describes only one
   will miss the other.

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
// BUG: only the rank is hashed, so every shape with the SAME rank collides
// regardless of its dimensions — [32, 64], [64, 32] and [128, 256] all fold
// in as the single value 2 and share one cached program.
size_t compute_program_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        operation_attributes.some_flag,
        input_tensor.padded_shape().rank());
}
```

```cpp
// BUG: a separate aliasing mode — volume distinguishes element counts but
// not tiling, so [32, 64] and [64, 32] (both volume 2048) collide even
// though they tile into different grids and need different kernels.
size_t compute_program_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        operation_attributes.some_flag,
        input_tensor.padded_shape().volume());
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
// GOOD: hashes the actual per-dimension padded shape, so same-rank shapes
// with different dims — and same-volume shapes with different tiling — all
// get distinct keys.
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
