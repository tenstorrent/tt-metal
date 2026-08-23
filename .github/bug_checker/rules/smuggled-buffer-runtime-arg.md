# Smuggled Buffer Address in Runtime Args

## Description

When the program cache hits, TTNN does **not** rebuild the `Program` — it reuses
the compiled program from the first call and only patches the specific runtime
args that were registered as dynamic (e.g. via a `BufferBinding` /
`get_dynamic_runtime_args`-style mechanism in `override_runtime_arguments()`).
Any buffer address written into a kernel's runtime-arg vector *without* being
registered that way is frozen at whatever value it had on the call that first
populated the cache entry.

This is easy to introduce because the buggy and correct code look almost
identical at the call site — `buffer->address()` compiles and runs fine on the
very first call (cache miss), and only misbehaves on a subsequent cache-hit
call with a different tensor at the same buffer-binding slot, which is often
several calls or a different test entirely.

The correct form is the declared binding: the *only* way the framework learns an
arg slot holds an address is the `Buffer*` / `const MeshTensor&` overload of
`emplace_runtime_args()` (or `emplace_common_runtime_args()` for common args),
which records a `BufferBinding`. Smuggle the address instead and one of two
things happens, both bad: the op must re-patch the slot by hand on every hit, or
— with no binding and no override — the adapter falls back to a **full
descriptor rebuild on every dispatch**, which is exactly the cost that blew the
perf budget.

```cpp
// WRONG: smuggled
reader_args.push_back(input_tensor.buffer()->address());
kernel.runtime_args.emplace_back(core, reader_args);

// RIGHT: declared binding
kernel.emplace_runtime_args(core, {input_tensor.buffer(), num_tiles, start_id});
```

Tenstorrent has hit this repeatedly enough that a pre-commit guard was added,
plus a nine-op-family sweep to backfill existing violations (moreh,
moreh_adam, reduction, pool, normalization, eltwise, data-movement,
experimental-matmul, examples) and several targeted point fixes
(`point_to_point`, `all_to_all_combine`, `moreh_mean_backward`, `ttnn::pad`).
The guard is `scripts/detect_smuggled_rta.py`, run as the `detect-smuggled-rta`
pre-commit hook over `ttnn/**/device/**/*.{cpp,hpp}`; it flags `.address()`
flowing into a descriptor sink. A deliberate exception is suppressed per line
with a `// smuggled-rta-ok: <reason>` comment — the script accepts the bare
marker, so the *reason* is unverified and is exactly the thing review must
judge. Legitimate reasons look like a workload-scoped `GlobalSemaphore` address
that is stable for the workload's lifetime, or a one-element metadata tensor
whose address the kernel dereferences on device. "I couldn't get the overload to
compile" is not one. The pre-commit guard only catches locally-run, changed
files — it does not protect the whole tree, so this is worth checking on every
diff that touches a program factory.

**Critically, the fix is not always mechanical.** Adding a binding flips a
descriptor op from rebuild-every-hit to fast-patch-only, and fast-patch
refreshes *only* what is declared. So a binding alone is correct only if the
address is the sole per-dispatch value — otherwise adding it *removes* a rebuild
that was silently keeping other stale values correct, converting a slow-but-right
op into a fast-and-wrong one.

## What to Look For

1. **Raw `buffer->address()` in `SetRuntimeArgs`**: a program factory's
   `create()` calling `SetRuntimeArgs(program, kernel_id, core, {..., buffer->
   address(), ...})` where `buffer` comes from an input/output `Tensor`, with
   no matching registration in `override_runtime_arguments()`.

2. **Incomplete `override_runtime_arguments()`**: an `override_runtime_arguments`
   override that patches some but not all of the buffer addresses that
   `create()` wrote into runtime args for the same kernel — check that every
   buffer-derived RTA index has a corresponding patch.

3. **Optional-output aliasing frozen at cache-miss time**: patterns like
   `auto& out = output.has_value() ? output.value() : input;` where the
   resulting buffer address is written into RTAs during `create()`, but the
   aliasing decision isn't re-evaluated (or the new buffer isn't re-patched)
   on a cache-hit call with a different optional-output choice.

4. **Runtime-arg vectors captured by value in `shared_variables`**: a
   `shared_variables_t` struct that stores a fully-built runtime-arg vector
   (including addresses) from `create()`, when it should instead store the
   *indices* of the address slots so `override_runtime_arguments()` can patch
   them.

5. **Work-distribution trap — a binding added without re-keying the hash**: if
   the op's custom hash drops shape or volume, one cached program is shared
   across different work splits, and the per-core tile counts baked at the first
   miss are wrong for the next call. The rebuild that a smuggled address forced
   was silently covering this. When a diff *adds* a binding (or otherwise
   removes a rebuild), check that shape/volume are actually in the hash;
   otherwise the change trades a perf bug for a correctness bug.

6. **Scalar trap — a hash-excluded scalar baked into runtime args**: values like
   `update_idx`, `step`, `lr`, or a packed scalar that are deliberately excluded
   from the program hash but written into rt-args at `create()` time freeze at
   their first-miss value. Every hash-excluded scalar must be re-applied per
   dispatch, not just the addresses.

7. **Suppression comment without a real reason**: a `// smuggled-rta-ok` marker
   whose stated justification is not a workload-scoped stable address. The
   script does not validate the reason, so an unjustified suppression silently
   reintroduces the bug.

For each of items 5 and 6, the deliberate choices are: re-key the hash so the
varying case becomes a cache miss, keep the rebuild, or cover everything in an
override. Picking one implicitly is the bug.

## Bad Code Examples

```cpp
// BUG: buffer address baked directly into the runtime args at create()
// time, with no BufferBinding registration — a cache-hit call with a
// different tensor still reads this stale address
std::vector<uint32_t> reader_rt_args = {
    src_buffer->address(),
    num_tiles,
};
tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

// override_runtime_arguments() exists, but never touches reader_rt_args[0]
void override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t&,
    tensor_return_value_t&) {
    // only num_tiles-related state is patched here — address is never
    // re-written, so it stays pinned to the first call's buffer
}
```

```cpp
// BUG: optional-output aliasing decided once at create() time and baked
// into the RTAs; a later cache-hit call with a different has_value()
// result keeps writing to the first call's aliased buffer
auto& out_tensor = output.has_value() ? output.value() : input_tensor;
rt_args.push_back(out_tensor.buffer()->address());
```

```cpp
// BUG: work-distribution trap. The binding is correct, but the custom hash
// omits shape, so one cached program is reused across different work splits
// and `num_tiles_per_core` — baked at the first miss — is wrong on the next
// call. Adding the binding removed the rebuild that was hiding this.
kernel.emplace_runtime_args(core, {input.buffer(), num_tiles_per_core, start_id});

size_t compute_program_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(operation_attributes.mode);
    // shape/volume excluded, but the program bakes a per-core work split
}
```

```cpp
// BUG: scalar trap. `update_idx` is deliberately excluded from the hash so
// successive decode steps hit the cache, but it is written once at
// create() time and never re-applied — it freezes at its first-miss value.
kernel.emplace_runtime_args(core, {cache.buffer(), input.buffer(), update_idx});
// no override re-writes the update_idx slot
```

```cpp
// BUG: suppression marker with a non-reason — this address is a per-dispatch
// tensor buffer, not a workload-scoped semaphore, so it really is smuggled.
args.push_back((uint32_t)input_tensor.buffer()->address());  // smuggled-rta-ok: overload wouldn't compile
```

## Good Code Examples

```cpp
// GOOD: buffer address registered as a dynamic binding so the program
// cache knows to patch it on every call, hit or miss
tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);
shared_variables.buffer_bindings.push_back(
    BufferBinding{reader_kernel_id, core, /*arg_index=*/0, src_buffer});

void override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t&) {
    for (auto& binding : cached_program.shared_variables.buffer_bindings) {
        auto& rt_args = GetRuntimeArgs(cached_program.program, binding.kernel_id, binding.core);
        rt_args[binding.arg_index] = tensor_args.input.buffer()->address();
    }
}
```

```cpp
// GOOD: optional-output aliasing is re-resolved and re-patched every call,
// not just baked in once at create() time
void override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& out_tensor = output.has_value() ? output.value() : tensor_args.input;
    for (auto& binding : cached_program.shared_variables.output_bindings) {
        auto& rt_args = GetRuntimeArgs(cached_program.program, binding.kernel_id, binding.core);
        rt_args[binding.arg_index] = out_tensor.buffer()->address();
    }
}
```

```cpp
// GOOD: declared binding via the Buffer* overload, and the work split is
// re-keyed into the hash so a different split is a cache miss rather than a
// silently-reused program.
kernel.emplace_runtime_args(core, {input.buffer(), num_tiles_per_core, start_id});

size_t compute_program_hash() const {
    return tt::stl::hash::hash_objects_with_default_seed(
        operation_attributes.mode,
        input_tensor.padded_shape());   // work split is derived from shape
}
```

```cpp
// GOOD: the hash-excluded scalar is re-applied on every dispatch alongside
// the bound addresses, so it never freezes at its first-miss value.
void override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t&) {
    auto& rt_args = GetRuntimeArgs(cached_program.program, cached_program.shared_variables.kernel_id, core);
    rt_args[cached_program.shared_variables.update_idx_slot] = attrs.update_idx;
}
```

```cpp
// GOOD: a suppression whose reason is real — the semaphore is created once
// for the whole workload, so its address is stable across dispatches.
args.push_back((uint32_t)exit_semaphore.address());  // smuggled-rta-ok: persistent GlobalSemaphore, allocated once per workload
```
