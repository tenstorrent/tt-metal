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

Tenstorrent has hit this repeatedly enough that a pre-commit guard was added,
plus a nine-op-family sweep to backfill existing violations (moreh,
moreh_adam, reduction, pool, normalization, eltwise, data-movement,
experimental-matmul, examples) and several targeted point fixes
(`point_to_point`, `all_to_all_combine`, `moreh_mean_backward`, `ttnn::pad`).
The pre-commit guard only catches locally-run, changed files — it does not
protect the whole tree, so this is worth checking on every diff that touches a
program factory.

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
