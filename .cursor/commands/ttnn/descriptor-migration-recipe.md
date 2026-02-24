# Migrate Operation to ProgramDescriptor Pattern

Migrate a device operation from the old `CachedProgram` / `ProgramFactoryConcept` architecture to the new `ProgramDescriptor` / `ProgramDescriptorFactoryConcept` architecture.

## Usage

When you need to migrate a device operation to the descriptor pattern, use this command and provide:
- The operation name you're migrating (e.g., 'FullLike', 'Bernoulli')
- The location of the old device operation code

## Overview

The migration has three phases:

1. **Create** a `_new` descriptor-based operation alongside the old one.
2. **Test** that both produce identical results and that the new path has acceptable
   performance overhead (< 3-5 %).
3. **Replace** the old operation's ProgramFactory with the new descriptor-based one,
   delete the `_new` directory, and clean up CMake/test references.

This recipe was validated on the Bernoulli, Matmul, Conv2d, and FullLike operations.

---

## Phase 1 — Create the `_new` descriptor operation

### 1.1 Create directory

```
ttnn/cpp/ttnn/operations/<op_name>_new/
├── CMakeLists.txt
├── <op_name>_new.hpp          # Public API (ttnn::<op_name>_new)
├── <op_name>_new.cpp
└── device/
    ├── <op_name>_new_device_operation.hpp
    ├── <op_name>_new_device_operation.cpp
    └── <op_name>_new_program_factory.cpp   # or factory/ directory for multi-factory ops
```

### 1.2 Device operation header

Replace the old ProgramFactory pattern:

```cpp
// OLD pattern (CachedProgram)
struct ProgramFactory {
    struct shared_variables_t { /* kernel handles, core lists, etc. */ };
    using cached_program_t = CachedProgram<shared_variables_t>;
    static cached_program_t create(...);
    static void override_runtime_arguments(cached_program_t&, ...);
};
```

with the new descriptor pattern:

```cpp
// NEW pattern (ProgramDescriptor)
struct ProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&);

    // Optional: only needed if non-address runtime args change on cache hits
    // (e.g., random seeds, dynamic parameters).
    // Buffer addresses and dynamic CB addresses are auto-patched by the framework.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&);
};
```

**Key points:**
- No `shared_variables_t`. No `cached_program_t`.
- `create_descriptor()` returns a `ProgramDescriptor` that declares CBs, kernels,
  and runtime args declaratively.
- The framework handles buffer address patching on cache hits automatically.
- The framework handles dynamic circular buffer address patching automatically
  (set `.buffer` on `CBDescriptor` for sharded ops).
- `override_runtime_arguments` is only needed for truly dynamic parameters
  (random seeds, semaphore addresses, etc.) — not for buffer addresses.
- Include `<tt-metalium/program_descriptors.hpp>`.

**Optional `prepare_resources` hook:**

If `create_descriptor` needs a device-side resource that isn't in `tensor_args` or
the output tensor (e.g., a config tensor that must be allocated once and kept alive
across cache hits), add a `prepare_resources` static method:

```cpp
struct ProgramFactory {
    // Called once on cache miss. Return value is stored by the framework
    // and passed to create_descriptor on that same cache miss.
    static tt::tt_metal::DeviceStorage prepare_resources(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&);

    // When prepare_resources exists, create_descriptor takes an extra parameter.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&,
        tt::tt_metal::DeviceStorage& resources);  // from prepare_resources
};
```

Most factories do NOT need this. It was needed for Conv2d because it allocates
sliding-window config tensors that must live as long as the cached program.

### 1.3 Program factory implementation (`create_descriptor`)

The descriptor declares everything in a `ProgramDescriptor` struct:

```cpp
ProgramDescriptor desc;

// Circular buffers
desc.cbs.push_back(CBDescriptor{
    .total_size = num_tiles * tile_size,
    .core_ranges = all_cores,
    .format_descriptors = {{CBFormatDescriptor{
        .buffer_index = cb_id,
        .data_format = data_format,
        .page_size = tile_size,
    }}},
});

// Kernels
KernelDescriptor reader_desc;
reader_desc.kernel_source = "path/to/reader.cpp";
reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
reader_desc.core_ranges = all_cores;
reader_desc.compile_time_args = {...};
reader_desc.config = ReaderConfigDescriptor{};

// Runtime args per core
reader_desc.runtime_args.emplace_back(
    core, KernelDescriptor::CoreRuntimeArgs{addr, offset, count});

desc.kernels.push_back(std::move(reader_desc));
return desc;
```

**Reuse existing kernels** — point `kernel_source` to the old operation's kernels directory.
No need to duplicate kernel files.

### 1.4 `compute_program_hash`

**Important for cache-hit performance.** During coexistence (old + new), include a
type hash to avoid cache collisions:

```cpp
tt::stl::hash::hash_t MyNewDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    auto hashable_attrs = attrs;
    hashable_attrs.seed = 0;  // Zero out dynamic fields that don't affect compilation
    return tt::stl::hash::hash_objects_with_default_seed(
        tt::stl::hash::type_hash<MyNewDeviceOperation>,  // Prevents collision with old op
        hashable_attrs, tensors);
}
```

Keep the `type_hash<>` after migration as well — it prevents collisions between
different operations in the shared device cache. Rename it from `type_hash<MyNewOp>`
to `type_hash<MyOp>` in Phase 3.

If all attributes are compile-time deterministic, you can omit `compute_program_hash`
and the framework will use a sensible default that hashes `type_hash<YourDeviceOperation>`,
all of `operation_attributes_t`, and all of `tensor_args_t`.

### 1.5 CMakeLists.txt

Create a `CMakeLists.txt` for the `_new` operation and add it to `ttnn/CMakeLists.txt`:

```cmake
add_subdirectory(cpp/ttnn/operations/<op_name>_new)
```

And add the target to the `ttnn` library's link dependencies.

### 1.6 Public API

In `<op_name>_new.hpp`, expose `ttnn::<op_name>_new(...)` that calls
`ttnn::prim::<op_name>_new(...)`.

---

## Phase 2 — Test correctness and performance

### 2.1 Create comparison test

Create `tests/ttnn/unit_tests/gtests/test_<op_name>_descriptor_benchmark.cpp`:

```cpp
// Correctness tests (non-cached and cached)
TEST_F(MyBenchmark, CorrectnessNonCached) {
    auto old_result = call_old(...);
    auto new_result = call_new(...);
    ASSERT_TRUE(allclose(old_result, new_result));
}

TEST_F(MyBenchmark, CorrectnessCached) {
    // Run once to populate cache, then compare
    call_old(...); call_new(...);
    auto old_result = call_old(...);
    auto new_result = call_new(...);
    ASSERT_TRUE(allclose(old_result, new_result));
}

// Performance test
TEST_F(MyBenchmark, DispatchPerformance) {
    constexpr int N = 1'000'000;  // Use 100k for heavy ops like conv2d

    // Run new FIRST to avoid instruction cache bias
    auto t_new = time([&]{ for (int i = 0; i < N; i++) call_new(...); });
    auto t_old = time([&]{ for (int i = 0; i < N; i++) call_old(...); });

    double overhead = (double(t_new) / double(t_old) - 1.0) * 100.0;
    std::cout << "Overhead: " << overhead << "%" << std::endl;
    EXPECT_LT(overhead, 3.0);  // < 3% overhead threshold
}
```

Register the test in `tests/ttnn/unit_tests/gtests/sources.cmake`.

### 2.2 Run tests (multiple times)

```bash
# Build
./build_metal.sh --build-all            # Release (for performance)
./build_metal.sh --debug --build-all    # Debug (for correctness)

# Run correctness (both debug and release)
TT_METAL_HOME=$PWD ./build_Release/bin/tt-nn-validation-basic \
    --gtest_filter="*DescriptorBenchmark.Correctness*"

# Run performance (release only — debug perf is not representative)
TT_METAL_HOME=$PWD ./build_Release/bin/tt-nn-validation-basic \
    --gtest_filter="*DescriptorBenchmark.DispatchPerformance*"
```

Run the performance test **3-5 times** and compute the average overhead.

### 2.3 Acceptance criteria

| Metric       | Threshold |
|--------------|-----------|
| Correctness  | Bit-exact or within tolerance for stochastic ops |
| Performance  | < 3% overhead (release), < 5% for complex ops |

If performance exceeds the threshold, check:
- Is `compute_program_hash` doing unnecessary work?
- Is `override_runtime_arguments` doing too much? Buffer addresses are auto-patched.
- In `mesh_device_operation_adapter.hpp`, verify the hash path is efficient.

---

## Phase 3 — Replace old with new

### 3.1 Update the old device operation header

In `<op_name>_device_operation.hpp`:

1. Add `#include <tt-metalium/program_descriptors.hpp>`
2. Replace each `ProgramFactory` struct: remove `shared_variables_t`,
   `cached_program_t`, `create()`, and `override_runtime_arguments(cached_program_t&, ...)`
   — replace with `create_descriptor()` and optionally
   `override_runtime_arguments(Program&, ...)`.
3. Update the `program_factory_t` variant if factory types changed.
4. **Preserve the original copyright year.**

### 3.2 Replace the program factory `.cpp` file(s)

Copy the descriptor-based implementation from the `_new` directory into the old
directory. Update:
- Namespace (from `<op_name>_new` to `<op_name>`)
- Include paths (from `<op_name>_new_device_operation.hpp` to `<op_name>_device_operation.hpp`)
- Class names (from `<Op>NewDeviceOperation` to `<Op>DeviceOperation`)

### 3.3 Update `compute_program_hash`

Change `type_hash<MyNewDeviceOperation>` to `type_hash<MyDeviceOperation>` to match
the renamed type. Keep the `type_hash` — it prevents cache collisions between
different operations that happen to hash the same attributes. The default fallback
in `compute_mesh_workload_hash` already includes `type_hash`, so custom hashes
should too for consistency.

### 3.4 Update CMakeLists.txt

- In `ttnn/CMakeLists.txt`: remove the `_new` target from link dependencies and
  `add_subdirectory`.
- In the operation's own `CMakeLists.txt`: replace old factory `.cpp` entries with
  the new descriptor `.cpp` entries.

### 3.5 Delete the `_new` directory

```bash
rm -rf ttnn/cpp/ttnn/operations/<op_name>_new/
```

### 3.6 Delete comparison tests

Remove the benchmark test `.cpp` file and its entry in
`tests/ttnn/unit_tests/gtests/sources.cmake`.

### 3.7 Check for external consumers

**Critical step.** Search for references to old factory types:

```bash
rg "OldFactoryTypeName" --type cpp
```

If external code (e.g., experimental CCL ops, sparse matmul) directly uses the old
factory's types or `shared_variables_t`, you must either:
- Keep the old factory files alongside the new ones (add them back to CMakeLists.txt)
- Migrate the external consumer as well

### 3.8 Build and verify

```bash
./build_metal.sh --debug --build-all    # Debug
./build_metal.sh --build-all            # Release
```

Both must succeed with zero errors.

---

## Quick reference: file changes per operation

| Step | Files changed |
|------|--------------|
| Header | `device/<op>_device_operation.hpp` — ProgramFactory struct |
| Factory impl | `device/<op>_program_factory.cpp` (or `device/factory/*.cpp`) |
| Hash | `device/<op>_device_operation.cpp` — `compute_program_hash` |
| CMake (op) | `<op>/CMakeLists.txt` — source entries |
| CMake (ttnn) | `ttnn/CMakeLists.txt` — remove `_new` subdirectory and link target |
| Tests | `tests/.../sources.cmake` — remove benchmark entries |
| Cleanup | Delete `<op>_new/` directory |

---

## Common pitfalls

1. **Namespace resolution after moving factories.** If factories move to a different
   namespace, unqualified type names (e.g., `MatmulParams`) may stop resolving.
   Use the same parent namespace as the types, or add `using` declarations.

2. **`detail::` namespace ambiguity.** Functions like `detail::preferred_noc_for_dram_read()`
   live in `tt::tt_metal::detail`. If your factory is in a namespace where `detail::`
   resolves differently, fully qualify as `tt::tt_metal::detail::`.

3. **Cache collisions.** Always include `type_hash<YourOp>` in
   `compute_program_hash`. During coexistence use `type_hash<NewOp>`; after migration
   rename to `type_hash<Op>`. Do NOT remove `type_hash` — it prevents collisions
   between different operations in the shared per-device program cache.

4. **`override_runtime_arguments` signature.** The descriptor pattern uses
   `Program&` (not `cached_program_t&`). The kernel handle is an integer index
   matching the order kernels were pushed into `desc.kernels`.

5. **External consumers of old factories.** Check sparse matmul, CCL fusion ops, and
   any other code that directly instantiates your factory type.

6. **Don't delete useful comments.** When copying factory implementations, preserve
   algorithmic comments from the original code. These explain non-obvious hardware
   behavior, NOC bandwidth constraints, padding rules, etc.

## Example Reference

See the FullLike operation for the simplest complete example:
- Descriptor-based: `ttnn/cpp/ttnn/operations/full_like/device/full_like_factory.cpp`
- Header: `ttnn/cpp/ttnn/operations/full_like/device/full_like_device_operation.hpp`

See the Bernoulli operation for an example with `override_runtime_arguments` (seed patching):
- Factory: `ttnn/cpp/ttnn/operations/bernoulli/device/bernoulli_program_factory.cpp`
- Header: `ttnn/cpp/ttnn/operations/bernoulli/device/bernoulli_device_operation.hpp`

See the Conv2d operation for an example with `prepare_resources`:
- Factory: `ttnn/cpp/ttnn/operations/conv/conv2d/device/factory/sharded_descriptor.cpp`
