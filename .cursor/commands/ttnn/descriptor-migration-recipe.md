# Migrate Operation to ProgramDescriptor Pattern

Migrate a device operation from the old `CachedProgram` / `ProgramFactoryConcept` architecture to the new `ProgramDescriptor`-based architecture.

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
// NEW pattern (ProgramDescriptor) — single descriptor, direct on the struct
struct MyDeviceOperation {
    // ... operation_attributes_t, tensor_args_t, etc. ...

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&);

    // No program_factory_t needed!
};
```

**Key points:**
- No `shared_variables_t`. No `cached_program_t`. No `ProgramFactory` wrapper struct.
- `create_descriptor()` is a static method directly on the operation struct.
- The framework synthesizes the variant dispatch wrapper internally.
- No `program_factory_t` alias or `select_program_factory` needed for single-descriptor ops.
- The framework handles buffer address patching on cache hits automatically.
- The framework handles dynamic circular buffer address patching automatically
  (set `.buffer` on `CBDescriptor` for sharded ops).
- Include `<tt-metalium/program_descriptors.hpp>`.

**Multi-variant programs (advanced):**

When an operation needs different program strategies, define named structs with
`create_descriptor` and put them in a variant:

```cpp
struct SmallInput {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};
struct LargeInput {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};
using program_factory_t = std::variant<SmallInput, LargeInput>;
static program_factory_t select_program_factory(
    const operation_attributes_t&, const tensor_args_t&);
```

**Mesh-workload ops with workload-scoped state — `WorkloadDescriptor` pattern:**

Most ops only need `create_descriptor`. But mesh-workload ops that allocate
`GlobalSemaphore`s or call `Synchronize` need to do that **once per workload**
(not once per coord, not once per dispatch). For those ops, define a
declarative `WorkloadDescriptor` that owns both the workload-scoped resources
**and** the per-coord program descriptors:

```cpp
struct MyMeshFactory {
    // Op-defined struct holding the entire workload:
    //   - workload-scoped resources (GlobalSemaphores, Synchronize tokens,
    //     anything that must outlive the per-coord programs)
    //   - a `programs` vector with one ProgramDescriptor per coord (or per
    //     coord-range, if multiple coords share the same program).
    //
    // GlobalSemaphore has no default constructor; wrap each in
    // std::optional<> so WorkloadDescriptor is value-initialisable.
    struct WorkloadDescriptor {
        std::optional<ttnn::GlobalSemaphore> semaphore;
        // ... any other workload-wide resources
        std::vector<std::pair<ttnn::MeshCoordinateRange, tt::tt_metal::ProgramDescriptor>> programs;
    };

    // Builds the entire workload in one call. Invoked ONCE per workload
    // (cache miss). The right place to:
    //   1. Allocate GlobalSemaphores / run Synchronize.
    //   2. Loop over `tensor_coords` and push a ProgramDescriptor per coord
    //      into `programs`.
    // The framework iterates `programs` verbatim to build the MeshWorkload.
    static WorkloadDescriptor create_workload_descriptor(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};
```

When the framework adapter sees `T::WorkloadDescriptor` + `T::create_workload_descriptor`
+ a `programs` field on the descriptor, it dispatches through this contract:

1. **Cache miss**: calls `create_workload_descriptor` ONCE. The factory allocates
   resources and populates `programs`. The adapter then turns each
   `(MeshCoordinateRange, ProgramDescriptor)` pair into a `Program` and adds
   it to the cached `MeshWorkload`.
2. **Cache hit**: the factory is **not** invoked. The framework's
   `BufferBinding` fast path patches buffer addresses directly into the
   cached programs. Declarative factories MUST use `emplace_runtime_args()`
   with `Buffer*` args for every position that can change between dispatches
   — there is no rebuild fallback for declarative ops (a rebuild would
   reallocate GlobalSemaphores).

> Single-device ops without workload-scoped state continue to use just
> `create_descriptor` (no workload concept). Only ops that need to allocate
> something once per workload (`GlobalSemaphore`s, halo lookup tables uploaded
> to device, etc.) need the declarative `WorkloadDescriptor` pattern above.

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

### 1.4 Hashing

No custom `compute_program_hash` is needed. The framework automatically hashes
`type_hash<YourDeviceOperation>` + all of `operation_attributes_t` + all of
`tensor_args_t`.

While both the old and `_new` operations exist side by side (Phase 1), their program
caches won't collide because the default hash includes `type_hash<YourDeviceOperation>`,
which differs between the two distinct operation types.

**Never write a custom `compute_program_hash` to exclude a per-call value.** A legacy op
often hand-wrote a hash that dropped a value which varies per call but doesn't change the
program structure (RNG `seed`, a fused `scalar`, a `from`/`to` range, a semaphore address).
Under the descriptor framework that is a trap: on a cache **hit** only `Buffer*` address
slots are re-patched, so a non-`Buffer` value baked into the runtime args is **frozen** at
the first miss (silent wrong result), unless the op happens to fall to the slow-path rebuild.

The correct pattern (static/dynamic split):

- **Static** (program identity → hashed): exclude the per-call value from the default hash by
  giving `operation_attributes_t` an `attribute_names` / `attribute_values()` pair that lists
  only the structural fields and omits the dynamic one. Do **not** add a `compute_program_hash`.

  ```cpp
  struct operation_attributes_t {
      Shape shape; DataType dtype; MemoryConfig memory_config;
      uint32_t seed;  // dynamic — omitted below
      static constexpr auto attribute_names = std::forward_as_tuple("shape", "dtype", "memory_config");
      auto attribute_values() const { return std::forward_as_tuple(shape, dtype, memory_config); }
  };
  ```

- **Dynamic** (re-patched every dispatch): declare a `get_dynamic_runtime_args` hook returning the
  current value at the (kernel, core, arg) slot `create_descriptor()` wrote it to. The framework
  re-applies these on every cache hit (the non-`Buffer` analog of a `BufferBinding`):

  ```cpp
  static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
      const operation_attributes_t& attrs, const tensor_args_t&, tensor_return_value_t& output,
      const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
  ```

  Each `DynamicRuntimeArg` is `{kernel_idx, core, arg_idx, value}`. Ops without the hook compile
  to nothing — only declare it when you excluded a value from the hash.

  > **Gotcha:** `attribute_values()` is also how the framework **discovers the mesh device** (via
  > `get_first_object_of_type<MeshDevice*>`). For ops with no input tensor (e.g. `rand`) the device
  > lives in the attrs, so it must appear in `attribute_values()` — and as the **first** element,
  > because that helper's tuple path only inspects element 0. Put `device` first; otherwise dispatch
  > throws "No mesh device found". Ops with an input tensor source the device from `tensor_args`, so
  > this only bites device-in-attrs ops.

- **Do not copy-paste** the work-split / core enumeration between `create_descriptor` and
  `get_dynamic_runtime_args`. Extract a shared helper both call, so the miss-build and the
  hit-patch derive the identical core list and value by construction.

- **In-place ops** (output aliases an input) take the fast path fine — register both as `Buffer*`
  bindings via `emplace_runtime_args`; the framework allows the output==input alias. (A duplicate
  among two *distinct* inputs, e.g. `matmul(X, X)`, still bails to the slow path.)

Compile-time values (CB sizes, `#define`s, compile args) can **never** be dynamic — they bake the
kernel ELF. They must stay hashed (keep them in `attribute_values()`).

> **Precondition for the fast path: the program hash must include everything the per-core runtime
> args depend on — in particular the shape.** The fast cache-hit path (buffer bindings +
> `get_dynamic_runtime_args`) only re-patches buffer addresses and your declared dynamic scalars; it
> does **not** recompute the rest of the runtime args. So it's only correct when "same hash" implies
> "same program structure" — same shape, same work-split, same per-core tile counts/offsets.
>
> Some ops deliberately do the opposite: they **exclude shape from the hash** so one program is
> reused across shapes, with the per-core args (num_tiles, offsets, num_cores) carrying the shape and
> the **slow-path rebuild** recomputing them every dispatch (e.g. `binary_ng` hashes `shard_volumes`,
> not shape). Such shape-agnostic ops **cannot** use buffer bindings — the fast path would leave the
> shape-dependent args stale and miscompute. Leave them on the slow path (raw addresses, no
> bindings). Forcing the fast path would require either putting shape back in the hash (losing the
> cross-shape reuse) or re-deriving every per-core arg (which is just the slow-path rebuild).

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

If you're running from IRD where local docker/SSH credentials are not available, run these commands on a standard
dev environment (or CI runner) with normal build access instead of from IRD.

Run the performance test **3-5 times** and compute the average overhead.

### 2.3 Acceptance criteria

| Metric       | Threshold |
|--------------|-----------|
| Correctness  | Bit-exact or within tolerance for stochastic ops |
| Performance  | < 3% overhead (release), < 5% for complex ops |

If performance exceeds the threshold, check:
- In `mesh_device_operation_adapter.hpp`, verify the hash path is efficient.

---

## Phase 3 — Replace old with new

### 3.1 Update the old device operation header

In `<op_name>_device_operation.hpp`:

1. Add `#include <tt-metalium/program_descriptors.hpp>`
2. For single-descriptor operations: remove the `ProgramFactory` wrapper struct,
   `program_factory_t` alias, and `select_program_factory`. Place `create_descriptor`
   directly on the operation struct.
3. For multi-variant operations: replace each factory struct's `shared_variables_t`,
   `cached_program_t`, `create()`, and `override_runtime_arguments(cached_program_t&, ...)`
   with `create_descriptor()`. Keep `program_factory_t` and `select_program_factory`.
4. **Preserve the original copyright year.**

### 3.2 Replace the program factory `.cpp` file(s)

Copy the descriptor-based implementation from the `_new` directory into the old
directory. Update:
- Namespace (from `<op_name>_new` to `<op_name>`)
- Include paths (from `<op_name>_new_device_operation.hpp` to `<op_name>_device_operation.hpp`)
- Class names (from `<Op>NewDeviceOperation` to `<Op>DeviceOperation`)

### 3.3 Delete `compute_program_hash`

If the old operation had a custom `compute_program_hash`, delete it. The framework
handles hashing automatically.

If that custom hash existed to **exclude a per-call value** (seed, scalar, range, semaphore
address), deleting it would put the value back in the key (recompile per value). Instead apply
the static/dynamic split from §1.4: omit the value from `attribute_values()` and re-apply it via
`get_dynamic_runtime_args`. Verify with a regression test: a differing value must NOT add a cache
entry (it's not in the key) AND must change the output (it's re-patched, not frozen).

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
| Hash | Delete `compute_program_hash` if present (framework handles it) |
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

3. **External consumers of old factories.** Check sparse matmul, CCL fusion ops, and
   any other code that directly instantiates your factory type.

4. **Don't delete useful comments.** When copying factory implementations, preserve
   algorithmic comments from the original code. These explain non-obvious hardware
   behavior, NOC bandwidth constraints, padding rules, etc.

## Example Reference

See the FullLike operation for the simplest complete example:
- Descriptor-based: `ttnn/cpp/ttnn/operations/full_like/device/full_like_factory.cpp`
- Header: `ttnn/cpp/ttnn/operations/full_like/device/full_like_device_operation.hpp`

See the Bernoulli operation for another complete example:
- Factory: `ttnn/cpp/ttnn/operations/bernoulli/device/bernoulli_program_factory.cpp`
- Header: `ttnn/cpp/ttnn/operations/bernoulli/device/bernoulli_device_operation.hpp`

See `pool/generic` for a complete declarative `WorkloadDescriptor` example
whose descriptor carries device-uploaded helper tensors (halo lookup
table, avg-pool scalar config) alongside the per-coord programs:
- Header: `ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.hpp`
- Factory: `ttnn/cpp/ttnn/operations/pool/generic/device/pool_multi_core_program_factory.cpp`
