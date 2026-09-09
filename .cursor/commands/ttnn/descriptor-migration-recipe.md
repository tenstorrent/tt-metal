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

> **Background reading:** "Descriptors and Specs: how a TTNN op describes its program" is the
> full "what good looks like and why" document (cache mechanics, keying, refresh, bad
> practices); this recipe is the mechanical procedure. It lives at
> <https://gist.github.com/dgomezTT/7584e4eb0dc6ddc5214f9a7e90e77181>. For a brand-new op (not
> a migration), also consider a **Spec** factory (`create_program_artifacts`) — Specs are the
> Metal 2.0 direction; see §3 and §6 of that document for whether your op's shape fits yet.

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
- Single-descriptor ops with **no override**: `create_descriptor` can sit directly on the op
  struct — no `program_factory_t`, no `select_program_factory`. Ops **with** an
  `override_runtime_arguments` need a factory struct holding both hooks plus a
  single-alternative `program_factory_t = std::variant<ProgramFactory>` (still no
  `select_program_factory`); the override is rejected on the op struct itself.
- Buffer addresses are patched on cache hits via the bindings you declare with
  `emplace_runtime_args` (§1.3) — declare a binding for every address; never bake one.
- Tensor-backed circular buffer addresses are patched the same way
  (set `.buffer` on `CBDescriptor` for sharded ops).
- Include `<tt-metalium/program_descriptors.hpp>`.

**Know which cache-hit branch your op lands in.** On a cache hit the framework picks one of
three branches, in priority order:

- **(a)** The program factory defines `override_runtime_arguments()` → the framework calls it
  and does **nothing else** — no binding patching. The op owns the whole refresh. This is the
  target mechanism; write it surgically (§1.4). The hook MUST live on the factory struct: a
  static_assert rejects it on the DeviceOperation itself (only the factory is probed there,
  so an operation-scope hook would be silently dead code).
- **(b)** Else, if the factory declared runtime-arg buffer bindings → fast patch: only the
  bound address slots are rewritten. Correct only when addresses are the *only* thing that
  varies between two calls sharing a key.
- **(c)** Else → slow path: `create_descriptor()` re-runs on **every dispatch**. Correct but
  expensive — this is the host cost that blew the perf budget. Don't land a new op here.

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
`GlobalSemaphore`s / call `Synchronize`, own device scratch (halo lookup tables),
or emit different programs per coordinate (CCL rings) need to do that **once per
workload** (cache miss), not once per dispatch. For those ops, return the
framework's `tt::tt_metal::WorkloadDescriptor`
(`<tt-metalium/workload_descriptor.hpp>`):

```cpp
struct WorkloadDescriptor {
    std::vector<GlobalSemaphore> semaphores;   // workload-scoped resources, kept alive
    std::vector<WorkloadBuffer> buffers;       //   for the cached workload's lifetime
    struct PerCoordProgram { distributed::MeshCoordinateRange range; ProgramDescriptor descriptor; };
    std::vector<PerCoordProgram> programs;     // which program goes where
};

static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
    const operation_attributes_t&,
    const tensor_args_t&,
    tensor_return_value_t&,
    const ttnn::MeshCoordinateRangeSet& tensor_coords);
```

The contract:

1. **Cache miss**: `create_workload_descriptor` is called ONCE. Allocate
   `GlobalSemaphore`s / run `Synchronize`, park them (and any op-owned scratch, via
   `WorkloadBuffer{owner, buffer}` — `owner` is typically a `shared_ptr<Tensor>` deferring
   deallocation) on the descriptor so they outlive the call, and push one `PerCoordProgram`
   per coordinate — or per range, since `programs` is range-keyed: one program replicated
   across the mesh is a single entry. The framework materialises `programs` verbatim into
   the cached `MeshWorkload`.
2. **Cache hit**: the factory is **not** invoked and there is **no rebuild fallback** (a
   rebuild would reallocate the semaphores and re-run the barrier). Only declared
   `BufferBinding`s are patched, so use `emplace_runtime_args()` with `Buffer*` for every
   position that can change between dispatches.
3. **The per-`Program` `override_runtime_arguments` hook is NEVER called on this path**
   (#52554) — it compiles, looks right, and everything it was meant to re-apply freezes at
   the first miss. If the op also has a hash-excluded scalar, wrap the descriptor adapter in
   a hand-written `<Op>MeshWorkloadFactory` whose override takes the whole cached workload:
   delegate to `descriptor_adapter_t::apply_descriptor(...)` first (binding behavior stays
   bit-identical), then patch the extra slots per coordinate. See
   `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.hpp`.

> Single-device ops without workload-scoped state continue to use just `create_descriptor`
> (no workload concept). An op whose program merely *differs per mesh coordinate* (no shared
> resources) instead takes the 4-arg `create_descriptor` with a
> `const std::optional<ttnn::MeshCoordinate>&` — the framework then calls it once per
> coordinate instead of once per range; returning an empty descriptor means "no work here".
> Only take the coordinate if the program genuinely differs (cost: one program per device).

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

// Runtime args per core — pass Buffer* (NOT ->address()) so the framework
// records a BufferBinding and can patch the address on cache hits
reader_desc.emplace_runtime_args(core, {src_buffer, offset, count});

desc.kernels.push_back(std::move(reader_desc));
return desc;
```

**Reuse existing kernels** — point `kernel_source` to the old operation's kernels directory.
No need to duplicate kernel files.

**Never smuggle an address.** The two ways to push runtime args are not equivalent:

```cpp
kernel.runtime_args.emplace_back(core, CoreRuntimeArgs{num_tiles});   // raw values only
kernel.emplace_runtime_args(core, {src_buffer, num_tiles, start_id}); // Buffer* => binding
```

Passing a `Buffer*` (or `const MeshTensor&`) records a `BufferBinding`; on cache hit the
framework writes the current address into that slot. Pushing `buffer->address()` as a raw
`uint32_t` hides the pointer: the op then needs a hand-written override or falls to the
rebuild-every-hit slow path. Same for common runtime args (`emplace_common_runtime_args`) and
for tensor-backed CBs (set `.buffer` / `.tensor`, never the address). The
`detect-smuggled-rta` pre-commit hook flags violations; a deliberate exception needs
`// smuggled-rta-ok: <real reason>` (e.g. a workload-scoped `GlobalSemaphore` address).

Compile notes for the binding overload: a bare `0` literal is ambiguous against `Buffer*` →
write `0u`; `page_size()` is `uint64` → `static_cast<uint32_t>`; dynamically built or mixed
lists need `RTArgList` or the `vector<variant<uint32_t, Buffer*>>` overload; an absent
optional tensor passes a null `Buffer*` (the framework emits 0 and registers no binding).

### 1.4 Hashing and keying

No custom `compute_program_hash` is needed by default. The framework automatically hashes
`type_hash<YourDeviceOperation>` + all of `operation_attributes_t` + all of `tensor_args_t`.
The tensor part of the key is the tensor **specs** (shape, dtype, layout, memory config,
shard spec) — **no buffer addresses**. That single fact generates every rule here: anything
derivable from a `TensorSpec` may be baked into the cached program; anything derived from an
address must be re-applied on every dispatch.

While both the old and `_new` operations exist side by side (Phase 1), their program
caches won't collide because the default hash includes `type_hash<YourDeviceOperation>`,
which differs between the two distinct operation types.

**If the default key is wrong, write `compute_program_hash`.** That is the one supported
mechanism for changing the key, and there are exactly two reasons to reach for it:

- **Hashing a derived quantity** the reflection hash can't express (unary hashes shard
  volumes, and padded shape for row-major, because page size depends on width).
- **Excluding a per-call value** that doesn't change the program structure (RNG `seed`, a
  fused `scalar`, a cache slot index, a token range): hashing it would recompile per call,
  so leave it out of the hash and re-apply it per dispatch (below). Put a comment next to
  the hash naming the mechanism that re-applies the excluded value, so both halves of the
  decision are findable from either end.

**Do not narrow `attribute_names` / `attribute_values()` to drop a field from the hash.**
Existing code does this (dropout omits `seed` that way) and it works today, but the loophole
is being removed: `attribute_names` is a reflection/printing contract that happens to feed
the hash, so keying through it makes the key invisible at the place a reader looks for it,
and silently changes what graph capture and op logging report. Express keying in
`compute_program_hash`, where it is one function a reviewer can read.

The cost of a custom hash is real: the op is **opted out of canonical-key collision
resolution**, so a 64-bit hash collision between two configurations of *your* op resolves to
a wrong hit instead of a rebuild. Hash everything structural, exhaustively.

**Every excluded value must be re-applied on every cache hit.** There is no third category:
every attribute is either structural (in the key, baked into the program) or dynamic (out of
the key **and** re-applied per dispatch). A field that is neither is a stale-value bug. The
mechanism is a **surgical `override_runtime_arguments` on the program factory struct**
(never on the DeviceOperation — the adapter static-asserts against that, because only the
factory is probed and an operation-scope hook would be silently dead code):

```cpp
static void override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&,
    const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
```

When present, the framework calls it on every cache hit and does **nothing else** — it
supersedes binding patching (branch (a) of §1.2). Rules for writing it:

1. **Single-source the work split.** Put the core split / per-core layout in one helper
   called by *both* `create_descriptor` and the override, so arg indices can't drift.
   **But keep the override lean**: the descriptor path has no `shared_variables_t`, so
   whatever the helper computes is recomputed on EVERY hit, and the cost scales with core
   count. A full `split_work_to_cores`/`grid_to_cores`/`CoreRangeSet` re-derivation measured
   ~+3µs/dispatch (+8-9%) on a 64-core grid vs the legacy op that stashed `cores` (randn
   trial, interleaved A/B; single-core shapes were within noise) — that alone can breach the
   §2.3 perf threshold. Re-derive only what the patch loop actually needs (often just the
   core coords), and benchmark the hit path, not only correctness.
2. **Must re-apply:** every buffer-address slot (anything emplaced as `Buffer*`), every
   scalar derived from a hash-excluded attribute, and every globally-allocated CB base
   address.
3. **May skip:** anything derived purely from hashed inputs — a hit means it's identical.
   Trace each value to its inputs; never skip on "looks static".
4. **Cover every core** the descriptor emplaced args for, including zero-work / no-op cores.
5. **Never gate the address patching behind an early return.** An override that returns
   early when it has no scalar to write also skips the addresses, and nothing else refreshes
   them (this bit rotary_embedding's prefill path). Gate only the scalars.
6. **Match CBs by `CBIndex`, not by `desc.cbs` position** — positions shift between
   descriptor variants.
7. **Never call `create_descriptor()` from inside the override.** That pays the cache-miss
   host cost on every hit (the ResNet50 20x cliff). The `detect-override-rebuild` pre-commit
   hook rejects it; don't hide a rebuild inside a shared helper either.
8. Pin the `kernel_idx` / `arg_idx` coupling to `create_descriptor`'s `push_back` order with
   named constants or a comment; multi-factory ops mirror `select_program_factory` through a
   shared helper too.

See `sparse_sdpa_program_factory.cpp` for the model: a few addresses and one hash-excluded
scalar written in place, everything else deliberately untouched because a hit guarantees it is
already right. (`interleaved_to_sharded_partial_program_factory.cpp` does the same job but
builds a positional placeholder list and indexes `desc.cbs[i]`; do not copy that part — rule 6
above requires matching CBs by `CBIndex`.)

> **`get_dynamic_runtime_args` is FORBIDDEN in new code.** The older per-slot
> `{kernel_idx, core, arg_idx, value}` dynamic-args hook is legacy and being removed
> (elimination campaign in progress; remaining users are backlog, not examples). The adapter
> static-asserts that an op does not declare both it and `override_runtime_arguments`. A
> migration must never introduce it; when the op being ported has it, replace it with the
> surgical override.

- **In-place ops** (output aliases an input) take the fast path fine — register both as `Buffer*`
  bindings via `emplace_runtime_args`; the framework allows the output==input alias. A duplicate
  among two *distinct* inputs (e.g. `matmul(X, X)`) still bails to the slow path. An optional
  `output_tensor` carried inside `tensor_args` lands in the *input* region and looks ambiguous —
  read the `allow_inplace_output_tensor_alias` comment in
  `tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp` before touching it.

Compile-time values (CB sizes, `#define`s, compile args) can **never** be dynamic — they bake the
kernel ELF. They must stay in the hash.

> **Gotcha for ops with no input tensor** (e.g. `rand`): the framework discovers the mesh device
> via `get_first_object_of_type<MeshDevice*>` over the attrs. A plain struct (no
> `attribute_names`) is searched member-by-member, so `device` is found at any position. But if
> the struct defines `attribute_names` / `attribute_values()`, discovery recurses into that
> tuple and only inspects **element 0** — `device` must be first or dispatch throws "No mesh
> device found". One more reason not to define the pair for keying. Ops with an input tensor
> source the device from `tensor_args`.

> **Precondition for the fast path: the program hash must include everything the per-core runtime
> args depend on — in particular the shape.** The fast cache-hit path (buffer bindings, or a
> surgical override that patches only addresses and hash-excluded scalars) does **not** recompute
> the rest of the runtime args. So it's only correct when "same hash" implies
> "same program structure" — same shape, same work-split, same per-core tile counts/offsets.
>
> Some ops deliberately do the opposite: they **exclude shape from the hash** so one program is
> reused across shapes, with the per-core args (num_tiles, offsets, num_cores) carrying the shape and
> the **slow-path rebuild** recomputing them every dispatch (`binary_ng` hashes `shard_volumes`, not
> shape; it now re-derives those args in
> `BinaryNgDeviceOperation::ProgramFactory::override_runtime_arguments`, not via a rebuild).
> Such shape-agnostic ops **cannot** use bindings-only fast patching — it would leave the
> shape-dependent args stale and miscompute; adding a binding *removes* the rebuild that was
> silently saving them. Pick deliberately between: re-key the hash so a work-split change is a
> miss (losing the cross-shape reuse), keep the slow-path rebuild, or re-derive every per-core arg
> in the override (which is just the rebuild wearing a different name).

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

**The cached test is the one that matters.** A single call per configuration exercises only
`create_descriptor` and proves nothing about the refresh path, which is where the bugs are.
Make the second call use **different tensors of the same spec** (so buffer addresses actually
change between miss and hit), add an in-place call if the op supports it, and if the op has a
hash-excluded value, assert that a differing value does NOT add a cache entry AND does change
the output (re-applied, not frozen).

**Parity check (descriptor ops):** configure the build with
`-DENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK=ON`. Every cache hit is then byte-compared against
a full rebuild (the rebuild is the oracle) and fails with op/kernel/core/arg detail instead
of PCC garbage. Caveats: `ninja` alone leaves the Python-imported `.so` stale (run the
install step and verify the loaded library carries the define); it only covers factories your
suite dispatches *twice* with reallocated tensors; it is wired into the
`ProgramDescriptor` hit branches only — `create_workload_descriptor` ops get no parity
coverage; and for ops whose per-dispatch values are *intentionally* nondeterministic (an
unseeded RNG drawing fresh seeds per call), the rebuild oracle legitimately differs — run
the parity suite with a fixed seed so miss and hit derive identical values.

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
- Which §1.2 cache-hit branch the op lands in. Branch (c) — the slow-path rebuild — is the
  usual culprit: add bindings, or a surgical `override_runtime_arguments`.
- In `mesh_device_operation_adapter.hpp`, verify the hash path is efficient.

---

## Phase 3 — Replace old with new

### 3.1 Update the old device operation header

In `<op_name>_device_operation.hpp`:

1. Add `#include <tt-metalium/program_descriptors.hpp>`
2. For single-descriptor operations **without** an override: remove the `ProgramFactory`
   wrapper struct, `program_factory_t` alias, and `select_program_factory`; place
   `create_descriptor` directly on the operation struct. **With** a surgical override, keep a
   slim factory struct holding `create_descriptor` + `override_runtime_arguments` and a
   single-alternative `program_factory_t` (drop `shared_variables_t`, `cached_program_t`,
   and `select_program_factory`).
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

### 3.3 Reconcile `compute_program_hash`

If the old operation had a custom `compute_program_hash`, classify it before deleting:

- **It re-hashed exactly what the default hash covers** → delete it. The default hash is
  strictly better: it keeps canonical-key collision resolution.
- **It hashed a buffer address** → delete that part; the value it actually wanted is a
  per-dispatch refresh (§1.4). An address in the key means a new cache entry per allocation.
- **It excluded a per-call value** (seed, scalar, range, semaphore address) → deleting it
  would put the value back in the key (recompile per value). Keep a custom hash that omits
  the value — and hashes *everything* structural, exhaustively — and re-apply the value in a
  surgical `override_runtime_arguments` (§1.4). Do **not** re-express the exclusion by
  narrowing `attribute_names`.
- **It hashed a derived quantity** (shard volumes, padded shape for row-major) → keep it.

Verify exclusions with a regression test: a differing value must NOT add a cache entry (it's
not in the key) AND must change the output (it's re-applied, not frozen).

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

### 3.8 Legacy MeshWorkload factories (`create_mesh_workload` / `create_at`)

CCL-family ops sit on `MeshWorkloadFactoryConcept` instead of `ProgramFactoryConcept`. Two
legacy shapes exist, and both migrate to the same target — a factory struct whose only hook is
`create_workload_descriptor` (§1.2):

- **`create_mesh_workload(attrs, tensor_coords, tensor_args, out)`** returning
  `cached_mesh_workload_t = AdaptedCachedMeshWorkload<shared_variables_t>`, usually
  allocating `GlobalSemaphore`s then delegating per coord to a `create_at(attrs, coord,
  tensor_args, out, <resource params...>)` (e.g. `reduce_scatter`).
- **`create_at`-only** — the framework builds the workload by calling `create_at` per coord
  (e.g. `ccl/mesh_partition`).

Teardown mapping:

| Legacy | Becomes |
|---|---|
| `create_mesh_workload` body | `create_workload_descriptor(attrs, tensor_args, out, tensor_coords)` — same skeleton: allocate resources + `Synchronize` once, then loop `tensor_coords` |
| `create_at` body (builds a `Program`) | a per-coord helper that builds a `ProgramDescriptor` (§1.3) → `programs.push_back({MeshCoordinateRange(coord), std::move(desc)})`; extra resource params stay as plain helper params |
| `shared_variables_t`: `GlobalSemaphore`s | `WorkloadDescriptor::semaphores` (framework keeps them alive for the cache entry) |
| `shared_variables_t`: op-owned tensors/buffers | `WorkloadDescriptor::buffers` as `WorkloadBuffer{owner, buffer}` (`owner` = `shared_ptr<Tensor>` deferring dealloc) |
| `shared_variables_t`: kernel handles, core lists, per-coord metadata for the override | deleted — their only consumer was the override |
| `cached_mesh_workload_t` / `AdaptedCachedMeshWorkload` | deleted |
| `override_runtime_arguments(cached_mesh_workload_t&, ...)` | deleted — replaced by `BufferBinding`s: `emplace_runtime_args` with `Buffer*` for **every** address that varies per dispatch |
| `program_factory_t` variant alternative | the new factory struct (keep the variant) |

Rules specific to this path (they differ from the per-`Program` flow, see §1.2/§4):

1. **There is no rebuild fallback and no per-`Program` override.** On a cache hit only the
   declared bindings are patched. Anything varying that isn't a binding is frozen at first
   miss. If a hash-excluded scalar survives the port, you need the hand-written
   `<Op>MeshWorkloadFactory` wrapper (delegate to `descriptor_adapter_t::apply_descriptor`,
   then patch) — `ring_joint_sdpa_program_factory.hpp`.
2. **Workload-scoped `GlobalSemaphore` addresses are stable** for the cache entry's lifetime,
   so they may be baked as raw args — suppress the guard with
   `// smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)`.
3. **Reconcile the custom hash per §3.3.** Legacy CCL hashes sometimes key on semaphore or
   buffer addresses — that's the §4.4 antipattern (unbounded cache growth); the semaphore is
   workload-scoped now, so drop it from the key entirely.
4. **The parity check does not cover this path** (§2 caveats), so the cache-hit regression
   test — second call with different tensors of the same spec — is the *only* net. Non-trivial
   CCL ops should also run twice under a shifted allocation to prove every binding is declared.
5. Coordinates that should do nothing: emit no `PerCoordProgram` for them (don't push empty
   descriptors).

Reference conversion: `all_gather_via_broadcast_factory.cpp` (semaphores + `Synchronize` on
miss, per-coord descriptors, bindings-only hits).

### 3.9 Build and verify

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
| Hash | Reconcile `compute_program_hash` per §3.3 (delete redundant ones; keep exclusions/derived quantities) |
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

5. **By-value runtime-args grid copies.** `GetRuntimeArgs(program, kernel_id)` (the grid
   form) returns `std::vector<std::vector<RuntimeArgsData>>&`; binding it to a by-value local
   (or plain `auto`) deep-copies the whole per-core arg grid. Invisible in testing —
   `RuntimeArgsData` is a view, so writes through the copy still land — but pure wasted work
   on every cache hit. Hoist above the loops and take `auto&`.

6. **A per-`Program` override on a `create_workload_descriptor` op is never called.** It
   compiles and silently freezes everything it was meant to re-apply. Use the
   mesh-workload-scoped `<Op>MeshWorkloadFactory` shape (§1.2), and prove reachability with
   `clang++ -fsyntax-only` static_asserts rather than assuming (#52554).

7. **Pre-commit guards.** `detect-smuggled-rta` (raw `.address()` into a descriptor sink) and
   `detect-override-rebuild` (`create_descriptor` called from an override) both run on
   `ttnn/**/device/**`. A migration should *remove* lines from
   `scripts/detect_override_rebuild_baseline.txt`, never add them.

## Example Reference

See the FullLike operation for the simplest complete example:
- Descriptor-based: `ttnn/cpp/ttnn/operations/full_like/device/full_like_factory.cpp`
- Header: `ttnn/cpp/ttnn/operations/full_like/device/full_like_device_operation.hpp`

See the Bernoulli operation for another complete example:
- Factory: `ttnn/cpp/ttnn/operations/bernoulli/device/bernoulli_program_factory.cpp`
- Header: `ttnn/cpp/ttnn/operations/bernoulli/device/bernoulli_device_operation.hpp`

Surgical `override_runtime_arguments` (the target cache-hit mechanism, §1.4):
- `ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp`
- Hash-excluded scalar re-applied per dispatch:
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sparse_sdpa_program_factory.cpp`
- Complete override for a shape-agnostic key + CB refresh by `CBIndex`:
  `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`
- Override done *wrong* (rebuild — do not copy): `eltwise/unary` and the other entries in
  `scripts/detect_override_rebuild_baseline.txt`

Declarative `WorkloadDescriptor` examples:
- Workload-scoped semaphores:
  `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_via_broadcast_factory.cpp`
- Op-owned device buffers (halo lookup table, avg-pool scalar config):
  `ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.hpp` +
  `pool_multi_core_program_factory.cpp`
- Workload-scoped override for a hash-excluded scalar:
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.hpp`

Per-coordinate program (4-arg `create_descriptor`):
- `ttnn/cpp/ttnn/operations/debug/device/apply_device_delay_device_operation.cpp`

Full doctrine, reviewer checklist, and source map:
<https://gist.github.com/dgomezTT/7584e4eb0dc6ddc5214f9a7e90e77181>.
