## Program cache issues review prompt

You are a senior C++ reviewer focusing on program caching correctness and efficiency in this repository.

### Context from this codebase

Device-level program cache API (excerpt from `tt_metal/api/tt-metalium/program_cache.hpp`):

```cpp
// tt_metal/api/tt-metalium/program_cache.hpp (excerpt)
struct ProgramCache {
    bool contains(uint64_t program_hash) const { return this->cache_.count(program_hash) > 0; }

    CachedProgramFactory& get(uint64_t program_hash) { return this->cache_.at(program_hash); }

    void insert(uint64_t program_hash, CachedProgramFactory&& program) {
        this->cache_.insert({program_hash, std::move(program)});
    }

    void enable() { is_enabled_ = true; }
    void disable() { is_enabled_ = false; }
    bool is_enabled() const { return is_enabled_; }

    void set_cache_misses_allowed(bool allowed) { allow_cache_misses_ = allowed; }
    bool cache_misses_allowed() const { return allow_cache_misses_; }

    void clear() { this->cache_.clear(); }
    std::size_t num_entries() const { return this->cache_.size(); }

private:
    bool is_enabled_ = true;
    bool allow_cache_misses_ = true;
    std::unordered_map<uint64_t, CachedProgramFactory> cache_{};
};
```

Default vs. custom program hash (excerpt from `ttnn/api/ttnn/device_operation.hpp`):

```cpp
// ttnn/api/ttnn/device_operation.hpp (excerpt)
template <typename device_operation_t>
auto compute_program_hash(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    if constexpr (DeviceOperationWithCustomProgramCacheConcept<device_operation_t>) {
        ZoneScopedN("Compute custom program hash");
        return device_operation_t::compute_program_hash(operation_attributes, tensor_args);
    } else {
        ZoneScopedN("Compute default program hash");
        return tt::stl::hash::hash_objects_with_default_seed(
            tt::stl::hash::type_hash<device_operation_t>, operation_attributes, tensor_args);
    }
}
```

### How the default program hash is computed (when no custom hash is provided)

When an operation does not provide `compute_program_hash(...)`, the framework computes a key as:

`hash_objects_with_default_seed(type_hash<device_operation_t>, operation_attributes, tensor_args)`

What this implies:
- **Operation type included**: `type_hash<device_operation_t>` ensures different ops never collide even with identical attributes/args.
- **Structured hashing**: `operation_attributes` and `tensor_args` are hashed via `tt::stl` reflection/hash utilities. Standard containers (vector, array, pair, tuple, optional, variant) are hashed element-wise; enums as integral; arithmetic types by value.
- **Determinism requirement**: The hash is deterministic only if all members iterate in a deterministic order. Avoid `unordered_*` in attributes/args or convert them to sorted vectors before hashing. Do not include raw pointers/addresses.
- **Scope of inclusion**: Only fields present in `operation_attributes` and `tensor_args` participate. If these structs contain per-invocation scalars that are intended to be set via `override_runtime_arguments`, the default hash will over-key. Conversely, if they omit fields that affect codegen (dtype, tile size, grid, sharding), the default hash will under-key.
- **Seed**: A fixed default seed is used to keep keys stable across runs of the same build; do not rely on process randomness.

Guidance:
- If your op’s compiled structure depends on data not represented (or represented non-deterministically) in `operation_attributes`/`tensor_args`, implement a custom `compute_program_hash` that serializes exactly those determinants in a stable order.
- Move pure runtime-only values (written by `override_runtime_arguments`) out of hashable attributes to prevent cache fragmentation.

Mesh workload hash adds tensor coordinates (excerpt from `ttnn/api/ttnn/mesh_device_operation_adapter.hpp`):

```cpp
// ttnn/api/ttnn/mesh_device_operation_adapter.hpp (excerpt)
static tt::stl::hash::hash_t compute_mesh_workload_hash(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args) {
    auto hash = compute_program_hash(attrs, tensor_args);
    for (const auto& coord : mesh_device_operation_utils::extract_tensor_coordinates(tensor_args)) {
        tt::utils::hash_combine(hash, coord);
    }
    return hash;
}
```

### How OP execution uses the program cache

1) Program definition and kernels
- Each OP defines a program that launches kernels on device. Compile-time parameters are set during `create(...)` and determine kernel selection, CB sizes, grids, and codegen. Values that vary per invocation should be passed as runtime arguments and updated via `override_runtime_arguments(...)`.

2) Program hash selection
- Each OP has an associated program hash. If the OP implements a custom `compute_program_hash(...)`, that result is used (often to intentionally reuse the same compiled program across shapes/layouts when legal). If not, the default hash is computed from the OP type, `operation_attributes`, and `tensor_args` (tensor shape, dtype, layout, etc.). See the “How the default program hash is computed” section above.

3) Launch flow and cache behavior
- On the first Python invocation, the infra computes the hash and queries the device `ProgramCache`.
  - Miss: Build the full program via the selected program factory’s `create(...)`, insert into the cache.
  - Hit: Fetch the cached program and call `override_runtime_arguments(...)` to update all per-invocation values that may differ from the original creation (e.g., buffer addresses/offsets, sizes, scalar params, seed/step counters). No recompilation occurs.

Why step 3 matters
- Correctness depends on OPs accurately updating runtime arguments on cache hits. Under-keyed hashes without proper overrides can cause stale arguments to be reused. Over-keyed hashes negate caching benefits. The review checklist in this document is intended to detect these issues.

Operational note
- Program cache enables efficient back-to-back model execution. A typical workflow runs the model once to populate the cache and kernel binaries; subsequent runs are fast because each OP skips compilation and only performs the short `override_runtime_arguments(...)` path.

### Two OP infrastructures (syntax differences)

There are two OP infrastructures in this repo, with slightly different APIs for creating programs and overriding runtime arguments:

- Type-erased (old) infra: `ttnn/api/ttnn/operation.hpp`
  - Program creation typically returns a `ProgramWithCallbacks` and stores an `override_runtime_arguments` callback.
  - Overrides are applied via the stored callback during the cache-hit path. See adapter usage in `ttnn/core/old_infra_device_operation.cpp`.

- Fully templated (new) infra: `ttnn/api/ttnn/device_operation.hpp`
  - Defines a typed `ProgramFactory` with `create(...)` and `override_runtime_arguments(...)`, plus optional `compute_program_hash(...)`.
  - Mesh variants use `ttnn/api/ttnn/mesh_device_operation_adapter.hpp` to build per-range workloads and apply per-range overrides.

Reviewer note: When checking caching correctness, account for both styles. In the old infra, ensure the callback-based override is wired and invoked on cache hits; in the new infra, verify the factory’s `override_runtime_arguments(...)` and `shared_variables_t` semantics are correct and that hashing matches factory selection.

### Your task

Review code (or a diff) for potential program cache issues.

Focus on:
- Correctness of cache keys (collisions and overkeying)
- Determinism and stability of hashing
- Cache usage patterns on hit/miss paths
- Runtime argument overrides and shared state safety
- Invalidation/clearing behavior and lifecycle

### Checklist

- **Hash contents**
  - Include every attribute/shape/sharding/tile/layout/dtype/compile-time flag that changes codegen, grid, buffer sizes, or kernel selection.
  - Exclude values that should not fragment the cache (pure runtime scalars written via `override_runtime_arguments`, tensor contents if not shaping codegen).
  - Avoid unstable inputs: pointers/addresses, unordered container iteration order, nondeterministic containers, float rounding mismatches, and environment variables unless they change codegen.
  - For mesh ops, ensure sharding/coordinates are included only where programs differ per range; confirm use of `compute_mesh_workload_hash` is appropriate.

- **Consistency with program creation**
  - Match `select_program_factory` branches with what the hash includes; any branch that affects kernels/CBs/grid must be in the hash.
  - Ensure optional inputs/options used in `create(...)` are represented in the hash if they change compiled structure.

- **Runtime overrides and shared state**
  - On cache hit, verify `override_runtime_arguments` updates all per-invocation device arguments.
  - `shared_variables_t` should contain only state safe to reuse across hits and across ranges for mesh workloads.
  - For mesh workloads, confirm proxying uses the correct `shared_variables` per coordinate range.

- **Cache usage/controls**
  - On miss, if `cache_misses_allowed()` is false, ensure a throw occurs; no silent fallbacks.
  - Respect `ProgramCache::is_enabled()`; avoid reading/inserting when disabled.
  - Verify `program_factory_index` aligns with the selected variant on cache hit.

- **Invalidation/lifecycle**
  - Device reconfiguration/fabric changes should call `disable_and_clear_program_cache()` or `clear_program_cache()`.
  - Review TODOs about stale entries (e.g., program command sequence caching) for risk of stale reuse after manager removal or config changes.

- **Performance anti-patterns**
  - Over-keying that explodes unique hashes for equivalent binaries (e.g., including runtime literal scalars).
  - Under-keying that causes reuse with mismatched kernel args, CB sizes, or grid topology.

- **Thread safety**
  - For global/static caches, check locking, wait/notify, and double-insert paths.
  - Watch for races between existence checks and insertions.

### Red flags to identify

- Custom `compute_program_hash` missing fields present in `create(...)` or `select_program_factory(...)` that change compiled structure.
- Hash includes raw pointers/addresses or iterates unordered containers without defined order.
- Cache-hit path missing `override_runtime_arguments` or using incorrect/shared `shared_variables` across incompatible invocations.
- Mesh workload overrides mismatch by range (wrong `shared_variables` for a coordinate range).
- Cache used when disabled, or entries inserted while disabled.
- Misses silently accepted when `.cache_misses_allowed() == false`.
- Cache not cleared after device reconfiguration/fabric changes.
- Any change to kernel selection/config not reflected in hash (dtype, tile size, tensor rank/shape divisibility, sharding mode, compute kernel config, quantization flags).
- Inclusion of per-invocation scalars in the hash that are set at runtime.
- TODOs noting stale caches that could cause correctness bugs.

### Files/areas to search first

- `ttnn/cpp/**/device/*_device_operation*.{hpp,cpp}` and `*_program_factory.cpp` for `compute_program_hash`, `select_program_factory`, `create`, `override_runtime_arguments`.
- `ttnn/api/ttnn/device_operation.hpp` for default hashing path.
- `ttnn/api/ttnn/mesh_device_operation_adapter.hpp` for mesh hash behavior.
- `tt_metal/api/tt-metalium/program_cache.hpp` for cache behavior and flags.
- `tt_metal/impl/device/device.cpp` for enable/clear/disable-and-clear.
- `tt_metal/impl/program/program.cpp` around program command sequence caching, prefetcher cache, and any TODOs on stale entries.
- `tt_metal/detail/kernel_cache.hpp` and `tt_metal/impl/dispatch/ringbuffer_cache.*` for related caches and eviction logic.

### Report format (use per finding)

- **Title**: short summary
- **Location**: file:line(s)
- **Severity**: High/Medium/Low
- **Type**: Collision risk | Cache fragmentation | Stale reuse | Correctness | Perf | Thread safety
- **Why it’s a problem**: concise rationale tied to this repo’s patterns
- **Evidence**: brief code quotes or logic description
- **Suggested fix**: minimal change to make it correct
- **Hash impact**: under-keyed/over-keyed; what to add/remove from the hash
- **Test hook**: a quick scenario that would have caught it

### Optional quick queries to run

- Find all overrides: search for `compute_program_hash(` and compare each to its `create` and `select_program_factory`.
- Verify all `override_runtime_arguments` sites are invoked on the cache-hit path.
- Check for hashing of `unordered_*` contents without deterministic ordering.
- Look for `set_cache_misses_allowed(false)` and confirm miss handling throws.

---

Do you want this prompt applied to a specific PR or subtree and the findings summarized here?
