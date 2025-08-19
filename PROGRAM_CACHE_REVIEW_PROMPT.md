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

### Guidelines for updating runtime arguments on cache hit

This section is critical for correctness when a cached program is reused.

- **Do not update hashed properties**
  - Tensor and OP properties that participate in the program hash are safe to leave unchanged in `override_runtime_arguments`. They can be compile-time or runtime kernel args. Any variables derived solely from hashed inputs are also safe to skip.

- **Always update tensor buffer addresses**
  - Buffer base addresses are not hashed and are allocated at runtime; they must be updated on every cache hit.
  - Non-sharded tensors: update the single buffer base address (and any dynamic size/offset) in all reader/writer/compute kernels that reference it.
  - Sharded tensors: update per-shard base addresses and any per-shard offsets/strides for each core/range that consumes the shard. Ensure the override iterates the same ranges used during program creation.

#### Examples: updating tensor buffer addresses in overrides

- Sharded input tensors: adjust circular buffer sizes and page sizes, and update per-core addresses.

```cpp
// In override_runtime_arguments(...): update CB layout and per-core args for a sharded input
void SomeOp::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    const auto& cores = cached_program.shared_variables.cores;

    // CB handle and buffer index captured during create(...)
    CBHandle cb_src0 = cached_program.shared_variables.cb_src0;
    uint8_t src0_cb_index = cached_program.shared_variables.src0_cb_index;
    auto reader_kernel_id = cached_program.shared_variables.reader_kernel_id;

    const auto& in0 = tensor_args.input_tensor_a; // sharded tensor
    const uint32_t tile_bytes = tt::tt_metal::tile_size(
        tt::tt_metal::datatype_to_dataformat_converter(in0.dtype()));

    for (const auto& core : cores) {
        // Derive per-core shard tile count/offset from the sharding spec
        uint32_t tiles_for_core = /* derive from in0.shard_spec for this core */;

        // Ensure CB capacity and page size match current run requirements
        UpdateCircularBufferTotalSize(program, cb_src0, tiles_for_core * tile_bytes);
        UpdateCircularBufferPageSize(program, cb_src0, src0_cb_index, tile_bytes);

        // Update per-core runtime args that include base address/offsets
        auto& reader_args = GetRuntimeArgs(program, reader_kernel_id, core);
        reader_args[0] = in0.buffer()->address();
        // reader_args[1] = per_core_offset_tiles; // if applicable
        // reader_args[2] = per_core_stride_tiles; // if applicable
    }
}
```

- Interleaved tensor buffer addresses: update DRAM base addresses directly per core.

```cpp
// In override callback: update interleaved buffer base addresses
auto override_runtime_arguments_callback = [reader_kernel_id, cores](
    const void* /*operation*/, tt::tt_metal::Program& program,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& /*optional_input_tensors*/,
    const std::vector<Tensor>& output_tensors) {
    auto index_dram_buffer = input_tensors.at(0).buffer();
    auto grad_dram_buffer = input_tensors.at(1).buffer();
    auto output_dram_buffer = output_tensors.at(0).buffer();

    auto& runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
    for (const auto& core : cores) {
        auto& runtime_args = runtime_args_by_core[core.x][core.y];
        runtime_args[0] = grad_dram_buffer->address();
        runtime_args[1] = index_dram_buffer->address();
        runtime_args[2] = output_dram_buffer->address();
    }
};
```

- **Update any property not in the hash**
  - Tensor/OP properties not included in the hash must be set via runtime arguments in `override_runtime_arguments`.
  - Such properties should not be compiled in as constants. If they are used as compile-time constants, that is a potential bug: either add them to the hash or convert them to runtime arguments.

- **Maintain exact runtime-argument ordering**
  - The order and indices of runtime arguments passed to kernels in `override_runtime_arguments` must match the order used during program creation. Mismatches lead to silent corruption or kernel faults on cache hits.
  - Typical failure mode: OP writer changes kernel arg order or adds new args in `create(...)` but forgets to update the override.
  - Mitigations:
    - Centralize argument indices (e.g., enums/constants) shared by both `create(...)` and override code paths.
    - Keep per-kernel helper functions that push args in one place, reused by both paths.
    - Add targeted tests that launch the same cached program with varied runtime-only values to catch mis-ordered args.

### Focused task: review override runtime args and write a failing program-cache test

What to do:
- Identify issues in the `override_runtime_arguments` section of an OP’s program factory.
- Write a targeted test that exposes the issue by exercising the program cache hit path.

How to review the override section:
- Trace every runtime-only value used by kernels (buffer base addresses, per-core offsets/strides, sizes, scalar params like seeds or indices) and ensure they are updated on cache hits.
- Cross-check the order of runtime args in override against their order during program creation. Indices must match.
- For sharded tensors, verify per-core iteration covers the same cores/ranges as in `create(...)` and uses the correct `shared_variables` entry.
- Confirm that only non-hashed properties are updated at runtime; hashed properties should not change between runs that hit the same cache entry.

Test recipe (two-run cache test):
1) First run: build cache and validate correctness
   - Allocate inputs/outputs with a given shape/dtype/layout.
   - Call the OP once to seed the program cache. Optionally assert that `device.num_program_cache_entries()` increased by 1.
   - Compute a golden reference and assert PCC passes and the call completes (no hang).
2) Second run: hit cache and trigger override path
   - Reallocate new input/output tensors with the same hashed properties so the same cache entry is used, but with different runtime-only values: new buffer addresses; and optionally different runtime scalars (e.g., seed, offsets) that should be overridden.
   - Call the OP again and expect failure that reveals the bug: PCC mismatch or a hang.

Example reference test names: see `test_group_attn_matmul_with_program_cache` in `tests/tt_eager/python_api_testing/unit_testing/misc/test_attn_matmul.py`.

Starter pytest template (fill in OP specifics):

```python
import pytest, torch, ttnn
from models.utility_functions import comp_pcc

@pytest.mark.timeout(60)
def test_<op_name>_program_cache_override_rtargs(device):
    torch.manual_seed(0)

    # 1) First run compiles and seeds the cache
    a1 = torch.randn(<shape>).bfloat16()
    b1 = torch.randn(<shape_b>).bfloat16()
    tt_a1 = ttnn.Tensor(a1, <in0_dtype>).to(ttnn.TILE_LAYOUT).to(device, <mem_config>)
    tt_b1 = ttnn.Tensor(b1, <in1_dtype>).to(ttnn.TILE_LAYOUT).to(device, <mem_config>)

    num_cache_start = device.num_program_cache_entries()
    out1 = ttnn.experimental.<op_name>(tt_a1, tt_b1, <other_attrs>)
    num_cache_end = device.num_program_cache_entries()
    assert num_cache_end == num_cache_start + 1, "Expected one new program cache entry on first run"

    # Validate correctness
    out1_host = out1.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    golden1 = <compute_golden_from>(a1, b1)
    ok, pcc = comp_pcc(out1_host, golden1)
    assert ok, f"First run PCC failed: {pcc}"

    # 2) Second run hits cache and triggers override path
    # Reallocate inputs to ensure different buffer addresses; keep hashed props identical
    a2 = torch.randn(<shape>).bfloat16()
    b2 = torch.randn(<shape_b>).bfloat16()
    tt_a2 = ttnn.Tensor(a2, <in0_dtype>).to(ttnn.TILE_LAYOUT).to(device, <mem_config>)
    tt_b2 = ttnn.Tensor(b2, <in1_dtype>).to(ttnn.TILE_LAYOUT).to(device, <mem_config>)

    # Optionally vary a runtime-only scalar that must be overridden (e.g., seed/offset)
    out2 = ttnn.experimental.<op_name>(tt_a2, tt_b2, <same_other_attrs_but_runtime_value_changed>)

    # Expect a failure that reveals the bug on cache hit
    out2_host = out2.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    golden2 = <compute_golden_from>(a2, b2)
    ok, pcc = comp_pcc(out2_host, golden2)
    assert not ok, "Second run unexpectedly succeeded; override runtime args likely incomplete"
```

Sharded variant hints:
- Convert tensors with `ttnn.interleaved_to_sharded(...)` to force per-core addresses and offsets to change between runs.
- In the second run, keep sharding config the same (so the hash matches) but reallocate tensors or vary runtime-only scalars to exercise per-core override logic.
