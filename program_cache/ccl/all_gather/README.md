## Program cache review — ccl/all_gather

Status: Reviewed — no program cache issues found.

Context
- This OP uses the old/type-erased infra (`ttnn/api/ttnn/operation.hpp`) and returns a `ProgramWithCallbacks` with an override callback.

Hashing
- Default hash path is used via `hash_operation<AllGather>(operation, input_tensors)` from the old infra.
  - Reference: `ttnn/api/ttnn/operation.hpp` default compute hash path at `compute_program_hash_impl_`.
- Determinants captured through `AllGather` fields and input tensor properties (shape/layout/dtype/memory config). No raw buffer addresses are hashed.

Create-time program details
- Program factory: `ttnn/cpp/ttnn/operations/ccl/all_gather/device/multi_core/all_gather_op_multi_core.cpp`.
- CB sizes and page sizes are derived from input/output page sizes and `AllGatherConfig` and remain stable for identical hashed inputs.
  - Example: CB setup around `L458-L470` using `input_page_size`, `max_pages_per_chunk`.
- Compile-time args/defines for kernels are derived from layout (RM/TILE), sharding, page sizes, topology, etc., all captured by the hash via input tensor properties and operation attributes.

Override runtime arguments (cache-hit path)
- Defined as a lambda returned with the program: see `L1275-L1315` in `all_gather_op_multi_core.cpp`.
- Correctly updated on cache hit:
  - Receiver writer kernel: updates output DRAM base address at arg index 0 for all receiver worker cores.
    - Reference: `L1292-L1299`.
  - Sender reader kernel: updates input and output DRAM base addresses at arg indices 0 and 1.
    - Reference: `L1300-L1308`.
  - Sender writer kernel: updates output DRAM base address at arg index 0.
    - Reference: `L1309-L1311`.
- Other runtime args (counts, offsets, topology-dependent indices, EDM addresses/semaphores) are compile-time-configured for a given hashed config and do not change across runs that hit the same cache entry.
- Sharded tensors: additional runtime args emitted by `ShardedAddrGenArgBuilder::emit_rt_args` contain per-core NOC maps derived from shard spec and device; these remain stable and appropriately are not overridden.
  - References: emitter impl in `ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.cpp:L155-L168`.

Notes
- The override enumerates the same sender/receiver core sets used during creation, ensuring address updates cover all kernels that consume DRAM addresses.
- The OP does not vary any runtime-only scalar that would need override beyond base addresses; counts and offsets are derived from hashed properties (shape, layout, topology, ring index/size, buffer sizes).

Suggested tests
- None required; observed override adequately updates all runtime-only values. If desired, a smoke two-run test could be authored to validate cache entry increment and successful second run with reallocated buffers.
