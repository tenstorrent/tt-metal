# Plan: Encapsulate selective_reduce_combine Like all_gather_async

**Status: Implemented.** The program factory now uses `SelectiveReduceCombineProgramArtifacts`, `build_selective_reduce_combine_program_artifacts`, and `selective_reduce_combine_helper_override_runtime_arguments` as described below.

## Goal
Refactor the selective_reduce_combine program factory so the program-building logic lives in a **reusable builder function** and runtime-arg updates in a **reusable override helper**, matching the pattern used in `experimental/ccl/all_gather_async` (AllGatherProgramArtifacts + build_all_gather_async_minimal_default_program_artifacts + all_gather_async_minimal_default_helper_override_runtime_arguments).

## Reference: all_gather_async Pattern
- **Artifacts struct** (`AllGatherProgramArtifacts`): kernel IDs + core list + layout counts (no semaphores).
- **Builder** (`build_all_gather_async_minimal_default_program_artifacts`): takes `Program&` and all params (tensors, coords, semaphores, etc.), builds CBs/kernels/RT args, returns artifacts.
- **Override helper** (`all_gather_async_minimal_default_helper_override_runtime_arguments`): takes program, artifacts (kernel IDs, cores, layout), semaphores, tensors; updates only buffer/semaphore addresses in RT args.
- **create_at**: allocates Program, calls builder, builds `shared_variables_t` from artifacts + semaphores, returns cached program.
- **override_runtime_arguments**: for each program, calls override helper with stored shared_variables and current tensors/semaphores.

## Steps for selective_reduce_combine

### 1. Add program artifacts struct and declarations (header)
**File:** `selective_reduce_combine_program_factory.hpp`

- Add **`SelectiveReduceCombineProgramArtifacts`** (in `ttnn::experimental::prim` or exposed in `ttnn`):
  - `reader_kernel_id` (KernelHandle)
  - `writer_kernel_id` (KernelHandle)
  - `sender_cores` (std::vector<CoreCoord>)
  - No semaphores (caller owns them).

- Declare **`build_selective_reduce_combine_program_artifacts`**:
  - Signature: `(Program& program, const SelectiveReduceCombineParams& operation_attributes, const MeshCoordinate& mesh_coordinate, const std::vector<MeshCoordinate>& all_mesh_coordinates, const SelectiveReduceCombineTensors& tensor_args, Tensor& output_tensor, const GlobalSemaphore& init_semaphore, const GlobalSemaphore& cross_device_semaphore) -> SelectiveReduceCombineProgramArtifacts`
  - Reusable: any caller (device op or future fused op) can build a selective_reduce_combine program and get back artifacts.

- Declare **`selective_reduce_combine_helper_override_runtime_arguments`**:
  - Signature: `(Program& program, KernelHandle reader_kernel_id, KernelHandle writer_kernel_id, const std::vector<CoreCoord>& sender_cores, const SelectiveReduceCombineTensors& tensor_args, Tensor& output_tensor, const GlobalSemaphore& init_semaphore, const GlobalSemaphore& cross_device_semaphore, const std::optional<GlobalSemaphore>& optional_cross_device_semaphore)`
  - Updates reader RT args [0,1] (dense_token_maps, dense_token_counts addresses) and writer RT args [0,3,4] (output, init_semaphore, cross_device semaphore; use optional_cross_device_semaphore when provided).

- Keep **`UnifiedSelectReduce`** and **`shared_variables_t`** unchanged: `shared_variables_t` still holds kernel IDs, cores, and the two semaphores (for create_at return and override). The factory will build `shared_variables_t` from artifacts + semaphores.

### 2. Implement builder in .cpp
**File:** `selective_reduce_combine_program_factory.cpp`

- Add **`build_selective_reduce_combine_program_artifacts`**:
  - Move the entire body of current **`create_at`** (from `Program program{}` through the loop that sets reader/writer runtime args) into this function.
  - Take `Program& program` and the same parameters as above (operation_attributes, mesh_coordinate, all_mesh_coordinates, tensor_args, output_tensor, init_semaphore, cross_device_semaphore).
  - Keep all **detail** helpers used as-is: `data_parallel_split`, `launch_mux_workers`, `add_termination_master_rt_args` (they stay in `namespace detail` and are called only from the builder).
  - At the end, **return** `SelectiveReduceCombineProgramArtifacts{ .reader_kernel_id = ..., .writer_kernel_id = ..., .sender_cores = std::move(sender_cores) }` (no semaphores in return).

### 3. Implement override helper in .cpp
- Add **`selective_reduce_combine_helper_override_runtime_arguments`**:
  - Loop over `sender_cores`; for each core get reader and writer runtime args and set:
    - reader: at(0) = dense_token_maps_tensor.buffer()->address(), at(1) = dense_token_counts_tensor.buffer()->address().
    - writer: at(0) = output_tensor.buffer()->address(), at(3) = init_semaphore.address(), at(4) = optional_cross_device_semaphore.has_value() ? optional_cross_device_semaphore->address() : cross_device_semaphore.address().
  - Use `tt::tt_metal::GetRuntimeArgs` with correct namespace.

### 4. Refactor create_at
- Replace the inline program-building body with:
  - `Program program{};`
  - `auto artifacts = build_selective_reduce_combine_program_artifacts(program, operation_attributes, mesh_coordinate, all_mesh_coordinates, tensor_args, tensor_return_value, init_semaphore, cross_device_semaphore);`
  - `return { std::move(program), shared_variables_t{ .reader_kernel_id = artifacts.reader_kernel_id, .writer_kernel_id = artifacts.writer_kernel_id, .cores = std::move(artifacts.sender_cores), .init_semaphore = init_semaphore, .cross_device_semaphore = cross_device_semaphore } };`

### 5. Refactor override_runtime_arguments
- For each (range, program) in the cached workload, get `shared_variables` and call:
  - `selective_reduce_combine_helper_override_runtime_arguments(program, shared_variables.reader_kernel_id, shared_variables.writer_kernel_id, shared_variables.cores, tensor_args, tensor_return_value, shared_variables.init_semaphore, shared_variables.cross_device_semaphore, operation_attributes.optional_cross_device_semaphore);`

### 6. Include and namespace
- In the .cpp, include the program factory header if not already (it already includes the device operation types via the existing header).
- Builder and override helper can live in `namespace ttnn::experimental::prim` (same as UnifiedSelectReduce) so they can use the same types; optionally expose the builder in `namespace ttnn` with a type alias for artifacts (like all_gather_async) if other namespaces need to call it.

## Files to touch
| File | Changes |
|------|--------|
| `selective_reduce_combine_program_factory.hpp` | Add `SelectiveReduceCombineProgramArtifacts`, declare `build_selective_reduce_combine_program_artifacts` and `selective_reduce_combine_helper_override_runtime_arguments`. |
| `selective_reduce_combine_program_factory.cpp` | Implement builder (move create_at body, return artifacts), implement override helper, slim create_at to call builder and assemble shared_variables_t, slim override_runtime_arguments to call override helper. |

## Optional (later)
- Expose `SelectiveReduceCombineProgramArtifacts` and the builder in `namespace ttnn` for use from fused ops (e.g. moe_routed_expert) without pulling in `experimental::prim` types, if desired.
