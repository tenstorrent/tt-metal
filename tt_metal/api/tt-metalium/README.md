# tt-metalium public headers

Include them as `<tt-metalium/<folder>/<header>.hpp>`. Headers are grouped by the job they do for a user of the runtime. To find something, start from what you are trying to do:

| Folder | Look here for | Main headers |
|---|---|---|
| `vocabulary/` | Plain value types the other folders share | `core_coord`, `mesh_coord`, `shape`, `device_types`, `hal_types` |
| `data_formats/` | Element and tile representation: data formats, tiles, conversions | `data_format`, `tile`, `bfloat16`, `tilize_utils`, `tile_constants` |
| `discovery/` | What hardware is present and what it can do, before you open a device | `system_mesh`, `hal`, `cluster` |
| `mesh/` | Opening, configuring and closing devices; sub-devices; the allocator | `mesh_device`, `mesh_config`, `sub_device`, `allocator` |
| `mesh/multihost/` | Coordination between hosts | `distributed_context` |
| `memory/buffer/` | Device and host buffers and how they are laid out | `mesh_buffer`, `buffer`, `buffer_types`, `host_buffer` |
| `memory/tensor/` | Tensors and their specs (shape, layout, memory config) | `mesh_tensor`, `host_tensor`, `spec/tensor_spec` |
| `memory/cross_program/` | Device memory that outlives a single program | `global_semaphore`, `global_circular_buffer` |
| `memory/utils/` | Lifetime of borrowed host storage | `memory_pin` |
| `workload/` | Programs and mesh workloads: the work you send to devices | `program`, `mesh_workload` |
| `workload/specification/` | Describing a program: kernels, dataflow buffers, semaphores (Metal 2.0) | `program_spec`, `kernel_spec`, `dataflow_buffer_spec` |
| `workload/run_args/` | Per-run arguments of a program (Metal 2.0) | `program_run_args` |
| `execution/` | Running work: command queues and events | `mesh_command_queue`, `mesh_event` |
| `observability/` | Profiling, capture hooks, memory reporting | `profiler/device_profiler`, `capture_hooks`, `memory_reporter` |
| `utils/` | Convenience code supporting usage of the runtime | `math`, `work_split`, `program_cache` |

## Stability

- **Folders above:** stable API.
- **`legacy/` sub-folders** (`mesh/legacy/`, `workload/legacy/`): still supported, but a stable replacement exists, and new code should use it. `mesh/legacy/device.hpp` (`IDevice`) is replaced by `MeshDevice`. `workload/legacy/` holds the Metal 1.0 way of building programs (`CreateProgram`, `CreateKernel`, circular buffers, runtime args, `ProgramDescriptor`), replaced by `workload/specification/` and `workload/run_args/`.
- **`experimental/`:** for users who accept change. Things graduate out of it into the folders above.
- Headers under `tt_metal/api/internal/` are for Tenstorrent components only, not part of the public API.

The full stability and dependency rules are in [`scripts/validate_api/README.md`](../../../scripts/validate_api/README.md).

## Headers at the top level

Files directly in `tt-metalium/` are forwarding headers left at the old flat paths so existing includes keep compiling. Each one names its new location; include that instead.

## Adding a header

Put it in the folder whose job it does, list it in `tt_metal/sources.cmake`, and run `python3 scripts/validate_api/validate_api.py tt_metal/api`.
