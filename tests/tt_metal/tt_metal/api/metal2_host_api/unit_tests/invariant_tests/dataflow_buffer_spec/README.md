# dataflow_buffer_spec invariant tests

Local invariants of `DataflowBufferSpec`, as written in `dataflow_buffer_spec.hpp`.

Most DFB rules depend on the kernels that bind the DFB, so they are structural and live in
[`../program_spec/`](../program_spec/): `dfb_endpoints.cpp`, `work_unit_bindings.cpp`, `dfbs_per_node.cpp`,
`borrowed_memory.cpp` and `dfb_aliasing.cpp`. `DFBAdvancedOptions` rules are in
[`../advanced_options/`](../advanced_options/).

Fixture: `ProgramSpecTestQuasar` (Q).

## Files

| File | Tests | Covers |
|---|---|---|
| `dataflow_buffer_spec.cpp` | 1 | `data_format_metadata` |

## Coverage of the header invariants

| Invariant | Tests |
|---|---|
| `entry_size > 0` and `num_entries > 0` | **Untested** on the spec. The run-time overrides are tested in [`../../program_run_args/dfb_run_overrides.cpp`](../../program_run_args/dfb_run_overrides.cpp) (`DFBEntrySizeOverrideZeroFails`, `DFBNumEntriesOverrideZeroFails`) |
| `data_format_metadata`, when set, is supported on the target architecture | Q `DataFormatNotSupportedOnTargetArchitectureFails` |
| `tile_format_metadata` set requires `data_format_metadata` set | **Untested** |
