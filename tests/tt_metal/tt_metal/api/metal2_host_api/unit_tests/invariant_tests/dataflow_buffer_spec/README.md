# dataflow_buffer_spec invariant tests

Local invariants of `DataflowBufferSpec` (`dataflow_buffer_spec.hpp`). Most DFB rules depend on the kernels that
bind the DFB, so they are structural and live in `../program_spec/`; `DFBAdvancedOptions` rules are in
`../advanced_options/`.

## Files

| File | Tests | Covers |
|---|---|---|
| `dataflow_buffer_spec.cpp` | 1 | `data_format_metadata` |

## Coverage

| Invariant | Tests |
|---|---|
| `entry_size > 0` and `num_entries > 0` | **Untested** on the spec. The run-time overrides are tested in `../../program_run_args/dfb_run_overrides.cpp` |
| `data_format_metadata`, when set, is supported on the target architecture | Q `DataFormatNotSupportedOnTargetArchitectureFails` |
| `tile_format_metadata` set requires `data_format_metadata` set | **Untested** |
