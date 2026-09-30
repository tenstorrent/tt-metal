# dataflow_buffer_spec invariant tests

Local invariants of `DataflowBufferSpec` (`dataflow_buffer_spec.hpp`). `borrowed_from` is a pointer, so rules on the
TensorParameter it names are local and tested here; that the name resolves is structural. Most other DFB rules depend
on the kernels that bind the DFB, so they are structural and live in `../program_spec/`; `DFBAdvancedOptions` rules
are in `../advanced_options/`.

## Files

| File | Tests | Covers |
|---|---|---|
| `dataflow_buffer_spec.cpp` | 1 | `data_format_metadata` |
| `borrowed_memory.cpp` | 6 | `borrowed_from`: the named TensorParameter is L1-resident and large enough |

## Coverage

| Invariant | Tests |
|---|---|
| `entry_size > 0` and `num_entries > 0` | **Untested** on the spec. The run-time overrides are tested in `../../program_run_args/dfb_run_overrides.cpp` |
| `data_format_metadata`, when set, is supported on the target architecture | Q `DataFormatNotSupportedOnTargetArchitectureFails` |
| `tile_format_metadata` set requires `data_format_metadata` set | **Untested** |
| When `borrowed_from` is set, the named TensorParameter is L1-resident (L1 or L1_SMALL) | Q `BorrowedMemoryDFBNonL1TensorParameterFails`. Accepted: Q `BorrowedMemoryDFBSucceeds` |
| The DFB fits in the named TensorParameter's per-bank allocation. Needs the allocator, so not an invariant by the header rules | Q `BorrowedMemoryDFBOversizedFails`, Q `BorrowedMemoryDFBLargerThanShardStillFails`. Accepted: Q `BorrowedMemoryDFBShardLargerThanWholeTensorSucceeds`, Q `BorrowedMemoryDFBNdShardLargerThanWholeTensorSucceeds` |
