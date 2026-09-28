# ProgramRunArgs tests

`ProgramRunArgs` (`program_run_args.hpp`) carries the mutable arguments of a Program. It is always checked against
the ProgramSpec the Program was built from. These tests cover `SetProgramRunArgs`, `UpdateProgramRunArgs` and
`UpdateTensorArgs` (declared in `program.hpp`) and `MergeProgramRunArgs`. They cover both validation and where the
values land in the kernel's argument buffers. Programs are never enqueued; the tests inspect the Program's internal
state on a mock device.

The rules come from the comments in `program_run_args.hpp` and `program.hpp`:

- `kernel_run_args` names each kernel at most once: `DuplicateKernelParamsFails` in `kernel_run_args.cpp`. This is
  the one invariant the header marks as such.
- `SetProgramRunArgs` needs every declared argument for every node the kernel runs on, and nothing undeclared.
- `UpdateProgramRunArgs` and `UpdateTensorArgs` accept a subset of the arguments; omitted arguments keep their
  previous values.
- A tensor argument's `TensorSpec` must match its TensorParameter, as loosened by any `TensorSpecRelaxations`.

Fixtures: `ProgramRunArgsTestQuasar` (RQ), `ProgramRunArgsTestGen1` (RW), `PrefetcherPipeSpecTestQuasar` (PQ) and
`PrefetcherPipeSpecTestGen1TwoChips`. Shared builders are in `run_args_test_helpers.hpp`.

| File | Tests | What is covered |
|---|---|---|
| `kernel_run_args.cpp` | 25 | Kernel and node coverage, vararg counts, repeated `SetProgramRunArgs` calls, kernels that may be omitted |
| `named_runtime_args.cpp` | 8 | Named RTAs and CRTAs, and where their values land |
| `runtime_varargs.cpp` | 7 | Vararg-only RTAs, including per-node vararg count overrides |
| `dfb_run_overrides.cpp` | 13 | DFB size overrides, including aliased DFB groups |
| `tensor_args.cpp` | 14 | Tensor argument completeness and TensorSpec matching (strict, `dynamic_tensor_shape`, `relax_logical_rank`, `match_padded_shape_only`); runtime shape written into CRTAs |
| `update_tensor_args.cpp` | 8 | `UpdateTensorArgs`: patches only tensor binding address slots |
| `update_program_run_args.cpp` | 8 | `UpdateProgramRunArgs`: partial updates retain omitted arguments |
| `borrowed_dfb_attach.cpp` | 6 | Borrowed-memory DFBs take their address from the tensor argument; resizing them |
| `merge_program_run_args.cpp` | 3 | `MergeProgramRunArgs` (no device) |
| `prefetcher_pipe_args.cpp` | 19 | Binding PrefetcherPipe objects to pipe parameters; rebinding; pipe lifetime; a pipe from another mesh |
