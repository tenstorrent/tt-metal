# ProgramRunArgs tests

Tests for `SetProgramRunArgs`, `UpdateProgramRunArgs`, `UpdateTensorArgs` and `MergeProgramRunArgs`: which
arguments are accepted, and where the values land in the kernel's argument buffers. Programs are never enqueued;
the tests inspect the Program on a mock device.

The rules come from `program_run_args.hpp` and `program.hpp`:

- `kernel_run_args` names each kernel at most once (`DuplicateKernelParamsFails`). This is the one rule the header
  marks as an invariant.
- `SetProgramRunArgs` needs every declared argument for every node the kernel runs on, and nothing undeclared.
- `UpdateProgramRunArgs` and `UpdateTensorArgs` accept a subset; omitted arguments keep their previous values.
- A tensor argument's `TensorSpec` must match its TensorParameter, as loosened by its `TensorSpecRelaxations`.
