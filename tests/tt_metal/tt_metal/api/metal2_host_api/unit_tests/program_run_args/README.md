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
