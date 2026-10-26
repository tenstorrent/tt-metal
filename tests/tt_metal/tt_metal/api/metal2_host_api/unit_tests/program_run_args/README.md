# ProgramRunArgs tests

Tests for `SetProgramRunArgs`, `UpdateProgramRunArgs`, `UpdateTensorArgs` and `MergeProgramRunArgs`: which
arguments are accepted, and where the values land in the kernel's argument buffers. Programs are never enqueued;
the tests inspect the Program on a mock device.

The rules come from `program_run_args.hpp` and `program.hpp`:

- `kernel_run_args` names each kernel at most once (`DuplicateKernelParamsFails`). This is the one rule listed as an
  invariant below.
- `SetProgramRunArgs` needs every declared argument for every node the kernel runs on, and nothing undeclared.
- `UpdateProgramRunArgs` and `UpdateTensorArgs` accept a subset; omitted arguments keep their previous values.
- A tensor argument's `TensorSpec` must match its TensorParameter, as loosened by its `TensorSpecRelaxations`.

## Listed invariants

`ProgramRunArgs` as declared in `program_run_args.hpp`, with every field and only its invariants.

```cpp
struct ProgramRunArgs {
    struct KernelRunArgs {
        KernelSpecName kernel;

        using RuntimeArgValues = Table<std::string, Table<NodeCoord, uint32_t>>;
        RuntimeArgValues runtime_arg_values;

        using CommonRuntimeArgValues = Table<std::string, uint32_t>;
        CommonRuntimeArgValues common_runtime_arg_values;

        AdvancedKernelRunArgs advanced_options;
    };
    // Invariant:
    // - kernel must be unique across kernel_run_args.
    Group<KernelRunArgs> kernel_run_args;

    using TensorArgument = std::variant<std::reference_wrapper<const MeshTensor>>;
    Table<TensorParamName, TensorArgument> tensor_args;

    struct DFBRunOverrides {
        DFBSpecName dfb;
        std::optional<uint32_t> entry_size = std::nullopt;
        std::optional<uint32_t> num_entries = std::nullopt;
    };
    Group<DFBRunOverrides> dfb_run_overrides;

    AdvancedProgramRunArgs advanced_options;
};
```
