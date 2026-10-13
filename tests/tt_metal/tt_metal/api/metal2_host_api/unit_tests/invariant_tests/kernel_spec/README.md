# kernel_spec invariant tests

Local invariants of `KernelSpec` and its nested structs (`kernel_spec.hpp`).
Tests such as the name resolves ("the bound DFB is declared"), and rules that need every kernel binding an object, are
structural and live in `../program_spec/`.

## Listed invariants

`KernelSpec` as declared in `kernel_spec.hpp`, with every field and only its invariants.

```cpp
struct KernelSpec {
    KernelSpecName unique_id;

    struct SourceCode {
        // Invariant:
        // - Must be non-empty.
        std::string code;
    };
    // Invariant for the path:
    // - Must be non-empty.
    // - Must point to a file that exists.
    // - The file must be readable.
    std::variant<std::filesystem::path, SourceCode> source;

    // Invariant on Gen1 architectures (Wormhole, Blackhole): must be 1.
    // Invariant on Gen2 architecture (Quasar):
    //   - If is_data_movement_kernel(), the valid range is [1, 6]
    //   - If is_compute_kernel(), the valid values are [1, 2, 4]
    uint32_t num_threads = 1;

    bool is_data_movement_kernel() const;
    bool is_compute_kernel() const;

    struct CompilerOptions {
        using IncludePaths = std::vector<std::filesystem::path>;
        using Defines = Table<std::string, std::string>;
        using OptLevel = tt::tt_metal::KernelBuildOptLevel;

        IncludePaths include_paths;
        Defines defines;
        OptLevel opt_level = OptLevel::O2;
    };
    CompilerOptions compiler_options = {};

    struct DFBBinding {
        enum class EndpointType { PRODUCER, CONSUMER };
        enum class AccessPattern { STRIDED, ALL, BLOCKED };

        DFBSpecName dfb_spec_name;

        // Invariant: A valid C++ identifier shorter than (or equal to) MAX_ACCESSOR_NAME_LENGTH.
        std::string accessor_name;

        EndpointType endpoint_type;

        // Invariant:
        // - Cannot be blocked (not yet supported).
        // - For a producer binding, must be STRIDED.
        AccessPattern access_pattern = AccessPattern::STRIDED;
    };
    // Invariant:
    // - Each DFB has at most one PRODUCER binding and at most one CONSUMER binding.
    //   (A kernel that binds a DFB in both roles "self-loops" it.)
    // - Two bindings may share an accessor_name only if they are the PRODUCER and CONSUMER
    //   bindings of the same DFB. (A self-loop may also use two different accessor_names.)
    // - Gen2: a data-movement kernel must not self-loop a DFB.
    // - A compute kernel that self-loops a DFB must use STRIDED on its CONSUMER binding.
    // - A CONSUMER binding with access_pattern ALL requires num_threads <= 4.
    // - If is_compute_kernel(), every DFB this kernel binds sets data_format_metadata.
    Group<DFBBinding> dfb_bindings;

    struct SemaphoreBinding {
        SemaphoreSpecName semaphore_spec_name;

        // Invariant: A valid C++ identifier shorter than (or equal to) MAX_ACCESSOR_NAME_LENGTH.
        std::string accessor_name;
    };
    // Invariant:
    // - semaphore_spec_name must be unique across all semaphore_bindings.
    // - accessor_name must be unique across all semaphore_bindings.
    // - Gen 2 & wormhole: Must be empty if is_compute_kernel().
    Group<SemaphoreBinding> semaphore_bindings;

    struct ScratchpadBinding {
        ScratchpadSpecName scratchpad_spec_name;

        // Invariant: A valid C++ identifier shorter than (or equal to) MAX_ACCESSOR_NAME_LENGTH.
        std::string accessor_name;
    };
    // Invariant:
    // - scratchpad_spec_name must be unique across all scratchpad_bindings.
    // - accessor_name must be unique across all scratchpad_bindings.
    Group<ScratchpadBinding> scratchpad_bindings;

    struct TensorBinding {
        TensorParamName tensor_parameter_name;

        // Invariant: A valid C++ identifier shorter than (or equal to) MAX_ACCESSOR_NAME_LENGTH.
        std::string accessor_name;
    };
    // Invariant:
    // - accessor_name must be unique across all tensor_bindings.
    Group<TensorBinding> tensor_bindings;

    using CompileTimeArgs = Table<std::string, uint32_t>;
    // Table key represents the accessor name of the CTA.
    // Invariant:
    // - The key must be a valid C++ identifier.
    // - Must not have repeated name with runtime_arg_schema.
    CompileTimeArgs compile_time_args;

    struct RuntimeArgSchema {
        // Invariant:
        // - Must not have repeated names.
        // - All argument names must be valid C++ identifiers.
        Group<std::string> runtime_arg_names;

        // Invariant:
        // - Must not have repeated names.
        // - All argument names must be valid C++ identifiers.
        Group<std::string> common_runtime_arg_names;
    };
    // Invariant:
    // - No repeated names across runtime_arg_names and common_runtime_arg_names.
    // - Must not have repeated names with compile_time_args.
    RuntimeArgSchema runtime_arg_schema{};

    // Invariant for ComputeHardwareConfig:
    // - Every unpack_modes key names a DFB in this kernel's dfb_bindings (either role).
    // - For each DFB this kernel binds as CONSUMER:
    //   - An UnpackToDest entry requires enable_32_bit_dest when the DFB's data_format_metadata is 32-bit
    //     (Float32, Int32, UInt32 or RawUInt32). On Gen1 it requires enable_32_bit_dest for every format.
    //   - If the DFB's data_format_metadata is Float32 and enable_32_bit_dest is set, unpack_modes must
    //     have an entry for it (either mode; no default is assumed).
    //
    // Invariant for DataMovementHardwareConfig:
    // - Every config_2xx->disable_dfb_implicit_sync_for entry names a DFB in this kernel's dfb_bindings.
    std::variant<DataMovementHardwareConfig, ComputeHardwareConfig> hw_config;

    KernelAdvancedOptions advanced_options;
};
```
