# compute_hardware_config invariant tests

Local invariants of `ComputeHardwareConfig` (`compute_hardware_config.hpp`). Their tests live in `../kernel_spec/hardware_config.cpp`. Rules that compare kernels are tested in `../program_spec/`.

## Listed invariants

`ComputeHardwareConfig` as declared in `compute_hardware_config.hpp`, with every field and only its invariants.
This is to be filled in as the header grows.

```cpp
struct ComputeHardwareConfig {
    tt::tt_metal::MathFidelity fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4;

    Precision sfpu_precision_mode = Precision::Precise;

    bool enable_32_bit_dest = false;

    bool double_buffer_dest = true;

    using ComputeUnpackModes = Table<DFBSpecName, tt::tt_metal::UnpackMode>;
    ComputeUnpackModes unpack_modes;

    struct Compute1XXConfig {
        Precision bfp_pack_precision_mode = Precision::Approximate;
    };
    std::optional<Compute1XXConfig> config_1xx = std::nullopt;

    struct Compute12XConfig {
        bool enable_pack_rvv = false;
    };
    std::optional<Compute12XConfig> config_12x = std::nullopt;

    struct Compute2XXConfig {
        bool enable_unpack_rvv = false;

        Group<DFBSpecName> sfpu_pipeline_consumed_dfbs;
        Group<DFBSpecName> sfpu_pipeline_produced_dfbs;
    };
    std::optional<Compute2XXConfig> config_2xx = std::nullopt;
};
```
