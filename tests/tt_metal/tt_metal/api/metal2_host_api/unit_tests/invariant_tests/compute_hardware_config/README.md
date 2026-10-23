# compute_hardware_config invariant tests

Local invariants of `ComputeHardwareConfig` (`compute_hardware_config.hpp`). It has none of its own for now.

## Listed invariants

`ComputeHardwareConfig` as declared in `compute_hardware_config.hpp`, with every field and only its invariants.
This is to be filled in as the header grows.

```cpp
struct ComputeHardwareConfig {
    MathFidelity fpu_math_fidelity = MathFidelity::HiFi4;

    Precision sfpu_precision_mode = Precision::Precise;

    bool enable_32_bit_dest = false;

    bool double_buffer_dest = true;

    using ComputeUnpackModes = Table<DFBSpecName, tt::tt_metal::UnpackMode>;
    ComputeUnpackModes unpack_modes;

    struct Compute1XXConfig {
        Precision bfp_pack_precision_mode = Precision::Approximate;
    };
    std::optional<Compute1XXConfig> config_1xx = std::nullopt;

    struct Compute2XXConfig {};
    std::optional<Compute2XXConfig> config_2xx = std::nullopt;
};
```
