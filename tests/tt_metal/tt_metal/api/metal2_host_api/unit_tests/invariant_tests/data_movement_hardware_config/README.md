# data_movement_hardware_config invariant tests

Local invariants of `DataMovementHardwareConfig` (`data_movement_hardware_config.hpp`). Rules that compare DM kernels
with each other live in `../program_spec/`: `gen1_dm_placement.cpp` (local to `WorkUnitSpec`) and
`gen2_dm_core_assignment.cpp` (structural).

## Listed invariants

`DataMovementHardwareConfig` as declared in `data_movement_hardware_config.hpp`, with every field and only its
invariants.

```cpp
struct DataMovementHardwareConfig {
    struct DataMovement1XXConfig {
        // Invariant:
        // - Either RISCV_0 or RISCV_1.
        tt::tt_metal::DataMovementProcessor processor;

        tt::tt_metal::NOC noc;

        tt::tt_metal::NOC_MODE noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
    };
    // Invariant:
    // - If this kernel is built for TT-1.x.x, config_1xx must not be empty (Processor and NOC have no default).
    std::optional<DataMovement1XXConfig> config_1xx = std::nullopt;

    struct DataMovement2XXConfig {
        // Invariant: All DFBSpecNames must be unique.
        Group<DFBSpecName> disable_dfb_implicit_sync_for;

        bool disable_dfb_implicit_sync_for_all = false;
    };
    std::optional<DataMovement2XXConfig> config_2xx = std::nullopt;
};
```
