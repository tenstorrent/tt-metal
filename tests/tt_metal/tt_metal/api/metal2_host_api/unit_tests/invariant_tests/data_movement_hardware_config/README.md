# data_movement_hardware_config invariant tests

Local invariants of `DataMovementHardwareConfig`, as written in `data_movement_hardware_config.hpp`.

The rules that compare data-movement kernels with each other (distinct processors and NOCs within a WorkUnitSpec,
Gen2 DM core assignment) are structural; they live in
[`../program_spec/gen1_dm_placement.cpp`](../program_spec/gen1_dm_placement.cpp) and
[`../program_spec/gen2_dm_core_assignment.cpp`](../program_spec/gen2_dm_core_assignment.cpp).

Fixtures: `ProgramSpecTestQuasar` (Q) and `ProgramSpecTestGen1` (WH).

## Files

| File | Tests | Covers |
|---|---|---|
| `config_1xx.cpp` | 3 | `config_1xx` (Wormhole, Blackhole) |

## Coverage of the header invariants

| Invariant | Tests |
|---|---|
| `DataMovement1XXConfig::processor` is RISCV_0 or RISCV_1 | WH `DMProcessorBeyondRiscv1Fails` |
| `config_1xx` is set when the kernel is built for TT-1.x.x | WH `DMKernelWithoutGen1SpecificFails`. Accepted on Gen2 without it: Q `DMKernelWithDefaultConfigSucceeds` |
| `DataMovement2XXConfig::disable_dfb_implicit_sync_for` has no repeated names | **Untested** |
