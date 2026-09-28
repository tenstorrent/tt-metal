# Metal 2.0 Host API integration tests

This folder holds Metal 2.0 unit tests that tests integration of Metal 2.0 with other parts of the low level systems (e.g. LLK).

Tests contained in this directory may need the host-attached accelarator or compatible emulation platforms to be present.

Tests should prefer [unit tests suite](../unit_tests/) or [kernel compilation tests suite](../kernel_compilation_tests/) when tests are self-contained and are able to be run without silicon/ emulation environment.

## Fixtures

- `ProgramSpecHWTest` (`program_spec_hw_fixture.hpp`): a `MeshDeviceFixture` for Wormhole B0 and Blackhole. Needs
  slow dispatch (`TT_METAL_SLOW_DISPATCH_MODE=1`).
- `UnitMeshCQSingleCardFixture`: fast dispatch, one card.
- `MeshWorkloadFactoryHWTest` (fast dispatch), `MeshWorkloadFactorySlowDispatchHWTest` (slow dispatch) and
  `MeshWorkloadFactory1x2HWTest` (a 1x2 mesh, which needs two devices): defined in `mesh_workload_factories.cpp`.

## Files

| File | Tests | Fixture | What is checked |
|---|---|---|---|
| `binding_loopbacks.cpp` | 4 | `ProgramSpecHWTest` | DFB and TensorAccessor bindings in DRAM-to-DRAM loopbacks; two kernels syncing on one semaphore through different accessor names; a LocalTensorAccessor binding compiled for and wired into a compute kernel |
| `kernel_args_loopbacks.cpp` | 5 | `ProgramSpecHWTest` | Named RTAs, CRTAs and CTAs plus varargs, from DM and compute kernels; TT_KERNEL kernels (DM and compute); all CRTA sections after a set and a partial update |
| `scratchpad.cpp` | 2 | `ProgramSpecHWTest` | Scratchpad write and readback under slow dispatch; the scratchpad base address is delivered again after a DFB resize |
| `scratchpad_fast_dispatch.cpp` | 2 | `UnitMeshCQSingleCardFixture` | Scratchpad write and readback under fast dispatch; a scratchpad as either end of a NoC transfer |
| `compute_semaphore.cpp` | 7 | `ProgramSpecHWTest` | Compute-kernel semaphores (`SemScope::COMPUTE_ATOMIC`). Blackhole only; they skip on Wormhole |
| `mesh_workload_factories.cpp` | 5 | `MeshWorkloadFactory*` | `MakeMeshWorkloadFromSpec(s)`: repeated enqueue, DFB resize between enqueues, the map overload, slow dispatch, distinct specs on a 1x2 mesh |
