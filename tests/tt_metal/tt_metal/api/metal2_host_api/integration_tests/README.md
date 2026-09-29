# Metal 2.0 Host API integration tests

Tests that run kernels on silicon or an emulator and check their results, including Metal 2.0's integration with
the rest of the low-level stack (for example LLK). Put a test here only if a mock device cannot cover it.

## Fixtures

- `ProgramSpecHWTest` (`program_spec_hw_fixture.hpp`): one Wormhole B0 or Blackhole device, slow dispatch
  (`TT_METAL_SLOW_DISPATCH_MODE=1`).
- `UnitMeshCQSingleCardFixture`: one card, fast dispatch.
- `MeshWorkloadFactoryHWTest` (fast dispatch), `MeshWorkloadFactorySlowDispatchHWTest` (slow dispatch) and
  `MeshWorkloadFactory1x2HWTest` (a 1x2 mesh, two devices), defined in `mesh_workload_factories.cpp`.

## Files

| File | Tests | Fixture | What is checked |
|---|---|---|---|
| `binding_loopbacks.cpp` | 4 | `ProgramSpecHWTest` | DFB, semaphore and TensorAccessor accessor names in loopbacks; a LocalTensorAccessor binding in a compute kernel |
| `kernel_args_loopbacks.cpp` | 5 | `ProgramSpecHWTest` | Named RTAs, CRTAs, CTAs and varargs from DM and compute kernels, including TT_KERNEL kernels; all CRTA sections after a set and a partial update |
| `scratchpad.cpp` | 2 | `ProgramSpecHWTest` | Scratchpad write and readback; the base address is delivered again after a DFB resize |
| `scratchpad_fast_dispatch.cpp` | 2 | `UnitMeshCQSingleCardFixture` | Scratchpad write and readback; a scratchpad as either end of a NoC transfer |
| `compute_semaphore.cpp` | 7 | `ProgramSpecHWTest` | Compute-kernel semaphores (`SemScope::COMPUTE_ATOMIC`). Blackhole only |
| `llk_operand_mul.cpp` | 3 | `ProgramSpecHWTest` | `mul_tiles` with LLK operands from a DFB, a LocalTensorAccessor and a Scratchpad. Blackhole only |
| `mesh_workload_factories.cpp` | 5 | `MeshWorkloadFactory*` | `MakeMeshWorkloadFromSpec(s)`: repeated enqueue, DFB resize between enqueues, the map overload, slow dispatch, distinct specs on a 1x2 mesh |
