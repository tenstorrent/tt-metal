# Metal 2.0 Host API integration tests

Tests that run kernels on silicon or an emulator and check their results, including Metal 2.0's integration with
the rest of the low-level stack (for example LLK). Put a test here only if a mock device cannot cover it.

## Fixtures

- `ProgramSpecHWTest` (`program_spec_hw_fixture.hpp`): one Wormhole B0 or Blackhole device, slow dispatch
  (`TT_METAL_SLOW_DISPATCH_MODE=1`).
- `UnitMeshCQSingleCardFixture`: one card, fast dispatch.
- `MeshWorkloadFactoryHWTest` (fast dispatch), `MeshWorkloadFactorySlowDispatchHWTest` (slow dispatch) and
  `MeshWorkloadFactory1x2HWTest` (a 1x2 mesh, two devices), defined in `mesh_workload_factories.cpp`.
