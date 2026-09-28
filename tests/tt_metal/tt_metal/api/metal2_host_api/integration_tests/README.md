# Metal 2.0 Host API integration tests

End-to-end tests on real Wormhole B0 or Blackhole silicon: build a Program from a ProgramSpec, compile, dispatch,
and check the results on the device. The fixtures skip on any other architecture. Test names have no `CPU_` prefix,
so CI runs them on silicon runners.

A test belongs here only if it needs silicon. A test that only calls `MakeProgramFromSpec` or compiles kernels
should use a mock fixture in [`../unit_tests/`](../unit_tests/) or
[`../kernel_compilation_tests/`](../kernel_compilation_tests/).

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

## Running

Slow-dispatch and fast-dispatch tests need separate runs:

```sh
export TT_METAL_HOME=$PWD
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_api \
    --gtest_filter='ProgramSpecHWTest.*:MeshWorkloadFactorySlowDispatchHWTest.*'
./build/test/tt_metal/unit_tests_api \
    --gtest_filter='UnitMeshCQSingleCardFixture.Scratchpad*:MeshWorkloadFactoryHWTest.*:MeshWorkloadFactory1x2HWTest.*'
```
