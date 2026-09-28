# MakeProgramFromSpec output tests

These tests check what `MakeProgramFromSpec` (declared in `program.hpp`) produces from a valid spec: lowered
hardware configs, kernel argument layouts, resource slots and scopes. They inspect the Program's internals through
`program.impl()` on a mock device. Whether a spec is accepted at all is tested in
[`../invariant_tests/`](../invariant_tests/).

Fixtures: `ProgramSpecTestQuasar` (Q), `ProgramSpecTestGen1` (WH), `ProgramSpecTestBlackhole` (BH),
`PrefetcherPipeSpecTestQuasar` (PQ) and `ProgramRunArgsTestQuasar` (RQ).

| File | Tests | What is pinned |
|---|---|---|
| `compute_config_lowering.cpp` | 4 | `ComputeHardwareConfig` defaults, inverted fields and enums map to the internal compute config (Quasar and Wormhole); an UnpackToDest mode lands at the DFB's slot |
| `kernel_args_layout.cpp` | 4 | Layout of the lowered kernel's CRTA buffer: named CRTAs, varargs and tensor binding runtime fields, including the per-binding shape slots added by `dynamic_tensor_shape` |
| `kernel_config.cpp` | 2 | `compiler_options.include_paths` reach the kernel config; `disable_dfb_implicit_sync_for_all` turns off producer implicit sync on the DFB |
| `semaphore_scope.cpp` | 3 | Semaphore mechanism (`SemScope`): Blackhole compute-bound resolves to COMPUTE_ATOMIC, DM-only to LOCAL_NONATOMIC; Wormhole compute-bound is not COMPUTE_ATOMIC |
| `prefetcher_pipe_slots.cpp` | 3 | One pipe slot per accessor group; the relay DFB is created without an address; one pipe accessor token per binding |
| `graph_tracking.cpp` | 4 | DFBs and scratchpads a Program reports to `GraphTracker`: aliased DFBs collapse, borrowed DFBs are flagged, scratchpads are reported, DFBs are not reported when the Program isn't hooked |
