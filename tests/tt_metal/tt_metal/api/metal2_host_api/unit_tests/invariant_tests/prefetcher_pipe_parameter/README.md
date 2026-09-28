# prefetcher_pipe_parameter invariant tests

Local invariants of `PrefetcherPipeParameter` geometry, as written in `prefetcher_pipe_parameter.hpp`.

Related PrefetcherPipe tests elsewhere:

- `PrefetcherPipeBinding` and relay-list rules: [`../advanced_options/`](../advanced_options/)
- Roles, accessor groups, credit lanes and relay DFBs (structural): `prefetcher_pipe_roles.cpp`,
  `prefetcher_pipe_lanes.cpp` and `prefetcher_pipe_relays.cpp` in [`../program_spec/`](../program_spec/)
- Slot reservation in `MakeProgramFromSpec`: [`../../program/prefetcher_pipe_slots.cpp`](../../program/prefetcher_pipe_slots.cpp)
- Binding pipe objects at run time: [`../../program_run_args/prefetcher_pipe_args.cpp`](../../program_run_args/prefetcher_pipe_args.cpp)

Fixture: `PrefetcherPipeSpecTestQuasar` (PQ), from `prefetcher_pipe_test_helpers.hpp`. That header also defines
the test geometry: sender (0,0), receivers (0,1)..(0,3), entry size 2048, 4 entries, ring size 8192.

## Files

| File | Tests | Covers |
|---|---|---|
| `prefetcher_pipe_parameter.cpp` | 6 | `receivers`, `ring_size`, `entry_size` |

## Coverage of the header invariants

| Invariant | Tests |
|---|---|
| `receivers` is non-empty | PQ `EmptyReceiversFails` |
| `entry_size` is a multiple of the device's L1 alignment | PQ `UnalignedEntrySizeFails` |
| `entry_size <= ring_size` | PQ `EntrySizeLargerThanRingFails` |

Also enforced, though the header doesn't state them as invariants:

- `ring_size > 0`: PQ `ZeroRingSizeFails`
- `entry_size > 0`: PQ `ZeroEntrySizeFails`
- `receivers` lie on the device's worker grid: PQ `OutOfBoundsReceiverFails`. This depends on the device, not
  only on the spec.
