# prefetcher_pipe_parameter invariant tests

Local invariants of `PrefetcherPipeParameter` geometry (`prefetcher_pipe_parameter.hpp`). Binding and relay-list
rules are in `../advanced_options/`; roles, accessor groups, credit lanes and relay DFBs are structural and live in
`../program_spec/prefetcher_pipe_*.cpp`.

## Files

| File | Tests | Covers |
|---|---|---|
| `prefetcher_pipe_parameter.cpp` | 6 | `receivers`, `ring_size`, `entry_size` |

## Coverage

| Invariant | Tests |
|---|---|
| `receivers` is non-empty | PQ `EmptyReceiversFails` |
| `entry_size` is a multiple of the device's L1 alignment | PQ `UnalignedEntrySizeFails` |
| `entry_size <= ring_size` | PQ `EntrySizeLargerThanRingFails` |
| `ring_size > 0` (not stated in the header) | PQ `ZeroRingSizeFails` |
| `entry_size > 0` (not stated in the header) | PQ `ZeroEntrySizeFails` |
| `receivers` lie on the device's worker grid (not stated in the header; depends on the device) | PQ `OutOfBoundsReceiverFails` |
