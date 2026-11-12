# prefetcher_pipe_parameter invariant tests

Local invariants of `PrefetcherPipeParameter` geometry (`prefetcher_pipe_parameter.hpp`). Binding and relay-list
rules are in `../advanced_options/`; roles, accessor groups, credit lanes and relay DFBs are structural and live in
`../program_spec/prefetcher_pipe_*.cpp`.

## Listed invariants

`PrefetcherPipeParameter` as declared in `prefetcher_pipe_parameter.hpp`, with every field and only its invariants.

```cpp
struct PrefetcherPipeParameter {
    PrefetcherPipeParamName unique_id;

    // Invariant: Must be non-empty.
    Nodes receivers;

    // Invariant: Must be greater than 0.
    uint32_t ring_size = 0;

    // Invariant:
    // - Must be greater than 0.
    // - Multiples of the L1 alignment on respective device.
    // - At most ring_size.
    uint32_t entry_size = 0;
};
```
