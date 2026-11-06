# advanced_options invariant tests

Local invariants of the `*AdvancedOptions` structs in `advanced_options.hpp`.  Rules that span several structs (DFB alias
groups, compute-bound semaphore options, PrefetcherPipe roles, lanes and relays) are structural and live in
`../program_spec/`.

## Listed invariants

`DFBAdvancedOptions` as declared in `advanced_options.hpp`.
Listings for the other advanced options remain to be added.

```cpp
struct DFBAdvancedOptions {
    // Invariant: Each name refers to another DFB; a DFB cannot alias itself.
    Group<DFBSpecName> alias_with;

    // Invariant: May be true only on Gen1 (Wormhole and Blackhole).
    bool allow_instance_multi_binding = false;

    // Invariant: Must not contain duplicate pipe names.
    Group<PrefetcherPipeParamName> prefetcher_pipe_relays;

    // Invariant:
    // - When set, must be non-empty and within the logical worker grid.
    // - Cannot be set together with prefetcher_pipe_relays.
    std::optional<CoreRangeSet> override_placement = std::nullopt;
};
```
