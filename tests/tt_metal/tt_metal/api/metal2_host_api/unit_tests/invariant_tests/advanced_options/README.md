# advanced_options invariant tests

Local invariants of the `*AdvancedOptions` structs in `advanced_options.hpp`.  Rules that span several structs (DFB alias
groups, compute-bound semaphore options, PrefetcherPipe roles, lanes and relays) are structural and live in
`../program_spec/`.

## Listed invariants

`DFBAdvancedOptions` as declared in `advanced_options.hpp`.
Listings for the other advanced options remain to be added.

```cpp
struct DFBAdvancedOptions {
    // Invariant:
    // - Each name refers to another DFB; a DFB cannot alias itself.
    // - Aliased DFBs have instances on identical node sets (from override_placement or the bound
    //   kernels), regardless of how ranges are split.
    Group<DFBSpecName> alias_with;

    // Invariant: May be true only on Gen1 (Wormhole and Blackhole).
    bool allow_instance_multi_binding = false;

    // Invariant: Must not contain duplicate pipe names.
    Group<PrefetcherPipeParamName> prefetcher_pipe_relays;

    // Invariant: When set:
    // - A non-empty subset of the bound kernels' nodes (equality is allowed).
    // - Every node has at least one producer and one consumer bound to this DFB.
    // - Cannot be set together with prefetcher_pipe_relays or DataflowBufferSpec::borrowed_from.
    std::optional<NodeRangeSet> override_placement = std::nullopt;
};
```
