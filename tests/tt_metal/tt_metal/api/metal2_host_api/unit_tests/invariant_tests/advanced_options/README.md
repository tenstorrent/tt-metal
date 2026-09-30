# advanced_options invariant tests

Local invariants of the `*AdvancedOptions` structs in `advanced_options.hpp`. They have no "Listed invariants"
section yet, so these tests follow the checks in the implementation. Rules that span several structs (DFB alias
groups, compute-bound semaphore options, PrefetcherPipe roles, lanes and relays) are structural and live in
`../program_spec/`.
