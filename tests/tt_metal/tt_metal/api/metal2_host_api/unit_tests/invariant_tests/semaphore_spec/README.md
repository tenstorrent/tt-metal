# semaphore_spec invariant tests

Local invariants of `SemaphoreSpec` (`semaphore_spec.hpp`).
The rules for binding a semaphore are in `../kernel_spec/`, its advanced options in
`../advanced_options/`, and the rules that need every kernel binding a semaphore in `../program_spec/`.

## Listed invariants

`SemaphoreSpec` as declared in `semaphore_spec.hpp`, with every field and only its invariants.

```cpp
struct SemaphoreSpec {
    SemaphoreSpecName unique_id;

    // Invariant:
    // - Must be non-empty
    Nodes target_nodes;

    SemaphoreAdvancedOptions advanced_options;
};
```
