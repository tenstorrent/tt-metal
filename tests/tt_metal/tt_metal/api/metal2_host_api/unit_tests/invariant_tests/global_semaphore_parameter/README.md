# global_semaphore_parameter invariant tests

Local invariants of `GlobalSemaphoreParameter` (`global_semaphore_parameter.hpp`). The requirement that every
parameter is bound, and that every binding names a declared parameter, is structural and is listed in
`../program_spec/`. The binding invariants are listed in `../advanced_options/`.

## Listed invariants

`GlobalSemaphoreParameter` as declared in `global_semaphore_parameter.hpp`, with every field and only its invariants.

```cpp
struct GlobalSemaphoreParameter {
    GlobalSemaphoreParamName unique_id;

    // Invariant: L1 or L1_SMALL.
    BufferType memory_type = BufferType::L1;
};
```
