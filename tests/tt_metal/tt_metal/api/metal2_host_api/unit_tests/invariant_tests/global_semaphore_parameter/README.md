# global_semaphore_parameter invariant tests

Local invariants of `GlobalSemaphoreParameter` (`global_semaphore_parameter.hpp`). It has none of its own: the name
uniqueness, binding and usage rules are structural and live in `../program_spec/global_semaphores.cpp`; the rules on
the supplied `GlobalSemaphore` (persistent, same MeshDevice, covers every binding kernel's nodes) are checked by
`ProgramRunArgs` and tested in `../../program_run_args/global_semaphore_args.cpp`.

## Listed invariants

`GlobalSemaphoreParameter` as declared in `global_semaphore_parameter.hpp`, with every field and only its invariants.

```cpp
struct GlobalSemaphoreParameter {
    GlobalSemaphoreParamName unique_id;
};
```
