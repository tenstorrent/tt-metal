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

## Runtime rule (not a spec invariant)

With a sub-device manager loaded, `target_nodes` must not reach the cores of another sub-device than the one the
program's kernels run on: fast dispatch writes the semaphore's initial value into the kernel-config ring of every node
it is placed on, ordered only against the programs of the kernels' sub-device. `ValidateProgramSpec` has no sub-device
context; the workload is refused when it is built or enqueued under the manager
(`TensixTestSubDeviceSemaphoreSpecOnAnotherSubDeviceIsRefusedWhenBuilt` in
`tests/tt_metal/tt_metal/dispatch/dispatch_program/test_sub_device.cpp`).
