# scratchpad_spec invariant tests

Local invariants of `ScratchpadSpec`, as written in `scratchpad_spec.hpp`.

Related scratchpad tests elsewhere:

- Binding rules (accessor names, bound once per kernel): [`../kernel_spec/scratchpad_binding.cpp`](../kernel_spec/scratchpad_binding.cpp)
- Every scratchpad is bound, and names a declared spec: [`../program_spec/declarations_used.cpp`](../program_spec/declarations_used.cpp), [`../program_spec/references.cpp`](../program_spec/references.cpp)
- At most one binding kernel per node: [`../program_spec/work_unit_bindings.cpp`](../program_spec/work_unit_bindings.cpp)
- `size_per_node` feeds the kernel hash: [`../../kernel_hash/scratchpad_spec.cpp`](../../kernel_hash/scratchpad_spec.cpp)

Fixture: `ProgramSpecTestQuasar` (Q).

## Files

| File | Tests | Covers |
|---|---|---|
| `scratchpad_spec.cpp` | 2 | `size_per_node` |

## Coverage of the header invariants

| Invariant | Tests |
|---|---|
| `size_per_node > 0` | Q `ZeroSizeScratchpadFails`. Accepted: Q `ValidScratchpadSucceeds` |
