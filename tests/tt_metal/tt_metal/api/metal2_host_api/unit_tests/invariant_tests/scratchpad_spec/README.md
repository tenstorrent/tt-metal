# scratchpad_spec invariant tests

Local invariants of `ScratchpadSpec`, as written in `scratchpad_spec.hpp`.

Related scratchpad tests elsewhere:

- Binding rules (accessor names, bound once per kernel): [`../kernel_spec/scratchpad_binding.cpp`](../kernel_spec/scratchpad_binding.cpp)
- Every scratchpad is bound, and names a declared spec: [`../program_spec/declarations_used.cpp`](../program_spec/declarations_used.cpp), [`../program_spec/references.cpp`](../program_spec/references.cpp)
- At most one binding kernel per node: [`../program_spec/work_unit_bindings.cpp`](../program_spec/work_unit_bindings.cpp)
- `size_per_node` and `data_format_metadata` feed the kernel hash: [`../../kernel_hash/scratchpad_spec.cpp`](../../kernel_hash/scratchpad_spec.cpp)
- The format and tile metadata as seen by device code (`LLKOperandFrom`): [`../../../kernel_compilation_tests/bindings/llk_operand.cpp`](../../../kernel_compilation_tests/bindings/llk_operand.cpp)

Fixture: `ProgramSpecTestQuasar` (Q).

## Files

| File | Tests | Covers |
|---|---|---|
| `scratchpad_spec.cpp` | 4 | `size_per_node`, `data_format_metadata`, `tile_format_metadata` |

## Coverage of the header invariants

| Invariant | Tests |
|---|---|
| `size_per_node > 0` | Q `ZeroSizeScratchpadFails`. Accepted: Q `ValidScratchpadSucceeds` |
| `data_format_metadata`, when set, is supported on the target architecture | Q `ScratchpadFormatUnsupportedOnArchFails` |
| `tile_format_metadata` set requires `data_format_metadata` set | Q `ScratchpadTileWithoutFormatFails` |
