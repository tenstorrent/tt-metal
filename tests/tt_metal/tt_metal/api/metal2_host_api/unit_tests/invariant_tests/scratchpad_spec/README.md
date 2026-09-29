# scratchpad_spec invariant tests

Local invariants of `ScratchpadSpec` (`scratchpad_spec.hpp`). Binding rules are in `../kernel_spec/`; the rules that
every scratchpad is bound, names a declared spec, and has at most one binding kernel per node are structural and
live in `../program_spec/`.

## Files

| File | Tests | Covers |
|---|---|---|
| `scratchpad_spec.cpp` | 4 | `size_per_node`, `data_format_metadata`, `tile_format_metadata` |

## Coverage

| Invariant | Tests |
|---|---|
| `size_per_node > 0` | Q `ZeroSizeScratchpadFails`. Accepted: Q `ValidScratchpadSucceeds` |
| `data_format_metadata`, when set, is supported on the target architecture | Q `ScratchpadFormatUnsupportedOnArchFails` |
| `tile_format_metadata` set requires `data_format_metadata` set | Q `ScratchpadTileWithoutFormatFails` |
