# scratchpad_spec invariant tests

Local invariants of `ScratchpadSpec` (`scratchpad_spec.hpp`). Binding rules are in `../kernel_spec/`. The rules that
every scratchpad is bound and every binding names a declared spec are structural, and the rule of at most one
binding kernel per node is local to `WorkUnitSpec`; both live in `../program_spec/`.

## Listed invariants

`ScratchpadSpec` as declared in `scratchpad_spec.hpp`, with every field and only its invariants.

```cpp
struct ScratchpadSpec {
    ScratchpadSpecName unique_id;

    // Invariant: Must be greater than 0.
    uint32_t size_per_node = 0;

    // Invariant:
    // - When data_format_metadata is set, it must be supported on the target architecture.
    std::optional<tt::DataFormat> data_format_metadata = std::nullopt;

    // Invariant:
    // - When tile_format_metadata is set, the data_format_metadata must also be set.
    std::optional<tt::tt_metal::Tile> tile_format_metadata = std::nullopt;
};
```
