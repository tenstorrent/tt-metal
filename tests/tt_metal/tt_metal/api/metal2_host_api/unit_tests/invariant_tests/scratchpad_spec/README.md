# scratchpad_spec invariant tests

Invariants of `ScratchpadSpec` (`scratchpad_spec.hpp`) and its `ScratchpadAdvancedOptions`
(`advanced_options.hpp`), including the arena allocation constraints below. Binding rules are in `../kernel_spec/`. The rules that
every scratchpad is bound and every binding names a declared spec are structural, and the rule of at most one
binding kernel per node (unless multi-kernel binding is enabled) is local to `WorkUnitSpec`; both live in `../program_spec/`.

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

    // Invariant: on Gen2 (Quasar), multi-binding on a node is permitted only between DM kernels.
    ScratchpadAdvancedOptions advanced_options;
};

struct ScratchpadAdvancedOptions {
    bool allow_instance_multi_binding = false;

    // Invariant: When set:
    // - Names an SRAMArenaParameter declared in ProgramAdvancedOptions::sram_arena_parameters.
    // - The offset is a multiple of the L1 alignment.
    // - The arena covers every node occupied by the scratchpad.
    // - allocation_override->offset + ScratchpadSpec::size_per_node does not exceed
    //   the arena's declared per-bank size.
    std::optional<SRAMArenaAllocation> allocation_override = std::nullopt;
};
```
