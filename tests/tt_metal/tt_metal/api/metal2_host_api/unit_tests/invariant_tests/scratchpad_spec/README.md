# scratchpad_spec invariant tests

Invariants of `ScratchpadSpec` (`scratchpad_spec.hpp`) and its `ScratchpadAdvancedOptions`
(`advanced_options.hpp`), including the borrowed-memory constraints below. Binding rules are in `../kernel_spec/`. The rules that
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

    struct SRAMArenaBorrowing {
        // Invariant: Names an SRAMArenaParameter declared in
        // ProgramAdvancedOptions::sram_arena_parameters.
        SRAMArenaParamName allocation;

        // Invariant: Must be a multiple of the L1 alignment.
        uint32_t offset = 0;  // in bytes
    };

    // Invariant: When set:
    // - The supplied SRAMArena must cover every node occupied by the scratchpad.
    // - borrowed_from->offset + ScratchpadSpec::size_per_node must not exceed
    //   the SRAMArena's per-node size on any occupied node.
    std::optional<SRAMArenaBorrowing> borrowed_from = std::nullopt;
};
```
