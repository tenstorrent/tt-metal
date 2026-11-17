# dataflow_buffer_spec invariant tests

Invariants of `DataflowBufferSpec` (`dataflow_buffer_spec.hpp`), including tensor borrowing and arena allocation constraints
from `DFBAdvancedOptions` (`advanced_options.hpp`) below. Structural rules involving kernel bindings live
in `../program_spec/`.

## Listed invariants

`DataflowBufferSpec` as declared in `dataflow_buffer_spec.hpp`, with every field and only its invariants.

```cpp
struct DataflowBufferSpec {
    DFBSpecName unique_id;

    // Invariant: Must be greater than 0.
    uint32_t entry_size = 0;
    // Invariant: Must be greater than 0.
    uint32_t num_entries = 0;

    // Invariant:
    // - When data_format_metadata is set, it must be supported on the target architecture.
    std::optional<tt::DataFormat> data_format_metadata = std::nullopt;

    // Invariant:
    // - When tile_format_metadata is set, the data_format_metadata must also be set.
    std::optional<tt::tt_metal::Tile> tile_format_metadata = std::nullopt;

    // Invariant: When set:
    // - Names a TensorParameter declared in ProgramSpec::tensor_parameters.
    // - The tensor has L1/SRAM storage on every node occupied by the DFB.
    // - Its TensorSpec is L1-resident (L1 or L1_SMALL), and each occupied node's
    //   allocation can hold entry_size * num_entries bytes.
    // - Cannot be combined with an allocation override or a PrefetcherPipe relay.
    std::optional<TensorParamName> borrowed_from = std::nullopt;

    DFBAdvancedOptions advanced_options;
};

struct DFBAdvancedOptions {
    // Other fields to be filled in later.

    // Invariant: When set:
    // - Names an SRAMArenaParameter declared in ProgramAdvancedOptions::sram_arena_parameters.
    // - The offset is a multiple of the L1 alignment.
    // - The arena covers every node occupied by the DFB.
    // - The offset plus DataflowBufferSpec::entry_size * DataflowBufferSpec::num_entries
    //   does not exceed the arena's declared per-bank size.
    // - Cannot be combined with DataflowBufferSpec::borrowed_from or a PrefetcherPipe relay.
    std::optional<SRAMArenaAllocation> allocation_override = std::nullopt;
};
```
