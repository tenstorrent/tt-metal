# dataflow_buffer_spec invariant tests

Invariants of `DataflowBufferSpec` (`dataflow_buffer_spec.hpp`), including the borrowed-memory constraints
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
    // - Names a TensorParameter declared in ProgramSpec::tensor_parameters or a
    //   RawAllocationParameter declared in ProgramAdvancedOptions::raw_allocation_parameters.
    // - The bound memory object has L1/SRAM storage on every node occupied by the DFB.
    // - When borrowing from a tensor, its TensorSpec is L1-resident (L1 or L1_SMALL),
    //   and each occupied node's allocation can hold entry_size * num_entries bytes.
    // - When borrowing from a RawAllocation, advanced_options.borrowed_memory_offset
    //   + entry_size * num_entries must not exceed its per-node size on any occupied node.
    std::optional<std::variant<TensorParamName, RawAllocationParamName>> borrowed_from = std::nullopt;

    DFBAdvancedOptions advanced_options;
};

struct DFBAdvancedOptions {
    // Other fields to be filled in later.

    // Invariant:
    // - Must be a multiple of the L1 alignment.
    // - Must be zero when DataflowBufferSpec::borrowed_from is unset or names a TensorParameter.
    // - When borrowing from a RawAllocation, borrowed_memory_offset
    //   + DataflowBufferSpec::entry_size * DataflowBufferSpec::num_entries must not exceed
    //   the RawAllocation's per-node size on any node occupied by the DFB.
    uint32_t borrowed_memory_offset = 0;  // in bytes
};
```
