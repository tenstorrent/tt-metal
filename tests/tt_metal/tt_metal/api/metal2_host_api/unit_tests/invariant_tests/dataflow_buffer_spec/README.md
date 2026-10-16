# dataflow_buffer_spec invariant tests

Local invariants of `DataflowBufferSpec` (`dataflow_buffer_spec.hpp`).  Most other DFB rules depend
on the kernels that bind the DFB, so they are structural and live in `../program_spec/`; `DFBAdvancedOptions` rules
are in `../advanced_options/`.

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

    // Invariant:
    // - When set, the named TensorParameter's TensorSpec is L1-resident (L1 or L1_SMALL).
    std::optional<TensorParamName> borrowed_from = std::nullopt;

    DFBAdvancedOptions advanced_options;
};
```
