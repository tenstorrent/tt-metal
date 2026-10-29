# sram_arena_parameter invariant tests

Local invariants of `SRAMArenaParameter` (`sram_arena_parameter.hpp`) and its contained allocation
properties (`experimental/sram_arena/sram_arena_properties.hpp`). Borrower name resolution and the
requirement that every arena parameter is used are structural and are listed in `../program_spec/`. Borrower coverage and capacity constraints are listed in
`../dataflow_buffer_spec/` and `../scratchpad_spec/`.

## Listed invariants

`SRAMArenaParameter` as declared in `sram_arena_parameter.hpp`, with every field and only its invariants.

```cpp
struct SRAMArenaParameter {
    SRAMArenaParamName unique_id;

    // Invariant: The selected allocation properties satisfy their invariants below.
    tt::tt_metal::experimental::SRAMArenaProperties properties;
};
```

The contained property types from `sram_arena_properties.hpp`:

```cpp
struct LockstepSRAMArenaProperties {
    using size_type = uint32_t;

    // Invariant: Must be greater than 0.
    size_type per_bank_size;
};

struct RangeLockstepSRAMArenaProperties {
    using size_type = uint32_t;

    // Invariant: Must be non-empty.
    NodeRangeSet banks_to_be_allocated_on;

    // Invariant: Must be greater than 0.
    size_type per_bank_size;
};

using SRAMArenaProperties = std::variant<LockstepSRAMArenaProperties, RangeLockstepSRAMArenaProperties>;
```
