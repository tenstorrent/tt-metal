# raw_allocation_parameter invariant tests

Local invariants of `RawAllocationParameter` (`raw_allocation_parameter.hpp`) and its contained allocation
properties (`experimental/allocation/raw_allocation_properties.hpp`). Borrower name resolution is structural
and is listed in `../program_spec/`. Borrower coverage and capacity constraints are listed in
`../dataflow_buffer_spec/` and `../scratchpad_spec/`.

## Listed invariants

`RawAllocationParameter` as declared in `raw_allocation_parameter.hpp`, with every field and only its invariants.

```cpp
struct RawAllocationParameter {
    RawAllocationParamName unique_id;

    // Invariant: The selected allocation properties satisfy their invariants below.
    tt::tt_metal::experimental::RawAllocationProperties properties;
};
```

The contained property types from `raw_allocation_properties.hpp`:

```cpp
struct LockstepAllocationProperties {
    using size_type = uint32_t;

    // Invariant: Must be greater than 0.
    size_type per_bank_size;
};

struct RangeLockstepAllocationProperties {
    using size_type = uint32_t;

    // Invariant: Must be non-empty.
    NodeRangeSet banks_to_be_allocated_on;

    // Invariant: Must be greater than 0.
    size_type per_bank_size;
};

using RawAllocationProperties = std::variant<LockstepAllocationProperties, RangeLockstepAllocationProperties>;
```
